"""M6D5e(-r) E06c production-runner qualification harness: QUALIFICATION_ONLY, never scientific.

M6D5e-r clean requalification. Every process uses qualification seed 60506 (not an
experiment seed) and writes only under <runtime_root>/qualification/m6d5e/E06c/q60506-<run_id>/,
the dedicated build root builds/e06c_dsdg_m6d5e_r and TMPDIR /tmp/gpat-m6d5er. The initial
M6D5e root (seed 60505) is never reused. Real TRAIN data (manifests/pairs_train_v1.parquet +
canonical faces of its sample_ids) is used; VAL and TEST have no code path. An audit-hook
firewall denies every other manifest (the split manifest included), faces_256 enumeration,
faces outside the TRAIN relation, <runtime_root>/runs and repository data roots, and logs
every benchmark-path access (DataLoader workers included) to an append-only audit file.

  loader  ID-only DataLoader order determinism: two constructions with seed 60506 over an
          index-only dataset (no image bytes); epoch-1 and epoch-2 pair-order hashes.
  train   ONE real-TRAIN epoch through the production engine (methods/dsdg/runner.py):
          8838 pairs, 36 x 240 + 198, 37 V2 global steps, the upstream epoch-1 visualization
          block (CPU RNG replay-proven), the epoch-1 official checkpoint event and engineering
          resume sidecar, then ONE in-memory epoch-2 reference step.
  resume  fresh process: explicit epoch-1 resume state, RNG/loader restore, ONE epoch-2 step
          appended to the same trajectory, compared against the reference probe.

The resulting checkpoints and sidecars are QUALIFICATION_ONLY / NOT_SCIENTIFIC_CHECKPOINT /
NOT_ELIGIBLE_FOR_SYNTHETIC_BANK / NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION / NOT_ELIGIBLE_FOR_REPORTING.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.common.config import sha256_file  # noqa: E402
from methods.dsdg import runner as engine  # noqa: E402
from methods.dsdg import runner_io as rio  # noqa: E402
from methods.dsdg import runtime as m6d5a  # noqa: E402  (unchanged M6D5a helpers)
from methods.dsdg import training_graph as tg  # noqa: E402
from methods.dsdg import training_qualification as m6d5b  # noqa: E402  (unchanged M6D5b helpers)

SEED = rio.QUALIFICATION_SEED
MODE = rio.QUALIFICATION
BUILD_PARTS = ('builds', 'e06c_dsdg_m6d5e_r')
# DataLoader workers pass tensor file descriptors through a multiprocessing AF_UNIX socket created
# under TMPDIR; a TMPDIR inside the (long) build root exceeds the 108-byte socket-path limit and
# deadlocks the loader, so the harness uses this short dedicated directory instead.
SHORT_TMP = Path('/tmp/gpat-m6d5er')
CLEAN_GPU_MAX_USED_MIB = 1024
OUTPUTS = {'loader': 'M6D5E_R_E06C_LOADER_DETERMINISM.json', 'train': 'M6D5E_R_E06C_REAL_TRAIN_EPOCH1.json',
           'reference': 'M6D5E_R_E06C_RESUME_REFERENCE.json', 'resume': 'M6D5E_R_E06C_RESUME_FRESH_PROCESS.json'}
ACCESS_LOG = 'M6D5E_R_E06C_BENCHMARK_ACCESS_{}.tsv'
# benchmark-path categories; every one except TRAIN_RELATION / TRAIN_FACE is denied and must stay at 0
CATEGORIES = ('TRAIN_RELATION', 'TRAIN_FACE', 'SPLIT_METADATA_VAL_TEST', 'VAL_METADATA', 'TEST_METADATA',
              'OTHER_MANIFEST', 'NON_TRAIN_FACE', 'FACES_ENUMERATION_OR_WRITE', 'SCIENTIFIC_RUN_ROOT',
              'OTHER_BENCHMARK_DATA')
PROBE = 'qualification_probes/reference_epoch2_step1.pt'
EXIT = {'PASS': 0, 'STOP_TRAINING_GATE': 3, 'STOP_RESOURCE_CONTAMINATION': 4, 'STOP_RESUME_MISMATCH': 5,
        'STOP_LOADER_NONDETERMINISM': 6, 'STOP_EXISTING_QUALIFICATION_ROOT': 7}
require = rio.require


class Firewall:
    """Audit hook: TRAIN relation + TRAIN canonical faces only; no other data, no scientific root.

    Every benchmark-path event (allowed or denied) is appended as one line to `access_fd`, an
    O_APPEND descriptor opened before the hook is installed and inherited by forked DataLoader
    workers, so worker face reads are audited too (os.write raises no audit event).
    """
    WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    EVENTS = ('open', 'os.listdir', 'os.scandir', 'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'shutil.rmtree')

    def __init__(self, runtime_root, faces_root, write_roots, sample_datasets, allow_faces, access_fd=None):
        self.runtime_root, self.faces = Path(runtime_root), Path(faces_root)
        self.write_roots = [str(Path(p)) + '/' for p in write_roots]
        self.samples, self.allow_faces = dict(sample_datasets), allow_faces
        self.manifest = str(ROOT / rio.TRAIN_RELATION)
        self.lightcnn = str(m6d5a.LIGHTCNN)
        self.events = dict.fromkeys(self.EVENTS, 0)
        self.denied, self.manifest_opens, self.face_opens_main_process = [], {}, 0
        self.access_fd = access_fd

    def category(self, event, path):
        """Benchmark category of a path (None for code, environment and model assets)."""
        p = Path(path)
        if p.is_relative_to(ROOT / 'manifests'):
            if path == self.manifest:
                return 'TRAIN_RELATION'
            name = p.name.lower()
            if name.startswith('split'):
                return 'SPLIT_METADATA_VAL_TEST'       # the split manifest holds TRAIN, VAL and TEST rows
            if 'val' in name:
                return 'VAL_METADATA'
            if 'test' in name:
                return 'TEST_METADATA'
            return 'OTHER_MANIFEST'
        if p.is_relative_to(self.faces):
            rel = p.relative_to(self.faces).parts
            if event != 'open':
                return 'FACES_ENUMERATION_OR_WRITE'
            ok = len(rel) == 2 and rel[1].endswith('.png') and self.samples.get(rel[1][:-4]) == rel[0]
            return 'TRAIN_FACE' if ok else 'NON_TRAIN_FACE'
        if p.is_relative_to(self.runtime_root / 'runs'):
            return 'SCIENTIFIC_RUN_ROOT'
        if p.is_relative_to(self.runtime_root / 'data') or any(p.is_relative_to(ROOT / d) for d in ('data', 'cache', 'runs')):
            return 'OTHER_BENCHMARK_DATA'
        return None

    def writable(self, path):
        return (path == '/dev/null' or path.startswith('/dev/shm/') or
                any(path == r[:-1] or path.startswith(r) for r in self.write_roots))

    def reason(self, event, path, args):
        p = Path(path)
        writing = event != 'open' and event not in ('os.listdir', 'os.scandir')
        if event == 'open':
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            writing = bool(flags & self.WRITE_FLAGS) or bool(set(mode) & set('wax+'))
        if p.is_relative_to(self.runtime_root / 'runs'):
            return 'scientific run root'
        if p.is_relative_to(ROOT / 'manifests'):
            if event == 'open' and path == self.manifest and not writing:
                self.manifest_opens[path] = self.manifest_opens.get(path, 0) + 1
                return None
            return 'manifest other than the TRAIN relation'
        if p.is_relative_to(self.faces):
            if event != 'open':
                return 'faces_256 enumeration/mutation'
            rel = p.relative_to(self.faces).parts
            ok = (len(rel) == 2 and rel[1].endswith('.png') and self.samples.get(rel[1][:-4]) == rel[0])
            if writing or not ok:
                return 'face outside the TRAIN relation or write'
            if not self.allow_faces:
                return 'image bytes are not permitted in this mode'
            self.face_opens_main_process += 1
            return None
        if p.is_relative_to(self.runtime_root / 'data') or any(
                p.is_relative_to(ROOT / d) for d in ('data', 'cache', 'runs')):
            return 'benchmark data root'
        if p.suffix == '.parquet':
            return 'parquet outside the TRAIN relation'
        if writing and not self.writable(path):
            return 'write outside the qualification/build roots'
        if p.suffix in ('.pth', '.pt', '.tar', '.ckpt', '.pkl') and path != self.lightcnn and not self.writable(path):
            return 'weight file other than LightCNN or this qualification run'
        return None

    def __call__(self, event, args):
        if event not in self.events or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        self.events[event] += 1
        paths = [args[0]] + ([args[1]] if event == 'os.rename' and len(args) > 1 else [])
        for raw in paths:
            path = os.path.abspath(os.fsdecode(raw))
            why = self.reason(event, path, args)
            cat = self.category(event, path)
            if cat is not None and self.access_fd is not None:
                os.write(self.access_fd, f'{os.getpid()}\t{event}\t{cat}\t{"DENIED" if why else "ALLOWED"}\t'
                                         f'{path}\n'.encode('utf-8'))
            if why:
                self.denied.append({'event': event, 'path': path, 'reason': why})
                raise PermissionError(f'E06c QUALIFICATION FIREWALL ({why}): {path}')

    def report(self):
        return {'denied': self.denied, 'event_counts': self.events,
                'manifest_opens': {os.path.relpath(k, ROOT): v for k, v in self.manifest_opens.items()},
                'face_opens_in_main_process': self.face_opens_main_process,
                'note': 'DataLoader worker processes inherit this hook (fork); their face reads are counted from the '
                        'consumed batches (2 canonical faces per pair row). pyarrow reads the TRAIN relation natively '
                        'after the Python SHA256 open.'}


def access_audit(path, train_sample_ids):
    """Summarize the append-only benchmark access log (all processes that inherited the hook)."""
    rows = [line.split('\t') for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(all(len(r) == 5 for r in rows), 'access log format')
    counts = dict.fromkeys(CATEGORIES, 0)
    denied = 0
    faces, pids = set(), set()
    for pid, event, cat, verdict, p in rows:
        counts[cat] += 1
        denied += verdict == 'DENIED'
        pids.add(pid)
        if cat == 'TRAIN_FACE':
            faces.add(Path(p).stem)
    val_meta = counts['SPLIT_METADATA_VAL_TEST'] + counts['VAL_METADATA']
    test_meta = counts['SPLIT_METADATA_VAL_TEST'] + counts['TEST_METADATA']
    out = {'log': Path(path).name, 'log_sha256': sha256_file(path), 'events_logged': len(rows),
           'processes_logged': len(pids), 'counts_by_category': counts, 'denied_events': denied,
           'train_faces_opened_distinct': len(faces),
           'train_faces_opened_all_in_relation': faces <= set(train_sample_ids),
           'VAL_metadata_accesses': val_meta, 'TEST_metadata_accesses': test_meta,
           'split_manifest_accesses': counts['SPLIT_METADATA_VAL_TEST'],
           'non_train_face_accesses': counts['NON_TRAIN_FACE'],
           'VAL_image_accesses': counts['NON_TRAIN_FACE'], 'TEST_image_accesses': counts['NON_TRAIN_FACE'],
           'image_access_note': 'Every canonical face path is classified against the frozen TRAIN relation only (split '
                                'metadata is never read). VAL/TEST image accesses are bounded above by the non-TRAIN face '
                                'accesses, which must be 0.',
           'coverage_note': 'Python audit events (open/os.open/listdir/scandir/mkdir/remove/rename) in the main process '
                            'and every forked DataLoader worker. pyarrow reads the TRAIN relation natively after the '
                            'audited Python SHA256 open, and is only ever given that exact path.'}
    return out


def instrument(torch, counters):
    """Count backward and zero_grad calls; forbid autograd.grad and activation checkpointing."""
    tensor_backward, autograd_backward = torch.Tensor.backward, torch.autograd.backward
    zero_grad = torch.optim.Optimizer.zero_grad

    def backward(self, *a, **k):
        counters['backward_calls'] += 1
        return tensor_backward(self, *a, **k)

    def autograd(*a, **k):
        counters['autograd_backward_calls'] += 1
        return autograd_backward(*a, **k)

    def zero(self, set_to_none=True):
        counters['zero_grad_calls'] += 1
        counters['zero_grad_set_to_none_values'] = sorted(set(counters['zero_grad_set_to_none_values']) | {set_to_none})
        return zero_grad(self, set_to_none=set_to_none)

    def forbid(name):
        def forbidden(*a, **k):
            counters[name] += 1
            raise RuntimeError('M6D5e forbids ' + name)
        return forbidden
    import torch.utils.checkpoint  # noqa: F401
    torch.Tensor.backward, torch.autograd.backward, torch.optim.Optimizer.zero_grad = backward, autograd, zero
    torch.autograd.grad = forbid('autograd_grad_calls')
    torch.utils.checkpoint.checkpoint = forbid('activation_checkpoint_calls')


def new_counters():
    return {'backward_calls': 0, 'autograd_backward_calls': 0, 'zero_grad_calls': 0,
            'zero_grad_set_to_none_values': [], 'autograd_grad_calls': 0, 'activation_checkpoint_calls': 0}


# ================================================================= setup shared by every mode
def setup(build, mode, *, allow_faces, needs_gpu):
    storage = rio.faces_root_from_exec_config()
    runtime_root = Path(storage['runtime_root'])
    require(build == runtime_root / BUILD_PARTS[0] / BUILD_PARTS[1], 'dedicated build root')
    access_log = build / ACCESS_LOG.format(mode)
    access_fd = os.open(access_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)   # fresh per mode
    # the firewall is active before any manifest byte is read; face paths resolve only once the
    # TRAIN relation has been projected (until then the sample map is empty, so every face is denied)
    firewall = Firewall(runtime_root, storage['faces_256_root'],
                        [build, runtime_root.joinpath(*rio.QUALIFICATION_PARTS[:2]), SHORT_TMP], {}, allow_faces,
                        access_fd)
    sys.addaudithook(firewall)
    contract = rio.load_contract()                      # verifies bound input SHA256 (incl. TRAIN relation bytes)
    ids = rio.identities(contract)
    source, adapter = m6d5a.source_identity()
    config = adapter.config
    lock = json.loads((ROOT / engine.LOCK_PATH).read_text())
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']) and
            source['files_sha256'] == lock['source']['files_sha256'], 'source identity equals environment lock')
    records, datasets, relation = rio.read_train_relation(adapter)
    firewall.samples = dict(datasets)
    result = {'mode': mode, 'qualification_seed': SEED, 'experiment_seed': None,
              'labels': list(rio.QUALIFICATION_LABELS), 'runner_mode': MODE, 'execution_mode': rio.EXECUTION_MODE,
              'storage': storage, 'identities': ids, 'contract_path': rio.CONTRACT_PATH,
              'train_relation': relation, 'source_before': source, 'overlays': engine.verify_overlays(),
              'm6d5e_file_sha256': rio.m6d5e_file_hashes(), 'pid': os.getpid(),
              'launch_environment': {k: os.environ.get(k) for k in (
                  'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
                  'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TMPDIR')}}
    require(result['launch_environment']['PYTHONHASHSEED'] == str(SEED), f'PYTHONHASHSEED={SEED}')
    if needs_gpu:
        result['asset_before'] = m6d5a.asset_identity(adapter)
        result['gpu_before'] = m6d5b.gpu_snapshot()
        foreign = [p for p in result['gpu_before']['compute_processes'] if int(p['pid']) != os.getpid()]
        result['resource_clean'] = not foreign and result['gpu_before']['used_mib'] <= CLEAN_GPU_MAX_USED_MIB
        if not result['resource_clean']:
            result.update(status='STOP_RESOURCE_CONTAMINATION', foreign_compute_processes=foreign)
            return result, None
    import torch
    import torch.nn.functional as F
    counters = new_counters()
    instrument(torch, counters)
    ctx = dict(torch=torch, F=F, adapter=adapter, config=config, records=records, datasets=datasets, ids=ids,
               firewall=firewall, counters=counters, runtime_root=runtime_root, storage=storage, source=source,
               lock=lock, access_log=access_log, access_fd=access_fd)
    if needs_gpu:
        result['precision'] = engine.configure_precision(torch)
        result['seeding'] = engine.seed_process(torch, MODE, SEED, config)
        ctx['cpu_rng_after_seeding'] = torch.get_rng_state()
        env = m6d5a.environment()
        diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
                if k != 'launch_environment' and env.get(k) != v}
        require(not diff, 'runtime identity equals environment lock: ' + json.dumps(diff))
        result['environment_before'] = env
        result['environment_lock_identity_match'] = sorted(k for k in lock['identity'] if k != 'launch_environment')
        ctx['env'] = env
    return result, ctx


def finish(result, ctx, needs_gpu=True):
    torch, fw = ctx['torch'], ctx['firewall']
    result['counters'] = ctx['counters']
    result['source_after'] = m6d5a.source_identity()[0]
    require(result['source_after'] == ctx['source'], 'source-cache integrity after execution')
    if needs_gpu:
        result['asset_after'] = m6d5a.asset_identity(ctx['adapter'])
        require(result['asset_after'] == result['asset_before'], 'LightCNN unchanged')
        result['environment_after'] = m6d5a.environment()
        require(result['environment_after'] == ctx['env'], 'environment stable')
        require(not torch.is_autocast_enabled('cuda'), 'autocast off')
        result['gpu_after'] = m6d5b.gpu_snapshot()
    require(ctx['counters']['autograd_grad_calls'] == ctx['counters']['activation_checkpoint_calls'] == 0,
            'no forbidden call')
    result['firewall'] = fw.report()
    require(not fw.denied, 'firewall denials')
    os.close(ctx['access_fd'])          # workers have exited: every audited event is in the log
    audit = access_audit(ctx['access_log'], ctx['datasets'])
    result['benchmark_access_audit'] = audit
    require(audit['denied_events'] == 0 and audit['train_faces_opened_all_in_relation'] and
            all(v == 0 for k, v in audit['counts_by_category'].items() if k not in ('TRAIN_RELATION', 'TRAIN_FACE')) and
            audit['VAL_metadata_accesses'] == audit['TEST_metadata_accesses'] == 0 and
            audit['VAL_image_accesses'] == audit['TEST_image_accesses'] == 0, 'TRAIN-only benchmark access audit')
    result.update(VAL_access=False, TEST_access=False, scientific_training=False, scientific_checkpoint_created=False,
                  synthetic_bank=False, downstream_evaluation=False, fidelity='CONTROLLED_ADAPTATION',
                  compatibility_patch='NONE')
    return result


def components(ctx, result, modules, dataset):
    torch = ctx['torch']
    root = Path(ctx['source']['root']) / ctx['config']['source']['relevant_path']
    models, result['binding'] = m6d5b.build_models(torch, modules['networks'], root, ctx['config'])
    result['lightcnn'] = m6d5a.load_lightcnn(torch, models['netIP'])
    models['netE_nir'].train(); models['netE_vis'].train(); models['netG'].train(); models['netIP'].eval()
    params = {n: m6d5a.parameter_report(m) for n, m in models.items()}
    for n, (count, tensors) in m6d5a.EXPECTED_PARAMETERS.items():
        require((params[n]['parameters'], params[n]['parameter_tensors']) == (count, tensors), 'live count ' + n)
    require(params['netIP']['requires_grad_parameters'] == 0, 'netIP frozen')
    result['parameter_counts'] = {n: {'parameters': p['parameters'], 'parameter_tensors': p['parameter_tensors']}
                                  for n, p in params.items()}
    optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                   ctx['config']['optimizer']['learning_rate'])
    result['optimizer'] = m6d5b.optimizer_evidence(torch, optimizer, models)
    result['scheduler'] = 'NONE (pinned trainer defines none)'
    loader, generator, result['dataloader'] = engine.build_loader(torch, dataset, MODE, SEED, ctx['config'])
    trainer = engine.Trainer(torch, ctx['F'], modules['misc.util'], models, optimizer, loader, generator,
                             ctx['records'], MODE, SEED, tg.frozen_lambdas(ctx['config']),
                             visualization_every=rio.visualization_every(ctx['config']))
    return trainer, params


def run_context(ctx, resume):
    env, reasons = engine.environment_record(ctx['torch'])
    return engine.E06cRunContext(mode=MODE, seed=SEED, runtime_root=ctx['runtime_root'], environment=env,
                                 missing_environment_reasons=reasons, identities=ctx['ids'], resume=resume,
                                 command_line=' '.join([sys.executable] + sys.argv))


def model_sha(models):
    return {n: {k: engine.tsha(p) for k, p in m.named_parameters()} for n, m in models.items()}


def public_probe(r):
    probe = {k: v for k, v in r['probe'].items() if k not in ('gradients', 'parameters_after')}
    return dict({k: v for k, v in r.items() if k not in ('probe', 'pair_ids')}, probe=probe,
                pair_ids_first5=r['pair_ids'][:5])


# ================================================================= loader determinism (ID-only)
def loader_mode(build):
    result, ctx = setup(build, 'loader', allow_faces=False, needs_gpu=False)
    torch, records = ctx['torch'], ctx['records']
    ids = [r['pair_id'] for r in records]
    reps = {}
    for rep in ('A', 'B'):
        loader, generator, evidence = engine.build_loader(torch, rio.IndexOnlyDataset(len(records)), MODE, SEED,
                                                          ctx['config'])
        out = {'generator_state_initial_sha256': rio.sha(generator.get_state().numpy().tobytes()), 'loader': evidence}
        for epoch in (1, 2):
            order, sizes = [], []
            for batch in loader:
                sizes.append(len(batch))
                order.extend(ids[i] for i in batch.tolist())
            out[f'epoch{epoch}'] = {'batch_sizes': sizes, 'rows': len(order), 'unique': len(set(order)),
                                    'pair_order_sha256': rio.order_sha256(order),
                                    'first_batch_pair_sha256': rio.order_sha256(order[:sizes[0]]),
                                    'generator_state_after_sha256': rio.sha(generator.get_state().numpy().tobytes())}
            require(sizes == [240] * 36 + [198] and len(order) == len(set(order)) == 8838, 'ID-only epoch plan')
        reps[rep] = out
    result['constructions'] = reps
    same = {k: reps['A'][k] == reps['B'][k] for k in reps['A']}
    result['repeat_equal'] = same
    result['epoch1_pair_order_sha256'] = reps['A']['epoch1']['pair_order_sha256']
    result['epoch2_first_batch_pair_sha256'] = reps['A']['epoch2']['first_batch_pair_sha256']
    result['epochs_differ'] = reps['A']['epoch1']['pair_order_sha256'] != reps['A']['epoch2']['pair_order_sha256']
    result['image_bytes_read'] = 0
    finish(result, ctx, needs_gpu=False)
    result['status'] = 'PASS' if all(same.values()) and result['epochs_differ'] else 'STOP_LOADER_NONDETERMINISM'
    return result


# ================================================================= real TRAIN epoch 1 + reference continuation
def train_mode(build):
    result, ctx = setup(build, 'train', allow_faces=True, needs_gpu=True)
    if ctx is None:
        return result, None
    torch = ctx['torch']
    from methods.common.upstream import upstream_modules
    root = Path(ctx['source']['root']) / ctx['config']['source']['relevant_path']
    reader = rio.CanonicalFaceReader(ctx['storage']['faces_256_root'], ctx['datasets'])
    dataset = rio.IndexedPairDataset(ctx['adapter'], ctx['records'], reader)
    run = run_context(ctx, resume=False)
    result['run_dir'], result['run_id'] = str(run.run_dir), run.run_id
    if run.run_dir.exists():
        result['status'] = 'STOP_EXISTING_QUALIFICATION_ROOT'
        return result, None
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(root, m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        trainer, params = components(ctx, result, modules, dataset)
        result['parameters_initial_sha256'] = {n: p['sha256'] for n, p in params.items()}
        run.open()
        streams = rio.tee_run_logs(run.run_dir)
        try:
            run.log_event('e06c_run_start', {'labels': list(rio.QUALIFICATION_LABELS), 'seeding': result['seeding'],
                                             'precision': result['precision'], 'loader': result['dataloader'],
                                             'relation': result['train_relation']})
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            try:
                summary, losses = trainer.run_epoch(1, run, t0)
            except engine.TrainingStop as stop:
                result.update(status='STOP_TRAINING_GATE', stop=str(stop), completed_steps=trainer.global_step)
                run.close(completion_status='failed', failure_reason=str(stop))
                return result, None
            result['epoch_seconds'] = time.monotonic() - t0
            result['epoch1'] = summary
            result['epoch1_losses'] = losses
            result['counters_after_epoch1'] = dict(ctx['counters'])
            result['optimizer_applications_epoch1'] = trainer.optimizer_applications
            result['backward_calls_epoch1'] = trainer.backward_calls
            result['peak_step_gpu_memory_allocated_bytes'] = trainer.peak_step_gpu_memory_bytes
            result['visualization'] = visualization_point(ctx, trainer, run)     # source order: before the save
            written, entry = trainer.checkpoint_event(1, run)
            result['checkpoint_event'] = verify_checkpoints(torch, trainer, run, written, entry)
            result['visualization']['resume_sidecar_torch_cpu_rng_equals_post_visualization'] = (
                result['checkpoint_event']['resume_state']['torch_cpu_rng_sha256'] ==
                result['visualization']['evidence']['torch_cpu_rng_after_sha256'])
            require(result['visualization']['resume_sidecar_torch_cpu_rng_equals_post_visualization'],
                    'resume sidecar carries the post-visualization CPU RNG')
            result['parameters_end_epoch1_sha256'] = model_sha(trainer.models)
            result['optimizer_state_end_epoch1'] = engine.optimizer_state_hashes(trainer.optimizer, trainer.owned)
            result['rng_end_epoch1'] = engine.rng_hashes(torch)
            result['loader_generator_end_epoch1_sha256'] = rio.sha(trainer.generator.get_state().numpy().tobytes())
            run.close(completion_status='interrupted', summary={
                'failure_reason': 'QUALIFICATION_STOP_AFTER_EPOCH_1 (planned; not a failure)',
                'training_duration_seconds': result['epoch_seconds'],
                'peak_vram_bytes': trainer.peak_step_gpu_memory_bytes,
                'final_or_selected_checkpoint_path': None, 'final_or_selected_checkpoint_sha256': None,
                'labels': list(rio.QUALIFICATION_LABELS),
                'missing_field_reasons': {
                    'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific selection exists',
                    'final_or_selected_checkpoint_sha256': 'QUALIFICATION_ONLY: no scientific selection exists',
                    'seed_level_evaluation_metrics': 'QUALIFICATION_ONLY: not evaluated, not reportable'}})
        finally:
            rio.untee(streams)
        result['run_files'] = run_files(run.run_dir)
        # ---- in-memory reference: exactly ONE epoch-2 global step from the epoch-1 state (not logged)
        trainer.set_modes()
        it = iter(trainer.loader)
        batch = next(it)
        del it
        ref = trainer.global_step_run(batch, 2, 0, capture=True)
        probe_path = run.run_dir / PROBE
        probe_path.parent.mkdir()
        torch.save({'gradients': ref['probe']['gradients'], 'parameters_after': ref['probe']['parameters_after'],
                    'labels': list(rio.QUALIFICATION_LABELS), 'kind': 'QUALIFICATION_RESUME_PROBE_NOT_A_CHECKPOINT'},
                   probe_path)
        reference = {'status': 'PASS', 'qualification_seed': SEED, 'labels': list(rio.QUALIFICATION_LABELS),
                     'role': 'UNINTERRUPTED_IN_MEMORY_REFERENCE_CONTINUATION', 'epoch': 2, 'steps_executed': 1,
                     'logged_to_metrics': False, 'run_dir': str(run.run_dir),
                     'from_state': 'in-memory state at the end of epoch 1 (after the checkpoint event)',
                     'step': public_probe(ref), 'probe_file': {'path': PROBE, 'sha256': sha256_file(probe_path),
                                                               'bytes': probe_path.stat().st_size},
                     'optimizer_applications_total': trainer.optimizer_applications,
                     'pid': os.getpid()}
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    finish(result, ctx)
    c = ctx['counters']
    require(trainer.optimizer_applications == 38 and trainer.backward_calls == 36 * 12 + 10 + 12 and
            c['backward_calls'] == c['autograd_backward_calls'] == 454 and c['zero_grad_calls'] == 38 and
            c['zero_grad_set_to_none_values'] == [False], 'accounting: 37 + 1 steps, 442 + 12 backward')
    result['status'] = 'PASS'
    reference['counters_total_process'] = dict(c)
    return result, reference


def visualization_point(ctx, trainer, run):
    """Execute the upstream epoch-1 block through the engine and prove its RNG semantics independently.

    Proof: a separate torch.Generator restored to the pre-block CPU state and drawing two
    zeros(240,128).normal_(0,1) in source order reproduces both noise tensors and ends in exactly
    the global CPU state observed after the block. CUDA RNG, parameters and Adam are unchanged.
    """
    torch = ctx['torch']
    before = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
              'params': model_sha(trainer.models),
              'adam': engine.optimizer_state_hashes(trainer.optimizer, trainer.owned)}
    evidence = trainer.visualize(1, run)
    after_cpu = torch.get_rng_state()
    g = torch.Generator()
    g.set_state(before['cpu'])
    replay = [torch.zeros(*rio.VISUALIZATION_NOISE_SHAPE).normal_(0, 1, generator=g) for _ in range(2)]
    proof = {
        'replay_noise_sha256_equal': engine.tsha(replay[0]) == evidence['noise_sha256'],
        'replay_noise_s_sha256_equal': engine.tsha(replay[1]) == evidence['noise_s_sha256'],
        'replay_end_state_equals_global_cpu_state': torch.equal(g.get_state(), after_cpu),
        'cpu_rng_advanced': not torch.equal(before['cpu'], after_cpu),
        'cuda_rng_unchanged': all(torch.equal(a, b) for a, b in zip(before['cuda'], torch.cuda.get_rng_state_all())),
        'parameters_unchanged': model_sha(trainer.models) == before['params'],
        'adam_state_unchanged': engine.optimizer_state_hashes(trainer.optimizer, trainer.owned) == before['adam']}
    require(all(proof.values()), 'visualization RNG semantics: ' + json.dumps(proof))
    return {'status': 'UPSTREAM_VISUALIZATION_RNG_SEMANTICS_PRESERVED', 'executed': 'FULL_BLOCK_WITH_IMAGE_GRIDS',
            'source': 'train_generator.py:197-211 (after the epoch loop, before save_checkpoint at :214)',
            'evidence': evidence, 'proof': proof,
            'training_consumed_no_global_cpu_rng': torch.equal(ctx['cpu_rng_after_seeding'], before['cpu']),
            'torch_cpu_rng_after_seeding_sha256': rio.sha(ctx['cpu_rng_after_seeding'].numpy().tobytes())}


def verify_checkpoints(torch, trainer, run, written, entry):
    """Official files reload to the in-memory modules; the sidecar reloads (weights_only) to the in-memory state."""
    out = {'official': [], 'resume_state': None}
    for w in written:
        p = run.run_dir / w['path']
        blob = torch.load(p, map_location='cuda', weights_only=False)   # this process's own pinned-writer file
        model = trainer.models[w['model']]
        loaded, live = blob['model'].state_dict(), model.state_dict()
        same = set(loaded) == set(live) and all(torch.equal(loaded[k], live[k]) for k in live)
        require(set(blob) == {'epoch', 'model'} and blob['epoch'] == 1 and same and
                type(blob['model']) is type(model) and type(blob['model'].module) is type(model.module),
                'official checkpoint content ' + w['path'])
        out['official'].append({**{k: w[k] for k in ('path', 'model', 'epoch', 'global_step', 'file_size_bytes',
                                                     'sha256', 'checkpoint_type', 'selected_for_final',
                                                     'selection_reason')},
                                'sha256_recomputed': sha256_file(p), 'keys': sorted(blob),
                                'module_class': type(blob['model'].module).__name__,
                                'state_dict_equal_in_memory': same})
        del blob
    p = run.run_dir / entry['path']
    state = torch.load(p, map_location=None, weights_only=True)
    ok_models = all(torch.equal(v, trainer.models[n].state_dict()[k]) for n, sd in state['models'].items()
                    for k, v in sd.items())
    live_opt = trainer.optimizer.state_dict()
    ok_opt = (state['optimizer']['param_groups'] == live_opt['param_groups'] and
              all(all(torch.equal(state['optimizer']['state'][i][k], live_opt['state'][i][k]) for k in live_opt['state'][i])
                  for i in live_opt['state']))
    ok_rng = (torch.equal(state['rng']['torch_cpu'], torch.get_rng_state()) and
              all(torch.equal(a, b) for a, b in zip(state['rng']['torch_cuda'], torch.cuda.get_rng_state_all())) and
              torch.equal(state['loader_generator'], trainer.generator.get_state()))
    require(ok_models and ok_opt and ok_rng and state['completed_epoch'] == 1 and state['global_step'] == 37,
            'resume sidecar equals the in-memory state')
    out['resume_state'] = {'path': entry['path'], 'sha256': entry['sha256'], 'sha256_recomputed': sha256_file(p),
                           'file_size_bytes': entry['file_size_bytes'], 'weights_only_load': True,
                           'completed_epoch': state['completed_epoch'], 'global_step': state['global_step'],
                           'inventory': sorted(state), 'models': sorted(state['models']),
                           'optimizer_state_entries': len(state['optimizer']['state']),
                           'rng': sorted(state['rng']), 'netIP': state['netIP'],
                           'torch_cpu_rng_sha256': rio.sha(state['rng']['torch_cpu'].numpy().tobytes()),
                           'equal_in_memory': {'models': ok_models, 'optimizer': ok_opt, 'rng_and_loader': ok_rng},
                           'scientific_checkpoint': state['scientific_checkpoint'],
                           'selection_candidate': state['selection_candidate']}
    return out


def run_files(run_dir):
    rows = {}
    for p in sorted(run_dir.rglob('*')):
        if p.is_file() and not p.name.startswith('.'):
            rows[str(p.relative_to(run_dir))] = {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    return rows


# ================================================================= fresh-process resume
def resume_mode(build):
    result, ctx = setup(build, 'resume', allow_faces=True, needs_gpu=True)
    if ctx is None:
        return result
    torch = ctx['torch']
    from methods.common.upstream import upstream_modules
    root = Path(ctx['source']['root']) / ctx['config']['source']['relevant_path']
    reader = rio.CanonicalFaceReader(ctx['storage']['faces_256_root'], ctx['datasets'])
    dataset = rio.IndexedPairDataset(ctx['adapter'], ctx['records'], reader)
    run = run_context(ctx, resume=True)
    result['run_dir'], result['run_id'] = str(run.run_dir), run.run_id
    require(run.run_dir.is_dir(), 'qualification run directory from the train process')
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(root, m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        trainer, _ = components(ctx, result, modules, dataset)
        metrics_before = run.path('metrics').read_bytes()
        run.open()
        streams = rio.tee_run_logs(run.run_dir)
        try:
            path, entry = rio.resume_entry(run.run_dir, 'checkpoints/' + rio.resume_state_name(1))
            restored = trainer.load_resume_state(path, run)
            agreement = rio.reconcile_metrics(run.path('metrics'), completed_epoch=restored['completed_epoch'],
                                              global_step=restored['global_step'])
            run.log_event('e06c_resume_reconciliation', {'resume_state': entry['path'],
                                                         'resume_state_sha256': entry['sha256'], **restored,
                                                         **agreement, 'overwrite': False,
                                                         'selection': 'explicit path (no latest-file discovery)'})
            result['resume'] = {'entry': entry, 'restored': restored, 'metrics_agreement': agreement}
            result['parameters_restored_sha256'] = model_sha(trainer.models)
            result['optimizer_state_restored'] = engine.optimizer_state_hashes(trainer.optimizer, trainer.owned)
            result['rng_restored'] = engine.rng_hashes(torch)
            result['loader_generator_restored_sha256'] = rio.sha(trainer.generator.get_state().numpy().tobytes())
            trainer.set_modes()
            it = iter(trainer.loader)
            batch = next(it)
            del it
            t0 = time.monotonic()
            r = trainer.global_step_run(batch, 2, 0, capture=True)
            run.log_epoch(rio.step_record(
                epoch=2, global_step=r['global_step'], iteration=0, learning_rate=r['learning_rate'], losses=r['losses'],
                total_loss=r['total_loss'], wall_clock_seconds=time.monotonic() - t0, gpu_memory_bytes=r['gpu_memory_bytes'],
                batch_size=r['batch_size'], chunk_sizes=r['chunk_sizes'], batch_pair_sha256=r['batch_pair_sha256'],
                epsilon_sha256=r['epsilon_sha256'], mode=MODE, seed=SEED,
                extra={'step_seconds': r['step_seconds'], 'owned_gradients_finite': r['owned_gradients']['finite'],
                       'optimizer_applications_total': trainer.optimizer_applications,
                       'resumed_process': True}))
            run.close(completion_status='interrupted', summary={
                'failure_reason': 'QUALIFICATION_STOP_AFTER_RESUME_PROBE_STEP (planned; not a failure)',
                'labels': list(rio.QUALIFICATION_LABELS),
                'missing_field_reasons': {
                    'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific selection exists',
                    'final_or_selected_checkpoint_sha256': 'QUALIFICATION_ONLY: no scientific selection exists',
                    'seed_level_evaluation_metrics': 'QUALIFICATION_ONLY: not evaluated, not reportable'}})
        finally:
            rio.untee(streams)
        metrics_after = run.path('metrics').read_bytes()
        result['metrics_append_only'] = metrics_after.startswith(metrics_before) and len(metrics_after) > len(metrics_before)
        result['metrics_records_after'] = [json.loads(x).get('record_type') for x in metrics_after.decode().splitlines()]
        result['step'] = public_probe(r)
        reference = torch.load(run.run_dir / PROBE, map_location='cpu', weights_only=True)
        result['difference_vs_reference_probe'] = {
            'gradients': tensor_diff(torch, reference['gradients'], r['probe']['gradients']),
            'parameters_after': tensor_diff(torch, reference['parameters_after'], r['probe']['parameters_after'])}
        result['run_files'] = run_files(run.run_dir)
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    finish(result, ctx)
    c = ctx['counters']
    require(trainer.optimizer_applications == 38 and c['backward_calls'] == 12 and c['zero_grad_calls'] == 1,
            'resume accounting: one step, 12 backward')
    result['status'] = 'PASS'
    return result


def tensor_diff(torch, a, b):
    require(list(a) == list(b), 'probe tensor inventory')
    dot = na = nb = nd = 0.0
    bitwise, max_abs = 0, 0.0
    for k in a:
        x, y = a[k].double(), b[k].double()
        dot += float((x * y).sum()); na += float(x.pow(2).sum()); nb += float(y.pow(2).sum())
        nd += float((x - y).pow(2).sum()); max_abs = max(max_abs, float((x - y).abs().max()))
        bitwise += int(torch.equal(a[k], b[k]))
    return {'tensors': len(a), 'bitwise_equal_tensors': bitwise, 'max_abs_diff': max_abs,
            'cosine': dot / max((na * nb) ** 0.5, 1e-300), 'relative_l2': nd ** 0.5 / max(na ** 0.5, 1e-300)}


# ================================================================= comparison (torch-free; also re-derived by the preflight)
BITWISE_FIELDS = (('batch_pair_sha256',), ('batch_size',), ('chunk_sizes',), ('epsilon_sha256',), ('losses',),
                  ('total_loss',), ('owned_gradients',), ('netCls_grad_nonzero',), ('probe', 'pass1'),
                  ('probe', 'gradient_sha256'), ('probe', 'gradient_nonzero_tensors'), ('probe', 'optimizer_state'),
                  ('probe', 'parameters_before_sha256'), ('probe', 'parameters_after_sha256'), ('probe', 'rng_before'),
                  ('probe', 'rng_before_step'), ('probe', 'rng_after'), ('probe', 'surrogate_sums'),
                  ('probe', 'sum_of_chunk_objectives'), ('global_step',), ('epoch',), ('learning_rate',))


def compare(reference, fresh):
    def get(d, path):
        for k in path:
            d = d[k]
        return d
    rows = {'.'.join(p): get(reference['step'], p) == get(fresh['step'], p) for p in BITWISE_FIELDS}
    diff = fresh['difference_vs_reference_probe']
    rows['gradient_tensors_bitwise'] = diff['gradients']['bitwise_equal_tensors'] == diff['gradients']['tensors'] == 55
    rows['parameter_tensors_bitwise'] = (diff['parameters_after']['bitwise_equal_tensors'] ==
                                         diff['parameters_after']['tensors'] == 55)
    return {'fields': rows, 'all_bitwise_equal': all(rows.values()), 'quantified': diff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('loader', 'train', 'resume'), required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT) and build.parts[-2:] == BUILD_PARTS, 'dedicated build root')
    build.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp == SHORT_TMP and tmp.is_dir() and not tmp.is_symlink() and tmp.resolve() == tmp,
            'TMPDIR is the dedicated short directory ' + str(SHORT_TMP))
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    outputs = {}
    if args.mode == 'loader':
        outputs['loader'] = loader_mode(build)
    elif args.mode == 'train':
        outputs['train'], reference = train_mode(build)
        if reference is not None:
            outputs['reference'] = reference
    else:
        outputs['resume'] = resume_mode(build)
        ref_path = build / OUTPUTS['reference']
        if outputs['resume']['status'] == 'PASS':
            fresh = json.loads(json.dumps(outputs['resume'], sort_keys=True))
            cmp = compare(json.loads(ref_path.read_text()), fresh)
            outputs['resume']['comparison_vs_reference'] = cmp
            if not cmp['all_bitwise_equal']:
                outputs['resume']['status'] = 'STOP_RESUME_MISMATCH'
    ended = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    status = 'PASS'
    for key, value in outputs.items():
        value['started_utc'], value['ended_utc'] = started, ended
        (build / OUTPUTS[key]).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        if value['status'] != 'PASS':
            status = value['status']
    print(json.dumps({'status': status, 'outputs': [str(build / OUTPUTS[k]) for k in outputs]}))
    sys.exit(EXIT[status])


if __name__ == '__main__':
    main()
