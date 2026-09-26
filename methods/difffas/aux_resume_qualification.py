"""M6D6f E07c auxiliary exact epoch-boundary resume qualification (Amendment A8): QUALIFICATION_ONLY.

Every process is a FRESH OS process seeded with qualification seed 60606 (never the auxiliary seed
42 or an experiment seed). Each one drives the SAME production engine (methods/difffas/aux_runner.py)
and the SAME A8 driver (methods/difffas/aux_resume.py::LogicalRun) that tools/run_e07c_aux.py uses,
on real TRAIN data (split_v1 TRAIN relation + K7 class map + TRAIN canonical faces), behind the
M6D6e audit-hook firewall (unchanged; reused), which denies every other manifest, every non-TRAIN
face, faces_256 enumeration, <runtime_root>/runs and every other benchmark data root.

  reference      fresh root q60606-<run_id>-reference: epoch-0 sidecar, full epoch 1 (step 1 probed),
                 epoch-1 whole-module save + committed epoch-1 sidecar, then exactly the FIRST
                 production step of epoch 2 (probed), then a planned qualification stop.
  interrupted    fresh root q60606-<run_id>-interrupted: the same up to the committed epoch-1 sidecar,
                 then exactly ONE epoch-2 step, logged to metrics.jsonl, then a deliberate
                 interruption after the record is fsynced (no epoch-2 sidecar).
  restore        NEW process: resumes the interrupted root from the explicit epoch-1 sidecar (SHA256
                 before torch.load, weights_only=True), reconciles append-only (the logged epoch-2
                 step -> SUPERSEDED_BY_RESUME_ROLLBACK), restores and replays exactly the first step of
                 epoch 2; bitwise comparison with reference AND with the superseded step.
  e0interrupted  fresh root q60606-<run_id>-e0interrupted: epoch-0 sidecar, ONE epoch-1 step logged,
                 deliberate interruption (a crash before epoch 1 completes).
  e0restore      NEW process: resumes it from the epoch-0 sidecar and replays the first B256 step;
                 bitwise comparison with the reference epoch-1 step and the superseded one.

Outputs are QUALIFICATION_ONLY / NOT_A_SCIENTIFIC_CHECKPOINT / NOT_ELIGIBLE_FOR_BANK /
NOT_ELIGIBLE_FOR_DOWNSTREAM / NOT_ELIGIBLE_FOR_REPORTING. Nothing here is a scientific attempt.
"""
import argparse
import gc
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
from methods.difffas import aux_resume as ar  # noqa: E402
from methods.difffas import aux_runner as engine  # noqa: E402
from methods.difffas import aux_runner_io as aio  # noqa: E402
from methods.difffas import aux_runner_qualification as m6d6e  # noqa: E402  (unchanged M6D6e firewall/audit)
from methods.difffas import runtime_qualification as rq  # noqa: E402  (unchanged M6D6a helpers)

SEED = aio.RESUME_QUALIFICATION_SEED
MODE = aio.QUALIFICATION
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6f')
SHORT_TMP = Path('/tmp/gpat-m6d6f')
PROCESSES = ('reference', 'interrupted', 'restore', 'e0interrupted', 'e0restore')
ROOT_LABEL = {'reference': 'reference', 'interrupted': 'interrupted', 'restore': 'interrupted',
              'e0interrupted': 'e0interrupted', 'e0restore': 'e0interrupted'}
PREREQUISITES = {'reference': (), 'interrupted': ('reference',), 'restore': ('reference', 'interrupted'),
                 'e0interrupted': ('reference', 'interrupted', 'restore'),
                 'e0restore': ('reference', 'interrupted', 'restore', 'e0interrupted')}
OUTPUT = 'M6D6F_E07C_AUX_RESUME_{}.json'
ACCESS_LOG = 'M6D6F_E07C_AUX_ACCESS_{}.tsv'
PROBE = 'm6d6f_probe_{}.pt'
# torch.save / torch.load budget per process (sidecars, whole-module save, probes)
SAVES = {'reference': 4, 'interrupted': 4, 'restore': 1, 'e0interrupted': 2, 'e0restore': 1}
LOADS = {'reference': 0, 'interrupted': 0, 'restore': 3, 'e0interrupted': 0, 'e0restore': 3}
EXIT = {'PASS': 0, 'STOP_TRAINING_GATE': 3, 'STOP_RESOURCE_CONTAMINATION': 4,
        'BLOCKED_BY_AUX_RESUME_NONDETERMINISM': 6, 'STOP_EXISTING_QUALIFICATION_ROOT': 7, 'STOP_RESUME_GATE': 8}
STEP_E1S1, STEP_E2S1 = 1, aio.STEPS_PER_EPOCH + 1
require = aio.require


class QualificationInterruption(RuntimeError):
    """Deliberate, planned qualification interruption right after a step record was fsynced."""


class InterruptAfterLogged:
    """run_epoch's ctx: every call is forwarded; after the record of `global_step` is appended, interrupt."""
    def __init__(self, ctx, global_step):
        self._ctx, self._step = ctx, global_step

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def log_epoch(self, record):
        self._ctx.log_epoch(record)                      # appended + flushed + fsynced (RunContext._append)
        if record['global_step'] == self._step:
            raise QualificationInterruption(f'deliberate qualification interruption after logged step {self._step}')


# ================================================================= setup (M6D6e pattern, M6D6f roots/seed)
def setup(build, process):
    storage = aio.faces_root_from_exec_config()
    runtime_root = Path(storage['runtime_root'])
    require(build == runtime_root.joinpath(*BUILD_PARTS), 'dedicated build root')
    qroot = runtime_root.joinpath(*aio.RESUME_QUALIFICATION_PARTS)
    access_log = build / ACCESS_LOG.format(process)
    access_fd = os.open(access_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)
    firewall = m6d6e.Firewall(runtime_root, storage['faces_256_root'], [build, qroot, SHORT_TMP], [build, qroot],
                              allow_faces=False, access_fd=access_fd)
    sys.addaudithook(firewall)
    import pyarrow.parquet as pq
    guard = m6d6e.PyarrowGuard(pq, ROOT / aio.SPLIT_MANIFEST)
    contract = aio.load_contract()
    ids = dict(aio.identities(contract), **ar.a8_identities())
    source, semantics, adapter, contracts = rq.source_identity()
    config = adapter.config
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']) and
            source['worktree_status'] == '', 'pinned source identity, clean')
    from methods.difffas import execution_policy as ep
    from methods.difffas.aux_training_qualification import upstream_training_semantics
    policy = ep.load_policy(config)
    require(policy['sha256'] == ids['a7_overlay_sha256'], 'A7 identity')
    a8 = ar.load_policy()
    upstream = upstream_training_semantics(source)
    require({k: upstream['loader'][k] for k in ('batch_size', 'shuffle', 'num_workers', 'drop_last')} ==
            {'batch_size': 256, 'shuffle': True, 'num_workers': 6, 'drop_last': True} and upstream['epochs'] == 200,
            'pinned pretrain_classifier.py loader/epochs')
    records, datasets, population = aio.read_train_population(pq=pq)
    firewall.samples, firewall.allow_faces = dict(datasets), True
    result = {'milestone': 'M6D6f', 'process': process, 'label': 'QUALIFICATION_ONLY', 'qualification_seed': SEED,
              'auxiliary_encoder_training_seed': None, 'experiment_seed': None, 'scientific_seed_consumed': False,
              'scientific_attempt': False,
              'seed_note': '60606 seeds this qualification only; auxiliary 42 and main 42/1337/2026 are not used',
              'labels': list(aio.QUALIFICATION_LABELS), 'runner_mode': MODE, 'storage': storage, 'identities': ids,
              'a8': {'overlay_sha256': a8['sha256'], 'document_sha256': a8['document_sha256']},
              'population': population, 'pyarrow_calls': guard.calls, 'source_before': source,
              'executable_semantics': semantics, 'upstream_training_semantics': upstream,
              'a7': {'overlay_sha256': policy['sha256'], 'document': policy['policy']['amendment_document']},
              'code_sha256': ar.code_hashes(), 'harness_sha256': sha256_file(Path(__file__)), 'pid': os.getpid(),
              'launch_environment': {k: os.environ.get(k) for k in (
                  'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
                  'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TMPDIR')}}
    require(result['launch_environment']['PYTHONHASHSEED'] == str(SEED), f'PYTHONHASHSEED={SEED}')
    result['gpu_before'] = m6d6e.gpu_snapshot()
    foreign = [p for p in result['gpu_before']['compute_processes'] if int(p['pid']) != os.getpid()]
    result['resource_clean'] = not foreign and result['gpu_before']['used_mib'] <= m6d6e.CLEAN_GPU_MAX_USED_MIB
    if not result['resource_clean']:
        result.update(status='STOP_RESOURCE_CONTAMINATION', foreign_compute_processes=foreign)
        return result, None
    import inspect
    import torch
    import torchvision.transforms as transforms
    rq.TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(inspect.signature(torch.load).parameters['weights_only'].default)
    counters = m6d6e.new_counters()
    m6d6e.instrument(torch, counters, saves_allowed=SAVES[process], loads_allowed=LOADS[process])
    load_calls = []
    counted_load = torch.load

    def recorded_load(*a, **k):                        # every load must be weights_only=True (sidecar + probes)
        load_calls.append({'weights_only': k.get('weights_only', 'DEFAULT'), 'map_location': repr(k.get('map_location'))})
        require(k.get('weights_only') is True, 'torch.load without weights_only=True is forbidden in M6D6f')
        return counted_load(*a, **k)
    torch.load = recorded_load
    result['precision'] = engine.configure_precision(torch, config)
    lock = json.loads((ROOT / engine.LOCK_PATH).read_text())
    env = rq.environment()
    diff = {k: {'lock': v, 'runtime': env.get(k)} for k, v in lock['identity'].items()
            if k != 'launch_environment' and env.get(k) != v}
    require(not diff, 'runtime identity equals environment lock: ' + json.dumps(diff))
    strip = lambda d: {k: v for k, v in d.items() if k not in ('PYTHONHASHSEED', 'TMPDIR')}
    require(strip(env['launch_environment']) == strip(lock['identity']['launch_environment']), 'launch env = lock')
    result['environment_before'] = env
    ctx = dict(torch=torch, transforms=transforms, config=config, records=records, datasets=datasets, ids=ids,
               firewall=firewall, counters=counters, runtime_root=runtime_root, storage=storage, source=source,
               access_log=access_log, access_fd=access_fd, env=env, guard=guard, load_calls=load_calls)
    return result, ctx


# ================================================================= probes
class Probe:
    """Bitwise capture of one production step: batch, input, labels, loss, gradients, parameters, momentum, RNG."""
    def __init__(self, torch, trainer, targets):
        self.torch, self.trainer, self.targets = torch, trainer, set(targets)
        self.batches, self.tensors, self.summary = {}, {}, {}
        original = trainer.step

        def step(batch, epoch, iteration, on_stage=None):       # observe the batch the engine receives
            if trainer.global_step + 1 in self.targets:
                x, y, idx = batch
                self.batches[trainer.global_step + 1] = {
                    'input_sha256': engine.tsha(x), 'input_shape': list(x.shape), 'input_dtype': str(x.dtype),
                    'labels_sha256': engine.tsha(y), 'labels': y.tolist(),
                    'indices_sha256': aio.sha(json.dumps(idx.tolist()).encode())}
            return original(batch, epoch, iteration, on_stage)
        trainer.step = step                                      # instance attribute; run_epoch calls self.step

    def on_step(self, r):
        g = r['global_step']
        if g not in self.targets:
            return
        torch, t = self.torch, self.trainer
        grads = {n: p.grad.detach().cpu().clone() for n, p in t.named if p.grad is not None}
        params = {n: p.detach().cpu().clone() for n, p in t.named}
        buffers = {n: b.detach().cpu().clone() for n, b in t.model.named_buffers()}
        moments = {n: t.optimizer.state[p]['momentum_buffer'].detach().cpu().clone() for n, p in t.named
                   if p in t.optimizer.state}
        rng = {'torch_cpu': torch.get_rng_state(), 'torch_cuda_device': torch.cuda.get_rng_state(ar.DEVICE)}
        self.tensors[g] = {'gradients': grads, 'parameters_after': params, 'buffers_after': buffers,
                           'momentum_after': moments, 'rng_after': rng}
        self.summary[g] = dict(self.batches[g], epoch=r['epoch'], global_step=g, iteration=r['iteration'],
                               batch_size=r['batch_size'], sample_ids=r['sample_ids'],
                               sample_ids_sha256=r['batch_sample_sha256'], class_histogram=r['class_histogram'],
                               loss=r['loss'], loss_hex=r['loss_hex'],
                               gradients_sha256=ar.tensor_digest(grads), gradient_tensors=len(grads),
                               parameters_after_sha256=ar.tensor_digest(params), parameter_tensors=len(params),
                               buffers_after_sha256=ar.tensor_digest(buffers),
                               momentum_after_sha256=ar.tensor_digest(moments), momentum_buffers=len(moments),
                               rng_after=ar.rng_digest(torch),
                               without_grad=sorted(n for n, p in t.named if p.grad is None))

    def save(self, build, process):
        path = build / PROBE.format(process)
        require(not path.exists(), 'fresh probe')
        self.torch.save({'kind': 'M6D6F_BITWISE_PROBE_NOT_A_CHECKPOINT', 'labels': list(aio.QUALIFICATION_LABELS),
                         'steps': self.tensors}, path)
        return {'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def tensor_diff(torch, a, b):
    require(list(a) == list(b), 'probe tensor inventory')
    bitwise, max_abs = 0, 0.0
    for k in a:
        bitwise += int(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]))
        if a[k].is_floating_point():
            max_abs = max(max_abs, float((a[k].double() - b[k].double()).abs().max()))
        elif not torch.equal(a[k], b[k]):
            max_abs = max(max_abs, float('inf'))
    return {'tensors': len(a), 'bitwise_equal_tensors': bitwise, 'max_abs_diff': max_abs}


FIELDS = ('sample_ids', 'sample_ids_sha256', 'indices_sha256', 'input_sha256', 'input_shape', 'input_dtype',
          'labels_sha256', 'labels', 'class_histogram', 'loss_hex', 'gradients_sha256', 'parameters_after_sha256',
          'buffers_after_sha256', 'momentum_after_sha256', 'without_grad', 'rng_after')


def compare_steps(reference, fresh):
    """Torch-free field comparison of two probe summaries (re-derived by the static preflight)."""
    rows = {f: reference[f] == fresh[f] for f in FIELDS}
    return {'fields': rows, 'all_bitwise_equal': all(rows.values()), 'tolerance': 'NONE (bitwise)'}


def compare_tensors(torch, ref_steps, fresh_steps, g):
    a, b = ref_steps[g], fresh_steps[g]
    out = {k: tensor_diff(torch, a[k], b[k]) for k in ('gradients', 'parameters_after', 'buffers_after', 'momentum_after')}
    out['rng_after'] = {k: bool(torch.equal(a['rng_after'][k], b['rng_after'][k])) for k in a['rng_after']}
    out['all_bitwise_equal'] = (all(v['bitwise_equal_tensors'] == v['tensors'] and v['max_abs_diff'] == 0.0
                                    for k, v in out.items() if k != 'rng_after') and all(out['rng_after'].values()))
    return out


# ================================================================= process bodies
def open_run(ctx, label, resume):
    env, reasons = engine.environment_record(ctx['torch'])
    return engine.E07cAuxRunContext(mode=MODE, seed=SEED, runtime_root=ctx['runtime_root'], environment=env,
                                    missing_environment_reasons=reasons, identities=ctx['ids'],
                                    command_line=' '.join([sys.executable] + sys.argv), resume=resume, label=label)


def metrics_view(run):
    raw, records = ar.read_records(run.path('metrics'))
    logical = ar.logical_trajectory(records)
    steps = [r['global_step'] for r in records if r.get('record_type') == 'epoch']
    return {'bytes': len(raw), 'sha256': aio.sha(raw), 'records': len(records),
            'physical_step_records': len(steps), 'physical_global_steps_tail': steps[-3:],
            'logical_step_records': len(logical), 'logical_global_steps_tail': [r['global_step'] for r in logical][-3:],
            'events': [r['event'] for r in records if r.get('record_type') == 'event']}


def run_process(build, process):
    result, ctx = setup(build, process)
    if ctx is None:
        return result
    torch = ctx['torch']
    label = ROOT_LABEL[process]
    resume = process in ('restore', 'e0restore')
    run = open_run(ctx, label, resume)
    result.update(run_dir=str(run.run_dir), run_id=run.run_id, root_label=label, continuation_process=resume)
    require(not run.run_dir.is_relative_to(ctx['runtime_root'] / 'runs'), 'qualification root outside runs/')
    if run.run_dir.exists() != resume:
        result['status'] = 'STOP_EXISTING_QUALIFICATION_ROOT'
        return result
    if resume:                                              # a fresh (non-resume) context must refuse this root
        probe_ctx = open_run(ctx, label, False)
        try:
            probe_ctx.open()
            refused = False
        except engine.RunDirectoryError as exc:
            refused = 'never overwritten' in str(exc)
        result['fresh_context_refuses_existing_root'] = refused
        require(refused, 'a non-resume context must refuse the existing root')
        result['metrics_before_resume'] = metrics_view(run)
    result['seeding'] = engine.seed_process(torch, MODE, SEED, ctx['config'])
    result['rng_after_seeding'] = ar.rng_digest(torch)
    targets = {'reference': (STEP_E1S1, STEP_E2S1), 'interrupted': (STEP_E2S1,), 'restore': (STEP_E2S1,),
               'e0interrupted': (STEP_E1S1,), 'e0restore': (STEP_E1S1,)}[process]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with engine.production_components(torch, ctx['transforms'], ctx['config'], ctx['records'], ctx['datasets'],
                                          ctx['storage']['faces_256_root'], MODE, SEED) as trainer:
            result['components'] = trainer.evidence
            result['constructed_state'] = {'parameters_sha256': ar.tensor_digest(
                {n: p.detach().cpu() for n, p in trainer.named}), 'rng': ar.rng_digest(torch)}
            logical = ar.LogicalRun(torch, trainer, run, ctx['config'], storage=ctx['storage'],
                                    precision=result['precision'])
            if resume:
                committed = ar.read_index(run.run_dir)['committed']
                supplied = str(run.run_dir / committed['path'])
                result['explicit_resume_path'] = supplied
                verified = logical.verify_before_open(supplied)
                result['verified_sidecar'] = {'path': str(verified['path'].relative_to(run.run_dir)),
                                              'entry': verified['entry'], 'sha256_verified_before_load': True,
                                              'weights_only': True,
                                              'torch_load_calls_so_far': list(ctx['load_calls'])}
            probe = Probe(torch, trainer, targets)
            run.open()
            streams = aio.tee_run_logs(run.run_dir)
            try:
                if resume:
                    result['reconciliation'] = logical.begin_resume()
                    result['rng_after_restore'] = ar.rng_digest(torch)
                else:
                    run.log_event('e07c_aux_run_start', {'labels': list(aio.QUALIFICATION_LABELS), 'milestone': 'M6D6f',
                                                         'process': process, 'seeding': result['seeding'],
                                                         'precision': result['precision'],
                                                         'components': trainer.evidence, 'resume_policy': 'A8'})
                    result['rng_before_epoch0_sidecar'] = ar.rng_digest(torch)
                    result['epoch0_sidecar'] = logical.begin_fresh()
                    result['rng_after_epoch0_sidecar'] = ar.rng_digest(torch)
                    require(result['rng_before_epoch0_sidecar'] == result['rng_after_epoch0_sidecar'],
                            'writing the sidecar consumes no RNG')
                t0 = time.monotonic()
                stop_step = max(targets)
                if process in ('reference', 'interrupted'):
                    summary, losses = trainer.run_epoch(1, run, t0, probe.on_step)
                    result['epoch1'] = summary
                    result['epoch1_losses_hex'] = [float(v).hex() for v in losses]
                    ck, entry = logical.commit_boundary(1)
                    result['epoch1_checkpoint_event'] = ck
                    result['epoch1_sidecar'] = entry
                    result['rng_at_epoch1_boundary'] = ar.rng_digest(torch)
                epoch = trainer.completed_epoch + 1
                result['rng_before_iterator'] = ar.rng_digest(torch)
                try:
                    trainer.run_epoch(epoch, InterruptAfterLogged(run, stop_step), t0, probe.on_step)
                    raise RuntimeError('the planned interruption did not happen')
                except QualificationInterruption as stop:
                    result['interruption'] = {'kind': 'DELIBERATE_QUALIFICATION_INTERRUPTION', 'message': str(stop),
                                              'after_logged_global_step': stop_step, 'epoch': epoch,
                                              'epoch_completed': False, 'sidecar_written_for_epoch': False,
                                              'utc': ar.utc()}
                gc.collect()                                     # the interrupted iterator's workers are shut down
                result['step_accounting'] = logical.summary()
                result['index_after'] = ar.read_index(run.run_dir)
                run.close(completion_status='interrupted', summary={
                    'failure_reason': f'QUALIFICATION_INTERRUPTION_AFTER_LOGGED_STEP_{stop_step} (planned; not a failure)',
                    'training_duration_seconds': time.monotonic() - t0,
                    'peak_vram_bytes': trainer.peak_step_gpu_memory_bytes,
                    'final_or_selected_checkpoint_path': None, 'final_or_selected_checkpoint_sha256': None,
                    'labels': list(aio.QUALIFICATION_LABELS), 'step_accounting': result['step_accounting'],
                    'missing_field_reasons': {
                        'final_or_selected_checkpoint_path': 'QUALIFICATION_ONLY: no scientific final state exists',
                        'final_or_selected_checkpoint_sha256': 'QUALIFICATION_ONLY: no scientific final state exists',
                        'seed_level_evaluation_metrics': 'QUALIFICATION_ONLY: not evaluated, not reportable'}})
            finally:
                aio.untee(streams)
            result['steps'] = json.loads(json.dumps({str(g): probe.summary[g] for g in sorted(probe.summary)}))
            result['probe'] = probe.save(build, process)
        result['metrics_after'] = metrics_view(run)
        if resume:
            before, after = result['metrics_before_resume'], run.path('metrics').read_bytes()
            result['metrics_append_only'] = {'prefix_bytes': before['bytes'],
                                             'prefix_sha256_unchanged': aio.sha(after[:before['bytes']]) == before['sha256'],
                                             'grew_by_bytes': len(after) - before['bytes']}
            require(result['metrics_append_only']['prefix_sha256_unchanged'], 'metrics.jsonl never truncated/rewritten')
        result['run_files'] = {str(p.relative_to(run.run_dir)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                               for p in sorted(run.run_dir.rglob('*')) if p.is_file() and not p.name.startswith('.')}
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        comparisons = {}
        if resume:
            g = STEP_E2S1 if process == 'restore' else STEP_E1S1
            superseded = 'interrupted' if process == 'restore' else 'e0interrupted'
            mine = probe.tensors                                  # this process's own capture (in memory)
            for other in ('reference', superseded):
                ev = json.loads((build / OUTPUT.format(other.upper())).read_text())
                theirs = torch.load(build / PROBE.format(other), map_location='cpu', weights_only=True)['steps']
                comparisons[other] = {'global_step': g, 'fields': compare_steps(ev['steps'][str(g)], result['steps'][str(g)]),
                                      'tensors': compare_tensors(torch, theirs, mine, g)}
                del theirs
            del mine
            result['comparisons'] = comparisons
    result['load_calls'] = ctx['load_calls']
    m6d6e.finish(result, ctx)
    c = ctx['counters']
    steps_here = {'reference': STEP_E2S1, 'interrupted': STEP_E2S1, 'restore': 1, 'e0interrupted': 1, 'e0restore': 1}[process]
    require(c['sgd_step_calls'] == c['backward_calls'] == c['zero_grad_calls'] == steps_here,
            f'physical optimizer steps in this process = {steps_here}')
    require(c['torch_save_calls'] == SAVES[process] and c['torch_load_calls'] == LOADS[process], 'save/load accounting')
    result['physical_optimizer_steps_this_process'] = steps_here
    result['status'] = 'PASS'
    if comparisons and not all(v['fields']['all_bitwise_equal'] and v['tensors']['all_bitwise_equal']
                               for v in comparisons.values()):
        result['status'] = 'BLOCKED_BY_AUX_RESUME_NONDETERMINISM'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process', choices=PROCESSES, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT) and build.parts[-3:] == BUILD_PARTS, 'dedicated build root')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp == SHORT_TMP and tmp.is_dir() and not tmp.is_symlink() and tmp.resolve() == tmp,
            'TMPDIR is the dedicated short directory ' + str(SHORT_TMP))
    out = build / OUTPUT.format(args.process.upper())
    require(not out.exists(), 'fresh output ' + out.name)
    for pre in PREREQUISITES[args.process]:
        require(json.loads((build / OUTPUT.format(pre.upper())).read_text())['status'] == 'PASS', pre + ' passed first')
    started = ar.utc()
    try:
        result = run_process(build, args.process)
    except ar.ResumeStop as stop:
        result = {'status': 'STOP_RESUME_GATE', 'stop': str(stop), 'process': args.process}
    except engine.TrainingStop as stop:
        result = {'status': 'STOP_TRAINING_GATE', 'stop': str(stop), 'process': args.process}
    result['started_utc'], result['ended_utc'] = started, ar.utc()
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(out)}))
    sys.exit(EXIT[result['status']])


if __name__ == '__main__':
    main()
