"""M6D6j E07c MAIN production-runner QUALIFICATION (seed 60608; experiment_seed = null). QUALIFICATION ONLY.

One fresh GPU process under gpat-m6-e07c drives the SAME production code a scientific seed uses
(main_runner.build_production, MainRunner.iteration, MainRunner.visualize_slot, MainRunner.checkpoint_transition,
main_checkpoint.CheckpointStore, E07cMainRunContext) inside a non-scientific run root
<runtime_root>/qualification/m6d6j/E07c/q60608-<id>/ :

  A  real B=4 production iteration (global step 1) from the real TRAIN DataLoader first batch;
  B  real B=2 tail-shaped iteration (global step 2): two real relation rows through the same dataset/transform and the
     DataLoader's own default_collate (the scientific DataLoader policy is unchanged);
  C  the production visualization slot invoked directly once (labelled; the scientific trigger is only ever
     iters % 1000 == 0), with RNG / buffer / parameter / optimizer / scheduler / EMA before-after evidence;
  D  run_logging_v1 outputs (per-step metrics, events, checkpoint_index, run_summary);
  E  the checkpoint lifecycle with the real full main state at LABELLED qualification steps 10000 -> 20000 -> terminal
     884000 (write, verify, successor-gated prune, protected-terminal refusal), plus one SHA-gated round-trip of the
     qualification's own first file; then qualification cleanup of the remaining bytes (not the M6D6iR policy).

Firewall: real TRAIN canonical faces only (ids bound by A1 relation + split_v1 TRAIN membership), the two frozen
manifests only, the frozen encoder's exact path only, writes only in the qualification/build roots.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.difffas import runtime_qualification as rq  # noqa: E402
from methods.difffas import main_runner_io as mio  # noqa: E402
from methods.difffas import main_checkpoint as mc  # noqa: E402
from methods.difffas import main_runner as mr  # noqa: E402
from methods.difffas import main_graph as mg  # noqa: E402
from methods.difffas import aux_checkpoint as seam  # noqa: E402

MILESTONE = 'M6D6j'
SEED = mio.QUALIFICATION_SEED
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6j')
TAIL_INDICES = (mio.TRAIN_ROWS - 2, mio.TRAIN_ROWS - 1)
LABELLED_STEPS = (10000, 20000)
WEIGHT_SUFFIXES = {'.pt', '.pth', '.pkl', '.ckpt', '.safetensors', '.bin'}
ALLOWED_SUBPROCESSES = ('git', 'nvidia-smi')
ALLOWED_EXACT_ARGV = (['/sbin/ldconfig', '-p'], ['uname', '-p'])
QUALIFIED = ['MAIN_PRODUCTION_RUNNER', 'A1_TRAIN_DATASET_INTEGRATION', 'CANONICAL_FACE_READER', 'GUIDE_MAPPING',
             'RUN_LOGGING_V1_MAIN_RUN', 'MAIN_CHECKPOINT_CADENCE', 'MAIN_TERMINAL_CHECKPOINT_CREATION',
             'MAIN_CHECKPOINT_RETENTION_RUNTIME_INTEGRATION']
NOT_QUALIFIED = ['MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
EXPECTED_VIS = {'model_forwards': 2 * mio.SAMPLE_INITIAL_NOISE, 'encoder_calls': 2, 'items': mio.BATCH_SIZE}


def require(value, message):
    if not value:
        raise RuntimeError('M6D6j gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Firewall:
    """Audit hook for the first real MAIN TRAIN access. Every face open is attributed to a phase and a sample id."""
    WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    EVENTS = ('open', 'os.listdir', 'os.scandir', 'os.mkdir', 'os.remove', 'os.rename', 'shutil.rmtree',
              'subprocess.Popen', seam.EVENT_VERIFIED, seam.EVENT_REJECTED)

    def __init__(self, runtime_root, faces_root, write_roots, frozen_path, weight_root):
        self.runtime_root, self.faces = Path(runtime_root), Path(faces_root)
        self.write_roots = [str(Path(p)) + '/' for p in write_roots]
        self.weight_root = str(Path(weight_root)) + '/'
        self.frozen = str(frozen_path)
        self.relation, self.split = str(ROOT / mio.RELATION), str(ROOT / mio.SPLIT_MANIFEST)
        self.samples, self.phase = {}, 'setup'
        self.events = dict.fromkeys(self.EVENTS, 0)
        self.denied, self.face_opens, self.manifest_opens, self.subprocesses = [], [], {}, []
        self.frozen_opens, self.weight_events, self.sha_events = [], [], []

    @staticmethod
    def in_roots(path, roots):
        return any(path == r[:-1] or path.startswith(r) for r in roots)

    def writing(self, event, args):
        if event in ('os.listdir', 'os.scandir'):
            return False
        if event != 'open':
            return True
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        return bool(flags & self.WRITE_FLAGS) or bool(set(mode) & set('wax+'))

    def check(self, event, path, args):
        p, writing = Path(path), self.writing(event, args)
        if p.is_relative_to(ROOT / 'manifests'):
            cat = {self.relation: 'A1_RELATION', self.split: 'SPLIT_MANIFEST'}.get(path)
            if cat and event == 'open' and not writing:
                self.manifest_opens[cat] = self.manifest_opens.get(cat, 0) + 1
                return None
            return 'manifest outside the two frozen inputs, or enumeration/write'
        if p.is_relative_to(self.faces):
            rel = p.relative_to(self.faces).parts
            if event != 'open' or writing:
                return 'faces_256 enumeration or write'
            sid = rel[1][:-4] if len(rel) == 2 and rel[1].endswith('.png') else None
            if sid is None or self.samples.get(sid) != rel[0]:
                return 'face outside the A1 TRAIN relation ids'
            self.face_opens.append((self.phase, sid))
            return None
        if path == self.frozen:
            if writing:
                return 'write to the frozen auxiliary encoder'
            self.frozen_opens.append(self.phase)
            return None
        if p.is_relative_to(self.runtime_root / 'runs') or p.is_relative_to(ROOT / 'runs'):
            return 'scientific run root'
        if p.is_relative_to(self.runtime_root / 'data') or any(p.is_relative_to(ROOT / d) for d in ('data', 'cache')):
            return 'other benchmark data'
        if p.suffix.lower() in WEIGHT_SUFFIXES or path.endswith('.pt.partial'):
            if not path.startswith(self.weight_root):
                return 'weight file outside the qualification checkpoints/'
            self.weight_events.append((self.phase, event, Path(path).name, writing))
        if p.suffix.lower() == '.parquet':
            return 'parquet outside the frozen inputs'
        if writing and not (path == '/dev/null' or self.in_roots(path, self.write_roots)):
            return 'write outside the qualification/build roots'
        return None

    def __call__(self, event, args):
        if event not in self.events:
            return
        self.events[event] += 1
        if event in (seam.EVENT_VERIFIED, seam.EVENT_REJECTED):
            self.sha_events.append({'event': event.rsplit('.', 1)[1], 'path': str(args[0]), 'sha256': str(args[1])})
            return
        if event == 'subprocess.Popen':
            argv = [os.fsdecode(a) for a in (args[1] or [])] if len(args) > 1 else []
            self.subprocesses.append(argv[:4])
            if Path(argv[0]).name not in ALLOWED_SUBPROCESSES and argv not in [list(a) for a in ALLOWED_EXACT_ARGV]:
                self.denied.append({'event': event, 'path': ' '.join(argv[:4]), 'reason': 'subprocess'})
                raise PermissionError('E07c M6D6j SUBPROCESS FIREWALL: ' + ' '.join(argv[:2]))
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        paths = [args[0]] + ([args[1]] if event == 'os.rename' and len(args) > 1 else [])
        for raw in paths:
            path = os.path.abspath(os.fsdecode(raw))
            why = self.check(event, path, args)
            if why:
                self.denied.append({'event': event, 'path': path, 'reason': why, 'phase': self.phase})
                raise PermissionError('E07c M6D6j FIREWALL: ' + why + ': ' + path)


class PyarrowGuard:
    """pyarrow reads bypass Python open(): only the A1 relation and split_v1 may be read."""
    def __init__(self, pq, allowed):
        self.allowed, self.calls = {str(Path(p)) for p in allowed}, []
        for name in ('read_table', 'read_schema', 'read_metadata'):
            setattr(pq, name, self.wrap(name, getattr(pq, name)))

    def wrap(self, name, original):
        def guarded(source, *args, **kwargs):
            path = str(Path(source))
            require(path in self.allowed, 'pyarrow read outside the frozen inputs: ' + path)
            self.calls.append({'call': name, 'path': os.path.relpath(path, ROOT),
                               'columns': list(kwargs.get('columns') or []), 'filters': repr(kwargs.get('filters'))})
            return original(source, *args, **kwargs)
        return guarded


class Counters:
    """Pass-through counters of the training/serialization calls; unexpected torch.load is refused."""
    def __init__(self, torch, qual_ckpt_dir):
        import tensorfn.optim.lr_scheduler as tl
        self.n = {k: 0 for k in ('backward', 'zero_grad', 'optimizer_step', 'scheduler_step', 'torch_save',
                                 'torch_load_seam', 'torch_load_roundtrip', 'autocast_entries')}
        o = {'save': torch.save, 'load': torch.load, 'backward': torch.autograd.backward,
             'zero': torch.optim.Optimizer.zero_grad, 'step': torch.optim.AdamW.step, 'sched': tl.PhaseScheduler.step}
        n, ckpt = self.n, str(qual_ckpt_dir) + '/'
        import io

        def save(obj, f, *a, **k):
            n['torch_save'] += 1
            return o['save'](obj, f, *a, **k)

        def load(f, *a, **k):
            if isinstance(f, io.BytesIO) and k == {'weights_only': False}:
                n['torch_load_seam'] += 1
            elif isinstance(f, str) and f.startswith(ckpt) and k.get('weights_only') is False:
                n['torch_load_roundtrip'] += 1
            else:
                raise RuntimeError('M6D6j refuses torch.load of ' + repr(f))
            return o['load'](f, *a, **k)

        def backward(*a, **k):
            n['backward'] += 1
            return o['backward'](*a, **k)

        def zero(self_, *a, **k):
            n['zero_grad'] += 1
            return o['zero'](self_, *a, **k)

        def step(self_, *a, **k):
            n['optimizer_step'] += 1
            return o['step'](self_, *a, **k)

        def sched(self_, *a, **k):
            n['scheduler_step'] += 1
            return o['sched'](self_, *a, **k)

        def forbid(*a, **k):
            n['autocast_entries'] += 1
            raise RuntimeError('M6D6j forbids autocast')
        torch.save, torch.load, torch.autograd.backward = save, load, backward
        torch.optim.Optimizer.zero_grad, torch.optim.AdamW.step, tl.PhaseScheduler.step = zero, step, sched
        torch.amp.autocast_mode.autocast.__enter__ = forbid


def rng(torch):
    import numpy as np
    state = np.random.get_state()
    return {'cpu_sha256': sha(torch.get_rng_state().numpy().tobytes()),
            'cuda_sha256': sha(torch.cuda.get_rng_state().numpy().tobytes()),
            'python_random_sha256': sha(repr(random.getstate()).encode()),
            'numpy_sha256': sha(state[1].tobytes() + repr((state[0], *state[2:])).encode())}


def digest(pairs):
    h = hashlib.sha256()
    n = 0
    for name, t in pairs:
        h.update(name.encode() + b'\0' + t.detach().cpu().contiguous().numpy().tobytes())
        n += 1
    return {'sha256': h.hexdigest(), 'tensors': n}


def optimizer_digest(optimizer):
    h = hashlib.sha256()
    for group in optimizer.param_groups:
        h.update(json.dumps({k: v for k, v in group.items() if k != 'params'}, sort_keys=True, default=str).encode())
        for p in group['params']:
            for k, v in sorted(optimizer.state.get(p, {}).items()):
                h.update(k.encode() + v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state(torch, runner):
    return {'model_parameters': digest(runner.model.named_parameters()),
            'model_buffers': digest(runner.model.named_buffers()),
            'ema_parameters': digest(runner.ema.named_parameters()), 'ema_buffers': digest(runner.ema.named_buffers()),
            'encoder_parameters': digest(runner.encoder.named_parameters()),
            'encoder_buffers': digest(runner.encoder.named_buffers()),
            'optimizer_sha256': optimizer_digest(runner.optimizer),
            'scheduler': json.loads(json.dumps(runner.scheduler.state_dict(), default=str)), 'rng': rng(torch)}


def step_evidence(torch, runner, before, after, record, batch):
    grads = [(n, p.grad) for n, p in runner.model.named_parameters()]
    present = [(n, g) for n, g in grads if g is not None]
    require(all(bool(torch.isfinite(g).all().item()) for _, g in present), 'finite gradients')
    require(all(p.grad is None for p in runner.encoder.parameters()), 'encoder receives no gradient')
    require(after['encoder_parameters'] == before['encoder_parameters'] and
            after['encoder_buffers'] == before['encoder_buffers'], 'encoder unchanged')
    require(after['model_parameters'] != before['model_parameters'], 'main parameters updated')
    require(all(torch.equal(pe, pm) for (_, pe), (_, pm) in zip(runner.ema.named_parameters(),
                                                               runner.model.named_parameters())),
            'EMA named_parameters == main after decay-0 accumulate (iters < 5000)')
    steps = {float(runner.optimizer.state[p]['step']) for n, p in runner.model.named_parameters() if p.grad is not None}
    return {'batch_shapes': {k: list(batch[k].shape) for k in sorted(batch)}, 'loss': record['train_losses'],
            'learning_rate': record['learning_rate'], 'gpu_memory_bytes': record['gpu_memory_bytes'],
            'grad_tensors_present': len(present), 'grad_tensors_none': len(grads) - len(present),
            'nonzero_grad_tensors': sum(1 for _, g in present if bool((g != 0).any().item())),
            'optimizer_state_step_values': sorted(steps), 'scheduler': after['scheduler'],
            'model_parameters_before': before['model_parameters']['sha256'],
            'model_parameters_after': after['model_parameters']['sha256'],
            'model_buffers_changed': after['model_buffers'] != before['model_buffers'],
            'ema_equals_main': True, 'encoder_unchanged': True, 'cpu_rng_changed': before['rng']['cpu_sha256'] !=
            after['rng']['cpu_sha256'], 'cuda_rng_changed': before['rng']['cuda_sha256'] != after['rng']['cuda_sha256']}


def roundtrip(torch, runner, entry):
    """SHA-gated round-trip of the qualification's OWN first file: exact payload keys and tensors."""
    path = runner.store.exact_path(entry['path'])
    require(mc.file_sha256(path) == entry['sha256'], 'SHA256 before the qualification round-trip load')
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    require(tuple(obj) == mc.PAYLOAD_KEYS, 'payload keys model, ema, scheduler, optimizer, conf')
    live = runner.model.state_dict()
    require(list(obj['model']) == list(live) and all(torch.equal(obj['model'][k], live[k].cpu()) for k in live),
            'model tensors == live state')
    ema = runner.ema.state_dict()
    require(all(torch.equal(obj['ema'][k], ema[k].cpu()) for k in ema), 'EMA tensors == live state')
    require(obj['scheduler'] == runner.scheduler.state_dict(), 'scheduler state == live')
    opt = obj['optimizer']
    require(len(opt['state']) == len(runner.optimizer.state) and opt['param_groups'][0]['lr'] ==
            runner.optimizer.param_groups[0]['lr'], 'optimizer state entries and lr == live')
    require(type(obj['conf']).__name__ == 'DiffusionConfig' and obj['conf'].training.ckpt_path ==
            str(runner.store.ckpt_dir), 'conf = pinned DiffusionConfig with ckpt_path = run checkpoints/')
    out = {'keys': list(obj), 'model_tensors': len(obj['model']), 'ema_tensors': len(obj['ema']),
           'optimizer_state_entries': len(opt['state']), 'conf_type': type(obj['conf']).__name__,
           'model_equal_live': True, 'ema_equal_live': True, 'scheduler_equal_live': True}
    del obj
    return out


def qualify(build):
    t_start = time.monotonic()
    contract = mio.load_contract()
    ids = mio.identities(contract)
    storage = mio.faces_root_from_exec_config()
    runtime_root = Path(storage['runtime_root'])
    require(runtime_root == rq.RUNTIME, 'runtime root from the frozen execution config')
    for k, v in (('PYTHONHASHSEED', str(SEED)), ('CUDA_VISIBLE_DEVICES', '0'), ('PYTHONDONTWRITEBYTECODE', '1'),
                 ('PYTHONNOUSERSITE', '1'), *mio.LAUNCH_ENVIRONMENT.items()):
        require(os.environ.get(k) == v, f'launch environment {k}={v}')
    from methods.common.runlog import git_commit
    from methods.difffas import DiffFASAdapter
    adapter = DiffFASAdapter()
    config = adapter.config
    qid = mio.qualification_id(config['_runtime']['config_sha256'], git_commit())
    run_dir = mio.run_root(runtime_root, mio.QUALIFICATION, SEED, qid)
    frozen = seam.frozen_checkpoint_path(runtime_root, config)
    fw = Firewall(runtime_root, storage['faces_256_root'], [run_dir.parent, build], frozen, run_dir / 'checkpoints')
    sys.addaudithook(fw)
    source = adapter.validate_source()
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']), 'source pin')
    source_evidence = mr.verify_production_source(source)
    import pyarrow.parquet as pq
    guard = PyarrowGuard(pq, [ROOT / mio.RELATION, ROOT / mio.SPLIT_MANIFEST])
    import torch
    from torch.utils.data import default_collate
    counters = Counters(torch, run_dir / 'checkpoints')
    env, reasons = mr.environment_record(torch)
    ctx = mr.E07cMainRunContext(mode=mio.QUALIFICATION, seed=SEED, runtime_root=runtime_root, environment=env,
                                missing_environment_reasons=reasons, identities=ids)
    require(ctx.run_dir == run_dir and not run_dir.is_relative_to(runtime_root / 'runs'), 'non-scientific root')
    stages, result = [], {}

    def on_stage(name, **objs):
        stages.append(name)
        if name == 'dataset':
            fw.samples = dict(objs['reader']._datasets)

    with ctx:
        streams = mio.tee_run_logs(ctx.run_dir)
        try:
            with mr.build_production(torch, mode=mio.QUALIFICATION, seed=SEED, runtime_root=runtime_root,
                                     faces_root=storage['faces_256_root'], ctx=ctx, source=source, config=config,
                                     schedule=mr.Schedule.frozen(), tqdm=mr.quiet_tqdm(), on_stage=on_stage) as runner:
                require(stages == ['seed', 'precision', 'transform', 'dataset', 'dataloader', 'objects', 'encoder_eval'],
                        'A7 construction order ' + json.dumps(stages))
                require(runner.loader.collate_fn is default_collate, 'DataLoader default_collate')
                require([e['sha256'] for e in fw.sha_events] == [seam.OWNER_FROZEN_SHA256] and
                        counters.n['torch_load_seam'] == 1 and len(fw.frozen_opens) == 2, 'frozen encoder via the seam')
                ev = {k: v for k, v in runner.evidence.items() if k != 'upstream_modules'}
                ctx.log_event('e07c_main_qualification_start', ev | {'source': source_evidence})
                ua = runner.evidence['upstream_modules']['unet_autoenc']
                calls = {'model_forwards': 0, 'encoder_calls': 0}
                source_forward = runner.model.forward

                def counted_forward(*a, **k):
                    # pass-through observer: nn.Module.__call__ AND the pinned forward_with_cond_scale (which calls
                    # self.forward directly, bypassing forward hooks; unet_autoenc.py:129,134) both resolve here
                    calls['model_forwards'] += 1
                    return source_forward(*a, **k)
                runner.model.forward = counted_forward
                runner.encoder.register_forward_hook(lambda *a: calls.__setitem__('encoder_calls',
                                                                                 calls['encoder_calls'] + 1))
                # ---------------- A: real B=4 production iteration (global step 1)
                fw.phase = 'b4_step'
                progress_bar = mg.epoch_progress(runner.tqdm, runner.loader)
                before = state(torch, runner)
                t0 = time.monotonic()
                for batch in progress_bar:
                    try:
                        iters = runner.iteration(batch, 0, 0, progress_bar)
                    except torch.cuda.OutOfMemoryError as exc:
                        raise mr.MainMemoryBlocked('BLOCKED_BY_MAIN_B4_MEMORY: ' + str(exc)[:500]) from exc
                    break
                torch.cuda.synchronize()
                b4_seconds = time.monotonic() - t0
                after = state(torch, runner)
                require(iters == 1 and runner.trace == [('train', 1), ('report', 1)], 'step 1 slots (no save / vis)')
                result['b4_step'] = step_evidence(torch, runner, before, after, json.loads(
                    ctx.path('metrics').read_text().splitlines()[-1]), batch) | {'seconds': round(b4_seconds, 3),
                                                                             'global_step': iters, 'forwards': dict(calls)}
                # ---------------- B: real B=2 tail-shaped iteration (global step 2)
                fw.phase = 'b2_tail_step'
                before = after
                calls.update(model_forwards=0, encoder_calls=0)
                tail = default_collate([runner.dataset[i] for i in TAIL_INDICES])
                t0 = time.monotonic()
                iters = runner.iteration(tail, 0, iters, progress_bar)
                torch.cuda.synchronize()
                after = state(torch, runner)
                require(iters == 2 and runner.trace[-2:] == [('train', 2), ('report', 2)], 'step 2 slots')
                result['b2_tail_step'] = step_evidence(torch, runner, before, after, json.loads(
                    ctx.path('metrics').read_text().splitlines()[-1]), tail) | {
                    'seconds': round(time.monotonic() - t0, 3), 'global_step': iters, 'indices': list(TAIL_INDICES),
                    'forwards': dict(calls), 'construction': 'default_collate of two real relation rows (the '
                    'DataLoader collate_fn); scientific DataLoader policy unchanged'}
                # ---------------- C: production visualization slot (direct, labelled)
                fw.phase = 'visualization'
                before = state(torch, runner)
                calls.update(model_forwards=0, encoder_calls=0)
                torch.cuda.reset_peak_memory_stats()
                info = runner.visualize_slot(iters)
                torch.cuda.synchronize()
                vis_peak = torch.cuda.max_memory_allocated()
                after = state(torch, runner)
                require(after['rng']['cpu_sha256'] != before['rng']['cpu_sha256'] and
                        after['rng']['cuda_sha256'] != before['rng']['cuda_sha256'], 'visualization advances CPU + CUDA RNG')
                for k in ('model_parameters', 'ema_parameters', 'ema_buffers', 'optimizer_sha256', 'scheduler',
                          'encoder_parameters', 'encoder_buffers'):
                    require(after[k] == before[k], 'visualization leaves unchanged: ' + k)
                require(dict(calls) == {k: EXPECTED_VIS[k] for k in ('model_forwards', 'encoder_calls')},
                        'pinned DDPM: 500 model forwards (250 steps x cond/null), 2 encoder calls ' + json.dumps(calls))
                require(all(m.training for m in runner.model.modules()), 'model stays in TRAIN mode (source)')
                result['visualization'] = info | {
                    'invocation': 'DIRECT production visualize_slot (labelled); scientific trigger iters % 1000 == 0',
                    'global_step_label': iters, 'model_forwards': calls['model_forwards'],
                    'encoder_calls': calls['encoder_calls'], 'peak_allocated_bytes': vis_peak,
                    'rng_before': before['rng'], 'rng_after': after['rng'], 'cpu_rng_changed': True,
                    'cuda_rng_changed': True, 'model_buffers_changed': after['model_buffers'] != before['model_buffers'],
                    'model_buffers_before': before['model_buffers']['sha256'],
                    'model_buffers_after': after['model_buffers']['sha256'],
                    'unchanged': ['model_parameters', 'ema_parameters', 'ema_buffers', 'optimizer', 'scheduler',
                                  'encoder'], 'rng_restored': False, 'model_mode': 'train'}
                # ---------------- E: checkpoint lifecycle (labelled qualification steps)
                fw.phase = 'checkpoint_lifecycle'
                life, torch_saves_before = [], counters.n['torch_save']
                torch.cuda.reset_peak_memory_stats()
                a, pruned = runner.checkpoint_transition(LABELLED_STEPS[0], 'periodic')
                require(pruned is None, 'first checkpoint has no predecessor')
                trip = roundtrip(torch, runner, a)
                b, pruned_a = runner.checkpoint_transition(LABELLED_STEPS[1], 'periodic')
                require(pruned_a and pruned_a['path'] == a['path'] and not (runner.store.run_dir / a['path']).exists(),
                        'A pruned only after B verified')
                t, pruned_b = runner.checkpoint_transition(mio.TERMINAL_STEP, 'terminal')
                require(pruned_b and pruned_b['path'] == b['path'], 'B pruned only after terminal verified')
                refusal = None
                try:
                    runner.store.prune(t['path'], t['path'])
                except mc.ProtectedCheckpoint as exc:
                    refusal = str(exc)
                require(refusal is not None and (runner.store.run_dir / t['path']).is_file(), 'terminal prune refused')
                ckpt_peak = torch.cuda.max_memory_allocated()
                index = runner.store.read_index()['checkpoints']
                require([e['path'] for e in index] == [a['path'], b['path'], t['path']] and
                        [e['bytes_present'] for e in index] == [False, False, True] and
                        [e['bytes_pruned'] for e in index] == [True, True, False] and
                        all(e['index_verified'] for e in index), 'index history retained through pruning')
                for e in index:
                    life.append({k: e.get(k) for k in (*mc.INDEX_FIELDS, 'bytes_present', 'bytes_pruned',
                                                        'index_verified', 'logical_roles', 'write_seconds', 'hash_seconds',
                                                        'pruned_utc', 'prune_reason', 'successor_checkpoint_global_step',
                                                        'successor_checkpoint_sha256')})
                cleanup = runner.store.remove_qualification_bytes(t['path'])
                ctx.log_event('e07c_main_qualification_cleanup', cleanup)
                result['checkpoint_lifecycle'] = {
                    'labelled_steps': [*LABELLED_STEPS, mio.TERMINAL_STEP], 'executed_global_step': iters,
                    'label': 'LABELLED QUALIFICATION STEPS of the real full main state; not executed global steps',
                    'index': life, 'roundtrip_first_file': trip, 'protected_refusal': refusal,
                    'prune_events': [pruned_a, pruned_b], 'torch_save_calls': counters.n['torch_save'] - torch_saves_before,
                    'peak_allocated_bytes': ckpt_peak, 'qualification_cleanup': [cleanup],
                    'bytes_created': sum(e['file_size_bytes'] for e in index),
                    'bytes_remaining_after_cleanup': 0, 'partial_files_remaining': 0}
                require(not list(fw.denied), 'firewall clean')
                result['trace'] = runner.trace
                result['runner_evidence'] = ev
                result['store_events'] = runner.store.events
                ctx.close(summary={
                    'final_or_selected_checkpoint_path': None, 'final_or_selected_checkpoint_sha256': None,
                    'training_duration_seconds': time.monotonic() - t_start, 'peak_vram_bytes': max(
                        runner.peak_step_gpu_memory_bytes, vis_peak, ckpt_peak), 'test_split_accessed': False,
                    'qualification_optimizer_steps': 2, 'scientific_optimizer_steps': 0,
                    'missing_field_reasons': {
                        'final_or_selected_checkpoint_path': 'QUALIFICATION: no scientific checkpoint exists',
                        'final_or_selected_checkpoint_sha256': 'QUALIFICATION: no scientific checkpoint exists',
                        'seed_level_evaluation_metrics': 'QUALIFICATION: no evaluation'}})
        finally:
            mio.untee(streams)
    metrics = [json.loads(line) for line in ctx.path('metrics').read_text().splitlines()]
    steps = [m for m in metrics if m['record_type'] == 'step']
    require([m['global_step'] for m in steps] == [1, 2] and all(m['experiment_seed'] is None and m['qualification_seed']
            == SEED and m['train_metrics'] is None and m['val_losses'] is None and m['val_metrics'] is None and
            set(m['missing_field_reasons']) >= {'train_metrics', 'val_losses', 'val_metrics'} for m in steps),
            'per-step metrics with explicit nulls')
    phases = {}
    for phase, sid in fw.face_opens:
        phases.setdefault(phase, []).append(sid)
    require({k: len(v) for k, v in phases.items()} == {'b4_step': 12, 'b2_tail_step': 6, 'visualization': 12},
            'physical face opens per phase ' + json.dumps({k: len(v) for k, v in phases.items()}))
    files = sorted(str(p.relative_to(ctx.run_dir)) for p in ctx.run_dir.rglob('*') if p.is_file())
    result.update(
        status='PASS', milestone=MILESTONE, qualification_seed=SEED, experiment_seed=None,
        experiment_seed_reason='QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN', qualification_id=qid,
        run_root=str(ctx.run_dir), stages=stages, source=source_evidence, identities=ids,
        counters=counters.n, pyarrow_reads=guard.calls, manifest_opens=fw.manifest_opens,
        frozen_encoder={'sha256_events': fw.sha_events, 'read_opens': len(fw.frozen_opens), 'torch_load_seam': 1},
        reads={'physical_face_opens': {k: len(v) for k, v in phases.items()},
               'unique_face_ids': len({s for _, s in fw.face_opens}), 'train_ids': len({s for _, s in fw.face_opens}),
               'val_ids': 0, 'test_ids': 0, 'prefetch_reads': 0, 'num_workers': 0,
               'attribution': 'every open is a TRAIN relation id (firewall allowlist); 3 opens per item (content, GT, guide)'},
        run_logging={'files': files, 'metrics_records': len(metrics), 'step_records': len(steps),
                     'step_record_example': steps[0], 'run_summary': json.loads(ctx.path('run_summary').read_text())},
        firewall={'denied': fw.denied, 'events': fw.events, 'subprocesses': fw.subprocesses,
                  'weight_events': len(fw.weight_events)},
        qualification_counters={'qualification_processes': 1, 'qualification_optimizer_steps': counters.n['optimizer_step'],
                                'qualification_backward_calls': counters.n['backward'],
                                'qualification_scheduler_steps': counters.n['scheduler_step'],
                                'qualification_checkpoint_deserializations': counters.n['torch_load_seam'],
                                'qualification_checkpoint_roundtrip_loads': counters.n['torch_load_roundtrip'],
                                'qualification_visualizations': 1, 'qualification_checkpoints_written': 3},
        scientific_counters={'scientific_main_runs': 0, 'scientific_optimizer_steps': 0, 'experiment_seed_runs': 0,
                             'scientific_checkpoint_writes': 0},
        wall_seconds=round(time.monotonic() - t_start, 3), candidate_sha256=mio.m6d6j_file_hashes(),
        owner_decisions={'D1_visualization': 'EXECUTE_EXACT_SOURCE',
                         'D2_terminal_position': 'AFTER_FINAL_VISUALIZATION_BEFORE_RUN_COMPLETION'},
        qualified_statuses=QUALIFIED, not_qualified=NOT_QUALIFIED, method_status='IMPLEMENTED_NOT_EXECUTED',
        fidelity_class='CONTROLLED_ADAPTATION', deviation='DEV-021', TRAIN_access=True, VAL_access=False,
        TEST_access=False, M8_outputs=0, MAIN_CHECKPOINT_RESUME='UNQUALIFIED',
        FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION=True, O4_resume_before_science='PENDING')
    require(counters.n == {'backward': 2, 'zero_grad': 2, 'optimizer_step': 2, 'scheduler_step': 2, 'torch_save': 3,
                           'torch_load_seam': 1, 'torch_load_roundtrip': 1, 'autocast_entries': 0},
            'exact counters ' + json.dumps(counters.n))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT) and build.parts[-3:] == BUILD_PARTS and build.is_relative_to(rq.RUNTIME),
            'dedicated M6D6j build path')
    require(Path(args.output).name == args.output and args.output.endswith('.json') and not (build / args.output).exists(),
            'fresh evidence filename')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    try:
        result = qualify(build)
    except mr.MainMemoryBlocked as exc:
        (build / args.output).write_text(json.dumps({'status': 'BLOCKED_BY_MAIN_B4_MEMORY', 'error': str(exc)}) + '\n')
        print(json.dumps({'status': 'BLOCKED_BY_MAIN_B4_MEMORY'}))
        sys.exit(3)
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))


if __name__ == '__main__':
    main()
