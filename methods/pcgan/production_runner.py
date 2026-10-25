"""E05 PCGAN (controlled architecture resolution) production runner: real-TRAIN pairs, 4000 D -> G iterations.

CONTROLLED_ADAPTATION (A2-07 architecture resolution, A5 blur operator, M6D4c training-runner resolution). Not a
native/faithful PCGAN and not an official reproduction: no official PCGAN code exists; the executable architecture
basis is taesungp/swapping-autoencoder-pytorch@6baa180f (disclosed).

Reused WITHOUT modification:
  * methods/pcgan/training_runner.py::TrainingRunner -- the qualified explicit-pair FP32 D -> G cycle (two Adam
    optimizers, lr 1e-6, betas (.9,.999); one D application, then one G application, then benchmark_iteration += 1;
    a failed partial iteration marks the runner failed and can never be retried);
  * methods/pcgan/training_qualification.py::existing_runtime -- the M6D4a/M6D4e qualified generated source, the
    two existing real CUDA extensions (no JIT rebuild, no fallback) and the pinned architecture/crop builders;
  * methods/pcgan/runtime_qualification.py::environment -- the environment record compared with the E05 lock;
  * the E06c canonical face reader and M2 faces-root containment (methods/dsdg/runner_io.py), unchanged.
Added here: the TRAIN pair dataset/loader, the per-iteration trajectory, the BASELINE_FINAL_STATE_V1 checkpoint writer,
the run_logging_v1 lifecycle (LearnedRunContext) and the fresh-only scientific entry used by tools/run_e05.py.
Torch is imported inside functions; importing this module is static.
"""
import fcntl
import getpass
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import uuid

import numpy as np

from methods.common.config import ROOT, load_method_config, sha256_file
from methods.common.learned import PreparationError, apply_framework_seed, seed_torch_worker, torch_loader_options
from methods.common.learned_runlog import LearnedRunContext
from methods.common.runlog import RunContext, RunDirectoryError, compute_run_id, git_commit, git_dirty, \
    load_logging_contract, atomic_write_json
from methods.pcgan import production_io as pio

require = pio.require
G_PARAMETERS, D_PARAMETERS = 4629193, 53381666


class TrainingStop(RuntimeError):
    """A real-data gate failed (contract, finiteness, accounting, OOM): STOP_AND_REPORT, never retried."""


def gate(value, message):
    if not value:
        raise TrainingStop('E05 STOP: ' + message)


# ================================================================= run context (run_logging_v1)
class E05RunContext(LearnedRunContext):
    """LearnedRunContext with the E05 production identities made visible.

    SCIENTIFIC is the unchanged LearnedRunContext (experiment seed, runs/m6/E05/seed_<seed>, checkpoint_metadata
    validation, fresh only). QUALIFICATION binds the same writers to <rt>/qualification/e05_production/E05/...;
    RunContext refuses non-experiment seeds by design, so the identity attributes are set here explicitly
    (experiment_seed stays null) with the run_logging_v1 run_id formula applied to the qualification seed.
    """
    def __init__(self, *, mode, seed, runtime_root, environment, missing_environment_reasons, identities, case=None,
                 command_line=None):
        pio.validate_mode_seed(mode, seed)
        config = load_method_config(pio.METHOD_ID)
        self.mode, self.case, self.identities = mode, case, dict(identities)
        require(self.identities['config_sha256'] == config['_runtime']['config_sha256'], 'config identity')
        if mode == pio.SCIENTIFIC:
            require(case is None, 'scientific runs have no qualification case')
            super().__init__(method_id=pio.METHOD_ID, seed=seed, config=config, runtime_root=runtime_root,
                             command_line=command_line, resume=False, environment=environment,
                             missing_environment_reasons=missing_environment_reasons)
            self.qualification_seed = None
            return
        self.contract = load_logging_contract()
        self.method_id, self.seed, self.qualification_seed = pio.METHOD_ID, None, seed
        self.config, self.runtime_root, self.resume = config, Path(runtime_root), False
        self.source_commits = {}          # identical to SCIENTIFIC: the E05 config has no top-level source pin
        self.command_line = command_line if command_line is not None else ' '.join(sys.argv)
        self.config_sha256 = config['_runtime']['config_sha256']
        self.config_path = config['_runtime']['config_path']
        self.git_commit = git_commit()
        self.run_id = compute_run_id(pio.METHOD_ID, seed, self.config_sha256, self.git_commit)
        self.run_dir = pio.run_root(runtime_root, mode, seed, case=case, run_id=self.run_id)
        self.start_utc = self._metrics_fh = self._generation_fh = self._closed_summary = self._lock_fh = None
        self._records, self.run_uuid, self._started = 0, str(uuid.uuid4()), time.monotonic()
        required = {'host', 'user', 'platform', 'python_version', 'numpy_version', 'gpu_model', 'gpu_count',
                    'cuda_version', 'cudnn_version', 'framework', 'framework_version', 'pytorch_version',
                    'dependency_fingerprint', 'environment_lock_path', 'tensorflow_version'}
        if not required <= set(environment):
            raise PreparationError('explicit learned environment fields are missing')
        if any(v is None and not missing_environment_reasons.get(k) for k, v in environment.items()):
            raise PreparationError('null environment fields require reasons')
        self._validate_metadata(environment)
        self._environment, self._environment_reasons = dict(environment), dict(missing_environment_reasons)

    @property
    def seed_value(self):
        return self.seed if self.mode == pio.SCIENTIFIC else self.qualification_seed

    def open(self):
        if self.mode == pio.SCIENTIFIC:
            return super().open()
        if self._metrics_fh is not None or self._closed_summary is not None:
            raise RunDirectoryError('RunContext objects may be opened only once')
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fh = (self.run_dir.parent / f'.{self.run_dir.name}.lock').open('a')
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._open_locked()
        except BaseException:
            for handle in (self._metrics_fh, self._generation_fh, self._lock_fh):
                if handle is not None:
                    handle.close()
            self._metrics_fh = self._generation_fh = self._lock_fh = None
            raise

    def _open_locked(self):
        if self.run_dir.exists():          # fresh only, in both modes (resume is not qualified for E05)
            raise RunDirectoryError(f'STOP_AND_REPORT: {self.run_dir} exists; E05 runs are fresh only '
                                    '(resume is not qualified)')
        result = super()._open_locked()
        index = json.loads(self.path('checkpoint_index').read_text())
        index.update(note='Model checkpoints actually written: pinned-predecessor cadence (iteration 1 periodic) plus '
                          'the BASELINE_FINAL_STATE_V1 terminal state at benchmark iteration 4000. No resume sidecar.',
                     runner_mode=self.mode, checkpoint_rule=self.config['checkpoint']['rule'],
                     selected_final=('checkpoints/' + pio.checkpoint_basename(pio.TOTAL_ITERATIONS)
                                     if self.mode == pio.SCIENTIFIC else None),
                     scientific_checkpoints=self.mode == pio.SCIENTIFIC, resume_sidecars=None)
        if self.mode == pio.QUALIFICATION:
            index.update(labels=self.labels(), qualification_seed=self.qualification_seed, qualification_case=self.case,
                         selected_final_reason='QUALIFICATION_ONLY: no scientific selection exists')
        atomic_write_json(self.path('checkpoint_index'), index)
        return result

    def labels(self):
        if self.mode == pio.SCIENTIFIC:
            return []
        extra = [pio.TERMINAL_PROBE_LABEL] if self.case == 'terminal_probe' else []
        return list(pio.QUALIFICATION_LABELS) + extra

    def _resolved_config(self):
        resolved = super()._resolved_config()
        ids = self.identities
        resolved['_resolved'].update({
            'runner_mode': self.mode, 'qualification_seed': self.qualification_seed,
            'qualification_case': self.case, 'scientific_run': self.mode == pio.SCIENTIFIC,
            'qualification_only': self.mode == pio.QUALIFICATION, 'labels': self.labels(),
            'note': 'Verbatim copy of the frozen config above; every execution choice is listed here and was NOT '
                    'part of the frozen YAML. No scientific value was overridden.',
            'identities': ids, 'fidelity_class': 'CONTROLLED_ADAPTATION',
            'reporting_label': self.config['reporting_label'],
            'source_pin': {'repository': self.config['source']['executable_architecture_basis']['repository'],
                           'commit': ids['source_commit'], 'tree': ids['source_tree'],
                           'role': 'executable architecture basis (no official PCGAN code release)'},
            'amendments': {'A2': ids['a2_sha256'], 'A5_amendment': ids['a5_amendment_sha256'],
                           'A5_overlay': {'path': pio.A5_OVERLAY_PATH, 'sha256': ids['a5_overlay_sha256']},
                           'M6D4c_overlay': {'path': pio.M6D4C_OVERLAY_PATH, 'sha256': ids['m6d4c_overlay_sha256']}},
            'environment_lock': {'path': pio.LOCK_PATH, 'sha256': ids['environment_lock_sha256']},
            'pair_manifest': {'path': pio.PAIR_MANIFEST, 'sha256': ids['pair_manifest_sha256'],
                              'columns_read': list(pio.ALLOWLIST)},
            'training': {'total_iterations': pio.TOTAL_ITERATIONS, 'batch_size': pio.BATCH_SIZE,
                         'iterations_executed_limit': (pio.TOTAL_ITERATIONS if self.mode == pio.SCIENTIFIC
                                                       else pio.QUALIFICATION_ITERATIONS),
                         'iteration_unit': 'one D Adam application then one G Adam application',
                         'optimizers': 'D: ImageD+PatchD, G: Encoder+Generator; Adam lr 1e-6 betas (0.9, 0.999)'},
            'execution_choices': pio.resolved_overlays(),
            'dataloader': dict(pio.LOADER, generator_seed=self.seed_value,
                               worker_init_fn='methods.common.learned.seed_torch_worker (inactive: 0 workers)'),
            'checkpoint_rule': {'rule': self.config['checkpoint']['rule'], 'iterations': pio.checkpoint_iterations(),
                                'selected': 'checkpoints/' + pio.checkpoint_basename(pio.TOTAL_ITERATIONS)},
            'logging_contract': {'version': 'run_logging_v1', 'sha256': ids['logging_contract_sha256'],
                                 'record_per': 'BENCHMARK_ITERATION'},
            'run_id_rule': 'run_logging_v1: sha256(method_id|seed|config_sha256|git_commit)[:16]'})
        return resolved

    def _manifest(self, **kwargs):
        m = super()._manifest(**kwargs)
        m.update(runner_mode=self.mode, scientific_run=self.mode == pio.SCIENTIFIC,
                 qualification_only=self.mode == pio.QUALIFICATION, qualification_seed=self.qualification_seed,
                 qualification_case=self.case, fidelity_class='CONTROLLED_ADAPTATION',
                 reporting_label=self.config['reporting_label'], identities=self.identities,
                 source_pin={'commit': self.identities['source_commit'], 'tree': self.identities['source_tree']},
                 environment_lock_sha256=self.identities['environment_lock_sha256'],
                 resume='NOT_QUALIFIED_FRESH_ONLY', val_split_accessed=False)
        if self.mode == pio.QUALIFICATION:
            m['labels'] = self.labels()
            m['missing_field_reasons']['experiment_seed'] = (f'QUALIFICATION_ONLY: seed {self.qualification_seed} '
                                                             'is not an experiment seed')
        return m

    def record_checkpoint(self, **kwargs):
        if self.mode == pio.SCIENTIFIC:
            return super().record_checkpoint(**kwargs)
        step = kwargs['global_step']
        kind, selected, reason = pio.checkpoint_kind(step)
        require((kwargs['checkpoint_type'], kwargs['selected_for_final'], kwargs['selection_reason']) ==
                (kind, selected, reason), 'qualification checkpoint semantics equal the production cadence')
        require(selected is (self.case == 'terminal_probe'), 'only the terminal-semantics probe records iteration 4000')
        return RunContext.record_checkpoint(self, **kwargs)


# ================================================================= runtime setup
def configure_precision(torch):
    """FP32 only, exactly the M6D4a/d/e qualified flags: no TF32, no autocast, highest matmul, deterministic cuDNN."""
    require(os.environ.get('NVIDIA_TF32_OVERRIDE') == '0', 'NVIDIA_TF32_OVERRIDE=0 required')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_WORKSPACE_CONFIG=:4096:8 required')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    require(not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'), 'autocast disabled')
    return {'default_dtype': str(torch.get_default_dtype()), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'autocast_cuda': False, 'autocast_cpu': False,
            'matmul_precision': torch.get_float32_matmul_precision(), 'amp': False}


def seed_process(torch, mode, seed, config):
    """Scientific: methods.common.learned.apply_framework_seed; qualification: the same calls, qualification seed."""
    pio.validate_mode_seed(mode, seed)
    require(os.environ.get('PYTHONHASHSEED') == str(seed), f'launch with PYTHONHASHSEED={seed}')
    if mode == pio.SCIENTIFIC:
        plan = apply_framework_seed(config, seed, 'torch', cuda=True)
        return {'seed': seed, 'role': 'EXPERIMENT_SEED', 'plan': 'methods.common.learned.apply_framework_seed',
                'python_numpy_torch_cuda': True, 'cudnn_benchmark': plan['cudnn_benchmark'],
                'cudnn_deterministic': plan['cudnn_deterministic']}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    return {'seed': seed, 'role': 'QUALIFICATION_SEED_NOT_EXPERIMENT_SEED', 'python_numpy_torch_cuda': True,
            'plan': 'random/numpy/torch/torch.cuda seeded identically to apply_framework_seed',
            'cudnn_benchmark': False, 'cudnn_deterministic': True}


def verify_environment(seed):
    """The live environment record equals the E05 lock identity (production_io.environment_differences)."""
    from methods.pcgan.runtime_qualification import environment
    lock = json.loads((ROOT / pio.LOCK_PATH).read_text())
    live = environment()
    diff, driver = pio.environment_differences(lock['identity'], live, seed)
    require(not diff, 'gpat-m6-e05 environment differs from the lock identity: ' + ', '.join(diff))
    return live, driver


def environment_record(torch):
    env = {'host': socket.gethostname(), 'user': getpass.getuser(), 'platform': platform.platform(),
           'python_version': platform.python_version(), 'numpy_version': np.__version__,
           'gpu_model': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(),
           'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
           'framework': 'torch', 'framework_version': torch.__version__, 'pytorch_version': torch.__version__,
           'tensorflow_version': None, 'dependency_fingerprint': sha256_file(ROOT / pio.LOCK_PATH),
           'environment_lock_path': pio.LOCK_PATH, 'environment_name': pio.ENVIRONMENT_NAME,
           'python_executable': sys.executable,
           'gpu_note': 'single RTX 3090; batch 1 explicit pair; FP32 (M6D4e peak ~3.7 GB)'}
    return env, {'tensorflow_version': 'E05 is a PyTorch method; TensorFlow is not part of gpat-m6-e05'}


def build_models(build):
    """The M6D4d/e qualified runtime: existing generated source + real CUDA binaries, pinned builders and crop."""
    from methods.pcgan import training_qualification as m6d4d
    runtime = {}
    models, crop, _ = m6d4d.existing_runtime(Path(build), runtime)
    return models, crop, runtime


# ================================================================= TRAIN pairs
class RecordingDecode:
    """Decode hook for the unchanged canonical reader: records sha256 of the PNG bytes actually read."""

    def __init__(self):
        self.reads = []

    def __call__(self, raw):
        from methods.dsdg import runner_io as rio
        self.reads.append(pio.sha(raw))
        return rio._cv2_decode(raw)


def face_reader(faces_root, sample_datasets):
    from methods.dsdg import runner_io as rio
    decode = RecordingDecode()
    return rio.CanonicalFaceReader(faces_root, sample_datasets, datasets=pio.DATASETS, decode=decode), decode


class PairDataset:
    """Row i of the frozen TRAIN pair manifest -> x_src = face(source_spoof_id), x_tgt = face(target_live_id)."""

    def __init__(self, rows, reader, decode):
        self.rows, self.reader, self.decode = tuple(rows), reader, decode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        before = len(self.decode.reads)
        x_src = pio.canonical_input(self.reader(r['source_spoof_id']))
        x_tgt = pio.canonical_input(self.reader(r['target_live_id']))
        require(len(self.decode.reads) == before + 2, 'two canonical face reads per pair')
        return {'x_src': x_src, 'x_tgt': x_tgt, 'index': np.int64(index), 'pair_id': r['pair_id'],
                'source_spoof_id': r['source_spoof_id'], 'target_live_id': r['target_live_id'],
                'dataset': r['dataset'], 'source_png_sha256': self.decode.reads[before],
                'target_png_sha256': self.decode.reads[before + 1]}


def build_loader(torch, dataset, mode, seed, config):
    """DataLoader(batch 1, shuffle, 0 workers, drop_last=False) with the seeded generator (torch_loader_options)."""
    if mode == pio.SCIENTIFIC:
        opts = torch_loader_options(config, seed)          # validates the experiment seed
    else:
        pio.validate_mode_seed(mode, seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        opts = {'generator': generator, 'worker_init_fn': seed_torch_worker}
    loader = torch.utils.data.DataLoader(dataset, **pio.LOADER, **opts)
    s, b = loader.sampler, loader.batch_sampler
    require((loader.batch_size, loader.drop_last, loader.num_workers, loader.pin_memory, loader.persistent_workers)
            == (1, False, 0, False, False), 'DataLoader batch/drop_last/workers/pin_memory')
    require(type(s) is torch.utils.data.RandomSampler and s.replacement is False and s.generator is opts['generator']
            and s.num_samples == len(dataset), 'RandomSampler without replacement on the seeded generator')
    require(type(b) is torch.utils.data.BatchSampler and b.batch_size == 1 and b.drop_last is False, 'BatchSampler')
    evidence = {k: v for k, v in pio.LOADER.items()}
    evidence.update(persistent_workers=False, sampler='RandomSampler(replacement=False, generator=G)',
                    batch_sampler='BatchSampler(1, drop_last=False)', generator_seed=seed, rows=len(dataset),
                    len_batches=len(loader))
    return loader, opts['generator'], evidence


# ================================================================= trainer around the qualified TrainingRunner
class E05Trainer:
    """Drives the unchanged TrainingRunner; adds per-iteration gates, independent application counters and records."""

    def __init__(self, torch, runner, mode, seed, device='cuda'):
        self.torch, self.runner, self.mode, self.seed, self.device = torch, runner, mode, seed, device
        self.hook_applications = {'D': 0, 'G': 0}
        self.peak_gpu_memory_bytes = 0
        self.membership = {}
        for group, entries in runner.groups.items():
            opt = runner.optimizers[group]
            pg = opt.param_groups
            require(len(pg) == 1 and [id(p) for p in pg[0]['params']] == [id(p) for _, p in entries],
                    'live optimizer membership ' + group)
            require((pg[0]['lr'], tuple(pg[0]['betas']), pg[0]['weight_decay']) == (pio.LEARNING_RATE, pio.BETAS, 0),
                    'Adam lr/betas/weight_decay ' + group)
            self.membership[group] = {'tensors': len(entries), 'parameters': sum(p.numel() for _, p in entries),
                                      'lr': pg[0]['lr'], 'betas': list(pg[0]['betas']),
                                      'weight_decay': pg[0]['weight_decay']}

            def counted(_opt, _args, _kwargs, key=group):
                self.hook_applications[key] += 1
            opt.register_step_post_hook(counted)
        require({g: (m['tensors'], m['parameters']) for g, m in self.membership.items()} ==
                {'G': (pio.GROUP_TENSORS['G'], G_PARAMETERS), 'D': (pio.GROUP_TENSORS['D'], D_PARAMETERS)},
                'optimizer group tensor/parameter counts')

    def applications(self):
        return {'D': self.runner.applications['D'], 'G': self.runner.applications['G'],
                'hook_D': self.hook_applications['D'], 'hook_G': self.hook_applications['G'],
                'total': self.runner.applications['D'] + self.runner.applications['G']}

    def step(self, batch, iteration, extra_observer=None):
        """Exactly one complete D -> G cycle for benchmark iteration `iteration` (1-based)."""
        torch, runner = self.torch, self.runner
        gate(runner.benchmark_iteration == iteration - 1 and not runner.failed, 'iteration boundary')
        gate(self.applications() == {'D': iteration - 1, 'G': iteration - 1, 'hook_D': iteration - 1,
                                     'hook_G': iteration - 1, 'total': 2 * (iteration - 1)}, 'prior application count')
        gate(all(len(batch[k]) == 1 for k in ('pair_id', 'source_spoof_id', 'target_live_id')), 'batch size 1')
        src, tgt = batch['x_src'].to(self.device), batch['x_tgt'].to(self.device)
        gate(tuple(src.shape) == tuple(tgt.shape) == (1, 3, 256, 256) and src.dtype == tgt.dtype == torch.float32,
             'explicit FP32 [1,3,256,256] pair')
        lo, hi = min(float(src.min()), float(tgt.min())), max(float(src.max()), float(tgt.max()))
        gate(pio.INPUT_RANGE[0] <= lo and hi <= pio.INPUT_RANGE[1], 'inputs in [-1,1]')
        seen = {}

        def observer(stage, live, state):
            if stage in ('D_forward', 'G_forward'):
                seen[stage[0] + '_losses'] = {k: float(v.detach()) for k, v in state['losses'].items()}
            if stage in ('D_backward', 'G_backward'):
                group = stage[0]
                other = 'G' if group == 'D' else 'D'
                grads = [p.grad for _, p in live.groups[group]]
                present = [g for g in grads if g is not None]
                seen['grad_' + group] = {
                    'tensors': len(grads), 'non_none': len(present),
                    'finite': int(torch.stack([torch.isfinite(g).all() for g in present]).sum()) if present else 0,
                    'nonzero': int(torch.stack([torch.count_nonzero(g) > 0 for g in present]).sum()) if present else 0,
                    'inactive_group_non_none': sum(p.grad is not None for _, p in live.groups[other])}
            if extra_observer is not None:
                extra_observer(stage, live, state)
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        try:
            runner.cycle(src, tgt, observer)
        except torch.OutOfMemoryError as error:
            raise TrainingStop(f'CUDA OOM at benchmark iteration {iteration}: {error}') from error
        step_seconds = time.monotonic() - t0
        gate(runner.benchmark_iteration == iteration, 'benchmark_iteration advanced exactly once')
        gate(self.applications() == {'D': iteration, 'G': iteration, 'hook_D': iteration, 'hook_G': iteration,
                                     'total': 2 * iteration}, 'exactly one D and one G Adam application')
        for group in ('D', 'G'):
            g = seen.get('grad_' + group)
            gate(g is not None and g['tensors'] == g['non_none'] == g['finite'] == pio.GROUP_TENSORS[group] and
                 g['inactive_group_non_none'] == 0 and g['nonzero'] > 0, f'{group} gradients: {g}')
        losses = {**seen.get('D_losses', {}), **seen.get('G_losses', {})}
        gate(set(losses) == set(pio.TRAIN_LOSS_KEYS) and pio.finite(list(losses.values())), 'finite D/G losses')
        gate(all(losses[k] > 0.0 for k in pio.G_TERMS), 'five G losses active (positive)')
        mem = torch.cuda.max_memory_allocated() if self.device == 'cuda' else None
        self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes, mem or 0)
        return {'losses': losses, 'step_seconds': step_seconds, 'gpu_memory_bytes': mem,
                'gradients': {g: seen['grad_' + g] for g in ('D', 'G')}, 'input_range': [lo, hi],
                'input_shape': list(src.shape), 'input_dtype': str(src.dtype),
                'learning_rates': {g: runner.optimizers[g].param_groups[0]['lr'] for g in ('D', 'G')}}


def run_iterations(trainer, loader, ctx, *, iterations, t_start, extra_observer=None, on_record=None):
    """The production loop: `iterations` complete cycles, one trajectory record each, checkpoints on the cadence."""
    if ctx.mode == pio.SCIENTIFIC:
        require(iterations == pio.TOTAL_ITERATIONS, 'scientific runs execute exactly 4000 benchmark iterations')
    else:
        require(iterations == pio.QUALIFICATION_ITERATIONS, 'qualification iteration limit')
    batches = iter(loader)
    written = []
    for t in range(1, iterations + 1):
        batch = next(batches)
        r = trainer.step(batch, t, extra_observer=extra_observer)
        pair = {k: batch[k][0] for k in ('pair_id', 'source_spoof_id', 'target_live_id', 'dataset')}
        record = pio.step_record(iteration=t, learning_rates=r['learning_rates'], losses=r['losses'],
                                 wall_clock_seconds=time.monotonic() - t_start, step_seconds=r['step_seconds'],
                                 gpu_memory_bytes=r['gpu_memory_bytes'], pair=pair,
                                 sampler_index=int(batch['index'][0]), applications=trainer.applications(),
                                 gradients=r['gradients'], mode=ctx.mode, seed=ctx.seed_value,
                                 input_range=r['input_range'])
        record.update(source_png_sha256=batch['source_png_sha256'][0], target_png_sha256=batch['target_png_sha256'][0],
                      input_shape=r['input_shape'], input_dtype=r['input_dtype'])
        ctx.log_epoch(record)
        if on_record is not None:
            on_record(record)
        if pio.needs_saving(t):
            written.append(write_checkpoint(trainer.torch, ctx, trainer, t))
    return written


# ================================================================= checkpoint writer (BASELINE_FINAL_STATE_V1)
def write_checkpoint(torch, ctx, trainer, iteration, *, terminal_semantics_probe=False):
    """Atomic state write + weights_only round-trip check + checkpoint_index record. Never steps an optimizer.

    terminal_semantics_probe exists only for the QUALIFICATION 'terminal_probe' case: it writes the iteration-4000
    path/type/selection semantics from a state that is NOT iteration 4000 and is labelled as such.
    """
    kind, selected, reason = pio.checkpoint_kind(iteration)
    runner = trainer.runner
    actual = runner.benchmark_iteration
    apps_before = trainer.applications()
    if terminal_semantics_probe:
        require(ctx.mode == pio.QUALIFICATION and ctx.case == 'terminal_probe' and
                iteration == pio.TOTAL_ITERATIONS and actual < pio.TOTAL_ITERATIONS,
                'terminal-semantics probe is qualification-only and never claims a real iteration 4000')
    else:
        require(ctx.mode == pio.SCIENTIFIC or ctx.case == 'train', 'checkpoint case')
        require(iteration == actual and apps_before == {'D': actual, 'G': actual, 'hook_D': actual, 'hook_G': actual,
                                                        'total': 2 * actual},
                'a checkpoint records exactly the completed benchmark iteration')
        require(not runner.failed, 'no checkpoint after a failed iteration')
    rel = 'checkpoints/' + pio.checkpoint_basename(iteration)
    path = ctx.run_dir / rel
    require(not path.exists() and not path.with_name(path.name + '.partial').exists(), 'checkpoint path is fresh')
    provenance = json.loads(json.dumps({
        'method_id': pio.METHOD_ID, 'runner_mode': ctx.mode, 'run_id': ctx.run_id, 'git_commit': ctx.git_commit,
        'experiment_seed': ctx.seed, 'qualification_seed': ctx.qualification_seed, 'config_sha256': pio.CONFIG_SHA,
        'source_commit': pio.SOURCE_COMMIT, 'benchmark_iteration': iteration, 'actual_benchmark_iteration': actual,
        'optimizer_applications': apps_before, 'checkpoint_type': kind, 'selected_for_final': selected,
        'selection_reason': reason, 'checkpoint_rule': pio.CHECKPOINT_RULE, 'labels': ctx.labels(),
        'terminal_semantics_probe': bool(terminal_semantics_probe),
        'contents': 'state_dict of Encoder, Generator, ImageD, PatchD; no optimizer state (fresh-only runs)'}))
    states = {name: model.state_dict() for name, model in runner.models.items()}
    payload = {'format': 'E05_BASELINE_STATE_V1', 'benchmark_iteration': iteration, 'models': states,
               'provenance': provenance}
    tmp = path.with_name(path.name + '.partial')
    torch.save(payload, tmp)
    os.replace(tmp, path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    require(loaded['format'] == payload['format'] and loaded['provenance'] == provenance and
            loaded['benchmark_iteration'] == iteration and set(loaded['models']) == set(states), 'checkpoint structure')
    tensors = 0
    for name, state in states.items():
        require(list(loaded['models'][name]) == list(state), 'state keys ' + name)
        for key, value in state.items():
            require(torch.equal(loaded['models'][name][key], value.detach().cpu()), f'round-trip {name}.{key}')
            tensors += 1
    require(trainer.applications() == apps_before, 'no optimizer application while writing a checkpoint')
    meta = dict(path=rel, epoch=pio.PASS_INDEX, global_step=iteration, file_size_bytes=path.stat().st_size,
                sha256=sha256_file(path), checkpoint_type=kind, selected_for_final=selected, selection_reason=reason)
    ctx.record_checkpoint(**meta)
    ctx.log_event('e05_checkpoint_event', {**meta, 'benchmark_iteration': iteration,
                                           'actual_benchmark_iteration': actual, 'tensors_round_trip_equal': tensors,
                                           'optimizer_applications': apps_before,
                                           'terminal_semantics_probe': bool(terminal_semantics_probe),
                                           'labels': ctx.labels(), 'resume_sidecar': None})
    return dict(meta, tensors_round_trip_equal=tensors, actual_benchmark_iteration=actual,
                terminal_semantics_probe=bool(terminal_semantics_probe))


# ================================================================= scientific entry (tools/run_e05.py)
def run_scientific(*, seed, runtime_root, faces_root, command_line=None):
    """Full 4000-iteration SCIENTIFIC run, fresh only. Never called by the qualification harness."""
    import torch
    from methods.dsdg import runner_io as rio
    from methods.pcgan import PCGANAdapter
    from methods.pcgan.training_runner import TrainingRunner
    mode = pio.SCIENTIFIC
    pio.validate_mode_seed(mode, seed)
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    errors = pio.launch_environment_errors(os.environ, seed)
    require(not errors, 'launch environment: ' + ', '.join(errors))
    config, contracts = pio.verify_contract()
    ids = pio.identities(config, contracts)
    source = pio.verify_source(PCGANAdapter(config))
    rt = Path(runtime_root)
    run_dir = pio.run_root(rt, mode, seed)
    require(not run_dir.exists(), f'{run_dir} exists; E05 scientific runs are fresh only (resume is not qualified)')
    precision = configure_precision(torch)
    seeding = seed_process(torch, mode, seed, config)
    env_identity, driver = verify_environment(seed)
    models, crop, runtime = build_models(pio.build_root(rt))
    runner = TrainingRunner(models, crop, root=ROOT)
    trainer = E05Trainer(torch, runner, mode, seed)
    rows, sample_datasets, manifest = pio.read_pair_manifest()
    reader, decode = face_reader(faces_root, sample_datasets)
    loader, _, loader_evidence = build_loader(torch, PairDataset(rows, reader, decode), mode, seed, config)
    env, reasons = environment_record(torch)
    ctx = E05RunContext(mode=mode, seed=seed, runtime_root=rt, environment=env, missing_environment_reasons=reasons,
                        identities=ids, command_line=command_line)
    with ctx:
        streams = rio.tee_run_logs(ctx.run_dir)
        try:
            ctx.log_event('e05_run_start', {
                'seeding': seeding, 'precision': precision, 'identities': ids, 'source_commit': source['commit'],
                'environment_identity_equals_lock': True, 'environment_python_sha256': env_identity['python_sha256'],
                'gpu_driver': driver,
                'runtime_closure': {k: runtime[k] for k in ('build_closure', 'models')},
                'optimizers': trainer.membership, 'loader': loader_evidence, 'pair_manifest': manifest,
                'execution_choices': pio.resolved_overlays(), 'fidelity_class': 'CONTROLLED_ADAPTATION'})
            t0 = time.monotonic()
            written = run_iterations(trainer, loader, ctx, iterations=pio.TOTAL_ITERATIONS, t_start=t0)
            final = [w for w in written if w['selected_for_final']]
            require(len(final) == 1 and final[0]['global_step'] == pio.TOTAL_ITERATIONS, 'one selected terminal state')
            ctx.close(completion_status='completed', summary={
                'final_or_selected_checkpoint_path': str(ctx.run_dir / final[0]['path']),
                'final_or_selected_checkpoint_sha256': final[0]['sha256'],
                'checkpoint_selection_rule': config['checkpoint']['rule'],
                'training_duration_seconds': time.monotonic() - t0,
                'peak_vram_bytes': trainer.peak_gpu_memory_bytes,
                'benchmark_iterations': runner.benchmark_iteration,
                'optimizer_applications': trainer.applications(),
                'missing_field_reasons': {
                    'seed_level_evaluation_metrics': 'Downstream evaluation is a later milestone'}})
        finally:
            rio.untee(streams)
    return ctx.run_dir
