"""M6D5e E06c DSDG-BIN-IDFREE production training runner (CONTROLLED_ADAPTATION, DEV-020).

Binds, unchanged: the frozen E06c config + A1 adaptation, the pinned FaceX-Zoo DSDG
networks/misc.util, LightCNN-29 v2 (frozen, eval, SHA-verified), the M6D5c V1 and M6D5d
V2 owner overlays (GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2: nominal global batch 240,
max microbatch 20, drop_last=False, 36 x 240 + 198 = 37 optimizer steps per epoch) and the
repository run_logging_v1 RunContext (LearnedRunContext writers reused as-is).
The pinned training-sample visualization block (train_generator.py:197-211) runs at its
source cadence, after the epoch loop and before the checkpoint event, so its two CPU
standard-normal draws advance the torch CPU RNG exactly as in the source.

Two modes share this engine and nothing else:
  SCIENTIFIC     seeds 42/1337/2026, 200 epochs, <runtime_root>/runs/m6/E06c/seed_<seed>/,
                 launched only by tools/run_e06c.py from a clean worktree.
  QUALIFICATION  seed 60506, <runtime_root>/qualification/m6d5e/E06c/q60506-<run_id>/,
                 launched only by methods/dsdg/runner_qualification.py; never scientific.
Torch is imported by the caller; importing this module is static.
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
import yaml

from methods.common.config import ROOT, load_method_config, sha256_file
from methods.common.learned import PreparationError, apply_framework_seed, seed_torch_worker, torch_loader_options
from methods.common.learned_runlog import LearnedRunContext
from methods.common.runlog import (RunContext, RunDirectoryError, atomic_write_json, compute_run_id, git_commit,
                                   git_dirty, load_logging_contract)
from methods.dsdg import microbatch_execution as v1
from methods.dsdg import microbatch_execution_v2 as v2
from methods.dsdg import runner_io as rio
from methods.dsdg import training_graph as tg

MODELS = ('netE_nir', 'netE_vis', 'netG', 'netCls', 'netIP')
LOCK_PATH = 'environments/e06c.lock.json'
require = rio.require


class TrainingStop(RuntimeError):
    """A real-data gate failed (gradient coverage, finiteness, batch plan, OOM): STOP_AND_REPORT."""


def gate(value, message):
    if not value:
        raise TrainingStop('E06c STOP: ' + message)


def tsha(t):
    return rio.sha(t.detach().cpu().contiguous().numpy().tobytes())


# ================================================================= run context (run_logging_v1)
class E06cRunContext(LearnedRunContext):
    """LearnedRunContext with the E06c overlay identities made visible.

    SCIENTIFIC mode is the unchanged LearnedRunContext (experiment seed, runs/m6 layout,
    checkpoint_metadata validation). QUALIFICATION mode binds the same writers to the
    qualification root; RunContext refuses non-experiment seeds by design, so the
    identity attributes are set here explicitly (experiment_seed stays null) with the
    run_logging_v1 run_id formula applied to the qualification seed.
    """
    def __init__(self, *, mode, seed, runtime_root, environment, missing_environment_reasons, identities,
                 resume=False, command_line=None):
        rio.validate_mode_seed(mode, seed)
        config = load_method_config(rio.METHOD_ID)
        self.mode, self.identities = mode, dict(identities)
        require(self.identities['config_sha256'] == config['_runtime']['config_sha256'], 'config identity')
        if mode == rio.SCIENTIFIC:
            super().__init__(method_id=rio.METHOD_ID, seed=seed, config=config, runtime_root=runtime_root,
                             command_line=command_line, resume=resume, environment=environment,
                             missing_environment_reasons=missing_environment_reasons)
            self.qualification_seed = None
            return
        self.contract = load_logging_contract()
        self.method_id, self.seed, self.qualification_seed = rio.METHOD_ID, None, seed
        self.config, self.runtime_root, self.resume = config, Path(runtime_root), resume
        self.source_commits = {config['source']['repository']: config['source']['pinned_commit']}
        self.command_line = command_line if command_line is not None else ' '.join(sys.argv)
        self.config_sha256 = config['_runtime']['config_sha256']
        self.config_path = config['_runtime']['config_path']
        self.git_commit = git_commit()
        self.run_id = compute_run_id(rio.METHOD_ID, seed, self.config_sha256, self.git_commit)
        self.run_dir = rio.run_root(runtime_root, mode, seed, self.run_id)
        self.start_utc = self._metrics_fh = self._generation_fh = self._closed_summary = self._lock_fh = None
        self._records, self.run_uuid, self._started = 0, str(uuid.uuid4()), time.monotonic()
        # LearnedRunContext environment validation, unchanged in substance
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
        return self.seed if self.mode == rio.SCIENTIFIC else self.qualification_seed

    def open(self):
        if self.mode == rio.SCIENTIFIC:
            return super().open()
        # RunContext.open with the lock named after the qualification run directory
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
        if self.mode == rio.QUALIFICATION and not self.resume and self.run_dir.exists():
            raise RunDirectoryError(f'STOP_AND_REPORT: qualification directory already exists: {self.run_dir}')
        if self.resume:
            require(self.path('resolved_config').is_file(), 'resume requires resolved_config.yaml')
            expected = yaml.safe_dump(self._resolved_config(), sort_keys=False, allow_unicode=True).encode('utf-8')
            if self.path('resolved_config').read_bytes() != expected:
                raise RunDirectoryError('resume refused: resolved_config.yaml bytes differ (identity/overlay drift)')
        result = super()._open_locked()
        index = json.loads(self.path('checkpoint_index').read_text())
        index.update(note='Official model checkpoints actually written (train_generator.py:214-217 cadence). '
                          'Engineering resume sidecars are indexed separately in resume_state_index.json.',
                     runner_mode=self.mode, checkpoint_rule=self.config['checkpoint']['rule'],
                     selected_final=('netG_model_epoch_200_iter_0.pth' if self.mode == rio.SCIENTIFIC else None),
                     scientific_checkpoints=self.mode == rio.SCIENTIFIC)
        if self.mode == rio.QUALIFICATION:
            index.update(labels=list(rio.QUALIFICATION_LABELS), qualification_seed=self.qualification_seed,
                         selected_final_reason='QUALIFICATION_ONLY: no scientific selection exists')
        atomic_write_json(self.path('checkpoint_index'), index)
        rio.init_resume_index(self.run_dir, run_id=self.run_id, mode=self.mode, seed=self.seed_value)
        return result

    def _resolved_config(self):
        resolved = super()._resolved_config()
        ids = self.identities
        resolved['_resolved'].update({
            'runner_mode': self.mode, 'qualification_seed': self.qualification_seed,
            'scientific_run': self.mode == rio.SCIENTIFIC, 'qualification_only': self.mode == rio.QUALIFICATION,
            'labels': list(rio.QUALIFICATION_LABELS) if self.mode == rio.QUALIFICATION else [],
            'note': 'Verbatim copy of the frozen config above; every execution adaptation is listed here and was '
                    'NOT part of the frozen YAML.',
            'identities': ids,
            'base_config': {'path': self.config_path, 'sha256': ids['config_sha256']},
            'a1_adaptation': {'path': 'configs/frozen/dsdg_bin_idfree_v1.yaml', 'sha256': ids['a1_adaptation_sha256']},
            'execution_overlays': [
                {'path': v1.RESOLUTION_PATH, 'sha256': ids['m6d5c_v1_overlay_sha256'], 'mode': v1.EXECUTION_MODE},
                {'path': v2.RESOLUTION_PATH, 'sha256': ids['m6d5d_v2_overlay_sha256'], 'mode': v2.EXECUTION_MODE},
                {'path': rio.CONTRACT_PATH, 'sha256': ids['m6d5e_contract_sha256'], 'mode': 'PRODUCTION_RUNNER_CONTRACT'}],
            'execution_adaptations': {
                'classification': 'CONTROLLED_EXECUTION_ADAPTATION', 'execution_mode': rio.EXECUTION_MODE,
                'physical_batch_240': 'OOM_RETAINED', 'nominal_global_batch': 240, 'max_microbatch': 20,
                'tail_batch': 198, 'tail_chunks': [20] * 9 + [18], 'drop_last': False,
                'optimizer_steps_per_epoch': 37, 'epsilon': 'drawn once per actual global batch [B,128], cls->nir->vis',
                'lightcnn_target_features': 'no_grad (approved dead-gradient elimination)',
                'official_visualization_block': (
                    'executed at epoch 1 and every 10 epochs after the epoch loop, before the checkpoint event '
                    '(train_generator.py:197-211): two CPU standard-normal [240,128] draws (noise, then noise_s), '
                    'netG(cat(noise_s, noise, noise)) under no_grad (memory only: no backward follows in the source), '
                    'six TRAIN-sample PNG grids under diagnostics/visualization/; diagnostic only, not validation, '
                    'selects nothing')},
            'source_pin': {'repository': self.config['source']['repository'], 'commit': ids['source_commit'],
                           'tree': ids['source_tree']},
            'environment_lock': {'path': LOCK_PATH, 'sha256': ids['environment_lock_sha256']},
            'lightcnn_sha256': ids['lightcnn_sha256'],
            'training': {'all_epochs': self.config['training']['all_epochs'],
                         'epochs_executed_limit': (self.config['training']['all_epochs']
                                                   if self.mode == rio.SCIENTIFIC else 1),
                         'global_batch': 240, 'max_microbatch': 20, 'drop_last': False},
            'dataloader': dict(rio.LOADER, generator_seed=self.seed_value,
                               worker_init_fn='methods.common.learned.seed_torch_worker'),
            'optimizer_contract': 'torch.optim.Adam(netE_nir + netE_vis + netG, lr=2e-4); netCls/netIP excluded; no scheduler',
            'loss_coefficients': tg.FROZEN_LAMBDAS, 'warmup': 'epoch < 2: 0.01 x every term except loss_rec',
            'checkpoint_rule': {'rule': self.config['checkpoint']['rule'], 'cadence': 'epoch 1 and every 10 epochs',
                                'selected': 'netG_model_epoch_200_iter_0.pth'},
            'resume_state': 'checkpoints/runner_state_epoch_<E>.pth + resume_state_index.json (engineering only)',
            'logging_contract': {'version': 'run_logging_v1', 'sha256': ids['logging_contract_sha256'],
                                 'record_per': 'GLOBAL_OPTIMIZER_STEP'},
            'run_id_rule': 'run_logging_v1: sha256(method_id|seed|config_sha256|git_commit)[:16]; resolved_config_sha256 '
                           'is recorded in run_manifest.json and the resume state (no second run-id convention)'})
        return resolved

    @property
    def resolved_config_sha256(self):
        p = self.path('resolved_config')
        return sha256_file(p) if p.is_file() else None

    def _manifest(self, **kwargs):
        m = super()._manifest(**kwargs)
        m.update(runner_mode=self.mode, scientific_run=self.mode == rio.SCIENTIFIC,
                 qualification_only=self.mode == rio.QUALIFICATION, qualification_seed=self.qualification_seed,
                 resolved_config_path='resolved_config.yaml', resolved_config_sha256=self.resolved_config_sha256,
                 execution_mode=rio.EXECUTION_MODE, physical_batch_240='OOM_RETAINED', nominal_global_batch=240,
                 max_microbatch=20, tail_batch=198, drop_last=False, fidelity_class='CONTROLLED_ADAPTATION',
                 execution_classification='CONTROLLED_EXECUTION_ADAPTATION', deviation='DEV-020',
                 identities=self.identities, environment_lock_sha256=self.identities['environment_lock_sha256'],
                 m6d5e_file_sha256=rio.m6d5e_file_hashes(), val_split_accessed=False)
        if self.mode == rio.QUALIFICATION:
            m['labels'] = list(rio.QUALIFICATION_LABELS)
            m['missing_field_reasons']['experiment_seed'] = (f'QUALIFICATION_ONLY: seed {self.qualification_seed} '
                                                             'is not an experiment seed')
        return m

    def record_checkpoint(self, **kwargs):
        if self.mode == rio.SCIENTIFIC:
            return super().record_checkpoint(**kwargs)
        require(kwargs['checkpoint_type'] == 'periodic' and kwargs['selected_for_final'] is False,
                'qualification checkpoints are never terminal/selected')
        require(rio.is_checkpoint_epoch(kwargs['epoch']), 'qualification checkpoint epoch outside the official cadence')
        return RunContext.record_checkpoint(self, **kwargs)


# ================================================================= runtime setup
def configure_precision(torch):
    """FP32 only: no TF32, no autocast, highest matmul precision, deterministic cuDNN without benchmark."""
    require(os.environ.get('NVIDIA_TF32_OVERRIDE') == '0', 'NVIDIA_TF32_OVERRIDE=0 required')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_WORKSPACE_CONFIG=:4096:8 required')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    require(not torch.is_autocast_enabled('cuda'), 'autocast disabled')
    return {'default_dtype': str(torch.get_default_dtype()), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'autocast_cuda': False,
            'matmul_precision': torch.get_float32_matmul_precision(), 'amp': False, 'activation_checkpointing': False}


def seed_process(torch, mode, seed, config):
    """Scientific: learned.apply_framework_seed; qualification: the same calls with the qualification seed."""
    rio.validate_mode_seed(mode, seed)
    require(os.environ.get('PYTHONHASHSEED') == str(seed), f'launch with PYTHONHASHSEED={seed}')
    if mode == rio.SCIENTIFIC:
        plan = apply_framework_seed(config, seed, 'torch', cuda=True)
        return {'seed': seed, 'role': 'EXPERIMENT_SEED', 'plan': 'methods.common.learned.apply_framework_seed',
                'cudnn_deterministic': plan['cudnn_deterministic']}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    return {'seed': seed, 'role': 'QUALIFICATION_SEED_NOT_EXPERIMENT_SEED',
            'plan': 'random/numpy/torch/torch.cuda seeded identically to apply_framework_seed'}


def environment_record(torch):
    env = {'host': socket.gethostname(), 'user': getpass.getuser(), 'platform': platform.platform(),
           'python_version': platform.python_version(), 'numpy_version': np.__version__,
           'gpu_model': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(),
           'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
           'framework': 'torch', 'framework_version': torch.__version__, 'pytorch_version': torch.__version__,
           'tensorflow_version': None, 'dependency_fingerprint': sha256_file(ROOT / LOCK_PATH),
           'environment_lock_path': LOCK_PATH, 'environment_name': 'gpat-m6-e06c', 'python_executable': sys.executable,
           'gpu_note': 'single RTX 3090; physical B=240 OOM retained; GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2'}
    return env, {'tensorflow_version': 'E06c is a PyTorch method; TensorFlow is not part of gpat-m6-e06c'}


def verify_overlays():
    """Explicit M6D5c/M6D5d owner-overlay binding (adapter.validate_batch is not an execution gate)."""
    policy_v1 = v1.execution_guard(v1.load_resolution())
    policy_v2 = v2.execution_guard(v2.load_resolution())
    plan = v2.epoch_batch_plan()
    require((plan['full_batches'], plan['final_batch'], plan['optimizer_steps_per_epoch'], plan['rows_covered']) ==
            (36, 198, 37, 8838), 'epoch plan 36 x 240 + 198')
    require([s.stop - s.start for s in v2.chunk_slices_v2(198)] == [20] * 9 + [18], 'tail chunks')
    return {'v1': policy_v1, 'v2': policy_v2, 'epoch_plan': {k: v for k, v in plan.items() if k != 'batch_sizes'}}


def build_loader(torch, dataset, mode, seed, config):
    """DataLoader(batch 240, shuffle, 8 workers, pin_memory, drop_last=False) with the seeded generator."""
    if mode == rio.SCIENTIFIC:
        opts = torch_loader_options(config, seed)      # validates the experiment seed
    else:
        rio.validate_mode_seed(mode, seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        opts = {'generator': generator, 'worker_init_fn': seed_torch_worker}
    loader = torch.utils.data.DataLoader(dataset, **rio.LOADER, **opts)
    return loader, opts['generator'], verify_loader(torch, loader, opts['generator'], len(dataset))


def verify_loader(torch, loader, generator, n):
    s, b = loader.sampler, loader.batch_sampler
    require((loader.batch_size, loader.drop_last, loader.num_workers, loader.pin_memory, loader.persistent_workers)
            == (240, False, 8, True, False), 'DataLoader batch/drop_last/workers/pin_memory')
    require(type(s) is torch.utils.data.RandomSampler and s.replacement is False and s.generator is generator and
            s.num_samples == n, 'RandomSampler without replacement on the seeded generator')
    require(type(b) is torch.utils.data.BatchSampler and b.batch_size == 240 and b.drop_last is False, 'BatchSampler')
    require(loader.worker_init_fn is seed_torch_worker and loader.generator is generator, 'worker seeding')
    return {'batch_size': 240, 'shuffle': True, 'num_workers': 8, 'pin_memory': True, 'drop_last': False,
            'persistent_workers': False, 'sampler': 'RandomSampler(replacement=False, generator=G)',
            'batch_sampler': 'BatchSampler(240, drop_last=False)', 'worker_init_fn': 'seed_torch_worker',
            'len_batches': len(loader), 'rows': n}


def lightcnn_identity(asset_path):
    raw_sha, size = sha256_file(asset_path), Path(asset_path).stat().st_size
    require((raw_sha, size) == ('d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964', 123844849),
            'LightCNN SHA256/bytes')
    return {'path': str(asset_path), 'sha256': raw_sha, 'bytes': size}


# ================================================================= trainer
class Trainer:
    """One E06c training process: models, Adam, loader and the V2 global step."""

    def __init__(self, torch, F, util, models, optimizer, loader, generator, records, mode, seed, lam, device='cuda',
                 visualization_every=rio.VISUALIZATION_EVERY):
        self.torch, self.F, self.util = torch, F, util
        self.visualization_every, self.visual = visualization_every, None
        self.models, self.optimizer, self.loader, self.generator = models, optimizer, loader, generator
        self.records, self.mode, self.seed, self.lam, self.device = records, mode, seed, lam, device
        self.nets = [models[n] for n in MODELS]
        self.owned = [(f'{n}.{k}', p) for n in tg.OPTIMIZER_OWNED for k, p in models[n].named_parameters()]
        self.criterion_type, self.criterionL2 = tg.criteria(torch) if device == 'cuda' else (
            torch.nn.CrossEntropyLoss(), torch.nn.MSELoss())
        self.completed_epoch, self.global_step = 0, 0
        self.optimizer_applications = self.backward_calls = 0
        self.peak_step_gpu_memory_bytes = 0
        self.ctx = None
        optimizer.register_step_post_hook(lambda *a: setattr(self, 'optimizer_applications',
                                                             self.optimizer_applications + 1))

    # ------------------------------------------------------------- one global batch
    def set_modes(self):
        """train_generator.py:105-108 at the start of every epoch."""
        m = self.models
        m['netE_nir'].train(); m['netE_vis'].train(); m['netG'].train(); m['netIP'].eval()

    def global_step_run(self, batch, epoch, iteration, capture=False, keep_visual=False):
        torch = self.torch
        indices = batch['index'].tolist()
        pair_ids = [self.records[i]['pair_id'] for i in indices]
        x_spoof, x_live = batch['0'].to(self.device), batch['1'].to(self.device)
        label = batch['type'].to(self.device)
        B = x_spoof.shape[0]
        gate(len(set(indices)) == B and 1 <= B <= 240, 'batch rows unique, 1 <= B <= 240')
        gate(tuple(x_spoof.shape) == tuple(x_live.shape) == (B, 3, 256, 256) and
             x_spoof.dtype == x_live.dtype == torch.float32, 'inputs FP32 [B,3,256,256]')
        gate(label.dtype == torch.int64 and tuple(label.shape) == (B,) and not bool(label.any()),
             'single-spoof CE index 0 for every row')
        lo, hi = min(float(x_spoof.min()), float(x_live.min())), max(float(x_spoof.max()), float(x_live.max()))
        gate(0.0 <= lo and hi <= 1.0, 'inputs in [0,1] (adapter /255 only)')
        chunks = [s.stop - s.start for s in v2.chunk_slices_v2(B)]
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        probe = {}
        if capture:
            probe['rng_before'] = rng_hashes(torch)
            probe['parameters_before_sha256'] = {n: tsha(p) for n, p in self.owned}
        eps = v2.draw_epsilon(torch, B, self.device)
        eps_sha = {k: tsha(eps[k]) for k in v2.EPS_ORDER}
        seen = {}

        def before_step():
            grads = [p.grad for _, p in self.owned]
            present = [g for g in grads if g is not None]
            finite_count = int(torch.stack([torch.isfinite(g).all() for g in present]).sum()) if present else 0
            seen['owned'] = {'tensors': len(grads), 'non_none': len(present), 'finite': finite_count}
            gate(len(grads) == len(present) == finite_count == 55, '55/55 owned finite gradients: ' + repr(seen['owned']))
            gate(all(p.grad is None for p in self.models['netIP'].parameters()), 'netIP receives no gradient')
            cls_nonzero = sum(int(torch.count_nonzero(p.grad)) for p in self.models['netCls'].parameters()
                              if p.grad is not None)
            gate(cls_nonzero == 0, 'netCls gradient is exactly zero (one-logit CE)')
            seen['netCls_grad_nonzero'] = cls_nonzero
            if capture:
                probe['gradient_sha256'] = {n: tsha(p.grad) for n, p in self.owned}
                probe['gradients'] = {n: p.grad.detach().cpu().clone() for n, p in self.owned}
                probe['gradient_nonzero_tensors'] = sum(int(bool(p.grad.ne(0).any())) for _, p in self.owned)
                probe['rng_before_step'] = rng_hashes(torch)
        recs = []

        def on_chunk(i, out):   # the last batch's 128-px reconstructions (pre-step), as train_generator.py:156-157
            recs.append((out['rec_nir'].detach().clone(), out['rec_vis'].detach().clone()))
        before_apps = self.optimizer_applications
        try:
            run = v2.run_global_batch_v2(torch, self.F, self.util, self.nets, self.optimizer, x_spoof, x_live, label,
                                         eps, self.lam, self.criterion_type, self.criterionL2, epoch,
                                         on_chunk=on_chunk if keep_visual else None, before_step=before_step)
        except torch.OutOfMemoryError as error:
            raise TrainingStop(f'CUDA OOM at epoch {epoch} iteration {iteration} (B={B}): {error}') from error
        gate(self.optimizer_applications == before_apps + 1, 'exactly one optimizer.step() per global batch')
        gate(run['chunk_sizes'] == chunks and run['global_batch'] == B, 'executed chunk plan')
        self.backward_calls += len(chunks)
        if keep_visual:
            gate(len(recs) == len(chunks), 'one reconstruction per chunk')
            size = (128, 128)
            self.visual = {'epoch': epoch, 'global_step': self.global_step + 1, 'batch_size': B, 'grids': {
                'img_spoof': self.F.interpolate(x_spoof, size=size, mode='bilinear'),
                'img_live': self.F.interpolate(x_live, size=size, mode='bilinear'),
                'rec_spoof': torch.cat([r[0] for r in recs]), 'rec_live': torch.cat([r[1] for r in recs])}}
        losses = run['global_losses']
        total = tg.total_loss(losses, epoch)
        gate(rio.finite([float(v) for v in losses.values()] + [float(total)]), 'finite global losses')
        gate(losses['loss_cls'] == 0.0 and losses['loss_pair'] == 0.0, 'one-logit CE and lambda_pair=0 give zero')
        self.global_step += 1
        step_seconds = time.monotonic() - t0
        mem = torch.cuda.max_memory_allocated() if self.device == 'cuda' else None
        self.peak_step_gpu_memory_bytes = max(self.peak_step_gpu_memory_bytes, mem or 0)
        result = {'epoch': epoch, 'global_step': self.global_step, 'iteration': iteration, 'batch_size': B,
                  'chunk_sizes': chunks, 'pair_ids': pair_ids, 'batch_pair_sha256': rio.order_sha256(pair_ids),
                  'epsilon_sha256': eps_sha, 'losses': {k: float(v) for k, v in losses.items()},
                  'total_loss': float(total), 'step_seconds': step_seconds, 'gpu_memory_bytes': mem,
                  'owned_gradients': seen['owned'], 'netCls_grad_nonzero': seen['netCls_grad_nonzero'],
                  'learning_rate': self.optimizer.param_groups[0]['lr']}
        if capture:
            st = run['stats']
            probe.update(pass1={'delta_sha256': tsha(st['delta']), 'ort_mean': float(st['ort_mean']),
                                'ort_mean_sha256': tsha(st['ort_mean']), 'mmd_sign_sha256': tsha(st['mmd_sign'])},
                         surrogate_sums=run['surrogate_sums'], sum_of_chunk_objectives=run['sum_of_chunk_objectives'],
                         parameters_after_sha256={n: {k: tsha(p) for k, p in m.named_parameters()}
                                                  for n, m in self.models.items()},
                         parameters_after={n: p.detach().cpu().clone() for n, p in self.owned},
                         optimizer_state=optimizer_state_hashes(self.optimizer, self.owned),
                         rng_after=rng_hashes(torch))
            result['probe'] = probe
        return result

    # ------------------------------------------------------------- epochs
    def run_epoch(self, epoch, ctx=None, t_start=None):
        """One full pass: 36 x 240 + 198, one trajectory record per global optimizer step."""
        self.set_modes()
        sizes, order, losses = [], [], []
        t_start = time.monotonic() if t_start is None else t_start
        peak = 0
        last = len(self.loader) - 1
        visual_epoch = rio.is_visualization_epoch(epoch, self.visualization_every)
        for iteration, batch in enumerate(self.loader):
            r = self.global_step_run(batch, epoch, iteration, keep_visual=visual_epoch and iteration == last)
            sizes.append(r['batch_size'])
            order.extend(r['pair_ids'])
            losses.append({'global_step': r['global_step'], 'batch_size': r['batch_size'], **r['losses'],
                           'total_loss': r['total_loss']})
            peak = max(peak, r['gpu_memory_bytes'] or 0)
            if ctx is not None:
                ctx.log_epoch(rio.step_record(
                    epoch=epoch, global_step=r['global_step'], iteration=iteration, learning_rate=r['learning_rate'],
                    losses=r['losses'], total_loss=r['total_loss'], wall_clock_seconds=time.monotonic() - t_start,
                    gpu_memory_bytes=r['gpu_memory_bytes'], batch_size=r['batch_size'], chunk_sizes=r['chunk_sizes'],
                    batch_pair_sha256=r['batch_pair_sha256'], epsilon_sha256=r['epsilon_sha256'], mode=self.mode,
                    seed=self.seed, extra={'step_seconds': r['step_seconds'],
                                           'owned_gradients_finite': r['owned_gradients']['finite'],
                                           'optimizer_applications_total': self.optimizer_applications}))
        plan = v2.epoch_batch_plan(len(self.records))
        gate(sizes == plan['batch_sizes'], f'epoch batch plan {sizes} != 36 x 240 + 198')
        ids = {r['pair_id'] for r in self.records}
        gate(len(order) == len(ids) == len(set(order)) and set(order) == ids,
             'every TRAIN pair exactly once (no drop, no duplicate)')
        self.completed_epoch = epoch
        summary = {'epoch': epoch, 'batch_sizes': sizes, 'optimizer_steps': len(sizes), 'rows': len(order),
                   'unique_pair_ids': len(set(order)), 'epoch_pair_order_sha256': rio.order_sha256(order),
                   'global_step_end': self.global_step, 'peak_step_gpu_memory_bytes': peak}
        if ctx is not None:
            ctx.log_event('e06c_epoch_complete', {k: v for k, v in summary.items() if k != 'batch_sizes'})
        return summary, losses

    # ------------------------------------------------------------- upstream visualization block
    def visualize(self, epoch, ctx):
        """train_generator.py:197-211, after the epoch loop and before the checkpoint event. Diagnostic only.

        The two CPU draws are made exactly as in the source (shape [240,128], noise then noise_s),
        so the torch CPU RNG advances as the pinned trainer's does. netG runs in its training-time
        mode (InstanceNorm without running statistics, no dropout: no buffer or parameter effect)
        under no_grad, because the source never backpropagates `fake`. The grids are the last
        TRAIN batch (128 px), its pre-step reconstructions and the noise fakes.
        """
        torch = self.torch
        require(rio.is_visualization_epoch(epoch, self.visualization_every), 'not a visualization epoch')
        require(self.visual is not None and self.visual['epoch'] == epoch, 'last-batch grids of this epoch')
        import torchvision.utils as vutils
        cpu_before = torch.get_rng_state()
        noise = torch.zeros(*rio.VISUALIZATION_NOISE_SHAPE).normal_(0, 1)
        noise_s = torch.zeros(*rio.VISUALIZATION_NOISE_SHAPE).normal_(0, 1)
        cpu_after = torch.get_rng_state()
        z = torch.cat((noise_s, noise, noise), dim=1).to(self.device)
        with torch.no_grad():
            fake = self.models['netG'](z)
        grids = dict(self.visual['grids'], fake_spoof=fake[:, 0:3, :, :], fake_live=fake[:, 3:6, :, :])
        out = ctx.run_dir / rio.VISUALIZATION_DIR
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for name in rio.VISUALIZATION_FILES:
            rel = rio.VISUALIZATION_DIR + '/' + rio.visualization_basename(epoch, name)
            require(not (ctx.run_dir / rel).exists(), 'visualization file already exists (no overwrite)')
            vutils.save_image(grids[name].data, str(ctx.run_dir / rel))
            files.append({'path': rel, 'images': int(grids[name].shape[0]), 'tensor_shape': list(grids[name].shape),
                          'bytes': (ctx.run_dir / rel).stat().st_size, 'sha256': sha256_file(ctx.run_dir / rel)})
        evidence = {'epoch': epoch, 'global_step': self.global_step, 'role': 'UPSTREAM_TRAINING_SAMPLE_VISUALIZATION',
                    'not_validation': True, 'selects_nothing': True, 'data': 'TRAIN (last global batch of the epoch)',
                    'last_batch_global_step': self.visual['global_step'], 'last_batch_size': self.visual['batch_size'],
                    'noise_draw_order': ['noise', 'noise_s'], 'noise_shape': list(rio.VISUALIZATION_NOISE_SHAPE),
                    'noise_sha256': tsha(noise), 'noise_s_sha256': tsha(noise_s),
                    'torch_cpu_rng_before_sha256': rio.sha(cpu_before.numpy().tobytes()),
                    'torch_cpu_rng_after_sha256': rio.sha(cpu_after.numpy().tobytes()),
                    'netG_forward': 'no_grad (source builds a graph it never uses)', 'files': files}
        self.visual = None
        ctx.log_event('e06c_visualization_diagnostic', evidence)
        return evidence

    # ------------------------------------------------------------- checkpoints and resume state
    def checkpoint_event(self, epoch, ctx):
        """Pinned misc/util.py::save_checkpoint for netE_spoof/netE_live/netG, then the engineering sidecar."""
        require(rio.is_checkpoint_epoch(epoch), 'not an official checkpoint epoch')
        ckdir = ctx.run_dir / 'checkpoints'
        written = []
        for model, prefix in rio.OFFICIAL_FILES:
            self.util.save_checkpoint(str(ckdir) + '/', self.models[model], epoch, 0, prefix)
            rel = 'checkpoints/' + rio.official_basename(prefix, epoch)
            kind, selected = rio.checkpoint_kind(model, epoch)
            if ctx.mode == rio.QUALIFICATION:
                reason = ('QUALIFICATION_ONLY: official cadence event (epoch-1 source exception) in the qualification '
                          'root; NOT_SCIENTIFIC_CHECKPOINT; never a selection candidate')
            else:
                reason = ctx.config['checkpoint']['rule'] if selected else 'Official cadence; not final selection'
            p = ctx.run_dir / rel
            meta = dict(path=rel, epoch=epoch, global_step=self.global_step, file_size_bytes=p.stat().st_size,
                        sha256=sha256_file(p), checkpoint_type=kind, selected_for_final=selected,
                        selection_reason=reason)
            ctx.record_checkpoint(**meta)
            written.append(dict(meta, model=model))
        state_rel = 'checkpoints/' + rio.resume_state_name(epoch)
        tmp = ctx.run_dir / (state_rel + '.partial')
        self.torch.save(self.resume_state(ctx), tmp)
        os.replace(tmp, ctx.run_dir / state_rel)
        entry = rio.record_resume_state(ctx.run_dir, path=state_rel, epoch=epoch, global_step=self.global_step,
                                        resolved_config_sha256=ctx.resolved_config_sha256)
        ctx.log_event('e06c_checkpoint_event', {'epoch': epoch, 'global_step': self.global_step,
                                                'official': [{k: w[k] for k in ('path', 'sha256', 'checkpoint_type')}
                                                             for w in written],
                                                'resume_state': {'path': entry['path'], 'sha256': entry['sha256']}})
        return written, entry

    def resume_state(self, ctx):
        torch = self.torch
        npname, keys, pos, has_gauss, cached = np.random.get_state()
        return {'kind': rio.RESUME_KIND, 'schema_version': 1, 'scientific_checkpoint': False,
                'selection_candidate': False, 'method_id': rio.METHOD_ID, 'runner_mode': ctx.mode,
                'seed': self.seed, 'run_id': ctx.run_id, 'completed_epoch': self.completed_epoch,
                'global_step': self.global_step, 'resolved_config_sha256': ctx.resolved_config_sha256,
                'identities': dict(ctx.identities), 'execution_mode': rio.EXECUTION_MODE,
                'models': {n: self.models[n].state_dict() for n in ('netE_nir', 'netE_vis', 'netG', 'netCls')},
                'netIP': 'NOT_STORED: immutable LightCNN asset re-read and SHA-verified on resume',
                'optimizer': self.optimizer.state_dict(),
                'rng': {'python': random.getstate(),
                        'numpy': {'name': npname, 'keys': torch.from_numpy(keys.astype(np.int64)), 'pos': int(pos),
                                  'has_gauss': int(has_gauss), 'cached_gaussian': float(cached)},
                        'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()},
                'loader_generator': self.generator.get_state(),
                'metrics': {'trajectory_records': self.global_step, 'last_global_step': self.global_step},
                'optimizer_applications_total': self.optimizer_applications}

    def load_resume_state(self, path, ctx):
        """Validate identity then restore models, Adam, RNG and loader-generator state exactly."""
        torch = self.torch
        state = torch.load(path, map_location=None, weights_only=True)
        expected = {'kind': rio.RESUME_KIND, 'method_id': rio.METHOD_ID, 'runner_mode': ctx.mode, 'seed': self.seed,
                    'run_id': ctx.run_id, 'resolved_config_sha256': ctx.resolved_config_sha256,
                    'identities': dict(ctx.identities), 'execution_mode': rio.EXECUTION_MODE,
                    'scientific_checkpoint': False, 'selection_candidate': False}
        bad = sorted(k for k, v in expected.items() if state.get(k) != v)
        require(not bad, 'resume state identity mismatch: ' + ', '.join(bad))
        require(type(state['completed_epoch']) is int and type(state['global_step']) is int and
                state['global_step'] == len(self.loader) * state['completed_epoch'],
                'completed epoch / global step consistency (epoch-boundary sidecar)')
        for n, sd in state['models'].items():
            self.models[n].load_state_dict(sd, strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        rng = state['rng']
        random.setstate(rng['python'])
        n = rng['numpy']
        np.random.set_state((n['name'], n['keys'].numpy().astype(np.uint32), n['pos'], n['has_gauss'],
                             n['cached_gaussian']))
        torch.set_rng_state(rng['torch_cpu'])
        torch.cuda.set_rng_state_all(rng['torch_cuda'])
        self.generator.set_state(state['loader_generator'])
        self.completed_epoch, self.global_step = state['completed_epoch'], state['global_step']
        self.optimizer_applications = state['optimizer_applications_total']
        return {'completed_epoch': self.completed_epoch, 'global_step': self.global_step,
                'restored': ['netE_nir', 'netE_vis', 'netG', 'netCls', 'Adam', 'python RNG', 'numpy RNG',
                             'torch CPU RNG', 'torch CUDA RNG', 'DataLoader generator'],
                'netIP': 'reloaded from the SHA-verified LightCNN asset'}


def rng_hashes(torch):
    return {'python_sha256': rio.sha(repr(random.getstate()).encode()),
            'numpy_sha256': rio.sha(repr(np.random.get_state()[1].tolist()).encode() +
                                    repr(np.random.get_state()[2:]).encode()),
            'torch_cpu_sha256': rio.sha(torch.get_rng_state().numpy().tobytes()),
            'torch_cuda_sha256': rio.sha(torch.cuda.get_rng_state().numpy().tobytes())}


def optimizer_state_hashes(optimizer, owned):
    out = {}
    for name, p in owned:
        s = optimizer.state[p]
        out[name] = {'step': float(s['step']), 'exp_avg_sha256': tsha(s['exp_avg']),
                     'exp_avg_sq_sha256': tsha(s['exp_avg_sq'])}
    return out


# ================================================================= scientific entry (tools/run_e06c.py)
def run_scientific(*, seed, runtime_root, faces_root, resume_state=None, command_line=None):
    """Full 200-epoch SCIENTIFIC run. Never called by M6D5e (qualification uses its own harness)."""
    import torch
    import torch.nn.functional as F
    from methods.common.upstream import upstream_modules
    from methods.dsdg import DSDGAdapter
    from methods.dsdg import runtime as m6d5a
    from methods.dsdg import training_qualification as m6d5b
    mode = rio.SCIENTIFIC
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    contract = rio.load_contract()
    ids = rio.identities(contract)
    adapter = DSDGAdapter()
    config = adapter.config
    overlays = verify_overlays()
    precision = configure_precision(torch)
    seeding = seed_process(torch, mode, seed, config)
    source = adapter.validate_source()
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']), 'source identity')
    lock = json.loads((ROOT / LOCK_PATH).read_text())
    require(source['files_sha256'] == lock['source']['files_sha256'], 'source closure equals environment lock')
    lightcnn = lightcnn_identity(m6d5a.LIGHTCNN)
    records, datasets, relation = rio.read_train_relation(adapter)
    reader = rio.CanonicalFaceReader(faces_root, datasets)
    dataset = rio.IndexedPairDataset(adapter, records, reader)
    env, reasons = environment_record(torch)
    ctx = E06cRunContext(mode=mode, seed=seed, runtime_root=runtime_root, environment=env,
                         missing_environment_reasons=reasons, identities=ids, resume=resume_state is not None,
                         command_line=command_line)
    root = Path(source['root']) / config['source']['relevant_path']
    with upstream_modules(root, m6d5a.UPSTREAM_MODULES, m6d5a.UPSTREAM_ROOTS) as modules:
        models, _ = m6d5b.build_models(torch, modules['networks'], root, config)
        m6d5a.load_lightcnn(torch, models['netIP'])
        optimizer = tg.build_optimizer(torch, models['netE_nir'], models['netE_vis'], models['netG'],
                                       config['optimizer']['learning_rate'])
        m6d5b.optimizer_evidence(torch, optimizer, models)
        loader, generator, loader_evidence = build_loader(torch, dataset, mode, seed, config)
        trainer = Trainer(torch, F, modules['misc.util'], models, optimizer, loader, generator, records, mode, seed,
                          tg.frozen_lambdas(config), visualization_every=rio.visualization_every(config))
        with ctx:
            streams = rio.tee_run_logs(ctx.run_dir)
            try:
                if resume_state is not None:
                    path, entry = rio.resume_entry(ctx.run_dir, resume_state)
                    restored = trainer.load_resume_state(path, ctx)
                    agreement = rio.reconcile_metrics(ctx.path('metrics'), completed_epoch=restored['completed_epoch'],
                                                      global_step=restored['global_step'])
                    ctx.log_event('e06c_resume_reconciliation', {'resume_state': entry['path'],
                                                                 'resume_state_sha256': entry['sha256'],
                                                                 **restored, **agreement, 'overwrite': False})
                else:
                    ctx.log_event('e06c_run_start', {'seeding': seeding, 'precision': precision, 'overlays': overlays,
                                                     'loader': loader_evidence, 'relation': relation,
                                                     'lightcnn': lightcnn})
                t0 = time.monotonic()
                for epoch in range(trainer.completed_epoch + 1, rio.ALL_EPOCHS + 1):
                    trainer.run_epoch(epoch, ctx, t0)
                    if rio.is_visualization_epoch(epoch, trainer.visualization_every):
                        trainer.visualize(epoch, ctx)             # source order: visualization, then checkpoint
                    if rio.is_checkpoint_epoch(epoch):
                        trainer.checkpoint_event(epoch, ctx)
                final = ctx.run_dir / 'checkpoints' / 'netG_model_epoch_200_iter_0.pth'
                summary = {'final_or_selected_checkpoint_path': str(final),
                           'final_or_selected_checkpoint_sha256': sha256_file(final),
                           'checkpoint_selection_rule': config['checkpoint']['rule'],
                           'training_duration_seconds': time.monotonic() - t0,
                           'peak_vram_bytes': trainer.peak_step_gpu_memory_bytes,
                           'optimizer_applications': trainer.optimizer_applications,
                           'missing_field_reasons': {'seed_level_evaluation_metrics':
                                                     'Downstream evaluation is a later milestone'}}
                ctx.close(completion_status='resumed_completed' if resume_state else 'completed', summary=summary)
            finally:
                rio.untee(streams)
    return ctx.run_dir
