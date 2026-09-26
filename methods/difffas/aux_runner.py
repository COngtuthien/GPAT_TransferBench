"""M6D6e E07c auxiliary conditioning-encoder production training engine (CONTROLLED_ADAPTATION, DEV-021).

Executes pinned murphytju/DiffFAS models/pretrain_classifier.py under Amendment A3 (K7 head,
frozen TRAIN population, auxiliary seed 42, one run), A6 (unchanged source architecture) and
A7 (FP32, TF32 off, cuDNN deterministic without benchmark, no autocast/GradScaler), in the
source order:

    model = custom_rn.resnet18(); model.fc = Linear(512, 7); model = model.cuda()      :15-17
    transform = Resize((256,256)) -> ToTensor() -> Normalize([.5]*3, [.5]*3)          :18-23
    dataset (ImageFolder-equivalent frozen TRAIN K7 population)                         :25
    DataLoader(dataset, batch_size=256, shuffle=True, num_workers=6, drop_last=True)   :26
    SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-3)                  :27
    CrossEntropyLoss()                                                                  :28
    for epoch in range(200): for batch: zero_grad; _,_,_,out = model(x); CE; backward;
        step; running_loss += loss.item() * x.size(0); rus = running_loss / len(dataset)
        after the epoch: torch.save(model, path)  (same path overwritten)                :30-45

Two modes share this engine and nothing else:
  SCIENTIFIC     auxiliary seed 42, 200 epochs, <runtime_root>/runs/m6/E07c/aux_encoder/seed_42/,
                 launched only by tools/run_e07c_aux.py from a clean worktree. NOT launched in M6D6e/M6D6f.
  QUALIFICATION  seed 60605 (M6D6e) under <runtime_root>/qualification/m6d6e/E07c_aux/q60605-<run_id>/, or
                 seed 60606 (M6D6f) under <runtime_root>/qualification/m6d6f/E07c_aux/q60606-<run_id>-<label>/,
                 launched only by the qualification harnesses; never scientific.
Recovery (Amendment A8, M6D6f): ONE_LOGICAL_RUN, exact epoch-boundary continuation only. The
engineering sidecars, their verified loading and every state restoration live in
methods/difffas/aux_resume.py; this engine's step, epoch loss and whole-module save are unchanged.
Torch is imported by the caller.
"""
from contextlib import contextmanager
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
from methods.common.learned import PreparationError
from methods.common.learned_runlog import LearnedRunContext
from methods.common.runlog import (RunContext, RunDirectoryError, atomic_write_json, git_commit,
                                   load_logging_contract)
from methods.difffas import aux_runner_io as aio

LOCK_PATH = 'environments/e07c.lock.json'
DISCONNECTED = ('norm.bias', 'norm.weight')     # source-native BatchNorm1d(18), never called (M6D6b)
require = aio.require


class TrainingStop(RuntimeError):
    """A real-data gate failed (batch plan, finiteness, gradient coverage, OOM): STOP_AND_REPORT."""
    def __init__(self, message, stage=None):
        super().__init__(message)
        self.stage = stage


class AuxMemoryBlocked(TrainingStop):
    """CUDA OOM at exact B=256: BLOCKED_BY_AUX_B256_MEMORY. No fallback of any kind exists."""


def gate(value, message):
    if not value:
        raise TrainingStop('E07c aux STOP: ' + message)


def tsha(t):
    return aio.sha(t.detach().cpu().contiguous().numpy().tobytes())


# ================================================================= run context (run_logging_v1)
class E07cAuxRunContext(LearnedRunContext):
    """run_logging_v1 writers bound to the auxiliary-encoder root.

    The auxiliary run is not an experiment-seed run, so experiment_seed stays null (with a
    reason) in BOTH modes and the seed is recorded as auxiliary_encoder_training_seed (42) or
    qualification_seed (60605). The run id uses the run_logging_v1 formula with the method
    label E07c/aux_encoder. A fresh context refuses an existing root; a continuation context
    (resume=True, A8) requires the existing root and keeps its run_id / run_uuid / start_utc.
    """
    def __init__(self, *, mode, seed, runtime_root, environment, missing_environment_reasons, identities,
                 command_line=None, resume=False, label=None):
        aio.validate_mode_seed(mode, seed)
        config = load_method_config(aio.METHOD_ID)
        self.mode, self.identities = mode, dict(identities)
        require(self.identities['config_sha256'] == config['_runtime']['config_sha256'], 'config identity')
        self.contract = load_logging_contract()
        self.method_id, self.seed, self.aux_seed = aio.METHOD_ID, None, seed
        self.config, self.runtime_root, self.resume, self.label = config, Path(runtime_root), bool(resume), label
        self.source_commits = {config['source']['repository']: config['source']['pinned_commit']}
        self.command_line = command_line if command_line is not None else ' '.join(sys.argv)
        self.config_sha256 = config['_runtime']['config_sha256']
        self.config_path = config['_runtime']['config_path']
        self.git_commit = git_commit()
        self.run_id = aio.run_id(seed, self.config_sha256, self.git_commit)
        self.run_dir = aio.run_root(runtime_root, mode, seed, self.run_id, label)
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
    def seed_field(self):
        return 'auxiliary_encoder_training_seed' if self.mode == aio.SCIENTIFIC else 'qualification_seed'

    def open(self):
        if self._metrics_fh is not None or self._closed_summary is not None:
            raise RunDirectoryError('RunContext objects may be opened only once')
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fh = (self.run_dir.parent / f'.{self.run_dir.name}.lock').open('a')
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.run_dir.exists() and not self.resume:
                raise RunDirectoryError(f'STOP_AND_REPORT: auxiliary run root already exists: {self.run_dir} '
                                        '(never overwritten; A8 continuation requires --resume-state <EXACT_PATH>)')
            if self.resume and not (self.run_dir.is_dir() and self.path('run_manifest').is_file()):
                raise RunDirectoryError(f'STOP_AND_REPORT: continuation requires the existing run root: {self.run_dir}')
            return self._open_locked()
        except BaseException:
            for handle in (self._metrics_fh, self._generation_fh, self._lock_fh):
                if handle is not None:
                    handle.close()
            self._metrics_fh = self._generation_fh = self._lock_fh = None
            raise

    def _open_locked(self):
        result = RunContext._open_locked(self)
        index = json.loads(self.path('checkpoint_index').read_text())
        index.update(note=('Every whole-module epoch-boundary save actually written (pretrain_classifier.py:45). The '
                           'same path is overwritten after every epoch; only the epoch-200 bytes are the final state.'),
                     experiment_seed=None, runner_mode=self.mode, **{self.seed_field: self.aux_seed},
                     checkpoint_rule=aio.CHECKPOINT_RULE, authoritative_for_main_difffas=False,
                     authoritative_reason=('the SHA256 is frozen only by a later owner milestone after epoch 200; '
                                           'main DiffFAS loads only through aux_checkpoint.load_frozen_aux_encoder'))
        if self.mode == aio.QUALIFICATION:
            index.update(labels=list(aio.QUALIFICATION_LABELS))
        atomic_write_json(self.path('checkpoint_index'), index)
        return result

    def _resolved_config(self):
        resolved = RunContext._resolved_config(self)
        ids = self.identities
        resolved['_resolved'].update({
            'experiment_seed': None, self.seed_field: self.aux_seed, 'role': 'AUXILIARY_CONDITIONING_ENCODER',
            'runner_mode': self.mode, 'scientific_run': self.mode == aio.SCIENTIFIC,
            'qualification_only': self.mode == aio.QUALIFICATION,
            'labels': list(aio.QUALIFICATION_LABELS) if self.mode == aio.QUALIFICATION else [],
            'note': 'Verbatim copy of the frozen E07c config above; the auxiliary training contract is '
                    'conditioning_encoder.training_contract. Nothing was overridden.',
            'identities': ids,
            'bound_overlays': [
                {'path': 'configs/amendments/e07c_a6_feature_interface_source_correction.yaml',
                 'sha256': ids['a6_overlay_sha256']},
                {'path': 'configs/amendments/e07c_a7_execution_policy.yaml', 'sha256': ids['a7_overlay_sha256']},
                {'path': aio.CONTRACT_PATH, 'sha256': ids['m6d6e_contract_sha256'], 'mode': 'PRODUCTION_RUNNER_CONTRACT'}]
            + ([{'path': 'configs/amendments/e07c_a8_aux_resume_policy.yaml', 'sha256': ids['a8_overlay_sha256'],
                 'mode': 'A8_EXACT_EPOCH_BOUNDARY_RESUME_POLICY'}] if 'a8_overlay_sha256' in ids else []),
            'population': {'split_manifest': aio.SPLIT_MANIFEST, 'sha256': ids['split_manifest_sha256'],
                           'class_map': aio.CLASS_MAP, 'class_map_sha256': ids['class_map_sha256'],
                           'rows': aio.TRAIN_ROWS, 'split': 'TRAIN', 'classes': list(aio.CLASSES),
                           'order': 'IMAGEFOLDER_EQUIVALENT (class index, sample_id)'},
            'training': {'epochs': aio.EPOCHS,
                         'epochs_executed_limit': aio.EPOCHS if self.mode == aio.SCIENTIFIC else 1,
                         'batch_size': aio.BATCH_SIZE, 'drop_last': aio.DROP_LAST, 'shuffle': aio.SHUFFLE,
                         'num_workers': aio.WORKERS, 'steps_per_epoch': aio.STEPS_PER_EPOCH,
                         'consumed_per_epoch': aio.CONSUMED_PER_EPOCH, 'dropped_per_epoch': aio.DROPPED_PER_EPOCH,
                         'epoch_loss_denominator': aio.EPOCH_LOSS_DENOMINATOR},
            'optimizer_contract': 'torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.005); no scheduler',
            'criterion': 'torch.nn.CrossEntropyLoss() on the fourth forward output',
            'precision': 'A7: FP32, no autocast, no GradScaler, TF32 off, cudnn.benchmark=False, cudnn.deterministic=True',
            'checkpoint': {'path': 'checkpoints/' + aio.CHECKPOINT_NAME, 'rule': aio.CHECKPOINT_RULE,
                           'cadence': 'after every epoch, same path overwritten',
                           'format': 'torch.save of the WHOLE nn.Module'},
            'resume': ('A8: ONE_LOGICAL_RUN; EXACT_EPOCH_BOUNDARY_ONLY from the one committed engineering sidecar '
                       'authorized by the A8 index (methods/difffas/aux_resume.py); never the scientific checkpoint'),
            'logging_contract': {'version': 'run_logging_v1', 'sha256': ids['logging_contract_sha256'],
                                 'record_per': 'OPTIMIZER_STEP'},
            'run_id_rule': "run_logging_v1 formula with method label 'E07c/aux_encoder'"})
        return resolved

    @property
    def resolved_config_sha256(self):
        p = self.path('resolved_config')
        return sha256_file(p) if p.is_file() else None

    def _manifest(self, **kwargs):
        m = LearnedRunContext._manifest(self, **kwargs)
        m.update(experiment_seed=None, runner_mode=self.mode, role='AUXILIARY_CONDITIONING_ENCODER',
                 run_label=aio.RUN_LABEL, scientific_run=self.mode == aio.SCIENTIFIC,
                 qualification_only=self.mode == aio.QUALIFICATION, **{self.seed_field: self.aux_seed},
                 resolved_config_path='resolved_config.yaml', resolved_config_sha256=self.resolved_config_sha256,
                 fidelity_class='CONTROLLED_ADAPTATION', deviation='DEV-021', identities=self.identities,
                 environment_lock_sha256=self.identities['environment_lock_sha256'],
                 m6d6e_file_sha256=aio.m6d6e_file_hashes(), val_split_accessed=False,
                 resume_policy='A8_ONE_LOGICAL_RUN_EXACT_EPOCH_BOUNDARY', continuation_process=self.resume)
        m['missing_field_reasons']['experiment_seed'] = (
            'AUXILIARY_ENCODER run: seeded by auxiliary_encoder_training_seed (A3 5.4b), not by an experiment seed'
            if self.mode == aio.SCIENTIFIC else f'QUALIFICATION_ONLY: seed {self.aux_seed} is not an experiment seed')
        if self.mode == aio.QUALIFICATION:
            m['labels'] = list(aio.QUALIFICATION_LABELS)
        return m

    def log_event(self, event, payload=None):
        self._append({'record_type': 'event', 'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'method_id': self.method_id, 'experiment_seed': None, self.seed_field: self.aux_seed,
                      'payload': payload or {}})

    def record_checkpoint(self, *, path, epoch, global_step, file_size_bytes, sha256, checkpoint_type,
                          selected_for_final, selection_reason):
        final = self.mode == aio.SCIENTIFIC and epoch == aio.EPOCHS
        require(type(epoch) is int and 1 <= epoch <= aio.EPOCHS and path == 'checkpoints/' + aio.CHECKPOINT_NAME,
                'auxiliary checkpoint epoch/path')
        require((checkpoint_type, selected_for_final) == (('selected', True) if final else ('periodic', False)),
                'only the scientific epoch-200 bytes are the final state')
        require(type(global_step) is int and global_step == epoch * aio.STEPS_PER_EPOCH and
                type(file_size_bytes) is int and file_size_bytes > 0 and len(sha256) == 64, 'checkpoint metadata')
        return RunContext.record_checkpoint(self, path=path, epoch=epoch, global_step=global_step,
                                            file_size_bytes=file_size_bytes, sha256=sha256,
                                            checkpoint_type=checkpoint_type, selected_for_final=selected_for_final,
                                            selection_reason=selection_reason)

    def close(self, *, completion_status='completed', summary=None, failure_reason=None):
        supplied = dict(summary or {})
        require(not supplied.get('test_split_accessed'), 'TEST is prohibited')
        reasons = dict(supplied.get('missing_field_reasons', {}))
        reasons.setdefault('experiment_seed', 'AUXILIARY_ENCODER run: no experiment seed (A3 5.4b)')
        for key in ('final_or_selected_checkpoint_path', 'final_or_selected_checkpoint_sha256',
                    'seed_level_evaluation_metrics', 'training_duration_seconds', 'peak_vram_bytes',
                    'failure_reason', 'output_manifest_path'):
            if supplied.get(key) is None:
                reasons.setdefault(key, 'Not supplied by caller; no execution/evaluation result inferred')
        supplied.update(missing_field_reasons=reasons, checkpoint_selection_rule=aio.CHECKPOINT_RULE,
                        runner_mode=self.mode, **{self.seed_field: self.aux_seed})
        supplied.setdefault('peak_vram_reason', reasons.get('peak_vram_bytes'))
        self._validate_metadata(supplied)
        return RunContext.close(self, completion_status=completion_status, summary=supplied,
                                failure_reason=failure_reason)


# ================================================================= runtime setup (A7 + seed)
def configure_precision(torch, config):
    """A7 in THIS process: the committed execution_policy seam, plus the frozen lock launch environment."""
    from methods.difffas import execution_policy as ep
    for key, value in aio.LAUNCH_ENVIRONMENT.items():
        require(os.environ.get(key) == value, f'launch with {key}={value} (environments/e07c.lock.json)')
    state = ep.apply_e07c_precision_policy(config)
    require(not torch.is_autocast_enabled('cuda'), 'autocast disabled')
    return dict(state, grad_scaler=False, amp=False, activation_checkpointing=False, api=(
        'methods/difffas/execution_policy.py::apply_e07c_precision_policy'))


def seed_process(torch, mode, seed, config):
    """SCIENTIFIC: seed_adapter.apply_seed(42, cuda=True, auxiliary=True). QUALIFICATION: the same calls, seed 60605.

    Called before model construction (the source constructs the model first, so every CPU
    draw from here on is the source's). No DataLoader generator: shuffle draws its seed from
    this process-seeded CPU generator, exactly as the unseeded source loader would.
    """
    from methods.difffas import execution_policy as ep
    from methods.difffas import seed_adapter
    aio.validate_mode_seed(mode, seed)
    require(os.environ.get('PYTHONHASHSEED') == str(seed), f'launch with PYTHONHASHSEED={seed}')
    if mode == aio.SCIENTIFIC:
        plan = seed_adapter.apply_seed(seed, cuda=True, auxiliary=True, config=config)
        out = {'seed': seed, 'role': 'AUXILIARY_ENCODER_TRAINING_SEED', 'plan_scope': plan['scope'],
               'hook': 'methods/difffas/seed_adapter.py::apply_seed(auxiliary=True)'}
    else:
        # identical framework calls to seed_adapter.apply_seed -> learned.apply_framework_seed(cuda=True)
        # followed by torch.cuda.manual_seed; the scientific API is NOT called with 60605.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        out = {'seed': seed, 'role': 'QUALIFICATION_SEED_NOT_AUXILIARY_SEED',
               'hook': 'aux_runner.seed_process (same calls as seed_adapter.apply_seed; scientific API not called)'}
    require(ep.precision_state() == ep.EXPECTED_STATE, 'A7 state intact after seeding')
    out.update(calls=['random.seed', 'numpy.random.seed', 'torch.manual_seed', 'cudnn.benchmark=False',
                      'cudnn.deterministic=True', 'torch.cuda.manual_seed_all', 'torch.cuda.manual_seed'],
               pythonhashseed=os.environ.get('PYTHONHASHSEED'),
               torch_cpu_rng_sha256=aio.sha(torch.get_rng_state().numpy().tobytes()),
               torch_cuda_rng_sha256=aio.sha(torch.cuda.get_rng_state().numpy().tobytes()))
    return out


def environment_record(torch):
    env = {'host': socket.gethostname(), 'user': getpass.getuser(), 'platform': platform.platform(),
           'python_version': platform.python_version(), 'numpy_version': np.__version__,
           'gpu_model': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(),
           'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
           'framework': 'torch', 'framework_version': torch.__version__, 'pytorch_version': torch.__version__,
           'tensorflow_version': None, 'dependency_fingerprint': sha256_file(ROOT / LOCK_PATH),
           'environment_lock_path': LOCK_PATH, 'environment_name': 'gpat-m6-e07c', 'python_executable': sys.executable,
           'gpu_note': 'single RTX 3090; exact B=256 FP32 (A7); no microbatch/accumulation/AMP'}
    return env, {'tensorflow_version': 'E07c is a PyTorch method; TensorFlow is not part of gpat-m6-e07c'}


# ================================================================= components (pretrain_classifier.py:15-28)
def build_transform(transforms):
    """pretrain_classifier.py:18-23, verbatim."""
    return transforms.Compose([transforms.Resize((256, 256)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def transform_evidence(transforms, transform):
    t = transform.transforms
    require([type(x) for x in t] == [transforms.Resize, transforms.ToTensor, transforms.Normalize] and
            list(t[0].size) == [256, 256] and list(t[2].mean) == [0.5] * 3 and list(t[2].std) == [0.5] * 3 and
            not t[2].inplace, 'transform is exactly Resize((256,256)) -> ToTensor -> Normalize(0.5, 0.5)')
    return {'steps': [repr(x) for x in t], 'augmentation': 'NONE', 'random_transforms': 0,
            'resize_interpolation': str(t[0].interpolation), 'resize_antialias': t[0].antialias}


def build_loader(torch, dataset):
    """pretrain_classifier.py:26: no generator, no worker_init_fn, no pin_memory, no sampler override."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=aio.BATCH_SIZE, shuffle=aio.SHUFFLE,
                                         num_workers=aio.WORKERS, drop_last=aio.DROP_LAST)
    return loader, verify_loader(torch, loader, len(dataset))


def verify_loader(torch, loader, n):
    s, b = loader.sampler, loader.batch_sampler
    require((loader.batch_size, loader.drop_last, loader.num_workers, loader.pin_memory, loader.persistent_workers,
             loader.prefetch_factor, loader.generator, loader.worker_init_fn, loader.timeout) ==
            (256, True, 6, False, False, 2, None, None, 0), 'DataLoader matches pretrain_classifier.py:26 defaults')
    require(type(s) is torch.utils.data.RandomSampler and s.replacement is False and s.generator is None and
            s.num_samples == n, 'RandomSampler without replacement on the process CPU generator (no explicit generator)')
    require(type(b) is torch.utils.data.BatchSampler and b.batch_size == 256 and b.drop_last is True,
            'BatchSampler(256, drop_last=True)')
    require(len(loader) == aio.STEPS_PER_EPOCH and n == aio.TRAIN_ROWS, '56 batches over 14467 rows')
    return {'batch_size': 256, 'shuffle': True, 'num_workers': 6, 'drop_last': True, 'pin_memory': False,
            'persistent_workers': False, 'prefetch_factor': 2, 'generator': None, 'worker_init_fn': None,
            'sampler': 'RandomSampler(replacement=False, generator=None)',
            'batch_sampler': 'BatchSampler(256, drop_last=True)', 'len_batches': len(loader), 'rows': n,
            'consumed_per_epoch': aio.CONSUMED_PER_EPOCH, 'dropped_per_epoch': aio.DROPPED_PER_EPOCH,
            'weighted_sampling': False, 'class_balancing': False, 'replacement': False}


def build_optimizer(torch, model):
    """pretrain_classifier.py:27."""
    optimizer = torch.optim.SGD(model.parameters(), lr=aio.LR, momentum=aio.MOMENTUM, weight_decay=aio.WEIGHT_DECAY)
    groups = optimizer.param_groups
    params = list(model.parameters())
    g = groups[0]
    require(type(optimizer) is torch.optim.SGD and len(groups) == 1 and len(g['params']) == len(params) and
            all(a is b for a, b in zip(g['params'], params)), 'one SGD group over model.parameters()')
    require((g['lr'], g['momentum'], g['weight_decay'], g['dampening'], g['nesterov'], g['maximize']) ==
            (0.002, 0.9, 0.005, 0, False, False), 'SGD lr=0.002 momentum=0.9 weight_decay=5e-3')
    return optimizer, {'class': 'torch.optim.SGD', 'lr': g['lr'], 'momentum': g['momentum'],
                       'weight_decay': g['weight_decay'], 'dampening': g['dampening'], 'nesterov': g['nesterov'],
                       'param_groups': 1, 'parameter_tensors': len(params),
                       'parameters': sum(p.numel() for p in params), 'scheduler': 'NONE'}


def build_criterion(torch):
    """pretrain_classifier.py:28."""
    c = torch.nn.CrossEntropyLoss()
    require((c.reduction, c.weight, c.ignore_index, c.label_smoothing) == ('mean', None, -100, 0.0),
            'CrossEntropyLoss() defaults')
    return c, {'class': 'torch.nn.CrossEntropyLoss', 'reduction': 'mean', 'weight': None, 'label_smoothing': 0.0}


@contextmanager
def production_components(torch, transforms, config, records, sample_datasets, faces_root, mode, seed):
    """Source order :15-28 on the ALREADY seeded process. custom_rn stays imported for the whole-module save."""
    from methods.difffas.encoder import encoder_model
    aio.validate_mode_seed(mode, seed)
    with encoder_model(config) as model:            # :15-16 custom_rn.resnet18(); fc = Linear(512, 7)
        model = model.cuda()                         # :17
        require(model.training and next(model.parameters()).dtype == torch.float32, 'FP32 module in training mode')
        transform = build_transform(transforms)      # :18-23
        reader = aio.CanonicalFaceReader(faces_root, sample_datasets)
        dataset = aio.AuxTrainDataset(records, reader, transform)    # :25
        loader, loader_ev = build_loader(torch, dataset)            # :26
        optimizer, opt_ev = build_optimizer(torch, model)           # :27
        criterion, crit_ev = build_criterion(torch)                 # :28
        trainer = Trainer(torch, model, optimizer, criterion, loader, dataset, mode, seed)
        trainer.evidence = {'transform': transform_evidence(transforms, transform), 'loader': loader_ev,
                            'optimizer': opt_ev, 'criterion': crit_ev,
                            'model': {'class': f'{type(model).__module__}.{type(model).__name__}',
                                      'fc': [model.fc.in_features, model.fc.out_features],
                                      'parameters': sum(p.numel() for p in model.parameters()),
                                      'parameter_tensors': len(list(model.parameters())),
                                      'device': str(next(model.parameters()).device), 'training': model.training,
                                      'disconnected_parameters': list(DISCONNECTED)}}
        yield trainer


# ================================================================= trainer
class Trainer:
    """One auxiliary training process: model, SGD, CE, loader and the source step."""

    def __init__(self, torch, model, optimizer, criterion, loader, dataset, mode, seed):
        self.torch, self.model, self.optimizer, self.criterion = torch, model, optimizer, criterion
        self.loader, self.dataset, self.mode, self.seed = loader, dataset, mode, seed
        self.records = dataset.records
        self.named = list(model.named_parameters())
        self.completed_epoch = self.global_step = 0
        self.last_epoch_order = ()
        self.optimizer_applications = self.backward_calls = 0
        self.peak_step_gpu_memory_bytes = 0
        self.evidence = {}
        optimizer.register_step_post_hook(lambda *a: setattr(self, 'optimizer_applications',
                                                             self.optimizer_applications + 1))

    def gradient_gate(self):
        torch = self.torch
        missing = sorted(n for n, p in self.named if p.grad is None)
        gate(missing == sorted(DISCONNECTED), 'gradient coverage: only norm.weight/norm.bias lack a gradient; got '
             + repr(missing))
        grads = [p.grad for n, p in self.named if p.grad is not None]
        gate(bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()), 'finite gradients')
        return {'with_grad': len(grads), 'without_grad': missing, 'finite': True}

    def step(self, batch, epoch, iteration, on_stage=None):
        """pretrain_classifier.py:34-43 for one batch; `on_stage` only observes (memory / hashes)."""
        torch = self.torch
        inputs_cpu, labels_cpu, index = batch
        indices = index.tolist()
        B = len(indices)
        gate(B == aio.BATCH_SIZE and len(set(indices)) == B, f'exact batch 256 of unique TRAIN rows (got {B})')
        gate(tuple(inputs_cpu.shape) == (B, 3, 256, 256) and inputs_cpu.dtype == torch.float32,
             'inputs FP32 [256,3,256,256]')
        gate(tuple(labels_cpu.shape) == (B,) and labels_cpu.dtype == torch.int64, 'int64 labels [256]')
        gate(all(self.records[i]['label'] == int(y) for i, y in zip(indices, labels_cpu.tolist())), 'labels = population')
        observe = on_stage or (lambda stage: None)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        stage = 'host_to_device'
        before_apps = self.optimizer_applications
        try:
            inputs, labels = inputs_cpu.cuda(), labels_cpu.cuda()       # :34
            observe('after_transfer')
            stage = 'zero_grad'
            self.optimizer.zero_grad()                                  # :36
            stage = 'forward'
            _, _, _, outputs = self.model(inputs)                       # :37
            observe('after_forward')
            stage = 'loss'
            loss = self.criterion(outputs, labels)                      # :38-39
            stage = 'backward'
            loss.backward()                                             # :40
            self.backward_calls += 1
            observe('after_backward')
            stage = 'gradient_gate'
            grad = self.gradient_gate()
            stage = 'optimizer_step'
            self.optimizer.step()                                       # :41
            observe('after_optimizer_step')
            stage = 'loss_item'
            value = loss.item()                                         # :42
        except torch.OutOfMemoryError as error:
            raise AuxMemoryBlocked(f'CUDA OOM at stage {stage}, epoch {epoch} iteration {iteration} (B={B}): {error}',
                                   stage) from error
        except RuntimeError as error:
            if 'out of memory' in str(error).lower():
                raise AuxMemoryBlocked(f'CUDA OOM at stage {stage} (B={B}): {error}', stage) from error
            raise
        gate(self.optimizer_applications == before_apps + 1, 'exactly one optimizer.step() per batch')
        gate(aio.finite([value]), 'finite loss')
        gate(bool(torch.stack([torch.isfinite(p).all() for _, p in self.named]).all()), 'finite parameters after step')
        self.global_step += 1
        mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        self.peak_step_gpu_memory_bytes = max(self.peak_step_gpu_memory_bytes, mem or 0)
        ids = [self.records[i]['sample_id'] for i in indices]
        hist = [0] * aio.K
        for y in labels_cpu.tolist():
            hist[y] += 1
        return {'epoch': epoch, 'global_step': self.global_step, 'iteration': iteration, 'batch_size': B,
                'indices': indices, 'sample_ids': ids, 'batch_sample_sha256': aio.order_sha256(ids),
                'class_histogram': hist, 'loss': value, 'loss_hex': float(value).hex(),
                'step_seconds': time.monotonic() - t0, 'gpu_memory_bytes': mem, 'gradients': grad,
                'learning_rate': self.optimizer.param_groups[0]['lr']}

    def run_epoch(self, epoch, ctx=None, t_start=None, on_step=None):
        """One pass: exactly 56 x 256 (drop_last), source running loss / len(dataset)."""
        t_start = time.monotonic() if t_start is None else t_start
        print('Epoch: {}/{}'.format(epoch, aio.EPOCHS))                  # :31
        running_loss = 0                                                  # :32
        sizes, order, losses, peak = [], [], [], 0
        for i, data in enumerate(self.loader):                            # :33
            r = self.step(data, epoch, i)
            running_loss += r['loss'] * r['batch_size']                   # :42 (inputs.size(0) == 256)
            rus = running_loss / len(self.dataset)                        # :43
            sizes.append(r['batch_size'])
            order.extend(r['sample_ids'])
            losses.append(r['loss'])
            peak = max(peak, r['gpu_memory_bytes'] or 0)
            if on_step is not None:
                on_step(r)
            if ctx is not None:
                ctx.log_epoch(aio.step_record(
                    mode=self.mode, seed=self.seed, epoch=epoch, global_step=r['global_step'], iteration=i,
                    learning_rate=r['learning_rate'], loss=r['loss'], running_loss=running_loss,
                    batch_size=r['batch_size'], batch_sample_sha256=r['batch_sample_sha256'],
                    batch_class_histogram=r['class_histogram'], wall_clock_seconds=time.monotonic() - t_start,
                    gpu_memory_bytes=r['gpu_memory_bytes'],
                    extra={'step_seconds': r['step_seconds'], 'loss_hex': r['loss_hex'],
                           'optimizer_applications_total': self.optimizer_applications}))
        print('\t Training: Loss: {:.4f}'.format(rus))                   # :44
        gate(sizes == [aio.BATCH_SIZE] * aio.STEPS_PER_EPOCH, f'epoch batch plan {len(sizes)} x 256 != 56 x 256')
        population = {r['sample_id'] for r in self.records}
        consumed = set(order)
        gate(len(order) == len(consumed) == aio.CONSUMED_PER_EPOCH and consumed <= population,
             '14336 unique TRAIN rows consumed')
        dropped = sorted(population - consumed)
        gate(len(dropped) == aio.DROPPED_PER_EPOCH, '131 TRAIN rows dropped by drop_last')
        loss = aio.epoch_loss(losses)
        gate(loss['running_loss'] == running_loss and loss['source_epoch_loss'] == rus, 'source epoch loss')
        self.completed_epoch, self.last_epoch_order = epoch, tuple(order)
        dropped_hist = [0] * aio.K
        label = {r['sample_id']: r['label'] for r in self.records}
        for sid in dropped:
            dropped_hist[label[sid]] += 1
        summary = {'epoch': epoch, 'optimizer_steps': len(sizes), 'batch_sizes_distinct': sorted(set(sizes)),
                   'consumed_examples': len(order), 'unique_consumed': len(consumed),
                   'dropped_examples': len(dropped), 'dropped_sample_ids_sha256': aio.order_sha256(dropped),
                   'dropped_class_histogram': dropped_hist,
                   'epoch_sample_order_sha256': aio.order_sha256(order), 'global_step_end': self.global_step,
                   'running_loss': running_loss, 'source_epoch_loss': rus, 'source_epoch_loss_hex': float(rus).hex(),
                   'epoch_loss_denominator': len(self.dataset),
                   'epoch_loss_denominator_semantics': 'len(train_dataset) = 14467 (source; NOT the 14336 consumed)',
                   'peak_step_gpu_memory_bytes': peak}
        if ctx is not None:
            ctx.log_event('e07c_aux_epoch_complete', summary)
        return summary, losses

    def checkpoint_event(self, epoch, ctx, config):
        """pretrain_classifier.py:45 torch.save(model, path) through the M6D6c seam; same path, atomic replace."""
        from methods.difffas.aux_checkpoint import save_whole_module
        require(epoch == self.completed_epoch, 'checkpoint only at a completed epoch boundary')
        final = aio.checkpoint_path(ctx.run_dir)
        partial = final.with_name(final.name + '.partial')
        require(not partial.exists(), 'no stale partial checkpoint')
        written = save_whole_module(self.model, partial, config)
        os.replace(partial, final)
        digest, size = sha256_file(final), final.stat().st_size
        require(digest == written['sha256'] and size == written['size_bytes'], 'replaced bytes equal the written bytes')
        is_final = ctx.mode == aio.SCIENTIFIC and epoch == aio.EPOCHS
        if is_final:
            kind, selected, reason = 'selected', True, aio.CHECKPOINT_RULE + ' (A3 5.4 checkpoint_rule)'
        elif ctx.mode == aio.QUALIFICATION:
            kind, selected, reason = 'periodic', False, ('QUALIFICATION_ONLY: epoch-boundary whole-module save; '
                                                         'NOT_A_SCIENTIFIC_CHECKPOINT; never consumable')
        else:
            kind, selected, reason = 'periodic', False, ('epoch-boundary overwrite (pretrain_classifier.py:45); NOT '
                                                         'authoritative before epoch 200; superseded by the next epoch')
        meta = dict(path='checkpoints/' + aio.CHECKPOINT_NAME, epoch=epoch, global_step=self.global_step,
                    file_size_bytes=size, sha256=digest, checkpoint_type=kind, selected_for_final=selected,
                    selection_reason=reason)
        ctx.record_checkpoint(**meta)
        ctx.log_event('e07c_aux_checkpoint_event', dict(meta, serialization_api=written['serialization_api'],
                                                        format=written['format'], identity=written['identity'],
                                                        authoritative_for_main_difffas=False))
        return dict(meta, identity=written['identity'], serialization_api=written['serialization_api'],
                    format=written['format'])


# ================================================================= scientific entry (tools/run_e07c_aux.py)
def run_scientific(*, runtime_root, faces_root, storage, command_line=None, resume_sidecar=None):
    """The ONE logical 200-epoch SCIENTIFIC auxiliary run (seed 42). Never called in M6D6e / M6D6f.

    Fresh (resume_sidecar=None): the seed_42 root must not exist; the epoch-0 sidecar is committed
    before the first iterator. Continuation (A8): resume_sidecar is the exact committed sidecar
    path; the process seeds and constructs exactly as a fresh run, then aux_resume verifies
    (SHA256 before torch.load, weights_only=True), reconciles append-only and restores, and the
    loop continues at completed_epoch + 1. Same run_id / run_uuid / root; one scientific run.
    """
    import torch
    import torchvision.transforms as transforms
    from methods.common.runlog import git_dirty
    from methods.difffas import DiffFASAdapter
    from methods.difffas import aux_resume as ar
    mode, seed = aio.SCIENTIFIC, aio.SCIENTIFIC_SEED
    require(not git_dirty(), 'SCIENTIFIC runs require a clean git worktree')
    contract = aio.load_contract()
    ids = dict(aio.identities(contract), **ar.a8_identities())
    adapter = DiffFASAdapter()
    config = adapter.config
    source = adapter.validate_source()
    require((source['commit'], source['tree']) == (ids['source_commit'], ids['source_tree']), 'source identity')
    precision = configure_precision(torch, config)
    records, datasets, population = aio.read_train_population()
    env, reasons = environment_record(torch)
    continuing = resume_sidecar is not None
    ctx = E07cAuxRunContext(mode=mode, seed=seed, runtime_root=runtime_root, environment=env,
                            missing_environment_reasons=reasons, identities=ids, command_line=command_line,
                            resume=continuing)
    require(ctx.run_dir.exists() == continuing,
            'a fresh run requires an absent seed_42 root; a continuation requires the existing root')
    seeding = seed_process(torch, mode, seed, config)                        # before model construction
    with production_components(torch, transforms, config, records, datasets, faces_root, mode, seed) as trainer:
        run = ar.LogicalRun(torch, trainer, ctx, config, storage=storage, precision=precision)
        if continuing:
            run.verify_before_open(resume_sidecar)                          # read-only; SHA256 before torch.load
        with ctx:
            streams = aio.tee_run_logs(ctx.run_dir)
            try:
                if continuing:
                    run.begin_resume()                                       # restore; nothing consumes RNG after
                else:
                    ctx.log_event('e07c_aux_run_start', {'seeding': seeding, 'precision': precision,
                                                         'population': population, 'components': trainer.evidence,
                                                         'resume_policy': 'A8'})
                    run.begin_fresh()                                        # epoch-0 boundary, before iter(loader)
                t0 = time.monotonic()
                for epoch in range(trainer.completed_epoch + 1, aio.EPOCHS + 1):   # :30
                    trainer.run_epoch(epoch, ctx, t0)
                    run.commit_boundary(epoch)                               # :45, then the engineering sidecar
                require(trainer.completed_epoch == aio.EPOCHS and trainer.global_step == aio.EPOCHS * aio.STEPS_PER_EPOCH,
                        'epoch-200 final state')
                final = aio.checkpoint_path(ctx.run_dir)
                digest = sha256_file(final)
                committed = ar.read_index(ctx.run_dir)['committed']
                require(committed['completed_epoch'] == aio.EPOCHS and
                        committed['scientific_checkpoint_file']['sha256'] == digest, 'final bytes = committed epoch 200')
                steps = run.summary()
                require(steps['logical_authoritative_optimizer_steps'] == aio.EPOCHS * aio.STEPS_PER_EPOCH,
                        '11200 logical optimizer steps')
                ctx.close(completion_status='resumed_completed' if steps['process_sessions'] > 1 else 'completed',
                          summary={
                    'final_or_selected_checkpoint_path': str(final), 'final_or_selected_checkpoint_sha256': digest,
                    'final_checkpoint_bytes': final.stat().st_size,
                    'final_checkpoint_status': 'SHA256_RECORDED_PENDING_OWNER_FREEZE',
                    'authoritative_for_main_difffas': False,
                    'consumption_rule': 'main DiffFAS may load it only after a later milestone freezes this SHA256 and '
                                        'only through aux_checkpoint.load_frozen_aux_encoder',
                    'training_duration_seconds': time.monotonic() - t0,
                    'training_duration_semantics': 'this process only',
                    'peak_vram_bytes': trainer.peak_step_gpu_memory_bytes,
                    'optimizer_applications': trainer.optimizer_applications,
                    'step_accounting': steps, 'scientific_auxiliary_runs': 1, 'logical_run': 'ONE_LOGICAL_RUN',
                    'missing_field_reasons': {'seed_level_evaluation_metrics':
                                              'Auxiliary encoder: no evaluation metric exists in the pinned source'}})
            finally:
                aio.untee(streams)
    return ctx.run_dir
