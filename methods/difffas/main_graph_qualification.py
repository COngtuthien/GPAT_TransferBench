"""M6D6i E07c MAIN DiffFAS encoder-load integration + bounded training-graph qualification. QUALIFICATION ONLY.

Two modes, run on the GPU host under gpat-m6-e07c with qualification_seed 60607:

  --mode process   ONE fresh qualification process: A7 source order with the REAL owner-frozen auxiliary encoder
                   loaded ONLY through methods.difffas.execution_policy.main_runner_encoder, a synthetic in-memory
                   N=8 dataset, exact B=4, and exactly ONE source-order iteration of methods/difffas/main_graph.py
                   (loss -> zero_grad -> backward -> scheduler.step -> optimizer.step -> EMA accumulate).
  --mode launch    parent launcher: GPU pre-launch gates, frozen-checkpoint stat/SHA256 before process 1 and after
                   process 2, spawns process 1 and process 2 as independent fresh OS processes, compares their
                   pre-backward observations (must be equal) and characterizes post-backward differences from
                   tensors streamed over an anonymous pipe and held only in memory (never written to disk).

Nothing here reads TRAIN/VAL/TEST, a manifest or an image file; nothing calls torch.save; no main-model checkpoint,
no resume, no visualization/sampling branch, no run_logging_v1 run root and no scientific seed. The two qualification
optimizer steps are QUALIFICATION TRAINING-GRAPH EXECUTION, not scientific training, and their state is discarded
with the processes.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.difffas import runtime_qualification as rq  # noqa: E402  (M6D6a helpers, unchanged)
from methods.difffas import aux_checkpoint as seam  # noqa: E402  (constants + frozen path only; no loader call)
from methods.difffas import main_graph as mg  # noqa: E402

MILESTONE = 'M6D6i'
SEED = 60607  # qualification_seed only; never the auxiliary (42) or an experiment seed (42/1337/2026)
EXPERIMENT_SEED_REASON = 'QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN'
LABEL = 'MAIN_ENCODER_LOAD_AND_TRAINING_GRAPH_QUALIFICATION_ONLY'
CLASSIFICATION = 'M6D6I_E07C_MAIN_ENCODER_LOAD_AND_TRAINING_GRAPH'
AUTHORITY = '84023aed3959d7ffda380b7612b860652579b443'
BUILD_PARTS = ('builds', 'e07c_difffas', 'm6d6i')
BATCH = 4
N_ITEMS = 8
K = 7
ENCODER_PARAMETERS = 46233707
MAIN_PARAMETERS = 159363974
ENCODER_SHAPES = {'x32x32': [BATCH, 256, 32, 32], 'x16x16': [BATCH, 512, 16, 16], 'x8x8': [BATCH, 512, 8, 8],
                  'embg': [BATCH, K]}
MAIN_OUTPUT_SHAPE = [BATCH, 6, 256, 256]
ROLES = ('content', 'style_spoof', 'GT')  # FAS_dataset.py returns {'content', 'style_spoof', 'GT'}
ROLE_PHASE = {'content': 1, 'GT': 2, 'style_spoof': 3}
M6D6C_EVIDENCE = 'outputs/audit/M6D6C_E07C_AUX_CHECKPOINT_COMPATIBILITY.json'
GRAD_REGIONS = ('time_embed', 'input_blocks', 'middle_block', 'output_blocks', 'out')
# pinned get_model_conf: the final self.out convolution is zero-initialised (M6D6a), so at the FIRST step every
# gradient upstream of it is exactly 0; only the out projection receives nonzero values (source-native).
NONZERO_GRAD_REGIONS = ('out',)
SEAM_EVENTS = ('encoder_helper_call', 'resnet18_call', 'ResNet_init_call', 'resnet18_return', 'encoder_helper_return',
               'load_frozen_aux_encoder_call', 'load_freeze_record_call', 'verify_future_checkpoint_call',
               'checkpoint_open', 'load_verified_whole_module_call', 'read_verified_call', 'sha256_verified',
               'sha256_rejected', 'torch_load', 'find_class', 'encoder_identity_call', 'encoder_eval')
EXPECTED_SEAM = ['encoder_helper_call', 'resnet18_call', 'ResNet_init_call', 'resnet18_return', 'encoder_helper_return',
                 'load_frozen_aux_encoder_call', 'load_freeze_record_call', 'verify_future_checkpoint_call',
                 'checkpoint_open', 'load_verified_whole_module_call', 'read_verified_call', 'checkpoint_open',
                 'sha256_verified', 'torch_load', 'find_class', 'encoder_identity_call', 'encoder_eval']
EXPECTED_ORDER = ['seed', 'precision_policy', 'transform', 'dataset', 'dataloader', 'model_conf', 'model_conf',
                  'optimizer_init', 'scheduler_init', 'create_gaussian_diffusion', 'fresh_run_asserted',
                  'encoder_helper_call', 'encoder_eval', 'epoch_loop', 'tqdm', 'dataloader_iter', 'batch', 'randint',
                  'training_losses_enter', 'encoder_forward', 'model_forward', 'training_losses_exit', 'zero_grad',
                  'backward', 'scheduler_step', 'optimizer_step', 'accumulate']
PROFILED = {('consume_upstream_encoder_loader_rng', 'execution_policy.py'): 'encoder_helper',
            ('resnet18', 'custom_rn.py'): 'resnet18', ('ResNet.__init__', 'custom_rn.py'): 'ResNet_init',
            ('load_frozen_aux_encoder', 'aux_checkpoint.py'): 'load_frozen_aux_encoder',
            ('load_freeze_record', 'aux_checkpoint.py'): 'load_freeze_record',
            ('verify_future_checkpoint', 'encoder.py'): 'verify_future_checkpoint',
            ('load_verified_whole_module', 'aux_checkpoint.py'): 'load_verified_whole_module',
            ('read_verified', 'aux_checkpoint.py'): 'read_verified',
            ('encoder_identity', 'aux_checkpoint.py'): 'encoder_identity'}
COUNTERS = ('throwaway_constructor', 'resnet_init_calls', 'sha_verified_event', 'sha_rejected_event', 'torch_load',
            'checkpoint_read_opens', 'checkpoint_write_attempts', 'AdamW_construction', 'optimizer_constructions',
            'scheduler_construction', 'DataLoader_iterator', 'batch', 'randint_in_iteration', 'training_losses',
            'encoder_forward', 'model_forward', 'zero_grad', 'backward', 'scheduler_step', 'optimizer_step',
            'accumulate', 'torch_save', 'sampling', 'autograd_grad', 'autocast_entries', 'grad_scaler_constructions',
            'torch_lr_scheduler_constructions', 'deterministic_algorithms_calls', 'TRAIN_reads', 'VAL_reads',
            'TEST_reads', 'manifest_opens', 'M8_outputs', 'scientific_runs')
EXPECTED_COUNTERS = {k: 0 for k in COUNTERS} | {
    'throwaway_constructor': 1, 'resnet_init_calls': 1, 'sha_verified_event': 1, 'torch_load': 1,
    'checkpoint_read_opens': 2, 'AdamW_construction': 1, 'optimizer_constructions': 1, 'scheduler_construction': 1,
    'DataLoader_iterator': 1, 'batch': 1, 'randint_in_iteration': 1, 'training_losses': 1, 'encoder_forward': 1,
    'model_forward': 1, 'zero_grad': 1, 'backward': 1, 'scheduler_step': 1, 'optimizer_step': 1, 'accumulate': 1}
RNG_PROOF_COUNTERS = {'post_hoc_direct_constructor_replays': 1, 'post_hoc_dataloader_replay_iterators': 2}
QUALIFIED = ['E07c_MAIN_RUNNER_ENCODER_LOAD_INTEGRATION_QUALIFIED', 'MAIN_RUNNER_ENCODER_LOAD_INTEGRATION',
             'E07c_MAIN_DIFFFAS_TRAINING_GRAPH_QUALIFIED', 'MAIN_DIFFFAS_TRAINING_GRAPH',
             'E07c_MAIN_B4_TRAINING_MEMORY_QUALIFIED', 'E07c_MAIN_A7_ORDER_RNG_COMPATIBILITY_QUALIFIED']
NOT_QUALIFIED = ['MAIN_PRODUCTION_RUNNER', 'MAIN_CHECKPOINT_RESUME', 'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK']
NOT_CLAIMED = ['A1_TRAIN_DATASET_INTEGRATION', 'CANONICAL_FACE_READER', 'GUIDE_MAPPING', 'PRODUCTION_DATA_PATH',
               'BITWISE_DETERMINISTIC_MAIN_TRAINING', 'MAIN_CHECKPOINT_CADENCE_OR_TERMINAL_CHECKPOINT',
               'MAIN_VISUALIZATION_BRANCH', 'RUN_LOGGING_V1_MAIN_RUN']
ELIGIBILITY = {'QUALIFICATION_ONLY': True, 'NOT_A_SCIENTIFIC_CHECKPOINT': True, 'NOT_ELIGIBLE_FOR_BANK': True,
               'NOT_ELIGIBLE_FOR_DOWNSTREAM': True, 'NOT_ELIGIBLE_FOR_REPORTING': True,
               'NOT_ELIGIBLE_FOR_MAIN_SCIENTIFIC_TRAINING': True, 'state_persisted': False}
LAUNCH_ENV = {'CUDA_VISIBLE_DEVICES': '0', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
              'PYTHONHASHSEED': str(SEED), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
PRELAUNCH_MAX_USED_MIB = 512


def require(value, message):
    if not value:
        raise RuntimeError('M6D6i gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MainFirewall(rq.Firewall):
    """rq.Firewall plus ONE narrow exception: read-only opens of the exact canonical frozen checkpoint path.

    Records the ordered security trace (SHA verdict audit events, checkpoint opens, pickle.find_class). A write
    open of the frozen path, any other weight file, anything else under runs/, manifests, parquet, faces_256,
    frames or image files are denied by the inherited classification.
    """

    def __init__(self, build, frozen_path, trace):
        super().__init__(build)
        self.frozen = os.path.abspath(str(frozen_path))
        self.trace = trace
        self.checkpoint_opens = []
        self.find_class = []

    @staticmethod
    def writing(args):
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) or set(mode) & set('wax+'))

    def __call__(self, event, args):
        if event in (seam.EVENT_VERIFIED, seam.EVENT_REJECTED):
            self.trace.append({'event': 'sha256_verified' if event == seam.EVENT_VERIFIED else 'sha256_rejected',
                               'path': str(args[0]), 'sha256': str(args[1])})
            return
        if event == 'pickle.find_class':
            self.find_class.append(f'{args[0]}.{args[1]}')
            self.trace.append({'event': 'find_class', 'global': f'{args[0]}.{args[1]}'})
            return
        if (event == 'open' and args and isinstance(args[0], (str, bytes)) and
                os.path.abspath(os.fsdecode(args[0])) == self.frozen):
            self.events['open'] += 1
            if self.writing(args):
                self.denied.append({'event': event, 'path': self.frozen, 'classes': ['frozen_checkpoint_write']})
                self.attempts['write_outside_build'] += 1
                raise RuntimeError('FROZEN CHECKPOINT WRITE FIREWALL: ' + self.frozen)
            record = {'event': 'checkpoint_open', 'path': self.frozen,
                      'mode': args[1] if len(args) > 1 and isinstance(args[1], str) else None,
                      'flags': args[2] if len(args) > 2 and isinstance(args[2], int) else None, 'read_only': True}
            self.checkpoint_opens.append(record)
            self.trace.append(record)
            return
        return super().__call__(event, args)


def analytic_rgb(np, index, role):
    """Deterministic analytic uint8 RGB 256x256 image; no RNG, no file."""
    phase = 11 * index + 37 * ROLE_PHASE[role]
    yy, xx = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256), indexing='ij')
    fx, fy = 1 + phase % 5, 1 + (3 * phase) % 7
    arr = np.stack((np.sin(2 * np.pi * fx * xx + 0.1 * phase) * np.cos(2 * np.pi * fy * yy),
                    np.cos(2 * np.pi * fy * xx - 0.1 * phase) * (2 * yy - 1),
                    np.sin(2 * np.pi * (fx * xx + fy * yy) + 0.2 * phase)), -1)
    return np.clip(np.rint((arr + 1) * 127.5), 0, 255).astype(np.uint8)


class SyntheticTriplets:
    """STRUCTURAL QUALIFICATION SUBSTITUTE: N=8 in-memory items shaped like FAS_dataset.py items (no label key).

    Not the A1 TRAIN dataset, not the canonical face reader and not the guide mapping.
    """

    def __init__(self, np, Image, transform):
        self.np, self.Image, self.transform, self.accessed = np, Image, transform, []

    def __len__(self):
        return N_ITEMS

    def raw(self, idx, role):
        return analytic_rgb(self.np, idx, role)

    def item(self, idx):
        content_img, gt_img, style_img = (self.Image.fromarray(self.raw(idx, r), 'RGB') for r in ('content', 'GT',
                                                                                                    'style_spoof'))
        content_img = self.transform(content_img)          # FAS_dataset.py:43-45 transform order
        style_img = self.transform(style_img)
        gt_img = self.transform(gt_img)
        return {'content': content_img, 'style_spoof': style_img, 'GT': gt_img}

    def __getitem__(self, idx):
        self.accessed.append(int(idx))
        return self.item(int(idx))


def tensor_sha(t):
    return sha(t.detach().cpu().contiguous().numpy().tobytes())


def rng_snapshot(torch, np):
    state = np.random.get_state()
    return {'cpu_sha256': sha(torch.get_rng_state().numpy().tobytes()),
            'cuda_sha256': sha(torch.cuda.get_rng_state().numpy().tobytes()),
            'python_random_sha256': sha(repr(random.getstate()).encode()),
            'numpy_sha256': sha(state[1].tobytes() + repr((state[0], *state[2:])).encode())}


def custom_resnets():
    gc.collect()
    out = []
    for o in gc.get_objects():
        if type(o).__name__ == 'ResNet' and type(o).__module__ in ('custom_rn', 'models.custom_rn'):
            out.append({'module': type(o).__module__, 'fc_out': o.fc.out_features,
                        'devices': sorted({str(p.device) for p in o.parameters()})})
    return out


def digests(pairs):
    return {n: tensor_sha(t) for n, t in pairs}


def aggregate(d):
    return sha(json.dumps(d, sort_keys=True).encode())


class Probe:
    """Pass-through observers (counts, order trace, RNG snapshots); forbids everything outside the one-step contract."""

    def __init__(self, torch, np, trace):
        self.torch, self.np, self.trace = torch, np, trace
        self.n = {k: 0 for k in COUNTERS}
        self.window = None          # None | 'train' | 'replay'
        self.observed = {}

    def mark(self, event, **extra):
        self.trace.append({'event': event, **extra})

    def install(self, tl):
        torch, n, probe = self.torch, self.n, self
        orig = {'load': torch.load, 'opt_init': torch.optim.Optimizer.__init__, 'adamw_step': torch.optim.AdamW.step,
                'zero_grad': torch.optim.Optimizer.zero_grad, 'backward': torch.autograd.backward,
                'randint': torch.randint, 'dl_iter': torch.utils.data.DataLoader.__iter__,
                'dl_next': torch.utils.data.dataloader._BaseDataLoaderIter.__next__,
                'ps_init': tl.PhaseScheduler.__init__, 'ps_step': tl.PhaseScheduler.step}

        def forbid(name):
            def forbidden(*args, **kwargs):
                n[name] += 1
                raise RuntimeError('M6D6i forbids ' + name)
            return forbidden

        def load(f, *args, **kwargs):
            n['torch_load'] += 1
            call = {'argument_type': f'{type(f).__module__}.{type(f).__qualname__}', 'extra_positional': len(args),
                    'kwargs': sorted(kwargs), 'weights_only': kwargs.get('weights_only', 'NOT_PASSED')}
            probe.observed['torch_load_call'] = call
            probe.mark('torch_load', **call)
            require(n['torch_load'] == 1 and isinstance(f, io.BytesIO) and not args and
                    kwargs == {'weights_only': False}, 'exactly one torch.load(BytesIO(verified bytes), weights_only=False)')
            return orig['load'](f, *args, **kwargs)

        def opt_init(self_, *args, **kwargs):
            n['optimizer_constructions'] += 1
            require(type(self_) is torch.optim.AdamW and n['optimizer_constructions'] == 1, 'exactly one AdamW')
            n['AdamW_construction'] += 1
            probe.mark('optimizer_init')
            return orig['opt_init'](self_, *args, **kwargs)

        def adamw_step(self_, *args, **kwargs):
            n['optimizer_step'] += 1
            require(n['optimizer_step'] == 1 and n['backward'] == 1 and n['scheduler_step'] == 1,
                    'exactly one optimizer.step, after backward and after scheduler.step')
            probe.observed['lr_at_optimizer_step'] = [g['lr'] for g in self_.param_groups]
            probe.mark('optimizer_step')
            return orig['adamw_step'](self_, *args, **kwargs)

        def zero_grad(self_, *args, **kwargs):
            n['zero_grad'] += 1
            require(n['zero_grad'] == 1 and n['backward'] == 0, 'zero_grad once, before backward')
            probe.observed['lr_before_scheduler_step'] = [g['lr'] for g in self_.param_groups]
            probe.mark('zero_grad')
            return orig['zero_grad'](self_, *args, **kwargs)

        def backward(*args, **kwargs):
            n['backward'] += 1
            require(n['backward'] == 1 and n['zero_grad'] == 1, 'exactly one backward after zero_grad')
            probe.mark('backward')
            return orig['backward'](*args, **kwargs)

        def randint(*args, **kwargs):
            out = orig['randint'](*args, **kwargs)
            if probe.window == 'train':
                n['randint_in_iteration'] += 1
                probe.observed['time_t'] = out.tolist()
                probe.observed['cuda_rng_after_randint'] = sha(torch.cuda.get_rng_state().numpy().tobytes())
                probe.mark('randint')
            return out

        def dl_iter(self_):
            snap = rng_snapshot(torch, probe.np)
            if probe.window == 'replay':
                probe.observed.setdefault('replay_iterators', 0)
                probe.observed['replay_iterators'] += 1
            else:
                n['DataLoader_iterator'] += 1
                probe.observed['rng_at_dataloader_iter'] = snap
                probe.mark('dataloader_iter')
            return orig['dl_iter'](self_)

        def dl_next(self_):
            batch = orig['dl_next'](self_)
            if probe.window != 'replay':
                n['batch'] += 1
                probe.observed['rng_after_first_batch'] = rng_snapshot(torch, probe.np)
                probe.observed['batch_sha256'] = {k: tensor_sha(batch[k]) for k in sorted(batch)}
                probe.observed['batch_keys'] = sorted(batch)
                probe.observed['batch_shapes'] = {k: list(batch[k].shape) for k in sorted(batch)}
                probe.observed['batch_dtypes'] = {k: str(batch[k].dtype) for k in sorted(batch)}
                probe.mark('batch')
            return batch

        def ps_init(self_, *args, **kwargs):
            n['scheduler_construction'] += 1
            require(n['scheduler_construction'] == 1, 'exactly one tensorfn PhaseScheduler')
            probe.mark('scheduler_init')
            return orig['ps_init'](self_, *args, **kwargs)

        def ps_step(self_, *args, **kwargs):
            n['scheduler_step'] += 1
            require(n['scheduler_step'] == 1 and n['backward'] == 1 and n['optimizer_step'] == 0,
                    'exactly one scheduler.step, after backward and BEFORE optimizer.step (FAS_train.py:76-77)')
            probe.mark('scheduler_step')
            return orig['ps_step'](self_, *args, **kwargs)

        torch.load = load
        torch.save = forbid('torch_save')
        torch.optim.Optimizer.__init__ = opt_init
        torch.optim.AdamW.step = adamw_step
        for cls in {c for c in vars(torch.optim).values() if isinstance(c, type) and
                    issubclass(c, torch.optim.Optimizer) and c is not torch.optim.AdamW}:
            cls.step = forbid('optimizer_step')
        torch.optim.Optimizer.zero_grad = zero_grad
        torch.autograd.backward = backward
        torch.autograd.grad = forbid('autograd_grad')
        torch.randint = randint
        torch.utils.data.DataLoader.__iter__ = dl_iter
        torch.utils.data.dataloader._BaseDataLoaderIter.__next__ = dl_next
        torch.optim.lr_scheduler.LRScheduler.__init__ = forbid('torch_lr_scheduler_constructions')
        tl.PhaseScheduler.__init__, tl.PhaseScheduler.step = ps_init, ps_step
        torch.amp.autocast_mode.autocast.__enter__ = forbid('autocast_entries')
        torch.amp.grad_scaler.GradScaler.__init__ = forbid('grad_scaler_constructions')
        torch.use_deterministic_algorithms = forbid('deterministic_algorithms_calls')

    def install_upstream(self, dmod, diffusion):
        n, probe = self.n, self
        for name in ('p_sample_loop', 'p_sample_loop_progressive', 'p_sample_cond_loop', 'p_sample_cond_loop_progressive',
                     'p_sample', 'p_sample_cond', 'calc_bpd_loop'):
            setattr(dmod.GaussianDiffusion, name, self._forbid('sampling'))
        dmod.ddim_steps = self._forbid('sampling')
        bound = diffusion.training_losses

        def training_losses(*args, **kwargs):
            n['training_losses'] += 1
            require(n['training_losses'] == 1, 'exactly one training_losses')
            probe.observed['training_losses_kwargs'] = {k: (v if isinstance(v, (bool, int, float)) else type(v).__name__)
                                                        for k, v in sorted(kwargs.items()) if k not in ('cond_input',)}
            probe.mark('training_losses_enter')
            out = bound(*args, **kwargs)
            probe.mark('training_losses_exit')
            return out
        diffusion.training_losses = training_losses

    def _forbid(self, name):
        n = self.n

        def forbidden(*args, **kwargs):
            n[name] += 1
            raise RuntimeError('M6D6i forbids ' + name)
        return forbidden


def seam_profiler(torch, np, probe, captured):
    """Read-only sys.setprofile observer of the A7 seam: calls/returns, RNG snapshots, throwaway object facts."""
    def prof(frame, event, arg):
        if event not in ('call', 'return'):
            return
        code = frame.f_code
        key = PROFILED.get((code.co_qualname, Path(code.co_filename).name))
        if key is None:
            return
        if event == 'call':
            if key == 'resnet18':
                probe.n['throwaway_constructor'] += 1
            if key == 'ResNet_init':
                probe.n['resnet_init_calls'] += 1
            if key in ('encoder_helper', 'resnet18'):
                captured[key + '_call'] = {'rng': rng_snapshot(torch, np), 'cuda_allocated': torch.cuda.memory_allocated()}
                if key == 'encoder_helper':
                    captured['pre_helper_cpu_state'] = torch.get_rng_state().clone()
                    captured['pre_helper_cuda_state'] = torch.cuda.get_rng_state().clone()
            if key == 'load_frozen_aux_encoder':
                captured['load_call'] = {'rng': rng_snapshot(torch, np)}
            probe.mark(key + '_call')
        elif key in ('encoder_helper', 'resnet18'):
            info = {'rng': rng_snapshot(torch, np), 'cuda_allocated': torch.cuda.memory_allocated()}
            if key == 'resnet18':
                info['object'] = {'type': f'{type(arg).__module__}.{type(arg).__qualname__}',
                                  'fc': [arg.fc.in_features, arg.fc.out_features],
                                  'devices': sorted({str(p.device) for p in arg.parameters()}),
                                  'aggregate_parameter_sha256': aggregate(digests(arg.named_parameters()))}
            else:
                info['return_value'] = arg if rq_plain(arg) else f'NON_PLAIN:{type(arg).__qualname__}'
                info['return_value_plain'] = rq_plain(arg)
            captured[key + '_return'] = info
            probe.mark(key + '_return')
    return prof


def rq_plain(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and rq_plain(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(rq_plain(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


def stat_identity(path):
    st = os.stat(path)
    return {'size_bytes': st.st_size, 'inode': st.st_ino, 'device': st.st_dev, 'mode': oct(st.st_mode),
            'nlink': st.st_nlink, 'uid': st.st_uid, 'gid': st.st_gid, 'mtime_ns': st.st_mtime_ns,
            'ctime_ns': st.st_ctime_ns}


# ============================================================================= process mode
def qualify(build, process, runtime_root):
    trace = []
    frozen_path = seam.frozen_checkpoint_path(runtime_root)
    firewall = MainFirewall(build, frozen_path, trace)
    sys.addaudithook(firewall)
    for k, v in LAUNCH_ENV.items():
        require(os.environ.get(k) == v, f'launch environment {k}={v}')
    source, semantics, adapter, contracts = rq.source_identity()
    cfg = adapter.config
    static = rq.contract(adapter, contracts)
    from methods.difffas import execution_policy as ep
    loaded = ep.load_policy(cfg)
    policy = loaded['policy']
    order = ep.upstream_main_order(source, policy)
    loader_semantics = ep.upstream_loader_semantics(source)
    require(ep.upstream_precision_requests(source) == [], 'pinned source requests no AMP/TF32/half precision')
    freeze = seam.load_freeze_record(cfg)['frozen_asset']
    require((freeze['sha256'], freeze['bytes']) == (seam.OWNER_FROZEN_SHA256, seam.OWNER_FROZEN_BYTES),
            'M6D6h owner-frozen identity')
    require(SEED not in cfg['seeds']['experiment_seeds'] and
            SEED != cfg['conditioning_encoder']['auxiliary_encoder_training_seed'], 'qualification seed is not scientific')
    equivalence = mg.verify_source_equivalence(source)
    args, args_info = mg.source_args(source, cfg)
    require(args.batch_size == BATCH == cfg['training']['batch_size'], 'exact frozen B=4')
    c6d6 = json.loads((ROOT / M6D6C_EVIDENCE).read_bytes())
    approved_globals = sorted(c6d6['module_identity']['find_class_globals'])

    import inspect
    import numpy as np
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from PIL import Image
    from tqdm import tqdm
    import tensorfn.optim.lr_scheduler as tl
    from tensorfn import load_config
    from methods.common.upstream import upstream_modules
    rq.TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(inspect.signature(torch.load).parameters['weights_only'].default)
    probe = Probe(torch, np, trace)
    probe.install(tl)
    result = {'milestone': MILESTONE, 'classification': CLASSIFICATION, 'label': LABEL, 'process': process,
              'qualification_seed': SEED, 'experiment_seed': None, 'experiment_seed_reason': EXPERIMENT_SEED_REASON,
              'auxiliary_encoder_training_seed': None, 'scientific_seed_consumed': False,
              'authority_commit': AUTHORITY, 'eligibility': ELIGIBILITY,
              'repository_head': rq.command('git', '-C', str(ROOT), 'rev-parse', 'HEAD'),
              'repository_candidate_status': rq.command('git', '-C', str(ROOT), 'status', '--porcelain',
                                                        '--untracked-files=all').splitlines(),
              'candidate_sha256': {rel: sha((ROOT / rel).read_bytes()) for rel in
                                   ('methods/difffas/main_graph.py', 'methods/difffas/main_graph_qualification.py')},
              'contract': static, 'source_before': source, 'executable_semantics': semantics,
              'a7': {'overlay': ep.A7_OVERLAY, 'overlay_sha256': loaded['sha256'],
                     'document': policy['amendment_document'], 'upstream_main_order': order,
                     'upstream_loader_semantics': loader_semantics},
              'main_graph_source_equivalence': equivalence, 'source_args': args_info,
              'frozen_checkpoint': {'path': str(frozen_path), 'path_template': freeze['path_template'],
                                    'sha256': freeze['sha256'], 'bytes': freeze['bytes'],
                                    'freeze_record': seam.FREEZE_RECORD, 'freeze_record_sha256': seam.FREEZE_RECORD_SHA256},
              'runtime_root': str(runtime_root)}
    require(result['repository_head'] == AUTHORITY, 'GPU repository HEAD is the M6D6h authority')
    captured = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with upstream_modules(Path(source['root']), rq.UPSTREAM_MAIN, rq.UPSTREAM_MAIN_ROOTS) as modules:
            dc, dmod, ua = (modules[n] for n in rq.UPSTREAM_MAIN)
            DiffConf = load_config(dc.DiffusionConfig, str(Path(source['root']) / 'config/diffusion.conf'), (), False)
            result['diff_config'] = mg.verify_diff_config(DiffConf, cfg)
            # ---- A7 step 1: FAS_train.py:167-175 seed_torch semantics with the qualification seed; A7 precision
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            probe.mark('seed')
            rng = {'after_seed': rng_snapshot(torch, np)}
            applied = ep.apply_e07c_precision_policy(cfg)
            probe.mark('precision_policy')
            result['environment_before'] = rq.environment()   # after the A7 policy; consumes no RNG
            result['precision'] = {'applied_state': applied, 'expected_state': ep.EXPECTED_STATE,
                                   'use_deterministic_algorithms': 'NOT_SET', 'autocast_entered': False,
                                   'grad_scaler_constructed': False}
            # ---- A7 steps 2-4
            transform = mg.build_transform(transforms)
            probe.mark('transform')
            dataset = SyntheticTriplets(np, Image, transform)
            probe.mark('dataset')
            loader = mg.build_dataloader(DataLoader, dataset, args)
            probe.mark('dataloader')
            require((loader.batch_size, loader.num_workers, loader.drop_last, type(loader.sampler).__name__,
                     loader.generator) == (BATCH, 0, False, 'RandomSampler', None), 'source DataLoader semantics')
            # ---- A7 steps 5-13
            def model_conf():
                probe.mark('model_conf')
                return dc.get_model_conf()

            def gaussian(*a, **kw):
                probe.mark('create_gaussian_diffusion')
                return dmod.create_gaussian_diffusion(*a, **kw)
            model, ema, optimizer, scheduler, betas, diffusion = mg.build_training_objects(model_conf, DiffConf,
                                                                                            gaussian, args)
            require(args.pretrain_path is None, 'fresh run (FAS_train.py:226 branch not taken)')
            probe.mark('fresh_run_asserted')
            rng['after_construction'] = rng_snapshot(torch, np)
            probe.install_upstream(dmod, diffusion)
            construction = construction_facts(torch, ua, dmod, model, ema, optimizer, scheduler, betas, diffusion)
            result['construction'] = construction
            model_pre = {'parameters': digests(model.named_parameters()), 'buffers': digests(model.named_buffers())}
            ema_pre = {'parameters': digests(ema.named_parameters()), 'buffers': digests(ema.named_buffers())}
            torch.cuda.synchronize()
            memory = {'after_main_construction_allocated_bytes': torch.cuda.memory_allocated()}
            instances_before = custom_resnets()
            require(instances_before == [], 'no custom_rn ResNet exists before the seam')
            # ---- A7 steps 14-19: FAS_train.py:34-35 -> main_runner_encoder, then encoder.eval()
            sys.setprofile(seam_profiler(torch, np, probe, captured))
            with ep.main_runner_encoder(runtime_root, seam.OWNER_FROZEN_SHA256, cfg) as encoder:
                encoder.eval()
                probe.mark('encoder_eval')
                sys.setprofile(None)
                rng['post_eval'] = rng_snapshot(torch, np)
                post_eval_cpu = torch.get_rng_state().clone()
                torch.cuda.synchronize()
                memory['after_encoder_load_allocated_bytes'] = torch.cuda.memory_allocated()
                instances_after = custom_resnets()
                enc = encoder_facts(torch, encoder, contracts)
                enc_pre = {'parameters': digests(encoder.named_parameters()),
                           'buffers': digests(encoder.named_buffers())}
                enc_outputs = []

                def encoder_hook(module, a, o):   # observes the ONE loss-path encoder call; returns None
                    enc_outputs.append({'input_shape': list(a[0].shape), 'shapes': [list(t.shape) for t in o],
                                        'dtypes': [str(t.dtype) for t in o], 'sha256': [tensor_sha(t) for t in o]})
                    probe.n['encoder_forward'] += 1
                    probe.mark('encoder_forward')
                encoder.register_forward_hook(encoder_hook)
                model_calls = []

                def model_pre_hook(module, a, kw):
                    model_calls.append({'x_shape': list(kw['x'].shape), 'x_sha256': tensor_sha(kw['x']),
                                        't': kw['t'].tolist(), 'cond_mask': kw['cond_mask'].tolist(),
                                        'x_cond_sha256': tensor_sha(kw['x_cond']), 'prob': kw['prob'],
                                        'means_size': kw['means_size'], 'var_size': kw['var_size'],
                                        'positional_args': len(a)})

                def model_hook(module, a, kw, out):
                    model_calls[-1].update(output_shape=list(out.shape), output_sha256=tensor_sha(out),
                                           output_finite=bool(torch.isfinite(out).all().item()))
                    probe.n['model_forward'] += 1
                    probe.mark('model_forward')
                model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
                model.register_forward_hook(model_hook, with_kwargs=True)
                acc_calls = []
                original_accumulate = mg.accumulate

                def accumulate(model1, model2, decay=0.9999):
                    probe.n['accumulate'] += 1
                    require(probe.n['accumulate'] == 1 and probe.n['optimizer_step'] == 1, 'one EMA update after the step')
                    acc_calls.append({'decay': decay, 'model1_is_ema': model1 is ema, 'model2_is_model': model2 is model})
                    probe.mark('accumulate')
                    return original_accumulate(model1, model2, decay)
                mg.accumulate = accumulate
                torch.cuda.reset_peak_memory_stats()
                # ---- A7 steps 20-23 + FAS_train.py:42-83: ONE bounded source-order iteration
                iters = 0
                out = None
                for epoch in range(args.max_epochs):
                    probe.mark('epoch_loop', epoch=epoch)
                    progress_bar = mg.epoch_progress(tqdm, loader)
                    probe.mark('tqdm')
                    rng['before_iterator'] = rng_snapshot(torch, np)
                    for batch in progress_bar:
                        dataset_order = list(dataset.accessed)
                        rng['cuda_before_iteration'] = sha(torch.cuda.get_rng_state().numpy().tobytes())
                        probe.window = 'train'
                        try:
                            out = mg.train_iteration(torch, DiffConf, args, batch, iters, model, ema, encoder,
                                                     diffusion, betas, optimizer, scheduler, 'cuda')
                        except torch.cuda.OutOfMemoryError as exc:
                            raise MemoryBlocked(str(exc)) from exc
                        probe.window = None
                        iters = out['iters']
                        break
                    progress_bar.close()
                    break
                torch.cuda.synchronize()
                mg.accumulate = original_accumulate
                rng['after_iteration'] = rng_snapshot(torch, np)
                memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              after_iteration_allocated_bytes=torch.cuda.memory_allocated(),
                              after_iteration_reserved_bytes=torch.cuda.memory_reserved())
                memory['nvidia_smi_process_after_iteration'] = rq.command(
                    'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader')
                require(out is not None and iters == 1, 'exactly one iteration')
                step = step_facts(torch, np, out, model, ema, encoder, optimizer, scheduler, model_pre, ema_pre,
                                  enc_pre, probe, acc_calls, DiffConf)
                enc['forward_in_loss_path'] = enc_outputs
                enc_post = {'parameters': digests(encoder.named_parameters()),
                            'buffers': digests(encoder.named_buffers())}
                require(enc_post == enc_pre, 'encoder parameters and buffers bitwise unchanged')
                require(all(p.grad is None for p in encoder.parameters()), 'encoder receives no gradient')
                enc.update(parameter_sha256=enc_pre['parameters'], buffer_sha256=enc_pre['buffers'],
                           aggregate_parameter_sha256=aggregate(enc_pre['parameters']),
                           aggregate_buffer_sha256=aggregate(enc_pre['buffers']),
                           unchanged_after_step={'parameters': True, 'buffers': True}, grads_all_none=True)
                del encoder
            # ---- post-hoc RNG replays (outside the seam; RNG_PROOF_ONLY, after the qualified step)
            a7rng = rng_proof(torch, np, probe, captured, rng, post_eval_cpu, instances_after, dataset, dataset_order,
                              loader, args, DataLoader, upstream_modules, source)
            model_calls_final = model_calls
            result.update(encoder=enc, a7_rng=a7rng, **step)
            result['encoder_load'] = seam_facts(firewall, probe, trace, approved_globals, frozen_path)
            result['synthetic_input'] = synthetic_facts(np, dataset, dataset_order, probe)
            result['pre_backward'] = pre_backward(probe, out, dataset_order, model_calls_final, enc_outputs, rng,
                                                  model_pre, ema_pre, enc_pre, a7rng)
            result['cuda_memory'] = {'label': 'MAIN_B4_TRAINING_STEP_MEMORY', 'batch_size': BATCH, **memory,
                                     'resident': 'frozen encoder + main model + EMA + gradients + AdamW moments',
                                     'amp': False, 'microbatch': False, 'activation_checkpointing': False,
                                     'cpu_offload': False, 'oom': False}
            stream_payload = (model, ema)
            result['_stream'] = stream_payload
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    require(ep.precision_state() == applied, 'A7 precision state unchanged to the end')
    result['source_after'] = rq.source_identity()[0]
    require(result['source_after'] == source and source['worktree_status'] == '', 'pinned source unchanged, clean')
    result['environment_after'] = rq.environment()
    require(result['environment_before'] == result['environment_after'], 'environment stable')
    order_events = [e['event'] for e in trace if e['event'] in EXPECTED_ORDER]
    order_events = [e for i, e in enumerate(order_events) if not (e == 'encoder_helper_call' and
                                                                   'encoder_helper_call' in order_events[:i])]
    require(order_events == EXPECTED_ORDER, 'A7 + FAS_train source order ' + json.dumps(order_events))
    require(probe.n == EXPECTED_COUNTERS, 'exact counters ' + json.dumps(probe.n))
    require(not firewall.denied and not any(firewall.attempts.values()), 'firewall: zero denied accesses')
    require(all(Path(a[0]).name in rq.ALLOWED_SUBPROCESSES or a in rq.ALLOWED_EXACT_ARGV
                for a in firewall.subprocesses), 'subprocess allowlist')
    result.update(status='PASS', counters=probe.n, rng_proof_counters=RNG_PROOF_COUNTERS, order=order_events,
                  security_trace=[e for e in trace if e['event'] in SEAM_EVENTS],
                  qualification_counters={'qualification_processes': 1, 'qualification_checkpoint_deserializations': 1,
                                          'qualification_backward_calls': 1, 'qualification_optimizer_steps': 1,
                                          'qualification_scheduler_steps': 1, 'qualification_ema_updates': 1},
                  scientific_counters={'scientific_main_runs': 0, 'scientific_optimizer_steps': 0,
                                       'scientific_checkpoint_deserializations': 0, 'scientific_checkpoint_writes': 0,
                                       'experiment_seed_runs': 0},
                  torch_save_calls=0, main_checkpoints_written=0, qualification_checkpoints_written=0,
                  TRAIN_access=False, VAL_access=False, TEST_access=False, manifest_access=False, M8_outputs=0,
                  synthetic_bank=False, production_runner_claimed=False, resume_executed=False,
                  visualization_branch_executed=False, sampling_executed=False,
                  fidelity='CONTROLLED_ADAPTATION', deviation='DEV-021', method_status='IMPLEMENTED_NOT_EXECUTED',
                  source_patch='NONE', not_claimed=NOT_CLAIMED,
                  firewall={'denied': firewall.denied, 'attempts': firewall.attempts, 'event_counts': firewall.events,
                            'subprocesses': firewall.subprocesses,
                            'frozen_checkpoint_read_opens': firewall.checkpoint_opens})
    return result


class MemoryBlocked(Exception):
    """Exact B=4 does not fit: STOP_AND_REPORT as BLOCKED_BY_MAIN_B4_MEMORY (no workaround authorized)."""


def construction_facts(torch, ua, dmod, model, ema, optimizer, scheduler, betas, diffusion):
    params = list(model.named_parameters())
    require(type(model) is ua.BeatGANsAutoencModel and type(ema) is ua.BeatGANsAutoencModel, 'pinned main model x2')
    require(model.input_blocks[0][0].in_channels == 3 and model.conf.in_channels == 3, 'use_pair=False 3-channel input')
    n_params = sum(p.numel() for _, p in params)
    require(n_params == MAIN_PARAMETERS == sum(p.numel() for p in ema.parameters()), 'main/EMA parameter count')
    require({str(p.device) for p in model.parameters()} | {str(p.device) for p in ema.parameters()} == {'cuda:0'} and
            {p.dtype for p in model.parameters()} == {torch.float32}, 'main/EMA on cuda:0 float32')
    require(all(m.training for m in model.modules()) and all(m.training for m in ema.modules()),
            'source never calls .train()/.eval() on model/EMA: default training=True')
    group = optimizer.param_groups
    require(len(group) == 1 and [id(p) for p in group[0]['params']] == [id(p) for _, p in params],
            'optimizer set == model.parameters()')
    hyper = {k: v for k, v in group[0].items() if k != 'params'}
    require((type(optimizer) is torch.optim.AdamW, hyper['lr'], tuple(hyper['betas']), hyper['eps'],
             hyper['weight_decay'], hyper['amsgrad']) == (True, 1e-5, (0.9, 0.999), 1e-8, 0, False),
            'AdamW(lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False) from tensorfn')
    require(type(scheduler).__name__ == 'PhaseScheduler' and scheduler.optimizer is optimizer and
            scheduler.phase_param == [('linear', 1e-5 * 4e-2, 1e-5, 5000), ('flat', 1e-5, 1e-5 * 1e-5, 2400000 - 5000)],
            'tensorfn cycle PhaseScheduler phases')
    require(betas.dtype == torch.float64 and list(betas.shape) == [1000] and float(betas[0]) == 1e-4 and
            abs(float(betas[-1]) - 2e-2) < 1e-15, 'linear float64 betas')
    require(diffusion.model_mean_type == dmod.ModelMeanType.EPSILON and
            diffusion.model_var_type == dmod.ModelVarType.LEARNED_RANGE and diffusion.loss_type == dmod.LossType.MSE and
            diffusion.rescale_timesteps is False, 'EPSILON / LEARNED_RANGE / MSE(+vb) diffusion')
    attention = sorted(n for n, m in model.named_modules() if type(m).__name__ == 'AttentionBlock')
    bn = sorted(n for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d))
    return {'model_class': f'{type(model).__module__}.{type(model).__qualname__}', 'ema_class_equal': True,
            'main_parameters': n_params, 'parameter_tensors': len(params),
            'buffers': len(list(model.named_buffers())), 'device': 'cuda:0', 'dtype': 'torch.float32',
            'training_mode_all_modules': True, 'in_channels': 3, 'use_pair': False,
            'optimizer': {'class': 'torch.optim.AdamW', 'factory': 'tensorfn config AdamW.make (config/diffusion.conf)',
                          'hyperparameters': {k: (list(v) if isinstance(v, tuple) else v) for k, v in hyper.items()}},
            'scheduler': {'class': f'{type(scheduler).__module__}.{type(scheduler).__qualname__}',
                          'phase_param': [list(p) for p in scheduler.phase_param]},
            'betas': {'dtype': 'torch.float64', 'n': 1000, 'first': float(betas[0]), 'last': float(betas[-1])},
            'diffusion': {'model_mean_type': 'EPSILON', 'model_var_type': 'LEARNED_RANGE', 'loss_type': 'MSE',
                          'rescale_timesteps': False, 'predict_xstart': False},
            'attention_blocks': attention, 'batchnorm2d_modules': bn,
            'ema_init': 'separately constructed get_model_conf().make_model() (FAS_train.py:218), not a copy'}


def encoder_facts(torch, encoder, contracts):
    custom_rn = sys.modules[type(encoder).__module__]
    identity = seam.encoder_identity(encoder, custom_rn, contracts)   # pure identity check; no load
    params = list(encoder.parameters())
    require(type(encoder) is custom_rn.ResNet and type(encoder).__module__ == 'custom_rn', 'custom_rn.ResNet')
    require([len(getattr(encoder, f'layer{i}')) for i in (1, 2, 3, 4)] == [3, 4, 6, 3] and
            {type(b) for i in (1, 2, 3, 4) for b in getattr(encoder, f'layer{i}')} == {custom_rn.BasicBlock},
            'BasicBlock [3,4,6,3]')
    require(type(encoder.fc) is torch.nn.Linear and (encoder.fc.in_features, encoder.fc.out_features) == (512, K) and
            encoder.fc.bias is not None, 'fc = Linear(512, 7) with bias')
    n = sum(p.numel() for p in params)
    require(n == ENCODER_PARAMETERS, f'encoder parameter count {n}')
    require({p.dtype for p in params} == {torch.float32}, 'encoder float32')
    require({str(t.device) for t in (*params, *encoder.buffers())} == {'cuda:0'}, 'encoder on cuda:0')
    require(encoder.training is False and not any(m.training for m in encoder.modules()), 'encoder.eval() everywhere')
    return {'identity': identity, 'class': 'custom_rn.ResNet', 'module': 'custom_rn',
            'module_file_sha256': identity['module_file_sha256'], 'topology': [3, 4, 6, 3], 'block': 'BasicBlock',
            'fc': {'in_features': 512, 'out_features': K, 'bias': True}, 'parameters': n, 'dtype': 'torch.float32',
            'device': 'cuda:0', 'training': False, 'submodules_training_false': True,
            'eval_called_by': 'caller, first statement inside main_runner_encoder (FAS_train.py:35)'}


def step_facts(torch, np, out, model, ema, encoder, optimizer, scheduler, model_pre, ema_pre, enc_pre, probe,
               acc_calls, DiffConf):
    for k in ('loss', 'loss_mse', 'loss_vb'):
        require(bool(torch.isfinite(out[k]).all().item()) and out[k].dim() == 0, k + ' finite scalar')
    ld = out['loss_dict']
    require(sorted(ld) == ['loss', 'mse', 'vb'] and list(ld['loss'].shape) == [BATCH] and list(ld['vb'].shape) == [BATCH]
            and ld['mse'].dim() == 0, 'loss = mse (scalar mean) + vb [B]')
    # gradients (after backward; AdamW never modifies .grad)
    grads, none_grad, region_present, region_nonzero, attention_present = {}, [], {}, {}, 0
    attention = [n for n, m in model.named_modules() if type(m).__name__ == 'AttentionBlock']
    for name, p in model.named_parameters():
        region = name.split('.')[0]
        if p.grad is None:
            none_grad.append(name)
            continue
        g = p.grad.detach()
        finite = bool(torch.isfinite(g).all().item())
        require(finite, 'finite gradient ' + name)
        nz = int((g != 0).sum().item())
        grads[name] = {'sha256': tensor_sha(g), 'nonzero_elements': nz, 'numel': g.numel(),
                       'max_abs': float(g.abs().max().item())}
        region_present[region] = region_present.get(region, 0) + 1
        region_nonzero[region] = region_nonzero.get(region, 0) + (1 if nz else 0)
        if any(name.startswith(a + '.') for a in attention):
            attention_present += 1
    for region in GRAD_REGIONS:
        require(region_present.get(region, 0) > 0, 'autograd-connected gradient tensors in ' + region)
    require(attention_present > 0, 'autograd-connected gradients in the AttentionBlocks that consume encoder features')
    for region in NONZERO_GRAD_REGIONS:
        require(region_nonzero.get(region, 0) > 0, 'nonzero gradient in ' + region)
    nonzero_names = sorted(n for n, g in grads.items() if g['nonzero_elements'])
    # parameters after the ONE step
    post = digests(model.named_parameters())
    changed = sorted(n for n in post if post[n] != model_pre['parameters'][n])
    require(set(changed) <= set(grads), 'only parameters with gradients changed')
    require(not set(changed) & set(none_grad), 'None-grad parameters unchanged')
    zero_grad_changed = sorted(n for n in changed if grads[n]['nonzero_elements'] == 0)
    require(not zero_grad_changed, 'AdamW (weight_decay 0) leaves all-zero-gradient parameters bitwise unchanged')
    for region in NONZERO_GRAD_REGIONS:
        require(any(n.split('.')[0] == region for n in changed), 'parameters updated in ' + region)
    require(all(bool(torch.isfinite(p).all().item()) for p in model.parameters()), 'finite parameters after step')
    # optimizer
    group = optimizer.param_groups[0]
    states, steps = 0, set()
    for name, p in model.named_parameters():
        st = optimizer.state.get(p, {})
        if p.grad is None:
            require(not st, 'no optimizer state for None-grad ' + name)
            continue
        require(sorted(st) == ['exp_avg', 'exp_avg_sq', 'step'], 'AdamW state keys ' + name)
        steps.add(float(st['step']))
        states += 1
    require(steps == {1.0}, 'optimizer state step == 1 for every parameter with a gradient')
    lr0 = 1e-5 * 4e-2
    require(probe.observed['lr_before_scheduler_step'] == [1e-5], 'AdamW lr 1e-5 before scheduler.step')
    require(probe.observed['lr_at_optimizer_step'] == [lr0] == [scheduler.latest_lr] == [group['lr']] and
            lr0 == scheduler.phase_param[0][1], 'applied first-step lr == tensorfn anneal_linear(1e-5*0.04, 1e-5, 0/5000)')
    sd = scheduler.state_dict()
    require((sd['phase'], sd['phase_step'], sd['latest_lr']) == (0, 1, lr0), 'scheduler phase 0, phase_step 1')
    # EMA: decay 0 at iters=1 < warmup 5000
    require(acc_calls == [{'decay': 0, 'model1_is_ema': True, 'model2_is_model': True}] and
            DiffConf.training.scheduler.warmup == 5000, 'accumulate(ema, model, 0)')
    ema_equal, byte_mismatch, signed_zero_only = True, 0, True
    for (n1, pe), (n2, pm) in zip(ema.named_parameters(), model.named_parameters()):
        require(n1 == n2, 'EMA/model parameter order')
        ema_equal &= bool(torch.equal(pe, pm))
        diff = pe.view(torch.int32) != pm.view(torch.int32)
        count = int(diff.sum().item())
        byte_mismatch += count
        if count:
            signed_zero_only &= bool((pm[diff] == 0).all().item())
    require(ema_equal and signed_zero_only, 'EMA named_parameters == main named_parameters after decay-0 accumulate')
    ema_post = {'parameters': digests(ema.named_parameters()), 'buffers': digests(ema.named_buffers())}
    require(ema_post['buffers'] == ema_pre['buffers'], 'EMA buffers untouched (source copies named_parameters only)')
    model_buffers = digests(model.named_buffers())
    buffers_changed = sorted(n for n in model_buffers if model_buffers[n] != model_pre['buffers'][n])
    return {
        'iteration': {'iters': out['iters'], 'epoch': 0, 'time_t': out['time_t'].tolist(),
                      'loss': float(out['loss'].item()), 'loss_float32_hex': float(out['loss'].item()).hex(),
                      'mse': float(out['loss_mse'].item()), 'mse_hex': float(out['loss_mse'].item()).hex(),
                      'vb': float(out['loss_vb'].item()), 'vb_hex': float(out['loss_vb'].item()).hex(),
                      'loss_per_sample_sha256': tensor_sha(ld['loss']), 'vb_per_sample_sha256': tensor_sha(ld['vb']),
                      'loss_list': out['loss_list'], 'loss_mean_list': out['loss_mean_list'],
                      'loss_vb_list': out['loss_vb_list'], 'finite': True,
                      'target': 'epsilon (ModelMeanType.EPSILON: noise) for x_start = GT; learned-range vb; loss = mse + vb',
                      'training_losses_kwargs': probe.observed['training_losses_kwargs'],
                      'save_branch_taken': False, 'visualization_branch_taken': False,
                      'save_branch_reason': '1 % 10000 != 0 and the branch is not implemented',
                      'visualization_branch_reason': '1 % 1000 != 0 and the branch is not implemented'},
        'gradients': {'tensors_with_grad': len(grads), 'none_grad_tensors': none_grad,
                      'tensors_with_nonzero_grad': sum(1 for g in grads.values() if g['nonzero_elements']),
                      'all_finite': True, 'region_tensors_with_grad': region_present,
                      'region_tensors_with_nonzero_grad': region_nonzero,
                      'attention_tensors_with_grad': attention_present, 'nonzero_grad_tensors': nonzero_names,
                      'reachability_rule': ('structural: every region (and the AttentionBlocks consuming the encoder '
                                            'features) holds autograd-connected gradient tensors; numerically nonzero '
                                            'only in the zero-initialised final out projection at the first step '
                                            '(pinned resnet_use_zero_module / zero_module out conv)'),
                      'required_regions': list(GRAD_REGIONS), 'required_nonzero_regions': list(NONZERO_GRAD_REGIONS),
                      'per_tensor': grads,
                      'grad_presence_sha256': aggregate({'none': none_grad, 'present': sorted(grads)})},
        'scheduler': {'class': 'tensorfn.optim.lr_scheduler.PhaseScheduler', 'steps': 1,
                      'order': 'scheduler.step() BEFORE optimizer.step() (FAS_train.py:76-77)',
                      'lr_before_scheduler_step': probe.observed['lr_before_scheduler_step'],
                      'applied_lr_at_optimizer_step': probe.observed['lr_at_optimizer_step'],
                      'applied_lr_repr': repr(lr0), 'source_derivation': 'anneal_linear(1e-5*4e-2, 1e-5, 0/5000)',
                      'state_dict': {'phase': sd['phase'], 'phase_step': sd['phase_step'], 'latest_lr': sd['latest_lr'],
                                     'phase_param': [list(p) for p in sd['phase_param']]}},
        'optimizer': {'class': 'torch.optim.AdamW', 'steps': 1,
                      'hyperparameters_after': {k: (list(v) if isinstance(v, tuple) else v) for k, v in group.items()
                                                if k != 'params'},
                      'state_entries': states, 'state_step_values': sorted(steps),
                      'no_state_for_none_grad': True, 'second_step': False},
        'parameters': {'aggregate_sha256_before': aggregate(model_pre['parameters']),
                       'aggregate_sha256_after': aggregate(post), 'changed_tensors': len(changed),
                       'unchanged_tensors': len(post) - len(changed), 'changed': changed,
                       'changed_equals_nonzero_grad_tensors': changed == nonzero_names,
                       'changed_tensors_with_all_zero_grad': zero_grad_changed,
                       'per_tensor_sha256_before': model_pre['parameters'], 'per_tensor_sha256_after': post,
                       'buffers_changed_by_train_mode_forward': buffers_changed,
                       'buffers_note': 'AttentionBlock.bn BatchNorm2d on cond (train mode) updates running stats; '
                                       'source-native'},
        'ema': {'accumulate_calls': acc_calls, 'decay_applied': 0, 'warmup': 5000,
                'rule': '0 if iters < conf.training.scheduler.warmup else 0.9999 (FAS_train.py:83)',
                'parameters_equal_main_after': True, 'byte_mismatches_signed_zero_only': byte_mismatch,
                'aggregate_parameter_sha256_before': aggregate(ema_pre['parameters']),
                'aggregate_parameter_sha256_after': aggregate(ema_post['parameters']),
                'buffers_unchanged': True,
                'source_quirk_recorded_not_fixed': 'accumulate copies named_parameters only; EMA buffers (incl. '
                                                   'AttentionBlock BatchNorm running stats) are never updated'}}


def seam_facts(firewall, probe, trace, approved_globals, frozen_path):
    seam_events = [e['event'] for e in trace if e['event'] in SEAM_EVENTS]
    collapsed = [e for i, e in enumerate(seam_events) if not (e == 'find_class' and i and seam_events[i - 1] == e)]
    require(collapsed == EXPECTED_SEAM, 'secure seam order ' + json.dumps(collapsed))
    verified = [e for e in trace if e['event'] == 'sha256_verified']
    require(len(verified) == 1 and verified[0]['sha256'] == seam.OWNER_FROZEN_SHA256 and
            verified[0]['path'] == str(frozen_path), 'one SHA-verified event with the frozen digest')
    require(not [e for e in trace if e['event'] == 'sha256_rejected'], 'zero SHA-rejected events')
    probe.n['sha_verified_event'], probe.n['sha_rejected_event'] = len(verified), 0
    opens = firewall.checkpoint_opens
    probe.n['checkpoint_read_opens'] = len(opens)
    require(len(opens) == 2 and all(o['read_only'] and o['path'] == str(frozen_path) for o in opens),
            'exactly two read-only opens of the canonical frozen path (verify_asset + read_verified)')
    observed = sorted(set(firewall.find_class))
    require(observed == approved_globals, 'unpickled globals == M6D6c whole-module set')
    return {'seam': 'methods.difffas.execution_policy.main_runner_encoder -> '
                    'methods.difffas.aux_checkpoint.load_frozen_aux_encoder',
            'caller_sha256': seam.OWNER_FROZEN_SHA256, 'ordered_events': collapsed,
            'find_class_events': len(firewall.find_class), 'find_class_globals': observed,
            'find_class_globals_equal_m6d6c': True, 'sha256_verified_events': verified,
            'sha256_rejected_events': 0, 'checkpoint_read_opens': opens, 'checkpoint_write_opens': 0,
            'torch_load_call': probe.observed['torch_load_call'], 'torch_load_calls': 1,
            'sha_verified_before_torch_load': True, 'harness_direct_loader_calls': 0,
            'deserializations': 1}


def rng_proof(torch, np, probe, captured, rng, post_eval_cpu, instances_after, dataset, dataset_order, loader, args,
              DataLoader, upstream_modules, source):
    hc, hr = captured['encoder_helper_call'], captured['encoder_helper_return']
    rc, rr = captured['resnet18_call'], captured['resnet18_return']
    lc = captured['load_call']
    require(hc['rng']['cpu_sha256'] != hr['rng']['cpu_sha256'], 'throwaway constructor consumes CPU RNG')
    require(rc['rng']['cpu_sha256'] != rr['rng']['cpu_sha256'] and rr['rng'] == hr['rng'] and rc['rng'] == hc['rng'],
            'the only RNG consumption inside the helper is the constructor')
    for k in ('cuda_sha256', 'python_random_sha256', 'numpy_sha256'):
        require(hc['rng'][k] == hr['rng'][k], 'helper leaves ' + k + ' untouched')
    require(hc['cuda_allocated'] == hr['cuda_allocated'] == rc['cuda_allocated'] == rr['cuda_allocated'],
            'throwaway never touches CUDA memory')
    require(rr['object']['type'] == 'custom_rn.ResNet' and rr['object']['fc'] == [512, 17] and
            rr['object']['devices'] == ['cpu'], 'throwaway: pinned default 17-way head, CPU')
    require(hr['return_value_plain'] and hr['return_value']['role'] == 'RNG_COMPATIBILITY_ONLY' and
            hr['return_value']['object_returned'] is False, 'helper returns a plain record only')
    require(lc['rng'] == hr['rng'] and rng['post_eval'] == hr['rng'], 'secure load + eval consume neither CPU nor CUDA RNG')
    require([i['fc_out'] for i in instances_after] == [K] and not [i for i in instances_after if i['fc_out'] == 17],
            'no 17-way object survives; the only custom_rn.ResNet is the frozen K7 encoder')
    require(probe.observed['rng_at_dataloader_iter'] == rng['post_eval'] == rng['before_iterator'],
            'DataLoader iterator sees the post-helper / post-eval CPU RNG state')
    require(rng['after_iteration']['cpu_sha256'] == probe.observed['rng_after_first_batch']['cpu_sha256'],
            'the step consumes no CPU RNG (CUDA generator only)')
    require(rng['cuda_before_iteration'] == rng['post_eval']['cuda_sha256'], 'no CUDA RNG consumer before time_t')
    # replay 1: direct pinned constructor from the restored pre-helper state
    probe.window = 'replay'
    torch.set_rng_state(captured['pre_helper_cpu_state'])
    require(sha(torch.get_rng_state().numpy().tobytes()) == hc['rng']['cpu_sha256'], 'restored pre-helper state')
    with upstream_modules(Path(source['root']) / 'models', ('custom_rn',), {'custom_rn'}) as modules:
        obj = modules['custom_rn'].resnet18(pretrained=False)
        direct = {'fc': [obj.fc.in_features, obj.fc.out_features],
                  'devices': sorted({str(p.device) for p in obj.parameters()}),
                  'aggregate_parameter_sha256': aggregate(digests(obj.named_parameters()))}
        del obj
    direct_after = sha(torch.get_rng_state().numpy().tobytes())
    require(direct_after == hr['rng']['cpu_sha256'], 'direct replay CPU RNG state == helper CPU RNG state')
    require(direct['aggregate_parameter_sha256'] == rr['object']['aggregate_parameter_sha256'] and direct['fc'] == [512, 17],
            'direct replay object == helper throwaway (17-way head)')
    # replay 2: DataLoader order from the post-eval state; replay 3: bypass control (no constructor)
    orders = {}
    for name, state in (('post_eval_replay', post_eval_cpu), ('bypass_control', captured['pre_helper_cpu_state'])):
        torch.set_rng_state(state)
        dataset.accessed = []
        next(iter(mg.build_dataloader(DataLoader, dataset, args)))
        orders[name] = list(dataset.accessed)
    probe.window = None
    require(probe.observed.get('replay_iterators') == 2, 'two replay iterators')
    require(orders['post_eval_replay'] == dataset_order, 'replay from the post-eval state reproduces the batch order')
    require(hc['rng']['cpu_sha256'] != rng['post_eval']['cpu_sha256'], 'bypass state differs from the post-helper state')
    return {'helper': 'methods.difffas.execution_policy.consume_upstream_encoder_loader_rng',
            'replays': 'models/unet_autoenc.py:74 model_autoencoder = resnet18()', 'constructor_calls': 1,
            'resnet_init_calls': 1, 'throwaway_object': rr['object'], 'throwaway_returned': False,
            'helper_return_value': hr['return_value'], 'rng_after_seed': rng['after_seed'],
            'rng_after_construction': rng['after_construction'],
            'rng_at_helper_call': hc['rng'], 'rng_at_helper_return': hr['rng'], 'rng_at_load_call': lc['rng'],
            'rng_post_eval': rng['post_eval'], 'rng_at_dataloader_iter': probe.observed['rng_at_dataloader_iter'],
            'rng_after_first_batch': probe.observed['rng_after_first_batch'], 'rng_after_iteration': rng['after_iteration'],
            'cuda_rng_before_time_t': rng['cuda_before_iteration'],
            'cuda_rng_after_randint': probe.observed['cuda_rng_after_randint'],
            'cpu_changed_by_constructor': True, 'cuda_unchanged_by_helper': True,
            'secure_load_changed_cpu_rng': False, 'secure_load_changed_cuda_rng': False,
            'cuda_memory_delta_during_helper_bytes': 0, 'surviving_17_way_objects': 0,
            'custom_resnets_after_eval': instances_after,
            'direct_replay': {'label': 'RNG_PROOF_ONLY_POST_HOC', **direct, 'cpu_after_sha256': direct_after,
                              'equal_to_helper': True},
            'dataloader_order': {'training_first_batch_indices': dataset_order,
                                 'post_eval_replay_indices': orders['post_eval_replay'],
                                 'bypass_control_indices': orders['bypass_control'],
                                 'bypass_control_order_differs': orders['bypass_control'] != dataset_order,
                                 'bypass_control_state_differs': True,
                                 'cpu_draws': 'DataLoader iterator _base_seed + RandomSampler seed (CPU default '
                                              'generator); nothing between encoder.eval() and the iterator consumes RNG'},
            'step_cpu_rng_consumption': 0,
            'manual_rng_draw_substitute': False}


def synthetic_facts(np, dataset, dataset_order, probe):
    table = {}
    for i in range(N_ITEMS):
        item = dataset.item(i)
        table[str(i)] = {r: {'uint8_sha256': sha(dataset.raw(i, r).tobytes()), 'tensor_sha256': tensor_sha(item[r])}
                         for r in ROLES}
    raw = [v[r]['uint8_sha256'] for v in table.values() for r in ROLES]
    require(len(set(raw)) == 3 * N_ITEMS, 'all 24 synthetic images distinct')
    require(probe.observed['batch_keys'] == sorted(ROLES) and
            all(s == [BATCH, 3, 256, 256] for s in probe.observed['batch_shapes'].values()) and
            set(probe.observed['batch_dtypes'].values()) == {'torch.float32'}, 'batch = 3 x [4,3,256,256] float32, no label')
    return {'label': 'SYNTHETIC_STRUCTURAL_QUALIFICATION_SUBSTITUTE', 'n_items': N_ITEMS, 'batch_size': BATCH,
            'keys': list(ROLES), 'label_tensor': False, 'generator': 'analytic sin/cos uint8 RGB 256x256, no RNG, no file',
            'transform': 'Resize((256,256)) -> ToTensor() -> Normalize([0.5]*3, [0.5]*3) (FAS_train.py:184-188)',
            'shuffle': True, 'num_workers': 0, 'iterators': 1, 'batches': 1, 'first_batch_indices': dataset_order,
            'batch_sha256': probe.observed['batch_sha256'], 'items': table,
            'not_claimed': ['A1_TRAIN_DATASET_INTEGRATION', 'CANONICAL_FACE_READER', 'GUIDE_MAPPING',
                            'PRODUCTION_DATA_PATH']}


def pre_backward(probe, out, dataset_order, model_calls, enc_outputs, rng, model_pre, ema_pre, enc_pre, a7rng):
    require(len(model_calls) == 1 and len(enc_outputs) == 1, 'one model forward and one encoder forward')
    call, eo = model_calls[0], enc_outputs[0]
    require(call['x_shape'] == [BATCH, 3, 256, 256] and call['output_shape'] == MAIN_OUTPUT_SHAPE and
            call['output_finite'] and call['prob'] == 1 - 0.2 and (call['means_size'], call['var_size']) == (5, 3),
            'model(x=x_t [4,3,256,256]) -> [4,6,256,256], prob 0.8')
    require(eo['input_shape'] == [BATCH, 3, 256, 256] and eo['shapes'] == list(ENCODER_SHAPES.values()) and
            set(eo['dtypes']) == {'torch.float32'}, 'A6 encoder feature shapes in the loss path')
    return {'batch_indices': dataset_order, 'batch_sha256': probe.observed['batch_sha256'],
            'time_t': probe.observed['time_t'], 'model_input': {k: call[k] for k in ('x_sha256', 't', 'cond_mask',
                                                                                     'x_cond_sha256', 'prob')},
            'model_output_sha256': call['output_sha256'], 'encoder_output_sha256': eo['sha256'],
            'encoder_output_shapes': eo['shapes'],
            'loss_hex': float(out['loss'].item()).hex(), 'mse_hex': float(out['loss_mse'].item()).hex(),
            'vb_hex': float(out['loss_vb'].item()).hex(),
            'loss_per_sample_sha256': tensor_sha(out['loss_dict']['loss']),
            'vb_per_sample_sha256': tensor_sha(out['loss_dict']['vb']),
            'content_after_inplace_zeroing_sha256': tensor_sha(out['content_after_loss']),
            'rng': {'after_seed': rng['after_seed'], 'after_construction': rng['after_construction'],
                    'post_eval': rng['post_eval'], 'after_first_batch': a7rng['rng_after_first_batch'],
                    'cuda_after_randint': a7rng['cuda_rng_after_randint']},
            'main_init_parameter_aggregate_sha256': aggregate(model_pre['parameters']),
            'ema_init_parameter_aggregate_sha256': aggregate(ema_pre['parameters']),
            'encoder_parameter_aggregate_sha256': aggregate(enc_pre['parameters'])}


def stream_tensors(fd, model, ema):
    """Anonymous-pipe transfer to the parent launcher for in-memory cross-process characterization; no file."""
    with os.fdopen(fd, 'wb') as pipe:
        for kind, pairs in (('grad', ((n, p.grad) for n, p in model.named_parameters())),
                            ('param', model.named_parameters()), ('ema', ema.named_parameters())):
            for name, t in pairs:
                if t is None:
                    pipe.write(json.dumps({'kind': kind, 'name': name, 'numel': -1}).encode() + b'\n')
                    continue
                raw = t.detach().float().cpu().contiguous().numpy().tobytes()
                pipe.write(json.dumps({'kind': kind, 'name': name, 'numel': t.numel()}).encode() + b'\n')
                pipe.write(raw)
        pipe.write(json.dumps({'end': True}).encode() + b'\n')


def process_main(args):
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-3:] == BUILD_PARTS and build.is_relative_to(rq.RUNTIME), 'dedicated M6D6i build path')
    require(args.output == f'process_{args.process}.json' and not (build / args.output).exists(), 'fresh output name')
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    try:
        result = qualify(build, args.process, rq.RUNTIME)
    except MemoryBlocked as exc:
        (build / args.output).write_text(json.dumps({'status': 'BLOCKED_BY_MAIN_B4_MEMORY', 'batch_size': BATCH,
                                                     'error': str(exc)[:2000]}, indent=2) + '\n')
        print(json.dumps({'status': 'BLOCKED_BY_MAIN_B4_MEMORY'}))
        sys.exit(3)
    model, ema = result.pop('_stream')
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if args.stream_fd is not None:
        stream_tensors(args.stream_fd, model, ema)
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))


# ============================================================================= launch mode (parent)
def nvidia(*query):
    return subprocess.check_output(['nvidia-smi', *query, '--format=csv,noheader,nounits'], text=True).strip()


def gpu_gate():
    apps = nvidia('--query-compute-apps=pid,used_memory')
    used = int(nvidia('--query-gpu=memory.used').splitlines()[0])
    require(apps == '' and used <= PRELAUNCH_MAX_USED_MIB, f'GPU idle before launch (apps={apps!r}, used={used} MiB)')
    return {'compute_apps': [], 'memory_used_mib': used, 'max_allowed_mib': PRELAUNCH_MAX_USED_MIB}


def checkpoint_identity(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return {**stat_identity(path), 'sha256': h.hexdigest()}


class StreamReader(threading.Thread):
    """Reads one child's tensor stream; process 1 is kept in memory, process 2 is compared record by record."""

    def __init__(self, fd, reference=None):
        super().__init__(daemon=True)
        self.fd, self.reference, self.store, self.error = fd, reference, {}, None
        self.stats = {k: {'tensors': 0, 'tensors_differing': 0, 'elements': 0, 'elements_differing': 0,
                          'max_abs_diff': 0.0, 'max_rel_diff': 0.0, 'top': []} for k in ('grad', 'param', 'ema')}
        self.none = {'grad': [], 'param': [], 'ema': []}
        self.ended = False

    def run(self):
        import numpy as np
        try:
            with os.fdopen(self.fd, 'rb') as pipe:
                while True:
                    head = json.loads(pipe.readline())
                    if head.get('end'):
                        self.ended = True
                        break
                    key = (head['kind'], head['name'])
                    if head['numel'] < 0:
                        self.none[head['kind']].append(head['name'])
                        continue
                    raw = pipe.read(4 * head['numel'])
                    require(len(raw) == 4 * head['numel'], 'complete tensor record')
                    if self.reference is None:
                        self.store[key] = raw
                        continue
                    a = np.frombuffer(self.reference.pop(key), dtype=np.float32).astype(np.float64)
                    b = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
                    s = self.stats[head['kind']]
                    differ = int(np.count_nonzero(a != b))
                    s['tensors'] += 1
                    s['elements'] += a.size
                    if differ:
                        d = np.abs(a - b)
                        den = np.maximum(np.abs(a), np.abs(b))
                        rel = float((d[den > 0] / den[den > 0]).max()) if (den > 0).any() else 0.0
                        s['tensors_differing'] += 1
                        s['elements_differing'] += differ
                        s['max_abs_diff'] = max(s['max_abs_diff'], float(d.max()))
                        s['max_rel_diff'] = max(s['max_rel_diff'], rel)
                        s['top'] = sorted(s['top'] + [[head['name'], differ, float(d.max())]],
                                          key=lambda r: -r[2])[:10]
        except Exception as exc:  # noqa: BLE001  reported and gated by the parent
            self.error = repr(exc)


def run_child(n, build, reference=None):
    r, w = os.pipe()
    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--mode', 'process', '--process', str(n),
           '--build-root', str(build), '--output', f'process_{n}.json', '--stream-fd', str(w)]
    started = time.time()
    proc = subprocess.Popen(cmd, pass_fds=(w,))
    os.close(w)
    reader = StreamReader(r, reference)
    reader.start()
    peak, samples = 0, 0
    while proc.poll() is None:
        try:
            for line in nvidia('--query-compute-apps=pid,used_memory').splitlines():
                pid, used = (x.strip() for x in line.split(','))
                if int(pid) == proc.pid:
                    peak = max(peak, int(used))
            samples += 1
        except (subprocess.CalledProcessError, ValueError):
            pass
        time.sleep(0.5)
    reader.join()
    return {'returncode': proc.returncode, 'pid': proc.pid, 'wall_seconds': round(time.time() - started, 3),
            'nvidia_smi_peak_process_mib': peak, 'nvidia_smi_samples': samples, 'command': cmd[1:]}, reader


def launch_main(args):
    build = args.build_root.resolve()
    require(build.parts[-3:] == BUILD_PARTS and build.is_relative_to(rq.RUNTIME) and not build.is_relative_to(ROOT),
            'dedicated M6D6i build path')
    require(not (build / 'main_graph_qualification.json').exists(), 'fresh aggregate')
    for k, v in LAUNCH_ENV.items():
        require(os.environ.get(k) == v, f'launch environment {k}={v}')
    frozen = seam.frozen_checkpoint_path(rq.RUNTIME)
    pre = checkpoint_identity(frozen)
    require((pre['sha256'], pre['size_bytes']) == (seam.OWNER_FROZEN_SHA256, seam.OWNER_FROZEN_BYTES),
            'frozen checkpoint SHA256/size before process 1')
    gates = {'before_process_1': gpu_gate()}
    run1, reader1 = run_child(1, build)
    require(run1['returncode'] == 0 and reader1.ended and reader1.error is None, 'process 1 PASS: ' + json.dumps(run1))
    gates['before_process_2'] = gpu_gate()
    run2, reader2 = run_child(2, build, reader1.store)
    require(run2['returncode'] == 0 and reader2.ended and reader2.error is None, 'process 2 PASS: ' + json.dumps(run2))
    require(not reader1.store, 'every process-1 tensor compared')
    post = checkpoint_identity(frozen)
    require(post == pre, 'frozen checkpoint unchanged (size, SHA256, inode, mtime_ns, ctime_ns, mode)')
    p = [json.loads((build / f'process_{i}.json').read_text()) for i in (1, 2)]
    for ev in p:
        require(ev['status'] == 'PASS' and ev['counters'] == EXPECTED_COUNTERS, 'per-process PASS and counters')
    require(p[0]['pre_backward'] == p[1]['pre_backward'], 'PRE-BACKWARD repeatability: processes differ')
    require(reader1.none == reader2.none and p[0]['gradients']['none_grad_tensors'] ==
            p[1]['gradients']['none_grad_tensors'] == reader1.none['grad'] and
            p[0]['gradients']['grad_presence_sha256'] == p[1]['gradients']['grad_presence_sha256'],
            'identical gradient presence / None structure')
    characterization = {k: {**v, 'max_rel_diff_definition': '|a-b| / max(|a|,|b|) over elements with max > 0'}
                        for k, v in reader2.stats.items()}
    fields = sorted(p[0]['pre_backward'])
    result = {
        'milestone': MILESTONE, 'classification': CLASSIFICATION, 'label': LABEL, 'status': 'PASS',
        'authority_commit': AUTHORITY, 'qualification_seed': SEED, 'experiment_seed': None,
        'experiment_seed_reason': EXPERIMENT_SEED_REASON, 'eligibility': ELIGIBILITY,
        'candidate_sha256': p[0]['candidate_sha256'], 'repository_head': p[0]['repository_head'],
        'frozen_checkpoint': {'path_template': p[0]['frozen_checkpoint']['path_template'], 'path': str(frozen),
                              'sha256': seam.OWNER_FROZEN_SHA256, 'bytes': seam.OWNER_FROZEN_BYTES,
                              'before_process_1': pre, 'after_process_2': post, 'unchanged': True,
                              'chmod_performed': False,
                              'identity_fields': ['size_bytes', 'sha256', 'inode', 'device', 'mode', 'nlink', 'uid',
                                                  'gid', 'mtime_ns', 'ctime_ns']},
        'gpu_prelaunch_gates': gates, 'launch_environment': LAUNCH_ENV,
        'processes': {str(i + 1): {'run': r, 'status': ev['status'], 'counters': ev['counters'],
                                   'loss_hex': ev['iteration']['loss_float32_hex'], 'mse_hex': ev['iteration']['mse_hex'],
                                   'vb_hex': ev['iteration']['vb_hex'], 'loss': ev['iteration']['loss'],
                                   'mse': ev['iteration']['mse'], 'vb': ev['iteration']['vb'],
                                   'peak_allocated_bytes': ev['cuda_memory']['peak_allocated_bytes'],
                                   'peak_reserved_bytes': ev['cuda_memory']['peak_reserved_bytes'],
                                   'evidence': f'process_{i + 1}.json'}
                      for i, (r, ev) in enumerate(((run1, p[0]), (run2, p[1])))},
        'pre_backward_repeatability': {'equal': True, 'fields_compared': fields,
                                       'rule': 'bitwise equality required through the forward/loss boundary (D1-A)'},
        'post_backward_characterization': {
            'label': 'CHARACTERIZATION_ONLY_NOT_AN_ACCEPTANCE_GATE', 'decision': 'OWNER D1-C',
            'gradient_presence_identical': True, 'none_grad_tensors': reader1.none['grad'],
            'grad': characterization['grad'], 'main_parameters_after_step': characterization['param'],
            'ema_parameters_after_step': characterization['ema'],
            'transport': 'anonymous pipe to the parent launcher; float32 bytes held in memory only; nothing written',
            'tolerance_applied': None,
            'known_nondeterministic_source_op': ('F.interpolate(mode="bicubic", align_corners=True) backward on CUDA '
                                                 '(models/blocks.py AttentionBlock._reshape_and_interpolate); A7 leaves '
                                                 'torch.use_deterministic_algorithms NOT_SET')},
        'qualification_counters': {'qualification_processes': 2, 'qualification_checkpoint_deserializations': 2,
                                   'qualification_backward_calls': 2, 'qualification_optimizer_steps': 2,
                                   'qualification_scheduler_steps': 2, 'qualification_ema_updates': 2},
        'scientific_counters': {'scientific_main_runs': 0, 'scientific_optimizer_steps': 0,
                                'scientific_checkpoint_deserializations': 0, 'scientific_checkpoint_writes': 0,
                                'experiment_seed_runs': 0},
        'batch_size': BATCH, 'torch_save_calls': 0, 'main_checkpoints_written': 0,
        'qualification_checkpoints_written': 0, 'TRAIN_access': False, 'VAL_access': False, 'TEST_access': False,
        'TRAIN_reads': 0, 'VAL_reads': 0, 'TEST_reads': 0, 'manifest_opens': 0, 'M8_outputs': 0, 'M8_bank': False,
        'firewall_denials': sum(len(ev['firewall']['denied']) for ev in p),
        'qualified_statuses': QUALIFIED, 'not_qualified': NOT_QUALIFIED, 'not_claimed': NOT_CLAIMED,
        'method_status': 'IMPLEMENTED_NOT_EXECUTED', 'fidelity_class': 'CONTROLLED_ADAPTATION', 'deviation': 'DEV-021',
        'production_runner_claimed': False, 'bitwise_deterministic_main_training_claimed': False}
    (build / 'main_graph_qualification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(build / 'main_graph_qualification.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('process', 'launch'), required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--process', type=int, choices=(1, 2))
    parser.add_argument('--output')
    parser.add_argument('--stream-fd', type=int)
    args = parser.parse_args()
    if args.mode == 'process':
        require(args.process in (1, 2), '--process 1|2')
        process_main(args)
    else:
        launch_main(args)


if __name__ == '__main__':
    main()
