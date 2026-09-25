"""M6D6a E07c runtime/architecture diagnostics, never a DiffFAS training runner.

Run in an isolated GPU process under gpat-m6-e07c. Binds the unchanged pinned
murphytju/DiffFAS modules, constructs the source-native custom_rn "resnet18"
conditioning encoder through the committed A3 seam (methods.difffas.encoder,
fc -> Linear(512, 7)) and the main BeatGANsAutoencModel through the pinned
config/diffconfig.py::get_model_conf, then runs synthetic forwards only.

The in-memory encoder is UNTRAINED and exists only to prove interface
compatibility; it is not, and never becomes, the future auxiliary checkpoint.
No dataset, manifest, image reader, optimizer, backward pass, checkpoint load or
checkpoint writer exists here. The diffusion loss is evaluated as
FORWARD_LOSS_PATH_ONLY under torch.no_grad.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import sysconfig
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNTIME = Path(str(ROOT) + '_runtime')
SEED = 60601  # qualification_seed only; never an experiment seed (42/1337/2026)
LABEL = 'SYNTHETIC_FORWARD_RUNTIME_ONLY'
BUILD_PARTS = ('builds', 'e07c_difffas')
BATCH = 4  # equals the frozen main batch_size; forward-only, NOT a training-memory qualification
EXPECTED_FEATURES = {'x32x32': [256, 32, 32], 'x16x16': [512, 16, 16], 'x8x8': [512, 8, 8]}  # A6, CHW
K = 7
UPSTREAM_MAIN = ('config.diffconfig', 'diffusion', 'models.unet_autoenc')
UPSTREAM_MAIN_ROOTS = ('config', 'diffusion', 'models')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff', '.gif'}
BLOCKED_SUFFIXES = IMAGE_SUFFIXES | {'.parquet', '.npy', '.npz', '.pth', '.pt', '.pkl', '.ckpt', '.h5',
                                     '.tar', '.safetensors'}
ALLOWED_SUBPROCESSES = ('git', 'nvidia-smi')
# torch CUDA init -> cuda.pathfinder -> ctypes.util.find_library runs this read-only linker-cache query.
ALLOWED_EXACT_ARGV = (['/sbin/ldconfig', '-p'],)
TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = None  # captured from the genuine signature before torch.load is forbidden


def require(value, message):
    if not value:
        raise RuntimeError('M6D6a gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


class Firewall:
    """Python audit hook: TRAIN/VAL/TEST data, manifests, images, weights and runs are unreachable.

    E07c needs no external runtime asset for this milestone, so the whole
    runtime root is denied except the dedicated build root. Writes are limited
    to the build root. Every attempt is classified; the gates require zero.
    Subprocesses are recorded and restricted to git metadata, nvidia-smi and the exact
    read-only `/sbin/ldconfig -p` query issued by ctypes.util.find_library.
    """
    CLASSES = ('benchmark_manifest', 'benchmark_image', 'runtime_data', 'scientific_run_root',
               'blocked_weight_or_array', 'write_outside_build')

    def __init__(self, build):
        self.build = str(build) + '/'
        self.denied = []
        self.events = {k: 0 for k in ('open', 'os.listdir', 'os.scandir', 'subprocess.Popen')}
        self.attempts = {k: 0 for k in self.CLASSES}
        self.subprocesses = []

    def classify(self, event, args):
        path = os.path.abspath(os.fsdecode(args[0]))
        parts = set(Path(path).parts)
        suffix = Path(path).suffix.lower()
        hits = []
        if 'manifests' in parts or suffix == '.parquet':
            hits.append('benchmark_manifest')
        if 'faces_256' in parts or 'frames' in parts or suffix in IMAGE_SUFFIXES:
            hits.append('benchmark_image')
        if any(path == p or path.startswith(p + '/') for p in
               (str(ROOT / 'data'), str(ROOT / 'cache'), str(RUNTIME / 'data'), str(RUNTIME / 'cache'))):
            hits.append('runtime_data')
        if any(path == p or path.startswith(p + '/') for p in (str(ROOT / 'runs'), str(RUNTIME / 'runs'))):
            hits.append('scientific_run_root')
        if suffix in BLOCKED_SUFFIXES - IMAGE_SUFFIXES - {'.parquet'}:
            hits.append('blocked_weight_or_array')
        in_runtime = path == str(RUNTIME) or path.startswith(str(RUNTIME) + '/')
        if in_runtime and not path.startswith(self.build) and 'runtime_data' not in hits \
                and 'scientific_run_root' not in hits:
            hits.append('runtime_data')
        if event == 'open':
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) \
                or set(mode) & set('wax+')
            if writing and not (path.startswith(self.build) or path == '/dev/null'):
                hits.append('write_outside_build')
        return path, hits

    def __call__(self, event, args):
        if event == 'subprocess.Popen':
            self.events[event] += 1
            argv = [os.fsdecode(a) for a in (args[1] or [])] if len(args) > 1 else []
            exe = Path(argv[0]).name if argv else ''
            self.subprocesses.append(argv[:4])
            if exe not in ALLOWED_SUBPROCESSES and argv not in ALLOWED_EXACT_ARGV:
                self.denied.append({'event': event, 'path': ' '.join(argv[:4]), 'classes': ['subprocess']})
                raise RuntimeError('SUBPROCESS FIREWALL: ' + exe)
            return
        if event not in self.events or not args or not isinstance(args[0], (str, bytes)):
            return
        self.events[event] += 1
        path, hits = self.classify(event, args)
        if hits:
            for h in hits:
                self.attempts[h] += 1
            self.denied.append({'event': event, 'path': path, 'classes': hits})
            raise RuntimeError('DATA/WRITE FIREWALL: ' + path)


def environment():
    import numpy
    import torch
    import torchvision
    names = ('tensorfn', 'pydantic', 'pyhocon', 'tqdm', 'opencv-python-headless', 'pillow', 'scipy',
             'boto3', 'rich', 'termcolor', 'tabulate', 'pyparsing')
    packages = dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions(
        path=sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}))))
    lower = {k.lower(): v for k, v in packages.items()}
    return {'python': sys.version, 'executable': sys.executable,
        'python_sha256': sha(Path(sys.executable).resolve().read_bytes()),
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': numpy.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'runtime_packages': {n: lower.get(n) for n in names},
        'scipy_note': 'not installed: absent from the pinned source import closure (no scipy import in any DiffFAS file)',
        'packages': packages,
        'compiler': 'NOT_USED: no extension build; pinned DiffFAS uses standard torch.nn/functional operators only',
        'gpu_driver': command('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'),
        'gpu_capability': list(torch.cuda.get_device_capability()),
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_device_count': torch.cuda.device_count(),
        'torch_load_weights_only_default': TORCH_LOAD_WEIGHTS_ONLY_DEFAULT,
        'precision': {'default_dtype': str(torch.get_default_dtype()),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'autocast_cuda': torch.is_autocast_enabled('cuda'),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms_forced': torch.are_deterministic_algorithms_enabled()},
        'launch_environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE',
            'PYTHONNOUSERSITE', 'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'TMPDIR')}}


def summarize(t, *, device='cuda'):
    import torch
    require(t.dtype in (torch.float32, torch.float64), 'floating tensor')
    require(t.device.type == device, f'{device} tensor placement')
    require(torch.isfinite(t).all().item(), 'finite tensor')
    a = t.detach().cpu().contiguous().numpy()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device), 'finite': True,
            'sha256': sha(a.tobytes()), 'min': float(a.min()), 'max': float(a.max()),
            'mean': float(a.mean(dtype='float64')), 'l2': float((a.astype('float64') ** 2).sum() ** .5)}


def state_report(model):
    params = list(model.named_parameters())
    buffers = list(model.named_buffers())
    digest = {n: sha(p.detach().cpu().numpy().tobytes()) for n, p in params}
    bdigest = {n: sha(b.detach().cpu().numpy().tobytes()) for n, b in buffers}
    return {'parameters': sum(p.numel() for _, p in params), 'parameter_tensors': len(params),
            'trainable_parameters': sum(p.numel() for _, p in params if p.requires_grad),
            'buffers': len(buffers), 'devices': sorted({str(p.device) for _, p in params}),
            'dtypes': sorted({str(p.dtype) for _, p in params}),
            'shapes': {n: list(p.shape) for n, p in params},
            'parameter_sha256': digest, 'buffer_sha256': bdigest,
            'aggregate_parameter_sha256': sha(json.dumps(digest, sort_keys=True).encode()),
            'aggregate_buffer_sha256': sha(json.dumps(bdigest, sort_keys=True).encode())}


def synthetic(torch, b, phase):
    """Deterministic analytic FP32 images in [-1, 1] (the frozen Normalize(0.5, 0.5) range).

    Per-row frequencies/phases make every row distinct. CPU float64 construction,
    no RNG, file, decoder or manifest.
    """
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 256, dtype=torch.float64),
                            torch.linspace(0, 1, 256, dtype=torch.float64), indexing='ij')
    rows = []
    for i in range(b):
        fx, fy, ph = 1 + (i + phase) % 5, 1 + (3 * i + phase) % 7, 0.7 * i + phase
        rows.append(torch.stack((torch.sin(6.283 * fx * xx + ph) * torch.cos(6.283 * fy * yy),
                                 torch.cos(6.283 * fy * xx - ph) * (2 * yy - 1),
                                 torch.sin(6.283 * (fx * xx + fy * yy) + 2 * ph))))
    return torch.stack(rows).clamp(-1, 1).to(torch.float32)


def contract(adapter, contracts):
    """Static frozen-contract checks (section 14 of the M6D6a brief); nothing executes them."""
    from methods.difffas.sampler import sampler_plan
    from methods.difffas.encoder import encoder_plan
    cfg = adapter.config
    t, d, o, s = cfg['training'], cfg['diffusion'], cfg['optimizer'], cfg['sampler']
    aux = contracts['effective_encoder']
    tc = aux['training_contract']
    require(cfg['seeds']['experiment_seeds'] == [42, 1337, 2026], 'scientific seeds')
    require((t['max_epochs'], t['batch_size'], t['input_resolution'], t['use_pair'], t['model_in_channels'],
             t['guidance_prob'], t['means_size'], t['var_size']) == (400, 4, 256, False, 3, 0.2, 5, 3),
            'frozen main training')
    require((d['schedule'], d['n_timestep'], d['linear_start'], d['linear_end'], d['model_mean_type'],
             d['variance']) == ('linear', 1000, 1e-4, 2e-2, 'EPSILON', 'LEARNED_RANGE'), 'frozen diffusion')
    require((o['name'], o['learning_rate']) == ('AdamW', 1e-5) and o['scheduler'] ==
            {'type': 'cycle', 'lr': 1e-5, 'n_iter': 2400000, 'warmup': 5000, 'decay': ['linear', 'flat']} and
            o['ema'] == {'decay': 0.9999, 'before_warmup': 0.0}, 'frozen optimizer/scheduler/EMA')
    require((s['algorithm'], s['DDIM_skip'], s['sample_initial_noise'], s['effective_steps'], s['cond_scale'],
             s['tensor_set_loaded']) == ('ddim', 10, 250, 25, 2.0, 'model'), 'frozen sampler')
    require((aux['auxiliary_encoder_training_runs'], aux['auxiliary_encoder_training_seed'],
             aux['same_frozen_encoder_reused_for_main_seeds'], aux['trained_exactly_once']) ==
            (1, 42, [42, 1337, 2026], True), 'one auxiliary encoder')
    require((tc['head'], tc['epochs'], tc['batch_size'], tc['drop_last'], tc['shuffle'], tc['workers'],
             tc['augmentation'], tc['class_balancing'], tc['loss'], tc['optimizer'], tc['learning_rate'],
             tc['momentum'], tc['weight_decay'], tc['scheduler'], tc['checkpoint_rule']) ==
            ('fc = nn.Linear(512, 7)', 200, 256, True, True, 6, 'NONE', 'NONE',
             'CrossEntropyLoss_on_fourth_forward_output', 'SGD', 0.002, 0.9, 5e-3, 'NONE',
             'FINAL_STATE_AFTER_EPOCH_200'), 'frozen auxiliary training contract (not executed)')
    obj = aux['substitute_objective']
    require(obj['K'] == K and obj['classes'] == ['live', 'makeup', 'mask_2d', 'mask_3d', 'partial', 'print',
            'replay'] and obj['excluded'] == ['other_spoof'] and obj['split'] == 'TRAIN_ONLY', 'K7 objective')
    require(cfg['checkpoint']['rule'] == 'BASELINE_FINAL_STATE_V1' and
            cfg['conditioning_encoder']['checkpoint_format'].startswith('torch.save of the WHOLE nn.Module'),
            'checkpoint rules')
    require(cfg['fidelity_class'] == 'CONTROLLED_ADAPTATION' and cfg['deviation'] == 'DEV-021', 'fidelity')
    require(cfg['data']['splits']['TEST']['allowed'] is False, 'TEST forbidden')
    effective = {k: v['shape'] for k, v in aux['feature_interface_at_256'].items()}
    historical = {k: v['shape'] for k, v in cfg['conditioning_encoder']['feature_interface_at_256'].items()}
    require(effective == {'x32x32': [32, 32, 256], 'x16x16': [16, 16, 512], 'x8x8': [8, 8, 512],
                          'embg': ['B', 7]}, 'A6 effective interface')
    sampler = sampler_plan(cfg)
    plan = encoder_plan(cfg, contracts)
    require(not plan['trained'] and plan['checkpoint']['sha256_status'] == 'RECORDED_AFTER_TRAINING',
            'auxiliary encoder remains untrained')
    return {'experiment_seeds': cfg['seeds']['experiment_seeds'],
            'main_training': {k: t[k] for k in ('max_epochs', 'batch_size', 'input_resolution', 'use_pair',
                              'model_in_channels', 'guidance_prob', 'means_size', 'var_size')},
            'diffusion': dict(d), 'optimizer': dict(o), 'sampler': dict(s),
            'sampler_timesteps_ascending': sampler['timesteps_ascending'],
            'authoritative_generation_tensor_set': s['tensor_set_loaded'],
            'auxiliary_training_contract': dict(tc), 'auxiliary_objective_classes': obj['classes'],
            'auxiliary_seed': aux['auxiliary_encoder_training_seed'],
            'auxiliary_runs': aux['auxiliary_encoder_training_runs'],
            'reused_for_main_seeds': aux['same_frozen_encoder_reused_for_main_seeds'],
            'checkpoint_rule': cfg['checkpoint']['rule'],
            'feature_interface_effective_A6_HWC': effective,
            'feature_interface_historical_frozen_HWC': historical,
            'fidelity_class': cfg['fidelity_class'], 'deviation': cfg['deviation'],
            'contract_inspection': 'STATIC_ONLY_NOT_EXECUTED'}


def source_identity():
    from methods.difffas import DiffFASAdapter
    from methods.difffas.contract import validate_contract
    from methods.difffas.source import verify_executable_semantics
    adapter = DiffFASAdapter()
    contracts = validate_contract(adapter.config)
    source = adapter.validate_source()
    semantics = verify_executable_semantics(source, contracts)
    root = Path(source['root'])
    source['worktree_status'] = command('git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all')
    source['shallow'] = command('git', '-C', str(root), 'rev-parse', '--is-shallow-repository')
    source['a6_overlay_sha256'] = contracts['overlay_sha256']
    source['documents'] = contracts['documents']
    return source, semantics, adapter, contracts


class ObservedEncoder:
    """Transparent call observer: forwards the same input to the same module and returns
    the same four tensors unchanged; optional diagnostic replacement of ONE output."""

    def __init__(self, module, replace=None):
        self.module, self.replace, self.calls = module, replace, []

    def __call__(self, x):
        out = list(self.module(x))
        self.calls.append({'input_shape': list(x.shape), 'output_shapes': [list(o.shape) for o in out],
                           'output_ptrs': [o.data_ptr() for o in out]})
        if self.replace is not None:
            index, fn = self.replace
            out[index] = fn(out[index])
        return tuple(out)


def qualify(build):
    firewall = Firewall(build)
    sys.addaudithook(firewall)
    source, semantics, adapter, contracts = source_identity()
    result = {'qualification_seed': SEED, 'experiment_seed': None,
              'experiment_seed_reason': 'M6D6a is engineering qualification; 42/1337/2026 are never used',
              'label': LABEL, 'batch_size_used_for_qualification': BATCH,
              'contract': contract(adapter, contracts), 'source_before': source,
              'executable_semantics': semantics}
    import numpy as np
    import torch
    from methods.common.upstream import upstream_modules
    from methods.difffas.encoder import encoder_model
    global TORCH_LOAD_WEIGHTS_ONLY_DEFAULT
    TORCH_LOAD_WEIGHTS_ONLY_DEFAULT = repr(inspect.signature(torch.load).parameters['weights_only'].default)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cudnn = adapter.config['training']['cudnn']
    torch.backends.cudnn.benchmark = cudnn['benchmark']          # frozen: False
    torch.backends.cudnn.deterministic = cudnn['deterministic']  # frozen: True
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    counters = {'optimizer_constructions': 0, 'optimizer_applications': 0, 'backward_passes': 0,
                'autograd_grad_calls': 0, 'checkpoint_saves': 0, 'checkpoint_loads': 0}

    def forbid(name):
        def forbidden(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('M6D6a forbids ' + name)
        return forbidden
    torch.optim.Optimizer.__init__ = forbid('optimizer_constructions')
    for cls in {c for c in vars(torch.optim).values() if isinstance(c, type) and
                issubclass(c, torch.optim.Optimizer)}:
        cls.step = forbid('optimizer_applications')
    torch.Tensor.backward = forbid('backward_passes')
    torch.autograd.backward = forbid('backward_passes')
    torch.autograd.grad = forbid('autograd_grad_calls')
    torch.save = forbid('checkpoint_saves')
    torch.load = forbid('checkpoint_loads')
    before = environment()
    result['environment_before'] = before
    src_root = Path(source['root'])
    caught = []
    with warnings.catch_warnings(record=True) as caught, encoder_model(adapter.config) as encoder:
        warnings.simplefilter('always')
        # ---------------------------------------------------------------- A. conditioning encoder
        enc_cls = type(encoder)
        enc_file = Path(sys.modules[enc_cls.__module__].__file__).resolve()
        require(enc_file == (src_root / 'models/custom_rn.py').resolve(), 'encoder class from pinned custom_rn')
        require(sha(enc_file.read_bytes()) == contracts['overlay']['source']['sha256'], 'custom_rn SHA256 (A6)')
        import torchvision
        require(not isinstance(encoder, torchvision.models.ResNet) and
                not enc_cls.__module__.startswith('torchvision'), 'no torchvision ResNet substitute')
        custom_rn = sys.modules[enc_cls.__module__]
        layers = [len(getattr(encoder, f'layer{i}')) for i in (1, 2, 3, 4)]
        blocks = {type(b).__name__ for i in (1, 2, 3, 4) for b in getattr(encoder, f'layer{i}')}
        widths = [getattr(encoder, f'layer{i}')[-1].conv2.out_channels for i in (1, 2, 3, 4)]
        require(layers == [3, 4, 6, 3] and blocks == {'BasicBlock'} and custom_rn.BasicBlock.expansion == 1,
                'pinned [3,4,6,3] BasicBlock topology')
        require(widths == [64, 256, 512, 512], 'pinned stage widths')
        require(type(encoder.fc) is torch.nn.Linear and (encoder.fc.in_features, encoder.fc.out_features) ==
                (512, K), 'A3 head Linear(512, 7)')
        with torch.random.fork_rng(devices=[]):
            reference = custom_rn.resnet18(pretrained=False)  # structure only; RNG isolated
        ours = {n: (type(m).__name__, [list(p.shape) for p in m.parameters(recurse=False)])
                for n, m in encoder.named_modules()}
        ref = {n: (type(m).__name__, [list(p.shape) for p in m.parameters(recurse=False)])
               for n, m in reference.named_modules()}
        differing = sorted(n for n in set(ours) | set(ref) if ours.get(n) != ref.get(n))
        require(differing == ['fc'] and ref['fc'] == ('Linear', [[17, 512], [17]]) and
                ours['fc'] == ('Linear', [[K, 512], [K]]), 'only the A3 head differs from pinned resnet18()')
        del reference
        encoder = encoder.cuda()   # upstream BeatGANsAutoencModel.encoder(): model.cuda()
        encoder.eval()             # FAS_train.py:35 encoder.eval()
        result['encoder'] = {'class': enc_cls.__name__, 'module': enc_cls.__module__,
            'factory': 'models/custom_rn.py::resnet18(pretrained=False)',
            'construction_seam': 'methods/difffas/encoder.py::encoder_model (committed A3 seam)',
            'file_sha256': sha(enc_file.read_bytes()), 'topology': layers, 'block': 'BasicBlock',
            'stage_widths': widths, 'head': {'type': 'Linear', 'in_features': 512, 'out_features': K},
            'pinned_default_head': {'type': 'Linear', 'in_features': 512, 'out_features': 17},
            'structural_difference_vs_pinned_resnet18': differing,
            'projection_or_adapter_modules_added': False, 'torchvision_substitute': False,
            'untrained_in_memory_only': True, 'weights_loaded': False, 'mode': 'eval',
            'state': state_report(encoder)}
        # ---------------------------------------------------------------- B. main DiffFAS model
        with upstream_modules(src_root, UPSTREAM_MAIN, UPSTREAM_MAIN_ROOTS) as modules:
            dc, diffusion_mod, ua = (modules[n] for n in UPSTREAM_MAIN)
            from tensorfn import load_config
            conf_path = src_root / 'config/diffusion.conf'
            diff_conf = load_config(dc.DiffusionConfig, str(conf_path), (), False)  # FAS_train.py:277
            model_conf = dc.get_model_conf()
            require(model_conf.in_channels == 3 and model_conf.image_size == 256 and
                    model_conf.out_channels == 6, 'pinned model conf (3-channel input, learned sigma)')
            model = model_conf.make_model()          # FAS_train.py:216 (use_pair False branch)
            model = model.to('cuda')
            require(type(model) is ua.BeatGANsAutoencModel, 'pinned BeatGANsAutoencModel')
            require(Path(sys.modules[type(model).__module__].__file__).resolve() ==
                    (src_root / 'models/unet_autoenc.py').resolve(), 'model class from pinned source')
            betas = diff_conf.diffusion.beta_schedule.make()      # FAS_train.py:234
            diffusion = diffusion_mod.create_gaussian_diffusion(betas, predict_xstart=False)  # :235
            require(diffusion.model_mean_type == diffusion_mod.ModelMeanType.EPSILON and
                    diffusion.model_var_type == diffusion_mod.ModelVarType.LEARNED_RANGE and
                    diffusion.loss_type == diffusion_mod.LossType.MSE and diffusion.num_timesteps == 1000,
                    'EPSILON / LEARNED_RANGE / MSE / 1000')
            require(betas.dtype == torch.float64 and float(betas[0]) == 1e-4 and abs(float(betas[-1]) - 2e-2) < 1e-15,
                    'linear betas 1e-4 -> 2e-2')
            first_conv = model.input_blocks[0][0]
            require(first_conv.in_channels == 3, 'main input path is 3-channel (use_pair=false)')
            attention = [(n, m.channels) for n, m in model.named_modules() if type(m).__name__ == 'AttentionBlock']
            conf_keys = ('image_size', 'in_channels', 'model_channels', 'out_channels', 'num_res_blocks',
                         'embed_channels', 'attention_resolutions', 'dropout', 'channel_mult', 'num_heads',
                         'resblock_updown', 'resnet_two_cond', 'enc_out_channels', 'use_checkpoint')
            result['main_model'] = {'class': type(model).__name__, 'module': type(model).__module__,
                'construction': 'config/diffconfig.py::get_model_conf().make_model() (FAS_train.py:216)',
                'file_sha256': sha((src_root / 'models/unet_autoenc.py').read_bytes()),
                'conf': {k: getattr(model_conf, k) for k in conf_keys},
                'first_conv_in_channels': first_conv.in_channels, 'use_pair': False,
                'attention_blocks': [{'name': n, 'channels': c} for n, c in attention],
                'ema_constructed': False,
                'ema_note': 'FAS_train.py:218 builds EMA with the identical constructor; not needed for forward qualification',
                'state_initial': state_report(model)}
            result['diffusion'] = {'config_loader': 'tensorfn.load_config(DiffusionConfig, config/diffusion.conf)',
                'beta_schedule_node': dict(diff_conf.diffusion.beta_schedule),
                'optimizer_config_parsed_not_constructed': diff_conf.training.optimizer.dict(),
                'scheduler_config_parsed_not_constructed': diff_conf.training.scheduler.dict(),
                'betas': summarize(betas, device='cpu'), 'num_timesteps': diffusion.num_timesteps,
                'model_mean_type': diffusion.model_mean_type.name, 'model_var_type': diffusion.model_var_type.name,
                'loss_type': diffusion.loss_type.name, 'rescale_timesteps': diffusion.rescale_timesteps,
                'predict_xstart': False}
            t_cfg = adapter.config['training']
            means, var = t_cfg['means_size'], t_cfg['var_size']
            # ------------------------------------------------------------ C. synthetic forwards
            content = synthetic(torch, BATCH, 0).cuda()      # batch['content']: inert under use_pair=false
            target_img = synthetic(torch, BATCH, 11).cuda()  # batch['GT'] (x_start)
            target_pose = synthetic(torch, BATCH, 23).cuda()  # batch['style_spoof'] (encoder input)
            require(len({sha(x.cpu().numpy().tobytes()) for x in (content, target_img, target_pose)}) == 3,
                    'distinct synthetic inputs')
            out, captured = {}, []

            def record(name, tensor, shape):
                require(list(tensor.shape) == shape, f'{name} shape {list(tensor.shape)} != {shape}')
                out[name] = summarize(tensor)

            def attention_hook(name):
                def hook(module, args):
                    x, cond = args[0], args[1]
                    captured.append({'block': name, 'x_channels': x.shape[1], 'x_hw': list(x.shape[2:]),
                                     'cond_shape': list(cond.shape), 'cond_ptr': cond.data_ptr()})
                return hook
            attention_names = [n for n, m in model.named_modules() if type(m).__name__ == 'AttentionBlock']
            hooks = [m.register_forward_pre_hook(attention_hook(n)) for n, m in model.named_modules()
                     if type(m).__name__ == 'AttentionBlock']
            # Every UNet stage output (input/middle/output blocks, final projection) and every
            # AttentionBlock output, so proofs do not rely on the zero-initialised final projection.
            traced = ([(f'input_blocks.{i}', m) for i, m in enumerate(model.input_blocks)] +
                      [('middle_block', model.middle_block)] +
                      [(f'output_blocks.{i}', m) for i, m in enumerate(model.output_blocks)] +
                      [('out', model.out)] + [(n, model.get_submodule(n)) for n in attention_names])
            stage = {}

            def output_hook(name):
                def hook(module, args, output):
                    stage[name] = output.detach().clone()
                return hook
            hooks += [m.register_forward_hook(output_hook(n)) for n, m in traced]
            model_inputs = []
            hooks.append(model.register_forward_pre_hook(
                lambda m, args, kwargs: model_inputs.append(list(kwargs['x'].shape)), with_kwargs=True))
            torch.cuda.synchronize()
            memory = {'after_construction_allocated_bytes': torch.cuda.memory_allocated(),
                      'after_construction_reserved_bytes': torch.cuda.memory_reserved()}
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                record('content', content, [BATCH, 3, 256, 256])
                record('target_img', target_img, [BATCH, 3, 256, 256])
                record('target_pose', target_pose, [BATCH, 3, 256, 256])
                # C1. encoder interface, direct
                f32, f16, f8, embg = encoder(target_pose)
                for name, tensor in (('x32x32', f32), ('x16x16', f16), ('x8x8', f8)):
                    record('encoder_' + name, tensor, [BATCH, *EXPECTED_FEATURES[name]])
                record('encoder_embg', embg, [BATCH, K])
                # C2. eval-mode connectivity / discard counterfactuals (dropout off, deterministic)
                model.eval()
                t_fixed = torch.tensor([0, 250, 500, 999], device='cuda')
                noise = torch.randn_like(target_img)
                x_t = diffusion.q_sample(target_img, t_fixed, noise=noise)
                cond_mask = torch.ones(BATCH, dtype=torch.bool, device='cuda')

                def forward(enc):
                    captured.clear()
                    stage.clear()
                    y = model(x=torch.cat([x_t], 1), encoder=enc, t=diffusion._scale_timesteps(t_fixed),
                              cond_mask=cond_mask, x_cond=target_pose, prob=1, means_size=means, var_size=var)
                    return y, list(captured), dict(stage)
                observed = ObservedEncoder(encoder)
                y_ref, att_ref, stage_ref = forward(observed)
                record('eval_model_output', y_ref, [BATCH, 6, 256, 256])
                zero_output = bool((y_ref == 0).all())
                require(zero_output == all(bool((p == 0).all()) for p in model.out[-1].parameters()),
                        'final output is identically zero exactly when the pinned zero_module projection is zero')
                require(set(stage_ref) == {n for n, _ in traced}, 'every traced stage executed')
                for n, v in stage_ref.items():
                    require(torch.isfinite(v).all().item(), 'finite internal activation ' + n)
                internal = {n: summarize(v) for n, v in stage_ref.items()}
                ptrs = dict(zip(('x32x32', 'x16x16', 'x8x8', 'embg'), observed.calls[0]['output_ptrs']))
                consumed = {name: [a for a in att_ref if a['cond_ptr'] == ptrs[name]] for name in ptrs}
                require(len(observed.calls) == 1 and observed.calls[0]['input_shape'] == [BATCH, 3, 256, 256],
                        'encoder called once on x_cond')
                require(not consumed['embg'], 'fourth encoder output never reaches any AttentionBlock')
                for name, (c, h, w) in EXPECTED_FEATURES.items():
                    require(consumed[name] and all(a['x_channels'] == c and a['x_hw'] == [h, w] and
                            a['cond_shape'] == [BATCH, c, h, w] for a in consumed[name]),
                            f'{name} consumed by AttentionBlocks with matching {c} channels at {h}x{w}')
                require(sum(len(v) for v in consumed.values()) == len(att_ref) == len(attention_names),
                        'every AttentionBlock cond is one of the three encoder features')
                y_repeat, _, stage_repeat = forward(ObservedEncoder(encoder))
                require(torch.equal(y_ref, y_repeat) and all(torch.equal(stage_ref[n], stage_repeat[n])
                        for n in stage_ref), 'eval forward repeat bitwise equal at every traced stage')
                y_nan, _, stage_nan = forward(ObservedEncoder(encoder, (3, lambda e: torch.full_like(e, float('nan')))))
                require(torch.isfinite(y_nan).all().item() and torch.equal(y_nan, y_ref) and
                        all(torch.equal(stage_ref[n], stage_nan[n]) for n in stage_ref),
                        'NaN-filled fourth output leaves every traced stage bitwise unchanged')
                sensitivity = {}
                for index, name in enumerate(('x32x32', 'x16x16', 'x8x8')):
                    _, _, stage_zero = forward(ObservedEncoder(encoder, (index, torch.zeros_like)))
                    blocks = [a['block'] for a in consumed[name]]
                    sensitivity[name] = {b: float((stage_zero[b] - stage_ref[b]).abs().max()) for b in blocks}
                    require(all(v > 0 for v in sensitivity[name].values()),
                            f'zeroing {name} changes the output of every AttentionBlock consuming it')
                eval_peak = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                # C3. FORWARD_LOSS_PATH_ONLY in FAS_train.py:50-68 call form (model train mode, encoder eval)
                model.train()
                torch.cuda.reset_peak_memory_stats()
                time_t = torch.randint(0, diff_conf.diffusion.beta_schedule['n_timestep'], (BATCH,), device='cuda')
                model_inputs.clear()
                loss_encoder = ObservedEncoder(encoder)
                loss_dict = diffusion.training_losses(
                    model, loss_encoder, x_start=target_img, t=time_t, betas=betas.cuda(),
                    cond_input=[content.clone(), target_pose], prob=1 - t_cfg['guidance_prob'],
                    means_size=means, var_size=var, use_pair=t_cfg['use_pair'])
                require(model_inputs == [[BATCH, 3, 256, 256]], 'use_pair=false 3-channel model input in the loss path')
                require(len(loss_encoder.calls) == 1, 'encoder called once in the loss path')
                record('loss_mse', loss_dict['mse'].reshape(1), [1])
                record('loss_vb', loss_dict['vb'], [BATCH])
                record('loss_per_sample', loss_dict['loss'], [BATCH])
                loss_mean = loss_dict['loss'].mean()
                record('loss_mean', loss_mean.reshape(1), [1])
                require(not loss_mean.requires_grad and loss_mean.grad_fn is None, 'no autograd graph (no_grad)')
                loss_peak = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            for h in hooks:
                h.remove()
            torch.cuda.synchronize()
            state_after = state_report(model)
            enc_after = state_report(encoder)
            initial = result['main_model']['state_initial']
            require(state_after['parameter_sha256'] == initial['parameter_sha256'], 'main parameters unchanged')
            require(enc_after['parameter_sha256'] == result['encoder']['state']['parameter_sha256'] and
                    enc_after['buffer_sha256'] == result['encoder']['state']['buffer_sha256'],
                    'encoder parameters and buffers unchanged (eval)')
            changed_buffers = sorted(n for n in initial['buffer_sha256']
                                     if initial['buffer_sha256'][n] != state_after['buffer_sha256'][n])
            result['main_model']['state_after_parameters_unchanged'] = True
            result['main_model']['buffers_changed_by_train_mode_forward'] = changed_buffers
            result['main_model']['buffer_change_note'] = (
                'Native BatchNorm2d running statistics of AttentionBlock.bn update during the train-mode forward '
                '(FAS_train keeps the model in train mode); no parameter changes and no optimizer exists.')
            result['forward'] = out
            result['finite_all'] = all(v['finite'] for v in out.values())
            result['feature_interface_live_CHW'] = {k: out['encoder_' + k]['shape'][1:] for k in
                                                    ('x32x32', 'x16x16', 'x8x8')} | {'embg': out['encoder_embg']['shape'][1:]}
            result['connectivity'] = {
                'attention_block_calls_per_forward': len(att_ref),
                'feature_consumers': {k: [{'x_channels': a['x_channels'], 'x_hw': a['x_hw']} for a in v]
                                      for k, v in consumed.items()},
                'feature_consumer_blocks': {k: [a['block'] for a in v] for k, v in consumed.items()},
                'fourth_output_consumers': len(consumed['embg']),
                'fourth_output_discard_proof': ('NaN replacement of encoder output[3] -> every traced UNet stage '
                    '(input/middle/output blocks, AttentionBlocks, final projection) bitwise identical and finite'),
                'zeroing_sensitivity_max_abs_at_consumer_attention_blocks': sensitivity,
                'traced_stages': len(traced), 'internal_activations': internal,
                'final_output_identically_zero': zero_output,
                'final_output_zero_reason': ('pinned get_model_conf sets resnet_use_zero_module=True: the final '
                    'self.out convolution is zero-initialised, so an UNTRAINED model outputs exactly 0; proofs use '
                    'internal stages instead'),
                'eval_forward_repeat_bitwise_equal': True,
                'source_path': 'models/unet_autoenc.py:178 x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)',
                'diagnostic_replacements_are_harness_only': True}
            result['forward_loss_path'] = {'label': 'FORWARD_LOSS_PATH_ONLY', 'call': 'diffusion.training_losses '
                '(FAS_train.py:57-68 argument form) under torch.no_grad', 'model_mode': 'train', 'encoder_mode': 'eval',
                'prob': 1 - t_cfg['guidance_prob'], 'means_size': means, 'var_size': var, 'use_pair': False,
                'time_t': time_t.tolist(), 'mse_reduction': "F.mse_loss(reduction='elementwise_mean') -> mean (deprecated alias)",
                'loss_mean': float(loss_mean), 'backward_executed': False, 'training_graph_qualified': False}
            result['cuda_memory'] = {'label': 'SYNTHETIC_FORWARD_MEMORY_OBSERVATION', 'batch_size': BATCH, **memory,
                'eval_connectivity_forwards': eval_peak, 'forward_loss_path': loss_peak,
                'autograd': 'torch.no_grad; no activation graph retained',
                'qualifies_training_memory': False, 'qualifies_scientific_batch_size': False,
                'qualifies_400_epoch_feasibility': False}
    result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    result['source_after'] = source_identity()[0]
    require(result['source_after'] == source, 'source-cache integrity after execution')
    require(source['worktree_status'] == '', 'pinned source worktree clean (no bytecode or edits)')
    result['environment_after'] = environment()
    diff = {k: {'before': before[k], 'after': result['environment_after'][k]}
            for k in before if before[k] != result['environment_after'][k]}
    require(not diff, 'environment stable: ' + json.dumps(diff))
    require(not any(counters.values()), 'zero optimizer/backward/grad/checkpoint calls')
    require(not firewall.denied and not any(firewall.attempts.values()), 'firewall: zero denied accesses')
    require(all(Path(a[0]).name in ALLOWED_SUBPROCESSES or a in ALLOWED_EXACT_ARGV
                for a in firewall.subprocesses), 'subprocess allowlist')
    result.update(status='PASS', counters=counters, optimizer_constructed=False, optimizer_applications=0,
        backward_passes=0, checkpoint_created=False, checkpoint_loaded=False, training_launched=False,
        auxiliary_encoder_trained=False, benchmark_data_access=False, TRAIN_access=False, VAL_access=False,
        TEST_access=False, synthetic_bank=False, diffusion_sampling=False, training_graph_qualified=False,
        fidelity='CONTROLLED_ADAPTATION', compatibility_patch='NONE',
        firewall={'denied': firewall.denied, 'attempts': firewall.attempts, 'event_counts': firewall.events,
                  'subprocesses': firewall.subprocesses})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', default='process.json')
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-2:] == BUILD_PARTS, 'dedicated build path')
    require(Path(args.output).name == args.output and args.output.endswith('.json'), 'evidence filename')
    build.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get('TMPDIR', '/'))
    require(tmp.is_dir() and tmp.resolve().is_relative_to(build), 'TMPDIR inside the dedicated build root')
    result = qualify(build)
    (build / args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(build / args.output)}))


if __name__ == '__main__':
    main()
