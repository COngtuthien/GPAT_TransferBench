"""M6D5a E06c runtime/architecture diagnostics, never a DSDG training runner.

Run in an isolated GPU process. Binds the unchanged pinned FaceX-Zoo DSDG
networks/misc modules, loads the official LightCNN-29 v2 bytes exactly as
train_generator.py:75-84 does, and runs batch-one synthetic forwards.
No dataset, image reader, optimizer, backward pass or checkpoint writer exists here.
"""
import argparse
import hashlib
import importlib.metadata
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
SEED = 60501  # qualification only; not an experiment seed (42/1337/2026)
LABEL = 'DIAGNOSTIC_ARCHITECTURE_BATCH_ONLY'
BUILD_PARTS = ('builds', 'e06c_dsdg')
LIGHTCNN = Path('/home/student20261/workdir/GPAT_TransferBench_runtime/third_party_weights/'
                'lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar')
EXPECTED_PARAMETERS = {  # (parameters, parameter tensors); a mismatch is a STOP, never a fix
    'netE_nir': (13_790_048, 17), 'netE_vis': (11_692_640, 17),
    'netG': (19_887_494, 21), 'netCls': (129, 2), 'netIP': (10_475_872, 60)}
EXPECTED_CLASSES = {'netE_nir': 'Encoder_s', 'netE_vis': 'Encoder',
                    'netG': 'Decoder_s', 'netCls': 'Cls', 'netIP': 'network_29layers_v2'}
OPTIMIZER_OWNED = ('netE_nir', 'netE_vis', 'netG')  # train_generator.py:87; netCls excluded
UPSTREAM_MODULES = ('networks', 'misc.util')
UPSTREAM_ROOTS = ('networks', 'misc', 'data', 'train_generator')


def require(value, message):
    if not value:
        raise RuntimeError('M6D5a gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def contract(adapter):
    """Static frozen-contract checks; nothing here constructs an optimizer."""
    cfg = adapter.config
    t, o, l = cfg['training'], cfg['optimizer'], cfg['losses']
    require((t['input_resolution'], t['hdim'], t['attack_type'], t['all_epochs']) == (256, 128, 1, 200),
            'frozen training contract')
    require((l['lambda_mmd'], l['lambda_ip'], l['lambda_type'], l['lambda_ort'], l['lambda_pair'])
            == (50, 1000, 10, 1, 0), 'frozen loss coefficients')
    require((o['name'], o['learning_rate'], o['scope'], o['netCls_in_optimizer'])
            == ('Adam', 2e-4, 'netE_nir + netE_vis + netG', False), 'frozen optimizer contract')
    ip = cfg['identity_preserving_net']
    require(ip['architecture'] == 'LightCNN-29 v2' and ip['frozen'] and ip['eval_mode'], 'frozen netIP')
    require(cfg['seeds']['experiment_seeds'] == [42, 1337, 2026], 'frozen seeds')
    require(cfg['checkpoint']['rule'] == 'OFFICIAL_GENERATOR_EPOCH_200', 'checkpoint rule')
    require(cfg['fidelity_class'] == 'CONTROLLED_ADAPTATION' and cfg['deviation'] == 'DEV-020', 'fidelity')
    require(cfg['data']['splits']['TEST']['allowed'] is False, 'TEST forbidden')
    batch = adapter.validate_batch()
    try:
        adapter.validate_batch(physical_batch_size=1, gradient_accumulation_steps=240)
        refused = False
    except Exception:
        refused = True
    require(refused, 'batch-one accumulation must remain refused as a training batch')
    return {'input_resolution': t['input_resolution'], 'hdim': t['hdim'], 'attack_type': t['attack_type'],
            'all_epochs': t['all_epochs'], 'losses': {k: l[k] for k in
            ('lambda_mmd', 'lambda_ip', 'lambda_type', 'lambda_ort', 'lambda_pair')},
            'optimizer': {k: o[k] for k in ('name', 'learning_rate', 'scope', 'netCls_in_optimizer')},
            'experiment_seeds': cfg['seeds']['experiment_seeds'], 'checkpoint_rule': cfg['checkpoint']['rule'],
            'fidelity_class': cfg['fidelity_class'], 'deviation': cfg['deviation'],
            'physical_batch_policy': batch['execution_policy'],
            'physical_batch_required': batch['physical_batch_size'],
            'batch_one_training_refused_by_adapter': refused,
            'optimizer_contract_inspection': 'STATIC_ONLY'}


def source_identity():
    from methods.dsdg import DSDGAdapter
    adapter = DSDGAdapter()
    source = adapter.validate_source()
    root = Path(source['root'])
    source['worktree_status'] = command('git', '-C', str(root), 'status', '--porcelain')
    source['sparse_checkout'] = command('git', '-C', str(root), 'sparse-checkout', 'list')
    return source, adapter


def asset_identity(adapter):
    asset, = adapter.config['external_assets']
    require(LIGHTCNN.is_file() and not LIGHTCNN.resolve().is_relative_to(ROOT), 'external LightCNN path')
    raw = LIGHTCNN.read_bytes()
    require(len(raw) == asset['bytes'] == 123844849, 'LightCNN byte size')
    require(sha(raw) == asset['sha256'] ==
            'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964', 'LightCNN SHA256')
    return {'path': str(LIGHTCNN), 'bytes': len(raw), 'sha256': sha(raw),
            'config_external_runtime_path_laptop': asset['external_runtime_path']}


class Firewall:
    """Python audit hook: no benchmark roots, manifests, images or other weights.

    The only readable weight file is the exact LightCNN asset. Writes are limited
    to the dedicated external build root. Git child processes are not intercepted.
    """
    def __init__(self, build):
        self.build = str(build) + '/'
        self.denied = []
        self.events = {k: 0 for k in ('open', 'os.listdir', 'os.scandir')}
        self.lightcnn_opens = 0

    def __call__(self, event, args):
        if event not in self.events or not args or not isinstance(args[0], (str, bytes)):
            return
        self.events[event] += 1
        path = os.path.abspath(os.fsdecode(args[0]))
        if event == 'open' and path == str(LIGHTCNN):
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else 'r'
            if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) \
                    and not set(mode) & set('wax+'):
                self.lightcnn_opens += 1
                return
        denied = bool(set(Path(path).parts) & {'faces_256', 'manifests'})
        for root in (str(ROOT), str(ROOT) + '_runtime'):
            denied |= any(path == root + '/' + p or path.startswith(root + '/' + p + '/')
                          for p in ('data', 'cache', 'runs'))
        denied |= Path(path).suffix.lower() in {'.parquet', '.jpg', '.jpeg', '.png', '.webp', '.bmp',
            '.npy', '.npz', '.pth', '.pt', '.ckpt', '.h5', '.pkl', '.tar'}
        if event == 'open':
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) \
                or set(mode) & set('wax+')
            denied |= bool(writing and not (path.startswith(self.build) or path == '/dev/null'))
        if denied:
            self.denied.append({'event': event, 'path': path})
            raise RuntimeError('DATA/WRITE FIREWALL: ' + path)


def environment():
    import numpy
    import PIL
    import torch
    import torchvision
    return {'python': sys.version, 'executable': sys.executable,
        'python_sha256': sha(Path(sys.executable).read_bytes()),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': numpy.__version__, 'pillow': PIL.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions(
            path=sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})))),
        'compiler': 'NOT_USED: no extension build; pinned DSDG uses standard torch.nn operators only',
        'gpu_driver': command('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'),
        'gpu_capability': list(torch.cuda.get_device_capability()),
        'cuda_device_count': torch.cuda.device_count(),
        'precision': {'default_dtype': str(torch.get_default_dtype()),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'autocast_cuda': torch.is_autocast_enabled('cuda'),
            'matmul_precision': torch.get_float32_matmul_precision()},
        'launch_environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE',
            'PYTHONNOUSERSITE', 'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'TMPDIR')}}


def summarize(t):
    import torch
    require(t.dtype == torch.float32 and t.is_cuda, 'FP32 CUDA tensor')
    require(torch.isfinite(t).all().item(), 'finite tensor')
    a = t.detach().cpu().contiguous().numpy()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'finite': True, 'sha256': sha(a.tobytes()),
            'min': float(a.min()), 'max': float(a.max()), 'mean': float(a.mean(dtype='float64')),
            'l2': float((a.astype('float64') ** 2).sum() ** .5)}


def parameter_report(model):
    params = list(model.named_parameters())
    return {'parameters': sum(p.numel() for _, p in params), 'parameter_tensors': len(params),
            'buffers': len(list(model.buffers())),
            'requires_grad_parameters': sum(p.numel() for _, p in params if p.requires_grad),
            'shapes': {n: list(p.shape) for n, p in params},
            'sha256': {n: sha(p.detach().cpu().numpy().tobytes()) for n, p in params}}


def load_lightcnn(torch, netIP):
    """Exact train_generator.py:75-84 filter/update/load/freeze sequence.

    torch>=2.6 defaults torch.load to weights_only=True; it is passed explicitly.
    The legacy file needs only OrderedDict/FloatStorage/_rebuild_tensor (M6A4), so
    the tensor values are identical and no pickled code can execute.
    """
    checkpoint = torch.load(str(LIGHTCNN), weights_only=True)
    pretrained_dict = checkpoint['state_dict']
    model_dict = netIP.state_dict()
    matched = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    shape_mismatch = sorted(k for k, v in matched.items() if tuple(v.shape) != tuple(model_dict[k].shape))
    report = {'top_level_keys': sorted(checkpoint), 'checkpoint_tensor_count': len(pretrained_dict),
        'expected_model_tensor_count': len(model_dict), 'matched': len(matched),
        'missing': sorted(set(model_dict) - set(matched)), 'shape_mismatches': shape_mismatch,
        'filtered_extra_keys': {k: list(pretrained_dict[k].shape) for k in pretrained_dict if k not in model_dict},
        'non_tensor_entries': {k: checkpoint[k] for k in ('epoch', 'arch', 'prec1')},
        'loader': 'train_generator.py:75-84 filter k in model_dict -> update -> load_state_dict -> requires_grad=False',
        'torch_load_weights_only': True}
    require((report['checkpoint_tensor_count'], report['expected_model_tensor_count'], report['matched'])
            == (61, 60, 60), 'LightCNN 60/60 compatibility')
    require(not report['missing'] and not shape_mismatch, 'LightCNN missing/shape mismatch')
    require(list(report['filtered_extra_keys']) == ['module.fc2.weight'], 'only module.fc2.weight filtered')
    model_dict.update(matched)
    netIP.load_state_dict(model_dict)
    for param in netIP.parameters():
        param.requires_grad = False
    loaded = netIP.state_dict()
    report['loaded_values_equal_checkpoint'] = all(
        torch.equal(loaded[k].cpu(), matched[k].cpu()) for k in matched)
    require(report['loaded_values_equal_checkpoint'], 'loaded LightCNN values')
    return report


def qualify(build):
    firewall = Firewall(build)
    sys.addaudithook(firewall)
    source, adapter = source_identity()
    result = {'diagnostic_seed': SEED, 'label': LABEL, 'contract': contract(adapter),
              'source_before': source, 'asset_before': asset_identity(adapter)}
    import numpy as np
    import torch
    import torch.nn.functional as F
    from methods.common.upstream import upstream_modules
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    counters = {'optimizer_constructions': 0, 'optimizer_applications': 0, 'backward_passes': 0,
                'checkpoint_saves': 0}

    def forbid(name):
        def forbidden(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('M6D5a forbids ' + name)
        return forbidden
    torch.optim.Optimizer.__init__ = forbid('optimizer_constructions')
    torch.optim.Optimizer.step = forbid('optimizer_applications')
    torch.optim.Adam.step = forbid('optimizer_applications')
    torch.Tensor.backward = forbid('backward_passes')
    torch.autograd.backward = forbid('backward_passes')
    torch.save = forbid('checkpoint_saves')
    before = environment()
    result['environment_before'] = before
    torch.cuda.reset_peak_memory_stats()
    root = Path(source['root']) / adapter.config['source']['relevant_path']
    caught = []
    with warnings.catch_warnings(record=True) as caught, \
            upstream_modules(root, UPSTREAM_MODULES, UPSTREAM_ROOTS) as modules:
        warnings.simplefilter('always')
        networks, util = modules['networks'], modules['misc.util']
        t = adapter.config['training']
        netE_nir, netE_vis, netG, netCls = networks.define_G(hdim=t['hdim'], attack_type=t['attack_type'])
        netIP = networks.define_IP(is_train=False)
        models = dict(netE_nir=netE_nir, netE_vis=netE_vis, netG=netG, netCls=netCls, netIP=netIP)
        binding = {}
        for name, model in models.items():
            inner = type(model.module)
            require(type(model) is torch.nn.DataParallel, 'upstream DataParallel wrapper ' + name)
            require(inner.__name__ == EXPECTED_CLASSES[name], 'class binding ' + name)
            require(Path(sys.modules[inner.__module__].__file__).resolve().is_relative_to(root.resolve()),
                    'class from pinned source ' + name)
            binding[name] = {'wrapper': 'torch.nn.DataParallel', 'class': inner.__name__,
                             'module': inner.__module__, 'device_ids': list(model.device_ids),
                             'file_sha256': sha(Path(sys.modules[inner.__module__].__file__).read_bytes())}
        require(netE_nir.module.hdim == netE_vis.module.hdim == 128, 'encoder hdim')
        require(netCls.module.fc.out_features == 1 and netCls.module.fc.in_features == 128, 'Cls(128,1)')
        require(netIP.module.is_train is False and not hasattr(netIP.module, 'fc2_'), 'define_IP(is_train=False)')
        result['lightcnn'] = load_lightcnn(torch, netIP)
        netE_nir.train(); netE_vis.train(); netG.train(); netIP.eval()  # train_generator.py:105-108
        params = {n: parameter_report(m) for n, m in models.items()}
        for n, (count, tensors) in EXPECTED_PARAMETERS.items():
            require((params[n]['parameters'], params[n]['parameter_tensors']) == (count, tensors),
                    f'live parameter count {n}: {params[n]["parameters"]}/{params[n]["parameter_tensors"]}')
        require(params['netIP']['requires_grad_parameters'] == 0 and not netIP.training, 'netIP frozen eval')
        owned = sum(params[n]['parameters'] for n in OPTIMIZER_OWNED)
        owned_tensors = sum(params[n]['parameter_tensors'] for n in OPTIMIZER_OWNED)
        require((owned, owned_tensors) == (45_370_182, 55), 'optimizer-owned total')
        result['binding'] = binding
        result['parameters'] = params
        result['optimizer_ownership'] = {'owned_modules': list(OPTIMIZER_OWNED), 'parameters': owned,
            'parameter_tensors': owned_tensors, 'netCls_excluded': True, 'netIP_excluded': True,
            'source': 'addition_module/DSDG/train_generator.py:87-88', 'optimizer_constructed': False}
        result['modes'] = {n: m.training for n, m in models.items()}

        # Deterministic analytic in-memory FP32 patterns in [0,1]; no RNG, file or decoder.
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing='ij')
        x_spoof = torch.stack((xx, yy, 0.5 + 0.5 * torch.sin(xx * 7) * torch.cos(yy * 5)))[None].cuda()
        x_live = torch.stack((0.5 + 0.5 * torch.cos(xx * 3), 1 - yy, xx * yy))[None].cuda()
        require(not torch.equal(x_spoof, x_live), 'distinct synthetic inputs')
        require(all(float(x.min()) >= 0 and float(x.max()) <= 1 for x in (x_spoof, x_live)), '[0,1] inputs')
        out, shapes = {}, {}

        def record(name, tensor, shape):
            require(list(tensor.shape) == shape, f'{name} shape {list(tensor.shape)} != {shape}')
            out[name] = summarize(tensor)
            shapes[name] = list(tensor.shape)
        with torch.no_grad():
            record('x_spoof', x_spoof, [1, 3, 256, 256]); record('x_live', x_live, [1, 3, 256, 256])
            mu_nir, logvar_nir, mu_a, logvar_a = netE_nir(x_spoof)
            for n, v in [('mu_nir', mu_nir), ('logvar_nir', logvar_nir), ('mu_a', mu_a), ('logvar_a', logvar_a)]:
                record(n, v, [1, 128])
            mu_vis, logvar_vis = netE_vis(x_live)
            record('mu_vis', mu_vis, [1, 128]); record('logvar_vis', logvar_vis, [1, 128])
            z_cls = util.reparameterize(mu_a, logvar_a)  # train_generator.py:125-127 order
            z_nir = util.reparameterize(mu_nir, logvar_nir)
            z_vis = util.reparameterize(mu_vis, logvar_vis)
            for n, v in [('z_cls', z_cls), ('z_nir', z_nir), ('z_vis', z_vis)]:
                record(n, v, [1, 128])
            pre_spoof = netCls(z_cls)
            record('pre_spoof', pre_spoof, [1, 1])
            latent = torch.cat((z_cls, z_nir, z_vis), dim=1)
            record('latent', latent, [1, 384])
            rec = netG(latent)
            record('rec', rec, [1, 6, 256, 256])
            rec_spoof, rec_live = rec[:, 0:3, :, :], rec[:, 3:6, :, :]
            record('rec_spoof', rec_spoof, [1, 3, 256, 256]); record('rec_live', rec_live, [1, 3, 256, 256])
            for n, v in [('x_spoof', x_spoof), ('x_live', x_live), ('rec_spoof', rec_spoof), ('rec_live', rec_live)]:
                small = F.interpolate(v, size=(128, 128), mode='bilinear')  # train_generator.py:154-157
                record(n + '_128', small, [1, 3, 128, 128])
                gray = util.rgb2gray(small)
                record(n + '_gray', gray, [1, 1, 128, 128])
                record(n + '_ip_feature', netIP(gray), [1, 256])
            label_spoof = torch.zeros(1, dtype=torch.long, device='cuda')  # single spoof CE index 0
            ce = torch.nn.CrossEntropyLoss()(pre_spoof, label_spoof)
            require(ce.item() == 0.0, 'one-logit CE degeneracy')
            eps_mean_check = {n: float(((z - m) / (0.5 * lv).exp()).abs().max()) for n, z, m, lv in
                              [('z_cls', z_cls, mu_a, logvar_a), ('z_nir', z_nir, mu_nir, logvar_nir),
                               ('z_vis', z_vis, mu_vis, logvar_vis)]}
        result['warnings'] = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
        result['forward'] = out
        result['shapes'] = shapes
        result['finite_all'] = all(v['finite'] for v in out.values())
        result['one_logit_classifier'] = {'status': 'EXPECTED_DEGENERACY_CONFIRMED', 'logit_shape': [1, 1],
            'target_class_index': 0, 'cross_entropy': ce.item(),
            'reason': 'log_softmax over one logit is identically 0; A1 discloses loss_cls as degenerate',
            'converted_to_two_logits': False}
        result['reparameterization'] = {'function': 'misc/util.py::reparameterize',
            'distribution': 'eps ~ N(0,1) via torch.cuda.FloatTensor(std.size()).normal_(); z = eps*exp(0.5*logvar)+mu',
            'rng': 'global CUDA default generator, seeded by torch.cuda.manual_seed_all(diagnostic_seed)',
            'max_abs_standardized_eps': eps_mean_check, 'altered': False}
        torch.cuda.synchronize()
        result['cuda_memory'] = {'label': 'DIAGNOSTIC_BATCH1_ONLY',
            'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'autograd': 'torch.no_grad forward; no activation graph retained',
            'establishes_physical_batch_240_feasibility': False,
            'batch_240_extrapolation': 'NOT_PERFORMED'}
        final = {n: parameter_report(m)['sha256'] for n, m in models.items()}
        require(final == {n: p['sha256'] for n, p in params.items()}, 'parameters unchanged')
    result['source_after'] = source_identity()[0]
    require(result['source_after'] == source, 'source-cache integrity after execution')
    result['asset_after'] = asset_identity(adapter)
    require(result['asset_after'] == result['asset_before'], 'LightCNN unchanged')
    result['environment_after'] = environment()
    diff = {k: {'before': before[k], 'after': result['environment_after'][k]}
            for k in before if before[k] != result['environment_after'][k]}
    require(not diff, 'environment stable: ' + json.dumps(diff))
    require(not any(counters.values()), 'zero optimizer/backward/checkpoint calls')
    result.update(status='PASS', counters=counters, optimizer_constructed=False, optimizer_applications=0,
        backward_passes=0, checkpoint_created=False, training_launched=False, benchmark_data_access=False,
        TEST_access=False, synthetic_bank=False, physical_batch_240_qualified=False,
        training_graph_qualified=False, fidelity='CONTROLLED_ADAPTATION', compatibility_patch='NONE',
        firewall={'denied': firewall.denied, 'event_counts': firewall.events,
                  'lightcnn_read_opens': firewall.lightcnn_opens})
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
