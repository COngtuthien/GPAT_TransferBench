"""M6D4a synthetic architecture diagnostics, never a PCGAN training runner.

Run in an isolated process. All generated source/build/evidence lives under the
explicit external build root. No dataset, pretrained state, optimizer or image
writer is used. Scalar backward probes are diagnostics, not the five losses.
"""
import argparse
import difflib
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SEED = 60401  # qualification only; does not select an experiment seed
PIN = '6baa180f1184ee79a6b967f9d80ee0e02a979ac7'


def require(value, message):
    if not value:
        raise RuntimeError('M6D4a gate: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def source_identity():
    from methods.pcgan import PCGANAdapter
    adapter = PCGANAdapter()
    source = adapter.validate_source()
    root = Path(source['root'])
    # Code-only inventory; no weight/image bytes, even in upstream checkout.
    files = command('git', '-C', str(root), 'ls-files').splitlines()
    code = {}
    for rel in files:
        if Path(rel).suffix not in {'.py', '.cpp', '.cu', '.h', '.cuh'}:
            continue
        raw = (root / rel).read_bytes()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        expected = command('git', '-C', str(root), 'rev-parse', PIN + ':' + rel)
        require(blob == expected, 'pinned code blob ' + rel)
        code[rel] = {'sha256': sha(raw), 'git_blob': blob}
    source['code_files'] = code
    source['worktree_status'] = command('git', '-C', str(root), 'status', '--porcelain')
    return source, adapter


def prepare(build):
    source, _ = source_identity()
    original = Path(source['root'])
    generated = build / 'source'
    patches = []
    for rel, identity in source['code_files'].items():
        raw = (original / rel).read_bytes()
        changed = raw
        reason = None
        if rel == 'util/util.py':
            old = b'return int(major) >= 10 and int(minor) >= 1'
            require(raw.count(old) == 1, 'CUDA predicate patch context')
            changed = raw.replace(old, b'return (int(major), int(minor)) >= (10, 1)')
            reason = ('Correct CUDA >=10.1 version comparison: the original rejects 11.0/12.0/13.0. '
                      'Only selects the real pinned CUDA kernels; no tensor operation changes.')
        dest = generated / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(changed)
        if changed != raw:
            patches.append({'file': rel, 'original_sha256': identity['sha256'],
                'generated_sha256': sha(changed), 'reason_and_semantic_proof': reason,
                'diff': ''.join(difflib.unified_diff(raw.decode().splitlines(True),
                    changed.decode().splitlines(True), fromfile='pinned/'+rel, tofile='generated/'+rel))})
    manifest = {'source': source, 'generated_root': str(generated), 'patches': patches,
        'generated_files_sha256': {p: sha((generated/p).read_bytes()) for p in source['code_files']},
        'source_cache_modified': False, 'cuda_kernel_arithmetic_modified': False}
    (build/'compatibility_patch_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    require(source_identity()[0] == source, 'source identity after generation')
    return manifest


class Firewall:
    """Reject benchmark roots, sample records, weights and image I/O in Python.

    Subprocesses are limited by this runner to source Git and build/environment
    commands; this hook does not claim to intercept native compiler filesystem I/O.
    """
    def __init__(self, build):
        self.build = str(build) + '/'
        self.denied = []
        self.events = {k: 0 for k in ('open', 'os.listdir', 'os.scandir')}

    def __call__(self, event, args):
        if event not in self.events or not args or not isinstance(args[0], (str, bytes)):
            return
        self.events[event] += 1
        path = os.path.abspath(os.fsdecode(args[0]))
        parts = set(Path(path).parts)
        denied = bool(parts & {'faces_256', 'manifests'})
        for root in (str(ROOT), str(ROOT)+'_runtime'):
            denied |= any(path == root+'/'+p or path.startswith(root+'/'+p+'/')
                          for p in ('data', 'cache', 'runs'))
        denied |= Path(path).suffix.lower() in {'.parquet', '.jpg', '.jpeg', '.png', '.webp',
            '.bmp', '.npy', '.npz', '.pth', '.pt', '.ckpt', '.h5', '.pkl'}
        if event == 'open':
            flags = args[2] if len(args)>2 and isinstance(args[2], int) else 0
            writing = flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            denied |= bool(writing and not (path.startswith(self.build) or path.startswith('/tmp/')
                                            or path == '/dev/null'))
        if denied:
            self.denied.append({'event': event, 'path': path})
            raise RuntimeError('DATA/WRITE FIREWALL: '+path)


def environment():
    import torch
    import torchvision
    return {'python': sys.version, 'executable': sys.executable,
        'python_sha256': sha(Path(sys.executable).read_bytes()),
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        # Setuptools adds its vendored packages to sys.path at extension import.
        # Inventory installed distributions at fixed environment paths instead.
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions(
            path=sorted({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})))),
        'compiler': command('c++', '--version'),
        'nvcc': command(str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'), '--version'),
        'gpu_driver': command('nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'),
        'gpu_capability': list(torch.cuda.get_device_capability()),
        'precision': {'default_dtype': str(torch.get_default_dtype()),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'autocast_cuda': torch.is_autocast_enabled('cuda'),
            'autocast_cpu': torch.is_autocast_enabled('cpu'),
            'matmul_precision': torch.get_float32_matmul_precision()},
        'launch_environment': {k: os.environ.get(k) for k in ('CUDA_HOME','TORCH_EXTENSIONS_DIR',
            'TORCH_CUDA_ARCH_LIST','MAX_JOBS','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE',
            'NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED')}}


def summarize(t):
    import torch
    require(t.dtype == torch.float32 and t.is_cuda, 'FP32 CUDA tensor')
    require(torch.isfinite(t).all().item(), 'finite tensor')
    a = t.detach().cpu().contiguous().numpy()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device),
        'finite': True, 'sha256': sha(a.tobytes()), 'min': float(a.min()), 'max': float(a.max()),
        'mean': float(a.mean(dtype='float64')), 'l2': float((a.astype('float64')**2).sum()**.5)}


def gradients(probe, named, retain_graph=False):
    import torch
    named = list(named)
    grads = torch.autograd.grad(probe, [p for _,p in named], retain_graph=retain_graph,
                                allow_unused=False)
    result = {n: summarize(g) for (n,_),g in zip(named, grads)}
    require(any(v['l2'] > 0 for v in result.values()), 'nonzero gradient probe')
    return result


def qualify(build):
    firewall = Firewall(build)
    sys.addaudithook(firewall)
    source, adapter = source_identity()
    manifest = json.loads((build/'compatibility_patch_manifest.json').read_text())
    require(manifest['source'] == source, 'source matches generated-copy authority')
    generated = Path(manifest['generated_root'])
    for rel,h in manifest['generated_files_sha256'].items():
        require(sha((generated/rel).read_bytes()) == h, 'generated closure '+rel)
    import numpy as np
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    def forbidden(*args, **kwargs):
        raise RuntimeError('No optimizer, pretrained loading or checkpoint is allowed')
    torch.load = forbidden
    torch.save = forbidden
    torch.nn.Module.load_state_dict = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    torch.optim.Adam.step = forbidden
    before = environment()
    sys.path.insert(0, str(generated))
    import util
    require(util.is_custom_kernel_supported(), 'real CUDA branch enabled')
    from models.networks.encoder import StyleGAN2ResnetEncoder
    from models.networks.generator import StyleGAN2ResnetGenerator
    from models.networks.discriminator import StyleGAN2Discriminator
    from models.networks.patch_discriminator import StyleGAN2PatchDiscriminator
    act = importlib.import_module('models.networks.stylegan2_op.fused_act')
    up = importlib.import_module('models.networks.stylegan2_op.upfirdn2d')
    require(act.use_custom_kernel and up.use_custom_kernel, 'no custom-op fallback')
    result = {'diagnostic_seed': SEED, 'source_before': source, 'environment_before': before,
              'custom_ops': {}, 'models': {}, 'forward': {}, 'gradients': {}}
    a = torch.linspace(-1,1,2*3*8*8,device='cuda').reshape(2,3,8,8).requires_grad_()
    b = torch.zeros(3,device='cuda',requires_grad=True)
    out = act.fused_leaky_relu(a,b)
    require(list(out.shape)==[2,3,8,8], 'fused activation shape')
    oracle = torch.nn.functional.leaky_relu(a+b[None,:,None,None],.2)*2**.5
    torch.testing.assert_close(out,oracle)
    result['custom_ops']['fused_bias_act'] = {'output': summarize(out),
        'gradients': gradients(out.square().mean(), [('input',a),('bias',b)]),
        'reference_max_abs_error': float((out-oracle).abs().max().detach()), 'real_cuda': True}
    kernel = torch.ones(2,2,device='cuda')/4
    out = up.upfirdn2d(a,kernel,down=2)
    require(list(out.shape)==[2,3,4,4], 'upfirdn shape')
    oracle = torch.nn.functional.avg_pool2d(a,2,2)
    torch.testing.assert_close(out,oracle)
    result['custom_ops']['upfirdn2d'] = {'output': summarize(out),
        'gradients': gradients(out.square().mean(), [('input',a)]),
        'reference_max_abs_error': float((out-oracle).abs().max().detach()), 'real_cuda': True}
    for name, extension in [('fused_bias_act',act.fused),('upfirdn2d',up.upfirdn2d_op)]:
        path = Path(extension.__file__)
        require(path.is_relative_to(build), 'external extension location')
        result['custom_ops'][name]['extension'] = {'path': str(path), 'sha256': sha(path.read_bytes())}
        result['custom_ops'][name]['build_ninja'] = (path.parent/'build.ninja').read_text()
    from methods.pcgan.architecture import architecture_mapping
    arch = architecture_mapping(adapter.config,source)
    options = {k:v['value'] for k,v in arch['inherited_source_defaults'].items()}
    options.update({v['target'].removeprefix('opt.'):v['value'] for v in arch['settings']})
    opt = SimpleNamespace(**options)
    models = {}
    for name, cls in [('encoder',StyleGAN2ResnetEncoder),('generator',StyleGAN2ResnetGenerator),
                      ('image_discriminator',StyleGAN2Discriminator),('patch_discriminator',StyleGAN2PatchDiscriminator)]:
        m = cls(opt).cuda(); models[name] = m
        result['models'][name] = {'class': cls.__name__, 'source_symbol': cls.__module__+'.'+cls.__name__,
            'options': options, 'trainable_parameters': sum(p.numel() for p in m.parameters() if p.requires_grad),
            'buffers': {n:summarize(v) for n,v in m.named_buffers()}, 'module_tree': repr(m),
            'parameters': {n:list(p.shape) for n,p in m.named_parameters()}}
    E,G,D,P = (models[n] for n in ('encoder','generator','image_discriminator','patch_discriminator'))
    initial_parameters = {key: {n: sha(p.detach().cpu().numpy().tobytes())
        for n,p in model.named_parameters()} for key,model in models.items()}
    # CPU-generated deterministic analytic patterns, copied to CUDA; no image files.
    yy,xx = torch.meshgrid(torch.linspace(-1,1,256),torch.linspace(-1,1,256),indexing='ij')
    src = torch.stack((xx,yy,torch.sin(xx*7)*torch.cos(yy*5)))[None].cuda().requires_grad_()
    tgt = torch.stack((torch.cos(xx*3),torch.sin(yy*4),xx*yy))[None].cuda().requires_grad_()
    forward = result['forward']; grad = result['gradients']
    for n,x in [('src',src),('tgt',tgt)]:
        sp,gl = E(x)
        require(list(sp.shape)==[1,8,128,128], 'measured spatial code')
        forward[n+'_spatial'] = summarize(sp); forward[n+'_global'] = summarize(gl)
        rec = G(sp,gl)
        require(list(rec.shape)==[1,3,256,256], 'reconstruction RGB')
        forward[n+'_reconstruction'] = summarize(rec)
        grad[n+'_reconstruction'] = gradients(rec.square().mean(),
            [('input',x)]+[('E.'+n,p) for n,p in E.named_parameters()]+[('G.'+n,p) for n,p in G.named_parameters()])
        del sp,gl,rec
    sp_src,gl_src = E(src); sp_tgt,gl_tgt = E(tgt)
    mixed = G(sp_src,gl_tgt)  # no detach; exact pinned (spatial, global) interface
    require(list(mixed.shape)==[1,3,256,256], 'mixed RGB')
    forward['mixed'] = summarize(mixed)
    grad['mixed'] = gradients(mixed.square().mean(), [('source_input',src),('target_input',tgt),
        ('spatial_code',sp_src),('global_code',gl_tgt)]+[('E.'+n,p) for n,p in E.named_parameters()]+
        [('G.'+n,p) for n,p in G.named_parameters()],retain_graph=True)
    from methods.pcgan.blur import blur_inputs
    bt,bm = blur_inputs(tgt,mixed)
    require(list(bt.shape)==list(bm.shape)==[1,3,128,128], 'A5 shapes')
    for x,y in [(tgt,bt),(mixed,bm)]:
        oracle = x.reshape(1,3,128,2,128,2).mean((3,5))
        torch.testing.assert_close(y,oracle)
    forward['blur_target'] = summarize(bt); forward['blur_mixed'] = summarize(bm)
    grad['blur_mixed'] = gradients(bm.square().mean(), [('source_input',src),('target_input',tgt),
        ('spatial_code',sp_src),('global_code',gl_tgt)]+[('G.'+n,p) for n,p in G.named_parameters()],retain_graph=True)
    grad['blur_target'] = gradients(bt.sum(), [('target_input',tgt)],retain_graph=True)
    for n,x in [('real',src),('reconstruction',G(sp_src,gl_src)),('mixed',mixed)]:
        pred = D(x); forward['image_D_'+n] = summarize(pred)
        grad['image_D_'+n] = gradients(pred.mean(), [('image_input',x)]+
            [('D.'+k,p) for k,p in D.named_parameters()],retain_graph=True)
    # This is the active upstream patch interface, not the obsolete BasePatchD.forward
    # (which references an undefined sample_patches). Do not invent tile semantics.
    crops_src = util.apply_random_crop(src,opt.patch_size,(opt.patch_min_scale,opt.patch_max_scale),num_crops=opt.patch_num_crops)
    crops_mix = util.apply_random_crop(mixed,opt.patch_size,(opt.patch_min_scale,opt.patch_max_scale),num_crops=opt.patch_num_crops)
    feat_src = P.extract_features(crops_src,aggregate=opt.patch_use_aggregation)
    feat_mix = P.extract_features(crops_mix)
    pred = P.discriminate_features(feat_src,feat_mix)
    for n,x in [('patch_source',crops_src),('patch_mixed',crops_mix),('patch_source_features',feat_src),
                ('patch_mixed_features',feat_mix),('patch_D',pred)]: forward[n] = summarize(x)
    grad['patch_D'] = gradients(pred.mean(), [('source_input',src),('mixed_input',mixed)]+
        [('P.'+n,p) for n,p in P.named_parameters()])
    result['patch_interface'] = 'util.apply_random_crop -> extract_features(aggregate=patch_use_aggregation for reference) -> discriminate_features; upstream get_random_crops/compute_patch_discriminator_losses call path; no loss evaluated'
    result['source_after'] = source_identity()[0]
    require(result['source_after']==source, 'source-cache integrity after execution')
    result['environment_after'] = environment()
    differences = {k: {'before':before[k], 'after':result['environment_after'][k]}
                   for k in before if before[k]!=result['environment_after'][k]}
    require(not differences, 'environment stable during qualification: '+json.dumps(differences))
    final_parameters = {key: {n: sha(p.detach().cpu().numpy().tobytes())
        for n,p in model.named_parameters()} for key,model in models.items()}
    require(initial_parameters == final_parameters, 'parameters unchanged by diagnostic backward')
    result['parameter_sha256_before'] = initial_parameters
    result['parameter_sha256_after'] = final_parameters
    result.update(status='PASS', optimizer_steps=0, checkpoints=0, pretrained_loads=0,
        benchmark_data_access=False, synthetic_bank=False, five_loss_runner_qualified=False,
        fidelity='CONTROLLED_ADAPTATION', firewall={'denied':firewall.denied,'event_counts':firewall.events},
        generated_copy_manifest_sha256=sha((build/'compatibility_patch_manifest.json').read_bytes()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',default='process.json')
    args = parser.parse_args()
    build = args.build_root.resolve()
    require(not build.is_relative_to(ROOT), 'build root outside repository')
    require(build.parts[-2:] == ('builds','e05_pcgan'), 'dedicated build path')
    require(Path(args.output).name==args.output and args.output.endswith('.json'), 'evidence filename')
    build.mkdir(parents=True,exist_ok=True)
    result = prepare(build) if args.prepare else qualify(build)
    (build/args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result.get('status','PREPARED'),'output':str(build/args.output)}))


if __name__ == '__main__':
    main()
