"""M7C3 GPAT GPU runtime qualification harness (synthetic tensors only; run in gpat-m7-gpu on the RTX 3090).

Qualification only, NOT a training runner: every tensor is generated internally from fixed seeds; there is no dataset
input, no epoch loop, no restart logic, no checkpoint writer and no bank writer. The single optimizer-group smoke builds
temporary in-memory Adam optimizers and GradScalers, performs exactly one accumulation group and discards everything.
Teachers in the optimizer smoke are frozen synthetic stubs (the real teachers are exercised only by the separate R-04
Level-2 parity tool).

  python -B tools/m7c3_gpat_gpu_qualification.py --out <json>            full qualification
  python -B tools/m7c3_gpat_gpu_qualification.py --digests --out <json>  repeatability digests only (run twice)
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # before any CUDA context
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
import torchvision                                                  # noqa: E402

from methods.gpat import (composition, losses, naf_source, runtime_contract as rc, spectral,  # noqa: E402
                          teacher_preprocess as tp, wavelet)
from methods.gpat.artifact_encoder import encoder_input             # noqa: E402
from methods.gpat.config import VARIANTS                            # noqa: E402
from methods.gpat.generator import FiLM, naf_symbols                # noqa: E402
from methods.gpat.highpass import highpass                          # noqa: E402
from methods.gpat.model import parameter_counts                     # noqa: E402

AUTHORITY = '33955b05ff70289c42386c3ac1623bae4a5fae5e'
SEED = 20261003
DEV = 'cuda'
N = 4
EXPECTED_COUNTS = {'g_res': 31677421, 'e_art': 11204736, 'discriminator': 2767809, 'attack_head': 3078,
                   'identity_head': 30780}
FP32_REQUIRED = ('wavelet.dwt', 'wavelet.idwt', 'composition.activate', 'composition.compose', 'composition.artifact_map',
                 'spectral.s_radial', 'spectral.s_orient', 'losses.l_tv', 'losses.l_budget', 'losses.l_low',
                 'losses.l_type', 'losses.idadv_microbatch', 'losses.l_parse', 'losses.l_d', 'losses.l_gadv',
                 'losses.l_id', 'losses.l_lm', 'losses.l_artcon', 'losses.l_bg', 'teacher.facexformer_input',
                 'teacher.adaface_input')


def _ev_tool():
    spec = importlib.util.spec_from_file_location('m7c2b_ev', ROOT / 'tools/m7c2b_gpat_static_core_evidence.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVT = _ev_tool()


def digest(t) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def gen(seed):
    return torch.Generator().manual_seed(seed)


def uniform(shape, seed):
    return (torch.rand(*shape, generator=gen(seed)) * 2 - 1).to(DEV)


def build(variant, *, with_discriminator=True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return EVT.build(variant, SEED, with_discriminator=with_discriminator).to(DEV)


def determinism_state() -> dict:
    return {'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'use_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def setup() -> dict:
    state = rc.apply_qualification_determinism(SEED, gpu=True)
    return {'applied': state, 'observed': determinism_state()}


def environment() -> dict:
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    import cv2
    import numpy
    import PIL
    import pywt
    return {'python': platform.python_version(), 'executable': sys.executable, 'sys_prefix': sys.prefix,
            'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'numpy': numpy.__version__, 'Pillow': PIL.__version__,
            'opencv': cv2.__version__, 'opencv_python_headless': md.version('opencv-python-headless'),
            'ptwt': md.version('ptwt'), 'PyWavelets_distribution': md.version('PyWavelets'),
            'pywt___version__': pywt.__version__, 'pywt___file__': pywt.__file__,
            'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
            'nvidia_smi': smi, 'host': platform.node()}


# ----------------------------------------------------------------------------- static authority on GPU
def static_authority() -> dict:
    from methods.gpat.config import load_config
    cfg = {v: load_config(v).sha256 for v in VARIANTS}
    m = build('B3')
    counts = {n: parameter_counts(getattr(m, n)) for n in EXPECTED_COUNTS}
    del m
    torch.cuda.empty_cache()
    ok = all(counts[k]['total'] == v for k, v in EXPECTED_COUNTS.items())
    return {'config_sha256': cfg, 'parameter_counts': counts, 'counts_match': ok}


# ----------------------------------------------------------------------------- NAF on torch 2.12.1 / CUDA
def _ln_native(x, w, b, eps):
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    return w.view(1, -1, 1, 1) * ((x - mu) / (var + eps).sqrt()) + b.view(1, -1, 1, 1)


def naf_gpu() -> dict:
    verified = naf_source.verify_source()
    syms = naf_symbols()
    lnf = syms['LayerNormFunction']
    out = {'commit': verified['commit'], 'tree': verified['tree'],
           'files_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in verified['files'].items()},
           'basicsr_imported': 'basicsr' in sys.modules, 'lmdb_imported': 'lmdb' in sys.modules}
    parity = {}
    for dt, tol in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
        g = gen(SEED)
        x = (torch.randn(2, 64, 32, 32, generator=g, dtype=dt) * 2 + 0.5).to(DEV)
        w, b = torch.randn(64, generator=g, dtype=dt).to(DEV), torch.randn(64, generator=g, dtype=dt).to(DEV)
        go = torch.randn(2, 64, 32, 32, generator=g, dtype=dt).to(DEV)
        a = [t.clone().requires_grad_(True) for t in (x, w, b)]
        r = [t.clone().requires_grad_(True) for t in (x, w, b)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            y1 = lnf.apply(*a, 1e-6)
            y1.backward(go)
        y2 = _ln_native(*r, 1e-6)
        y2.backward(go)
        f = float((y1 - y2).detach().abs().max())
        gmax = max(float((p.grad - q.grad).abs().max()) / max(1.0, float(q.grad.abs().max())) for p, q in zip(a, r))
        parity[str(dt).split('.')[-1]] = {'forward_max_abs': f, 'grad_max_rel': gmax, 'tolerance': tol,
                                          'pass': f <= tol and gmax <= tol}
        out['saved_variables_warnings'] = sorted({f'{w.category.__name__}: {str(w.message)[:120]}' for w in caught})
    out['layernorm_native_parity'] = parity
    out['saved_variables_works'] = all(v['pass'] for v in parity.values())
    blocks = {}
    for mode in ('fp32', 'fp16_autocast'):
        torch.manual_seed(SEED)
        blk = syms['NAFBlock'](64).to(DEV)
        with torch.no_grad():
            blk.beta.normal_(0, 0.5)
            blk.gamma.normal_(0, 0.5)
        x = uniform((N, 64, 64, 64), SEED + 1).requires_grad_(True)
        ctx = torch.autocast('cuda', dtype=torch.float16) if mode != 'fp32' else contextlib.nullcontext()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with ctx:
                y = blk(x)
            y.float().square().mean().backward()
        grads = [p.grad for p in blk.parameters()] + [x.grad]
        blocks[mode] = {'output_dtype': str(y.dtype).split('.')[-1], 'shape_preserved': tuple(y.shape) == tuple(x.shape),
                        'finite_forward': bool(torch.isfinite(y).all()),
                        'finite_backward': all(bool(torch.isfinite(gg).all()) for gg in grads),
                        'max_activation_abs': float(y.detach().float().abs().max()),
                        'max_grad_abs': max(float(gg.abs().max()) for gg in grads)}
    out['nafblock'] = blocks
    out['pass'] = (out['saved_variables_works'] and not out['basicsr_imported'] and not out['lmdb_imported'] and
                   all(b['finite_forward'] and b['finite_backward'] and b['shape_preserved'] for b in blocks.values()))
    return out


# ----------------------------------------------------------------------------- deterministic op digests
def _op(fn, x, go_seed):
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    go = torch.randn(y.shape, generator=gen(go_seed)).to(DEV, y.dtype)
    y.backward(go)
    return y, x.grad


def op_digests() -> dict:
    torch.manual_seed(SEED)
    syms = naf_symbols()
    up_mod = nn.Conv2d(512, 256, 3, padding=1).to(DEV)
    film = FiLM(128).to(DEV)
    with torch.no_grad():
        film.affine.weight.normal_(0, 0.02)
    blk = syms['NAFBlock'](128).to(DEV)
    with torch.no_grad():
        blk.beta.normal_(0, 0.5)
        blk.gamma.normal_(0, 0.5)
    d = build('B0').discriminator
    z = uniform((N, 512), SEED + 5)
    x256 = uniform((N, 3, 256, 256), SEED + 2)
    ops = {
        'dwt': (lambda x: torch.cat(wavelet.dwt(x), 1), x256),
        'idwt': (lambda x: wavelet.idwt(x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9:12]), uniform((N, 12, 128, 128), SEED + 3)),
        'highpass': (highpass, x256),
        'bilinear_x2_source_bands': (lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False),
                                     uniform((N, 9, 128, 128), SEED + 4)),
        'decoder_up_bilinear_conv': (lambda x: up_mod(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)),
                                     uniform((N, 512, 8, 8), SEED + 6)),
        'avgpool_bottleneck': (nn.AvgPool2d(2, 2), uniform((N, 512, 16, 16), SEED + 7)),
        'film': (lambda x: film(x, z), uniform((N, 128, 32, 32), SEED + 8)),
        'nafblock': (blk, uniform((N, 128, 32, 32), SEED + 9)),
        'patchgan': (d, x256),
        's_radial': (spectral.s_radial, x256),
        's_orient': (spectral.s_orient, x256)}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        for name, (fn, x) in ops.items():
            y, g = _op(fn, x, SEED + 100)
            out[name] = {'forward': digest(y), 'grad': digest(g), 'finite': bool(torch.isfinite(y).all() and torch.isfinite(g).all())}
    cls = torch.randint(0, 11, (N, 224, 224), generator=gen(SEED + 10)).to(DEV)
    out['face_mask_nearest_exact'] = {'forward': digest(losses.face_mask_dilated(cls)), 'grad': None, 'finite': True}
    return out


def bilinear_matrix_parity() -> dict:
    """The audited fixed-matrix bilinear x2 (N-09) vs F.interpolate on this CUDA stack (evaluation; not adopted)."""
    x = uniform((N, 9, 128, 128), SEED + 4).double()
    m = torch.as_tensor(rc.bilinear_x2_matrix(128), dtype=torch.float64, device=DEV)
    out = {}
    for dt in (torch.float64, torch.float32):
        a = x.to(dt).clone().requires_grad_(True)
        b = x.to(dt).clone().requires_grad_(True)
        y1 = F.interpolate(a, scale_factor=2, mode='bilinear', align_corners=False)
        mm = m.to(dt)
        y2 = torch.einsum('oh,nchw,pw->ncop', mm, b, mm)
        go = torch.randn(y1.shape, generator=gen(SEED + 11), dtype=dt).to(DEV)
        y1.backward(go)
        y2.backward(go)
        out[str(dt).split('.')[-1]] = {'forward_max_abs': float((y1 - y2).detach().abs().max()),
                                      'grad_max_abs': float((a.grad - b.grad).abs().max())}
    out['forward_gate_1e-6'] = out['float32']['forward_max_abs'] <= 1e-6
    out['adopted'] = False
    out['reason_not_adopted'] = 'F.interpolate bilinear backward runs under use_deterministic_algorithms(True) and is bitwise repeatable'
    return out


# ----------------------------------------------------------------------------- AMP / fp32 boundaries
@contextlib.contextmanager
def instrument(table):
    patched = []

    def wrap(mod, name, label):
        orig = getattr(mod, name)

        def fn(*a, **k):
            out = orig(*a, **k)
            ins = sorted({str(t.dtype).split('.')[-1] for t in a if isinstance(t, torch.Tensor)} |
                         {str(t.dtype).split('.')[-1] for v in a if isinstance(v, (tuple, list)) for t in v
                          if isinstance(t, torch.Tensor)} |
                         {str(t.dtype).split('.')[-1] for v in a if isinstance(v, dict) for t in v.values()
                          if isinstance(t, torch.Tensor)})
            outs = out if isinstance(out, (tuple, list)) else (list(out.values()) if isinstance(out, dict) else (out,))
            od = sorted({str(t.dtype).split('.')[-1] for t in outs if isinstance(t, torch.Tensor)})
            row = table.setdefault(label, {'inputs': set(), 'outputs': set(), 'autocast_enabled': set()})
            row['inputs'].update(ins)
            row['outputs'].update(od)
            row['autocast_enabled'].add(torch.is_autocast_enabled('cuda'))
            return out
        setattr(mod, name, fn)
        patched.append((mod, name, orig))

    for name in ('dwt', 'idwt'):
        wrap(wavelet, name, 'wavelet.' + name)
    for name in ('activate', 'compose', 'artifact_map'):
        wrap(composition, name, 'composition.' + name)
    for name in ('s_radial', 's_orient'):
        wrap(spectral, name, 'spectral.' + name)
    for name in ('l_tv', 'l_budget', 'l_low', 'l_type', 'idadv_microbatch', 'l_parse', 'l_d', 'l_gadv', 'l_id', 'l_lm',
                 'l_artcon', 'l_bg'):
        wrap(losses, name, 'losses.' + name)
    for name in ('facexformer_input', 'adaface_input'):
        wrap(tp, name, 'teacher.' + name)
    try:
        yield
    finally:
        for mod, name, orig in reversed(patched):
            setattr(mod, name, orig)


def module_dtypes(model) -> tuple:
    table, hooks = {}, []
    groups = {'e_art': model.e_art, 'g_res': model.g_res, 'attack_head': model.attack_head,
              'identity_head': model.identity_head, 'discriminator': model.discriminator}
    for gname, g in groups.items():
        if g is None:
            continue
        for m in g.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                def hook(mod, inp, out, gname=gname):
                    table.setdefault(gname, set()).add(str(out.dtype).split('.')[-1])
                hooks.append(m.register_forward_hook(hook))
    return table, hooks


# ----------------------------------------------------------------------------- synthetic stub teachers (frozen)
class StubTeachers(nn.Module):
    """Frozen, randomly initialized stand-ins with the real teachers' interfaces; fp32, eval, requires_grad False."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED + 50)
        self.id_net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(16, 512))
        self.fx_parse = nn.Conv2d(3, 11, 1)
        self.fx_lm = nn.Sequential(nn.AvgPool2d(56), nn.Flatten(), nn.Linear(48, 136))   # 224 -> 4, deterministic
        r18 = torchvision.models.resnet18(weights=None)
        self.f_art = nn.Sequential(*list(r18.children())[:-1], nn.Flatten())
        self.eval().requires_grad_(False)

    def forward(self, x):
        with torch.autocast('cuda', enabled=False):
            fx = tp.facexformer_input(x.float())
            ad = tp.adaface_input(x.float())
            return {'id': F.normalize(self.id_net(ad), dim=1), 'parse': self.fx_parse(fx),
                    'lm': torch.tanh(self.fx_lm(fx)).view(-1, 68, 2), 'art': self.f_art(highpass(x.float()))}


# ----------------------------------------------------------------------------- forward N=4, D11/D12
def forward_n4(variant) -> dict:
    model = build(variant)
    model.train()
    x_s, x_t = uniform((N, 3, 256, 256), SEED + 20), uniform((N, 3, 256, 256), SEED + 21)
    torch.cuda.synchronize()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        out = model(x_s, x_t, scale_hf=0.15)
        dlog = model.discriminator(out.x_hat)
    rec = {'shapes': {k: list(getattr(out, k).shape) for k in ('z_a', 'spatial_code', 'layer4', 'raw', 'delta_LL',
                                                               'delta_LH', 'delta_HL', 'delta_HH', 'M', 'LL_syn',
                                                               'x_hat', 'A')},
           'dtypes': {k: str(getattr(out, k).dtype).split('.')[-1] for k in ('z_a', 'spatial_code', 'raw', 'delta_LH', 'M',
                                                                           'LL_syn', 'x_hat', 'A')},
           'finite': all(bool(torch.isfinite(getattr(out, k)).all()) for k in ('z_a', 'raw', 'x_hat', 'A', 'M')),
           'mask_range': [float(out.M.min()), float(out.M.max())],
           'x_hat_range_float': [float(out.x_hat.min()), float(out.x_hat.max())],
           'A_stats': {'min': float(out.A.min()), 'max': float(out.A.max()), 'mean': float(out.A.mean())},
           'D_logits_shape': list(dlog.shape), 'D_logits_dtype': str(dlog.dtype).split('.')[-1],
           'attack_logits': None if out.attack_logits is None else {'shape': list(out.attack_logits.shape),
                                                                    'finite': bool(torch.isfinite(out.attack_logits).all())},
           'identity_logits': None if out.identity_logits is None else {'shape': list(out.identity_logits.shape),
                                                                        'finite': bool(torch.isfinite(out.identity_logits).all())}}
    del model, out
    torch.cuda.empty_cache()
    return rec


def d11_d12() -> dict:
    x = (torch.rand(100, 3, 256, 256, generator=gen(42)) * 2 - 1).to(DEV)
    d11 = float((wavelet.idwt(*wavelet.dwt(x)) - x).abs().max())
    del x
    model = build('B0')
    x_s1, x_s2, x_t = (uniform((N, 3, 256, 256), SEED + 30 + i) for i in range(3))
    with torch.no_grad():
        with torch.no_grad():
            model.g_res.ending.weight.normal_(0, 0.05)
            model.g_res.ending.bias.normal_(0, 1.0)
        out = {}
        for amp in (False, True):
            ctx = torch.autocast('cuda', dtype=torch.float16) if amp else contextlib.nullcontext()
            with ctx:
                a = model(x_s1, x_t, scale_hf=0.15, artifact_scale=0.0)
                b = model(x_s2, x_t, scale_hf=0.15, artifact_scale=0.0)
                live1 = model(x_s1, x_t, scale_hf=0.15)
                live2 = model(x_s2, x_t, scale_hf=0.15)
            recon = wavelet.idwt(*wavelet.dwt(x_t))
            out['amp_fp16' if amp else 'fp32'] = {
                'zero_residual_vs_idwt_dwt_x_t_max_abs': float((a.x_hat - recon).abs().max()),
                'zero_residual_vs_x_t_max_abs': float((a.x_hat - x_t).abs().max()),
                'source_independence_max_abs': float((a.x_hat - b.x_hat).abs().max()),
                'live_source_dependence_max_abs': float((live1.x_hat - live2.x_hat).abs().max()),
                'gamma0_LL_exact': bool(torch.equal(live1.LL_syn, live1.target_bands[0]))}
    del model
    torch.cuda.empty_cache()
    gates = all(v['zero_residual_vs_idwt_dwt_x_t_max_abs'] < 1e-5 and v['zero_residual_vs_x_t_max_abs'] < 1e-5 and
                v['source_independence_max_abs'] < 1e-5 and v['live_source_dependence_max_abs'] > 1e-5 and
                v['gamma0_LL_exact'] for v in out.values())
    return {'D11_max_abs': d11, 'D11_pass': d11 < 1e-5, 'D12': out, 'D12_pass': gates}


# ----------------------------------------------------------------------------- GRL / DEV-022 on GPU
def grl_gpu() -> dict:
    from methods.gpat.heads import IdentityAdversaryHead
    torch.manual_seed(SEED)
    head = IdentityAdversaryHead().to(DEV)
    ref = nn.Linear(512, 60).to(DEV)
    ref.load_state_dict(head.fc.state_dict())
    z = F.normalize(uniform((N, 512), SEED + 40), dim=1)
    y = torch.tensor([0, 7, 33, 59], device=DEV)
    z1, z2 = z.clone().requires_grad_(True), z.clone().requires_grad_(True)
    F.cross_entropy(head(z1), y).backward()
    F.cross_entropy(ref(z2), y).backward()
    return {'z_grad_is_negated_max_abs': float((z1.grad + z2.grad).abs().max()),
            'head_grad_same_sign_max_abs': float((head.fc.weight.grad - ref.weight.grad).abs().max()),
            'pass': float((z1.grad + z2.grad).abs().max()) <= 1e-7 and torch.equal(head.fc.weight.grad, ref.weight.grad)}


def dev022_gpu() -> dict:
    la, lb = uniform((N, 60), SEED + 60) * 3, uniform((N, 60), SEED + 61) * 3
    la.requires_grad_(True)
    lb.requires_grad_(True)
    ya, yb = torch.tensor([5, -1, 17, -1], device=DEV), torch.tensor([-1, -1, 42, -1], device=DEV)
    va, vb = ya >= 0, yb >= 0
    (sa, ca), (sb, cb) = losses.idadv_microbatch(la, ya, va), losses.idadv_microbatch(lb, yb, vb)
    group = losses.idadv_group([sa, sb], [ca, cb])
    ref = F.cross_entropy(torch.cat([la[va], lb[vb]]), torch.cat([ya[va], yb[vb]]), reduction='sum') / (ca + cb)
    sample_weighted = 0.5 * (sa / ca) + 0.5 * (sb / cb)
    group.backward()
    masked_grad = float(la.grad[~va].abs().sum() + lb.grad[~vb].abs().sum())
    z = uniform((N, 60), SEED + 62).requires_grad_(True)
    s0, c0 = losses.idadv_microbatch(z, torch.full((N,), -1, device=DEV), torch.zeros(N, dtype=torch.bool, device=DEV))
    g0 = losses.idadv_group([s0, s0], [c0, c0])
    g0.backward()
    return {'labelled_counts': [ca, cb], 'group': float(group), 'reference_sum_over_labelled': float(ref),
            'abs_diff': float((group - ref).abs()), 'sample_weighted_microbatch_means': float(sample_weighted),
            'differs_from_sample_weighted': abs(float(group - sample_weighted)) > 1e-3,
            'masked_rows_grad_abs_sum': masked_grad, 'zero_labelled_value': float(g0),
            'zero_labelled_requires_grad': g0.requires_grad, 'zero_labelled_grad_all_zero': bool((z.grad == 0).all()),
            'pass': float((group - ref).abs()) <= 1e-5 and abs(float(group - sample_weighted)) > 1e-3 and
            masked_grad == 0.0 and float(g0) == 0.0 and g0.requires_grad and bool((z.grad == 0).all())}


# ----------------------------------------------------------------------------- one synthetic accumulation group
def _norm(params):
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.norm(torch.stack(gs))) if gs else 0.0


def optimizer_group_smoke() -> dict:
    """B3, physical batch 4, grad_accum 2, N-04 pre-update joint gradient; exactly one group; nothing persisted."""
    model = build('B3')
    model.train()
    teachers = StubTeachers().to(DEV)
    gmods = model.generator_modules()
    g_params = [p for m in gmods.values() for p in m.parameters()]
    d_params = list(model.discriminator.parameters())
    g_opt = torch.optim.Adam(g_params, lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0)
    d_opt = torch.optim.Adam(d_params, lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0)
    g_scaler, d_scaler = torch.amp.GradScaler('cuda'), torch.amp.GradScaler('cuda')
    cur = rc.curriculum(16576)
    weights = rc.group_weights([N, N])
    ident = [(torch.tensor([3, -1, 11, -1], device=DEV)), (torch.tensor([-1, 25, -1, -1], device=DEV))]
    group_count = sum(int((y >= 0).sum()) for y in ident)
    attack = [torch.tensor([0, 1, 4, 5], device=DEV), torch.tensor([2, 3, 4, 1], device=DEV)]
    face_mask = torch.ones(N, 1, 256, 256, device=DEV)
    before = {n: [p.detach().clone() for p in m.parameters()] for n, m in (('g', model.g_res), ('d', model.discriminator))}
    rec = {'microbatches': [], 'weights': weights, 'group_labelled_count': group_count}
    amp_table, mod_table = {}, None
    mod_table, hooks = module_dtypes(model)
    with instrument(amp_table):
        for i in range(2):
            x_s, x_t = uniform((N, 3, 256, 256), SEED + 70 + 2 * i), uniform((N, 3, 256, 256), SEED + 71 + 2 * i)
            model.discriminator.requires_grad_(False)
            d_norm_before_g = _norm(d_params)
            # everything below runs inside the fp16 autocast region on purpose: the fp32 pins must hold there
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x_s, x_t, scale_hf=cur['s_hf'])
                d_fake_for_g = model.discriminator(out.x_hat)
                with torch.no_grad():
                    t_ref = teachers(x_t)                              # detached x_t targets
                    t_src = teachers(x_s)
                t_hat = teachers(out.x_hat)
                spectral.s_radial(out.x_hat)
                spectral.s_orient(out.x_hat)
                comps = {'id': losses.l_id(t_ref['id'], t_hat['id']), 'lm': losses.l_lm(t_ref['lm'], t_hat['lm']),
                     'parse': losses.l_parse(t_ref['parse'], t_hat['parse']),
                     'low': losses.l_low(out.x_hat, out.target_bands[0]),
                     'artcon': losses.l_artcon(t_hat['art'], t_src['art'], t_ref['art']),
                     'spec': spectral.spec_loss(out.x_hat, x_s), 'gadv': losses.l_gadv(d_fake_for_g),
                     'type': losses.l_type(out.attack_logits, attack[i]),
                     'budget': losses.l_budget(out.A), 'tv': losses.l_tv(out.M, out.delta_LH, out.delta_HL, out.delta_HH),
                     'bg': losses.l_bg(out.x_hat, x_t, face_mask)}
                total, record = losses.assemble_generator_loss(comps, cur, lambda_type=0.2, lambda_idadv=0.0)
                ce_sum, cnt = losses.idadv_microbatch(out.identity_logits, ident[i], ident[i] >= 0)
                idadv = 0.1 * losses.idadv_share(ce_sum, group_count)    # never re-weighted by w_i (DEV-022)
                l_g = weights[i] * total + idadv
            g_scaler.scale(l_g).backward()
            d_unchanged_by_g = _norm(d_params) == d_norm_before_g
            g_norm_before_d = _norm(g_params)
            model.discriminator.requires_grad_(True)
            with torch.autocast('cuda', dtype=torch.float16):
                real, fake = model.discriminator(x_s), model.discriminator(out.x_hat.detach())
            l_d = losses.l_d(real, fake)
            d_scaler.scale(weights[i] * l_d).backward()
            rec['microbatches'].append({
                'L_G_weighted': float(l_g), 'L_D': float(l_d), 'idadv_share': float(idadv), 'labelled': cnt,
                'components': {k: float(v) for k, v in record['components'].items()},
                'finite': all(math.isfinite(float(v)) for v in (l_g, l_d, *record['components'].values())),
                'L_G_dtype': str(l_g.dtype).split('.')[-1],
                'D_grads_unchanged_by_G_backward': d_unchanged_by_g,
                'G_grad_norm_unchanged_by_D_backward': abs(_norm(g_params) - g_norm_before_d) == 0.0})
    for h in hooks:
        h.remove()
    # boundary sequence (N-04)
    scales = {'G_before': float(g_scaler.get_scale()), 'D_before': float(d_scaler.get_scale())}
    d_scaled_norm = _norm(d_params)
    d_scaler.unscale_(d_opt)
    d_unscaled_norm = _norm(d_params)
    d_total = float(torch.nn.utils.clip_grad_norm_(d_params, 1.0))
    d_after_clip = _norm(d_params)
    d_scaler.step(d_opt)
    d_scaler.update()
    g_scaled_norm = _norm(g_params)
    g_scaler.unscale_(g_opt)
    g_unscaled_norm = _norm(g_params)
    g_total = float(torch.nn.utils.clip_grad_norm_(g_params, 1.0))
    g_after_clip = _norm(g_params)
    g_scaler.step(g_opt)
    g_scaler.update()
    g_opt.zero_grad(set_to_none=True)
    d_opt.zero_grad(set_to_none=True)
    scales.update({'G_after': float(g_scaler.get_scale()), 'D_after': float(d_scaler.get_scale()),
                   'independent_objects': g_scaler is not d_scaler and g_scaler._scale is not d_scaler._scale})
    changed = {n: any(not torch.equal(a, b.detach()) for a, b in zip(before[n], (model.g_res if n == 'g' else
                                                                                  model.discriminator).parameters()))
               for n in before}
    finite_params = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    rec.update({
        'scales': scales,
        'D': {'scaled_norm': d_scaled_norm, 'unscaled_norm': d_unscaled_norm, 'clip_returned_total_norm': d_total,
              'norm_after_clip': d_after_clip,
              'clip_after_unscale': abs(d_total - d_unscaled_norm) <= 1e-3 * max(1.0, d_unscaled_norm) and
              abs(d_scaled_norm / scales['D_before'] - d_unscaled_norm) <= 1e-3 * max(1.0, d_unscaled_norm)},
        'G': {'scaled_norm': g_scaled_norm, 'unscaled_norm': g_unscaled_norm, 'clip_returned_total_norm': g_total,
              'norm_after_clip': g_after_clip,
              'clip_after_unscale': abs(g_total - g_unscaled_norm) <= 1e-3 * max(1.0, g_unscaled_norm) and
              abs(g_scaled_norm / scales['G_before'] - g_unscaled_norm) <= 1e-3 * max(1.0, g_unscaled_norm)},
        'params_changed': changed, 'params_finite_after_step': finite_params,
        'optimizer_steps': {'G': 1, 'D': 1}, 'checkpoint_writes': 0,
        'amp_function_boundaries': {k: {kk: sorted(map(str, vv)) for kk, vv in v.items()} for k, v in amp_table.items()},
        'amp_module_output_dtypes': {k: sorted(v) for k, v in mod_table.items()}})
    fp16_ok = all(v == ['float16'] for v in rec['amp_module_output_dtypes'].values())
    fp32_ok = all(rec['amp_function_boundaries'][k]['outputs'] == ['float32'] for k in rec['amp_function_boundaries'])
    rec['amp_pass'] = fp16_ok and fp32_ok
    rec['pass'] = (all(m['finite'] and m['D_grads_unchanged_by_G_backward'] and m['G_grad_norm_unchanged_by_D_backward']
                       for m in rec['microbatches']) and finite_params and all(changed.values()) and
                   rec['D']['clip_after_unscale'] and rec['G']['clip_after_unscale'] and
                   rec['D']['norm_after_clip'] <= 1.0 + 1e-4 and rec['G']['norm_after_clip'] <= 1.0 + 1e-4 and
                   scales['independent_objects'])
    del model, teachers, g_opt, d_opt
    torch.cuda.empty_cache()
    return rec


def scaler_clip_demo() -> dict:
    """Explicit cases the smoke cannot force: clipping that actually binds after unscale, and scaler independence
    when one optimizer's gradients overflow (that scaler skips its step and backs off; the other is untouched)."""
    torch.manual_seed(SEED + 95)
    g_lin, d_lin = nn.Linear(64, 64).to(DEV), nn.Linear(64, 64).to(DEV)
    g_opt = torch.optim.Adam(g_lin.parameters(), lr=1e-3, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d_lin.parameters(), lr=1e-3, betas=(0.5, 0.999))
    g_s, d_s = torch.amp.GradScaler('cuda'), torch.amp.GradScaler('cuda')
    x = uniform((N, 64), SEED + 96)
    lg = (g_lin(x) * 50).square().mean()                           # fp32, large but finite: clipping must bind
    with torch.autocast('cuda', dtype=torch.float16):
        ld = d_lin(x).float().sum()
    g_s.scale(lg).backward()
    d_s.scale(ld).backward()
    with torch.no_grad():
        d_lin.weight.grad[0, 0] = float('inf')                    # overflow only on the D side
    g_scale0, d_scale0 = g_s.get_scale(), d_s.get_scale()
    g_scaled = _norm(list(g_lin.parameters()))
    g_s.unscale_(g_opt)
    g_unscaled = _norm(list(g_lin.parameters()))
    total = float(torch.nn.utils.clip_grad_norm_(g_lin.parameters(), 1.0))
    g_clipped = _norm(list(g_lin.parameters()))
    g_before = [p.detach().clone() for p in g_lin.parameters()]
    d_before = [p.detach().clone() for p in d_lin.parameters()]
    d_s.unscale_(d_opt)
    torch.nn.utils.clip_grad_norm_(d_lin.parameters(), 1.0)
    d_s.step(d_opt)
    d_s.update()
    g_s.step(g_opt)
    g_s.update()
    g_stepped = any(not torch.equal(a, b) for a, b in zip(g_before, g_lin.parameters()))
    d_stepped = any(not torch.equal(a, b) for a, b in zip(d_before, d_lin.parameters()))
    rec = {'G_scaled_norm': g_scaled, 'G_unscaled_norm': g_unscaled, 'clip_returned_total_norm': total,
           'G_norm_after_clip': g_clipped, 'clip_binds_after_unscale': g_unscaled > 1.0 and abs(total - g_unscaled) <=
           1e-3 * g_unscaled and abs(g_clipped - 1.0) <= 1e-3 and abs(g_scaled / g_scale0 - g_unscaled) <= 1e-3 * g_unscaled,
           'G_scale': [g_scale0, g_s.get_scale()], 'D_scale': [d_scale0, d_s.get_scale()],
           'G_stepped': g_stepped, 'D_step_skipped_on_inf': not d_stepped}
    rec['scalers_independent'] = (rec['D_scale'][1] == d_scale0 * 0.5 and rec['G_scale'][1] == g_scale0 and g_stepped
                                  and not d_stepped)
    rec['pass'] = rec['clip_binds_after_unscale'] and rec['scalers_independent']
    return rec


def teacher_adapter_autocast_hazard() -> dict:
    """The M7C2a adapters do not disable autocast themselves: record their dtype inside and outside autocast."""
    x = uniform((1, 3, 256, 256), SEED + 80)
    with torch.autocast('cuda', dtype=torch.float16):
        inside = (str(tp.facexformer_input(x).dtype), str(tp.adaface_input(x).dtype))
        with torch.autocast('cuda', enabled=False):
            outside = (str(tp.facexformer_input(x).dtype), str(tp.adaface_input(x).dtype))
    return {'inside_autocast': inside, 'inside_explicit_fp32_region': outside,
            'call_site_contract': 'teacher adapters + teacher forwards run inside torch.autocast(enabled=False)',
            'pass': outside == ('torch.float32', 'torch.float32')}


# ----------------------------------------------------------------------------- VRAM
def _peak(fn) -> dict:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return {'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
            'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
            'max_memory_allocated_GiB': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
            'max_memory_reserved_GiB': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}


def vram() -> dict:
    def fwd(v):
        def run():
            m = build(v)
            m.train()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                o = m(uniform((N, 3, 256, 256), SEED + 90), uniform((N, 3, 256, 256), SEED + 91), scale_hf=0.15)
                m.discriminator(o.x_hat)
        return run
    out = {'A_B0_forward_N4': _peak(fwd('B0')), 'B_B3_forward_N4': _peak(fwd('B3'))}
    holder = {}
    out['C_B3_one_group_backward_and_step'] = _peak(lambda: holder.setdefault('smoke', optimizer_group_smoke()))
    out['physical_batch'] = N
    out['oom'] = False
    return out, holder['smoke']


# ----------------------------------------------------------------------------- digests (repeatability)
def core_digests() -> dict:
    model = build('B0')
    model.train()
    x_s, x_t = uniform((N, 3, 256, 256), SEED + 20), uniform((N, 3, 256, 256), SEED + 21)
    out = {}
    for amp in (False, True):
        for p in model.parameters():
            p.grad = None
        ctx = torch.autocast('cuda', dtype=torch.float16) if amp else contextlib.nullcontext()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with ctx:
                o = model(x_s, x_t, scale_hf=0.15)
                dl = model.discriminator(o.x_hat)
            loss = (spectral.spec_loss(o.x_hat, x_s) + losses.l_low(o.x_hat, o.target_bands[0]) +
                    losses.l_budget(o.A) + losses.l_gadv(dl))
            loss.backward()
        tag = 'amp_fp16' if amp else 'fp32'
        out[tag] = {'dwt_target_LL': digest(o.target_bands[0]), 'g_res_raw': digest(o.raw), 'x_hat': digest(o.x_hat),
                    'A': digest(o.A), 'D_logits': digest(dl), 'loss': digest(loss),
                    'grad_g_res_ending': digest(model.g_res.ending.weight.grad),
                    'grad_g_res_intro': digest(model.g_res.intro.weight.grad),
                    'grad_e_art_conv1': digest(model.e_art.conv1.weight.grad),
                    'grad_D_first_conv': digest(model.discriminator.model[0].weight.grad)}
    return {'core': out, 'ops': op_digests()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--digests', action='store_true')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('M7C3 GPU qualification needs CUDA')
    det = setup()
    if args.digests:
        res = {'kind': 'repeatability_digests', 'determinism': det, 'digests': core_digests(),
               'environment': environment()}
    else:
        res = {'milestone': 'M7C3', 'kind': 'gpu_runtime_qualification', 'authority_commit': AUTHORITY, 'seed': SEED,
               'synthetic_only': True, 'dataset_images_read': 0, 'checkpoint_writes': 0, 'bank_writes': 0,
               'training_runs': 0, 'environment': environment(), 'determinism': det,
               'static_authority': static_authority(), 'naf': naf_gpu(), 'bilinear_matrix_parity': bilinear_matrix_parity(),
               'forward_n4': {'B0': forward_n4('B0'), 'B3': forward_n4('B3')}, 'd11_d12': d11_d12(),
               'grl': grl_gpu(), 'dev022': dev022_gpu(), 'teacher_adapter_autocast': teacher_adapter_autocast_hazard(),
               'scaler_clip_demo': scaler_clip_demo()}
        res['vram'], res['optimizer_group_smoke'] = vram()
        res['determinism_after'] = determinism_state()
        f = res['forward_n4']
        res['gates'] = {
            'counts': res['static_authority']['counts_match'], 'naf': res['naf']['pass'],
            'D11': res['d11_d12']['D11_pass'], 'D12': res['d11_d12']['D12_pass'], 'grl': res['grl']['pass'],
            'dev022': res['dev022']['pass'], 'optimizer_group': res['optimizer_group_smoke']['pass'],
            'amp_boundaries': res['optimizer_group_smoke']['amp_pass'],
            'teacher_adapter_fp32_region': res['teacher_adapter_autocast']['pass'],
            'scaler_clip': res['scaler_clip_demo']['pass'],
            'forward_n4': all(v['finite'] and v['D_logits_shape'] == [N, 1, 30, 30] and v['shapes']['x_hat'] == [N, 3, 256, 256]
                              for v in f.values()) and f['B0']['shapes'] == f['B3']['shapes'],
            'determinism_still_enabled': res['determinism_after']['use_deterministic_algorithms'] and
            not res['determinism_after']['deterministic_warn_only'],
            'no_oom': not res['vram']['oom']}
        res['status'] = 'PASS' if all(res['gates'].values()) else 'FAIL'
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(res.get('gates', {'digests': 'written'}), indent=1))


if __name__ == '__main__':
    main()
