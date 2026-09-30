"""M7C2a CPU qualification: gpat-m7-cpu environment, pinned NAFNet source (R-05) and R-04 Level-1 parity.

    ~/.venvs/gpat-m7-cpu/bin/python -B tools/m7c2a_gpat_cpu_qualification.py [--write]

CPU only (refuses a visible CUDA device). Reads no TRAIN/VAL/TEST sample, no manifest and no teacher checkpoint
except the already-local, hash-verified torchvision ResNet-18 IMAGENET1K_V1 file used as F_art. The corpus is a
deterministic synthetic set. FaceXFormer is instantiated from its hash-verified pinned code with random, frozen
weights (gradient structure only); AdaFace code is not available on the laptop, so its gradient-structure check
uses a frozen stand-in network. Level 2 is not executed. --write stores the evidence JSON; without it the result
is printed and compared with the stored evidence gates.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import math
import platform
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from methods.gpat import naf_source, runtime_contract as rc, teacher_preprocess as tp  # noqa: E402

EVIDENCE = 'outputs/audit/M7C2A_GPAT_CPU_QUALIFICATION.json'
LOCK = 'environments/gpat_m7_cpu.lock.json'
CORPUS_SEED = 20260930
DETERMINISM_SEED = 42
RESNET18 = Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
RESNET18_SHA = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
FX_CODE = ROOT / 'third_party/source_cache/facexformer'
GATES = {'facexformer_matrix_vs_library_float_operator_max_abs': 1e-6,
         'facexformer_clip_emulating_vs_frozen_pil_uint8_max_lsb': 1.13,
         'facexformer_clip_emulating_vs_frozen_normalized_max_abs': 0.0198,
         'adaface_matrix_vs_library_float_operator_max_abs': 1e-6,
         'adaface_adapter_vs_frozen_cv2_uint8_max_abs': 0.5 / 127.5 + 1e-6,
         'highpass_float_path_max_abs': 1e-6,
         'artifact_probe_float_preprocessing_max_abs': 1e-6}
NAF_FP64_GATE = 1e-10          # forward/gradient parity in float64: only rounding differs
NAF_FP32_REL_GATE = 1e-5       # float32 parity relative to the tensor's max magnitude
EQUIV_FWD_GATE = 1e-6          # N-09 deterministic replacement pre-evidence (forward)
EQUIV_GRAD_GATE = 1e-5         # N-09 pre-evidence gradient parity (float32 accumulation order)


def require(ok, msg):
    if not ok:
        raise SystemExit('M7C2a CPU qualification FAILED: ' + msg)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------------------- synthetic corpus (no data)
def corpus():
    rng = np.random.default_rng(CORPUS_SEED)
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    imgs = {}
    for v in (0, 1, 127, 128, 254, 255):
        imgs[f'const_{v}'] = np.full((h, w, 3), v, np.uint8)
    imgs['const_rgb'] = np.broadcast_to(np.array([12, 200, 99], np.uint8), (h, w, 3)).copy()
    imgs['ramp_h'] = np.repeat(np.round(xx / (w - 1) * 255)[..., None], 3, 2).astype(np.uint8)
    imgs['ramp_v'] = np.repeat(np.round(yy / (h - 1) * 255)[..., None], 3, 2).astype(np.uint8)
    imgs['ramp_diag_rgb'] = np.stack([np.round((xx + yy) / 510 * 255), np.round(xx / 255 * 255),
                                      np.round((255 - yy) / 255 * 255)], 2).astype(np.uint8)
    for p in (1, 2, 3, 4, 7, 8, 16, 32):
        c = (((xx // p) + (yy // p)) % 2 * 255).astype(np.uint8)
        imgs[f'checker_{p}'] = np.repeat(c[..., None], 3, 2)
    for lo, hi in ((0, 255), (64, 192)):
        c = (((xx // 4) + (yy // 4)) % 2 * (hi - lo) + lo).astype(np.uint8)
        imgs[f'checker_4_{lo}_{hi}'] = np.repeat(c[..., None], 3, 2)
    for bg, fg in ((0, 255), (255, 0), (128, 255)):
        a = np.full((h, w), bg, np.uint8)
        a[::17, ::13] = fg
        imgs[f'impulses_{bg}_{fg}'] = np.repeat(a[..., None], 3, 2)
    a = np.zeros((h, w), np.uint8)
    a[128, 128] = 255
    imgs['impulse_single'] = np.repeat(a[..., None], 3, 2)
    for name, m in (('edge_v', xx >= 128), ('edge_h', yy >= 101), ('edge_diag', xx + yy >= 255),
                    ('edge_circle', (xx - 128) ** 2 + (yy - 120) ** 2 <= 70 ** 2)):
        imgs[name + '_0_255'] = np.repeat((m * 255).astype(np.uint8)[..., None], 3, 2)
        imgs[name + '_40_210'] = np.repeat((m * 170 + 40).astype(np.uint8)[..., None], 3, 2)
    for i in range(6):
        imgs[f'uniform_{i}'] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    for i, s in enumerate((1.0, 2.0, 4.0, 8.0, 3.0, 6.0)):
        n = rng.standard_normal((h, w, 3)).astype(np.float32)
        b = cv2.GaussianBlur(n, (0, 0), s, borderType=cv2.BORDER_REFLECT_101)
        b = (b - b.min()) / (b.max() - b.min())
        lo, hi = (0.0, 1.0) if i < 4 else (0.15, 0.85)
        imgs[f'smooth_s{s}_{i}'] = np.round((lo + (hi - lo) * b) * 255).astype(np.uint8)
    return imgs


INTERIOR = ('smooth_s3.0_4', 'smooth_s6.0_5', 'edge_v_40_210', 'checker_4_64_192')


def to_x(u8):
    return torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1)))[None].float() / 127.5 - 1.0


def stats(d):
    d = np.abs(np.asarray(d, np.float64)).ravel()
    return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()), 'rmse': float(np.sqrt((d ** 2).mean())),
            'p99_abs': float(np.percentile(d, 99))}


def impulse_matrix(fn, n_in):
    cols = []
    for j in range(n_in):
        e = np.zeros((2, n_in), np.float32)
        e[:, j] = 1.0
        cols.append(fn(e)[0])
    return np.stack(cols, 1).astype(np.float64)


# ----------------------------------------------------------------------------- environment
def environment():
    import ptwt
    import pywt
    require(not torch.cuda.is_available(), 'CPU only: a CUDA device is visible')
    freeze = sorted(f'{d.metadata["Name"]}=={d.version}' for d in md.distributions())
    lock = json.loads((ROOT / LOCK).read_text()) if (ROOT / LOCK).exists() else None
    env = {'python': platform.python_version(), 'executable': sys.executable, 'platform': platform.platform(),
           'torch': torch.__version__, 'torchvision': md.version('torchvision'), 'torch_cuda_build': torch.version.cuda,
           'cuda_available': torch.cuda.is_available(), 'numpy': np.__version__, 'Pillow': md.version('pillow'),
           'opencv': cv2.__version__, 'ptwt': md.version('ptwt'), 'PyWavelets': md.version('PyWavelets'),
           'pywt_module_version_string': pywt.__version__, 'ptwt_file': ptwt.__file__,
           'distributions_sha256': sha_bytes('\n'.join(freeze).encode())}
    if lock is not None:
        for k in ('python', 'torch', 'torchvision', 'numpy', 'Pillow', 'opencv', 'ptwt', 'PyWavelets'):
            require(env[k] == lock['versions'][k], f'environment {k} differs from {LOCK}')
    return env


def pywavelets_provenance():
    """M7C2A-OBS-02 PACKAGING_METADATA_ANOMALY: record, never patch, the runtime version string."""
    import pywt
    dist = md.distribution('PyWavelets')
    require(dist.version == '1.9.0', 'PyWavelets distribution metadata must report 1.9.0 (STOP)')
    prefix = Path(sys.prefix).resolve()
    module = Path(pywt.__file__).resolve()
    dist_info = Path(dist._path).resolve()
    record = {row.split(',')[0]: row.split(',')[1] for row in dist.read_text('RECORD').splitlines() if row}
    version_py = module.parent / 'version.py'
    import base64
    digest = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(version_py.read_bytes()).digest()).decode().rstrip('=')
    out = {'importlib_metadata_version': md.version('PyWavelets'), 'pywt___version__': pywt.__version__,
           'pywt___file__': str(module), 'sys_prefix': str(prefix), 'dist_info': str(dist_info),
           'module_in_environment': module.is_relative_to(prefix), 'environment_name': prefix.name,
           'record_version_py': record.get('pywt/version.py'), 'installed_version_py_matches_record': digest ==
           record.get('pywt/version.py'), 'classification': 'PACKAGING_METADATA_ANOMALY', 'patched': False}
    require(out['module_in_environment'] and prefix.name == 'gpat-m7-cpu', 'pywt must come from gpat-m7-cpu')
    require(dist_info.is_relative_to(prefix) and out['installed_version_py_matches_record'], 'pywt RECORD integrity')
    return out


def adversarial_checks():
    """M7C2A-OBS-01: L_D = 0.5 * (L_D_real + L_D_fake) equals the balanced concatenated mean BCE."""
    torch.manual_seed(5)
    real, fake = torch.randn(4, 1, 30, 30, dtype=torch.float64), torch.randn(4, 1, 30, 30, dtype=torch.float64)
    ld = rc.d_loss(real, fake)
    lr, lf = rc.d_bce_terms(real, fake)
    cat = nn.BCEWithLogitsLoss(reduction='mean')(torch.cat([real, fake], 0),
                                                 torch.cat([torch.ones_like(real), torch.zeros_like(fake)], 0))
    diff = float((ld - cat).abs())
    require(float((ld - 0.5 * (lr + lf)).abs()) == 0.0 and diff <= 1e-12, 'L_D equivalence')
    return {'L_D': float(ld), 'concatenated_mean_bce': float(cat), 'abs_difference_float64': diff,
            'L_D_real': float(lr), 'L_D_fake': float(lf)}


def ptwt_checks():
    import ptwt
    import pywt
    g = torch.Generator().manual_seed(42)
    x = torch.rand(100, 3, 256, 256, generator=g) * 2 - 1
    coeffs = ptwt.wavedec2(x, pywt.Wavelet('haar'), level=1, mode='reflect')
    rec = ptwt.waverec2(coeffs, pywt.Wavelet('haar'))
    d11 = float((rec - x).abs().max())
    require(d11 < 1e-5, f'D11 DWT/IDWT {d11}')
    ca, (ch, cv, cd) = coeffs
    ref = pywt.dwt2(x[:2].double().numpy(), 'haar', mode='reflect', axes=(-2, -1))
    order = max(float(np.abs(a[:2].double().numpy() - b).max()) for a, b in zip((ca, ch, cv, cd), (ref[0], *ref[1])))
    require(order < 1e-5, 'ptwt (cA, (cH, cV, cD)) differs from pywt.dwt2')
    stripes = torch.zeros(1, 1, 256, 256)
    stripes[..., 1::2, :] = 1.0                       # rows alternate: horizontal stripes
    _, (hs, vs, ds) = ptwt.wavedec2(stripes, pywt.Wavelet('haar'), level=1, mode='reflect')
    _, (ht, vt, dt) = ptwt.wavedec2(stripes.transpose(-1, -2).contiguous(), pywt.Wavelet('haar'), level=1,
                                    mode='reflect')
    directional = {'horizontal_stripes_cH_cV_cD_absmax': [float(hs.abs().max()), float(vs.abs().max()),
                                                          float(ds.abs().max())],
                   'vertical_stripes_cH_cV_cD_absmax': [float(ht.abs().max()), float(vt.abs().max()),
                                                        float(dt.abs().max())]}
    require(directional['horizontal_stripes_cH_cV_cD_absmax'][0] > 0.9 and
            max(directional['horizontal_stripes_cH_cV_cD_absmax'][1:]) < 1e-6 and
            directional['vertical_stripes_cH_cV_cD_absmax'][1] > 0.9 and
            max(directional['vertical_stripes_cH_cV_cD_absmax'][0::2]) < 1e-6, 'R-01 directional mapping')
    return {'D11_max_abs': d11, 'D11_pass_if_less_than': 1e-5, 'coefficient_order_vs_pywt_max_abs': order,
            'R01_mapping': {'LL': 'cA', 'LH': 'cH', 'HL': 'cV', 'HH': 'cD'}, 'R01_directional': directional,
            'coefficient_shape': list(ca.shape)}


# ----------------------------------------------------------------------------- R-05 NAFNet
def ln_native(x, w, b, eps):
    mu = x.mean(1, keepdim=True)
    var = (x - mu).pow(2).mean(1, keepdim=True)
    return w.view(1, -1, 1, 1) * ((x - mu) / (var + eps).sqrt()) + b.view(1, -1, 1, 1)


class LN2dNative(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        return ln_native(x, self.weight, self.bias, self.eps)


def naf_checks():
    import copy
    verified = naf_source.verify_source()
    segments = naf_source.source_segments(verified['files'])
    before = set(sys.modules)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        syms = naf_source.load()
        LNF, LN2d, NAF = syms['LayerNormFunction'], syms['LayerNorm2d'], syms['NAFBlock']
        x = torch.randn(2, 8, 5, 5, dtype=torch.float64, requires_grad=True)
        LNF.apply(x, torch.ones(8, dtype=torch.float64, requires_grad=True),
                  torch.zeros(8, dtype=torch.float64, requires_grad=True), 1e-6).sum().backward()
    new_modules = sorted(m for m in set(sys.modules) - before if m.split('.')[0] in ('basicsr', 'lmdb'))
    require(not new_modules, 'BasicSR/lmdb imported: ' + json.dumps(new_modules))
    deprecations = sorted({str(w.message) for w in caught if 'saved_variables' in str(w.message)})
    require(deprecations, 'ctx.saved_variables path not exercised')
    out = {'commit': verified['commit'], 'tree': verified['tree'],
           'files_sha256': {k: sha_bytes(v) for k, v in verified['files'].items()},
           'symbols': {k: {kk: v[kk] for kk in ('file', 'first_line', 'last_line')} for k, v in segments.items()},
           'symbol_source_sha256': {k: sha_bytes(v['source'].encode()) for k, v in segments.items()},
           'saved_variables_verbatim': 'ctx.saved_variables' in segments['LayerNormFunction']['source'],
           'saved_variables_warning': deprecations, 'basicsr_or_lmdb_imported': False}
    require(out['saved_variables_verbatim'], 'saved_variables not verbatim')
    # fp64 gradcheck: LayerNormFunction and a whole NAFBlock
    torch.manual_seed(0)
    x = torch.randn(2, 6, 4, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gc_ln = torch.autograd.gradcheck(lambda a, c, d: LNF.apply(a, c, d, 1e-6), (x, w, b), eps=1e-6, atol=1e-6)
        blk64 = NAF(4).double()
        with torch.no_grad():
            blk64.beta.normal_()
            blk64.gamma.normal_()
        xb = torch.randn(1, 4, 5, 5, dtype=torch.float64, requires_grad=True)
        gc_blk = torch.autograd.gradcheck(blk64, (xb,), eps=1e-6, atol=1e-6)
    require(gc_ln and gc_blk, 'fp64 gradcheck')
    out['gradcheck_fp64'] = {'LayerNormFunction': gc_ln, 'NAFBlock': gc_blk}
    # native-autograd parity (LayerNorm alone and whole NAFBlock), fp64 and fp32
    parity = {}
    for dt in (torch.float64, torch.float32):
        torch.manual_seed(1)
        x = torch.randn(4, 32, 16, 16, dtype=dt) * 3 + 1
        w, b, g = torch.randn(32, dtype=dt), torch.randn(32, dtype=dt), torch.randn(4, 32, 16, 16, dtype=dt)
        a = [t.clone().requires_grad_(True) for t in (x, w, b)]
        r = [t.clone().requires_grad_(True) for t in (x, w, b)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = LNF.apply(*a, 1e-6)
            y1.backward(g)
        y2 = ln_native(*r, 1e-6)
        y2.backward(g)
        ln = {'forward': float((y1 - y2).detach().abs().max()), 'grad_x': float((a[0].grad - r[0].grad).abs().max()),
              'grad_weight': float((a[1].grad - r[1].grad).abs().max()),
              'grad_bias': float((a[2].grad - r[2].grad).abs().max()),
              'scale': {'forward': float(y2.detach().abs().max()), 'grad_x': float(r[0].grad.abs().max()),
                        'grad_weight': float(r[1].grad.abs().max()), 'grad_bias': float(r[2].grad.abs().max())}}
        torch.manual_seed(2)
        blk = NAF(32).to(dt)
        with torch.no_grad():
            blk.beta.normal_()
            blk.gamma.normal_()
        ref = copy.deepcopy(blk)
        ref.norm1, ref.norm2 = LN2dNative(32).to(dt), LN2dNative(32).to(dt)
        ref.norm1.load_state_dict(blk.norm1.state_dict())
        ref.norm2.load_state_dict(blk.norm2.state_dict())
        require(list(ref.state_dict()) == list(blk.state_dict()), 'NAFBlock state_dict keys')
        xin = torch.randn(4, 32, 32, 32, dtype=dt)
        x1, x2 = xin.clone().requires_grad_(True), xin.clone().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            o1 = blk(x1)
            gg = torch.randn_like(o1)
            o1.backward(gg)
        o2 = ref(x2)
        o2.backward(gg)
        require(torch.isfinite(o1).all() and torch.isfinite(x1.grad).all(), 'NAFBlock finite')
        pg = max(float((p1.grad - p2.grad).abs().max()) for p1, p2 in zip(blk.parameters(), ref.parameters()))
        pscale = max(float(p2.grad.abs().max()) for p2 in ref.parameters())
        nb = {'forward': float((o1 - o2).detach().abs().max()), 'grad_x': float((x1.grad - x2.grad).abs().max()),
              'grad_params': pg, 'scale': {'forward': float(o2.detach().abs().max()), 'grad_x': float(x2.grad.abs().max()),
                                           'grad_params': pscale}}
        name = 'float64' if dt == torch.float64 else 'float32'
        parity[name] = {'LayerNorm': ln, 'NAFBlock': nb}
        for part in (ln, nb):
            for k, v in part.items():
                if k == 'scale':
                    continue
                if dt == torch.float64:
                    require(v <= NAF_FP64_GATE, f'fp64 parity {k} {v}')
                else:
                    require(v <= NAF_FP32_REL_GATE * max(part['scale'][k], 1.0), f'fp32 parity {k} {v}')
    out['native_autograd_parity'] = parity
    out['parity_gates'] = {'float64_max_abs': NAF_FP64_GATE, 'float32_rel_to_max_magnitude': NAF_FP32_REL_GATE}
    out['nafblock_params_c32'] = sum(p.numel() for p in NAF(32).parameters())
    return out


# ----------------------------------------------------------------------------- R-04 Level 1
def level1(imgs):
    names = sorted(imgs)
    wfx = tp.pil_bicubic_matrix()
    wfx_pil = impulse_matrix(lambda e: np.asarray(Image.fromarray(e).resize((224, 2), Image.BICUBIC)), 256)
    waf = tp.area_matrix(256, 112)
    waf_cv = impulse_matrix(lambda e: cv2.resize(e, (112, 2), interpolation=cv2.INTER_AREA), 256)
    wpr = tp.area_matrix(256, 224)
    wpr_cv = impulse_matrix(lambda e: cv2.resize(e, (224, 2), interpolation=cv2.INTER_AREA), 256)
    tfx = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                     T.Normalize(tp.IMAGENET_MEAN, tp.IMAGENET_STD)])
    from gpatbench.probe.preprocess import frozen_probe_input
    fx_lsb, fx_norm, af, hp, pr = [], [], [], [], []
    per_image = {}
    for n in names:
        u8 = imgs[n]
        x = to_x(u8)
        ref_u8 = np.asarray(Image.fromarray(u8).resize((224, 224), Image.BICUBIC)).astype(np.float64)
        ref_norm = tfx(Image.fromarray(u8))[None]
        unit = tp.facexformer_unit(x)[0].numpy().transpose(1, 2, 0) * 255.0
        cand = tp.facexformer_input(x)
        d_lsb = unit - ref_u8
        d_norm = (cand - ref_norm).numpy()
        img = cv2.resize(u8, (112, 112), interpolation=cv2.INTER_AREA)
        ref_af = torch.tensor(np.ascontiguousarray(((np.asarray(img)[:, :, ::-1] / 255.0 - 0.5) / 0.5)
                                                   .transpose(2, 0, 1))[None]).float()
        d_af = (tp.adaface_input(x) - ref_af).numpy()
        xf = x[0].numpy().transpose(1, 2, 0)
        ref_hp = xf - cv2.GaussianBlur(xf, (9, 9), 1.5, borderType=cv2.BORDER_REFLECT_101)
        d_hp = tp.highpass(x)[0].numpy() - ref_hp.transpose(2, 0, 1)
        d_pr = tp.artifact_probe_input_float(x)[0].numpy() - frozen_probe_input(u8)
        fx_lsb.append(d_lsb.ravel())
        fx_norm.append(d_norm.ravel())
        af.append(d_af.ravel())
        hp.append(d_hp.ravel())
        pr.append(d_pr.ravel())
        per_image[n] = {'fx_lsb_max': float(np.abs(d_lsb).max()), 'fx_norm_max': float(np.abs(d_norm).max()),
                        'adaface_max': float(np.abs(d_af).max())}
    cat = np.concatenate
    measured = {
        'facexformer_matrix_vs_library_float_operator_max_abs': float(np.abs(wfx - wfx_pil).max()),
        'facexformer_clip_emulating_vs_frozen_pil_uint8_max_lsb': float(np.abs(cat(fx_lsb)).max()),
        'facexformer_clip_emulating_vs_frozen_normalized_max_abs': float(np.abs(cat(fx_norm)).max()),
        'adaface_matrix_vs_library_float_operator_max_abs': float(np.abs(waf - waf_cv).max()),
        'adaface_adapter_vs_frozen_cv2_uint8_max_abs': float(np.abs(cat(af)).max()),
        'highpass_float_path_max_abs': float(np.abs(cat(hp)).max()),
        'artifact_probe_float_preprocessing_max_abs': float(np.abs(cat(pr)).max())}
    diagnostics = {'facexformer_uint8_lsb': stats(cat(fx_lsb)), 'facexformer_normalized': stats(cat(fx_norm)),
                   'adaface_input': stats(cat(af)), 'highpass_256': stats(cat(hp)),
                   'artifact_probe_float': stats(cat(pr)),
                   'probe_area_matrix_vs_cv2_float32_max_abs': float(np.abs(wpr - wpr_cv).max()),
                   'facexformer_matrix_max_row_abs_sum': float(np.abs(wfx).sum(1).max()),
                   'gaussian_kernel_vs_cv2_max_abs': float(np.abs(tp.gaussian_kernel_1d() -
                                                                  cv2.getGaussianKernel(9, 1.5, cv2.CV_64F).ravel()).max())}
    verdict = {k: measured[k] <= GATES[k] for k in GATES}
    require(all(verdict.values()), 'Level-1 gate: ' + json.dumps({k: measured[k] for k, v in verdict.items() if not v}))
    return {'corpus': {'seed': CORPUS_SEED, 'n_images': len(names), 'names_sha256': sha_bytes('\n'.join(names).encode()),
                       'dataset_images': 0},
            'gates': GATES, 'measured': measured, 'pass': verdict, 'diagnostics_not_thresholds': diagnostics,
            'per_image_max': per_image}


def load_facexformer_frozen_random():
    from gpatbench.preprocess.aux_models import FACEXFORMER_CODE_SHA256, verify_code
    import torchvision
    verify_code(FX_CODE, FACEXFORMER_CODE_SHA256)
    sys.path.insert(0, str(FX_CODE))
    try:
        import network.models.facexformer as fxm
        fxm.swin_b = lambda weights=None, **k: torchvision.models.swin_b(weights=None, **k)
        torch.manual_seed(DETERMINISM_SEED)
        model = fxm.FaceXFormer()
    finally:
        sys.path.remove(str(FX_CODE))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def gradient_structure(imgs):
    import torchvision
    require(RESNET18.is_file() and sha_bytes(RESNET18.read_bytes()) == RESNET18_SHA, 'F_art weights')
    fart = torchvision.models.resnet18()
    fart.load_state_dict(torch.load(RESNET18, map_location='cpu', weights_only=True))
    fart.fc = nn.Identity()
    fart.eval()
    for p in fart.parameters():
        p.requires_grad_(False)
    fx = load_facexformer_frozen_random()
    ada = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.PReLU(8), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 16))
    ada.eval()
    for p in ada.parameters():
        p.requires_grad_(False)
    out = {'F_art': 'torchvision ResNet-18 IMAGENET1K_V1 ' + RESNET18_SHA,
           'FaceXFormer': 'pinned architecture (code hash-verified), random frozen weights',
           'AdaFace': 'frozen stand-in network (AdaFace code not on the laptop)', 'cases': {}}
    task = torch.tensor([0])
    for n in INTERIOR:
        x_t = to_x(imgs[n])
        x_hat = (x_t + 0.01 * torch.tanh(torch.randn_like(x_t))).requires_grad_(True)
        with torch.no_grad():
            t_fart = fart(tp.highpass(x_t))
            t_ada = ada(tp.adaface_input(x_t))
            t_fx = fx(tp.facexformer_input(x_t), None, task)[-1]
        loss = ((fart(tp.highpass(x_hat)) - t_fart).abs().mean() + (ada(tp.adaface_input(x_hat)) - t_ada).abs().mean()
                + (fx(tp.facexformer_input(x_hat), None, task)[-1] - t_fx).abs().mean())
        loss.backward()
        g = x_hat.grad
        case = {'input_grad_finite': bool(torch.isfinite(g).all()), 'input_grad_abs_sum': float(g.abs().sum()),
                'teacher_param_grads_none': all(p.grad is None for m in (fart, fx, ada) for p in m.parameters()),
                'targets_detached': not (t_fart.requires_grad or t_ada.requires_grad or t_fx.requires_grad)}
        require(case['input_grad_finite'] and case['input_grad_abs_sum'] > 0 and case['teacher_param_grads_none']
                and case['targets_detached'], 'gradient structure ' + n)
        out['cases'][n] = case
    return out


def equivalence_pre_evidence():
    """N-09 CPU pre-evidence only; adoption requires the qualified CUDA stack."""
    torch.manual_seed(3)
    x = torch.randn(4, 512, 16, 16, dtype=torch.float32)
    a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = F.adaptive_avg_pool2d(a, 8), F.avg_pool2d(b, 2, 2)
    g = torch.randn_like(ya)
    ya.backward(g)
    yb.backward(g)
    pool = {'forward_max_abs': float((ya - yb).detach().abs().max()), 'grad_max_abs': float((a.grad - b.grad).abs().max())}
    res = {}
    for n in (8, 16, 64, 128):
        x = torch.randn(2, 3, n, n, dtype=torch.float32)
        a, b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        ya = F.interpolate(a, scale_factor=2, mode='bilinear', align_corners=False)
        m = torch.tensor(rc.bilinear_x2_matrix(n), dtype=torch.float32)
        yb = torch.einsum('oh,nchw,pw->ncop', m, b, m)
        g = torch.randn_like(ya)
        ya.backward(g)
        yb.backward(g)
        res[str(n)] = {'forward_max_abs': float((ya - yb).detach().abs().max()), 'grad_max_abs': float((a.grad - b.grad).abs().max())}
    ok = pool['forward_max_abs'] <= EQUIV_FWD_GATE and pool['grad_max_abs'] <= EQUIV_GRAD_GATE and all(
        r['forward_max_abs'] <= EQUIV_FWD_GATE and r['grad_max_abs'] <= EQUIV_GRAD_GATE for r in res.values())
    require(ok, 'N-09 CPU equivalence pre-evidence')
    return {'status': 'CPU_PRE_EVIDENCE_ONLY_NOT_ADOPTED', 'gates': {'forward': EQUIV_FWD_GATE, 'grad': EQUIV_GRAD_GATE},
            'adaptive_avg_pool2d_16_to_8_vs_avg_pool2d_2_2': pool, 'bilinear_x2_vs_fixed_matrix': res}


def contract_checks():
    lr = {u: rc.main_lr(u) for u in (1, 2, 5525, 5526, 16575, 16576, 66299, 66300)}
    wl = {s: rc.attack_warmup_lr(s) for s in (1, 2, 695, 1389, 1390)}
    cur = {u: rc.curriculum(u) for u in (1, 2763, 5525, 5526, 16575, 16576, 66300)}
    require(lr[1] == 0.0 and abs(lr[5525] - 2e-4) < 1e-18 and abs(lr[5526] - 2e-4) < 1e-18 and
            abs(lr[66300] - 2e-6) < 1e-18, 'main LR endpoints')
    require(abs(wl[1] - 1e-4) < 1e-18 and abs(wl[1390]) < 1e-18, 'attack warmup LR endpoints')
    require(cur[1]['s_hf'] == 0.02 and abs(cur[5525]['s_hf'] - 0.05) < 1e-15 and cur[5526]['s_hf'] == 0.10 and
            cur[16576]['s_hf'] == 0.15, 'curriculum')
    return {'main_lr': {str(k): v for k, v in lr.items()}, 'attack_warmup_lr': {str(k): v for k, v in wl.items()},
            'curriculum': {str(k): v for k, v in cur.items()},
            'updates_per_epoch': rc.UPDATES_PER_EPOCH, 'total_updates': rc.TOTAL_UPDATES,
            'tail_group_weights': rc.group_weights([4, 2])}


def run():
    det = rc.apply_qualification_determinism(DETERMINISM_SEED, gpu=False)
    env = environment()
    imgs = corpus()
    result = {'schema': 'gpat.m7c2a.cpu_qualification', 'milestone': 'M7C2a', 'status': 'PASS', 'environment': env,
              'determinism': det, 'pywavelets_provenance': pywavelets_provenance(),
              'adversarial_bce': adversarial_checks(), 'ptwt': ptwt_checks(), 'nafnet': naf_checks(), 'r04_level1': level1(imgs),
              'r04_gradient_structure': gradient_structure(imgs), 'n09_cpu_pre_evidence': equivalence_pre_evidence(),
              'contract': contract_checks(), 'level2_executed': False, 'gpu_used': False,
              'dataset_images_read': 0, 'teacher_checkpoints_read': ['F_art ResNet-18 IMAGENET1K_V1 (local cache)']}
    again = level1(imgs)
    require(again['measured'] == result['r04_level1']['measured'], 'Level-1 bitwise repeatability')
    result['r04_level1']['repeat_bitwise_identical'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    if args.write:
        (ROOT / EVIDENCE).write_text(text)
        print(json.dumps({'status': 'WRITTEN', 'path': EVIDENCE}))
    else:
        print(text)


if __name__ == '__main__':
    main()
