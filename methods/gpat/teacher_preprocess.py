"""R-04 differentiable teacher-input adapters (A10 D07 as refined by M7C2a).

Frozen references (gpatbench/preprocess/aux_models.py, gpatbench/probe/preprocess.py) are unchanged. These
adapters take the native GPAT float tensor x ([N, 3, 256, 256], RGB, range [-1, 1] nominal) and reproduce the
frozen resampling without uint8 rounding, so gradients reach x_hat. Teacher weights and code are not touched
here; this module builds only fixed resampling operators and the adapter transforms.

FaceXFormer: CLIP_EMULATING_DIFFERENTIABLE_COMPATIBILITY -- v = clamp((x + 1) / 2, 0, 1); horizontal PIL
  bicubic pass; clamp(0, 1); vertical pass; clamp(0, 1); ImageNet normalization. PIL clips after each pass
  (clip8), so the adapter is piecewise-linear, not a global linear operator.
AdaFace: AREA_MATRIX_DIFFERENTIABLE_COMPATIBILITY -- clamp(x, -1, 1); exact INTER_AREA 256->112 area
  matrix in the [-1, 1] domain; RGB->BGR flip.
ArtifactProbe (VAL only, no gradient): the frozen [0,1] path on float input (INTER_AREA 224, then
  x - GaussianBlur 9x9 sigma 1.5 reflect-101), unchanged by N-01.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CANONICAL = 256
FX_INPUT = 224
ADAFACE_INPUT = 112
PROBE_INPUT = 224
L2_SEED_PREFIX = 'M7C2A-R04-L2'
L2_REPLICAS = 8
L2_PER_DATASET = 64
L2_DATASETS = ('casia_fasd', 'msu_mfsd', 'siwmv2')


def pil_bicubic_matrix(in_size: int = CANONICAL, out_size: int = FX_INPUT) -> np.ndarray:
    """PIL Resample.c precompute_coeffs for BICUBIC (a = -0.5), float64, rows normalized to 1."""
    a = -0.5

    def kernel(x):
        x = abs(x)
        if x < 1.0:
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        if x < 2.0:
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        return 0.0

    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ss = 1.0 / filterscale
    w = np.zeros((out_size, in_size), np.float64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        k = np.array([kernel((x + xmin - center + 0.5) * ss) for x in range(xmax)])
        w[xx, xmin:xmin + xmax] = k / k.sum()
    return w


def area_matrix(in_size: int, out_size: int) -> np.ndarray:
    """Exact INTER_AREA downscale weights: overlap length / cell width, float64."""
    s = in_size / out_size
    w = np.zeros((out_size, in_size), np.float64)
    for i in range(out_size):
        a0, a1 = i * s, (i + 1) * s
        for j in range(int(math.floor(a0)), min(int(math.ceil(a1)), in_size)):
            overlap = min(a1, j + 1) - max(a0, j)
            if overlap > 1e-12:
                w[i, j] = overlap / s
    return w


def gaussian_kernel_1d(size: int = 9, sigma: float = 1.5) -> np.ndarray:
    i = np.arange(size) - (size - 1) / 2
    g = np.exp(-(i ** 2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _t(mat, like):
    import torch
    return torch.as_tensor(mat, dtype=torch.float32, device=like.device)


def facexformer_input(x, w=None):
    """Clip-emulating FaceXFormer adapter: [N,3,256,256] GPAT float -> ImageNet-normalized [N,3,224,224]."""
    import torch
    w = _t(pil_bicubic_matrix() if w is None else w, x)
    v = ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    h = torch.einsum('nchw,pw->nchp', v, w).clamp(0.0, 1.0)      # horizontal pass, PIL clip8
    o = torch.einsum('oh,nchp->ncop', w, h).clamp(0.0, 1.0)      # vertical pass, PIL clip8
    mean = torch.tensor(IMAGENET_MEAN, dtype=o.dtype, device=o.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=o.dtype, device=o.device).view(1, 3, 1, 1)
    return (o - mean) / std


def facexformer_unit(x, w=None):
    """The clip-emulating resampled image in [0, 1] before normalization (A(v) of the Level-2 contract)."""
    import torch
    w = _t(pil_bicubic_matrix() if w is None else w, x)
    v = ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    h = torch.einsum('nchw,pw->nchp', v, w).clamp(0.0, 1.0)
    return torch.einsum('oh,nchp->ncop', w, h).clamp(0.0, 1.0)


def adaface_input(x, w=None):
    """AdaFace adapter: [N,3,256,256] GPAT float -> BGR [N,3,112,112] in [-1, 1]."""
    import torch
    w = _t(area_matrix(CANONICAL, ADAFACE_INPUT) if w is None else w, x)
    xc = x.float().clamp(-1.0, 1.0)
    return torch.einsum('oh,nchw,pw->ncop', w, xc, w).flip(1)


def highpass(x):
    """GPAT canonical HP (N-01): x - GaussianBlur(x), 9x9, sigma 1.5, reflect-101, per channel, no normalization."""
    import torch
    import torch.nn.functional as F
    g = torch.as_tensor(gaussian_kernel_1d(), dtype=x.dtype, device=x.device)
    c, k = x.shape[1], g.numel()
    p = k // 2
    y = F.pad(x, (p, p, p, p), mode='reflect')
    y = F.conv2d(y, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(y, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x - y


def artifact_probe_input_float(x):
    """VAL-only float entry of the frozen ArtifactProbe path (R-03): no uint8, no clamp, no gradient."""
    import torch
    w = _t(area_matrix(CANONICAL, PROBE_INPUT), x)
    with torch.no_grad():
        v = (x.float() + 1.0) * 0.5
        r = torch.einsum('oh,nchw,pw->ncop', w, v, w)
        return highpass(r)


# ----------------------------------------------------------------------------- Level-2 contract helpers
def level2_subset(sample_ids_by_dataset: dict, per_dataset: int = L2_PER_DATASET) -> dict:
    """Per dataset, the up-to-`per_dataset` unique sample ids with the smallest SHA256(UTF-8 id) digests."""
    out = {}
    for dataset in L2_DATASETS:
        ids = sorted(set(sample_ids_by_dataset.get(dataset, ())))
        if any(not isinstance(s, str) for s in ids):
            raise ValueError('canonical sample ids must be strings')
        ids.sort(key=lambda s: (hashlib.sha256(s.encode('utf-8')).hexdigest(), s))
        out[dataset] = ids[:per_dataset]
    return out


def level2_seed(dataset: str, sample_id: str, k: int) -> int:
    msg = f'{L2_SEED_PREFIX}|{dataset}|{sample_id}|{int(k)}'.encode('utf-8')
    return int(hashlib.sha256(msg).hexdigest()[:16], 16) % 2 ** 63


def requantization_replica(unit, dataset: str, sample_id: str, k: int):
    """clip(round(255 * A(v) + u_k), 0, 255) / 255 with u_k ~ U[-0.5, 0.5) from the derived CPU seed."""
    import torch
    g = torch.Generator(device='cpu').manual_seed(level2_seed(dataset, sample_id, k))
    u = torch.rand(unit.shape, generator=g, dtype=torch.float64) - 0.5
    q = torch.round(unit.detach().double().cpu() * 255.0 + u).clamp(0.0, 255.0) / 255.0
    return q.to(dtype=torch.float32)
