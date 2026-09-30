"""Reference forms of the M7C2a owner-frozen implementation/runtime clarifications.

Authority: configs/amendments/gpat_m7c2a_implementation_resolution.yaml (layered on A10 and the M7B record).
These are the single definitions later GPAT code must call; they hold no architecture and no training loop.
Torch is imported lazily so the schedule and identity-order functions stay usable without it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

RESOLUTION = 'configs/amendments/gpat_m7c2a_implementation_resolution.yaml'

# ----------------------------------------------------------------------------- N-07 schedules (closed form)
TOTAL_UPDATES = 66300
UPDATES_PER_EPOCH = 1105
WARMUP_UPDATES = 5525
PEAK_LR = 2e-4
MIN_LR = 2e-6
ATTACK_WARMUP_STEPS = 1390
ATTACK_WARMUP_LR = 1e-4
STAGE2_FIRST, STAGE3_FIRST = 5526, 16576


def _update(u: int, last: int) -> int:
    if isinstance(u, bool) or not isinstance(u, int) or not 1 <= u <= last:
        raise ValueError(f'update index must be an int in 1..{last}, got {u!r}')
    return u


def main_lr(u: int) -> float:
    """G_OPT and D_OPT learning rate of generator update u (1-based)."""
    u = _update(u, TOTAL_UPDATES)
    if u <= WARMUP_UPDATES:
        return PEAK_LR * (u - 1) / (WARMUP_UPDATES - 1)
    t = (u - (WARMUP_UPDATES + 1)) / (TOTAL_UPDATES - (WARMUP_UPDATES + 1))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * t))


def attack_warmup_lr(s: int) -> float:
    """B1/B3 WARMUP_OPT learning rate of warmup step s (1-based); no additional LR warmup."""
    s = _update(s, ATTACK_WARMUP_STEPS)
    return 0.5 * ATTACK_WARMUP_LR * (1.0 + math.cos(math.pi * (s - 1) / (ATTACK_WARMUP_STEPS - 1)))


def curriculum(u: int) -> dict:
    """s_hf (runtime residual scale), lambda_adv, lambda_con, lambda_spec of generator update u."""
    u = _update(u, TOTAL_UPDATES)
    if u < STAGE2_FIRST:
        return {'stage': 1, 's_hf': 0.02 + 0.03 * (u - 1) / (WARMUP_UPDATES - 1),
                'lambda_adv': 0.0, 'lambda_con': 0.5, 'lambda_spec': 0.25}
    if u < STAGE3_FIRST:
        return {'stage': 2, 's_hf': 0.10, 'lambda_adv': 0.05, 'lambda_con': 1.0, 'lambda_spec': 0.5}
    return {'stage': 3, 's_hf': 0.15, 'lambda_adv': 0.10, 'lambda_con': 1.0, 'lambda_spec': 0.5}


# N-05: D trains from generator update 1 (its real/fake update runs even while lambda_adv = 0).
D_FIRST_UPDATE = 1
EMA_DECAY = 0.999
EMA_START_UPDATE = WARMUP_UPDATES          # EMA is active after the G step of updates > 5525 (A10 D15)

# ----------------------------------------------------------------------------- N-04 update order
JOINT_UPDATE_MICROBATCH = (
    'compute x_hat once',
    'G losses with D parameters gradient-disabled; G_SCALER.scale(w_i * L_G).backward()',
    'D loss on x_hat.detach() with D trainable; D_SCALER.scale(w_i * L_D).backward()',
)
JOINT_UPDATE_BOUNDARY = (
    'D_SCALER.unscale_(D_OPT)', 'clip_grad_norm_(D, 1.0)', 'D_SCALER.step(D_OPT)', 'D_SCALER.update()',
    'G_SCALER.unscale_(G_OPT)', 'clip_grad_norm_(G_OPT params, 1.0)', 'G_SCALER.step(G_OPT)', 'G_SCALER.update()',
    'zero_grad(G_OPT, D_OPT)', 'EMA update after the G step when active',
)


def group_weights(sizes) -> list:
    """Sample weights w_i = n_i / n of the physical microbatches of one optimizer update (A10 D15)."""
    n = sum(sizes)
    return [s / n for s in sizes]


# ----------------------------------------------------------------------------- N-07 parsing losses
DICE_CLASSES = tuple(range(1, 11))
DICE_EPS = 1e-6
KL_FLOOR = 1e-8
KL_WEIGHT = 0.1


def soft_dice_loss(logits_t, logits_h):
    import torch
    p_t = torch.softmax(logits_t.float(), dim=1)[:, 1:11]
    p_h = torch.softmax(logits_h.float(), dim=1)[:, 1:11]
    inter = (p_t * p_h).sum(dim=(-2, -1))
    dice = (2.0 * inter + DICE_EPS) / (p_t.sum(dim=(-2, -1)) + p_h.sum(dim=(-2, -1)) + DICE_EPS)
    return 1.0 - dice.mean()


def kl_loss(logits_t, logits_h):
    import torch
    p_t = torch.softmax(logits_t.float(), dim=1)
    p_h = torch.softmax(logits_h.float(), dim=1)
    kl = (p_t * (p_t.clamp_min(KL_FLOOR).log() - p_h.clamp_min(KL_FLOOR).log())).sum(dim=1)
    return kl.mean()


def parse_loss(logits_t, logits_h):
    return soft_dice_loss(logits_t, logits_h) + KL_WEIGHT * kl_loss(logits_t, logits_h)


# ----------------------------------------------------------------------------- N-07 spectra
RADIAL_BINS = 128
ORIENT_BINS = 8
SPEC_EPS = 1e-12


def _shifted_power(x):
    import torch
    f = torch.fft.fft2(x.float())
    return torch.fft.fftshift(f.real ** 2 + f.imag ** 2, dim=(-2, -1))


def _offsets(h, w, device):
    import torch
    dy = torch.arange(h, device=device, dtype=torch.float64) - h // 2
    dx = torch.arange(w, device=device, dtype=torch.float64) - w // 2
    return torch.meshgrid(dy, dx, indexing='ij')


def radial_bin_index(h=256, w=256, device='cpu'):
    import torch
    dy, dx = _offsets(h, w, device)
    return torch.floor(torch.sqrt(dx * dx + dy * dy)).long()


def orient_bin_index(h=256, w=256, device='cpu'):
    """(bin index 0..7, support mask) with DC excluded and 0 < radius <= 0.5 cycles/pixel."""
    import torch
    dy, dx = _offsets(h, w, device)
    fy, fx = dy / h, dx / w
    radius = torch.sqrt(fx * fx + fy * fy)
    support = (radius > 0) & (radius <= 0.5)
    theta = torch.remainder(torch.atan2(fy, fx), math.pi)
    idx = torch.clamp(torch.floor(theta / (math.pi / ORIENT_BINS)).long(), max=ORIENT_BINS - 1)
    return idx, support


def s_radial(x):
    """[N, C, H, W] -> [N, C, 128]: log1p power summed per integer radius 0..127, normalized per channel."""
    import torch
    n, c, h, w = x.shape
    logp = torch.log1p(_shifted_power(x))
    r = radial_bin_index(h, w, x.device).flatten()
    keep = r < RADIAL_BINS
    vals = logp.flatten(-2)[..., keep]
    out = torch.zeros(n, c, RADIAL_BINS, dtype=vals.dtype, device=x.device)
    out = out.index_add(2, r[keep], vals)
    return out / (out.sum(-1, keepdim=True) + SPEC_EPS)


def s_orient(x):
    """[N, C, H, W] -> [N, C, 8]: raw power summed per orientation bin over the support, normalized."""
    import torch
    n, c, h, w = x.shape
    power = _shifted_power(x)
    idx, support = orient_bin_index(h, w, x.device)
    keep = support.flatten()
    vals = power.flatten(-2)[..., keep]
    out = torch.zeros(n, c, ORIENT_BINS, dtype=vals.dtype, device=x.device)
    out = out.index_add(2, idx.flatten()[keep], vals)
    return out / (out.sum(-1, keepdim=True) + SPEC_EPS)


def spec_loss(x_hat, x_s, lambda_dir=0.5):
    """L_spec = mean L1(S_radial) + lambda_dir * mean L1(S_orient) (A10 D01; lambda_dir frozen 0.5)."""
    return ((s_radial(x_hat) - s_radial(x_s)).abs().mean() +
            lambda_dir * (s_orient(x_hat) - s_orient(x_s)).abs().mean())


# ----------------------------------------------------------------------------- adversarial BCE
def _at_least_fp32(t):
    import torch
    return t.to(torch.promote_types(t.dtype, torch.float32))   # R-06: BCE logits/reduction never below fp32


def d_bce_terms(real_logits, fake_logits):
    """(L_D_real, L_D_fake) = (mean BCE(real, 1), mean BCE(fake, 0)); fake logits come from D(x_hat.detach())."""
    import torch
    import torch.nn.functional as F
    real, fake = _at_least_fp32(real_logits), _at_least_fp32(fake_logits)
    return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real)),
            F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))


def d_loss(real_logits, fake_logits):
    """L_D = 0.5 * (L_D_real + L_D_fake) (M7C2A-OBS-01, owner implementation clarification)."""
    real, fake = d_bce_terms(real_logits, fake_logits)
    return 0.5 * (real + fake)


def g_adv_bce(fake_logits):
    """Mean BCE(D(x_hat), 1) with x_hat not detached and D parameters gradient-disabled."""
    import torch
    import torch.nn.functional as F
    fake = _at_least_fp32(fake_logits)
    return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))


# ----------------------------------------------------------------------------- N-03 face mask
MASK_SIZE = 256
DILATION_KERNEL = 15


def face_mask_dilated(parsing_argmax_224):
    """[N, 224, 224] categorical mask -> [N, 1, 256, 256] float {0, 1}: nearest-exact, classes 1..10, 15x15 dilation."""
    import torch.nn.functional as F
    m = parsing_argmax_224[:, None].float()
    m = F.interpolate(m, size=(MASK_SIZE, MASK_SIZE), mode='nearest-exact')
    fg = ((m >= 1) & (m <= 10)).float()
    return F.max_pool2d(fg, kernel_size=DILATION_KERNEL, stride=1, padding=DILATION_KERNEL // 2)


# ----------------------------------------------------------------------------- N-06 EMA state policy
def ema_update(ema_state: dict, live_state: dict, decay: float = EMA_DECAY) -> None:
    """In place: floating tensors -> decay * ema + (1 - decay) * live; non-floating tensors copied."""
    import torch
    if list(ema_state) != list(live_state):
        raise ValueError('EMA and live state_dict keys differ')
    with torch.no_grad():
        for k, live in live_state.items():
            ema = ema_state[k]
            if torch.is_floating_point(live):
                ema.mul_(decay).add_(live.detach(), alpha=1.0 - decay)
            else:
                ema.copy_(live)


# ----------------------------------------------------------------------------- N-08 LFErr
def lferr_internal(ll_syn_internal, ll_target):
    """Per-sample ||LL_syn_internal - LL_target||_1 / (||LL_target||_1 + 1e-8) on internal LL coefficients."""
    diff = (ll_syn_internal - ll_target).abs().flatten(1).sum(1)
    return diff / (ll_target.abs().flatten(1).sum(1) + 1e-8)


# ----------------------------------------------------------------------------- identity class order
IDENTITY_DATASETS = ('casia_fasd', 'msu_mfsd')
IDENTITY_CLASSES = 60


class IdentityMapError(ValueError):
    """The identity class map cannot be built under the frozen algorithm (STOP)."""


def identity_class_map(rows, source_manifest_sha256: str, expected: int = IDENTITY_CLASSES) -> dict:
    """rows: iterable of (dataset, source_subject) from valid TRAIN identity metadata of the identity datasets."""
    keys = set()
    for dataset, subject in rows:
        if not isinstance(dataset, str) or not isinstance(subject, str) or not subject:
            raise IdentityMapError('dataset and source_subject must be canonical manifest strings')
        if dataset not in IDENTITY_DATASETS:
            raise IdentityMapError(f'dataset {dataset!r} carries no identity supervision (DEV-022)')
        keys.add((dataset, subject))
    ordered = sorted(keys, key=lambda k: (k[0].encode('utf-8'), k[1].encode('utf-8')))
    if len(ordered) != expected:
        raise IdentityMapError(f'identity classes K={len(ordered)} != {expected}')
    classes = [{'index': i, 'dataset': d, 'source_subject': s} for i, (d, s) in enumerate(ordered)]
    canonical = json.dumps(classes, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return {'classes': classes, 'source_manifest_sha256': source_manifest_sha256,
            'mapping_sha256': hashlib.sha256(canonical).hexdigest()}


# ----------------------------------------------------------------------------- N-09 determinism
CUBLAS_WORKSPACE_CONFIG = ':4096:8'


def apply_qualification_determinism(seed: int, *, gpu: bool) -> dict:
    """Seed Python/NumPy/torch (CPU and CUDA) and enforce deterministic algorithms; never relaxed on failure."""
    import numpy as np
    import torch
    if gpu:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS_WORKSPACE_CONFIG
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    return {'seed': seed, 'gpu': gpu, 'use_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def bilinear_x2_matrix(n: int):
    """Fixed matrix equal to F.interpolate(scale_factor=2, mode='bilinear', align_corners=False) along one axis."""
    import numpy as np
    m = np.zeros((2 * n, n), np.float64)
    for i in range(2 * n):
        src = max((i + 0.5) / 2.0 - 0.5, 0.0)
        i0 = min(int(math.floor(src)), n - 1)
        i1 = min(i0 + 1, n - 1)
        lam = src - i0
        m[i, i0] += 1.0 - lam
        m[i, i1] += lam
    return m


# ----------------------------------------------------------------------------- shape-trace clarifications
SHAPE_TRACE = {'bottleneck_film': False, 'encoder_skip': 'after_stage_film',
               'decoder_skip_addition': 'before_decoder_nafblocks', 'decoder_film': 'after_decoder_nafblocks',
               'upsample': 'bilinear_x2_align_corners_false_then_conv3x3_bias_true'}
