"""GPAT output activations, wavelet-domain composition and artifact provenance map (spec 9.1/9.3/9.4, A10 D10.4/D10.5,
D12, M7B R-02). Everything here runs in fp32 outside autocast; x_hat is never clamped.

    delta_LL = tanh(raw_LL) * 0.05          delta_HF = tanh(raw_HF) * scale_hf (curriculum value, explicit)
    M = sigmoid(mask_logit) [N, 1, 128, 128], shared across RGB, no threshold
    LL_syn = LL_t + gamma * delta_LL        XX_syn = XX_t + M * delta_XX  (XX = LH, HL, HH)
    x_hat = IDWT(LL_syn, LH_syn, HL_syn, HH_syn)
    A = clip(M_256 * (0.5 * A_rgb + 0.5 * A_freq_256) / 2, 0, 1)

`artifact_scale` is the A10 D12 qualification hook ("artifact_scale_zero_disables delta_HF, delta_LL"): 1.0 is the only
production value; 0.0 zeroes every residual contribution at the composition boundary. It is not a hyperparameter.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.gpat import wavelet
from methods.gpat.generator import split_output

SCALE_LL = 0.05
SCALE_HF_FINAL = 0.15
GAMMA_MAIN = 0.0
GAMMA_ALLOWED = (0.0, 0.02, 0.05, 0.10)          # spec 9.1: 0.0 main; the others are ablations only
ARTIFACT_SCALE_ALLOWED = (0.0, 1.0)
ARTIFACT_NORMALIZER = 2.0


def _check(gamma, scale_hf, artifact_scale):
    if float(gamma) not in GAMMA_ALLOWED:
        raise ValueError(f'gamma {gamma!r} not in the frozen set {GAMMA_ALLOWED}')
    if not 0.0 <= float(scale_hf) <= SCALE_HF_FINAL:
        raise ValueError(f'scale_hf {scale_hf!r} outside the curriculum range [0, {SCALE_HF_FINAL}]')
    if float(artifact_scale) not in ARTIFACT_SCALE_ALLOWED:
        raise ValueError('artifact_scale is a D12 qualification hook: 1.0 (production) or 0.0 (zero residual)')


def activate(raw: torch.Tensor, scale_hf: float, artifact_scale: float = 1.0) -> dict:
    """Raw G_res output -> bounded deltas and the soft mask, fp32. artifact_scale 0.0 zeroes every delta."""
    with torch.autocast(device_type=raw.device.type, enabled=False):
        parts = {k: v.float() for k, v in split_output(raw).items()}
        out = {'delta_LL': torch.tanh(parts['delta_LL']) * SCALE_LL}
        for band in ('LH', 'HL', 'HH'):
            out['delta_' + band] = torch.tanh(parts['delta_' + band]) * float(scale_hf)
        if float(artifact_scale) == 0.0:
            out = {k: torch.zeros_like(v) for k, v in out.items()}
        out['M'] = torch.sigmoid(parts['mask_logit'])
        return out


def compose(target: tuple, deltas: dict, gamma: float) -> dict:
    """(LL_t, LH_t, HL_t, HH_t) + deltas + M -> synthetic bands and x_hat (no clamp)."""
    ll_t, lh_t, hl_t, hh_t = target
    m = deltas['M']
    with torch.autocast(device_type=ll_t.device.type, enabled=False):
        syn = {'LL_syn': ll_t + float(gamma) * deltas['delta_LL'],
               'LH_syn': lh_t + m * deltas['delta_LH'],
               'HL_syn': hl_t + m * deltas['delta_HL'],
               'HH_syn': hh_t + m * deltas['delta_HH']}
        syn['x_hat'] = wavelet.idwt(syn['LL_syn'], syn['LH_syn'], syn['HL_syn'], syn['HH_syn'])
    return syn


def _up2(t):
    return F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)


def artifact_map(x_hat, x_t, deltas: dict, gamma: float, *, components: bool = False):
    """A [N, 1, 256, 256] = clip(u / 2.0, 0, 1), u = M_256 * (0.5 * A_rgb + 0.5 * A_freq_256) (A10 D10.4)."""
    with torch.autocast(device_type=x_hat.device.type, enabled=False):
        a_rgb = (x_hat.float() - x_t.float()).abs().mean(dim=1, keepdim=True)
        a_freq_128 = (deltas['delta_LH'].abs() + deltas['delta_HL'].abs() + deltas['delta_HH'].abs()
                      + float(gamma) * deltas['delta_LL'].abs()).mean(dim=1, keepdim=True)
        a_freq_256 = _up2(a_freq_128)
        m_256 = _up2(deltas['M'])
        u = m_256 * (0.5 * a_rgb + 0.5 * a_freq_256)
        a = torch.clamp(u / ARTIFACT_NORMALIZER, 0.0, 1.0)
    if components:
        return a, {'A_rgb': a_rgb, 'A_freq_128': a_freq_128, 'A_freq_256': a_freq_256, 'M_256': m_256, 'u': u}
    return a


def generate(raw, target: tuple, x_t, *, scale_hf: float, gamma: float = GAMMA_MAIN, artifact_scale: float = 1.0):
    """Full composition boundary: activations -> composition -> x_hat -> A."""
    _check(gamma, scale_hf, artifact_scale)
    deltas = activate(raw, scale_hf, artifact_scale)
    syn = compose(target, deltas, gamma)
    return {**deltas, **syn, 'A': artifact_map(syn['x_hat'], x_t, deltas, gamma)}
