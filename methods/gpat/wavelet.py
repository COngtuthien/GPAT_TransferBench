"""GPAT level-1 Haar wavelet transform (spec 9.1, A10 D11, M7B R-01): ptwt, mode reflect, per RGB channel, fp32.

Canonical band mapping: cA -> LL, cH -> LH, cV -> HL, cD -> HH over axes (-2, -1). The transform always runs in fp32
outside autocast (M7B R-06 fp32_outside_autocast: DWT, IDWT).
"""
from __future__ import annotations

import ptwt
import pywt
import torch

WAVELET = pywt.Wavelet('haar')
MODE = 'reflect'
AXES = (-2, -1)
BANDS = ('LL', 'LH', 'HL', 'HH')


def dwt(x: torch.Tensor):
    """[N, 3, H, W] -> (LL, LH, HL, HH), each [N, 3, H/2, W/2] fp32."""
    if x.dim() != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError(f'dwt expects [N, C, H, W] with even H, W; got {tuple(x.shape)}')
    with torch.autocast(device_type=x.device.type, enabled=False):
        ca, (ch, cv, cd) = ptwt.wavedec2(x.float(), WAVELET, level=1, mode=MODE, axes=AXES)
    return ca, ch, cv, cd


def idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Inverse of `dwt`: (LL, LH, HL, HH) -> [N, 3, 2h, 2w] fp32."""
    with torch.autocast(device_type=ll.device.type, enabled=False):
        return ptwt.waverec2([ll.float(), (lh.float(), hl.float(), hh.float())], WAVELET, axes=AXES)


def concat_bands(ll, lh, hl, hh) -> torch.Tensor:
    """Band order [LL_rgb, LH_rgb, HL_rgb, HH_rgb] -> [N, 12, h, w] (A10 D10.12)."""
    return torch.cat((ll, lh, hl, hh), dim=1)
