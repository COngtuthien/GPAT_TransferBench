"""GPAT spectral summaries S_radial / S_orient (A10 D10.1/D10.2, M7C2a N-07), fp32 outside autocast.

The single definitions are the M7C2a reference forms in `runtime_contract` (vectorized index_add over integer bin
maps built with tensor ops; no Python loop over pixels). This module pins the GPAT input contract and fp32.

S_radial: log1p(|FFT2|^2), fftshift, r = floor(sqrt(dx^2 + dy^2)), bins r = 0..127 (DC included, r > 127 ignored),
          per-channel normalization to sum 1 (eps 1e-12) -> [N, 3, 128].
S_orient: raw |FFT2|^2, DC excluded, 0 < radius <= 0.5 cycles/pixel, theta = atan2(fy, fx) mod pi, 8 bins over
          [0, pi), per-channel normalization to sum 1 (eps 1e-12) -> [N, 3, 8].
"""
from __future__ import annotations

import torch

from methods.gpat import runtime_contract as rc

RADIAL_BINS = rc.RADIAL_BINS
ORIENT_BINS = rc.ORIENT_BINS
LAMBDA_DIR = 0.5            # frozen coefficient of S_orient inside L_spec (A10 D01)
SIZE = 256


def _check(x):
    if x.dim() != 4 or x.shape[1] != 3 or tuple(x.shape[-2:]) != (SIZE, SIZE):
        raise ValueError(f'spectral summaries expect [N, 3, 256, 256], got {tuple(x.shape)}')


def s_radial(x: torch.Tensor) -> torch.Tensor:
    _check(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        return rc.s_radial(x.float())


def s_orient(x: torch.Tensor) -> torch.Tensor:
    _check(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        return rc.s_orient(x.float())


def spec_loss(x_hat: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    """L_spec = mean L1(S_radial) + 0.5 * mean L1(S_orient); no further lambda_dir outside this term."""
    _check(x_hat)
    _check(x_s)
    with torch.autocast(device_type=x_hat.device.type, enabled=False):
        return rc.spec_loss(x_hat.float(), x_s.float(), lambda_dir=LAMBDA_DIR)
