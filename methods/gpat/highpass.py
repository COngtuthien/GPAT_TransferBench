"""GPAT fixed high-pass HP(x) = x - GaussianBlur(x) for E_art, F_art and PatchGAN D (A10 D10.17, M7C2a N-01).

Input is the GPAT tensor in [-1, 1]: 9x9 separable Gaussian, sigma 1.5, reflect-101 padding, per RGB channel, fp32,
no ImageNet normalization and no [0, 1] remap. The single definition is the M7C2a reference
`teacher_preprocess.highpass`; this wrapper only pins fp32 outside autocast. ArtifactProbe keeps its own adapter.
"""
from __future__ import annotations

import torch

from methods.gpat.teacher_preprocess import highpass as _reference_highpass

KERNEL = 9
SIGMA = 1.5


def highpass(x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _reference_highpass(x.float())
