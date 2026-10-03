"""GPAT PatchGAN discriminator D (spec 10.4, A10 D10.13/D10.14, M7C2a N-01).

Input = concat(RGB image, HP(image)) -> [N, 6, 256, 256]; HP is the fixed GPAT high-pass on the [-1, 1] tensor (fp32).
pix2pix 70x70 PatchGAN: Conv4x4 s2 6->64, LeakyReLU(0.2); Conv4x4 s2 64->128, InstanceNorm, LeakyReLU; Conv4x4 s2
128->256, InstanceNorm, LeakyReLU; Conv4x4 s1 256->512, InstanceNorm, LeakyReLU; Conv4x4 s1 512->1. Padding 1 on every
conv; output [N, 1, 30, 30] logits (no sigmoid). Conv bias and InstanceNorm follow the pix2pix NLayerDiscriminator
convention for norm='instance' (InstanceNorm2d affine=False, no running stats, so every conv keeps its bias).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.gpat.highpass import highpass

KERNEL = 4
PADDING = 1
SLOPE = 0.2
OUTPUT_SIZE = 30
# (in, out, stride, instance_norm, leaky_relu)
LAYERS = ((6, 64, 2, False, True), (64, 128, 2, True, True), (128, 256, 2, True, True), (256, 512, 1, True, True),
          (512, 1, 1, False, False))


def discriminator_input(image: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=image.device.type, enabled=False):
        return torch.cat([image.float(), highpass(image)], dim=1)


class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout, stride, norm, act in LAYERS:
            layers.append(nn.Conv2d(cin, cout, kernel_size=KERNEL, stride=stride, padding=PADDING, bias=True))
            if norm:
                layers.append(nn.InstanceNorm2d(cout, affine=False, track_running_stats=False))
            if act:
                layers.append(nn.LeakyReLU(SLOPE, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image [N, 3, 256, 256] in [-1, 1] -> patch logits [N, 1, 30, 30]."""
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f'D expects an RGB image [N, 3, H, W], got {tuple(image.shape)}')
        return self.model(discriminator_input(image))
