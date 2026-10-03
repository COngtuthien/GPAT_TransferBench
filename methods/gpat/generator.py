"""GPAT residual generator G_res = NAFResidualUNet (spec 9.3, A10 D10.8-D10.12, M7C2a SHAPE_TRACE).

NAFBlock / LayerNorm2d / SimpleGate come only from the hash-verified pinned NAFNet source (M7C2a R-05 loader).

    input   [N, 12, 128, 128] = [LL_t, LH_t, HL_t, HH_t]; intro Conv3x3 12 -> 32
    Enc1..4 C = 32, 64, 128, 256 at 128, 64, 32, 16; blocks 2, 2, 4, 8; FiLM; skip taken AFTER FiLM;
            downsample Conv2d(C, 2C, k2, s2) (A10 D10.9)
    bottleneck  target [N, 512, 8, 8]; spatial z_a [N, 256, 16, 16] -> Conv1x1 256 -> 512 -> AvgPool2d(2, 2)
            (exact equivalent of adaptive_avg_pool2d(8) on a 16x16 input); concat [N, 1024, 8, 8]; Conv1x1 1024 -> 512;
            12 NAFBlocks; no FiLM
    Dec4..1 bilinear x2 (align_corners False) -> Conv3x3 (bias) halving C -> + encoder skip -> 2 NAFBlocks -> FiLM
    output  Conv3x3 32 -> 13 = raw dLL 0..2, dLH 3..5, dHL 6..8, dHH 9..11, mask logit 12 (no activation here)

FiLM: Linear(512, 2C) -> (gamma, beta); (1 + gamma) * h + beta. Its affine layer is zero-initialized so the initial
modulation is the identity (no frozen authority specifies otherwise; owner instruction M7C2b section 12).
Other layers keep the pinned/PyTorch default initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.gpat import naf_source

WIDTH = 32
ENC_BLOCKS = (2, 2, 4, 8)
MIDDLE_BLOCKS = 12
DEC_BLOCKS = (2, 2, 2, 2)
IN_CHANNELS = 12
OUT_CHANNELS = 13
CODE_DIM = 512
SPATIAL_CODE_CHANNELS = 256
BAND_SIZE = 128
OUTPUT_SLICES = {'delta_LL': (0, 3), 'delta_LH': (3, 6), 'delta_HL': (6, 9), 'delta_HH': (9, 12), 'mask_logit': (12, 13)}

_NAF = {}


def naf_symbols(root=None) -> dict:
    """The four pinned NAFNet symbols (verified and bound once per process and source root)."""
    key = str(root)
    if key not in _NAF:
        _NAF[key] = naf_source.load(root)
    return _NAF[key]


class FiLM(nn.Module):
    def __init__(self, channels: int, code_dim: int = CODE_DIM):
        super().__init__()
        self.channels = channels
        self.affine = nn.Linear(code_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h, z):
        gamma, beta = self.affine(z).chunk(2, dim=1)
        return (1.0 + gamma[:, :, None, None]) * h + beta[:, :, None, None]


class _Stage(nn.Module):
    def __init__(self, block, channels: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(*[block(channels) for _ in range(n_blocks)])
        self.film = FiLM(channels)

    def forward(self, h, z):
        return self.film(self.blocks(h), z)


class _Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=True)

    def forward(self, h):
        return self.conv(F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False))


class NAFResidualUNet(nn.Module):
    def __init__(self, naf_root=None):
        super().__init__()
        block = naf_symbols(naf_root)['NAFBlock']
        chans = [WIDTH * 2 ** i for i in range(4)]                       # 32, 64, 128, 256
        self.intro = nn.Conv2d(IN_CHANNELS, WIDTH, kernel_size=3, padding=1, bias=True)
        self.encoders = nn.ModuleList(_Stage(block, c, n) for c, n in zip(chans, ENC_BLOCKS))
        self.downs = nn.ModuleList(nn.Conv2d(c, 2 * c, kernel_size=2, stride=2) for c in chans)
        bottleneck = 2 * chans[-1]                                       # 512
        self.code_proj = nn.Conv2d(SPATIAL_CODE_CHANNELS, bottleneck, kernel_size=1)
        self.code_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fuse = nn.Conv2d(2 * bottleneck, bottleneck, kernel_size=1)
        self.middle = nn.Sequential(*[block(bottleneck) for _ in range(MIDDLE_BLOCKS)])
        self.ups = nn.ModuleList(_Up(2 * c, c) for c in reversed(chans))
        self.decoders = nn.ModuleList(_Stage(block, c, n) for c, n in zip(reversed(chans), DEC_BLOCKS))
        self.ending = nn.Conv2d(WIDTH, OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

    def forward(self, target_bands: torch.Tensor, z_a: torch.Tensor, spatial_code: torch.Tensor) -> torch.Tensor:
        n = target_bands.shape[0]
        if tuple(target_bands.shape[1:]) != (IN_CHANNELS, BAND_SIZE, BAND_SIZE):
            raise ValueError(f'G_res expects [N, 12, 128, 128], got {tuple(target_bands.shape)}')
        if tuple(z_a.shape) != (n, CODE_DIM) or tuple(spatial_code.shape) != (n, SPATIAL_CODE_CHANNELS, 16, 16):
            raise ValueError('G_res expects z_a [N, 512] and spatial code [N, 256, 16, 16]')
        h = self.intro(target_bands)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            h = enc(h, z_a)
            skips.append(h)                                              # skip after the stage FiLM
            h = down(h)
        code = self.code_pool(self.code_proj(spatial_code))              # [N, 512, 8, 8]
        h = self.middle(self.fuse(torch.cat([h, code], dim=1)))
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = dec(up(h) + skip, z_a)                                   # skip added before blocks; FiLM after
        return self.ending(h)


def split_output(raw: torch.Tensor) -> dict:
    """Raw [N, 13, 128, 128] -> named channel groups (A10 D10.12 / M7B R-01 output channel mapping)."""
    if raw.dim() != 4 or raw.shape[1] != OUT_CHANNELS:
        raise ValueError(f'G_res output must be [N, 13, h, w], got {tuple(raw.shape)}')
    return {k: raw[:, a:b] for k, (a, b) in OUTPUT_SLICES.items()}
