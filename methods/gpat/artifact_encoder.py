"""GPAT artifact encoder E_art (spec 9.2, A10 D10.16/D10.24, M7C2a N-01/N-06).

torchvision ResNet-18 architecture with conv1 widened 3 -> 12 input channels and no FC/avgpool head. Input channels:
0..2 HP(x_s) RGB, 3..5 / 6..8 / 9..11 the source LH / HL / HH bands upsampled bilinear x2 (align_corners False).
Outputs: spatial artifact code = layer3 [N, 256, 16, 16]; layer4 [N, 512, 8, 8]; z_a = L2Normalize(GAP(layer4)) [N, 512].
BatchNorm follows normal PyTorch train/eval semantics (the caller sets the mode; N-06: train mode during training).

Scientific initialization is IMAGENET1K_V1 (`imagenet_state_dict` + `apply_pretrained_init`). The weight file is never
downloaded: it is read from an explicit local path (default: the models/registry.yaml path) and SHA-256 verified.
Synthetic tests inject a 3-channel state dict through `apply_pretrained_init` instead.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from methods.gpat.highpass import highpass

IN_CHANNELS = 12
SPATIAL_CHANNELS = 256
CODE_DIM = 512
IMAGENET_WEIGHT_FILE = 'resnet18-f37072fd.pth'
IMAGENET_WEIGHT_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'   # models/registry.yaml
REGISTRY_WEIGHT_PATH = '/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/backbones/torchvision/' + IMAGENET_WEIGHT_FILE
_BODY = ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4')


class PretrainedWeightError(RuntimeError):
    """The IMAGENET1K_V1 weight file is missing or differs from the registry digest (never downloaded)."""


def encoder_input(x_s: torch.Tensor, lh_s: torch.Tensor, hl_s: torch.Tensor, hh_s: torch.Tensor) -> torch.Tensor:
    """[N, 12, 256, 256] fp32: HP RGB + bilinear-x2 LH_s, HL_s, HH_s (A10 D10.16)."""
    with torch.autocast(device_type=x_s.device.type, enabled=False):
        up = [F.interpolate(b.float(), scale_factor=2, mode='bilinear', align_corners=False) for b in (lh_s, hl_s, hh_s)]
        return torch.cat([highpass(x_s), *up], dim=1)


def widen_conv1(weight_rgb: torch.Tensor, in_channels: int = IN_CHANNELS) -> torch.Tensor:
    """[64, 3, 7, 7] -> [64, 12, 7, 7]: channels 0:3 pretrained, each extra = mean over RGB / 3 (A10 D10.24)."""
    if weight_rgb.dim() != 4 or weight_rgb.shape[1] != 3:
        raise ValueError('expected a 3-channel conv1 weight')
    extra = (weight_rgb.mean(dim=1, keepdim=True) / 3.0).expand(-1, in_channels - 3, -1, -1)
    return torch.cat([weight_rgb, extra], dim=1).contiguous()


class ArtifactEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for name in _BODY:
            self.add_module(name, getattr(base, name))

    def forward(self, inp: torch.Tensor):
        """inp [N, 12, 256, 256] -> (spatial layer3 [N, 256, 16, 16], layer4 [N, 512, 8, 8], z_a [N, 512])."""
        if inp.dim() != 4 or inp.shape[1] != IN_CHANNELS:
            raise ValueError(f'E_art expects [N, 12, H, W], got {tuple(inp.shape)}')
        h = self.maxpool(self.relu(self.bn1(self.conv1(inp))))
        spatial = self.layer3(self.layer2(self.layer1(h)))
        l4 = self.layer4(spatial)
        z_a = F.normalize(l4.float().mean(dim=(-2, -1)), p=2.0, dim=1)
        return spatial, l4, z_a


def apply_pretrained_init(encoder: ArtifactEncoder, state_rgb: dict) -> dict:
    """Load a 3-channel ResNet-18 state dict (fc.* ignored) with the frozen conv1 widening; returns what was used."""
    own = encoder.state_dict()
    new = {}
    for k, v in state_rgb.items():
        if k.startswith('fc.'):
            continue
        if k not in own:
            raise KeyError(f'unexpected ResNet-18 key {k}')
        new[k] = widen_conv1(v) if k == 'conv1.weight' else v
    # the official IMAGENET1K_V1 file predates BatchNorm num_batches_tracked; those counters keep their fresh value 0
    missing = sorted(set(own) - set(new))
    if any(not k.endswith('num_batches_tracked') for k in missing):
        raise KeyError(f'pretrained state misses {missing}')
    encoder.load_state_dict(new, strict=True)          # BatchNorm restores an absent num_batches_tracked itself
    return {'keys': len(new), 'ignored': sorted(k for k in state_rgb if k.startswith('fc.')),
            'num_batches_tracked_not_in_source': len(missing)}


def imagenet_state_dict(path=None) -> dict:
    """Read the local IMAGENET1K_V1 file (default: registry path) after SHA-256 verification; never downloads."""
    p = Path(path if path is not None else REGISTRY_WEIGHT_PATH)
    if not p.is_file():
        raise PretrainedWeightError(f'IMAGENET1K_V1 weight file unavailable (no download): {p}')
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != IMAGENET_WEIGHT_SHA256:
        raise PretrainedWeightError(f'IMAGENET1K_V1 weight SHA-256 mismatch: {p}')
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def build_artifact_encoder(*, pretrained_state: dict | None = None, weight_path=None) -> ArtifactEncoder:
    """Production: IMAGENET1K_V1 from the verified local file. Tests: pass a synthetic 3-channel `pretrained_state`."""
    encoder = ArtifactEncoder()
    apply_pretrained_init(encoder, pretrained_state if pretrained_state is not None else imagenet_state_dict(weight_path))
    return encoder
