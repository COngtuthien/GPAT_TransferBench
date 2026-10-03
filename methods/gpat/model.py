"""Static GPAT model facade (spec 9, A10, M7B, M7C2a): wiring only, no teacher, no loss, no optimizer, no training.

GPATCore owns E_art, G_res, the auxiliary heads enabled by the variant config and, optionally, PatchGAN D as a
separately owned submodule that the forward never calls. `forward_generator` returns every intermediate the later
loss/selection code needs. B0..B3 share one architecture; only the auxiliary heads differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from methods.gpat import composition, wavelet
from methods.gpat.artifact_encoder import ArtifactEncoder, build_artifact_encoder, encoder_input
from methods.gpat.config import GPATConfig
from methods.gpat.discriminator import PatchGANDiscriminator
from methods.gpat.generator import NAFResidualUNet
from methods.gpat.heads import AttackTypeHead, IdentityAdversaryHead

IMAGE_SHAPE = (3, 256, 256)


@dataclass
class GPATForward:
    source_bands: tuple
    target_bands: tuple
    encoder_input: torch.Tensor
    z_a: torch.Tensor
    spatial_code: torch.Tensor
    layer4: torch.Tensor
    raw: torch.Tensor
    delta_LL: torch.Tensor
    delta_LH: torch.Tensor
    delta_HL: torch.Tensor
    delta_HH: torch.Tensor
    M: torch.Tensor
    LL_syn: torch.Tensor
    LH_syn: torch.Tensor
    HL_syn: torch.Tensor
    HH_syn: torch.Tensor
    x_hat: torch.Tensor
    A: torch.Tensor
    scale_hf: float
    gamma: float
    attack_logits: Optional[torch.Tensor] = None
    identity_logits: Optional[torch.Tensor] = None


class GPATCore(nn.Module):
    def __init__(self, config: GPATConfig, *, pretrained_state: dict | None = None, weight_path=None,
                 naf_root=None, with_discriminator: bool = False):
        super().__init__()
        self.config = config
        self.e_art = build_artifact_encoder(pretrained_state=pretrained_state, weight_path=weight_path)
        self.g_res = NAFResidualUNet(naf_root)
        self.attack_head = AttackTypeHead() if config.attack_type_head else None
        self.identity_head = IdentityAdversaryHead() if config.identity_adversary else None
        self.discriminator = PatchGANDiscriminator() if with_discriminator else None

    def generator_modules(self) -> dict:
        """G_OPT membership (M7B R-07): G_res + E_art + every enabled auxiliary head. D is never a member."""
        mods = {'g_res': self.g_res, 'e_art': self.e_art}
        if self.attack_head is not None:
            mods['attack_head'] = self.attack_head
        if self.identity_head is not None:
            mods['identity_head'] = self.identity_head
        return mods

    def encode_source(self, x_source: torch.Tensor):
        bands = wavelet.dwt(x_source)
        inp = encoder_input(x_source, *bands[1:])
        spatial, layer4, z_a = self.e_art(inp)
        return bands, inp, spatial, layer4, z_a

    def forward_generator(self, x_source: torch.Tensor, x_target: torch.Tensor, *, scale_hf: float,
                          gamma: float | None = None, artifact_scale: float = 1.0) -> GPATForward:
        for name, x in (('x_source', x_source), ('x_target', x_target)):
            if x.dim() != 4 or tuple(x.shape[1:]) != IMAGE_SHAPE:
                raise ValueError(f'{name} must be [N, 3, 256, 256], got {tuple(x.shape)}')
        if x_source.shape[0] != x_target.shape[0]:
            raise ValueError('x_source and x_target batch sizes differ')
        gamma = self.config.gamma if gamma is None else gamma
        src_bands, inp, spatial, layer4, z_a = self.encode_source(x_source)
        tgt_bands = wavelet.dwt(x_target)
        raw = self.g_res(wavelet.concat_bands(*tgt_bands), z_a, spatial)
        out = composition.generate(raw, tgt_bands, x_target, scale_hf=scale_hf, gamma=gamma,
                                   artifact_scale=artifact_scale)
        return GPATForward(
            source_bands=src_bands, target_bands=tgt_bands, encoder_input=inp, z_a=z_a, spatial_code=spatial,
            layer4=layer4, raw=raw, scale_hf=float(scale_hf), gamma=float(gamma),
            attack_logits=self.attack_head(z_a) if self.attack_head is not None else None,
            identity_logits=self.identity_head(z_a) if self.identity_head is not None else None,
            **{k: out[k] for k in ('delta_LL', 'delta_LH', 'delta_HL', 'delta_HH', 'M', 'LL_syn', 'LH_syn', 'HL_syn',
                                   'HH_syn', 'x_hat', 'A')})

    def forward(self, x_source, x_target, *, scale_hf, gamma=None, artifact_scale=1.0):
        return self.forward_generator(x_source, x_target, scale_hf=scale_hf, gamma=gamma, artifact_scale=artifact_scale)


def parameter_counts(module: nn.Module) -> dict:
    return {'total': sum(p.numel() for p in module.parameters()),
            'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}


__all__ = ['GPATCore', 'GPATForward', 'ArtifactEncoder', 'parameter_counts']
