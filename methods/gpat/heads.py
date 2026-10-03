"""GPAT auxiliary heads on the global artifact code z_a (512-D).

Attack-type head (B1/B3, A10 D05): Linear(512, 6, bias=True), class order fixed below, no live class.
Identity adversary (B2/B3, A10 D02, M7B owner clarification): GRL(alpha 1.0) then Linear(512, 60, bias=True); the
objective adds +lambda_idadv * CE, the single reversal is the GRL.
"""
from __future__ import annotations

import torch.nn as nn

from methods.gpat.grl import GRL_ALPHA, GradientReversal

CODE_DIM = 512
ATTACK_CLASSES = ('makeup', 'mask_2d', 'mask_3d', 'partial', 'print', 'replay')
ATTACK_CLASS_INDEX = {name: i for i, name in enumerate(ATTACK_CLASSES)}
IDENTITY_CLASSES = 60


class AttackTypeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CODE_DIM, len(ATTACK_CLASSES), bias=True)

    def forward(self, z_a):
        return self.fc(z_a)


class IdentityAdversaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.grl = GradientReversal(GRL_ALPHA)
        self.fc = nn.Linear(CODE_DIM, IDENTITY_CLASSES, bias=True)

    def forward(self, z_a):
        return self.fc(self.grl(z_a))
