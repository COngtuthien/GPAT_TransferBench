"""EMA of E_art and G_res (spec 10.5, A10 D10.21, M7C2a N-06): decay 0.999 on every floating state_dict tensor
(parameters and BN running statistics); non-floating tensors (num_batches_tracked) are copied. No EMA of the attack
head, the identity head, D or any frozen teacher. No disk I/O here; the EMA copy is evaluated in eval mode.
"""
from __future__ import annotations

import copy

import torch

from methods.gpat import runtime_contract as rc
from methods.gpat.artifact_encoder import ArtifactEncoder
from methods.gpat.generator import NAFResidualUNet

EMA_DECAY = rc.EMA_DECAY
EMA_SCOPE = (ArtifactEncoder, NAFResidualUNet)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        if not isinstance(model, EMA_SCOPE):
            raise TypeError(f'EMA scope is E_art and G_res only, not {type(model).__name__}')
        if decay != EMA_DECAY:
            raise ValueError(f'EMA decay is frozen at {EMA_DECAY}')
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.updates = 0

    def initialize_from(self, model: torch.nn.Module) -> None:
        """Copy the live state exactly (start of EMA)."""
        self._same_type(model)
        ema = self.module.state_dict()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema[k].copy_(v)
        self.updates = 0

    def update_from(self, model: torch.nn.Module) -> None:
        """ema = decay * ema + (1 - decay) * live (floating); non-floating copied (rc.ema_update)."""
        self._same_type(model)
        rc.ema_update(self.module.state_dict(), model.state_dict(), self.decay)
        self.updates += 1

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'updates': self.updates, 'module': self.module.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if state['decay'] != self.decay:
            raise ValueError('EMA decay mismatch')
        self.module.load_state_dict(state['module'], strict=True)
        self.updates = int(state['updates'])

    def evaluation_copy(self) -> torch.nn.Module:
        """An independent eval-mode copy of the EMA weights."""
        return copy.deepcopy(self.module).eval()

    def _same_type(self, model):
        if type(model) is not type(self.module):
            raise TypeError('live model type differs from the EMA module')
