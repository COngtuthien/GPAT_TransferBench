"""Gradient reversal layer (A10 D02): forward identity, backward -alpha * grad; alpha frozen at 1.0."""
from __future__ import annotations

import torch

GRL_ALPHA = 1.0


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float = GRL_ALPHA) -> torch.Tensor:
    return _GradReverse.apply(x, float(alpha))


class GradientReversal(torch.nn.Module):
    def __init__(self, alpha: float = GRL_ALPHA):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return grad_reverse(x, self.alpha)

    def extra_repr(self):
        return f'alpha={self.alpha}'
