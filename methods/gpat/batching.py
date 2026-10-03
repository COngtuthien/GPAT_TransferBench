"""Deterministic GPAT batch accounting (A10 D15): 8838 TRAIN pairs, physical batch 4, grad accumulation 2, no
drop_last -> 2210 microbatches, 1105 optimizer groups, regular group [4, 4], tail group [4, 2], sample-weighted loss
w_i = n_i / n_group. Pure arithmetic: no data, no DataLoader, no permutation.
"""
from __future__ import annotations

from methods.gpat import runtime_contract as rc

TRAIN_PAIRS = 8838
PHYSICAL_BATCH = 4
GRAD_ACCUM = 2


def microbatch_sizes(n: int = TRAIN_PAIRS, batch: int = PHYSICAL_BATCH) -> list:
    full, tail = divmod(n, batch)
    return [batch] * full + ([tail] if tail else [])


def optimizer_groups(n: int = TRAIN_PAIRS, batch: int = PHYSICAL_BATCH, accum: int = GRAD_ACCUM) -> list:
    """Microbatch sizes grouped per optimizer update (the last group keeps whatever remains)."""
    sizes = microbatch_sizes(n, batch)
    return [sizes[i:i + accum] for i in range(0, len(sizes), accum)]


def group_weights(sizes) -> list:
    return rc.group_weights(sizes)


def accounting(n: int = TRAIN_PAIRS, batch: int = PHYSICAL_BATCH, accum: int = GRAD_ACCUM) -> dict:
    groups = optimizer_groups(n, batch, accum)
    return {'samples': n, 'microbatches': sum(len(g) for g in groups), 'optimizer_groups': len(groups),
            'regular_group': groups[0], 'last_group': groups[-1], 'regular_weights': group_weights(groups[0]),
            'last_weights': group_weights(groups[-1])}
