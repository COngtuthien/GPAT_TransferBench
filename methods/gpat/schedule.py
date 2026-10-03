"""Closed-form GPAT learning-rate and curriculum values (spec 10.5, A10 D10.22/D10.23, M7C2a N-07).

The single definitions live in `runtime_contract`; this module adds only the epoch/group -> update-index map.
Main LR (G_OPT and D_OPT): u = 1..5525 linear 0 -> 2e-4; u = 5526..66300 cosine 2e-4 -> 2e-6.
Attack warmup (B1/B3): s = 1..1390 cosine 1e-4 -> 0.
Curriculum: u 1..5525 s_hf 0.02 -> 0.05, adv 0, con 0.5, spec 0.25; 5526..16575 0.10 / 0.05 / 1 / 0.5;
16576..66300 0.15 / 0.10 / 1 / 0.5.
"""
from __future__ import annotations

from methods.gpat import runtime_contract as rc

TOTAL_UPDATES = rc.TOTAL_UPDATES
UPDATES_PER_EPOCH = rc.UPDATES_PER_EPOCH
GENERATOR_EPOCHS = TOTAL_UPDATES // UPDATES_PER_EPOCH
main_lr = rc.main_lr
attack_warmup_lr = rc.attack_warmup_lr
curriculum = rc.curriculum


def update_index(epoch: int, group: int) -> int:
    """1-based optimizer update index of 1-based (epoch, group within the epoch)."""
    for name, v, last in (('epoch', epoch, GENERATOR_EPOCHS), ('group', group, UPDATES_PER_EPOCH)):
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= last:
            raise ValueError(f'{name} must be an int in 1..{last}, got {v!r}')
    return (epoch - 1) * UPDATES_PER_EPOCH + group
