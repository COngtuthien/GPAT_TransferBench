"""E02 — Frequency Substitution generator (frozen contract: configs/methods/e02_freqsub.yaml)."""
from .e02 import (
    E02Generator,
    block_count,
    conjugate_index,
    conjugate_mask,
    eligible_blocks,
    pair_seed,
    pair_seed_preimage,
    select_block_indices,
    substitute,
)

__all__ = [
    "E02Generator", "block_count", "conjugate_index", "conjugate_mask", "eligible_blocks",
    "pair_seed", "pair_seed_preimage", "select_block_indices", "substitute",
]
