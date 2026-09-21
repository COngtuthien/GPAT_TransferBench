"""Deterministic seeding for M6 runs.

Process-level seeding only. Method-specific per-pair seed derivations stay method-local
(`methods.fas_aug.e01.operator_seed`, `methods.freq_sub.e02.pair_seed`) because they are distinct
frozen scientific contracts, not a shared utility.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> dict:
    """Seed Python and NumPy deterministically. Returns what was seeded, for the run manifest.

    Framework seeding (torch / tensorflow) is deliberately NOT imported here: E01 and E02 are
    non-learned and must not pull a deep-learning framework into their execution path. The learned
    baselines add their own hook at M6C2+.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    return {
        "python_random": seed,
        "numpy_legacy_global": seed,
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "framework_seed_hook": "NOT_APPLICABLE_NON_LEARNED_METHOD",
        "note": ("Per-pair scientific RNG is derived independently of these global seeds; global "
                 "seeding exists so that any incidental library randomness is reproducible."),
    }
