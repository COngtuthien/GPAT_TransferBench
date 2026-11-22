"""Split firewall for downstream and smoke stages: every sample must be a known split_v1 sample in an allowed split.

TEST is never an allowed split in this module: §0.1 forbids TEST metadata in training and TEST metrics in any
selection, and no M8 smoke rule authorizes TEST (DRAFT S5). Allowing TEST would need a new, explicit code change
tied to an owner ruling.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

SPLIT_MANIFEST = "manifests/split_v1.parquet"
SPLIT_MANIFEST_SHA256 = "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"
PERMITTED = frozenset({"TRAIN", "VAL"})


class TestFirewallViolation(RuntimeError):
    __test__ = False   # not a unittest case


def read_permitted_rows(root: str | Path, columns: list[str]) -> list[dict]:
    """TRAIN/VAL rows of the frozen split manifest, hash-checked before parsing. TEST rows are removed by the parquet
    reader's row filter, so no TEST row or TEST metadata value reaches Python code."""
    import pyarrow.parquet as pq
    p = Path(root) / SPLIT_MANIFEST
    if hashlib.sha256(p.read_bytes()).hexdigest() != SPLIT_MANIFEST_SHA256:
        raise TestFirewallViolation("split manifest SHA256 differs from the frozen value")
    cols = list(dict.fromkeys(list(columns) + ["split"]))
    rows = pq.read_table(p, columns=cols, filters=[("split", "in", sorted(PERMITTED))]).to_pylist()
    if {r["split"] for r in rows} - PERMITTED:
        raise TestFirewallViolation("row filter let a non-permitted split through")
    return rows


def load_split_lookup(root: str | Path) -> dict[str, str]:
    """sample_id -> split for TRAIN/VAL samples only (allowlist). A TEST sample is simply unknown and refused."""
    return {r["sample_id"]: r["split"] for r in read_permitted_rows(root, ["sample_id"])}


def assert_allowed(sample_ids: Iterable[str], lookup: dict[str, str], allowed: Iterable[str]) -> dict:
    allowed = frozenset(allowed)
    if not allowed or not allowed <= PERMITTED:
        raise TestFirewallViolation(f"allowed splits {sorted(allowed)} must be a non-empty subset of {sorted(PERMITTED)}")
    counts: dict[str, int] = {}
    for sid in sample_ids:
        sp = lookup.get(sid)
        if sp is None:
            raise TestFirewallViolation(f"unknown sample {sid!r}")
        if sp not in allowed:
            raise TestFirewallViolation(f"sample {sid!r} is in split {sp}, not in {sorted(allowed)}")
        counts[sp] = counts.get(sp, 0) + 1
    return counts
