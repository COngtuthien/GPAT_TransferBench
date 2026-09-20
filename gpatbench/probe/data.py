"""ArtifactProbeNet dataset and loaders (owner decisions D-M5-01, D-M5-08, Q-07).

Firewall: TRAIN optimises, VAL selects, and **TEST can never be constructed**. There is no code
path that accepts synthetic images -- every sample is resolved from the frozen M2 canonical face
store through the split manifest.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from . import preprocess as PP
from .contract import (CLASS_INDEX, CLASSES, ROOT, SEED, ProbeContractViolation,
                       SPLIT_MANIFEST)

ALLOWED_SPLITS = ("TRAIN", "VAL")
EXEC_CONFIG = ROOT / "configs/execution/m2b_laptop_external_storage.yaml"


def faces_root() -> Path:
    import yaml
    from gpatbench.preprocess import m2b
    cfg = yaml.safe_load(EXEC_CONFIG.read_text())
    return m2b.resolve_roots(cfg)["roots"]["faces_256_root"]


def load_rows(split: str, manifest: Path = SPLIT_MANIFEST) -> list:
    """Rows for one allowed split, in canonical deterministic `sample_id` order."""
    if split not in ALLOWED_SPLITS:
        raise ProbeContractViolation(
            f"ArtifactProbeNet may only load {ALLOWED_SPLITS}; refused {split!r}. "
            "TEST is never loaded by the probe.")
    import pyarrow.parquet as pq
    rows = [r for r in pq.read_table(manifest).to_pylist() if r["split"] == split]
    out = []
    for r in rows:
        if r["m2_status"] != "COMPLETE":
            raise ProbeContractViolation(f"{r['sample_id']} is not M2 COMPLETE")
        if r["attack_macro"] not in CLASS_INDEX:
            raise ProbeContractViolation(f"{r['sample_id']}: unknown class {r['attack_macro']!r}")
        out.append({"sample_id": r["sample_id"], "dataset": r["dataset"],
                    "attack_macro": r["attack_macro"], "label": CLASS_INDEX[r["attack_macro"]],
                    "sha256": r["sha256"]})
    out.sort(key=lambda r: r["sample_id"])
    return out


class ProbeDataset:
    """Frozen canonical face -> frozen high-pass tensor -> (float32 CHW, int label)."""

    def __init__(self, split: str, rows: list | None = None, root: Path | None = None):
        if split not in ALLOWED_SPLITS:
            raise ProbeContractViolation(f"refused split {split!r}; TEST is never loaded")
        self.split = split
        self.rows = rows if rows is not None else load_rows(split)
        self._root = root
        self.synthetic_samples = 0          # structurally impossible; asserted by tests

    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = faces_root()
        return self._root

    def __len__(self) -> int:
        return len(self.rows)

    def face_path(self, i: int) -> Path:
        r = self.rows[i]
        return self.root / r["dataset"] / f"{r['sample_id']}.png"

    def __getitem__(self, i: int):
        import cv2
        r = self.rows[i]
        p = self.face_path(i)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ProbeContractViolation(f"{r['sample_id']}: canonical face missing at {p}")
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        x = PP.frozen_probe_input(rgb)
        return x, int(r["label"])


def worker_init_fn(worker_id: int) -> None:
    """Deterministic per-worker seeds derived from the frozen seed (D-M5-08)."""
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    try:
        import torch
        torch.manual_seed(s)
    except Exception:                        # torch is always present in the trainer path
        pass


def make_loaders(*, num_workers: int = 4, pin_memory: bool = True,
                 persistent_workers: bool = True, prefetch_factor: int = 2,
                 batch_size: int = 64):
    """TRAIN shuffles under an explicit seeded generator; VAL never shuffles. drop_last=False."""
    import torch
    from torch.utils.data import DataLoader

    class _T(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            return torch.from_numpy(x), y

    train_ds, val_ds = ProbeDataset("TRAIN"), ProbeDataset("VAL")
    g = torch.Generator()
    g.manual_seed(SEED)
    common = dict(batch_size=batch_size, drop_last=False, num_workers=num_workers,
                  pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    if num_workers > 0:
        common.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return (DataLoader(_T(train_ds), shuffle=True, generator=g, **common),
            DataLoader(_T(val_ds), shuffle=False, **common),
            train_ds, val_ds)
