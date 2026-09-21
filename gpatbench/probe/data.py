"""ArtifactProbeNet dataset and loaders (owner decisions D-M5-01, D-M5-08, Q-07).

Firewall: TRAIN optimises, VAL selects, and **TEST can never be constructed**. There is no code
path that accepts synthetic images -- every sample is resolved from the frozen M2 canonical face
store through the split manifest.

**Physical storage is execution-only and portable.** The canonical faces live at a machine-specific
path, so the *execution* config that names it is selected by the `GPAT_M5_EXEC_CONFIG` environment
variable, defaulting to the historical laptop config when unset. The selected file is always passed
through `gpatbench.preprocess.m2b.resolve_roots()`, so the runtime-root containment firewall still
applies and a root that escapes -- by absolute path, `..` or symlink -- is refused. There is
deliberately no direct faces-root override: relocation has to go through a config that containment
can check. None of this touches sample membership, labels, splits, preprocessing or the frozen
scientific contract.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from . import preprocess as PP
from .contract import (CLASS_INDEX, CLASSES, ROOT, SEED, ProbeContractViolation,
                       SPLIT_MANIFEST)

ALLOWED_SPLITS = ("TRAIN", "VAL")

# Execution-only storage selector. Scientific identity never depends on these strings.
EXEC_CONFIG_ENV = "GPAT_M5_EXEC_CONFIG"
DEFAULT_EXEC_CONFIG = ROOT / "configs/execution/m2b_laptop_external_storage.yaml"
EXEC_CONFIG = DEFAULT_EXEC_CONFIG          # historical default; kept for compatibility


def resolve_exec_config(env: dict | None = None) -> Path:
    """Which execution config names the physical face store.

    `GPAT_M5_EXEC_CONFIG` set   -> exactly that file, and it must exist (no silent fallback).
    `GPAT_M5_EXEC_CONFIG` unset -> the historical laptop default, preserving previous behaviour.

    A relative value is resolved against the repository root so a run is not sensitive to `cwd`.
    """
    import os
    env = os.environ if env is None else env
    raw = env.get(EXEC_CONFIG_ENV)
    if raw is None or str(raw).strip() == "":
        if not DEFAULT_EXEC_CONFIG.is_file():
            raise ProbeContractViolation(
                f"default execution config missing: {DEFAULT_EXEC_CONFIG}")
        return DEFAULT_EXEC_CONFIG
    p = Path(str(raw).strip())
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise ProbeContractViolation(
            f"{EXEC_CONFIG_ENV}={raw!r} does not name an existing file ({p}). It was set "
            "explicitly, so there is no fallback to the default execution config.")
    return p


def load_exec_config(path: Path | None = None) -> dict:
    """Parse the selected execution config. A malformed or non-mapping file fails closed."""
    import yaml
    path = resolve_exec_config() if path is None else path
    try:
        cfg = yaml.safe_load(path.read_text())
    except Exception as exc:                                  # malformed YAML
        raise ProbeContractViolation(f"execution config {path} does not parse: {exc}") from exc
    if not isinstance(cfg, dict) or "storage" not in cfg:
        raise ProbeContractViolation(
            f"execution config {path} has no `storage` block; it cannot name a face store")
    return cfg


def resolve_storage(path: Path | None = None, *, require_faces: bool = True) -> dict:
    """Selected config -> contained roots -> the physical faces root, with provenance.

    Always goes through `m2b.resolve_roots`, so a root outside `storage.runtime_root` is refused
    there rather than silently used.
    """
    from gpatbench.preprocess import m2b
    path = resolve_exec_config() if path is None else path
    cfg = load_exec_config(path)
    try:
        resolved = m2b.resolve_roots(cfg)                     # containment firewall
    except KeyError as exc:
        raise ProbeContractViolation(
            f"execution config {path} is missing a required storage key: {exc}") from exc
    roots = resolved["roots"]
    if "faces_256_root" not in roots:
        raise ProbeContractViolation(
            f"execution config {path} declares no faces_256_root")
    faces = Path(roots["faces_256_root"]).resolve()
    if require_faces and not faces.is_dir():
        raise ProbeContractViolation(
            f"faces_256_root {faces} (from {path}) does not exist or is not a directory")
    from .contract import sha256_file
    return {"exec_config_path": path, "exec_config_sha256": sha256_file(path),
            "runtime_root": str(resolved["runtime_root"]),
            "faces_256_root": faces,
            "faces_256_root_resolved": faces.as_posix(),
            "selected_via_env": EXEC_CONFIG_ENV in __import__("os").environ}


def faces_root(*, require_exists: bool = True) -> Path:
    return resolve_storage(require_faces=require_exists)["faces_256_root"]


def execution_config_provenance() -> dict:
    """The three execution-storage fields every M5 artifact must carry.

    This is PROVENANCE, not readiness: it reports which execution config was selected, its SHA-256
    and where `faces_256_root` resolves to. None of that needs the directory to exist, so it does
    not require it -- selection, parsing and containment are still enforced. Physical readiness is
    checked where it matters: `faces_root()` and the authoritative `preflight()` both resolve with
    `require_faces=True` and fail closed on a missing face store.
    """
    s = resolve_storage(require_faces=False)
    return {"m5_execution_config_path": (
                s["exec_config_path"].relative_to(ROOT).as_posix()
                if s["exec_config_path"].is_relative_to(ROOT)
                else s["exec_config_path"].as_posix()),
            "m5_execution_config_sha256": s["exec_config_sha256"],
            "faces_256_root_resolved": s["faces_256_root_resolved"],
            "m5_execution_config_selected_via_env": s["selected_via_env"],
            "m5_execution_config_env_var": EXEC_CONFIG_ENV}


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
