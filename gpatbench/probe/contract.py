"""Frozen ArtifactProbeNet contract (spec §12.1 + the 2026-09-20 owner resolutions).

Loads `configs/frozen/artifact_probe.yaml`, verifies it against the authoritative manifests and
recomputes every derived quantity. Nothing here trains: it is the gate that the single trainer must
pass through, and it refuses rather than repairs.
"""
from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

FROZEN_CONFIG = ROOT / "configs/frozen/artifact_probe.yaml"
FROZEN_SNAPSHOT = ROOT / "frozen_config_snapshot/configs/frozen/artifact_probe.yaml"
SPLIT_MANIFEST = ROOT / "manifests/split_v1.parquet"

CONFIG_SHA256 = "3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe"
SPLIT_MANIFEST_SHA256 = "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"
RESNET18_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

CLASSES = ("live", "makeup", "mask_2d", "mask_3d", "partial", "print", "replay")
K = 7
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
CLIP_LO, CLIP_HI = 0.5, 3.0
SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
T_MAX = 30
ETA_MIN = 0.0
EMBED_DIM = 512
L2_EPS = 1.0e-12
TRAIN_ROWS = 14467
VAL_ROWS = 3121


class ProbeContractViolation(RuntimeError):
    """The frozen ArtifactProbeNet contract was violated. The trainer refuses rather than adapts."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path = FROZEN_CONFIG, *, verify_hash: bool = True) -> dict:
    if not path.is_file():
        raise ProbeContractViolation(f"frozen config missing: {path}")
    if verify_hash:
        got = sha256_file(path)
        if got != CONFIG_SHA256:
            raise ProbeContractViolation(
                f"config sha256 {got} != frozen {CONFIG_SHA256}; the contract changed")
        if not FROZEN_SNAPSHOT.is_file() or FROZEN_SNAPSHOT.read_bytes() != path.read_bytes():
            raise ProbeContractViolation("frozen snapshot is missing or not byte-identical")
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("status") != "FROZEN" or cfg.get("blocking_decisions"):
        raise ProbeContractViolation(
            f"config is not frozen (status={cfg.get('status')!r}, "
            f"open={cfg.get('blocking_decisions')!r})")
    if list(cfg["classes"]["order"]) != list(CLASSES) or cfg["classes"]["K"] != K:
        raise ProbeContractViolation("class order/K in the config does not match the frozen tuple")
    return cfg


def validation_epochs(cfg: dict | None = None) -> tuple:
    """The authoritative validation schedule: a VAL pass at the END of every epoch, 1..30.

    This is the single source of the sequence -- no other code path may hard-code it. The frozen
    contract states it as `evaluate_every_epoch` / `epoch_start` / `epoch_end` rather than a
    two-element list, so `[1, 30]` can never be misread as "epochs 1 and 30 only".
    """
    ck = (cfg or load_config())["validation"]["checkpoint"]
    if ck.get("evaluate_every_epoch") is not True:
        raise ProbeContractViolation(
            "the frozen contract requires evaluate_every_epoch: true "
            f"(got {ck.get('evaluate_every_epoch')!r})")
    start, end = ck["epoch_start"], ck["epoch_end"]
    if start != 1:
        raise ProbeContractViolation(f"epoch_start must be 1, got {start!r}")
    if end != EPOCHS:
        raise ProbeContractViolation(f"epoch_end must be {EPOCHS}, got {end!r}")
    seq = tuple(range(start, end + 1))
    if len(seq) != EPOCHS:
        raise ProbeContractViolation(f"validation sequence has {len(seq)} entries, expected {EPOCHS}")
    return seq


def select_best_epoch(macro_f1_by_epoch) -> tuple:
    """D-M5-06 selection over a per-epoch macro-F1 sequence: strict `>`, exact tie keeps earlier.

    Pure and side-effect free, so the rule can be tested without training anything.
    `macro_f1_by_epoch` is an iterable of (epoch, macro_f1) in ascending epoch order.
    """
    best_epoch, best_score = None, None
    for epoch, score in macro_f1_by_epoch:
        if best_score is None or score > best_score:      # strict >, never >=
            best_epoch, best_score = epoch, score
    if best_epoch is None:
        raise ProbeContractViolation("no epoch was evaluated")
    return best_epoch, best_score


def class_counts(split: str, manifest: Path = SPLIT_MANIFEST) -> dict:
    """Recount from the authoritative manifest. Hand-written numbers are never trusted."""
    import pyarrow.parquet as pq
    rows = [r for r in pq.read_table(manifest).to_pylist() if r["split"] == split]
    for r in rows:
        if r["m2_status"] != "COMPLETE":
            raise ProbeContractViolation(f"{r['sample_id']} is not M2 COMPLETE")
    counts = collections.Counter(r["attack_macro"] for r in rows)
    unknown = set(counts) - set(CLASSES)
    if unknown:
        raise ProbeContractViolation(f"{split}: unknown attack_macro tokens {sorted(unknown)}")
    return {c: int(counts.get(c, 0)) for c in CLASSES}


def verify_population(cfg: dict, manifest: Path = SPLIT_MANIFEST) -> dict:
    if sha256_file(manifest) != SPLIT_MANIFEST_SHA256:
        raise ProbeContractViolation("split manifest sha256 changed")
    tr, va = class_counts("TRAIN", manifest), class_counts("VAL", manifest)
    if tr != {k: int(v) for k, v in cfg["classes"]["train_counts"].items()}:
        raise ProbeContractViolation(f"TRAIN class counts differ: {tr}")
    if va != {k: int(v) for k, v in cfg["classes"]["val_counts"].items()}:
        raise ProbeContractViolation(f"VAL class counts differ: {va}")
    if sum(tr.values()) != TRAIN_ROWS or sum(va.values()) != VAL_ROWS:
        raise ProbeContractViolation("TRAIN/VAL totals differ from the frozen contract")
    return {"train": tr, "val": va}


def compute_class_weights(train_counts: dict) -> np.ndarray:
    """D-M5-02: w_raw = N/(K*n_c); w = clip(w_raw, 0.5, 3.0). float64, TRAIN only, no renorm."""
    n = np.array([train_counts[c] for c in CLASSES], dtype=np.float64)
    if not np.all(n > 0):
        raise ProbeContractViolation("a frozen class has zero TRAIN samples")
    w_raw = np.float64(n.sum()) / (np.float64(K) * n)
    w = np.minimum(np.float64(CLIP_HI), np.maximum(np.float64(CLIP_LO), w_raw))
    if not np.all(np.isfinite(w)) or not np.all((w >= CLIP_LO) & (w <= CLIP_HI)):
        raise ProbeContractViolation("class weights outside the frozen clip range")
    return w


def verify_class_weights(cfg: dict, train_counts: dict) -> np.ndarray:
    w = compute_class_weights(train_counts)
    frozen = np.array([float(cfg["class_weights"]["w_clipped"][c]) for c in CLASSES],
                      dtype=np.float64)
    if not np.array_equal(w, frozen):
        raise ProbeContractViolation(
            f"class weights differ from the frozen values: {list(w)} vs {list(frozen)}")
    return w


def lr_schedule(epochs: int = EPOCHS, lr: float = LR, t_max: int = T_MAX,
                eta_min: float = ETA_MIN) -> list:
    """Analytic CosineAnnealingLR sequence: the LR in force DURING each 1-based epoch."""
    import math
    return [eta_min + (lr - eta_min) * (1 + math.cos(math.pi * e / t_max)) / 2
            for e in range(epochs)]


def verify_environment(*, require_cuda: bool) -> dict:
    import torch
    import torchvision
    env = {"torch": torch.__version__, "torchvision": torchvision.__version__,
           "cuda_available": bool(torch.cuda.is_available()),
           "cuda_version": torch.version.cuda,
           "cudnn_version": torch.backends.cudnn.version()}
    if require_cuda and not env["cuda_available"]:
        raise ProbeContractViolation(
            "E-M5-01: authoritative M5 training requires CUDA with AMP. CPU execution is "
            "forbidden and AMP may not be disabled. Run on the approved GPU host recorded in "
            "the frozen config's `execution` block (see also any pending execution-host "
            "deviation record before an authoritative run).")
    return env
