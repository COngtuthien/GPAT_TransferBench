"""The single authoritative ArtifactProbeNet trainer (spec §12.1 + owner resolutions).

There is exactly one training code path in this project and it is `run()` below. It refuses rather
than adapts: a non-frozen config, a changed config or manifest hash, a wrong ResNet weight hash, a
CPU authoritative run, unavailable AMP, a TEST split, synthetic data, augmentation, forbidden
post-high-pass normalisation, a class-count mismatch or a class-weight mismatch all raise
`ProbeContractViolation`.

The physical face store is execution-only and portable: `GPAT_M5_EXEC_CONFIG` selects the
execution config that names it (default: the historical laptop config), and it is always resolved
through `m2b.resolve_roots` so the containment firewall applies. The selected path, its SHA-256 and
the resolved `faces_256_root` travel in the run provenance, the checkpoint and the audit records.

`run(dry_run=True)` performs the dry-run preflight: it validates the contract, builds the model,
loss, optimizer and scheduler, and returns -- it never calls `optimizer.step()` and never writes a
checkpoint.

`run(dry_run=False)` is the authoritative run. It requires CUDA, trains for
`contract.EPOCHS` epochs and, **for every epoch in `contract.validation_epochs()` = (1, ..., 30)**,
executes exactly this order:

    1. all TRAIN optimizer updates for the epoch
    2. END-OF-EPOCH validation over the whole VAL split
    3. best-checkpoint selection (and write, when it improves)
    4. `scheduler.step()`  -- exactly once per epoch, after 1-3

`_train_loop` records that order as an event trace so it can be asserted structurally rather than
inferred. The best checkpoint is replaced only on a strict `new_macro_f1 > best_macro_f1`, so an
exact tie keeps the earlier epoch. TEST never participates, in any mode.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as MET
from . import model as MODEL
from .contract import (BATCH_SIZE, CLASSES, CONFIG_SHA256, EPOCHS, ETA_MIN, K, LR,
                       RESNET18_WEIGHT_SHA256, ROOT, SEED, SPLIT_MANIFEST, SPLIT_MANIFEST_SHA256,
                       T_MAX, WEIGHT_DECAY, FROZEN_CONFIG, FROZEN_SNAPSHOT, ProbeContractViolation,
                       load_config, lr_schedule, sha256_file, select_best_epoch, validation_epochs,
                       verify_class_weights, verify_environment, verify_population)

CHECKPOINT_DIR = ROOT / "models/artifact_probe"
CHECKPOINT_PATH = CHECKPOINT_DIR / "artifact_probe_v1.pt"

# Paths are the ones the frozen config's `outputs:` block names; they are asserted against it.
TRAINING_LOG = ROOT / "outputs/audit/M5_PROBE_TRAINING_LOG.jsonl"
EPOCH_METRICS = ROOT / "outputs/audit/M5_PROBE_EPOCH_METRICS.csv"
SELECTION_REPORT = ROOT / "outputs/audit/M5_PROBE_SELECTION_REPORT.md"
CLASS_MAPPING = ROOT / "manifests/artifact_probe_classes_v1.json"
ENVIRONMENT_LOCK = ROOT / "environments/m5_probe_gpu.lock.txt"
CHECKPOINT_SHA_RECORD = ROOT / "outputs/audit/artifact_probe_v1.sha256"

# Ordered control-flow step names. The trace they build is the evidence for the frozen per-epoch
# order, so they are constants rather than ad-hoc strings.
STEP_TRAIN = "train"
STEP_VAL = "validate"
STEP_SELECT = "select_checkpoint"
STEP_SCHEDULER = "scheduler_step"

CHECKPOINT_SCHEMA_VERSION = "artifact_probe_v1"
EPOCH_METRICS_COLUMNS = ("epoch", "lr", "train_loss", "train_batches", "train_samples",
                         "train_optimizer_steps", "val_macro_f1", "val_accuracy", "val_samples",
                         "is_best", "best_epoch_so_far", "best_macro_f1_so_far",
                         "epoch_seconds")

# AMP scope -- OWNER-RESOLVED 2026-09-21 (outputs/audit/M5_GPU_EXECUTION_OWNER_RESOLUTION.md).
# The frozen config enables AMP for the run without scoping it to the training pass. The owner
# resolved that the VALIDATION forward pass uses the SAME CUDA FP16 autocast regime as the training
# forward pass, so the two passes never differ in precision. Validation is NOT run in FP32.
# The GradScaler applies to the TRAIN backward/update only -- validation runs under `no_grad()` and
# never scales, steps or updates. Recorded in the checkpoint and the audit provenance as
# `amp.val_autocast` so the decision travels with the artifact.
VAL_AUTOCAST = True


# ------------------------------------------------------------------ determinism / environment
def seed_everything(seed: int = SEED) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_determinism() -> dict:
    """Frozen determinism contract. Never silently relaxed: a missing kernel must stop the run.

    `CUBLAS_WORKSPACE_CONFIG` is *enabling*, not relaxing: without it cuBLAS raises under
    `use_deterministic_algorithms(True)`. It must be set before the CUDA context is created, so
    this runs before any device work.
    """
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"seed": SEED, "cudnn_benchmark": False, "cudnn_deterministic": True,
            "use_deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def resolve_device(*, require_cuda: bool):
    """CUDA for the authoritative run. There is no CPU fallback and none may be added."""
    import torch
    if require_cuda:
        if not torch.cuda.is_available():
            raise ProbeContractViolation(
                "E-M5-01: authoritative M5 training requires CUDA with AMP. CPU execution is "
                "forbidden and AMP may not be disabled. Run on the approved GPU host.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_amp(device):
    """CUDA float16 autocast + an ENABLED GradScaler. Fails closed on anything else."""
    import torch
    if device.type != "cuda":
        raise ProbeContractViolation(
            f"AMP is CUDA-only for authoritative training; got device {device.type!r}. "
            "bfloat16, full-float and CPU fallbacks are all forbidden.")
    try:                                                      # non-deprecated API when available
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):                       # older pinned builds
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    if not scaler.is_enabled():
        raise ProbeContractViolation("GradScaler is disabled; AMP may not be silently turned off")

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return autocast, scaler


def verify_backbone_weights() -> str:
    """Hash the actual torchvision cache file for IMAGENET1K_V1 and refuse any substitute."""
    from torchvision.models import ResNet18_Weights
    url = ResNet18_Weights.IMAGENET1K_V1.url
    name = url.rsplit("/", 1)[-1]
    candidates = [Path.home() / ".cache/torch/hub/checkpoints" / name,
                  Path("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache/backbones/"
                       "torchvision") / name]
    for p in candidates:
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != RESNET18_WEIGHT_SHA256:
                raise ProbeContractViolation(
                    f"ResNet-18 weight sha256 {h} != frozen {RESNET18_WEIGHT_SHA256} at {p}")
            return h
    raise ProbeContractViolation(
        f"ResNet-18 IMAGENET1K_V1 weight file {name} not found in {[str(c) for c in candidates]}")


def build_loss(weights_np: np.ndarray):
    import torch
    w = torch.tensor(weights_np, dtype=torch.float32)       # frozen class order
    return torch.nn.CrossEntropyLoss(weight=w, reduction="mean")


def build_optimizer_and_scheduler(model):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=ETA_MIN)
    return opt, sch


# ------------------------------------------------------------------ epoch primitives
def train_one_epoch(model, loader, criterion, optimizer, scaler, autocast, device) -> dict:
    """One full pass of optimizer updates. Returns only aggregate statistics -- never predictions."""
    import torch
    model.train()
    total_loss, n_batches, n_samples, n_steps = 0.0, 0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = torch.as_tensor(yb).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_steps += 1
        n_batches += 1
        n_samples += int(yb.shape[0])
        total_loss += float(loss.detach()) * int(yb.shape[0])
    if n_samples == 0:
        raise ProbeContractViolation("the TRAIN loader yielded no samples")
    return {"train_loss": total_loss / n_samples, "train_batches": n_batches,
            "train_samples": n_samples, "train_optimizer_steps": n_steps}


def validate(model, loader, autocast, device) -> dict:
    """END-OF-EPOCH validation over the whole VAL split. No gradients, no checkpoint side effects."""
    import torch
    model.eval()
    cm = np.zeros((K, K), dtype=np.int64)
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb)
            if VAL_AUTOCAST:
                with autocast():
                    logits = model(xb)
            else:
                logits = model(xb)
            pred = logits.float().argmax(dim=1).cpu().numpy()
            cm += MET.confusion_matrix(yb.cpu().numpy(), pred)
            n += int(yb.shape[0])
    rep = MET.report(cm)
    return {"confusion_matrix": cm, "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
            "per_class_f1": rep["f1"], "support": rep["support"], "predicted": rep["predicted"],
            "val_samples": n}


# ------------------------------------------------------------------ checkpoint
def checkpoint_payload(model, *, epoch: int, macro_f1: float, val: dict, provenance: dict) -> dict:
    """Everything a later validator needs, and no raw data."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "selected_epoch": int(epoch),
        "best_val_macro_f1": float(macro_f1),
        "val_per_class_f1": [float(x) for x in val["per_class_f1"]],
        "val_confusion_matrix": val["confusion_matrix"].tolist(),
        "val_samples": int(val["val_samples"]),
        "classes": list(CLASSES),
        "class_index": {c: i for i, c in enumerate(CLASSES)},
        "num_classes": K,
        "seed": SEED,
        "frozen_config": _rel(FROZEN_CONFIG),
        "frozen_config_sha256": provenance["frozen_config_sha256"],
        "frozen_config_snapshot_sha256": provenance["frozen_config_snapshot_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "resnet18_weight_sha256": provenance["resnet18_weight_sha256"],
        "m5_execution_config_path": provenance.get("m5_execution_config_path"),
        "m5_execution_config_sha256": provenance.get("m5_execution_config_sha256"),
        "faces_256_root_resolved": provenance.get("faces_256_root_resolved"),
        "class_weights": provenance["class_weights"],
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                      "param_groups": 1},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": T_MAX, "eta_min": ETA_MIN,
                      "step_granularity": "once_per_epoch",
                      "step_position": "after validation and checkpoint selection"},
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_epochs": list(validation_epochs()),
        "selection_rule": "max VAL macro-F1; strict >, exact tie keeps the earlier epoch",
        "amp": {"autocast_device_type": "cuda", "autocast_dtype": "float16",
                "grad_scaler_enabled": True, "val_autocast": VAL_AUTOCAST},
        "determinism": provenance["determinism"],
        "environment": provenance["environment"],
        "code_commit": provenance.get("code_commit"),
        "host": provenance.get("host"),
        "created_utc": provenance.get("created_utc"),
        "test_split_used": False,
        "synthetic_samples_used": False,
        "augmentation": False,
    }


def save_checkpoint(model, *, epoch: int, macro_f1: float, val: dict, provenance: dict,
                    path: Path = CHECKPOINT_PATH) -> Path:
    """Atomic write: tmp -> fsync -> rename, so a partial file never carries the final name."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = checkpoint_payload(model, epoch=epoch, macro_f1=macro_f1, val=val,
                                 provenance=provenance)
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ the 30-epoch control flow
def _train_loop(*, model, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
                autocast, device, provenance, checkpoint_path: Path = CHECKPOINT_PATH,
                epochs=None, train_fn=train_one_epoch, validate_fn=validate,
                save_fn=save_checkpoint, write_checkpoint: bool = True) -> dict:
    """The frozen per-epoch order, recorded as it happens.

    For every epoch in `epochs` (default `validation_epochs()` = 1..30):
        train updates -> validation -> checkpoint selection/write -> scheduler.step()
    `scheduler.step()` runs exactly once per epoch and always last.
    """
    epochs = tuple(validation_epochs()) if epochs is None else tuple(epochs)
    events, rows, log = [], [], []
    best_epoch, best_macro_f1, best_val = None, None, None
    checkpoint_written_at = []

    for epoch in epochs:
        t0 = time.time()
        lr_now = float(optimizer.param_groups[0]["lr"])

        # 1. all TRAIN optimizer updates for this epoch
        events.append((STEP_TRAIN, epoch))
        tr = train_fn(model, train_loader, criterion, optimizer, scaler, autocast, device)

        # 2. END-OF-EPOCH validation
        events.append((STEP_VAL, epoch))
        va = validate_fn(model, val_loader, autocast, device)

        # 3. best-checkpoint selection; strict >, so an exact tie keeps the earlier epoch
        events.append((STEP_SELECT, epoch))
        is_best = best_macro_f1 is None or MET.is_better(va["macro_f1"], best_macro_f1)
        if is_best:
            best_epoch, best_macro_f1, best_val = epoch, va["macro_f1"], va
            if write_checkpoint:
                save_fn(model, epoch=epoch, macro_f1=va["macro_f1"], val=va,
                        provenance=provenance, path=checkpoint_path)
                checkpoint_written_at.append(epoch)

        # 4. scheduler, exactly once per epoch and only after 1-3
        events.append((STEP_SCHEDULER, epoch))
        scheduler.step()

        row = {"epoch": epoch, "lr": lr_now, "train_loss": tr["train_loss"],
               "train_batches": tr["train_batches"], "train_samples": tr["train_samples"],
               "train_optimizer_steps": tr["train_optimizer_steps"],
               "val_macro_f1": va["macro_f1"], "val_accuracy": va["accuracy"],
               "val_samples": va["val_samples"], "is_best": bool(is_best),
               "best_epoch_so_far": best_epoch, "best_macro_f1_so_far": best_macro_f1,
               "epoch_seconds": round(time.time() - t0, 6)}
        rows.append(row)
        log.append({**row, "per_class_f1": [float(x) for x in va["per_class_f1"]],
                    "support": list(va["support"]), "predicted": list(va["predicted"])})

    if best_epoch is None:
        raise ProbeContractViolation("no epoch was validated; the contract requires 1..30")
    # independent cross-check of the selection rule against the recorded sequence
    rule_epoch, rule_score = select_best_epoch([(r["epoch"], r["val_macro_f1"]) for r in rows])
    if (rule_epoch, rule_score) != (best_epoch, best_macro_f1):
        raise ProbeContractViolation(
            f"selection disagreement: loop chose epoch {best_epoch} ({best_macro_f1}), "
            f"rule chose {rule_epoch} ({rule_score})")
    return {"events": events, "epoch_rows": rows, "training_log": log,
            "epochs_run": list(epochs), "validation_passes": len(epochs),
            "best_epoch": best_epoch, "best_macro_f1": best_macro_f1, "best_val": best_val,
            "checkpoint_written_at": checkpoint_written_at,
            "scheduler_steps": sum(1 for e, _ in events if e == STEP_SCHEDULER)}


# ------------------------------------------------------------------ audit writers
def _rel(path: Path) -> str:
    """Repo-relative when the path is inside the repository, absolute otherwise."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _canonical_json_bytes(obj) -> bytes:
    """The project-wide canonical JSON policy (see gpatbench.pairs.execute.canonical_json_bytes)."""
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
                       separators=(",", ": ")) + "\n").encode("utf-8")


def write_audit_outputs(loop: dict, provenance: dict, *, checkpoint_path: Path = CHECKPOINT_PATH,
                        paths: dict | None = None) -> dict:
    """Deterministic-schema audit records. The checkpoint SHA is taken only after the file exists."""
    p = {"training_log": TRAINING_LOG, "epoch_metrics": EPOCH_METRICS,
         "selection_report": SELECTION_REPORT, "class_mapping": CLASS_MAPPING,
         "environment_lock": ENVIRONMENT_LOCK, "checkpoint_sha": CHECKPOINT_SHA_RECORD}
    if paths:
        p.update(paths)
    for q in p.values():
        q.parent.mkdir(parents=True, exist_ok=True)

    with p["training_log"].open("w", encoding="utf-8") as f:
        for rec in loop["training_log"]:
            f.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")

    with p["epoch_metrics"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(EPOCH_METRICS_COLUMNS))
        w.writeheader()
        for row in loop["epoch_rows"]:
            w.writerow({k: row[k] for k in EPOCH_METRICS_COLUMNS})

    p["class_mapping"].write_bytes(_canonical_json_bytes({
        "schema_version": "artifact_probe_classes_v1",
        "classes": list(CLASSES), "class_index": {c: i for i, c in enumerate(CLASSES)},
        "num_classes": K, "excluded": ["other_spoof"],
        "excluded_reason": "zero TRAIN and zero VAL observations; no unused output logit",
        "frozen_config_sha256": provenance["frozen_config_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"]}))

    # The execution-storage fields are merged here too, so the lock records where the faces
    # physically came from even if a caller supplied a lock dict without them.
    lock = dict(provenance["environment_lock"])
    for k in ("m5_execution_config_path", "m5_execution_config_sha256",
              "faces_256_root_resolved"):
        if provenance.get(k) is not None:
            lock[k] = provenance[k]
    p["environment_lock"].write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(lock.items())) + "\n", encoding="utf-8")

    sha = None
    if checkpoint_path.is_file():               # only after the selected checkpoint exists
        sha = sha256_file(checkpoint_path)
        p["checkpoint_sha"].write_bytes(_canonical_json_bytes({
            "artifact": _rel(checkpoint_path),
            "sha256": sha, "size_bytes": checkpoint_path.stat().st_size,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "selected_epoch": loop["best_epoch"],
            "best_val_macro_f1": loop["best_macro_f1"],
            "validation_passes": loop["validation_passes"],
            "frozen_config_sha256": provenance["frozen_config_sha256"],
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "resnet18_weight_sha256": provenance["resnet18_weight_sha256"],
            "m5_execution_config_path": provenance.get("m5_execution_config_path"),
            "m5_execution_config_sha256": provenance.get("m5_execution_config_sha256"),
            "faces_256_root_resolved": provenance.get("faces_256_root_resolved"),
            "code_commit": provenance.get("code_commit"),
            "host": provenance.get("host"),
            "created_utc": provenance.get("created_utc"),
            "committed_to_git": False}))

    best = loop["best_val"]
    lines = [
        "# M5 — ArtifactProbeNet Checkpoint Selection Report", "",
        f"**Selected epoch:** {loop['best_epoch']} of {loop['validation_passes']}",
        f"**Best VAL macro-F1:** {loop['best_macro_f1']!r}",
        f"**Checkpoint:** `{_rel(checkpoint_path)}`"
        + (f" (sha256 `{sha}`)" if sha else " (not written)"), "",
        "Rule: maximum VAL macro-F1, replaced only on a strict `>` so an exact tie keeps the "
        "earlier epoch. Training loss is never a tie-break and TEST never participates.", "",
        "| class | support | predicted | F1 |", "|---|---|---|---|",
    ]
    if best is not None:
        for i, c in enumerate(CLASSES):
            lines.append(f"| {c} | {best['support'][i]} | {best['predicted'][i]} | "
                         f"{best['per_class_f1'][i]:.6f} |")
    lines += ["", "| epoch | lr | train loss | VAL macro-F1 | best |", "|---|---|---|---|---|"]
    for r in loop["epoch_rows"]:
        lines.append(f"| {r['epoch']} | {r['lr']:.6e} | {r['train_loss']:.6f} | "
                     f"{r['val_macro_f1']:.6f} | {'**yes**' if r['is_best'] else ''} |")
    p["selection_report"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"checkpoint_sha256": sha,
            "written": {k: _rel(v) for k, v in p.items() if k != "checkpoint_sha" or sha}}


# ------------------------------------------------------------------ gates
def preflight(config_path: Path | None = None, *, require_cuda: bool,
              require_faces: bool = True) -> dict:
    """Every gate the authoritative run must pass, with no side effects."""
    cfg = load_config(config_path) if config_path else load_config()
    pop = verify_population(cfg)
    weights = verify_class_weights(cfg, pop["train"])
    env = verify_environment(require_cuda=require_cuda)
    weight_sha = verify_backbone_weights()
    if cfg["augmentation"]["any"]:
        raise ProbeContractViolation("augmentation is forbidden for ArtifactProbeNet")
    if cfg["population"]["test_split_usage"] != "NEVER":
        raise ProbeContractViolation("TEST must never be used")
    if cfg["population"]["synthetic_samples"] != "NONE":
        raise ProbeContractViolation("synthetic samples must never enter probe training")
    n = cfg["input"]["post_hp_normalization"]
    if n["imagenet_mean_subtraction"] or n["imagenet_std_division"] or n["dataset_mean_std"] \
            or n["min_max"]:
        raise ProbeContractViolation("post-high-pass normalization is forbidden")
    if cfg["execution"]["cpu_authoritative_training"] != "FORBIDDEN" or \
            cfg["execution"]["disable_amp"] != "FORBIDDEN":
        raise ProbeContractViolation("the frozen execution contract was weakened")
    if sha256_file(FROZEN_SNAPSHOT) != sha256_file(FROZEN_CONFIG):
        raise ProbeContractViolation("frozen snapshot is not byte-identical to the frozen config")
    epochs = validation_epochs(cfg)          # (1, ..., 30); the single source of the schedule
    # Execution-only storage: which config names the physical face store, containment-checked.
    # Fails closed on a missing/unparseable config, an escaping root or an absent faces directory.
    storage = D.resolve_storage(require_faces=require_faces)
    return {"config": cfg, "population": pop, "class_weights": weights, "environment": env,
            "resnet18_weight_sha256": weight_sha, "validation_epochs": list(epochs),
            "storage": storage,
            "execution_config": D.execution_config_provenance()}


def _provenance(pre: dict, det: dict, device) -> dict:
    import subprocess

    import torch
    import torchvision

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "torchvision": torchvision.__version__,
           "cuda_available": bool(torch.cuda.is_available()),
           "cuda_version": torch.version.cuda,
           "cudnn_version": torch.backends.cudnn.version(),
           "device": str(device),
           "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
           "gpu_capability": (".".join(map(str, torch.cuda.get_device_capability(0)))
                              if torch.cuda.is_available() else None)}
    try:
        import cv2
        env["opencv"] = cv2.__version__
    except Exception:
        env["opencv"] = None
    exec_prov = pre["execution_config"]
    return {
        **exec_prov,                 # m5_execution_config_path / _sha256 / faces_256_root_resolved
        "frozen_config_sha256": sha256_file(FROZEN_CONFIG),
        "frozen_config_snapshot_sha256": sha256_file(FROZEN_SNAPSHOT),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "resnet18_weight_sha256": pre["resnet18_weight_sha256"],
        "class_weights": [float(x) for x in pre["class_weights"]],
        "determinism": det,
        "environment": env,
        "environment_lock": {**{f"env.{k}": v for k, v in env.items()},
                             **{f"determinism.{k}": v for k, v in det.items()},
                             **exec_prov,
                             "frozen_config_sha256": sha256_file(FROZEN_CONFIG),
                             "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
                             "resnet18_weight_sha256": pre["resnet18_weight_sha256"]},
        "code_commit": git("rev-parse", "HEAD"),
        "code_dirty": bool(git("status", "--porcelain")),
        "host": platform.node(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ------------------------------------------------------------------ the single entry point
def run(config_path: Path | None = None, *, dry_run: bool = False,
        allow_cpu_dry_run: bool = True) -> dict:
    """Authoritative training (dry_run=False) or the dry-run contract preflight (dry_run=True)."""
    require_cuda = not (dry_run and allow_cpu_dry_run)
    pre = preflight(config_path, require_cuda=require_cuda)
    seed_everything()
    det = apply_determinism()
    device = resolve_device(require_cuda=require_cuda)
    model = MODEL.build(SEED)
    report = MODEL.trainable_report(model)
    if not report["fully_trainable"] or report["out_features"] != K:
        raise ProbeContractViolation(f"model contract violated: {report}")
    criterion = build_loss(pre["class_weights"])
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    out = {"dry_run": dry_run, "determinism": det, "model": report,
           "class_weights": [float(x) for x in pre["class_weights"]],
           "classes": list(CLASSES), "environment": pre["environment"],
           "resnet18_weight_sha256": pre["resnet18_weight_sha256"],
           "lr_schedule": lr_schedule(), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
           "validation_epochs": pre["validation_epochs"],
           "validation_passes": len(pre["validation_epochs"]),
           "device": str(device), "checkpoint_written": False,
           **pre["execution_config"]}

    if dry_run:
        # No optimizer.step(), no loaders touched, no checkpoint, no audit file.
        out["note"] = ("Dry-run preflight only: no optimizer.step(), no checkpoint, "
                       "no scientific claim.")
        return out

    # ---- authoritative run: CUDA + AMP, then the frozen per-epoch control flow ----
    autocast, scaler = build_amp(device)
    model.to(device)
    criterion.to(device)
    dl = pre["config"]["dataloader"]
    gpu = dl["gpu_defaults"]
    train_loader, val_loader, train_ds, val_ds = D.make_loaders(
        num_workers=int(gpu["num_workers"]), pin_memory=bool(gpu["pin_memory"]),
        persistent_workers=bool(gpu["persistent_workers"]),
        prefetch_factor=int(gpu["prefetch_factor"]), batch_size=int(dl["train"]["batch_size"]))
    if train_ds.split != "TRAIN" or val_ds.split != "VAL":
        raise ProbeContractViolation("loaders must be TRAIN and VAL only")
    provenance = _provenance(pre, det, device)
    loop = _train_loop(model=model, train_loader=train_loader, val_loader=val_loader,
                       criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                       scaler=scaler, autocast=autocast, device=device, provenance=provenance)
    audit = write_audit_outputs(loop, provenance)
    out.update({
        "checkpoint_written": CHECKPOINT_PATH.is_file(),
        "checkpoint_path": _rel(CHECKPOINT_PATH),
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "best_epoch": loop["best_epoch"], "best_macro_f1": loop["best_macro_f1"],
        "epochs_run": loop["epochs_run"], "scheduler_steps": loop["scheduler_steps"],
        "events": loop["events"], "epoch_rows": loop["epoch_rows"],
        "audit": audit["written"], "provenance": provenance,
        "amp": {"autocast_device_type": "cuda", "autocast_dtype": "float16",
                "grad_scaler_enabled": True, "val_autocast": VAL_AUTOCAST},
        "test_split_used": False, "synthetic_samples_used": False,
    })
    return out
