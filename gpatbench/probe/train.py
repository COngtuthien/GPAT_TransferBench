"""The single authoritative ArtifactProbeNet trainer (spec §12.1 + owner resolutions).

There is exactly one training code path in this project and it is `run()` below. It refuses rather
than adapts: a non-frozen config, a changed config or manifest hash, a wrong ResNet weight hash, a
CPU authoritative run, disabled AMP, a TEST split, synthetic data, a class-count mismatch or a
class-weight mismatch all raise `ProbeContractViolation`.

`run(dry_run=True)` performs the CPU-safe preflight: it validates the contract, builds the model,
loss, metric and scheduler, and does a single forward pass -- but never calls `optimizer.step()`
and never writes a checkpoint.

**The training loop is NOT implemented yet.** `run(dry_run=False)` raises before any optimization,
and it stays that way until the GPU execution preflight is complete. When it is written it must
follow the frozen validation contract: a VAL pass at the END of **every** epoch --
`contract.validation_epochs()` returns `(1, 2, ..., 30)`, i.e. 30 evaluations -- with the best
checkpoint allowed to come from any epoch in that range, replaced only on a strict
`new_macro_f1 > best_macro_f1` so an exact tie keeps the earlier epoch
(`contract.select_best_epoch`). TEST never participates.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as MET
from . import model as MODEL
from .contract import (BATCH_SIZE, CLASSES, EPOCHS, ETA_MIN, K, LR, RESNET18_WEIGHT_SHA256, ROOT,
                       SEED, T_MAX, WEIGHT_DECAY, ProbeContractViolation, load_config,
                       lr_schedule, validation_epochs, verify_class_weights, verify_environment,
                       verify_population)

CHECKPOINT_DIR = ROOT / "models/artifact_probe"
CHECKPOINT_PATH = CHECKPOINT_DIR / "artifact_probe_v1.pt"


def seed_everything(seed: int = SEED) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_determinism() -> dict:
    """Frozen determinism contract. Never silently relaxed: a missing kernel must stop the run."""
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"cudnn_benchmark": False, "cudnn_deterministic": True,
            "use_deterministic_algorithms": True, "tf32_matmul": False, "tf32_cudnn": False,
            "float32_matmul_precision": "highest"}


def verify_backbone_weights() -> str:
    """Hash the actual torchvision cache file for IMAGENET1K_V1 and refuse any substitute."""
    import hashlib

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


def preflight(config_path: Path | None = None, *, require_cuda: bool) -> dict:
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
    if cfg["input"]["post_hp_normalization"]["imagenet_mean_subtraction"]:
        raise ProbeContractViolation("post-high-pass ImageNet normalization is forbidden")
    epochs = validation_epochs(cfg)          # (1, ..., 30); the single source of the schedule
    return {"config": cfg, "population": pop, "class_weights": weights, "environment": env,
            "resnet18_weight_sha256": weight_sha, "validation_epochs": list(epochs)}


def run(config_path: Path | None = None, *, dry_run: bool = False,
        allow_cpu_dry_run: bool = True) -> dict:
    """Authoritative training (dry_run=False) or the CPU-safe contract preflight (dry_run=True)."""
    import torch
    require_cuda = not (dry_run and allow_cpu_dry_run)
    pre = preflight(config_path, require_cuda=require_cuda)
    seed_everything()
    det = apply_determinism()
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
           "checkpoint_written": False}
    if dry_run:
        out["note"] = ("CPU-safe preflight only: no optimizer.step(), no checkpoint, "
                       "no scientific claim.")
        return out
    raise ProbeContractViolation(
        "the authoritative M5 training loop is NOT implemented yet. The frozen contract, the gates "
        "and the CPU-safe preflight are ready; the loop will validate at the end of every epoch "
        f"{validation_epochs()[0]}..{validation_epochs()[-1]} "
        f"({len(validation_epochs())} passes). Execution is gated on the GPU preflight "
        "(outputs/audit/M5_GPU_EXECUTION_PLAN.md).")
