"""Reproducibility manifest of one M10 scientific ResNet-18 detector run (stdlib only; validated before any record is
accepted). Every field below is required; a run manifest that lacks one, uses a wrong type or contradicts the matrix
(E00 real-only, detector seed = bank seed, frozen 40/3 schedule, VAL-only selection, TEST untouched) is refused.
"""
from __future__ import annotations

import re

SCHEMA = "m10_resnet18_run_manifest_v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
GROUPS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
SEEDS = (42, 1337, 2026)

REQUIRED = {
    "schema": str, "label": str, "method_id": str, "detector_seed": int, "bank_seed": (int, type(None)),
    "source_commit": str, "execution_commit": str, "evaluator_contract_sha256": str, "frozen_config_sha256": str,
    "split_manifest_sha256": str, "m2_accounting_sha256": str, "pool_sha256": dict, "synthetic_bank": (dict, type(None)),
    "environment": dict, "gpu": dict, "precision_policy": dict, "rng": dict, "command": str, "working_directory": str,
    "start_utc": str, "end_utc": str, "epochs": int, "warmup_epochs": int, "steps_per_epoch": int, "total_steps": int,
    "checkpoint": dict, "val_predictions": list, "selection": dict, "logs": dict, "audit": dict, "reproduction": list,
    "test_accessed": bool, "backup": str,
}
ENV_KEYS = ("name", "python", "lock_path", "lock_sha256", "torch", "torchvision", "cuda")
GPU_KEYS = ("name", "uuid", "driver")
SELECTION_KEYS = ("selected_epoch", "checkpoint_state_sha256", "checkpoint_file_sha256", "val_acer", "val_auc",
                  "frozen_val_threshold_hex", "selection_rule", "threshold_rule", "selection_source")


class ManifestError(ValueError):
    pass


def validate(doc: dict) -> dict:
    problems = []
    for k, t in REQUIRED.items():
        if k not in doc:
            problems.append(f"missing {k}")
        elif not isinstance(doc[k], t) or (t is int and isinstance(doc[k], bool)):
            problems.append(f"{k} has type {type(doc[k]).__name__}")
    if problems:
        raise ManifestError(problems)
    m, s = doc["method_id"], doc["detector_seed"]
    if doc["schema"] != SCHEMA or m not in GROUPS or s not in SEEDS:
        problems.append("schema / method / seed")
    if doc["label"] not in ("SCIENTIFIC", "NON_SCIENTIFIC_QUALIFICATION"):
        problems.append("label")
    if m == "E00":
        if doc["bank_seed"] is not None or doc["synthetic_bank"] is not None:
            problems.append("E00 is real-only")
    elif doc["bank_seed"] != s or not isinstance(doc["synthetic_bank"], dict) \
            or doc["synthetic_bank"].get("seed") != s or doc["synthetic_bank"].get("method_id") != m \
            or not SHA.match(str(doc["synthetic_bank"].get("manifest_sha256"))):
        problems.append("D12: the synthetic bank must be the seed-matched bank of the same method")
    for k in ("evaluator_contract_sha256", "frozen_config_sha256", "split_manifest_sha256", "m2_accounting_sha256"):
        if not SHA.match(doc[k]):
            problems.append(f"{k} is not a sha256")
    for k in ("source_commit", "execution_commit"):
        if not COMMIT.match(doc[k]):
            problems.append(f"{k} is not a full commit")
    if any(k not in doc["environment"] for k in ENV_KEYS) or any(k not in doc["gpu"] for k in GPU_KEYS):
        problems.append("environment / gpu identity incomplete")
    if doc["label"] == "SCIENTIFIC" and (doc["epochs"], doc["warmup_epochs"]) != (40, 3):
        problems.append("scientific runs use the frozen 40-epoch / 3-warmup schedule")
    if doc["total_steps"] != doc["epochs"] * doc["steps_per_epoch"]:
        problems.append("total_steps != epochs * steps_per_epoch")
    if any(k not in doc["selection"] for k in SELECTION_KEYS):
        problems.append("selection record incomplete")
    elif not (0 <= doc["selection"]["selected_epoch"] < doc["epochs"]):
        problems.append("selected epoch out of range")
    if len(doc["val_predictions"]) != doc["epochs"] or any(not SHA.match(str(p.get("sha256"))) for p in doc["val_predictions"]):
        problems.append("one VAL prediction file (sha256) per epoch is required")
    if not SHA.match(str(doc["checkpoint"].get("file_sha256"))) or not doc["checkpoint"].get("path"):
        problems.append("checkpoint path / sha256")
    if doc["test_accessed"] is not False or doc["selection"].get("selection_source", "").find("VAL") < 0:
        problems.append("selection must be VAL-only and TEST untouched")
    if "independent" not in doc["backup"] and "no independent" not in doc["backup"]:
        problems.append("backup statement must say whether an independent copy exists")
    if problems:
        raise ManifestError(problems)
    return {"ok": True, "method_id": m, "detector_seed": s}
