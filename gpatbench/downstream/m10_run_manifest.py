"""Reproducibility manifest v2 of one M10 ResNet-18 detector run (stdlib only). Additive to science_manifest (v1, kept
unchanged with its committed template): v2 adds every item required by the owner decisions record
(configs/amendments/m10_resnet18_owner_decisions.yaml reproducibility_requirements). A manifest that lacks a field,
uses a wrong type or contradicts the matrix / owner decisions is refused.

The manifest is written by the launcher at the end of a run; `audit` is always PENDING_INDEPENDENT_AUDIT there -- the
independent auditor (tools/m10_run_auditor.py) writes its own record and never edits the manifest.
"""
from __future__ import annotations

import re

SCHEMA = "m10_resnet18_run_manifest_v2"
SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
GROUPS = ("E00", "E01", "E02", "E03", "E04", "E05", "E06b", "E06c")
SEEDS = (42, 1337, 2026)
LABELS = ("SCIENTIFIC", "NON_SCIENTIFIC_QUALIFICATION")
TRACKS = {"E00": "binary", "E01": "A_FAIR_COMMON_IDENTITY_FREE", "E02": "A_FAIR_COMMON_IDENTITY_FREE",
          "E03": "A_FAIR_COMMON_IDENTITY_FREE", "E04": "A_FAIR_COMMON_IDENTITY_FREE", "E05": "A_FAIR_COMMON_IDENTITY_FREE",
          "E06b": "B_NATIVE_FULL_SECONDARY", "E06c": "A_FAIR_COMMON_IDENTITY_FREE"}       # O7 A_CONFIRM

REQUIRED = {
    "schema": str, "label": str, "run_id": str, "method_id": str, "track_id": str, "detector_seed": int,
    "bank_seed": (int, type(None)), "source_commit": str, "execution_commit": str, "code_sha256": dict,
    "evaluator_contract_sha256": str, "frozen_config_sha256": str, "owner_decisions_sha256": str,
    "precision_pin_sha256": str, "launch_authority": (dict, type(None)), "storage_authority": (dict, type(None)),
    "split_manifest_sha256": str, "m2_accounting_sha256": str, "pool_sha256": dict, "pool_counts": dict,
    "synthetic_bank": (dict, type(None)), "weights_sha256": str, "environment": dict, "gpu": dict,
    "precision_policy": dict, "grad_scaler": (dict, type(None)), "determinism": dict, "amp_observed": dict,
    "optimizer": dict, "scheduler": dict, "rng": dict, "command": str, "argv": list, "working_directory": str,
    "run_root": str, "start_utc": str, "end_utc": str, "epochs": int, "warmup_epochs": int, "steps_per_epoch": int,
    "total_steps": int, "optimizer_steps_applied": int, "optimizer_steps_skipped": int, "epoch_metrics": list,
    "checkpoint": dict, "val_predictions": list, "selection": dict, "logs": dict, "firewall": dict,
    "exit_status": dict, "audit": dict, "reproduction": list, "test_accessed": bool, "backup": str,
}
ENV_KEYS = ("name", "python", "sys_prefix", "lock_path", "lock_sha256", "torch", "torchvision", "cuda", "cudnn")
GPU_KEYS = ("name", "uuid", "driver")
SELECTION_KEYS = ("selected_epoch", "checkpoint_state_sha256", "checkpoint_file_sha256", "val_acer", "val_auc",
                  "frozen_val_threshold_hex", "selection_rule", "threshold_rule", "selection_source",
                  "selection_reasoning", "threshold_tie_break")
EPOCH_KEYS = ("epoch", "train_loss_mean", "optimizer_steps_applied", "optimizer_steps_skipped", "val_acer", "val_auc",
              "threshold_hex", "state_sha256", "predictions_sha256", "became_best")


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
    if doc["schema"] != SCHEMA or m not in GROUPS or s not in SEEDS or doc["label"] not in LABELS:
        problems.append("schema / method / seed / label")
    if doc["track_id"] != TRACKS.get(m):
        problems.append("O7: track_id differs from the confirmed method track")
    if m == "E00":
        if doc["bank_seed"] is not None or doc["synthetic_bank"] is not None:
            problems.append("E00 is real-only")
    elif doc["bank_seed"] != s or not isinstance(doc["synthetic_bank"], dict) \
            or doc["synthetic_bank"].get("seed") != s or doc["synthetic_bank"].get("method_id") != m \
            or not SHA.match(str(doc["synthetic_bank"].get("manifest_sha256"))):
        problems.append("D12: the synthetic bank must be the seed-matched bank of the same method")
    for k in ("evaluator_contract_sha256", "frozen_config_sha256", "split_manifest_sha256", "m2_accounting_sha256",
              "owner_decisions_sha256", "precision_pin_sha256", "weights_sha256"):
        if not SHA.match(doc[k]):
            problems.append(f"{k} is not a sha256")
    for k in ("source_commit", "execution_commit"):
        if not COMMIT.match(doc[k]):
            problems.append(f"{k} is not a full commit")
    if not doc["code_sha256"] or any(not SHA.match(str(v)) for v in doc["code_sha256"].values()):
        problems.append("code_sha256")
    if any(k not in doc["environment"] for k in ENV_KEYS) or any(k not in doc["gpu"] for k in GPU_KEYS):
        problems.append("environment / gpu identity incomplete")
    if doc["label"] == "SCIENTIFIC":
        if (doc["epochs"], doc["warmup_epochs"]) != (40, 3):
            problems.append("scientific runs use the frozen 40-epoch / 3-warmup schedule")
        if doc["launch_authority"] is None or doc["storage_authority"] is None:
            problems.append("scientific runs bind the launch and storage authorities")
        if doc["precision_policy"].get("name") != "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC":
            problems.append("O1 precision policy")
    if doc["total_steps"] != doc["epochs"] * doc["steps_per_epoch"]:
        problems.append("total_steps != epochs * steps_per_epoch")
    if doc["optimizer_steps_applied"] + doc["optimizer_steps_skipped"] != doc["total_steps"]:
        problems.append("applied + skipped optimizer steps != total_steps")
    if len(doc["epoch_metrics"]) != doc["epochs"] or any(any(k not in e for k in EPOCH_KEYS)
                                                         for e in doc["epoch_metrics"]):
        problems.append("one complete TRAIN/VAL metric record per epoch is required")
    if any(k not in doc["selection"] for k in SELECTION_KEYS):
        problems.append("selection record incomplete")
    elif not (0 <= doc["selection"]["selected_epoch"] < doc["epochs"]):
        problems.append("selected epoch out of range")
    if len(doc["val_predictions"]) != doc["epochs"] or any(not SHA.match(str(p.get("sha256")))
                                                           for p in doc["val_predictions"]):
        problems.append("one VAL prediction file (sha256) per epoch is required")
    if not SHA.match(str(doc["checkpoint"].get("file_sha256"))) or not doc["checkpoint"].get("path"):
        problems.append("checkpoint path / sha256")
    if doc["test_accessed"] is not False or "VAL" not in str(doc["selection"].get("selection_source", "")):
        problems.append("selection must be VAL-only and TEST untouched")
    if doc["firewall"].get("violation_count") != 0 or doc["firewall"].get("listing_attempts") != 0:
        problems.append("TEST firewall violation recorded")
    if doc["exit_status"].get("exit_code") != 0:
        problems.append("exit status is not 0")
    if doc["audit"].get("status") != "PENDING_INDEPENDENT_AUDIT":
        problems.append("the run manifest never self-certifies an audit")
    if "independent" not in doc["backup"]:
        problems.append("backup statement must say whether an independent copy exists")
    if not doc["reproduction"]:
        problems.append("reproduction instructions")
    if problems:
        raise ManifestError(problems)
    return {"ok": True, "method_id": m, "detector_seed": s}
