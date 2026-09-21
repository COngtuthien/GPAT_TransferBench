"""Frozen M6B config loading.

The M6B configs are immutable. This module *consumes* them: it never repairs, defaults or
overrides a scientific value. A config that is missing a required field, disagrees with its
frozen snapshot, or declares a TEST-reading intent is REFUSED, not silently fixed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "methods"
SNAPSHOT_DIR = ROOT / "frozen_config_snapshot" / "configs" / "methods"
LOGGING_CONTRACT = ROOT / "configs" / "run_logging_v1.yaml"
LOGGING_SNAPSHOT = ROOT / "frozen_config_snapshot" / "configs" / "run_logging_v1.yaml"

#: method_id -> frozen config filename, exactly as M6B froze them.
METHOD_FILES = {
    "E01": "e01_fas_aug.yaml",
    "E02": "e02_freqsub.yaml",
    "E03": "e03_stdn.yaml",
    "E04": "e04_physics_std.yaml",
    "E05": "e05_pcgan.yaml",
    "E06c": "e06c_dsdg_bin_idfree.yaml",
    "E07c": "e07c_difffas_bin_idfree.yaml",
}

REQUIRED_TOP_LEVEL = (
    "method_id", "status", "milestone_status", "fidelity_class", "fidelity_provenance",
    "seeds", "data", "checkpoint", "run_layout", "logging", "logging_contract",
)


class FrozenConfigError(RuntimeError):
    """A frozen M6B config is missing, altered, internally inconsistent or TEST-contaminated."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _require(cfg: dict, path: str, ctx: str) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise FrozenConfigError(f"{ctx}: frozen field '{path}' is missing. Refusing to proceed "
                                    f"(M6C1 §1: do not silently repair the config).")
        node = node[part]
    return node


def _assert_test_firewall(cfg: dict, ctx: str) -> None:
    test = _require(cfg, "data.splits.TEST", ctx)
    if test.get("allowed") is not False:
        raise FrozenConfigError(f"{ctx}: data.splits.TEST.allowed must be false.")
    if test.get("used_for") != []:
        raise FrozenConfigError(f"{ctx}: data.splits.TEST.used_for must be empty.")
    for flag in ("never_training", "never_checkpoint_selection",
                 "never_hyperparameter_selection", "never_reconstruction_choice", "never_seed_selection"):
        if test.get(flag) is not True:
            raise FrozenConfigError(f"{ctx}: data.splits.TEST.{flag} must be true.")
    if test.get("code_path_present") is not False:
        raise FrozenConfigError(f"{ctx}: data.splits.TEST.code_path_present must be false.")
    if _require(cfg, "data.splits.VAL", ctx).get("may_select_checkpoint") is not False:
        raise FrozenConfigError(f"{ctx}: VAL must not select a checkpoint.")


def load_method_config(method_id: str, *, config_path: str | Path | None = None,
                       snapshot_path: str | Path | None = None) -> dict:
    """Load, verify and return a frozen M6B method config.

    Verifies: the file exists; it parses; `method_id` matches; the required fields are present;
    the bytes are identical to the frozen snapshot; and the TEST firewall holds. Returns the
    parsed config with `_config_path`, `_config_sha256` and `_snapshot_path` attached under the
    reserved `_runtime` key (never a scientific field).
    """
    if method_id not in METHOD_FILES:
        raise FrozenConfigError(f"unknown method_id {method_id!r}; expected one of "
                                f"{sorted(METHOD_FILES)}")
    fn = METHOD_FILES[method_id]
    cfg_p = Path(config_path) if config_path else CONFIG_DIR / fn
    snap_p = Path(snapshot_path) if snapshot_path else SNAPSHOT_DIR / fn
    ctx = f"{method_id} ({cfg_p})"

    if not cfg_p.is_file():
        raise FrozenConfigError(f"{ctx}: frozen config not found.")
    if not snap_p.is_file():
        raise FrozenConfigError(f"{ctx}: frozen snapshot not found at {snap_p}.")

    raw = cfg_p.read_bytes()
    snap = snap_p.read_bytes()
    if raw != snap:
        raise FrozenConfigError(
            f"{ctx}: config bytes differ from the frozen snapshot "
            f"({sha256_bytes(raw)} != {sha256_bytes(snap)}). REFUSED.")

    cfg = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(cfg, dict):
        raise FrozenConfigError(f"{ctx}: config did not parse to a mapping.")
    if cfg.get("method_id") != method_id:
        raise FrozenConfigError(f"{ctx}: method_id mismatch "
                                f"({cfg.get('method_id')!r} != {method_id!r}).")
    for field in REQUIRED_TOP_LEVEL:
        _require(cfg, field, ctx)
    if cfg.get("status") != "CONFIG_FROZEN":
        raise FrozenConfigError(f"{ctx}: status must be CONFIG_FROZEN.")
    _assert_test_firewall(cfg, ctx)

    seeds = _require(cfg, "seeds.experiment_seeds", ctx)
    if seeds != [42, 1337, 2026]:
        raise FrozenConfigError(f"{ctx}: experiment_seeds must be [42, 1337, 2026], got {seeds}.")

    frozen = json.loads((ROOT / "outputs/audit/M6B_CONFIG_FREEZE.json").read_text())
    expected = next(m["config_sha256"] for m in frozen["methods"] if m["method_id"] == method_id)
    if sha256_bytes(raw) != expected:
        raise FrozenConfigError(f"{ctx}: SHA256 differs from M6B freeze record")
    cfg["_runtime"] = {
        "config_path": str(cfg_p.relative_to(ROOT)) if cfg_p.is_relative_to(ROOT) else str(cfg_p),
        "config_sha256": sha256_bytes(raw),
        "snapshot_path": str(snap_p.relative_to(ROOT)) if snap_p.is_relative_to(ROOT) else str(snap_p),
        "snapshot_verified": True,
    }
    return cfg


def load_logging_contract() -> dict:
    """Load `configs/run_logging_v1.yaml`, verified byte-identical against its snapshot."""
    raw = LOGGING_CONTRACT.read_bytes()
    if raw != LOGGING_SNAPSHOT.read_bytes():
        raise FrozenConfigError("run_logging_v1.yaml differs from its frozen snapshot. REFUSED.")
    frozen = json.loads((ROOT / "outputs/audit/M6B_CONFIG_FREEZE.json").read_text())
    if sha256_bytes(raw) != frozen["logging_contract"]["sha256"]:
        raise FrozenConfigError("logging contract SHA256 differs from M6B freeze record")
    cfg = yaml.safe_load(raw.decode("utf-8"))
    if cfg.get("version") != "run_logging_v1":
        raise FrozenConfigError("run_logging_v1.yaml: unexpected version.")
    return cfg
