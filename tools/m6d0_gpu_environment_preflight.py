#!/usr/bin/env python3
"""Validate the M6D0 static environment audit and proposal without installing.

This tool deliberately performs no SSH, imports no ML framework, constructs no
model, and reads no manifest or benchmark image. A successful exit validates
the internal audit/plan; it does not declare the GPU execution host ready.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/audit/M6D0_GPU_ENVIRONMENT_AUDIT.json"
PLAN = ROOT / "environments/m6_environment_plan.md"
EXPECTED_HEAD = "f2a625b1b2ee832e3517561e8f64a8cde3ae9d0c"
METHODS = ("E01", "E02", "E03", "E04", "E05", "E06c", "E07c")
CONFIGS = {
    "E01": "configs/methods/e01_fas_aug.yaml",
    "E02": "configs/methods/e02_freqsub.yaml",
    "E03": "configs/methods/e03_stdn.yaml",
    "E04": "configs/methods/e04_physics_std.yaml",
    "E05": "configs/methods/e05_pcgan.yaml",
    "E06c": "configs/methods/e06c_dsdg_bin_idfree.yaml",
    "E07c": "configs/methods/e07c_difffas_bin_idfree.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main() -> None:
    assert git("rev-parse", "HEAD") == EXPECTED_HEAD
    assert git("branch", "--show-current") == "m6-baselines"
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["starting_commit"] == EXPECTED_HEAD
    assert audit["audit_classification"] == "AUDIT_PLANNING_ONLY"
    assert audit["gpu_host"]["live_audit_status"] == "BLOCKED_AUTHENTICATION"
    assert audit["gpu_host"]["current_facts_verified"] is False
    assert audit["execution_readiness"] == "BLOCKED_PENDING_READ_ONLY_GPU_AUDIT_AND_COMPATIBILITY_PROBES"
    assert audit["method_matrix"].keys() == set(METHODS)
    allowed = {"SOURCE_PINNED", "SOURCE_RANGE_ONLY", "VERSION_UNSPECIFIED",
               "COMPATIBILITY_REQUIRES_PROBE"}
    for method, rel in CONFIGS.items():
        path = ROOT / rel
        snap = ROOT / "frozen_config_snapshot" / rel
        assert path.read_bytes() == snap.read_bytes(), rel
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["method_id"] == method
        assert audit["method_matrix"][method]["config_sha256"] == sha256(path)
        assert set(audit["method_matrix"][method]["requirement_status"]) <= allowed
    sources = audit["source_checkouts"]
    for row in sources.values():
        root = ROOT / row["path"]
        assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip() == row["commit"]
    assert audit["external_assets"]["E07c"]["status"] == "NOT_YET_TRAINED"
    assert audit["safety"]["package_install"] is False
    assert audit["safety"]["environment_creation"] is False
    assert audit["safety"]["model_construction"] is False
    text = PLAN.read_text(encoding="utf-8")
    for env in ("env-m6-core", "env-m6-stdn", "env-m6-physics-std",
                "env-m6-pcgan", "env-m6-dsdg", "env-m6-difffas"):
        assert env in text
    assert "PROPOSED — NOT AN EXECUTION LOCK" in text
    assert not list((ROOT / "environments").glob("m6_*.lock.txt"))
    result = {
        "result": "PLAN_VALIDATION_PASS",
        "execution_readiness": audit["execution_readiness"],
        "gpu_live_audit": "BLOCKED_AUTHENTICATION",
        "method_configs_verified": len(METHODS),
        "source_checkouts_verified": len(sources),
        "final_locks_created": 0,
        "packages_installed": 0,
        "environments_created": 0,
        "model_construction": False,
        "benchmark_image_opens": 0,
        "manifest_opens": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
