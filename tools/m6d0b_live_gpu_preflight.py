#!/usr/bin/env python3
"""Validate recorded M6D0b evidence; no SSH, ML imports, or scientific execution.

Default checks audit/contract integrity and the append-only ledger. --final also
streams indexed files to verify the finalized artifact index (including opaque
manifest bytes, never parsing manifests or opening their referenced images).
This is an infrastructure evidence validator, not an execution-readiness probe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HEAD = "a6f78e9ba403148cb92e7f46ad4c72037dfc0499"
PREFIX_SHA = "8855f49e9474f23aa7059a9b461db21e940f3a205ecb3dc219d98eda94169979"
AUDIT = "outputs/audit/M6D0B_LIVE_GPU_VERIFICATION.json"
LEDGER = "outputs/audit/EXECUTION_LEDGER.jsonl"
INDEX = "outputs/audit/ARTIFACT_INDEX.csv"
CREATED = {AUDIT, "outputs/audit/M6D0B_LIVE_GPU_VERIFICATION.md",
           "environments/m6_environment_live_addendum.md",
           "tools/m6d0b_live_gpu_preflight.py"}
ALLOWED = CREATED | {LEDGER, INDEX}
METHODS = {"E01", "E02", "E03", "E04", "E05", "E06c", "E07c"}
STATUSES = {"EXISTING_ENV_CANDIDATE", "NEW_ENV_REQUIRED",
            "BLOCKED_BY_VERSION_AMBIGUITY", "BLOCKED_BY_MISSING_ASSET",
            "BLOCKED_BY_GPU_REPO_NOT_SYNCED", "REQUIRES_COMPATIBILITY_PROBE"}


def require(ok, message):
    if not ok:
        raise SystemExit("FAIL: " + message)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="verify finalized index too")
    args = parser.parse_args()
    require(git("rev-parse", "HEAD").decode().strip() == HEAD, "laptop HEAD")
    require(git("branch", "--show-current").decode().strip() == "m6-baselines", "branch")
    require(git("rev-parse", "origin/m6-baselines").decode().strip() == HEAD, "origin tracking ref")
    require(git("rev-list", "--left-right", "--count", "HEAD...@{u}").split() == [b"0", b"0"], "divergence")
    changed = set(git("diff", "HEAD", "--name-only").decode().splitlines())
    untracked = set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    require(changed | untracked <= ALLOWED, "changes outside additive allowlist")
    require(CREATED <= untracked, "four deliverables must remain uncommitted additions")
    audit = json.loads((ROOT / AUDIT).read_text())
    require(audit["starting_commit"] == HEAD, "audit authority")
    require(audit["local_preflight"]["worktree_clean_at_start"] is True, "initial clean evidence")
    require(audit["ssh_authentication"] == "PUBLIC_KEY_WORKING", "SSH evidence")
    host = audit["gpu_repository"]
    require(host["clean_at_start"] and host["clean_at_end"], "GPU clean evidence")
    require(host["remote_branch_head"] == HEAD, "live remote head")
    require(host["branch"] == "m6-baselines", "GPU branch")
    counts = git("rev-list", "--left-right", "--count", host["head"] + "..." + HEAD).split()
    require([int(x) for x in counts] == [0, host["commits_behind"]], "ancestry evidence")
    require(audit["platform"]["gpu"] == "NVIDIA GeForce RTX 3090", "GPU identity")
    require(audit["runtime"]["exists"] and audit["runtime"]["faces_png_metadata_count"] == 20615, "runtime metadata")
    for rel, expected in audit["input_sha256"].items():
        require(sha(ROOT / rel) == expected, "input changed: " + rel)
    historical = json.loads((ROOT / "outputs/audit/M6D0_GPU_ENVIRONMENT_AUDIT.json").read_text())
    for method, row in historical["method_matrix"].items():
        rel = audit["method_matrix"][method]["config"]
        require(sha(ROOT / rel) == row["config_sha256"], "frozen config: " + method)
        require((ROOT / rel).read_bytes() == (ROOT / "frozen_config_snapshot" / rel).read_bytes(), "snapshot: " + method)
    for source in historical["source_checkouts"].values():
        actual = subprocess.check_output(["git", "-C", str(ROOT / source["path"]), "rev-parse", "HEAD"]).decode().strip()
        require(actual == source["commit"], "source pin: " + source["path"])
    packages = audit["gpat_m5"]["packages"]
    needed = {"torch", "torchvision", "tensorflow", "cv2", "numpy", "scipy", "sklearn", "pandas", "pyarrow", "PIL", "yaml", "tensorfn", "Cython", "onnxruntime", "lpips"}
    require(needed <= packages.keys(), "package inventory incomplete")
    for name, row in packages.items():
        require(row["status"] in {"AVAILABLE", "MISSING"}, "package status: " + name)
        require(bool(row.get("version") if row["status"] == "AVAILABLE" else row.get("error")), "package evidence: " + name)
    m5 = audit["gpat_m5"]
    live = {"python": m5["python"], "torch": packages["torch"]["version"],
            "torchvision": packages["torchvision"]["version"],
            "torch_cuda_runtime": m5["torch"]["cuda_runtime"], "cudnn": m5["torch"]["cudnn"]}
    for key, row in audit["historical_m5_comparison"].items():
        require(row["live"] == live[key], "live comparison: " + key)
        status = "MATCHES_HISTORICAL_M5" if row["historical"] == live[key] else "DIFFERS_FROM_HISTORICAL_M5"
        require(row["status"] == status, "comparison classification: " + key)
    require(set(audit["method_matrix"]) == METHODS, "method coverage")
    for method, row in audit["method_matrix"].items():
        require(bool(row["statuses"]) and set(row["statuses"]) <= STATUSES, "readiness vocabulary: " + method)
        require("BLOCKED_BY_GPU_REPO_NOT_SYNCED" in row["statuses"], "lag gate: " + method)
        if "EXISTING_ENV_CANDIDATE" in row["statuses"]:
            require(method in {"E01", "E02"} and row["candidate"] == "gpat-m5", "unsupported reuse")
    for name in ("numpy", "PIL", "PIL.ImageCms", "cv2", "yaml"):
        require(packages[name]["status"] == "AVAILABLE", "core candidate import: " + name)
    require(len(audit["environment_isolation_matrix"]) == 6, "isolation coverage")
    for row in audit["environment_isolation_matrix"].values():
        require(row["status"] in {"REUSE_EXISTING_CANDIDATE", "CREATE_NEW_CANDIDATE", "SPLIT_REQUIRED", "STILL_UNRESOLVED"}, "isolation status")
    expected_assets = historical["external_assets"]["E04"]["assets"]
    expected_assets = {x["path"]: (x["size"], x["sha256"]) for x in expected_assets}
    light = historical["external_assets"]["E06c"]
    expected_assets[Path(light["path"]).name] = (light["size"], light["sha256"])
    require(len(audit["external_assets"]["observations"]) == 5, "asset coverage")
    for row in audit["external_assets"]["observations"]:
        require((row["expected_size"], row["expected_sha256"]) == expected_assets[row["asset"]], "asset scientific identity")
        for candidate in row["candidates"]:
            if candidate["exists"]:
                require(candidate.get("match") is True, "asset mismatch or unreadable existing asset")
    require(audit["external_assets"]["E07c"]["status"] == "NOT_YET_TRAINED", "encoder status")
    require(all(value is False for value in audit["safety"].values()), "safety declarations")
    require(not list((ROOT / "environments").glob("m6*.lock*")), "premature M6 lock")
    prefix = git("show", HEAD + ":" + LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == 95 and hashlib.sha256(prefix).hexdigest() == PREFIX_SHA, "committed ledger prefix")
    require(current.startswith(prefix) and len(current.splitlines()) == 96, "append-only ledger")
    suffix = current[len(prefix):]
    require(len(suffix.splitlines()) == 1, "exactly one appended row")
    row = json.loads(suffix)
    require(row["classification"] == "M6D0B_LIVE_GPU_VERIFICATION", "ledger classification")
    require(row["previous_ledger_sha256"] == PREFIX_SHA, "ledger prefix hash")
    for rel, expected in row["file_sha256"].items():
        require(sha(ROOT / rel) == expected, "ledger artifact hash: " + rel)
    indexed = None
    if args.final:
        data = (ROOT / INDEX).read_bytes()
        require(data.count(b"\n") == data.count(b"\r\n"), "index CRLF")
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        paths = [r["path"] for r in rows]
        require(len(paths) == len(set(paths)), "index unique paths")
        require(CREATED <= set(paths), "index new deliverables")
        for item in rows:
            path = ROOT / item["path"]
            require(path.is_file() and path.stat().st_size == int(item["size_bytes"]), "index size: " + item["path"])
            require(sha(path) == item["sha256"], "index hash: " + item["path"])
        indexed = len(rows)
    print(json.dumps({"result": "LIVE_AUDIT_VALIDATION_PASS", "execution_ready": False,
                      "method_configs_verified": 7, "source_pins_verified": 6,
                      "ledger_prefix_rows": 95, "ledger_final_rows": 96,
                      "ledger_prefix_sha256": PREFIX_SHA, "ledger_prefix_byte_identical": True,
                      "firewall": "PASS", "index_paths_verified": indexed,
                      "scientific_execution": False}, indent=2))


if __name__ == "__main__":
    main()
