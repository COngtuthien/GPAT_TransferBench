#!/usr/bin/env python3
"""Validate recorded M6D0c synchronization evidence without SSH or mutation.

Only standard-library evidence checks and read-only local Git queries run here.
--final additionally validates index coverage, sizes, hashes and CRLF bytes.
Index hashing treats tracked manifests as opaque bytes; no images are opened.
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
PREVIOUS = "89ea43e3d32d87d0641ec57bb62e6c72694dbe8a"
TARGET = "4bc78bbabedc73e58d24874097506b28d50b0390"
PREFIX_SHA = "a3d8a59c8ba3eea4c7ff327dea99313e99c2b52428ba1713a67a8bec84293717"
AUDIT = "outputs/audit/M6D0C_GPU_REPOSITORY_SYNC.json"
REPORT = "outputs/audit/M6D0C_GPU_REPOSITORY_SYNC.md"
TOOL = "tools/m6d0c_gpu_repo_sync_preflight.py"
LEDGER = "outputs/audit/EXECUTION_LEDGER.jsonl"
INDEX = "outputs/audit/ARTIFACT_INDEX.csv"
CREATED = {AUDIT, REPORT, TOOL}
ALLOWED = CREATED | {LEDGER, INDEX}
FETCH = "git fetch --no-tags origin refs/heads/m6-baselines:refs/remotes/origin/m6-baselines"
MERGE = "git merge --ff-only origin/m6-baselines"
SPOTS = {
    "configs/methods/e01_fas_aug.yaml", "configs/methods/e03_stdn.yaml",
    "configs/methods/e04_physics_std.yaml", "configs/methods/e05_pcgan.yaml",
    "configs/methods/e06c_dsdg_bin_idfree.yaml",
    "configs/methods/e07c_difffas_bin_idfree.yaml", "configs/run_logging_v1.yaml",
    "environments/m6_environment_live_addendum.md",
    "outputs/audit/M6D0B_LIVE_GPU_VERIFICATION.json", "tools/m6d0b_live_gpu_preflight.py",
}
SAFETY = {
    "scientific_content_edit", "runtime_write", "package_install", "environment_creation",
    "environment_mutation", "asset_deployment", "model_construction", "weight_load",
    "benchmark_image_decoding", "training", "inference", "sampling", "synthetic_bank",
    "checkpoint", "test_execution", "optimizer", "backward", "laptop_commit",
    "laptop_push", "gpu_commit", "manual_gpu_repo_edit", "environment_lock_creation",
}


def require(ok, message):
    if not ok:
        raise SystemExit("FAIL: " + message)


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(rows, name):
    matches = [r for r in rows if r["command"] == name]
    require(len(matches) == 1, "unique evidence command: " + name)
    require(matches[0]["exit_code"] == 0, "command failed: " + name)
    return matches[0]["stdout"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    require(git("rev-parse", "HEAD").decode().strip() == TARGET, "laptop HEAD")
    require(git("rev-parse", "origin/m6-baselines").decode().strip() == TARGET, "laptop origin")
    require(git("branch", "--show-current").decode().strip() == "m6-baselines", "laptop branch")
    require(git("rev-list", "--left-right", "--count", "HEAD...@{u}").split() == [b"0", b"0"], "laptop divergence")
    changed = set(git("diff", "HEAD", "--name-only", "-z").decode().strip("\0").split("\0")) - {""}
    untracked = set(git("ls-files", "--others", "--exclude-standard", "-z").decode().strip("\0").split("\0")) - {""}
    require(changed | untracked <= ALLOWED, "scientific firewall")
    require(CREATED == untracked, "exactly three uncommitted additions")
    require(not git("diff", "--cached", "--name-only"), "no staged changes")
    if args.final:
        require(changed | untracked == ALLOWED, "exactly five final paths")

    audit = json.loads((ROOT / AUDIT).read_text())
    expected = {
        "previous_head": PREVIOUS, "target_head": TARGET, "remote_head": TARGET,
        "commit_distance": 16, "ancestry": "FAST_FORWARD", "post_sync_head": TARGET,
        "origin_head": TARGET, "branch": "m6-baselines", "divergence": [0, 0],
        "gpu_worktree_clean": True, "faces_png_metadata_count": 20615,
        "fetch_command": FETCH, "fast_forward_command": MERGE,
    }
    for key, value in expected.items():
        require(audit[key] == value and type(audit[key]) is type(value), "recorded " + key)
    require(audit["local_preflight"] == {
        "head": TARGET, "origin_head": TARGET, "branch": "m6-baselines",
        "divergence": [0, 0], "worktree_clean": True,
    }, "initial laptop evidence")
    require(audit["ssh"]["hostname"] == "nd-System-Product-Name" and
            audit["ssh"]["user"] == "student20261" and
            audit["ssh"]["status"] == "PUBLIC_KEY_WORKING", "SSH identity")
    require(set(audit["safety"]) == SAFETY and all(v is False for v in audit["safety"].values()), "safety declarations")
    require(audit["asset_status"] == {
        "E04": "MISSING_NOT_DEPLOYED", "E06c_LightCNN": "MISSING_NOT_DEPLOYED",
        "E07c_encoder": "NOT_YET_TRAINED",
    }, "preserved asset blockers")
    evidence = audit["command_evidence"]
    pre, fetched, post = (evidence[k] for k in ("presync", "fetch", "postchecks"))
    require(command(pre, "git rev-parse HEAD") == PREVIOUS, "pre-sync HEAD")
    require(command(pre, "git branch --show-current") == "m6-baselines", "pre-sync branch")
    require(command(pre, "git status --porcelain") == "", "pre-sync clean")
    require(command(pre, "git ls-remote origin refs/heads/m6-baselines").split() == [TARGET, "refs/heads/m6-baselines"], "remote target")
    command(fetched, FETCH)
    require(audit["fetch_count"] == 1, "one controlled fetch")
    require(command(fetched, "git rev-parse HEAD") == PREVIOUS, "post-fetch HEAD")
    require(command(fetched, "git rev-parse origin/m6-baselines") == TARGET, "post-fetch origin")
    command(fetched, f"git merge-base --is-ancestor {PREVIOUS} {TARGET}")
    require(command(fetched, f"git rev-list --count {PREVIOUS}..{TARGET}") == "16", "remote commit distance")
    require(command(fetched, f"git diff --stat {PREVIOUS} {TARGET}"), "diff stat captured")
    require(git("rev-list", "--left-right", "--count", PREVIOUS + "..." + TARGET).split() == [b"0", b"16"], "local ancestry corroboration")
    transcript = evidence["merge_transcript"]
    require(transcript["exit_code"] == 0 and MERGE in transcript["output"] and
            "Updating 89ea43e..4bc78bb" in transcript["output"] and
            "Fast-forward" in transcript["output"], "merge evidence")
    require(command(post, "git rev-parse HEAD") == TARGET, "post-sync HEAD evidence")
    require(command(post, "git rev-parse origin/m6-baselines") == TARGET, "post-sync origin evidence")
    require(command(post, "git branch --show-current") == "m6-baselines", "post-sync branch")
    require(command(post, "git status --porcelain") == "", "post-sync clean evidence")
    require(command(post, "git rev-list --left-right --count HEAD...origin/m6-baselines").split() == ["0", "0"], "post-sync divergence evidence")
    require(command(post, "git reflog -1 --format=%H %gs") == TARGET + " merge origin/m6-baselines: Fast-forward", "reflog corroboration")
    runtime = "/home/student20261/workdir/GPAT_TransferBench_runtime"
    command(post, "test -d " + runtime)
    command(post, "test -d " + runtime + "/data/processed/faces_256")
    count_command = "bash -o pipefail -c find " + runtime + "/data/processed/faces_256 -type f -name '*.png' -printf '.' | wc -c"
    require(command(post, count_command) == "20615", "runtime metadata count evidence")
    require(audit["runtime"]["root_exists"] is True and audit["runtime"]["faces_exists"] is True, "runtime directories")
    spots = audit["content_hash_spot_checks"]
    require(set(spots) == SPOTS, "ten prescribed spot checks")
    remote_hashes = [r for r in post if r["command"].startswith("sha256sum ")]
    require(len(remote_hashes) == 1 and remote_hashes[0]["exit_code"] == 0, "hash command")
    require(evidence["local_hashes"] == remote_hashes[0]["stdout"], "raw hash equality")
    parsed = {line.split()[1]: line.split()[0] for line in remote_hashes[0]["stdout"].splitlines()}
    require(set(parsed) == SPOTS, "raw hash coverage")
    for rel, row in spots.items():
        require(sha(ROOT / rel) == row["laptop_sha256"] == row["gpu_sha256"] == parsed[rel], "content hash: " + rel)
        require(row["match"] is True, "hash match declaration")

    prefix = git("show", TARGET + ":" + LEDGER)
    current = (ROOT / LEDGER).read_bytes()
    require(len(prefix.splitlines()) == 96 and hashlib.sha256(prefix).hexdigest() == PREFIX_SHA, "committed prefix")
    require(current.startswith(prefix) and len(current.splitlines()) == 97, "byte-identical prefix and 97 rows")
    suffix = current[len(prefix):]
    require(len(suffix.splitlines()) == 1, "exactly one append")
    ledger = json.loads(suffix)
    for source in (audit["ledger"], ledger):
        require(source["committed_prefix_rows"] == 96 and source["final_rows"] == 97 and
                source["committed_prefix_sha256"] == PREFIX_SHA, "ledger metadata")
    require(ledger["classification"] == "M6D0C_GPU_REPOSITORY_SYNC", "ledger classification")
    require(set(ledger["file_sha256"]) == CREATED, "ledger artifact coverage")
    for rel, digest in ledger["file_sha256"].items():
        require(sha(ROOT / rel) == digest, "ledger artifact hash: " + rel)

    indexed = None
    if args.final:
        data = (ROOT / INDEX).read_bytes()
        require(data.count(b"\n") == data.count(b"\r\n") > 0 and b"\r" not in data.replace(b"\r\n", b""), "index CRLF")
        rows = list(csv.DictReader(io.StringIO(data.decode())))
        paths = [r["path"] for r in rows]
        require(len(paths) == len(set(paths)) == audit["artifact_index"]["expected_path_count"], "unique index path count")
        candidates = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().strip("\0").split("\0")
        expected_paths = {
            rel for rel in candidates if (ROOT / rel).is_file()
            and rel not in {INDEX, LEDGER} and Path(rel).name != ".gitkeep"
            and not {".git", "__pycache__", ".venv"}.intersection(Path(rel).parts)
            and not rel.startswith(("data/raw/", "data/processed/", "cache/"))
        }
        require(set(paths) == expected_paths and CREATED <= set(paths), "complete index coverage")
        for row in rows:
            path = ROOT / row["path"]
            require(path.stat().st_size == int(row["size_bytes"]), "index size: " + row["path"])
            require(sha(path) == row["sha256"], "index hash: " + row["path"])
        indexed = len(rows)
    print(json.dumps({
        "result": "GPU_REPOSITORY_SYNC_VALIDATION_PASS", "mode": "final" if args.final else "evidence",
        "post_sync_head": TARGET, "ancestry": "FAST_FORWARD", "commit_distance": 16,
        "content_hash_matches": len(spots), "faces_png_metadata_count": 20615,
        "committed_prefix_rows": 96, "final_rows": 97, "prefix_byte_identical": True,
        "committed_prefix_sha256": PREFIX_SHA, "index_paths_verified": indexed,
        "firewall": "PASS", "scientific_execution": False,
    }, indent=2))


if __name__ == "__main__":
    main()
