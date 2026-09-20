"""Record the pinned third-party source provenance for the Track-A adaptation audits.

    <m2 venv>/bin/python tools/m4_pin_third_party_sources.py

Reads the local pinned checkouts under third_party/source_cache/ (git-ignored, never committed) and
writes third_party/source_pins.json: repository URL, exact commit, commit tree, commit date, and the
sha256 + git blob id of every file the adaptation analysis actually cites. Source only -- no model
weights are fetched, and any binary weight that arrived with a checkout is removed and recorded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E  # noqa: E402

CACHE = ROOT / "third_party/source_cache"
OUT = ROOT / "third_party/source_pins.json"

PINS = {
    "dsdg": {
        "method_ids": ["E06a", "E06b"],
        "repository": "https://github.com/JDAI-CV/FaceX-Zoo",
        "commit": "16b793a7564a4b9308cf94e62bdb2ffacb3a725a",
        "checkout": "facexzoo",
        "relevant_path": "addition_module/DSDG",
        "checkout_mode": "sparse (addition_module/DSDG only), blob-filtered, depth 1",
        "cited_files": [
            "addition_module/DSDG/train_generator.py",
            "addition_module/DSDG/data/generation_dataset.py",
            "addition_module/DSDG/data/make_train_list.py",
            "addition_module/DSDG/networks/__init__.py",
            "addition_module/DSDG/networks/generator.py",
            "addition_module/DSDG/misc/util.py",
        ],
        "binary_weights_removed": ["addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl"],
    },
    "difffas": {
        "method_ids": ["E07a", "E07b"],
        "repository": "https://github.com/murphytju/DiffFAS",
        "commit": "23f40519ec25a833ebc06842aa6fbab74fad4d15",
        "checkout": "difffas",
        "relevant_path": ".",
        "checkout_mode": "full tree, depth 1",
        "cited_files": [
            "FAS_dataset.py",
            "FAS_train.py",
            "FAS_sample.py",
            "diffusion.py",
            "models/unet_autoenc.py",
        ],
        "binary_weights_removed": [],
    },
}

WEIGHT_SUFFIXES = (".pt", ".pth", ".pkl", ".ckpt", ".onnx", ".safetensors", ".bin", ".h5")


def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    out = {"purpose": "Track-A adaptation source audit (Amendment A1); source only, no weights",
           "fetched_utc": "2026-09-20",
           "storage": "third_party/source_cache/ (git-ignored; see .gitignore)",
           "sources": {}}
    bad = []
    for key, spec in PINS.items():
        repo = CACHE / spec["checkout"]
        if not repo.is_dir():
            bad.append(f"{key}: checkout missing at {repo}")
            continue
        head = git(repo, "rev-parse", "HEAD")
        url = git(repo, "remote", "get-url", "origin")
        if head != spec["commit"]:
            bad.append(f"{key}: HEAD {head} != pinned {spec['commit']}")
        if url.rstrip("/").removesuffix(".git") != spec["repository"].rstrip("/"):
            bad.append(f"{key}: remote {url} != {spec['repository']}")
        stray = [p.relative_to(repo).as_posix() for p in repo.rglob("*")
                 if p.is_file() and ".git" not in p.parts and p.suffix in WEIGHT_SUFFIXES]
        files = {}
        for rel in spec["cited_files"]:
            path = repo / (spec["relevant_path"] + "/" + rel if spec["relevant_path"] != "."
                           and not rel.startswith(spec["relevant_path"]) else rel)
            if not path.is_file():
                bad.append(f"{key}: cited file missing {rel}")
                continue
            data = path.read_bytes()
            files[rel] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob": git(repo, "hash-object", str(path)),
                "size_bytes": len(data),
                "lines": data.decode("utf-8", "replace").count("\n") + 1,
            }
        out["sources"][key] = {
            "method_ids": spec["method_ids"],
            "repository": spec["repository"],
            "pinned_commit": spec["commit"],
            "commit_verified_from_remote": url,
            "commit_tree": git(repo, "log", "-1", "--format=%T"),
            "commit_date_utc": git(repo, "log", "-1", "--format=%cI"),
            "commit_subject": git(repo, "log", "-1", "--format=%s"),
            "relevant_path": spec["relevant_path"],
            "checkout_mode": spec["checkout_mode"],
            "cited_files": files,
            "binary_weights_removed_after_checkout": spec["binary_weights_removed"],
            "binary_weight_files_present_now": stray,
            "model_weights_downloaded": False,
        }
    out["verification_failures"] = bad
    out["status"] = "PASS" if not bad else "FAIL"
    OUT.write_bytes(E.canonical_json_bytes(out))
    summary = {"status": out["status"], "failures": bad}
    summary.update({k: v["pinned_commit"] for k, v in out["sources"].items()})
    print(json.dumps(summary, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
