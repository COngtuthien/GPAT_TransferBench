"""Append a detailed row to outputs/audit/EXECUTION_LEDGER.jsonl (append-only).

Superset of gpatbench.audit.ledger (same base fields) plus the owner-required provenance fields:
config path/sha256, environment lock/sha256, input/output artifacts WITH sha256, dataset roots,
seed, workers, counts, retry lineage, failure classification. gpatbench/ is intentionally not
modified (its code-tree hash is recorded in the M1 inventory run record).

Usage:
  python3 tools/ledger_rich.py --milestone M1 --purpose "..." --command "..." \
      [--config configs/frozen/x.yaml] [--env-lock environments/x.lock.txt] \
      [--inputs a b] [--outputs c d] [--dataset-roots r1 r2] [--seed 20260814] [--workers 8] \
      [--counts key=value ...] [--status OK] [--exit-code 0] [--classification ...] \
      [--retry-of <timestamp_utc of earlier row>] [--decision "..."] [--notes "..."]
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.audit.ledger import FAILURE_CLASSES, LEDGER_PATH, git_state  # noqa: E402


def _sha(p: str) -> str | None:
    path = (ROOT / p) if not os.path.isabs(p) else Path(p)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _arts(paths):
    return [{"path": p, "sha256": _sha(p)} for p in (paths or [])]


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--milestone", required=True)
    a.add_argument("--purpose", required=True)
    a.add_argument("--command", required=True)
    a.add_argument("--config")
    a.add_argument("--env-lock")
    a.add_argument("--inputs", nargs="*", default=[])
    a.add_argument("--outputs", nargs="*", default=[])
    a.add_argument("--dataset-roots", nargs="*", default=[])
    a.add_argument("--seed", type=int)
    a.add_argument("--workers", type=int)
    a.add_argument("--counts", nargs="*", default=[])
    a.add_argument("--status", default="OK")
    a.add_argument("--exit-code", type=int, default=0)
    a.add_argument("--classification", choices=FAILURE_CLASSES)
    a.add_argument("--retry-of")
    a.add_argument("--decision", default="")
    a.add_argument("--notes", default="")
    x = a.parse_args()
    head, dirty = git_state()
    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(), "user": getpass.getuser(), "cwd": os.getcwd(),
        "milestone": x.milestone, "purpose": x.purpose, "command": x.command,
        "git_commit": head, "git_dirty": dirty,
        "input_artifacts": x.inputs, "output_artifacts": x.outputs,
        "status": x.status, "exit_code": x.exit_code, "classification": x.classification, "notes": x.notes,
        "ledger_schema": "rich-v1",
        "config": {"path": x.config, "sha256": _sha(x.config)} if x.config else None,
        "environment_lock": {"path": x.env_lock, "sha256": _sha(x.env_lock)} if x.env_lock else None,
        "input_artifacts_sha256": _arts(x.inputs), "output_artifacts_sha256": _arts(x.outputs),
        "dataset_roots": x.dataset_roots, "seed": x.seed, "workers": x.workers,
        "counts": dict(kv.split("=", 1) for kv in x.counts),
        "retry_of": x.retry_of, "scientific_decision": x.decision,
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(row["timestamp_utc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
