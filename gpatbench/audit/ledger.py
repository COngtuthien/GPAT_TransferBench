"""Append-only execution ledger for GPAT-TransferBench.

The ledger (outputs/audit/EXECUTION_LEDGER.jsonl) is never rewritten: entries are
only appended. Corrections are made by appending a new entry that references the
earlier one, never by editing history.

CLI:
    python3 -m gpatbench.audit.ledger append --milestone M0 --purpose "..." \
        --command "..." [--inputs a b] [--outputs c d] [--status OK] \
        [--exit-code 0] [--notes "..."] [--classification TECHNICAL_FAILURE]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = PROJECT_ROOT / "outputs" / "audit" / "EXECUTION_LEDGER.jsonl"

FAILURE_CLASSES = (
    "TECHNICAL_FAILURE",
    "DEPENDENCY_BLOCK",
    "SOURCE_GAP",
    "DATA_BLOCK",
    "SCIENTIFIC_BLOCK",
)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path = PROJECT_ROOT) -> tuple[str | None, bool | None]:
    """Return (HEAD commit or None, dirty flag or None). Read-only."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD"],
            capture_output=True, text=True, env=env,
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, env=env, check=True,
        ).stdout
        return head, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def append_entry(
    *,
    milestone: str,
    purpose: str,
    command: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    status: str = "OK",
    exit_code: int | None = 0,
    notes: str = "",
    classification: str | None = None,
    ledger_path: Path = LEDGER_PATH,
) -> dict:
    if classification is not None and classification not in FAILURE_CLASSES:
        raise ValueError(f"unknown classification {classification!r}")
    head, dirty = git_state()
    entry = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "milestone": milestone,
        "purpose": purpose,
        "command": command,
        "git_commit": head,
        "git_dirty": dirty,
        "input_artifacts": inputs or [],
        "output_artifacts": outputs or [],
        "status": status,
        "exit_code": exit_code,
        "classification": classification,
        "notes": notes,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:  # append-only
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _main() -> None:
    p = argparse.ArgumentParser(prog="gpatbench.audit.ledger")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("append")
    a.add_argument("--milestone", required=True)
    a.add_argument("--purpose", required=True)
    a.add_argument("--command", required=True)
    a.add_argument("--inputs", nargs="*", default=[])
    a.add_argument("--outputs", nargs="*", default=[])
    a.add_argument("--status", default="OK")
    a.add_argument("--exit-code", type=int, default=0)
    a.add_argument("--notes", default="")
    a.add_argument("--classification", default=None, choices=FAILURE_CLASSES)
    args = p.parse_args()
    e = append_entry(
        milestone=args.milestone, purpose=args.purpose, command=args.command,
        inputs=args.inputs, outputs=args.outputs, status=args.status,
        exit_code=args.exit_code, notes=args.notes, classification=args.classification,
    )
    print(json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    _main()
