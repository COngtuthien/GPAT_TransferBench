"""GPAT-TransferBench CLI (spec §24). Only the M1 `inventory` command is implemented so far."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from gpatbench.audit.ledger import PROJECT_ROOT, git_state, sha256_file


def _code_tree_sha256() -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted((PROJECT_ROOT / "gpatbench").rglob("*.py")):
        h.update(p.relative_to(PROJECT_ROOT).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def cmd_inventory(args) -> int:
    from gpatbench.data.inventory import load_config, run_inventory
    cfg = load_config(args.config)
    head, dirty = git_state()
    run = run_inventory(args.config, workers=args.workers)
    lock = PROJECT_ROOT / cfg["environment_lock"]
    run.update({"command": "python -m gpatbench.cli inventory --config " + args.config,
                "git_commit": head, "git_dirty": dirty, "code_tree_sha256": _code_tree_sha256(),
                "python": sys.version.split()[0], "platform": platform.platform(),
                "environment_lock": cfg["environment_lock"],
                "environment_lock_sha256": sha256_file(lock) if lock.exists() else None})
    path = PROJECT_ROOT / cfg["outputs"]["run_json"]
    path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(run["counts"]), "->", path.relative_to(PROJECT_ROOT))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gpatbench.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory", help="M1: inventory local datasets (read-only, CPU)")
    inv.add_argument("--config", required=True)
    inv.add_argument("--workers", type=int, default=None, help="process count (engineering only)")
    inv.set_defaults(func=cmd_inventory)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
