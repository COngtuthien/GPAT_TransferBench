"""GPAT-TransferBench CLI (spec §24): `inventory` (M1), `split` and `audit-split` (M3)."""
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


def cmd_split(args) -> int:
    """M3: run the frozen Q-01 allocator and write the authoritative split manifest.

    This is the same code path as calling `gpatbench.split.execute.run_split` directly; there is no
    alternative execution route that bypasses the frozen allocator.
    """
    from gpatbench.split.execute import run_split
    cfg = Path(args.config)
    if cfg.resolve() != (PROJECT_ROOT / "configs/frozen/split_v1.yaml").resolve():
        raise SystemExit(f"STOP: M3 runs only against the frozen split config, got {args.config}")
    out = run_split(PROJECT_ROOT, write=not args.dry_run)
    head, dirty = git_state()
    record = {"command": f"python -m gpatbench.cli split --config {args.config}",
              "git_commit": head, "git_dirty": dirty, "code_tree_sha256": _code_tree_sha256(),
              "python": sys.version.split()[0], "platform": platform.platform(),
              "config_sha256": out["config_sha256"], "rows": out["rows"],
              "objectives": out["objectives"], "row_order": out["row_order"],
              "parquet_writer": out["parquet_writer"]}
    if not args.dry_run:
        record.update(manifest=out["manifest_path"], manifest_sha256=out["manifest_sha256"],
                      group_manifest=out["group_manifest_path"],
                      group_manifest_sha256=out["group_manifest_sha256"])
    print(json.dumps({k: record[k] for k in record if k != "objectives"}, indent=1))
    return 0


def cmd_audit_split(args) -> int:
    """M3: leakage and distribution audit of a written split manifest."""
    from gpatbench.split.execute import audit_split
    rep = audit_split(PROJECT_ROOT, Path(args.manifest))
    print(json.dumps({"manifest": rep["manifest"], "rows": rep["rows"], "ok": rep["ok"],
                      "checks": rep["checks"]}, indent=1, default=str))
    return 0 if rep["ok"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gpatbench.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory", help="M1: inventory local datasets (read-only, CPU)")
    inv.add_argument("--config", required=True)
    inv.add_argument("--workers", type=int, default=None, help="process count (engineering only)")
    inv.set_defaults(func=cmd_inventory)
    sp = sub.add_parser("split", help="M3: deterministic group-disjoint TRAIN/VAL/TEST split")
    sp.add_argument("--config", required=True)
    sp.add_argument("--dry-run", action="store_true", help="compute without writing the manifest")
    sp.set_defaults(func=cmd_split)
    au = sub.add_parser("audit-split", help="M3: leakage audit of a split manifest")
    au.add_argument("--manifest", required=True)
    au.set_defaults(func=cmd_audit_split)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
