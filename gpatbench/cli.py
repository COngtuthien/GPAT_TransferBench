"""GPAT-TransferBench CLI (spec §24): `inventory` (M1), `split`/`audit-split` (M3), `build-pairs` (M4),
`train-probe` (M5 — pre-flight guard only; it refuses to train while the contract is unfrozen)."""
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


def cmd_build_pairs(args) -> int:
    """M4: build the authoritative common pair manifests (spec §6, App. A).

    This is the same code path as calling `gpatbench.pairs.execute.run` directly. There is no
    alternative builder, and `--audit-only` computes without writing so it cannot change membership.
    """
    from gpatbench.pairs import execute as E
    cfg = Path(args.config)
    if cfg.resolve() != (PROJECT_ROOT / "configs/frozen/pairs_v1.yaml").resolve():
        raise SystemExit(f"STOP: M4 runs only against the frozen pair config, got {args.config}")
    out = E.run(write=not args.audit_only)
    head, dirty = git_state()
    record = {"command": f"python -m gpatbench.cli build-pairs --config {args.config}",
              "git_commit": head, "git_dirty": dirty, "code_tree_sha256": _code_tree_sha256(),
              "python": sys.version.split()[0], "platform": platform.platform(),
              "pairs_config_sha256": out["pop"]["pairs_config_sha256"],
              "split_manifest_sha256": out["pop"]["split_manifest_sha256"],
              "rows": {k: len(v) for k, v in out["rows"].items()},
              "row_order": list(E.ROW_ORDER), "schema_signature": E.schema_signature(),
              "parquet_writer": dict(E.PARQUET_WRITER), "json_policy": dict(E.JSON_POLICY),
              "written": not args.audit_only, "sha256": out["sha256"]}
    print(json.dumps(record, indent=1))
    return 0


def cmd_train_probe(args) -> int:
    """M5: ArtifactProbeNet (spec §12.1) — PRE-FLIGHT GUARD, not a trainer.

    The frozen contract does not exist yet: eight execution-affecting decisions are still open
    (see configs/proposed/artifact_probe.proposed.yaml). This subcommand therefore exists to hold
    the spec's CLI surface and to refuse, so nothing can train by accident and no second training
    implementation is created behind it.
    """
    import yaml
    cfg = Path(args.config)
    frozen = PROJECT_ROOT / "configs/frozen/artifact_probe.yaml"
    if not cfg.is_file():
        raise SystemExit(
            f"STOP: {args.config} does not exist. M5 is NOT_STARTED and the ArtifactProbeNet "
            "contract is not frozen; see configs/proposed/artifact_probe.proposed.yaml.")
    if cfg.resolve() != frozen.resolve():
        raise SystemExit(f"STOP: M5 runs only against {frozen.relative_to(PROJECT_ROOT)}, "
                         f"got {args.config}")
    doc = yaml.safe_load(cfg.read_text())
    blockers = list(doc.get("blocking_decisions") or [])
    if doc.get("status") != "FROZEN" or blockers:
        raise SystemExit(
            f"STOP: the ArtifactProbeNet contract is not frozen (status={doc.get('status')!r}, "
            f"open decisions={blockers}). Training is refused.")
    raise SystemExit(
        "STOP: the M5 trainer is not implemented in this pre-flight pass. The contract is frozen, "
        "so implementing it is the next authorized step.")


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
    bp = sub.add_parser("build-pairs", help="M4: authoritative common source->target pair manifests")
    bp.add_argument("--config", required=True)
    bp.add_argument("--audit-only", action="store_true",
                    help="compute and report hashes without writing (membership is unchanged)")
    bp.set_defaults(func=cmd_build_pairs)
    tp = sub.add_parser("train-probe",
                        help="M5: ArtifactProbeNet (pre-flight guard; refuses to train)")
    tp.add_argument("--config", required=True)
    tp.set_defaults(func=cmd_train_probe)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
