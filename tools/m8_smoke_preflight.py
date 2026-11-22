#!/usr/bin/env python3
"""M8 smoke real-data PREFLIGHT (no smoke stage is executed).

    python -B tools/m8_smoke_preflight.py --faces-root <production faces_256> --smoke-parent <dir> --out <new json>

Re-derives the approved toy manifest from TRAIN/VAL metadata, then reads every allowlisted face (toy roles plus the
toy pairs' targets and sources) with O_NOATIME under the audit-hook firewall and checks its bytes against the M2 SHA256.
Records disk, memory and load. Exit 0 only if every face is present and hash-verified, no atime changed, the plan matches
the approved feasibility record and resources are adequate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces-root", required=True)
    ap.add_argument("--smoke-parent", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    from gpatbench.downstream import audit_hook, toy_manifest, toy_subset
    from gpatbench.downstream.faceio import FaceReader
    man = toy_manifest.build_real(ROOT, a.faces_root)
    rec = json.loads((ROOT / "outputs/audit/M8_TOY_SUBSET_FEASIBILITY.json").read_text())
    plan = toy_subset.plan(ROOT)
    plan_ok = all(json.loads(json.dumps(plan[k])) == rec[k] for k in ("roles", "pooled", "toy_pairs", "feasibility"))
    audit_hook.install(a.faces_root, man["allowlist"], out.with_suffix(".firewall.json"))
    by_path = {}
    for f in [f for v in man["roles"].values() for f in v] + man["pair_samples"]:
        by_path.setdefault(f["face_rel_path"], f["face_sha256"])
    reader = FaceReader(a.faces_root)
    errors = []
    for rel, digest in sorted(by_path.items()):
        try:
            reader.read(rel, digest)
        except Exception as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")
    fw = audit_hook.write_report()
    du = shutil.disk_usage(a.smoke_parent)
    mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l}
    res = {"disk_free_gib": round(du.free / 2 ** 30, 1), "mem_available_gib": round(mem["MemAvailable"] / 2 ** 20, 1),
           "loadavg": os.getloadavg(), "cpus": os.cpu_count()}
    gates = {"plan_matches_feasibility_record": plan_ok,
             "toy_pairs_186": len(man["toy_pairs"]) == 186,
             "all_faces_present_and_hash_verified": not errors and len(reader.ledger) == len(by_path),
             "atime_unchanged": reader.summary()["atime_changed_files"] == 0,
             "firewall_clean": fw["violation_count"] == 0 and fw["listing_attempts"] == 0,
             "test_rows_read_zero": man["test_rows_read"] == 0,
             "disk_free_ge_10_gib": res["disk_free_gib"] >= 10, "mem_available_ge_6_gib": res["mem_available_gib"] >= 6}
    doc = {"record": "M8_SMOKE_REAL_PREFLIGHT", "gates": gates, "ok": all(gates.values()), "faces_checked": len(by_path),
           "frames": {k: len(v) for k, v in man["roles"].items()}, "pair_samples": len(man["pair_samples"]),
           "errors": errors[:50], "resources": res, "firewall": {k: fw[k] for k in ("violation_count", "listing_attempts",
                                                                                     "distinct_files_opened")},
           "toy_manifest_sha256": toy_manifest.canonical_sha256(man)}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"ok": doc["ok"], "gates": gates, "faces_checked": len(by_path), "resources": res}, indent=1))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
