#!/usr/bin/env python3
"""Write the M8 toy-subset feasibility plan (metadata only; no image, no TEST row, no smoke execution).

    python tools/m8_toy_plan.py --out outputs/audit/M8_TOY_SUBSET_FEASIBILITY.json

Refuses to overwrite. Exit 0 only if the plan has no shortfall, full batches are satisfiable, VAL select and holdout
each hold both classes per dataset and are disjoint; otherwise exit 1 (the record is still written as evidence).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import toy_subset  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 2
    plan = toy_subset.plan(ROOT)
    f = plan["feasibility"]
    ok = (not plan["shortfalls"] and f["full_batch_composition_satisfiable"]
          and all(f["val_both_classes_per_dataset"].values()) and all(f["select_holdout_disjoint"].values()))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"]).decode().strip())
    record = {"record": "M8_TOY_SUBSET_FEASIBILITY",
              "status": "TOY_SUBSET_FEASIBLE" if ok else "TOY_SUBSET_INFEASIBLE",
              "scope": "planning artifact from TRAIN/VAL metadata; authorizes no smoke run; not SMOKE evidence",
              "source_commit": commit, "worktree_dirty_at_run": dirty,
              "command": "python tools/m8_toy_plan.py --out " + str(a.out), **plan}
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "feasibility": f}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
