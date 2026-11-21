#!/usr/bin/env python3
"""E01/E02 production bank CLI (fail-closed).

    python tools/e01_e02_bank.py --method E01 --seed 42 --runtime-root <runtime_root> [--check-only]

Order of gates, each refusing before any image is read:
  1. owner ruling (`configs/amendments/e01_e02_bank_milestone_ruling.yaml`, approved, equal to the approved decisions);
     absent/deviating -> exit 3 with E01_E02_BLOCKED_BY_MILESTONE_RULING; approved but no bound, passing M8
     deterministic smoke-test record -> exit 3 with E01_E02_BLOCKED_BY_M8_SMOKE_GATE;
  2. clean git worktree (the run id binds the commit);
  3. the M6D1-locked gpat-m5 interpreter (A2-03 NumPy pin);
  4. frozen TRAIN population + M2 face hashes; faces root = <runtime_root>/data/processed/faces_256;
  5. no existing run directory (RunContext refuses; every file is O_EXCL).
Then the bank is written, file-verified and fully replayed. There is no resume, limit, subset or TEST option.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402
from methods.common.config import load_method_config  # noqa: E402
from methods.common.runlog import git_dirty  # noqa: E402

EXIT_BLOCKED = 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=bw.METHODS)
    ap.add_argument("--seed", required=True, type=int, choices=(42, 1337, 2026))
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--check-only", action="store_true", help="run every gate, then stop before writing")
    a = ap.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        auth = bw.load_production_authorization(a.method)
    except bw.BankBlocked as exc:
        print(json.dumps({"status": exc.status, "method": a.method, "seed": a.seed, "reason": str(exc)}))
        return EXIT_BLOCKED
    if git_dirty():
        print(json.dumps({"status": "REFUSED", "reason": "git worktree is dirty"}))
        return 2
    live = bw.verify_locked_environment()
    config = load_method_config(a.method)
    runtime_root = Path(a.runtime_root)
    inputs = bw.production_inputs(runtime_root, config)
    run_dir = runtime_root / "runs" / "m6" / a.method / f"seed_{a.seed}"
    if run_dir.exists():
        print(json.dumps({"status": "REFUSED", "reason": f"run directory exists: {run_dir}"}))
        return 2
    gates = {"status": "GATES_PASS", "method": a.method, "seed": a.seed, "authorization": auth.record(),
             "environment": live, "inputs": inputs.provenance, "run_dir": str(run_dir)}
    if a.check_only:
        print(json.dumps(gates, indent=2))
        return 0
    lock = {"verified": True, "path": bw.ENV_LOCK_PATH, "live": live}
    step = max(1, len(inputs.pairs) // 100)
    summary = bw.write_bank(a.method, a.seed, inputs, auth, runtime_root, config=config, environment_lock=lock,
                            progress=lambda i, n: print(f"{a.method} s{a.seed} {i}/{n}", flush=True)
                            if i % step == 0 or i == n else None)
    files = bw.verify_bank_files(run_dir, a.method)
    replay = bw.replay_bank(run_dir, a.method, a.seed, inputs, auth, config=config)
    print(json.dumps({"status": summary["completion_status"], "run_id": summary["run_id"],
                      "output_count": summary["output_count"], "failure_count": summary["failure_count"],
                      "bank_manifest_sha256": summary["bank_manifest_sha256"],
                      "file_verification": files["ok"], "replay": replay["ok"]}, indent=2))
    return 0 if files["ok"] and replay["ok"] and summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
