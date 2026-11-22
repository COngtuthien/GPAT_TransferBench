#!/usr/bin/env python3
"""E01/E02 production bank CLI (fail-closed).

    CUDA_VISIBLE_DEVICES= python -B tools/e01_e02_bank.py --method E01 --seed 42 --runtime-root <runtime_root> [--check-only]

Order of gates, each refusing (exit 3, JSON status) before any image is read:
  1. governance (bank_writer.load_production_authorization): the owner ruling, unchanged and approved
     (E01_E02_BLOCKED_BY_MILESTONE_RULING), then the additive M8 smoke binding with the strictly validated smoke record
     and every pinned identity (E01_E02_BLOCKED_BY_M8_SMOKE_GATE), smoke-source ancestry and seed scope;
  2. launch (bank_writer.verify_production_launch): the owner-approved production authority commit is an ancestor of
     this clean checkout and every authority file is byte-identical to it (E01_E02_BLOCKED_BY_PRODUCTION_AUTHORITY);
     approved host, locked gpat-m5 interpreter, CUDA hidden, approved runtime root on the approved filesystem UUID with
     the free-space floor, no existing run directory (E01_E02_BLOCKED_BY_PRODUCTION_PREFLIGHT);
  3. frozen TRAIN population + M2 face hashes; faces root = <runtime_root>/data/processed/faces_256;
  4. a file-open firewall (audit hook) over the faces root that admits only the referenced TRAIN faces and forbids
     listing it; then one O_NOATIME probe read of the first TRAIN face (atime stat-checked unchanged).
Then the bank is written, file-verified and fully replayed. Every face read is O_NOATIME. There is no resume, limit,
subset or TEST option. The last stdout line is the JSON evidence (gates, firewall report, atime ledger, results).
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

EXIT_BLOCKED = 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=bw.METHODS)
    ap.add_argument("--seed", required=True, type=int, choices=bw.PRODUCTION_SEEDS)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--check-only", action="store_true", help="run every gate and the probe read, then stop")
    a = ap.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    runtime_root = Path(a.runtime_root)
    try:
        auth = bw.load_production_authorization(a.method, seed=a.seed)
        launch = bw.verify_production_launch(a.method, a.seed, auth, runtime_root)
    except bw.BankBlocked as exc:
        print(json.dumps({"status": exc.status, "method": a.method, "seed": a.seed, "reason": str(exc)}))
        return EXIT_BLOCKED
    config = load_method_config(a.method)
    inputs = bw.production_inputs(runtime_root, config)

    from gpatbench.downstream import audit_hook
    allow = sorted(f"{inputs.sample_dataset[s]}/{s}.png" for s in inputs.face_sha256)
    audit_hook.install(inputs.faces_root, allow, os.devnull)
    probe_id = inputs.pairs[0].target_live_id
    _, probe_sha = inputs.read(probe_id)
    gates = {"status": "GATES_PASS", "method": a.method, "seed": a.seed, "authorization": auth.record(),
             "launch": launch, "inputs": inputs.provenance, "firewall_allowlist": len(allow),
             "noatime_probe": {"sample_id": probe_id, "sha256": probe_sha, "o_noatime": True, "atime_unchanged": True}}
    if a.check_only:
        gates["firewall"] = audit_hook.report()
        print(json.dumps(gates, indent=2))
        return 0
    lock = {"verified": True, "path": bw.ENV_LOCK_PATH, "live": launch["environment"]}
    step = max(1, len(inputs.pairs) // 100)
    summary = bw.write_bank(a.method, a.seed, inputs, auth, runtime_root, config=config, environment_lock=lock,
                            progress=lambda i, n: print(f"{a.method} s{a.seed} {i}/{n}", flush=True)
                            if i % step == 0 or i == n else None)
    run_dir = Path(launch["run_dir"])
    files = bw.verify_bank_files(run_dir, a.method)
    replay = bw.replay_bank(run_dir, a.method, a.seed, inputs, auth, config=config)
    fw = audit_hook.report()
    result = {"status": summary["completion_status"], "run_id": summary["run_id"], "run_dir": str(run_dir),
              "output_count": summary["output_count"], "failure_count": summary["failure_count"],
              "bank_manifest_sha256": summary["bank_manifest_sha256"], "file_verification": files,
              "replay": replay, "firewall": fw,
              "atime_ledger": {"o_noatime_reads": inputs.atime_checked_reads, "atime_change_policy": "abort_run",
                               "distinct_faces_opened": len(set(inputs.opened))},
              "gates": gates}
    print(json.dumps(result, sort_keys=True))
    ok = files["ok"] and replay["ok"] and summary["failure_count"] == 0 and fw["violation_count"] == 0 \
        and fw["listing_attempts"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
