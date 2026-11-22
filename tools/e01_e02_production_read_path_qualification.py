#!/usr/bin/env python3
"""Qualify the hardened E01/E02 writer against the passing M8 smoke banks (real TRAIN faces, toy population).

    CUDA_VISIBLE_DEVICES= nice -n 19 ionice -c3 <gpat-m5 python> -B tools/e01_e02_production_read_path_qualification.py \
        --smoke-root <M8 smoke real_A root> --faces-root <runtime_root>/data/processed/faces_256 --out <new root>

The production gate hardening changed how faces are read (O_NOATIME plus an atime stat check in every mode) and what
the authorization record holds; it changed no generator, encoder or naming code. This tool regenerates the six smoke
toy banks (E01/E02 x 42/1337/2026, 186 frozen TRAIN pairs each) with the current code under SMOKE authorization, in a
new root outside the runtime root, behind the same TRAIN allowlist firewall. It then compares every manifest row with
the bank of the passing smoke run (source dd89301): pair, position, status, deterministic seed, recipe metadata, PNG
SHA256, pixel SHA256 and edit-map SHA256 must be identical. It also requires every face atime to be unchanged.
Writes <out>/read_path_qualification.json. Exit 0 only if every gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOY_MANIFEST_SHA256 = "50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690"   # M8 smoke record
COMPARED = ("position", "pair_id", "method_id", "experiment_seed", "config_sha256", "target_live_id",
            "target_face_sha256", "source_spoof_id", "source_face_sha256", "generation_status", "deterministic_pair_seed",
            "recipe_or_operator_metadata", "synthetic_output_path", "output_sha256", "output_pixel_sha256",
            "edit_map_path", "edit_map_sha256", "selected_operator_name", "operator_magnitude_level",
            "selected_asset_filename", "is_level_zero_noop", "n_eligible", "k_selected",
            "selected_block_indices_sorted_ascending", "max_abs_imag")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-root", required=True)
    ap.add_argument("--faces-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from methods.common import bank_writer as bw
    from methods.common.config import load_method_config
    from methods.common.runlog import git_commit, git_dirty
    from gpatbench.downstream import audit_hook

    smoke, faces, out = Path(a.smoke_root), Path(a.faces_root), Path(a.out)
    out.mkdir(parents=False, exist_ok=False)
    tm_path = smoke / "select" / "toy_manifest.json"
    gates = {"toy_manifest_sha256": hashlib.sha256(tm_path.read_bytes()).hexdigest() == TOY_MANIFEST_SHA256,
             "worktree_clean": not git_dirty()}
    man = json.loads(tm_path.read_text())
    positions = [p["position"] for p in man["toy_pairs"]]
    allow = [s["face_rel_path"] for s in man["pair_samples"]]
    live = bw.verify_locked_environment()
    lock = {"verified": True, "path": bw.ENV_LOCK_PATH, "live": live}
    atime_before = {r: os.stat(faces / r).st_atime_ns for r in allow}
    audit_hook.install(faces, allow, out / "firewall.json")
    auth = bw.smoke_authorization()
    banks = {}
    for m in bw.METHODS:
        cfg = load_method_config(m)
        for s in bw.PRODUCTION_SEEDS:
            inputs = bw.smoke_toy_inputs(faces.parents[2], cfg, positions)
            summ = bw.write_bank(m, s, inputs, auth, out / "banks", config=cfg, environment_lock=lock)
            new_rows = bw.read_manifest(out / "banks" / "runs" / "m6" / m / f"seed_{s}")
            ref_rows = bw.read_manifest(smoke / "banks" / "runs" / "m6" / m / f"seed_{s}")
            diffs = [(r["pair_id"], k) for r, q in zip(new_rows, ref_rows) for k in COMPARED if r.get(k) != q.get(k)]
            key = f"{m}.seed_{s}"
            gates[f"{key}.completed_no_failures"] = summ["completion_status"] == "completed" and summ["failure_count"] == 0
            gates[f"{key}.rows_186"] = len(new_rows) == len(ref_rows) == 186
            gates[f"{key}.identical_to_smoke_bank"] = not diffs and len(new_rows) == len(ref_rows)
            gates[f"{key}.o_noatime_reads"] = inputs.atime_checked_reads == len(inputs.opened) > 0
            banks[key] = {"rows": len(new_rows), "differences": diffs[:20], "difference_count": len(diffs),
                          "bank_manifest_sha256": summ["bank_manifest_sha256"],
                          "smoke_bank_manifest_sha256": (smoke / "banks" / "runs" / "m6" / m / f"seed_{s}" /
                                                         bw.BANK_MANIFEST_SHA).read_text().split()[0],
                          "o_noatime_reads": inputs.atime_checked_reads}
    fw = audit_hook.write_report()
    atime_after = {r: os.stat(faces / r).st_atime_ns for r in allow}
    changed = sorted(r for r in allow if atime_before[r] != atime_after[r])
    gates["firewall_clean"] = fw["violation_count"] == 0 and fw["listing_attempts"] == 0
    gates["face_atime_unchanged"] = not changed
    record = {"record": "E01_E02_PRODUCTION_READ_PATH_QUALIFICATION", "git_commit": git_commit(),
              "worktree_dirty": git_dirty(), "environment": live, "smoke_root": str(smoke), "faces_root": str(faces),
              "out": str(out), "compared_columns": list(COMPARED), "banks": banks, "firewall": fw,
              "faces_checked_for_atime": len(allow), "atime_changed_faces": changed[:20],
              "gates": gates, "all_pass": all(gates.values()),
              "git_status": subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                           text=True).stdout}
    (out / "read_path_qualification.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_pass": record["all_pass"], "failed": [k for k, v in gates.items() if not v]}))
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
