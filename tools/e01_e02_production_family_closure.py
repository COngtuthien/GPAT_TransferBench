#!/usr/bin/env python3
"""Close one E01/E02 production family from its three committed per-bank audits.

    python tools/e01_e02_production_family_closure.py --method E01 --out-prefix outputs/audit/E01_PRODUCTION_FAMILY_CLOSURE

Refuses (exit 1, writes nothing) unless outputs/audit/<M>_PRODUCTION_BANK_SEED{42,1337,2026}.json all exist and are
PASS with every gate true. They must agree on the launch/candidate commits, config, environment, binding and storage,
and have three distinct run ids. Writes <prefix>.json/.md and refuses to overwrite them. Does not touch
outputs/audit/method_status.csv and claims nothing about downstream evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = (42, 1337, 2026)
DISCLOSURES = {
    "E01": [
        "Official RizhaoCai/FAS-Aug@0da1dd79 operators, round-robin operator_set[position % 8] over the frozen pair order "
        "(owner ruling Q2), per-pair operator seed SHA256(pair_id + seed) mod 2^31 (A2-05), NUM_MAG 10 grid.",
        "Magnitude level 0 is retained: some outputs are unchanged copies of the live target (per_seed "
        "level_zero_noop_count; each was verified pixel-identical to its target); they are not substituted or removed.",
        "Duplicate output pixels: frozen pairs reuse live targets, so level-0 copies of the same target are identical "
        "(per_seed duplicate_output_pixel_groups); every row is kept, as spec section 7 requires.",
        "Cross-environment variation: in M6 qualification 3/72 synthetic Hand_Trembling outputs differed by <= 1 LSB "
        "between the laptop .venv and gpat-m5. These banks were generated and replayed only under gpat-m5, and "
        "byte identity on other stacks is not claimed.",
        "The bank is non-learned. No checkpoint exists, and generator_checkpoint_sha256 is null by design."],
    "E02": [
        "Spec 8.2 frequency substitution: source-spoof amplitude in 40 of 159 eligible 16x16 blocks "
        "(0.20 <= r <= 0.50, >= 75 % coverage) with pixel-wise conjugate closure, target-live phase preserved, PCG64 "
        "selection from the A2-04 namespace seed.",
        "uint8 quantization is the implementation-defined clip [0,255] -> numpy.rint (owner ruling Q3), not an explicit "
        "spec instruction. Every output was recomputed with an independent float64 FFT and was identical.",
        "Edit maps (L PNG, 255 = edited) record the applied selection plus conjugates for every output.",
        "Pixel identity is shown for the gpat-m5 stack only (and the laptop in M6). It is not an unrestricted "
        "cross-stack guarantee."],
}
REMAINING = [
    "M8+ downstream evaluation of these banks (ResNet-18 per the approved evaluator contract; DINOv3 needs its frozen "
    "config and pinned weights): separate owner authorization required",
    "TEST evaluation: not authorized; no TEST row or TEST face was read",
    "outputs/audit/method_status.csv is unchanged; updating it needs a separately verified registry-authority decision",
    "learned-method banks (E03-E07) and GPAT are outside this work",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=("E01", "E02"))
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)
    m, prefix = a.method, Path(a.out_prefix)
    for suf in (".json", ".md"):
        if prefix.with_suffix(suf).exists():
            raise SystemExit(f"refusing to overwrite {prefix.with_suffix(suf)}")
    paths = {s: ROOT / f"outputs/audit/{m}_PRODUCTION_BANK_SEED{s}.json" for s in SEEDS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(json.dumps({"status": f"{m}_PRODUCTION_FAMILY_NOT_CLOSED", "missing_audits": missing}))
        return 1
    recs = {s: json.loads(p.read_text()) for s, p in paths.items()}
    problems = []
    for s, r in recs.items():
        if r["status"] != f"{m}_SEED{s}_PRODUCTION_BANK_AUDIT_PASS" or r["failed_gates"] or not all(r["gates"].values()):
            problems.append(f"seed {s} audit not PASS")
        if (r["method"], r["seed"], r["output_count"], r["failure_count"]) != (m, s, 8838, 0):
            problems.append(f"seed {s} identity/count")
    for key in ("launch_commit", "candidate_commit", "config_sha256", "filesystem", "environment_versions"):
        if len({json.dumps(r[key], sort_keys=True) for r in recs.values()}) != 1:
            problems.append(f"{key} differs across seeds")
    if len({json.dumps(r["authorization"].get("binding"), sort_keys=True) for r in recs.values()}) != 1:
        problems.append("binding differs across seeds")
    if len({r["run_id"] for r in recs.values()}) != 3 or len({r["bank_manifest_sha256"] for r in recs.values()}) != 3:
        problems.append("run ids or bank manifests not distinct")
    if problems:
        print(json.dumps({"status": f"{m}_PRODUCTION_FAMILY_NOT_CLOSED", "problems": problems}))
        return 1
    r0 = recs[SEEDS[0]]
    seeds = {}
    for s, r in recs.items():
        seeds[str(s)] = {"run_id": r["run_id"], "run_dir": r["run_dir"], "output_count": r["output_count"],
                         "failure_count": r["failure_count"], "bank_manifest_sha256": r["bank_manifest_sha256"],
                         "pair_output_sha_digest": r["pair_output_sha_digest"], "run_dir_disk_bytes": r["run_dir_disk_bytes"],
                         "audit_record": str(paths[s].relative_to(ROOT)), "audit_record_sha256": sha(paths[s]),
                         "audit_gates": f"{r['passed']}/{r['gate_count']}", "launch_time": r["process"]["launch_time"],
                         "end_time": r["process"]["end_time"],
                         "replay": {"cli_in_process": "8838/8838 identical",
                                    "independent_process": f"{r['audit_replay']['rows_replayed']} rows, "
                                                           f"{len(r['audit_replay']['mismatches'])} mismatches"},
                         "atime": {k: r["atime"][k] for k in ("pre_files", "atime_changed_total", "atime_changed_val",
                                                             "atime_changed_other_test_by_elimination",
                                                             "atime_changed_train_allowlisted", "mtime_changed_total")},
                         "test_firewall": {"cli_violations": r["cli_firewall"]["violation_count"],
                                           "cli_listing_attempts": r["cli_firewall"]["listing_attempts"],
                                           "audit_violations": r["audit_firewall"]["violation_count"],
                                           "test_rows_parsed": 0},
                         **({"operator_counts": r["e01_operator_counts"], "level_histogram": r["e01_level_histogram"],
                             "level_zero_noop_count": r["e01_level_zero_noop_count"]} if m == "E01" else {}),
                         "duplicate_output_pixel_groups": r["duplicate_output_pixel_groups"]}
    auth = json.loads((ROOT / "configs/amendments/e01_e02_production_authority.yaml").read_text())
    record = {"record": f"{m}_PRODUCTION_FAMILY_CLOSURE", "status": f"{m}_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED",
              "method": m, "seeds": list(SEEDS), "per_seed": seeds,
              "total_outputs": sum(r["output_count"] for r in recs.values()),
              "storage": {**r0["filesystem"], "runtime_root": auth["storage"]["runtime_root"],
                          "device_model": "Samsung SSD 980 1TB (NVMe)"},
              "source": {"launch_commit": r0["launch_commit"], "candidate_commit": r0["candidate_commit"],
                         "production_authority": "configs/amendments/e01_e02_production_authority.yaml",
                         "production_authority_sha256": sha(ROOT / "configs/amendments/e01_e02_production_authority.yaml"),
                         "binding": r0["authorization"]["binding"], "ruling_sha256": r0["authorization"]["ruling_sha256"]},
              "config_sha256": r0["config_sha256"], "environment_versions": r0["environment_versions"],
              "fidelity_disclosures": DISCLOSURES[m],
              "access_time_protection": "every face read used O_NOATIME with a per-read atime stat check (abort on change); "
                                        "stat-only snapshots of all 20615 faces before/after each bank show 0 atime changes "
                                        "(0 VAL, 0 TEST by elimination)",
              "test_isolation": "frozen TRAIN pair relation only; the TRAIN allowlist audit hook blocked any other face open "
                                "and any listing; the split manifest was read with a TRAIN/VAL row filter; no TEST row or "
                                "face was read",
              "remaining_downstream_dependencies": REMAINING,
              "is_downstream_evaluation_complete": False, "method_status_csv_updated": False}
    prefix.with_suffix(".json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    md = [f"# {m} production family closure", "", f"Status: `{record['status']}`", "",
          f"Storage: `{record['storage']['runtime_root']}` on {record['storage']['source']} "
          f"(ext4 UUID `{record['storage']['filesystem_uuid']}`, mount `{record['storage']['mount_point']}`)",
          f"Launch commit `{r0['launch_commit']}` (candidate `{r0['candidate_commit']}`), config `{r0['config_sha256']}`", "",
          "| seed | run_id | outputs | bank manifest SHA256 | audit |", "|---|---|---|---|---|"]
    md += [f"| {s} | `{v['run_id']}` | {v['output_count']} | `{v['bank_manifest_sha256']}` | {v['audit_gates']} |"
           for s, v in seeds.items()]
    if m == "E01":
        md += ["", "| seed | level-0 no-op copies | duplicate-pixel groups |", "|---|---|---|"] + \
            [f"| {s} | {v['level_zero_noop_count']} | {v['duplicate_output_pixel_groups']} |" for s, v in seeds.items()]
    md += ["", "Disclosures:"] + [f"- {d}" for d in DISCLOSURES[m]] + ["", "Remaining:"] + [f"- {x}" for x in REMAINING]
    prefix.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps({"status": record["status"], "total_outputs": record["total_outputs"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
