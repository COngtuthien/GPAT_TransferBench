"""M3 audit artifacts: counts, leakage, distributions, failed-sample accounting, determinism.

    <m2 venv>/bin/python tools/m3_split_reports.py [--reruns 2]

Reads the authoritative manifest and the frozen metadata; never modifies the split. The determinism
check re-executes the whole allocator in **fresh interpreters** (canonical order, shuffled metadata
order, and a different PYTHONHASHSEED) into clean temporary directories and compares the resulting
Parquet bytes with the authoritative manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.split import allocator as A  # noqa: E402
from gpatbench.split import execute as E  # noqa: E402

AUDIT = ROOT / "outputs/audit"

RERUN_CODE = """
import sys, json
sys.path.insert(0, {root!r})
from pathlib import Path
from gpatbench.split import execute as E
out = E.run_split(Path({root!r}), write=True,
                  manifest_path=Path({tmp!r}) / "split_v1.parquet",
                  group_manifest_path=Path({tmp!r}) / "split_groups_v1.parquet",
                  shuffle_seed={seed!r})
print(json.dumps({{"manifest_sha256": out["manifest_sha256"],
                   "group_manifest_sha256": out["group_manifest_sha256"],
                   "rows": out["rows"], "objectives": out["objectives"]}}))
"""


def rerun(label: str, seed, hashseed: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        code = RERUN_CODE.format(root=str(ROOT), tmp=tmp, seed=seed)
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, cwd=str(ROOT))
        if res.returncode != 0:
            raise RuntimeError(f"rerun {label} failed: {res.stderr[-2000:]}")
        out = json.loads([ln for ln in res.stdout.splitlines() if ln.startswith("{")][-1])
    out.update(label=label, shuffle_seed=seed, pythonhashseed=hashseed, fresh_process=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reruns", type=int, default=2)
    a = ap.parse_args()

    mp = ROOT / E.SPLIT_MANIFEST
    gp = ROOT / E.GROUP_MANIFEST
    rep = E.audit_split(ROOT)
    dist = E.distributions(ROOT)
    failed = E.failed_sample_accounting(ROOT)
    import pyarrow.parquet as pq
    rows = pq.read_table(mp).to_pylist()

    # ---------------- counts ----------------
    with open(AUDIT / "M3_SPLIT_COUNTS.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["dataset", "split", "allocation_groups", "canonical_videos", "usable_samples",
                "live_videos", "spoof_videos", "canonical_video_pct", "usable_sample_pct"]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for ds in E.DATASETS:
            tot = rep["counts"][ds]["_total"]
            for s in A.SPLITS:
                c = rep["counts"][ds][s]
                w.writerow({"dataset": ds, "split": s, **{k: c[k] for k in
                            ("allocation_groups", "canonical_videos", "usable_samples",
                             "live_videos", "spoof_videos")},
                            "canonical_video_pct": round(100.0 * c["canonical_videos"] / tot["canonical_videos"], 4),
                            "usable_sample_pct": round(100.0 * c["usable_samples"] / tot["usable_samples"], 4)})

    # ---------------- distributions ----------------
    for level, fname in (("binary", "M3_DISTRIBUTION_BINARY.csv"),
                         ("attack_macro", "M3_DISTRIBUTION_ATTACK_MACRO.csv"),
                         ("attack_raw", "M3_DISTRIBUTION_ATTACK_RAW.csv")):
        with open(AUDIT / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dist[level][0]), lineterminator="\n")
            w.writeheader()
            w.writerows(dist[level])

    # ---------------- failed-sample accounting ----------------
    with open(AUDIT / "M3_FAILED_SAMPLE_SPLIT_ACCOUNTING.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(failed[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(failed)

    # ---------------- determinism ----------------
    authoritative = E.sha256_file(mp)
    runs = [{"label": "authoritative", "manifest_sha256": authoritative,
             "group_manifest_sha256": E.sha256_file(gp), "rows": len(rows),
             "shuffle_seed": None, "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
             "fresh_process": False}]
    plans = [("rerun_A_canonical_order", None, "0"),
             ("rerun_B_shuffled_metadata_order", 20260814, "1"),
             ("rerun_C_shuffled_other_seed", 7, "12345")][:max(2, a.reruns)]
    for label, seed, hs in plans:
        runs.append(rerun(label, seed, hs))
    det = {"authoritative_manifest_sha256": authoritative,
           "runs": runs,
           "all_manifest_hashes_identical": len({r["manifest_sha256"] for r in runs}) == 1,
           "all_group_hashes_identical": len({r["group_manifest_sha256"] for r in runs}) == 1,
           "all_objectives_identical": len({json.dumps(r.get("objectives"), sort_keys=True)
                                            for r in runs if r.get("objectives")}) <= 1,
           "byte_identical": len({r["manifest_sha256"] for r in runs}) == 1}
    det["status"] = "PASS" if det["byte_identical"] and det["all_group_hashes_identical"] else "FAIL"
    (AUDIT / "M3_DETERMINISM_COMPARE.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")

    # ---------------- leakage json/md ----------------
    (AUDIT / "M3_LEAKAGE_AUDIT.json").write_text(json.dumps(rep, indent=2, default=str) + "\n",
                                                 encoding="utf-8")
    write_reports(rep, dist, failed, det, rows)
    print(json.dumps({"rows": rep["rows"], "audit_ok": rep["ok"], "determinism": det["status"],
                      "manifest_sha256": authoritative}, indent=1))
    return 0 if rep["ok"] and det["status"] == "PASS" else 1


def write_reports(rep, dist, failed, det, rows):
    C = rep["counts"]

    # ---- leakage ----
    L = ["# M3 — Leakage Audit", "",
         f"Manifest `{rep['manifest']}` · {rep['rows']:,} rows · sha256 `{rep['manifest_sha256']}`.",
         f"Overall: **{'PASS' if rep['ok'] else 'FAIL'}**.", "",
         "## Global checks", "", "| check | result |", "|---|---|"]
    for k, v in rep["checks"].items():
        if k == "missing_complete_rows":
            L.append(f"| missing M2 COMPLETE rows | {len(v)} |")
        else:
            L.append(f"| {k.replace('_', ' ')} | {v} |")
    L += ["", f"Canonical videos whose frames landed in more than one split: "
              f"**{len(rep['leakage']['videos_with_multiple_splits'])}**", "",
          "## Per dataset", ""]
    for ds in E.DATASETS:
        e = rep["leakage"][ds]
        L += [f"### {ds}", "",
              f"- grouping: `{e['grouping']}` (key `{e['group_key']}`) — fidelity **{e['fidelity']}**"]
        if ds == "siwmv2":
            L += [f"- **subject leakage: {e['subject_leakage']}** — {e['subject_leakage_reason']}",
                  f"- `subject_id_global` null for every SiW row: {e['subject_id_global_all_null']}",
                  f"- subject-disjoint claim: **{e['subject_disjoint_claim']}** "
                  "(SiW is canonical-video / content-group-disjoint and must never be described as subject-disjoint)",
                  f"- content-group intersections across splits: {e['content_group_leakage']}",
                  f"- content groups spanning more than one split: "
                  f"**{len(e['content_groups_spanning_splits'])}** "
                  "(all exact-byte duplicate videos stay together)"]
        else:
            L += [f"- subject intersections across splits: {e['subject_leakage']}"]
        L += [f"- allocation-group intersections: {e['allocation_group_leakage']}",
              f"- video intersections: {e['video_leakage']}",
              f"- sample intersections: {e['sample_leakage']}", ""]
    (AUDIT / "M3_LEAKAGE_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- determinism ----
    D = ["# M3 — Determinism Report", "", f"Status: **{det['status']}**.", "",
         "The whole allocator was re-executed in **fresh interpreters** into clean temporary "
         "directories, including a run whose input metadata order was deliberately permuted and runs "
         "under different `PYTHONHASHSEED` values. A manifest is accepted only if the Parquet bytes "
         "are identical, not merely the assignment.", "",
         "| run | fresh process | metadata order | PYTHONHASHSEED | rows | manifest sha256 |",
         "|---|---|---|---|---|---|"]
    for r in det["runs"]:
        order = "canonical" if r["shuffle_seed"] is None else f"shuffled (seed {r['shuffle_seed']})"
        D.append(f"| {r['label']} | {r['fresh_process']} | {order} | {r['pythonhashseed']} | "
                 f"{r['rows']:,} | `{r['manifest_sha256'][:32]}…` |")
    D += ["", f"- all manifest hashes identical: **{det['all_manifest_hashes_identical']}**",
          f"- all group-manifest hashes identical: **{det['all_group_hashes_identical']}**",
          f"- all allocator objectives identical: **{det['all_objectives_identical']}**",
          f"- byte-identical split manifest: **{det['byte_identical']}**", "",
          "Serialization is pinned by the frozen writer settings recorded in "
          "`M3_SPLIT_REPORT.md`; row order is frozen as "
          f"`{' > '.join(E.ROW_ORDER)}` and is applied inside the writer, so input row order cannot "
          "reach the file."]
    (AUDIT / "M3_DETERMINISM_REPORT.md").write_text("\n".join(D) + "\n", encoding="utf-8")

    # ---- main split report ----
    S = ["# M3 — Authoritative Split Report", "",
         f"`manifests/split_v1.parquet` · **{rep['rows']:,} rows** · sha256 "
         f"`{rep['manifest_sha256']}`. Produced by the frozen Q-01 allocator; this pass made no "
         "scientific choice of its own.", "",
         "## 1. Population", "",
         "| | |", "|---|---|",
         "| M2 sampled samples | 20,640 |",
         f"| M2 COMPLETE (usable, in manifest) | {rep['rows']:,} |",
         f"| M2 FAILED (provenance only, excluded) | {len(failed)} |",
         "", "Every M2 COMPLETE row appears exactly once; no FAILED row is present. A canonical "
         "video weighs exactly one video in the objective regardless of how many of its frames "
         "survived, so the 23 partially-failed videos were balanced like any other.", "",
         "## 2. Allocation and usable rows", "",
         "| dataset | split | groups | canonical videos | video % | usable samples | live vids | spoof vids |",
         "|---|---|---|---|---|---|---|---|"]
    for ds in E.DATASETS:
        tot = C[ds]["_total"]
        for s in A.SPLITS:
            c = C[ds][s]
            S.append(f"| {ds} | {s} | {c['allocation_groups']:,} | {c['canonical_videos']:,} | "
                     f"{100.0 * c['canonical_videos'] / tot['canonical_videos']:.3f} % | "
                     f"{c['usable_samples']:,} | {c['live_videos']:,} | {c['spoof_videos']:,} |")
        S.append(f"| **{ds}** | **all** | **{tot['allocation_groups']:,}** | "
                 f"**{tot['canonical_videos']:,}** | 100 % | **{tot['usable_samples']:,}** | | |")
    S += ["", "Ratios are the allocator's objective unit — **canonical videos**, not frames. Frame "
              "ratios differ slightly because a few videos contribute 6 or 7 usable frames.", "",
          "## 3. Distributions", "",
          "Full tables: `M3_DISTRIBUTION_BINARY.csv`, `M3_DISTRIBUTION_ATTACK_MACRO.csv`, "
          "`M3_DISTRIBUTION_ATTACK_RAW.csv` (canonical-video counts with per-category deviation in "
          "percentage points). These are audit results: the split was not adjusted after reading them.",
          "", "| dataset | category | videos | TRAIN % | VAL % | TEST % |", "|---|---|---|---|---|---|"]
    for r in dist["binary"]:
        lab = "live" if r["category"] == "0" else "spoof"
        S.append(f"| {r['dataset']} | {lab} | {r['total_videos']:,} | {r['TRAIN_pct']:.2f} | "
                 f"{r['VAL_pct']:.2f} | {r['TEST_pct']:.2f} |")
    worst = sorted(dist["attack_raw"], key=lambda r: -max(abs(r[f"{s}_deviation_pp"]) for s in A.SPLITS))[:8]
    S += ["", "Largest per-category deviations at the attack_raw level (percentage points from target):", "",
          "| dataset | attack_raw | videos | TRAIN pp | VAL pp | TEST pp |", "|---|---|---|---|---|---|"]
    for r in worst:
        S.append(f"| {r['dataset']} | {r['category']} | {r['total_videos']:,} | "
                 f"{r['TRAIN_deviation_pp']:+.2f} | {r['VAL_deviation_pp']:+.2f} | {r['TEST_deviation_pp']:+.2f} |")
    S += ["", "Deviation here is driven by group granularity and by the lexicographic order (P1 and "
              "P2 are satisfied first and may not be worsened), not by a tuning choice.", "",
          "## 4. Manifest", "", "| | |", "|---|---|",
          f"| rows | {rep['rows']:,} |",
          f"| columns | {', '.join('`' + c + '`' for c in rep['columns'])} |",
          f"| canonical row order | `{' > '.join(E.ROW_ORDER)}` |",
          f"| writer | pyarrow `pq.write_table` {json.dumps(E.PARQUET_WRITER)} |",
          f"| sha256 | `{rep['manifest_sha256']}` |", "",
          "`sha256` follows the existing lineage: CASIA keeps M1's `original_frame_bytes`, and MSU/SiW "
          "resolve M1's `PENDING_M2_CANONICAL_PNG` placeholder with M2's canonical frame PNG hash. "
          "`sha256_kind` records which applies. `subject_id_global` is left null for SiW — no subject "
          "identity is manufactured.", "",
          "## 5. M2 failed-sample accounting", "",
          f"All **{len(failed)}** M2 failures are accounted for in "
          "`M3_FAILED_SAMPLE_SPLIT_ACCOUNTING.csv`: none appears in the split manifest, and each is "
          "recorded together with the split its canonical video received. They are provenance only "
          "and were neither replaced nor resampled.", ""]
    (AUDIT / "M3_SPLIT_REPORT.md").write_text("\n".join(S) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
