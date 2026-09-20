"""Exhaustive audit of the authoritative M4 common pair manifests (spec §6; prompt §21-§26).

    <m2 venv>/bin/python tools/m4_audit.py

Re-derives every pair independently from the frozen inputs and checks the written manifests against
that recomputation. Nothing here can change pair membership: the audit only reads.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P          # noqa: E402
from gpatbench.pairs import execute as E         # noqa: E402

AUDIT = ROOT / "outputs/audit"
EXPECTED_ROWS = {"TRAIN": 8838, "VAL": 1905}
CAND_AUDIT_SALT = "gpatbench.m4_candidate_audit.v1|"
CAND_AUDIT_PER_CELL = 25


def quart(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main() -> int:
    pop = E.load_population()
    doc = E.build_train_stats(pop)
    stats = E.stats_to_posestats(doc)
    by_id = {s.sample_id: s for s in pop["samples"]}
    all_rows = {r["sample_id"]: r for r in pq.read_table(E.SPLIT_MANIFEST).to_pylist()}
    acct = {r["sample_id"]: r for r in pq.read_table(E.ACCOUNTING).to_pylist()}

    report = {"expected_rows": EXPECTED_ROWS, "splits": {}, "checks": {}, "failures": []}

    def fail(msg):
        report["failures"].append(msg)

    counts_csv, dist_csv, reuse_csv, cand_csv = [], [], [], []

    for split in P.SPLITS_WITH_PAIRS:
        path = E.MANIFEST_PATH[split]
        written = pq.read_table(path).to_pylist()
        schema = pq.read_schema(path)
        rebuilt = E.build_pairs(pop, split, stats)
        s = {"path": path.relative_to(ROOT).as_posix(), "rows": len(written),
             "sha256": E.sha256_file(path), "schema": [f"{f.name}:{f.type}" for f in schema]}

        # ---- §21 structural
        if len(written) != EXPECTED_ROWS[split]:
            fail(f"{split}: {len(written)} rows, expected {EXPECTED_ROWS[split]}")
        spoof = [x for x in pop["samples"] if x.split == split and x.label_binary == 1]
        s["spoof_sources_in_split"] = len(spoof)
        if len(written) != len(spoof):
            fail(f"{split}: rows != spoof-source count")
        if len({r["pair_id"] for r in written}) != len(written):
            fail(f"{split}: duplicate pair_id")
        if len({r["source_spoof_id"] for r in written}) != len(written):
            fail(f"{split}: duplicate source_spoof_id")
        if {r["source_spoof_id"] for r in written} != {x.sample_id for x in spoof}:
            fail(f"{split}: source set != spoof-source set")
        if written != rebuilt:
            fail(f"{split}: written manifest does not equal an independent rebuild")
        if [r[c] for r in written for c in E.ROW_ORDER] != \
           [r[c] for r in sorted(written, key=lambda r: tuple(r[c] for c in E.ROW_ORDER))
                for c in E.ROW_ORDER]:
            fail(f"{split}: rows are not in canonical (dataset, source_spoof_id) order")
        expected_ids = P.assign_pair_ids(split, P.canonical_source_order(spoof))
        if [r["pair_id"] for r in sorted(written, key=lambda r: tuple(r[c] for c in E.ROW_ORDER))] \
                != expected_ids:
            fail(f"{split}: pair_id sequence is not the canonical assignment")

        ev, el = [], []
        for r in written:
            src, tgt = by_id[r["source_spoof_id"]], by_id[r["target_live_id"]]
            if src.label_binary != 1:
                fail(f"{r['pair_id']}: source is not SPOOF")
            if tgt.label_binary != 0:
                fail(f"{r['pair_id']}: target is not LIVE")
            if src.dataset != tgt.dataset or src.dataset != r["dataset"]:
                fail(f"{r['pair_id']}: dataset mismatch")
            if src.split != split or tgt.split != split:
                fail(f"{r['pair_id']}: split crossing")
            for sid in (src.sample_id, tgt.sample_id):
                if all_rows[sid]["m2_status"] != "COMPLETE" or acct[sid]["final_status"] != "COMPLETE":
                    fail(f"{r['pair_id']}: {sid} is not M2 COMPLETE")
                if all_rows[sid]["split"] == "TEST":
                    fail(f"{r['pair_id']}: TEST reference {sid}")
            if r["split"] != split:
                fail(f"{r['pair_id']}: split column is {r['split']}")
            if r["seed"] != P.SPLIT_SEED:
                fail(f"{r['pair_id']}: seed {r['seed']}")
            if r["split_manifest_sha256"] != pop["split_manifest_sha256"]:
                fail(f"{r['pair_id']}: wrong split_manifest_sha256")
            if r["pairs_config_sha256"] != pop["pairs_config_sha256"]:
                fail(f"{r['pair_id']}: wrong pairs_config_sha256")
            if r["source_sha256"] != src.sha256 or r["target_sha256"] != tgt.sha256:
                fail(f"{r['pair_id']}: lineage sha mismatch")
            if r["attack_macro"] != src.attack_macro:
                fail(f"{r['pair_id']}: attack_macro is not the source's")
            # distances
            for k in ("d_pose", "d_scale", "d_luma", "d_pair"):
                v = r[k]
                if not np.isfinite(v) or v < 0.0:
                    fail(f"{r['pair_id']}: {k} = {v}")
            recomputed = 0.5 * r["d_pose"] + 0.3 * r["d_scale"] + 0.2 * r["d_luma"]
            if recomputed != r["d_pair"]:
                fail(f"{r['pair_id']}: d_pair != 0.5/0.3/0.2 recombination")
            d = P.d_pair(src, tgt, stats[src.dataset])
            for k in ("d_pose", "d_scale", "d_luma", "d_pair"):
                if d[k] != r[k]:
                    fail(f"{r['pair_id']}: {k} does not recompute from the frozen primitives")
            if r["face_area_fraction_source"] != src.face_area_fraction or \
               r["face_area_fraction_target"] != tgt.face_area_fraction:
                fail(f"{r['pair_id']}: face_area_fraction mismatch")
            ev.append(r["candidate_count_evaluated"])
            el.append(r["candidate_count_eligible"])
            # ---- §22 / §23 eligibility
            if src.dataset in ("casia_fasd", "msu_mfsd"):
                if not r["source_subject"] or not r["target_subject"]:
                    fail(f"{r['pair_id']}: null subject in {src.dataset}")
                if r["source_subject"] == r["target_subject"]:
                    fail(f"{r['pair_id']}: same subject")
            else:
                if r["source_subject"] is not None or r["target_subject"] is not None:
                    fail(f"{r['pair_id']}: SiW subject field is not null")
                if r["source_video_id"] == r["target_video_id"]:
                    fail(f"{r['pair_id']}: SiW same video")
                if r["source_content_group_id"] == r["target_content_group_id"]:
                    fail(f"{r['pair_id']}: SiW same content group")

        s["candidate_count_evaluated"] = {"min": min(ev), "max": max(ev),
                                          "all_equal_cap": min(ev) == max(ev) == P.CANDIDATE_CAP}
        s["candidate_count_eligible"] = {"min": min(el), "max": max(el),
                                         "all_at_least_cap": min(el) >= P.CANDIDATE_CAP}
        if min(el) < P.CANDIDATE_CAP:
            fail(f"{split}: a source has fewer than {P.CANDIDATE_CAP} eligible targets")

        # ---- winner is the minimum over the evaluated set (exhaustive)
        live_by_ds = collections.defaultdict(list)
        for x in pop["samples"]:
            if x.split == split and x.label_binary == 0:
                live_by_ds[x.dataset].append(x)
        for ds in live_by_ds:
            live_by_ds[ds].sort(key=lambda t: t.sample_id)
        wrong_winner = tie_checked = 0
        for r in written:
            src = by_id[r["source_spoof_id"]]
            cands = E.select_candidates(src, P.eligible_targets(src, live_by_ds[src.dataset]))
            keyed = sorted(((P.d_pair(src, t, stats[src.dataset])["d_pair"], t.sample_id)
                            for t in cands))
            if keyed[0][1] != r["target_live_id"]:
                wrong_winner += 1
            if len(keyed) > 1 and keyed[0][0] == keyed[1][0]:
                tie_checked += 1
        if wrong_winner:
            fail(f"{split}: {wrong_winner} winners are not the minimum of the evaluated set")
        s["exact_distance_ties_at_the_top"] = tie_checked

        # ---- §24 target reuse
        for ds in sorted({r["dataset"] for r in written}):
            sub = [r for r in written if r["dataset"] == ds]
            c = collections.Counter(r["target_live_id"] for r in sub)
            hist = collections.Counter(c.values())
            reuse_csv.append({"split": split, "dataset": ds, "pairs": len(sub),
                              "live_targets_available": len(live_by_ds[ds]),
                              "unique_targets_used": len(c),
                              "max_reuse": max(c.values()), "min_reuse": min(c.values()),
                              "mean_reuse": round(len(sub) / len(c), 6),
                              "reuse_histogram": json.dumps(dict(sorted(hist.items())))})
            counts_csv.append({"split": split, "dataset": ds, "pairs": len(sub),
                               "spoof_sources": sum(1 for x in spoof if x.dataset == ds),
                               "live_targets_available": len(live_by_ds[ds])})
            # ---- §25 distance distribution
            row = {"split": split, "dataset": ds}
            for k in ("d_pose", "d_scale", "d_luma", "d_pair"):
                q = quart([r[k] for r in sub])
                for stat in ("min", "median", "p95", "max"):
                    row[f"{k}_{stat}"] = q[stat]
            row["n"] = len(sub)
            dist_csv.append(row)
            if ds == "casia_fasd":
                uniq = sorted({r["d_scale"] for r in sub})
                s.setdefault("casia_d_scale_unique", {})[split] = uniq
                if uniq != [0.0]:
                    fail(f"{split}/casia: d_scale unique values {uniq} != [0.0]")

        # ---- §26 candidate-selection evidence on a deterministic small subset
        for ds in sorted({r["dataset"] for r in written}):
            sub = sorted([r for r in written if r["dataset"] == ds],
                         key=lambda r: E.sha256_text(CAND_AUDIT_SALT + r["source_spoof_id"]))
            for r in sub[:CAND_AUDIT_PER_CELL]:
                src = by_id[r["source_spoof_id"]]
                elig = P.eligible_targets(src, live_by_ds[ds])
                cands = E.select_candidates(src, elig)
                cand_csv.append({
                    "split": split, "dataset": ds, "pair_id": r["pair_id"],
                    "source_spoof_id": src.sample_id,
                    "eligible_count": len(elig),
                    "eligible_ids_sha256": E.sha256_text("\n".join(t.sample_id for t in elig)),
                    "selected_count": len(cands),
                    "selected_ids_sha256": E.sha256_text("\n".join(t.sample_id for t in cands)),
                    "winner_target_id": r["target_live_id"],
                    "winner_in_selected": r["target_live_id"] in {t.sample_id for t in cands},
                    "winner_d_pair": repr(r["d_pair"])})
        report["splits"][split] = s

    report["checks"]["structural"] = not report["failures"]
    report["stats_sha256"] = E.sha256_file(E.STATS_PATH)
    report["pairs_config_sha256"] = pop["pairs_config_sha256"]
    report["split_manifest_sha256"] = pop["split_manifest_sha256"]
    report["schema_signature"] = E.schema_signature()
    report["parquet_writer"] = dict(E.PARQUET_WRITER)
    report["failures_count"] = len(report["failures"])
    report["status"] = "PASS" if not report["failures"] else "FAIL"

    def wcsv(name, rows):
        p = AUDIT / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    wcsv("M4_PAIR_COUNTS.csv", counts_csv)
    wcsv("M4_PAIR_DISTANCE_DISTRIBUTION.csv", dist_csv)
    wcsv("M4_PAIR_TARGET_REUSE.csv", reuse_csv)
    wcsv("M4_CANDIDATE_SELECTION_AUDIT.csv", cand_csv)
    (AUDIT / "M4_COMMON_PAIR_AUDIT.json").write_bytes(E.canonical_json_bytes(report))
    print(json.dumps({"status": report["status"], "failures": report["failures"][:10],
                      "rows": {k: v["rows"] for k, v in report["splits"].items()}}, indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
