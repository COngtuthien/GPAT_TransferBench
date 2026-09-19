"""PRE-M3 feasibility analysis of the frozen Q-01 allocator on the real metadata.

    <m2 venv>/bin/python tools/m3_allocator_feasibility.py

This does NOT create the split. It measures whether the frozen allocator can reach a proven global
optimum on the real problem, and how big that problem is. The allocator is run in memory purely to
obtain that evidence; the resulting group->split membership is deliberately **discarded and never
written, printed or hashed**. The authoritative split is created later, by M3 itself.

What is persisted: problem dimensions, category counts, rare-category group counts, the objective
values reached, the aggregate per-split video counts that demonstrate feasibility, and the solver's
optimality evidence. What is not persisted: which group went where, and the per-class count vectors.

Writes outputs/audit/M3_ALLOCATOR_FEASIBILITY.md and M3_ALLOCATOR_OBJECTIVE.json.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.split import allocator as A  # noqa: E402

AUDIT = ROOT / "outputs/audit"
DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")


def build_groups(dataset: str):
    """Real allocation groups for one dataset, from the frozen M1 manifests + M2 accounting."""
    vids = [v for v in pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist()
            if v["dataset"] == dataset]
    acct = [r for r in pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet").to_pylist()
            if r["dataset"] == dataset]
    per_video = collections.defaultdict(lambda: {"c": 0, "f": 0})
    for r in acct:
        per_video[r["video_id"]]["c" if r["final_status"] == "COMPLETE" else "f"] += 1
    if dataset == "siwmv2":
        groups_map = {}
        with open(AUDIT / "siw_content_groups.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                groups_map[r["video_id"]] = r["content_group_id"]
        key = lambda v: groups_map[v["video_id"]]                      # noqa: E731
    else:
        key = lambda v: v["subject_id_global"]                         # noqa: E731

    buckets = collections.defaultdict(list)
    for v in vids:
        m = per_video[v["video_id"]]
        buckets[key(v)].append(A.Video(v["video_id"], int(v["label_binary"]), v["attack_macro"],
                                       v["attack_raw"], m["c"], m["f"]))
    return [A.Group(dataset, gid, tuple(sorted(vs, key=lambda x: x.video_id)))
            for gid, vs in sorted(buckets.items())]


def dimensions(groups) -> dict:
    sizes = collections.Counter(g.n_videos for g in groups)
    classes = collections.Counter(g.class_signature for g in groups)
    n_videos = sum(g.n_videos for g in groups)
    cats = {}
    for level in A.LEVELS[1:]:
        c = collections.Counter()
        for g in groups:
            for v in g.videos:
                for k in A._category_keys(level, v):
                    c[k] += 1
        cats[level] = dict(c.most_common())
    # model size after the profile-class reduction
    J, S = len(classes), len(A.SPLITS)
    n_abs = sum(len(cats[lvl]) for lvl in A.LEVELS[1:]) * S + S       # + P1's three terms
    return {
        "canonical_videos": n_videos,
        "allocation_groups": len(groups),
        "group_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "profile_classes_after_reduction": J,
        "usable_samples": sum(v.n_complete for g in groups for v in g.videos),
        "failed_samples_provenance_only": sum(v.n_failed for g in groups for v in g.videos),
        "category_counts": cats,
        "model_integer_variables": J * S,
        "model_deviation_variables": n_abs,
        "model_total_columns": J * S + n_abs,
        "model_constraint_rows": J + 2 * n_abs,
        "naive_variables_without_reduction": len(groups) * S,
    }


def main() -> int:
    report = {"purpose": "PRE-M3 feasibility only; no allocation is persisted",
              "allocator": "gpatbench.split.allocator (frozen Q-01 lexicographic contract)",
              "membership_persisted": False, "datasets": {}}
    for ds in DATASETS:
        groups = build_groups(ds)
        dim = dimensions(groups)
        rare = A.rare_category_report(groups)
        t0 = time.perf_counter()
        res = A.allocate(groups)                     # in-memory only
        elapsed = time.perf_counter() - t0
        # Aggregate feasibility evidence only. The assignment and the per-class count vectors are
        # intentionally NOT copied into the report and go out of scope with `res`.
        achieved = {s: {"canonical_videos": res.stats[s]["canonical_videos"],
                        "video_pct": res.stats[s]["video_pct"],
                        "usable_samples": res.stats[s]["usable_samples"],
                        "live_videos": res.stats[s]["live_videos"],
                        "spoof_videos": res.stats[s]["spoof_videos"]} for s in A.SPLITS}
        report["datasets"][ds] = {
            "dimensions": dim,
            "rare_categories": {lvl: {k: v for k, v in d.items() if not v["can_occupy_three_splits"]}
                                for lvl, d in rare.items()},
            "category_group_counts": {lvl: {k: v["groups"] for k, v in d.items()}
                                      for lvl, d in rare.items()},
            "objectives": res.objectives,
            "solver_evidence": res.evidence,
            "achieved_split_shape": achieved,
            "solve_seconds": round(elapsed, 3),
            "feasible": True,
        }
        del res
        print(json.dumps({"dataset": ds, "groups": dim["allocation_groups"],
                          "classes": dim["profile_classes_after_reduction"],
                          "objectives": report["datasets"][ds]["objectives"],
                          "seconds": report["datasets"][ds]["solve_seconds"]}))
    report["backend"] = A.solver_backend(A.DEFAULT_OPTIONS)
    (AUDIT / "M3_ALLOCATOR_OBJECTIVE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_md(report)
    return 0


def write_md(rep: dict) -> None:
    L = ["# M3 — Allocator Feasibility (PRE-M3, no split created)", "",
         "The frozen Q-01 allocator was run against the real metadata **only** to measure problem size, "
         "feasibility and optimality. The resulting group-to-split membership was discarded and is not "
         "written, printed or hashed anywhere; `manifests/split_v1.parquet` does not exist. M3 itself "
         "will produce the authoritative split.", "",
         f"Backend: **{rep['backend']['backend']}** (scipy {rep['backend']['scipy_version']}), "
         f"{rep['backend']['kind']}, threads {rep['backend']['threads']}, "
         f"options `{rep['backend']['deterministic_options']}`.", ""]
    for ds, d in rep["datasets"].items():
        dim = d["dimensions"]
        L += [f"## {ds}", "",
              "| dimension | value |", "|---|---|",
              f"| canonical videos | {dim['canonical_videos']:,} |",
              f"| allocation groups | {dim['allocation_groups']:,} |",
              f"| group size distribution | {dim['group_size_distribution']} |",
              f"| profile classes after reduction | **{dim['profile_classes_after_reduction']}** |",
              f"| usable samples (M2 COMPLETE) | {dim['usable_samples']:,} |",
              f"| failed samples (provenance only) | {dim['failed_samples_provenance_only']:,} |",
              f"| integer variables (reduced) | {dim['model_integer_variables']} |",
              f"| deviation variables | {dim['model_deviation_variables']} |",
              f"| total columns | {dim['model_total_columns']} |",
              f"| constraint rows | {dim['model_constraint_rows']} |",
              f"| variables without the class reduction | {dim['naive_variables_without_reduction']:,} |",
              f"| solve time (all levels + canonicalisation) | {d['solve_seconds']} s |", "",
              "Category counts:", ""]
        for lvl, c in dim["category_counts"].items():
            L.append(f"- **{lvl}** ({len(c)} categories): {c}")
        L += ["", "Objective values reached (all levels proven globally optimal):", "",
              "| level | objective | solver status | mip gap |", "|---|---|---|---|"]
        for e in d["solver_evidence"]:
            L.append(f"| {e['level']} | {e['objective']:,} | {e['message']} | {e['mip_gap']} |")
        L += ["", "Achieved split shape (aggregate feasibility evidence; **not** a membership table):", "",
              "| split | canonical videos | % | usable samples | live | spoof |", "|---|---|---|---|---|---|"]
        for s, v in d["achieved_split_shape"].items():
            L.append(f"| {s} | {v['canonical_videos']:,} | {v['video_pct']:.4f} % | "
                     f"{v['usable_samples']:,} | {v['live_videos']:,} | {v['spoof_videos']:,} |")
        rare = {lvl: v for lvl, v in d["rare_categories"].items() if v}
        L += ["", ("Categories that cannot occupy all three splits (fewer than 3 independent groups): "
                   + (json.dumps(rare) if rare else "**none**")), ""]
    L += ["## Why this is exact", "",
          "Groups carrying an identical multiset of (label_binary, attack_macro, attack_raw) are "
          "interchangeable for every P1-P4 term, so the allocation collapses to a per-class count "
          "vector. That reduction is what makes a proven global optimum cheap here: the real problems "
          "shrink to a handful of integer variables, and every level terminates with `mip_gap = 0`. "
          "Which concrete group receives which split is then fixed by the frozen seeded hash, not by "
          "the solver.", ""]
    (AUDIT / "M3_ALLOCATOR_FEASIBILITY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
