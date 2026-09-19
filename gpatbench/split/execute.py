"""M3 authoritative split execution and audit.

Runs the frozen Q-01 allocator (`gpatbench.split.allocator`) over the frozen M1/M2 metadata and
writes `manifests/split_v1.parquet`. This module contains no scientific choice of its own: the
objective, grouping, seed and tie-break all come from `configs/frozen/split_v1.yaml`, whose SHA-256
is asserted before anything is written.

Population
----------
Allocation happens at canonical-video level; each canonical video weighs exactly one video whatever
its surviving frame count. Only M2 `COMPLETE` rows become split rows, so the manifest holds 20,615
of the 20,640 sampled frames; the 25 `SCRFD_NO_FACE` rows stay in `m2_sample_accounting.parquet` and
are separately accounted for by the audit.

`sha256` lineage
----------------
Spec §3.2 defines the column as "hash of original frame bytes or canonical extracted PNG". M1 filled
it for CASIA (`original_frame_bytes`, the pre-cropped source PNG is itself the canonical frame) and
left MSU/SiW as `PENDING_M2_CANONICAL_PNG`. M2 produced exactly that canonical frame PNG, so the
placeholder is resolved with M2's `frame_png_sha256` and `sha256_kind` records which lineage applies.
No new hash meaning is invented, and for CASIA the two agree on all 4,800 rows (asserted).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import allocator as A

DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
SPLIT_CONFIG = "configs/frozen/split_v1.yaml"
SPLIT_MANIFEST = "manifests/split_v1.parquet"
GROUP_MANIFEST = "manifests/split_groups_v1.parquet"

# Required frozen-spec columns first, then stable audit columns.
SPEC_COLUMNS = ["sample_id", "dataset", "subject_id_global", "video_id", "frame_index",
                "label_binary", "attack_raw", "attack_macro", "split", "sha256"]
AUDIT_COLUMNS = ["sha256_kind", "content_group_id", "allocation_group_id", "m2_status"]
COLUMNS = SPEC_COLUMNS + AUDIT_COLUMNS

# Canonical row order, frozen. `frame_index` sorts numerically, the rest as strings.
ROW_ORDER = ("dataset", "video_id", "frame_index", "sample_id")

# Deterministic Parquet writer settings, frozen (M1 convention + explicit determinism controls).
PARQUET_WRITER = {"compression": "zstd", "compression_level": 9, "write_statistics": True,
                  "version": "2.6", "use_dictionary": False, "row_group_size": 65536,
                  "store_schema": True, "write_page_index": False}


def _maybe_shuffle(items: list, seed: int | None) -> list:
    """Permute an intermediate list to prove the result is order-invariant.

    Used only by the determinism audit: the authoritative run passes seed=None. Any dependence on
    input row order would show up as a different manifest, which is exactly what we want to detect.
    """
    if seed is None:
        return items
    import random
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_frozen_config(root: Path) -> dict:
    cfg = yaml.safe_load((root / SPLIT_CONFIG).read_text())
    if cfg["status"] != "FROZEN" or cfg["q01_status"] != "RESOLVED_BY_OWNER_LEXICOGRAPHIC_ALLOCATOR":
        raise A.AllocationError("split_v1.yaml is not the frozen Q-01 contract")
    if cfg["split_seed"] != A.SPLIT_SEED or cfg["normalization"]["scale"] != A.SCALE:
        raise A.AllocationError("frozen config disagrees with the allocator implementation")
    if cfg["integer_percentages"] != dict(A.PCT):
        raise A.AllocationError("frozen config ratios disagree with the allocator")
    return cfg


# ---------------------------------------------------------------- population
def load_population(root: Path, shuffle_seed: int | None = None) -> dict:
    """Authoritative per-dataset allocation groups, rebuilt from the frozen manifests."""
    inv = _maybe_shuffle(pq.read_table(root / "manifests/inventory.parquet").to_pylist(), shuffle_seed)
    acct = pq.read_table(root / "manifests/m2_sample_accounting.parquet").to_pylist()
    by_sample = {r["sample_id"]: r for r in acct}
    if {r["sample_id"] for r in inv} != set(by_sample):
        raise A.AllocationError("inventory and M2 accounting cover different sample sets")

    groups_map = {}
    with open(root / "outputs/audit/siw_content_groups.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups_map[r["video_id"]] = r["content_group_id"]

    per_video: dict = {}
    for r in inv:
        st = by_sample[r["sample_id"]]["final_status"]
        v = per_video.setdefault((r["dataset"], r["video_id"]), {"complete": 0, "failed": 0, "row": r})
        v["complete" if st == "COMPLETE" else "failed"] += 1

    out = {}
    for ds in DATASETS:
        buckets: dict = {}
        for (d, vid), v in _maybe_shuffle(list(per_video.items()), shuffle_seed):
            if d != ds:
                continue
            r = v["row"]
            gid = groups_map[vid] if ds == "siwmv2" else r["subject_id_global"]
            if not gid:
                raise A.AllocationError(f"{ds}/{vid}: no allocation group key")
            buckets.setdefault(gid, []).append(
                A.Video(vid, int(r["label_binary"]), r["attack_macro"], r["attack_raw"],
                        v["complete"], v["failed"]))
        out[ds] = _maybe_shuffle(
            [A.Group(ds, gid, tuple(sorted(vs, key=lambda x: x.video_id)))
             for gid, vs in sorted(buckets.items())], shuffle_seed)
    return out


def verify_profile_class_equivalence(groups) -> dict:
    """Prove the reduction is sound: groups sharing a profile signature are truly interchangeable.

    The solver optimises per-class counts, which is only valid if swapping any two groups of a class
    leaves every objective-relevant quantity unchanged. This recomputes those quantities per group
    and requires them to be identical inside each class — including the canonical-video count, since
    group size enters P1 directly.
    """
    classes: dict = {}
    for g in groups:
        classes.setdefault(g.class_signature, []).append(g)
    report = {"classes": len(classes), "groups": len(groups), "violations": []}
    for sig, members in classes.items():
        vectors = set()
        for g in members:
            vec = [("n_videos", g.n_videos)]
            for level in A.LEVELS[1:]:
                counts: dict = {}
                for v in g.videos:
                    for k in A._category_keys(level, v):
                        counts[k] = counts.get(k, 0) + 1
                vec.append((level, tuple(sorted(counts.items()))))
            vectors.add(tuple(vec))
        if len(vectors) != 1:
            report["violations"].append({"class": repr(sig)[:120], "distinct_vectors": len(vectors),
                                         "groups": [g.group_id for g in members][:5]})
    # and: two different signatures must never produce the same contribution vector by accident
    report["ok"] = not report["violations"]
    if not report["ok"]:
        raise A.AllocationError(f"profile-class reduction is unsound: {report['violations']}")
    return report


# ---------------------------------------------------------------- execution
def build_rows(root: Path, population: dict, assignment: dict, shuffle_seed: int | None = None) -> list:
    """One row per M2 COMPLETE sample, in the frozen canonical order."""
    inv = _maybe_shuffle(pq.read_table(root / "manifests/inventory.parquet").to_pylist(), shuffle_seed)
    acct = {r["sample_id"]: r for r in
            pq.read_table(root / "manifests/m2_sample_accounting.parquet").to_pylist()}
    group_of, split_of = {}, {}
    for ds, groups in population.items():
        for g in groups:
            for v in g.videos:
                group_of[(ds, v.video_id)] = g.group_id
                split_of[(ds, v.video_id)] = assignment[ds][g.group_id]
    groups_map = {}
    with open(root / "outputs/audit/siw_content_groups.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups_map[r["video_id"]] = r["content_group_id"]

    rows = []
    for r in inv:
        a = acct[r["sample_id"]]
        if a["final_status"] != "COMPLETE":
            continue                                    # FAILED rows never enter the split manifest
        ds, vid = r["dataset"], r["video_id"]
        if r["sha256"]:                                 # CASIA: M1 already hashed the source frame
            sha, kind = r["sha256"], r["sha256_kind"]
            if sha != a["frame_png_sha256"]:
                raise A.AllocationError(f"{r['sample_id']}: M1 sha256 disagrees with the M2 frame hash")
        else:                                           # MSU/SiW: resolve PENDING_M2_CANONICAL_PNG
            sha, kind = a["frame_png_sha256"], "m2_canonical_frame_png"
            if not sha:
                raise A.AllocationError(f"{r['sample_id']}: no canonical frame hash")
        rows.append({
            "sample_id": r["sample_id"], "dataset": ds,
            "subject_id_global": r["subject_id_global"],     # SiW keeps its authoritative null
            "video_id": vid, "frame_index": int(r["frame_index"]),
            "label_binary": int(r["label_binary"]), "attack_raw": r["attack_raw"],
            "attack_macro": r["attack_macro"], "split": split_of[(ds, vid)], "sha256": sha,
            "sha256_kind": kind,
            "content_group_id": groups_map.get(vid) if ds == "siwmv2" else None,
            "allocation_group_id": group_of[(ds, vid)], "m2_status": "COMPLETE"})
    rows.sort(key=lambda r: (r["dataset"], r["video_id"], r["frame_index"], r["sample_id"]))
    return rows


def write_manifest(rows: list, path: Path) -> str:
    """Deterministic Parquet write. Returns the file SHA-256."""
    schema = pa.schema([(c, pa.int64() if c in ("frame_index", "label_binary") else pa.string())
                        for c in COLUMNS])
    table = pa.table({c: pa.array([r[c] for r in rows], schema.field(c).type) for c in COLUMNS},
                     schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **PARQUET_WRITER)
    return sha256_file(path)


def write_group_manifest(root: Path, population: dict, assignment: dict, path: Path) -> str:
    rows = []
    for ds in DATASETS:
        for g in population[ds]:
            rows.append({"dataset": ds, "allocation_group_id": g.group_id,
                         "split": assignment[ds][g.group_id],
                         "canonical_video_count": g.n_videos,
                         "usable_sample_count": sum(v.n_complete for v in g.videos),
                         "failed_sample_count": sum(v.n_failed for v in g.videos),
                         "stable_group_rank": A.stable_group_rank(ds, g.group_id)})
    rows.sort(key=lambda r: (r["dataset"], r["allocation_group_id"]))
    schema = pa.schema([("dataset", pa.string()), ("allocation_group_id", pa.string()),
                        ("split", pa.string()), ("canonical_video_count", pa.int64()),
                        ("usable_sample_count", pa.int64()), ("failed_sample_count", pa.int64()),
                        ("stable_group_rank", pa.string())])
    table = pa.table({f.name: pa.array([r[f.name] for r in rows], f.type) for f in schema},
                     schema=schema)
    pq.write_table(table, path, **PARQUET_WRITER)
    return sha256_file(path)


def run_split(root: Path, write: bool = True, manifest_path: Path | None = None,
              group_manifest_path: Path | None = None, shuffle_seed: int | None = None) -> dict:
    """Execute the authoritative split. Pure function of frozen metadata + frozen config + code.

    `shuffle_seed` permutes the intermediate metadata order; the authoritative run leaves it None.
    """
    cfg = load_frozen_config(root)
    population = load_population(root, shuffle_seed)
    assignment, objectives, evidence, reduction, stats = {}, {}, {}, {}, {}
    for ds in DATASETS:                                  # independently per dataset; never pooled
        groups = population[ds]
        reduction[ds] = verify_profile_class_equivalence(groups)
        res = A.allocate(groups)
        assignment[ds] = res.assignment
        objectives[ds] = res.objectives
        evidence[ds] = res.evidence
        stats[ds] = res.stats
    rows = build_rows(root, population, assignment, shuffle_seed)
    out = {"config_sha256": sha256_file(root / SPLIT_CONFIG), "rows": len(rows),
           "objectives": objectives, "solver_evidence": evidence, "reduction_check": reduction,
           "allocator_stats": stats, "row_order": list(ROW_ORDER),
           "parquet_writer": dict(PARQUET_WRITER), "columns": COLUMNS}
    if write:
        mp = manifest_path or (root / SPLIT_MANIFEST)
        gp = group_manifest_path or (root / GROUP_MANIFEST)
        out["manifest_path"] = str(mp)
        out["manifest_sha256"] = write_manifest(rows, mp)
        out["group_manifest_path"] = str(gp)
        out["group_manifest_sha256"] = write_group_manifest(root, population, assignment, gp)
    out["_rows"] = rows
    out["_assignment"] = assignment
    return out


# ---------------------------------------------------------------- audit
def audit_split(root: Path, manifest_path: Path | None = None) -> dict:
    """Leakage and distribution audit of a written split manifest.

    Reads only the manifest plus the frozen M1/M2 metadata, so it validates the artifact rather than
    the in-memory state that produced it.
    """
    mp = manifest_path or (root / SPLIT_MANIFEST)
    rows = pq.read_table(mp).to_pylist()
    acct = pq.read_table(root / "manifests/m2_sample_accounting.parquet").to_pylist()
    complete = {r["sample_id"] for r in acct if r["final_status"] == "COMPLETE"}
    failed = [r for r in acct if r["final_status"] == "FAILED"]
    policy = yaml.safe_load((root / "configs/frozen/dataset_protocol_policy_v1.yaml").read_text())

    rep: dict = {"manifest": str(mp), "manifest_sha256": sha256_file(mp), "rows": len(rows),
                 "columns": [f.name for f in pq.read_schema(mp)],
                 "row_order": list(ROW_ORDER), "leakage": {}, "counts": {}, "checks": {}}

    # ---- schema and population ----
    rep["checks"]["required_columns_present"] = all(c in rep["columns"] for c in SPEC_COLUMNS)
    ids = [r["sample_id"] for r in rows]
    rep["checks"]["unique_sample_ids"] = len(set(ids)) == len(ids)
    rep["checks"]["rows_equal_m2_complete"] = set(ids) == complete
    rep["checks"]["no_failed_row_present"] = not ({r["sample_id"] for r in failed} & set(ids))
    rep["checks"]["missing_complete_rows"] = sorted(complete - set(ids))[:10]
    rep["checks"]["every_row_has_a_split"] = all(r["split"] in A.SPLITS for r in rows)
    rep["checks"]["canonical_row_order"] = rows == sorted(
        rows, key=lambda r: (r["dataset"], r["video_id"], r["frame_index"], r["sample_id"]))

    # ---- leakage: every frame of a canonical video shares its split ----
    video_splits: dict = {}
    for r in rows:
        video_splits.setdefault((r["dataset"], r["video_id"]), set()).add(r["split"])
    rep["leakage"]["videos_with_multiple_splits"] = sorted(
        f"{d}/{v}" for (d, v), s in video_splits.items() if len(s) > 1)

    # ---- per-dataset group/subject/content-group disjointness ----
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        by_split = {s: [r for r in sub if r["split"] == s] for s in A.SPLITS}
        grouping = policy["datasets"][ds]["split_grouping"]
        entry: dict = {"grouping": grouping, "group_key": policy["datasets"][ds]["group_key"]}

        def sets(field):
            return {s: {r[field] for r in by_split[s] if r[field] is not None} for s in A.SPLITS}

        def pairwise(d):
            return {f"{a}&{b}": sorted(d[a] & d[b])[:5]
                    for i, a in enumerate(A.SPLITS) for b in A.SPLITS[i + 1:]}

        entry["allocation_group_leakage"] = pairwise(sets("allocation_group_id"))
        entry["video_leakage"] = pairwise(sets("video_id"))
        entry["sample_leakage"] = pairwise(sets("sample_id"))
        if ds == "siwmv2":
            # Subject identity is unavailable for SiW; claiming subject-disjointness is forbidden.
            entry["subject_leakage"] = "N/A"
            entry["subject_leakage_reason"] = "trustworthy subject identity unavailable (Q-14)"
            entry["subject_id_global_all_null"] = all(r["subject_id_global"] is None for r in sub)
            entry["content_group_leakage"] = pairwise(sets("content_group_id"))
            groups_in = {}
            for r in sub:
                groups_in.setdefault(r["content_group_id"], set()).add(r["split"])
            entry["content_groups_spanning_splits"] = sorted(g for g, s in groups_in.items() if len(s) > 1)
            entry["fidelity"] = "canonical-video / content-group-disjoint"
            entry["subject_disjoint_claim"] = "FORBIDDEN"
        else:
            entry["subject_leakage"] = pairwise(sets("subject_id_global"))
            entry["fidelity"] = "subject-disjoint"
        rep["leakage"][ds] = entry

        # ---- counts ----
        rep["counts"][ds] = {
            s: {"usable_samples": len(by_split[s]),
                "canonical_videos": len({r["video_id"] for r in by_split[s]}),
                "allocation_groups": len({r["allocation_group_id"] for r in by_split[s]}),
                "live_videos": len({r["video_id"] for r in by_split[s] if r["label_binary"] == 0}),
                "spoof_videos": len({r["video_id"] for r in by_split[s] if r["label_binary"] == 1})}
            for s in A.SPLITS}
        rep["counts"][ds]["_total"] = {
            "usable_samples": len(sub), "canonical_videos": len({r["video_id"] for r in sub}),
            "allocation_groups": len({r["allocation_group_id"] for r in sub})}

    # ---- global checks ----
    def leaks(entry, key):
        v = entry.get(key)
        return sum(len(x) for x in v.values()) if isinstance(v, dict) else 0

    rep["checks"]["allocation_group_leakage_total"] = sum(
        leaks(rep["leakage"][ds], "allocation_group_leakage") for ds in DATASETS)
    rep["checks"]["video_leakage_total"] = sum(leaks(rep["leakage"][ds], "video_leakage") for ds in DATASETS)
    rep["checks"]["sample_leakage_total"] = sum(leaks(rep["leakage"][ds], "sample_leakage") for ds in DATASETS)
    rep["checks"]["subject_leakage_total_casia_msu"] = sum(
        leaks(rep["leakage"][ds], "subject_leakage") for ds in ("casia_fasd", "msu_mfsd"))
    rep["checks"]["content_group_leakage_total_siw"] = leaks(rep["leakage"]["siwmv2"], "content_group_leakage")
    rep["checks"]["every_dataset_in_every_split"] = all(
        rep["counts"][ds][s]["canonical_videos"] > 0 for ds in DATASETS for s in A.SPLITS)
    rep["ok"] = (rep["checks"]["required_columns_present"] and rep["checks"]["unique_sample_ids"]
                 and rep["checks"]["rows_equal_m2_complete"] and rep["checks"]["no_failed_row_present"]
                 and rep["checks"]["every_row_has_a_split"] and rep["checks"]["canonical_row_order"]
                 and not rep["leakage"]["videos_with_multiple_splits"]
                 and rep["checks"]["allocation_group_leakage_total"] == 0
                 and rep["checks"]["video_leakage_total"] == 0
                 and rep["checks"]["sample_leakage_total"] == 0
                 and rep["checks"]["subject_leakage_total_casia_msu"] == 0
                 and rep["checks"]["content_group_leakage_total_siw"] == 0
                 and not rep["leakage"]["siwmv2"]["content_groups_spanning_splits"]
                 and rep["checks"]["every_dataset_in_every_split"])
    return rep


def distributions(root: Path, manifest_path: Path | None = None) -> dict:
    """Canonical-video counts per dataset x split x category (the allocator's own unit)."""
    rows = pq.read_table(manifest_path or (root / SPLIT_MANIFEST)).to_pylist()
    seen, out = set(), {"binary": [], "attack_macro": [], "attack_raw": []}
    videos = []
    for r in rows:
        k = (r["dataset"], r["video_id"])
        if k in seen:
            continue
        seen.add(k)
        videos.append(r)
    for level, field in (("binary", "label_binary"), ("attack_macro", "attack_macro"),
                         ("attack_raw", "attack_raw")):
        cats = sorted({("(none)" if v[field] is None else str(v[field])) for v in videos})
        for ds in DATASETS:
            dsv = [v for v in videos if v["dataset"] == ds]
            for cat in cats:
                cv = [v for v in dsv if ("(none)" if v[field] is None else str(v[field])) == cat]
                if not cv:
                    continue
                row = {"dataset": ds, "category": cat, "total_videos": len(cv)}
                for s in A.SPLITS:
                    n = sum(1 for v in cv if v["split"] == s)
                    row[f"{s}_videos"] = n
                    row[f"{s}_pct"] = round(100.0 * n / len(cv), 4)
                    row[f"{s}_target_pct"] = A.PCT[s]
                    row[f"{s}_deviation_pp"] = round(100.0 * n / len(cv) - A.PCT[s], 4)
                out[level].append(row)
    return out


def failed_sample_accounting(root: Path, manifest_path: Path | None = None) -> list:
    """Where the canonical videos of the M2 FAILED samples ended up (provenance only)."""
    rows = pq.read_table(manifest_path or (root / SPLIT_MANIFEST)).to_pylist()
    split_of = {(r["dataset"], r["video_id"]): r["split"] for r in rows}
    acct = pq.read_table(root / "manifests/m2_sample_accounting.parquet").to_pylist()
    out = [{"sample_id": r["sample_id"], "dataset": r["dataset"], "video_id": r["video_id"],
            "frame_index": int(r["frame_index"]), "failure_reason": r["failure_reason"],
            "video_split": split_of.get((r["dataset"], r["video_id"]), "VIDEO_NOT_IN_SPLIT"),
            "in_split_manifest": "False"}
           for r in acct if r["final_status"] == "FAILED"]
    out.sort(key=lambda r: (r["dataset"], r["video_id"], r["frame_index"]))
    return out
