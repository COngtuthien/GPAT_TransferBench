"""Dataset-policy sanity audit + SiW protocol coverage (Q-17) + SiW exact-content groups.

Read-only on datasets and manifests. Produces:
  outputs/audit/SIW_PROTOCOL_COVERAGE_AUDIT.csv   per (protocol file, referenced name)
  outputs/audit/siw_content_groups.csv            per SiW canonical video: sha256-derived content_group_id
  outputs/audit/dataset_policy_sanity.json        §43 sanity facts (subject completeness, traceability, ...)

content_group_id = "siwmv2::sha256:<raw canonical video sha256>" (policy dataset_protocol_policy_v1).
It is NOT a person identity; it only prevents exact-content leakage across splits.
Run: .venv/bin/python tools/audit_dataset_policy.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
AUDIT = ROOT / "outputs/audit"


def content_group_id(sha256: str) -> str:
    return f"siwmv2::sha256:{sha256}"


def main() -> int:
    import audit_siw_resolution as siw
    siw.fetch()   # uses the git-ignored cache at the pinned commit; hashes re-recorded in the manifest
    videos = pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist()
    samples = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
    files = {(r["dataset"], r["rel_path"]): r for r in pq.read_table(ROOT / "manifests/raw_file_index.parquet").to_pylist()}
    amap = yaml.safe_load((ROOT / "configs/frozen/attack_map_v1.yaml").read_text())

    # ---- SiW content groups
    sv = sorted((v for v in videos if v["dataset"] == "siwmv2"), key=lambda r: r["video_id"])
    groups = collections.defaultdict(list)
    for v in sv:
        groups[files[("siwmv2", v["source_path"])]["sha256"]].append(v["video_id"])
    with open(AUDIT / "siw_content_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video_id", "raw_sha256", "content_group_id", "group_size", "attack_raw"])
        for v in sv:
            h = files[("siwmv2", v["source_path"])]["sha256"]
            w.writerow([v["video_id"], h, content_group_id(h), len(groups[h]), v["attack_raw"] or ""])

    # ---- Q-17 protocol coverage
    prefix_to_folder = {}
    for v in sv:
        m = json.loads(v["native_meta_json"])
        prefix_to_folder[m["file_prefix"]] = m["folder"]
    local_stems = {Path(v["source_path"]).stem for v in sv}
    cov_rows = []
    for rel in siw.LISTS:
        toks = (siw.CACHE / rel).read_text().split()
        cnt = collections.Counter(toks)
        for name in sorted(cnt):
            prefix = name.rsplit("_", 1)[0]
            present = name in local_stems
            if not present:
                cls = "MISSING_FROM_LOCAL_RELEASE"
            elif cnt[name] > 1:
                cls = "DUPLICATE_PROTOCOL_REFERENCE"
            else:
                cls = "PRESENT"
            cov_rows.append({"protocol_file": rel, "referenced_name": name, "local_present": "YES" if present else "NO",
                             "attack_type": prefix_to_folder.get(prefix, "UNKNOWN_PREFIX"), "references_in_file": cnt[name],
                             "classification": cls,
                             "notes": ("repeated for live/spoof balancing (official README)" if cnt[name] > 1 else "")
                                      + ("; not in local release" if not present else "")})
    with open(AUDIT / "SIW_PROTOCOL_COVERAGE_AUDIT.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(cov_rows)
    missing = sorted({r["referenced_name"] for r in cov_rows if r["local_present"] == "NO"})
    miss_by_type = collections.Counter({r["referenced_name"]: r["attack_type"] for r in cov_rows
                                        if r["local_present"] == "NO"}.values())   # distinct names per type

    # ---- §43 sanity
    by_ds = collections.defaultdict(list)
    for v in videos:
        by_ds[v["dataset"]].append(v)
    vid_keys = {(v["dataset"], v["video_id"]) for v in videos}
    mapped = {ds: set(t) for ds, t in (amap.get("mapped") or {}).items()}
    raw_tokens = collections.Counter((v["dataset"], v["attack_raw"]) for v in videos if v["attack_raw"] is not None)
    sanity = {
        "casia_subjects_complete": all(v["subject_id_raw"] is not None for v in by_ds["casia_fasd"]),
        "casia_subject_count": len({v["subject_id_global"] for v in by_ds["casia_fasd"]}),
        "msu_subjects_complete": all(v["subject_id_raw"] is not None for v in by_ds["msu_mfsd"]),
        "msu_subject_count": len({v["subject_id_global"] for v in by_ds["msu_mfsd"]}),
        "siw_subject_ids_all_null": all(v["subject_id_raw"] is None and v["subject_id_global"] is None for v in by_ds["siwmv2"]),
        "siw_subject_status_values": sorted({v["subject_id_status"] for v in by_ds["siwmv2"]}),
        "all_samples_trace_to_one_canonical_video": all((s["dataset"], s["video_id"]) in vid_keys for s in samples),
        "siw_samples": sum(1 for s in samples if s["dataset"] == "siwmv2"),
        "siw_videos": len(sv), "siw_content_groups": len(groups),
        "siw_multi_video_groups": sorted([sorted(g) for g in groups.values() if len(g) > 1]),
        "cross_video_exact_duplicates_casia_msu": _cross_dups(videos, files, samples),
        "attack_raw_tokens": {f"{ds}:{t}": {"videos": n, "mapped": t in mapped.get(ds, set())} for (ds, t), n in sorted(raw_tokens.items())},
        "unmapped_tokens": sorted(f"{ds}:{t}" for (ds, t) in raw_tokens if t not in mapped.get(ds, set())),
        "q17": {"protocol_files": len(siw.LISTS), "distinct_referenced": len({r["referenced_name"] for r in cov_rows}),
                "missing_from_local": len(missing), "missing_by_type": dict(miss_by_type),
                "missing_rows_all_files": sum(1 for r in cov_rows if r["local_present"] == "NO"),
                "local_videos_not_referenced": sorted(local_stems - {r["referenced_name"] for r in cov_rows})},
    }
    (AUDIT / "dataset_policy_sanity.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in sanity.items() if k != "attack_raw_tokens"}, indent=1))
    return 0


def _cross_dups(videos, files, samples):
    """Exact-byte duplicates spanning different canonical videos, per dataset (CASIA via frame files)."""
    out = {}
    for ds in ("casia_fasd", "msu_mfsd"):
        by_hash = collections.defaultdict(set)
        for (d, rel), r in files.items():
            if d == ds and r["category"].startswith("CANONICAL") and r["video_id"]:
                by_hash[r["sha256"]].add(r["video_id"])
        out[ds] = sum(1 for v in by_hash.values() if len(v) > 1)
    return out


if __name__ == "__main__":
    sys.exit(main())
