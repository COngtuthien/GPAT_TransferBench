"""M2 completion audit: cache integrity, manifest accounting, SCRFD report and padding audit.

    <m2 venv>/bin/python tools/m2b_audit.py --exec-config configs/execution/m2b_laptop_external_storage.yaml
        [--skip-blocks]     # index-level checks only (blocks are ~33 GiB to read back)

Writes:
  outputs/audit/M2_CACHE_INTEGRITY.json / .md
  manifests/m2_sample_accounting.parquet          (one row per frozen M1 sample, with provenance)
  outputs/audit/M2_CACHE_SHARD_INDEX.csv          (per-shard hashes; per-block rows stay in the cache)
  outputs/audit/M2_SCRFD_SUCCESS_REPORT.csv / .md
  outputs/audit/M2_PADDING_AUDIT.csv / .md

Nothing here can change a stored value: it only reads, hashes and counts. Detector thresholds and
the crop policy are never revisited on the basis of what these reports show.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.preprocess import contracts as C  # noqa: E402
from gpatbench.preprocess import logit_store as LS  # noqa: E402
from gpatbench.preprocess import m2b  # noqa: E402

AUDIT = ROOT / "outputs/audit"
GIB = 1024 ** 3


def load_field_index(field_root: Path) -> dict:
    """sample_id -> row, plus the per-shard metadata, for one cache field."""
    rows, shards = {}, {}
    for idx in sorted(field_root.glob("*.index.json")):
        meta = json.loads(idx.read_text())
        shards[meta["shard"]] = meta
        for r in meta["rows"]:
            rows.setdefault(r["sample_id"], []).append(r)
    return {"rows": rows, "shards": shards}


def audit_field(field_root: Path, field: str, check_blocks: bool) -> dict:
    idx = load_field_index(field_root)
    dup = sorted(s for s, v in idx["rows"].items() if len(v) > 1)
    rep = {"field": field, "root": str(field_root), "shards": len(idx["shards"]),
           "entries": len(idx["rows"]), "duplicate_sample_ids": dup,
           "missing_shard_files": [], "orphan_shard_files": [], "overlapping_blocks": [],
           "bad_block_hashes": [], "bad_shape_or_dtype": [], "non_finite": [],
           "shard_hash_failures": [], "bytes": 0}
    present = {p.name for p in field_root.glob("*.bin")}
    rep["orphan_shard_files"] = sorted(present - set(idx["shards"]))
    for name, meta in idx["shards"].items():
        path = field_root / name
        if not path.is_file():
            rep["missing_shard_files"].append(name)
            continue
        size = path.stat().st_size
        rep["bytes"] += size
        covered = sum(r["byte_length"] for r in meta["rows"])
        if covered != size:
            rep["missing_shard_files"].append(f"{name}: index covers {covered} of {size} bytes")
        spans = sorted((r["byte_offset"], r["byte_offset"] + r["byte_length"], r["sample_id"]) for r in meta["rows"])
        for (a0, a1, sa), (b0, _b1, sb) in zip(spans, spans[1:]):
            if b0 < a1:
                rep["overlapping_blocks"].append((sa, sb))
        if not check_blocks:
            continue
        blob_all = path.read_bytes()
        if hashlib.sha256(blob_all).hexdigest() != meta["shard_sha256"]:
            rep["shard_hash_failures"].append(name)
        for r in meta["rows"]:
            blob = blob_all[r["byte_offset"]:r["byte_offset"] + r["byte_length"]]
            if hashlib.sha256(blob).hexdigest() != r["block_sha256"]:
                rep["bad_block_hashes"].append(r["sample_id"])
                continue
            try:
                arr = LS.decode_block(blob) if field in m2b.COMPRESSED_FIELDS else np.load(
                    io.BytesIO(blob), allow_pickle=False)
            except Exception:
                rep["bad_block_hashes"].append(r["sample_id"])
                continue
            if list(arr.shape) != r["shape"] or arr.dtype.str != r["dtype"]:
                rep["bad_shape_or_dtype"].append(r["sample_id"])
            if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
                rep["non_finite"].append(r["sample_id"])
    rep["ok"] = not any(rep[k] for k in ("duplicate_sample_ids", "missing_shard_files", "orphan_shard_files",
                                         "overlapping_blocks", "bad_block_hashes", "bad_shape_or_dtype",
                                         "non_finite", "shard_hash_failures"))
    return rep, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-config", required=True)
    ap.add_argument("--skip-blocks", action="store_true")
    ap.add_argument("--sample-blocks", type=int, default=None,
                    help="verify blocks for a deterministic hash-selected subset of shards only")
    a = ap.parse_args()
    xcfg = yaml.safe_load((ROOT / a.exec_config).read_text())
    resolved = m2b.resolve_roots(xcfg)
    roots, runstate = resolved["roots"], resolved["runstate_root"]

    inv = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
    state = m2b.StateLog.load(runstate)
    by_id = {r["sample_id"]: r for r in inv}

    # ---------------- per-field cache integrity ----------------
    fields, indexes = {}, {}
    for f in m2b.ALL_FIELDS:
        froot = m2b.field_root(roots, f)
        rep, idx = audit_field(froot, f, check_blocks=not a.skip_blocks)
        fields[f], indexes[f] = rep, idx

    # ---------------- manifest accounting + per-sample provenance ----------------
    acc = {"expected": len(inv), "complete": 0, "failed": 0, "missing_state": 0}
    by_ds = collections.defaultdict(lambda: collections.Counter())
    failure_classes = collections.Counter()
    rows_out, orphan_cache_rows, missing_entries, face_missing = [], [], [], []
    cache_ids = {f: set(idx["rows"]) for f, idx in indexes.items()}
    for sid, r in by_id.items():
        st = state.get(sid)
        ds = r["dataset"]
        if st is None:
            acc["missing_state"] += 1
            by_ds[ds]["missing_state"] += 1
            rows_out.append({"sample_id": sid, "dataset": ds, "final_status": "MISSING_STATE"})
            continue
        status = st.get("state")
        by_ds[ds][status] += 1
        if status == "COMPLETE":
            acc["complete"] += 1
            for f in m2b.ALL_FIELDS:
                if sid not in cache_ids[f]:
                    missing_entries.append((sid, f))
            fp = Path(st["face_png_path"])
            if not fp.is_file():
                face_missing.append(sid)
        elif status == "FAILED":
            acc["failed"] += 1
            failure_classes[st.get("failure_reason", "UNCLASSIFIED").split(":")[0]] += 1
        rows_out.append({
            "sample_id": sid, "dataset": ds, "video_id": r["video_id"], "frame_index": r["frame_index"],
            "label_binary": r["label_binary"], "attack_raw": r["attack_raw"] or "",
            "route": st.get("route", ""), "final_status": status,
            "failure_reason": st.get("failure_reason", ""),
            "source_path": r["source_path"], "source_file_sha256": r["source_file_sha256"],
            "frame_png_path": st.get("frame_png_path", ""), "frame_png_sha256": st.get("frame_png_sha256", ""),
            "frame_hw": st.get("frame_hw", ""),
            "scrfd_applied": st.get("scrfd_applied", ""), "scrfd_status": st.get("scrfd_status", ""),
            "n_detections": st.get("n_detections", ""), "det_score": st.get("det_score", ""),
            "bbox": st.get("bbox", ""), "crop_requested": st.get("crop_requested", ""),
            "crop_source": st.get("crop_source", ""), "pad_left": st.get("pad_left", ""),
            "pad_top": st.get("pad_top", ""), "pad_right": st.get("pad_right", ""),
            "pad_bottom": st.get("pad_bottom", ""), "crop_side_px": st.get("crop_side_px", ""),
            "crop_pre_resize_hw": st.get("crop_pre_resize_hw", ""),
            "crop_touches_border": st.get("crop_touches_border", ""),
            "face_png_path": st.get("face_png_path", ""), "face_png_sha256": st.get("face_png_sha256", ""),
            "shard": st.get("shard", ""), "embedding_l2": st.get("embedding_l2", ""),
            "raw_norm": st.get("raw_norm", "")})
    for f, ids in cache_ids.items():
        orphan_cache_rows += [(sid, f) for sid in ids
                              if state.get(sid, {}).get("state") != "COMPLETE"]

    rows_out.sort(key=lambda r: (r["dataset"], r["sample_id"]))
    fieldnames = sorted({k for r in rows_out for k in r})
    import pyarrow as pa
    import pyarrow.parquet as pqw
    pqw.write_table(pa.table({c: pa.array([str(r.get(c, "")) for r in rows_out], pa.string()) for c in fieldnames}),
                    ROOT / "manifests/m2_sample_accounting.parquet", compression="zstd")

    # Per-shard hash summary small enough to live in Git; the per-block index stays with the cache.
    srows = []
    for f, idx in indexes.items():
        d = m2b.field_root(roots, f)
        for name, meta in sorted(idx["shards"].items()):
            ip = d / f"{name}.index.json"
            srows.append({"field": f, "role": m2b.ALL_FIELDS[f], "shard": name, "rows": len(meta["rows"]),
                          "bytes": (d / name).stat().st_size, "shard_sha256": meta["shard_sha256"],
                          "codec_id": meta["codec_id"],
                          "index_sha256": hashlib.sha256(ip.read_bytes()).hexdigest() if ip.is_file() else "",
                          "physical_dir": str(d)})
    srows.sort(key=lambda r: (r["field"], r["shard"]))
    if srows:
        with open(AUDIT / "M2_CACHE_SHARD_INDEX.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(srows)

    # ---------------- SCRFD success report ----------------
    scrfd_rows, scores, area_frac = [], collections.defaultdict(list), collections.defaultdict(list)
    for r in rows_out:
        if r.get("scrfd_applied") is not True and r.get("scrfd_applied") != "True":
            continue
        ds = r["dataset"]
        scrfd_rows.append(r)
        if r.get("scrfd_status") == "DETECTED":
            scores[ds].append(float(r["det_score"]))
            bb = json.loads(r["bbox"])
            fh_, fw = (int(x) for x in r["frame_hw"].split("x"))
            area_frac[ds].append(((bb[2] - bb[0]) * (bb[3] - bb[1])) / (fw * fh_))
    scrfd = {}
    for ds in sorted({r["dataset"] for r in scrfd_rows}):
        sub = [r for r in scrfd_rows if r["dataset"] == ds]
        det = [r for r in sub if r.get("scrfd_status") == "DETECTED"]
        nf = [r for r in sub if r.get("scrfd_status") == "SCRFD_NO_FACE"]
        other = [r for r in sub if r["final_status"] == "FAILED" and r.get("scrfd_status") != "SCRFD_NO_FACE"]
        # MSU/SiW rows that failed before the detector ever ran are not in the SCRFD denominator,
        # but they are counted here so no sample can quietly vanish between the two reports.
        not_reached = [r for r in rows_out if r["dataset"] == ds and r["final_status"] == "FAILED"
                       and r.get("scrfd_applied") not in (True, "True")]
        scrfd[ds] = {"applicable": len(sub), "success": len(det), "no_face": len(nf), "other_error": len(other),
                     "failed_before_detection": len(not_reached),
                     "success_pct": round(100.0 * len(det) / len(sub), 4) if sub else None,
                     "score_min": min(scores[ds]) if scores[ds] else None,
                     "score_p05": float(np.percentile(scores[ds], 5)) if scores[ds] else None,
                     "score_median": statistics.median(scores[ds]) if scores[ds] else None,
                     "score_max": max(scores[ds]) if scores[ds] else None,
                     "bbox_area_fraction_median": statistics.median(area_frac[ds]) if area_frac[ds] else None,
                     "bbox_area_fraction_min": min(area_frac[ds]) if area_frac[ds] else None,
                     "bbox_area_fraction_max": max(area_frac[ds]) if area_frac[ds] else None,
                     "padded_samples": sum(1 for r in det if r["crop_touches_border"] in (True, "True"))}
    casia_n = sum(1 for r in rows_out if r["dataset"] == "casia_fasd")
    with open(AUDIT / "M2_SCRFD_SUCCESS_REPORT.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["dataset", "applicable", "success", "no_face", "other_error", "failed_before_detection", "success_pct", "score_min",
                "score_p05", "score_median", "score_max", "bbox_area_fraction_min", "bbox_area_fraction_median",
                "bbox_area_fraction_max", "padded_samples"]
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for ds, v in scrfd.items():
            w.writerow({"dataset": ds, **{c: v.get(c) for c in cols[1:]}})
        w.writerow({"dataset": "casia_fasd", "applicable": 0, "success": "N/A", "no_face": "N/A",
                    "other_error": "N/A", "success_pct": "N/A"})

    # ---------------- padding audit ----------------
    pad = collections.defaultdict(lambda: collections.Counter())
    extremes = []
    for r in rows_out:
        if r.get("scrfd_status") != "DETECTED":
            continue
        ds = r["dataset"]
        sides = [s for s, k in (("left", "pad_left"), ("right", "pad_right"),
                                ("top", "pad_top"), ("bottom", "pad_bottom")) if int(r.get(k) or 0) > 0]
        pad[ds]["total_detected"] += 1
        if not sides:
            pad[ds]["no_padding"] += 1
        else:
            for s in sides:
                pad[ds][s] += 1
            pad[ds]["multiple_sides" if len(sides) > 1 else "single_side"] += 1
            side_px = int(r["crop_side_px"])
            frac = sum(int(r.get(k) or 0) for k in ("pad_left", "pad_right", "pad_top", "pad_bottom")) / side_px
            extremes.append((frac, r["sample_id"], ds, side_px,
                             [r["pad_left"], r["pad_top"], r["pad_right"], r["pad_bottom"]]))
    extremes.sort(reverse=True)
    with open(AUDIT / "M2_PADDING_AUDIT.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["dataset", "total_detected", "no_padding", "single_side", "multiple_sides",
                "left", "right", "top", "bottom"]
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for ds in sorted(pad):
            w.writerow({"dataset": ds, **{c: pad[ds].get(c, 0) for c in cols[1:]}})

    # ---------------- roll-up ----------------
    integrity = {
        "fields": fields,
        "blocks_verified": not a.skip_blocks,
        "accounting": {**acc, "by_dataset": {k: dict(v) for k, v in by_ds.items()}},
        "failure_classes": dict(failure_classes),
        "complete_samples_missing_cache_entry": missing_entries[:50],
        "n_complete_samples_missing_cache_entry": len(missing_entries),
        "orphan_cache_rows": orphan_cache_rows[:50], "n_orphan_cache_rows": len(orphan_cache_rows),
        "complete_samples_missing_face_file": face_missing[:50], "n_missing_face_files": len(face_missing),
        "scrfd": scrfd, "casia_scrfd": "N/A", "casia_samples": casia_n,
        "padding": {k: dict(v) for k, v in pad.items()},
        "padding_extremes_top10": [{"pad_fraction_of_side": round(f, 4), "sample_id": s, "dataset": d,
                                    "side_px": sp, "pads_LTRB": p} for f, s, d, sp, p in extremes[:10]],
        "cache_bytes": {f: v["bytes"] for f, v in fields.items()},
    }
    integrity["ok"] = (all(v["ok"] for v in fields.values())
                       and acc["missing_state"] == 0
                       and acc["complete"] + acc["failed"] == acc["expected"]
                       and not missing_entries and not orphan_cache_rows and not face_missing)
    (AUDIT / "M2_CACHE_INTEGRITY.json").write_text(json.dumps(integrity, indent=2, default=str) + "\n", encoding="utf-8")
    write_markdown(integrity)
    print(json.dumps({"integrity_ok": integrity["ok"], **acc,
                      "fields_ok": {f: v["ok"] for f, v in fields.items()},
                      "failure_classes": dict(failure_classes)}, indent=1))
    return 0 if integrity["ok"] else 1


def write_markdown(rep: dict) -> None:
    g = lambda b: f"{b / GIB:,.2f} GiB"                                      # noqa: E731
    acc, scrfd, pad = rep["accounting"], rep["scrfd"], rep["padding"]

    # ---- cache integrity ----
    lines = ["# M2 — Cache Integrity Audit", "",
             f"Overall: **{'PASS' if rep['ok'] else 'FAIL'}**. "
             f"Block-level verification: {'yes' if rep['blocks_verified'] else 'index-level only'}.", "",
             "| field | entries | shards | bytes | duplicates | orphan shards | bad block hashes | bad shape/dtype | non-finite | shard hash failures | ok |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f, v in rep["fields"].items():
        lines.append(f"| `{f}` | {v['entries']:,} | {v['shards']} | {g(v['bytes'])} | {len(v['duplicate_sample_ids'])} | "
                     f"{len(v['orphan_shard_files'])} | {len(v['bad_block_hashes'])} | {len(v['bad_shape_or_dtype'])} | "
                     f"{len(v['non_finite'])} | {len(v['shard_hash_failures'])} | {'PASS' if v['ok'] else 'FAIL'} |")
    lines += ["",
              "| accounting | value |", "|---|---|",
              f"| expected (frozen M1 inventory) | {acc['expected']:,} |",
              f"| COMPLETE | {acc['complete']:,} |",
              f"| FAILED | {acc['failed']:,} |",
              f"| missing state (unaccounted) | {acc['missing_state']:,} |",
              f"| COMPLETE samples missing a cache entry | {rep['n_complete_samples_missing_cache_entry']:,} |",
              f"| orphan cache rows (no COMPLETE sample) | {rep['n_orphan_cache_rows']:,} |",
              f"| COMPLETE samples missing a face file | {rep['n_missing_face_files']:,} |", "",
              "Per dataset:", "", "| dataset | " + " | ".join(sorted({k for v in acc["by_dataset"].values() for k in v})) + " |",
              "|---" * (1 + len({k for v in acc["by_dataset"].values() for k in v})) + "|"]
    states = sorted({k for v in acc["by_dataset"].values() for k in v})
    for ds, v in sorted(acc["by_dataset"].items()):
        lines.append(f"| {ds} | " + " | ".join(f"{v.get(s, 0):,}" for s in states) + " |")
    if rep["failure_classes"]:
        lines += ["", "Failure classes:", ""] + [f"- `{k}`: {v:,}" for k, v in sorted(rep["failure_classes"].items())]
    else:
        lines += ["", "No failures were recorded."]
    (AUDIT / "M2_CACHE_INTEGRITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- SCRFD report ----
    tot_app = sum(v["applicable"] for v in scrfd.values())
    tot_ok = sum(v["success"] for v in scrfd.values())
    lines = ["# M2 — SCRFD Success Report", "",
             "Frozen detector SCRFD_10G_KPS at the spec threshold 0.50, largest bbox, no fallback of any kind.",
             "**CASIA-FASD is N/A** and is excluded from every denominator: its route is the approved "
             "pre-cropped adaptation (DEV-011), so the detector is never run for it and it is never counted "
             "as a detector success.",
             "",
             "| dataset | applicable | success | no-face | other error | failed before detection | success % |",
             "|---|---|---|---|---|---|---|"]
    for ds, v in sorted(scrfd.items()):
        lines.append(f"| {ds} | {v['applicable']:,} | {v['success']:,} | {v['no_face']:,} | "
                     f"{v['other_error']:,} | {v['failed_before_detection']:,} | {v['success_pct']:.4f} % |")
    if tot_app:
        lines.append(f"| **all applicable** | **{tot_app:,}** | **{tot_ok:,}** | "
                     f"**{sum(v['no_face'] for v in scrfd.values()):,}** | "
                     f"**{sum(v['other_error'] for v in scrfd.values()):,}** | "
                     f"**{sum(v['failed_before_detection'] for v in scrfd.values()):,}** | "
                     f"**{100.0 * tot_ok / tot_app:.4f} %** |")
    lines.append("| casia_fasd | 0 (N/A) | N/A | N/A | N/A | N/A | N/A |")
    lines += ["", "## Detection-score and bbox distributions (descriptive only)", "",
              "These are reported for audit. The threshold and the crop policy are frozen and are **not** "
              "revisited on the basis of this table.", "",
              "| dataset | score min | score p05 | score median | score max | bbox area frac min | median | max | padded crops |",
              "|---|---|---|---|---|---|---|---|---|"]
    for ds, v in sorted(scrfd.items()):
        lines.append(f"| {ds} | {v['score_min']:.4f} | {v['score_p05']:.4f} | {v['score_median']:.4f} | "
                     f"{v['score_max']:.4f} | {v['bbox_area_fraction_min']:.5f} | "
                     f"{v['bbox_area_fraction_median']:.5f} | {v['bbox_area_fraction_max']:.5f} | "
                     f"{v['padded_samples']:,} |")
    (AUDIT / "M2_SCRFD_SUCCESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- padding audit ----
    lines = ["# M2 — Crop Padding Audit (Q-22 / DEV-016)", "",
             "Counts of detected crops whose requested 1.25x square extends past a frame edge and is "
             "therefore zero-padded on that side. The requested square is never shrunk or shifted, so a "
             "padded crop is still a true square before the 256 resize.",
             "",
             "**The crop policy is frozen and is not changed because of these frequencies.**", "",
             "| dataset | detected crops | no padding | single side | multiple sides | left | right | top | bottom |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds, v in sorted(pad.items()):
        lines.append(f"| {ds} | {v.get('total_detected', 0):,} | {v.get('no_padding', 0):,} | "
                     f"{v.get('single_side', 0):,} | {v.get('multiple_sides', 0):,} | {v.get('left', 0):,} | "
                     f"{v.get('right', 0):,} | {v.get('top', 0):,} | {v.get('bottom', 0):,} |")
    lines += ["", "CASIA-FASD performs no SCRFD crop, so it contributes no row here.", "",
              "## Most extreme padded samples (preserved for audit)", "",
              "| pad fraction of side | sample_id | dataset | side_px | pads L,T,R,B |", "|---|---|---|---|---|"]
    for e in rep["padding_extremes_top10"]:
        lines.append(f"| {e['pad_fraction_of_side']:.4f} | `{e['sample_id'][:16]}…` | {e['dataset']} | "
                     f"{e['side_px']:,} | {e['pads_LTRB']} |")
    (AUDIT / "M2_PADDING_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
