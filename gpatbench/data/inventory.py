"""M1 dataset inventory (spec §3.1-§3.4, §25 M1 gate).

Deterministic outputs (no timestamps inside): manifests/*.parquet and outputs/audit/dataset_*.csv,
unmapped_attack_tokens.csv, dataset_report.html. Run-specific facts (times, git state, hashes of
outputs) go to outputs/audit/m1_inventory_run.json.

Raw data policy: every raw file is opened read-only; the file-system snapshot (size, mtime_ns)
is compared before/after the run and any change is a hard failure.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import json
import os
import zipfile
import zlib
import multiprocessing as mp
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import casia_fasd, msu_mfsd, siwmv2
from .base import (ATTACK_MACROS, CANONICAL_FRAME_IMAGE, CANONICAL_VIDEO, DATASETS,
                   DERIVED_AUGMENTATION_COPY, FILE_CATEGORIES, SUBJECT_MISSING, VideoRecord,
                   sample_id, select_frame_indices)
from .frames import decoder_info, probe_image_sequence, probe_video

ADAPTERS = {"casia_fasd": casia_fasd, "msu_mfsd": msu_mfsd, "siwmv2": siwmv2}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_COLUMNS = {"split"}  # M3 owns the benchmark split


class InventoryError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- config
def load_config(path: str | os.PathLike) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if set(cfg["datasets"]) != set(DATASETS):
        raise InventoryError(f"config must define exactly {DATASETS}")
    pos = cfg["frame_sampling"]["positions"]
    if len(pos) != cfg["frame_sampling"]["frames_per_video"]:
        raise InventoryError("positions length != frames_per_video")
    return cfg


def load_attack_map(path: str | os.PathLike) -> dict:
    amap = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for ds, toks in (amap.get("mapped") or {}).items():
        for tok, ent in toks.items():
            if ent["attack_macro"] not in ATTACK_MACROS or ent["attack_macro"] == "live":
                raise InventoryError(f"attack map {ds}:{tok} has invalid macro {ent['attack_macro']!r}")
    return amap


# ----------------------------------------------------------------------------- workers
def _hash_file(args):
    root, rel = args
    h, crc = hashlib.sha256(), 0
    try:
        with open(Path(root) / rel, "rb") as f:  # read-only
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
                crc = zlib.crc32(block, crc)
        return rel, h.hexdigest(), f"{crc & 0xFFFFFFFF:08x}", True, None
    except OSError as e:
        return rel, None, None, False, f"{type(e).__name__}: {e}"


def _probe(args):
    root, kind, key, payload = args
    if kind == "image_sequence":
        return key, probe_image_sequence(root, payload)
    return key, probe_video(root, payload)


def _derived_check(args):
    import cv2
    import numpy as np
    root, rel_derived, rel_canon, aug = args
    a = cv2.imdecode(np.fromfile(Path(root) / rel_canon, np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.fromfile(Path(root) / rel_derived, np.uint8), cv2.IMREAD_COLOR)
    if a is None or b is None or a.shape != b.shape:
        return aug, None, None, None
    return (aug, bool(np.array_equal(b, a[:, ::-1])), bool(np.array_equal(a, b)),
            float(b.astype(np.float64).mean() - a.astype(np.float64).mean()))


# ----------------------------------------------------------------------------- helpers
def _walk(root: Path) -> tuple[list[str], list[str], list[str]]:
    files, dirs, links = [], [], []
    for dp, dn, fn in os.walk(root, followlinks=False):
        dn.sort()
        for d in dn:
            p = Path(dp) / d
            (links if p.is_symlink() else dirs).append(p.relative_to(root).as_posix())
        for f in sorted(fn):
            p = Path(dp) / f
            (links if p.is_symlink() else files).append(p.relative_to(root).as_posix())
    return sorted(files), sorted(dirs), sorted(links)


def _snapshot(root: Path, rels: list[str]) -> dict[str, tuple[int, int]]:
    out = {}
    for r in rels:
        st = (root / r).stat()
        out[r] = (st.st_size, st.st_mtime_ns)
    return out


def _write_parquet(rows: list[dict], schema: pa.Schema, path: Path) -> None:
    if FORBIDDEN_COLUMNS & set(schema.names):
        raise InventoryError(f"forbidden column(s) {FORBIDDEN_COLUMNS & set(schema.names)} (M3 owns the split)")
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=9, write_statistics=True)


def _write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


SAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset", pa.string()),
    ("subject_id_raw", pa.string()), ("subject_id_global", pa.string()), ("subject_id_status", pa.string()),
    ("video_id", pa.string()), ("frame_index", pa.int32()), ("sample_ordinal", pa.int8()),
    ("label_binary", pa.int8()), ("attack_raw", pa.string()), ("attack_macro", pa.string()),
    ("attack_map_status", pa.string()), ("style_id", pa.string()), ("source_path", pa.string()),
    ("source_file_sha256", pa.string()), ("sha256", pa.string()), ("sha256_kind", pa.string()),
    ("label_conflict", pa.string()), ("native_protocol_split", pa.string()),
])
VIDEO_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("video_id", pa.string()),
    ("subject_id_raw", pa.string()), ("subject_id_global", pa.string()), ("subject_id_status", pa.string()),
    ("subject_id_source", pa.string()), ("label_binary", pa.int8()), ("label_source", pa.string()),
    ("label_conflict", pa.string()), ("attack_raw", pa.string()), ("attack_macro", pa.string()),
    ("attack_map_status", pa.string()), ("style_id", pa.string()), ("native_protocol_split", pa.string()),
    ("native_meta_json", pa.string()), ("source_kind", pa.string()), ("source_path", pa.string()),
    ("source_file_sha256", pa.string()), ("n_source_files", pa.int32()), ("declared_frames", pa.int32()),
    ("n_valid_frames", pa.int32()), ("n_invalid_frames", pa.int32()), ("first_valid_index", pa.int32()),
    ("last_valid_index", pa.int32()), ("decode_status", pa.string()), ("resumed_after_failure", pa.bool_()),
    ("frame_sizes", pa.string()), ("fps", pa.float64()), ("n_sampled", pa.int8()),
    ("sampled_frame_indices", pa.string()), ("collision_resolution_used", pa.bool_()),
    ("decoder_error_lines", pa.int32()), ("frames_beyond_declared", pa.bool_()),
])
FILE_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("rel_path", pa.string()), ("size_bytes", pa.int64()), ("category", pa.string()),
    ("video_id", pa.string()), ("frame_index", pa.int32()), ("readable", pa.bool_()), ("read_error", pa.string()),
    ("sha256", pa.string()), ("crc32", pa.string()),
])


# ----------------------------------------------------------------------------- main
def run_inventory(config_path: str, workers: int | None = None, project_root: Path = PROJECT_ROOT,
                  make_report: bool = True) -> dict:
    t0 = dt.datetime.now(dt.timezone.utc)
    cfg = load_config(config_path)
    amap = load_attack_map(project_root / cfg["attack_map"])
    positions = [float(p) for p in cfg["frame_sampling"]["positions"]]
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    out = {k: project_root / v for k, v in cfg["outputs"].items()}

    file_rows, video_rows, sample_rows, issues = [], [], [], []
    archive_rows, dup_rows, derived_stats = [], [], {}
    snapshots, roots, listings = {}, {}, {}

    def issue(ds, severity, code, obj, detail):
        issues.append({"dataset": ds, "severity": severity, "code": code, "object": obj, "detail": detail})

    # forkserver: avoid fork() of a multi-threaded parent (pyarrow threads)
    with mp.get_context("forkserver").Pool(workers) as pool:
        for ds in DATASETS:
            dcfg = cfg["datasets"][ds]
            root = Path(dcfg["root"])
            if not root.is_dir():
                raise InventoryError(f"{ds}: selected root does not exist: {root}")
            roots[ds] = str(root)
            adapter = ADAPTERS[ds]
            files, dirs, links = _walk(root)
            listings[ds] = sorted(os.listdir(root))
            for l in links:
                issue(ds, "ERROR", "SYMLINK_UNDER_ROOT", l, "symlinks are not followed")
            # case-insensitive collisions among files and directories
            groups = collections.defaultdict(set)
            for p in files + dirs:
                groups[p.lower()].add(p)
            for g in sorted(v for v in groups.values() if len(v) > 1):
                issue(ds, "WARNING", "CASE_INSENSITIVE_PATH_COLLISION", " | ".join(sorted(g)), "paths differ only by case")
            snapshots[ds] = _snapshot(root, files)

            classified = [(rel, adapter.classify(rel)) for rel in files]  # raises on unclassified
            hashes = {r[0]: r for r in pool.imap(_hash_file, [(str(root), rel) for rel in files], chunksize=64)}
            for rel, fc in classified:
                _, sha, crc, ok, err = hashes[rel]
                file_rows.append({"dataset": ds, "rel_path": rel, "size_bytes": snapshots[ds][rel][0],
                                  "category": fc.category, "video_id": fc.video_key, "frame_index": fc.frame_index,
                                  "readable": ok, "read_error": err, "sha256": sha, "crc32": crc})
                if not ok:
                    issue(ds, "ERROR", "UNREADABLE_FILE", rel, err)

            videos: list[VideoRecord] = (adapter.build_videos(classified, root) if ds == "msu_mfsd"
                                         else adapter.build_videos(classified))
            ids = [v.video_id for v in videos]
            if len(ids) != len(set(ids)):
                raise InventoryError(f"{ds}: duplicate canonical video_id")

            # CASIA derived-copy evidence (measured on every derived file)
            if ds == "casia_fasd":
                jobs = []
                for rel, fc in classified:
                    if fc.category == DERIVED_AUGMENTATION_COPY:
                        d, name = rel.rsplit("/", 1)
                        jobs.append((str(root), rel, f"{d}/{name[1:]}", name[0]))
                stats = collections.defaultdict(lambda: {"n": 0, "exact_hflip": 0, "exact_equal": 0,
                                                         "unpaired": 0, "mean_delta_min": None, "mean_delta_max": None})
                for aug, hflip, equal, delta in pool.imap(_derived_check, jobs, chunksize=64):
                    s = stats[aug]
                    s["n"] += 1
                    if hflip is None:
                        s["unpaired"] += 1
                        continue
                    s["exact_hflip"] += hflip
                    s["exact_equal"] += equal
                    s["mean_delta_min"] = delta if s["mean_delta_min"] is None else min(s["mean_delta_min"], delta)
                    s["mean_delta_max"] = delta if s["mean_delta_max"] is None else max(s["mean_delta_max"], delta)
                derived_stats[ds] = {k: dict(v) for k, v in sorted(stats.items())}

            probe_jobs = [(str(root), v.source_kind, v.video_id,
                           v.frame_files if v.source_kind == "image_sequence" else v.source_path) for v in videos]
            probes = dict(pool.imap(_probe, probe_jobs, chunksize=1))

            mapped = (amap.get("mapped") or {}).get(ds, {})
            for v in videos:
                pr = probes[v.video_id]
                if v.label_binary == 0:
                    macro, mstatus = "live", "LIVE"
                elif v.attack_raw in mapped:
                    macro, mstatus = mapped[v.attack_raw]["attack_macro"], "MAPPED"
                else:
                    macro, mstatus = "other_spoof", "UNMAPPED_OTHER_SPOOF"
                style = f"{ds}::{v.attack_raw}" if v.attack_raw is not None else None
                chosen, fallback = select_frame_indices(pr["valid_indices"], positions)
                src_sha = hashes[v.source_path][1] if v.source_kind == "video_file" else None
                vi = pr["valid_indices"]
                video_rows.append({
                    "dataset": ds, "video_id": v.video_id, "subject_id_raw": v.subject_id_raw,
                    "subject_id_global": v.subject_id_global, "subject_id_status": v.subject_id_status,
                    "subject_id_source": v.subject_id_source, "label_binary": v.label_binary,
                    "label_source": v.label_source, "label_conflict": v.label_conflict, "attack_raw": v.attack_raw,
                    "attack_macro": macro, "attack_map_status": mstatus, "style_id": style,
                    "native_protocol_split": v.native_protocol_split,
                    "native_meta_json": json.dumps(v.native_meta, sort_keys=True, separators=(",", ":")),
                    "source_kind": v.source_kind, "source_path": v.source_path, "source_file_sha256": src_sha,
                    "n_source_files": len(v.frame_files) if v.source_kind == "image_sequence" else 1,
                    "declared_frames": pr["declared_frames"], "n_valid_frames": len(vi),
                    "n_invalid_frames": len(pr["invalid_indices"]),
                    "first_valid_index": vi[0] if vi else None, "last_valid_index": vi[-1] if vi else None,
                    "decode_status": pr["decode_status"], "resumed_after_failure": pr["resumed_after_failure"],
                    "frame_sizes": ";".join(pr["frame_sizes"]), "fps": pr["fps"], "n_sampled": len(chosen),
                    "sampled_frame_indices": json.dumps(chosen, separators=(",", ":")),
                    "collision_resolution_used": fallback,
                    "decoder_error_lines": pr["decoder_error_lines"],
                    "frames_beyond_declared": pr["frames_beyond_declared"],
                })
                if pr["decode_status"] not in ("OK",):
                    sev = ("ERROR" if pr["decode_status"] in ("OPEN_FAILED", "NO_DECODABLE_FRAMES", "DECLARED_COUNT_UNAVAILABLE")
                           else "WARNING" if pr["decode_status"] == "SOME_FRAMES_UNDECODABLE" else "INFO")
                    issue(ds, sev, f"DECODE_{pr['decode_status']}", v.video_id,
                          f"declared={pr['declared_frames']} valid={len(vi)} invalid={len(pr['invalid_indices'])}"
                          + (f" invalid_positions={pr['invalid_indices'][:20]}" if pr["invalid_indices"] else ""))
                if pr["frames_beyond_declared"]:
                    issue(ds, "WARNING", "FRAMES_BEYOND_DECLARED", v.video_id,
                          f"a read succeeded after declared={pr['declared_frames']} frames; extra frames are not sampled")
                if v.native_meta.get("folder_label_overridden"):
                    issue(ds, "INFO", "FOLDER_LABEL_OVERRIDDEN", v.video_id, v.label_source)
                if pr["decoder_error_lines"]:
                    issue(ds, "WARNING", "DECODER_ERRORS_REPORTED", v.video_id,
                          f"{pr['decoder_error_lines']} FFmpeg stderr lines; frames may be error-concealed; e.g. "
                          + " || ".join(pr["decoder_error_sample"][:3]))
                if v.source_kind == "image_sequence" and vi and vi[-1] - vi[0] + 1 != len(vi):
                    missing = sorted(set(range(vi[0], vi[-1] + 1)) - set(vi))
                    issue(ds, "INFO", "FRAME_INDEX_GAP", v.video_id, f"missing indices {missing[:20]}")
                if len(chosen) < len(positions):
                    issue(ds, "WARNING", "FEWER_THAN_8_VALID_FRAMES", v.video_id, f"valid={len(vi)}")
                if fallback:
                    issue(ds, "INFO", "SAMPLING_COLLISION_RESOLUTION_USED", v.video_id, json.dumps(chosen))
                if v.label_conflict:
                    issue(ds, "WARNING", "LABEL_CONFLICT", v.video_id, v.label_conflict)
                for k, fi in enumerate(chosen):
                    if v.source_kind == "image_sequence":
                        rel = v.frame_files[fi]
                        fsha, sha, kind = hashes[rel][1], hashes[rel][1], "original_frame_bytes"
                    else:
                        rel, fsha, sha, kind = v.source_path, src_sha, None, "PENDING_M2_CANONICAL_PNG"
                    sample_rows.append({
                        "sample_id": sample_id(ds, v.video_id, fi), "dataset": ds,
                        "subject_id_raw": v.subject_id_raw, "subject_id_global": v.subject_id_global,
                        "subject_id_status": v.subject_id_status, "video_id": v.video_id, "frame_index": fi,
                        "sample_ordinal": k, "label_binary": v.label_binary, "attack_raw": v.attack_raw,
                        "attack_macro": macro, "attack_map_status": mstatus, "style_id": style,
                        "source_path": rel, "source_file_sha256": fsha, "sha256": sha, "sha256_kind": kind,
                        "label_conflict": v.label_conflict, "native_protocol_split": v.native_protocol_split,
                    })
            n_missing = sum(v.subject_id_status == SUBJECT_MISSING for v in videos)
            if n_missing:
                issue(ds, "ERROR", "MISSING_SUBJECT_ID", f"{n_missing} videos",
                      "subject identity not recoverable from local files; M3 main split BLOCKED_BY_MISSING_SUBJECT_ID")

            # archive cross-check (central directory only: sizes + CRC32)
            by_rel = {r["rel_path"]: r for r in file_rows if r["dataset"] == ds}
            for arc in dcfg.get("archives_crosscheck", []) or []:
                ap, strip = Path(arc["path"]), arc.get("strip_prefix", "")
                row = {"dataset": ds, "archive": str(ap), "archive_size_bytes": ap.stat().st_size, "entries": 0,
                       "matched_size_crc": 0, "size_mismatch": 0, "crc_mismatch": 0, "missing_in_extracted": 0,
                       "extracted_not_in_archive": 0}
                seen = set()
                with zipfile.ZipFile(ap) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        row["entries"] += 1
                        rel = zi.filename[len(strip):] if zi.filename.startswith(strip) else None
                        r = by_rel.get(rel)
                        if r is None:
                            row["missing_in_extracted"] += 1
                            continue
                        seen.add(rel)
                        if r["size_bytes"] != zi.file_size:
                            row["size_mismatch"] += 1
                        elif r["crc32"] != f"{zi.CRC & 0xFFFFFFFF:08x}":
                            row["crc_mismatch"] += 1
                        else:
                            row["matched_size_crc"] += 1
                row["extracted_not_in_archive"] = len(set(by_rel) - seen)
                archive_rows.append(row)
                if row["size_mismatch"] or row["crc_mismatch"] or row["missing_in_extracted"]:
                    issue(ds, "WARNING", "ARCHIVE_EXTRACTED_MISMATCH", str(ap), json.dumps(row))

        # exact-byte duplicates among canonical sample-bearing files (within and across datasets)
        by_hash = collections.defaultdict(list)
        for r in file_rows:
            if r["category"] in (CANONICAL_VIDEO, CANONICAL_FRAME_IMAGE) and r["sha256"]:
                by_hash[r["sha256"]].append(r)
        sampled = {(r["dataset"], r["source_path"]) for r in sample_rows}
        for h, rows in sorted(by_hash.items()):
            if len(rows) > 1:
                paths = sorted(f"{r['dataset']}:{r['rel_path']}" for r in rows)
                dsets = sorted({r["dataset"] for r in rows})
                vids = sorted({(r["dataset"], r["video_id"]) for r in rows})
                n_sampled = sum((r["dataset"], r["rel_path"]) in sampled for r in rows)
                dup_rows.append({"sha256": h, "n_files": len(rows), "datasets": ";".join(dsets),
                                 "n_distinct_videos": len(vids), "n_sampled": n_sampled, "paths": " | ".join(paths)})
                if len(vids) > 1:
                    issue(";".join(dsets), "WARNING", "EXACT_DUPLICATE_ACROSS_VIDEOS", " | ".join(paths),
                          f"sha256={h}; distinct canonical videos={len(vids)}; no merge/delete performed")
                else:
                    issue(dsets[0], "INFO", "IDENTICAL_FRAMES_WITHIN_VIDEO", " | ".join(paths),
                          f"sha256={h}; sampled among them={n_sampled}")

    # ------------------------------------------------------------------ validation (hard gates)
    sids = [r["sample_id"] for r in sample_rows]
    if len(sids) != len(set(sids)):
        raise InventoryError("duplicate sample_id — M1 hard failure")
    for ds in DATASETS:  # raw immutability
        root = Path(roots[ds])
        after = _snapshot(root, list(snapshots[ds]))
        if after != snapshots[ds] or _walk(root)[0] != sorted(snapshots[ds]):
            raise InventoryError(f"{ds}: raw data changed during inventory")

    key = lambda r: (r["dataset"], r["video_id"])
    file_rows.sort(key=lambda r: (r["dataset"], r["rel_path"]))
    video_rows.sort(key=key)
    sample_rows.sort(key=lambda r: (r["dataset"], r["video_id"], r["frame_index"]))
    issues.sort(key=lambda r: (r["dataset"], r["severity"], r["code"], r["object"]))
    _write_parquet(sample_rows, SAMPLE_SCHEMA, out["inventory"])
    _write_parquet(video_rows, VIDEO_SCHEMA, out["inventory_videos"])
    _write_parquet(file_rows, FILE_SCHEMA, out["raw_file_index"])

    # ------------------------------------------------------------------ audit tables
    summary = []
    for ds in DATASETS:
        fr = [r for r in file_rows if r["dataset"] == ds]
        vr = [r for r in video_rows if r["dataset"] == ds]
        cat = collections.Counter(r["category"] for r in fr)
        row = {"dataset": ds, "selected_root": roots[ds], "indexed_files": len(fr),
               "total_bytes": sum(r["size_bytes"] for r in fr),
               "unreadable_files": sum(not r["readable"] for r in fr),
               **{f"files_{c.lower()}": cat.get(c, 0) for c in FILE_CATEGORIES},
               "canonical_videos": len(vr), "live_videos": sum(r["label_binary"] == 0 for r in vr),
               "spoof_videos": sum(r["label_binary"] == 1 for r in vr),
               "videos_with_subject_id": sum(r["subject_id_raw"] is not None for r in vr),
               "videos_missing_subject_id": sum(r["subject_id_raw"] is None for r in vr),
               "unique_subjects": len({r["subject_id_global"] for r in vr if r["subject_id_global"]}),
               "unique_subjects_live": len({r["subject_id_global"] for r in vr if r["subject_id_global"] and r["label_binary"] == 0}),
               "unique_subjects_spoof": len({r["subject_id_global"] for r in vr if r["subject_id_global"] and r["label_binary"] == 1}),
               "attack_raw_types": len({r["attack_raw"] for r in vr if r["attack_raw"] is not None}),
               "attack_macro_types_spoof": len({r["attack_macro"] for r in vr if r["label_binary"] == 1}),
               "unmapped_attack_tokens": len({r["attack_raw"] for r in vr if r["attack_map_status"] == "UNMAPPED_OTHER_SPOOF"}),
               "label_conflict_videos": sum(r["label_conflict"] is not None for r in vr),
               "videos_decode_not_ok": sum(r["decode_status"] != "OK" for r in vr),
               "videos_with_decoder_errors": sum(r["decoder_error_lines"] > 0 for r in vr),
               "videos_frames_beyond_declared": sum(bool(r["frames_beyond_declared"]) for r in vr),
               "videos_folder_label_overridden": sum(json.loads(r["native_meta_json"]).get("folder_label_overridden", False) for r in vr),
               "videos_lt_8_valid_frames": sum(r["n_sampled"] < len(positions) for r in vr),
               "sampled_frame_rows": sum(r["n_sampled"] for r in vr),
               "valid_frames_total": sum(r["n_valid_frames"] for r in vr)}
        summary.append(row)
    _write_csv(summary, list(summary[0]), out["summary_csv"])

    cov = collections.defaultdict(lambda: {"videos": 0, "subjects": set(), "samples": 0})
    for r in video_rows:
        k = (r["dataset"], r["label_binary"], r["attack_raw"] or "", r["attack_macro"], r["attack_map_status"])
        cov[k]["videos"] += 1
        cov[k]["samples"] += r["n_sampled"]
        if r["subject_id_global"]:
            cov[k]["subjects"].add(r["subject_id_global"])
    cov_rows = [{"dataset": k[0], "label_binary": k[1], "attack_raw": k[2], "attack_macro": k[3],
                 "attack_map_status": k[4], "videos": v["videos"],
                 "subjects_recovered": len(v["subjects"]), "sampled_frame_rows": v["samples"]}
                for k, v in sorted(cov.items())]
    _write_csv(cov_rows, list(cov_rows[0]), out["attack_coverage_csv"])

    pending = amap.get("unmapped_pending_approval") or {}
    un_rows = []
    for (ds, tok), n in sorted(collections.Counter((r["dataset"], r["attack_raw"]) for r in video_rows
                                                   if r["attack_map_status"] == "UNMAPPED_OTHER_SPOOF").items()):
        p = (pending.get(ds) or {}).get(tok) or {}
        un_rows.append({"dataset": ds, "attack_raw": tok, "videos": n, "assigned_attack_macro": "other_spoof",
                        "proposed_attack_macro": p.get("proposed_attack_macro"), "proposal_basis": p.get("basis"),
                        "status": "PENDING_OWNER_APPROVAL", "native_full_track": "BLOCKED"})
    _write_csv(un_rows, ["dataset", "attack_raw", "videos", "assigned_attack_macro", "proposed_attack_macro",
                         "proposal_basis", "status", "native_full_track"], out["unmapped_csv"])
    _write_csv(issues, ["dataset", "severity", "code", "object", "detail"], out["issues_csv"])
    _write_csv(archive_rows, ["dataset", "archive", "archive_size_bytes", "entries", "matched_size_crc",
                              "size_mismatch", "crc_mismatch", "missing_in_extracted", "extracted_not_in_archive"],
               out["archive_csv"])
    _write_csv(dup_rows, ["sha256", "n_files", "datasets", "n_distinct_videos", "n_sampled", "paths"], out["duplicates_csv"])

    facts = {"roots": roots, "root_listings": listings, "derived_copy_checks": derived_stats,
             "decoder": decoder_info(), "positions": positions, "workers": workers,
             "attack_map_status": amap.get("status")}
    (out["facts_json"]).write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if make_report:
        from .report import write_report
        write_report(cfg, project_root)

    run = {"started_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "config": str(config_path), "config_sha256": _sha(Path(config_path)),
           "attack_map_sha256": _sha(project_root / cfg["attack_map"]),
           "outputs_sha256": {k: _sha(p) for k, p in sorted(out.items()) if p.exists() and k != "run_json"},
           "counts": {"files": len(file_rows), "videos": len(video_rows), "samples": len(sample_rows),
                      "issues": len(issues)},
           "raw_data_unchanged": True}
    return run
