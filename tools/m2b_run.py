"""M2B full preprocessing runner (frozen contract, external storage, resumable).

    <m2 venv>/bin/python tools/m2b_run.py \
        --exec-config configs/execution/m2b_laptop_external_storage.yaml \
        [--worker 0 --n-workers 2] [--limit-shards N] [--dry-run]

Scientific behaviour comes entirely from `configs/frozen/preprocess_v1.yaml` and the frozen
adapters; this script only drives them. It refuses to start unless the three frozen auxiliary
weights hash-match `models/registry.yaml`.

Each worker takes the shards where `shard_index % n_workers == worker`. Workers never share a
shard, a state log or an output file, and every sample is computed independently with the frozen
determinism settings (torch 4 intra-op threads, 1 ORT thread), so the number of workers cannot
change a stored value. The deterministic final validation re-proves that afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.preprocess import contracts as C  # noqa: E402
from gpatbench.preprocess import logit_store as LS  # noqa: E402
from gpatbench.preprocess import m2b  # noqa: E402
from gpatbench.preprocess.frames import read_image_frame, read_video_frames  # noqa: E402

MODEL_CACHE = Path("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache")


def verify_models(reg: dict, cfg: dict) -> dict:
    want = {
        "scrfd": (Path(cfg["scrfd"]["model_path"]), reg["scrfd"]["weight_sha256"]),
        "facexformer": (MODEL_CACHE / "face_geometry/ckpts/model.pt", reg["facexformer"]["weight_sha256"]),
        "adaface_ir50": (Path(cfg["adaface"]["checkpoint_path"]), reg["adaface_ir50"]["weight_sha256"]),
    }
    out = {}
    for k, (path, expected) in want.items():
        got = m2b.sha256_file(path)
        if got != expected:
            raise SystemExit(f"STOP: {k} weight hash mismatch: {got} != {expected}")
        out[k] = {"path": str(path), "sha256": got}
    return out


def free_bytes(path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-config", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--limit-shards", type=int, default=None, help="process at most N shards (smoke/debug)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    xcfg = yaml.safe_load((ROOT / a.exec_config).read_text())
    cfg = yaml.safe_load((ROOT / xcfg["scientific_contract"]).read_text())
    reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]
    if m2b.sha256_file(ROOT / xcfg["scientific_contract"]) != xcfg["scientific_contract_sha256"]:
        raise SystemExit("STOP: the frozen scientific contract changed since this execution config was written")
    shard_rows = int(xcfg["execution"]["shard_rows"])
    if shard_rows != LS.SHARD_ROWS:
        raise SystemExit(f"STOP: shard_rows {shard_rows} != frozen {LS.SHARD_ROWS}")

    try:
        resolved = m2b.resolve_roots(xcfg)
    except m2b.RootContainmentError as ex:
        raise SystemExit(f"STOP: {ex}")
    roots, runtime_root, runstate = resolved["roots"], resolved["runtime_root"], resolved["runstate_root"]
    reserve = int(xcfg["storage"]["min_free_reserve_bytes"])
    models = verify_models(reg, cfg)

    data_cfg = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
    inv = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
    plan = m2b.build_plan(inv, shard_rows)
    n_shards = (len(plan) + shard_rows - 1) // shard_rows
    my_shards = [s for s in range(n_shards) if s % a.n_workers == a.worker]
    if a.limit_shards is not None:
        my_shards = my_shards[:a.limit_shards]

    field_roots = {f: m2b.field_root(roots, f) for f in m2b.ALL_FIELDS}
    print(json.dumps({"samples": len(plan), "shards_total": n_shards, "shards_this_worker": len(my_shards),
                      "worker": a.worker, "n_workers": a.n_workers, "models": {k: v["sha256"][:12] for k, v in models.items()},
                      "free_gib": round(free_bytes(runtime_root) / 1024 ** 3, 2)}), flush=True)
    if a.dry_run:
        return 0

    from gpatbench.preprocess.aux_models import AdaFaceAdapter, FaceXFormerAdapter, set_deterministic
    from gpatbench.preprocess.scrfd import SCRFD
    set_deterministic(int(xcfg["execution"]["torch_threads"]))
    det = SCRFD(cfg["scrfd"]["model_path"], cfg["scrfd"]["weight_sha256"], input_size=C.SCRFD_INPUT)
    fx = FaceXFormerAdapter(MODEL_CACHE / "code/facexformer", MODEL_CACHE / "face_geometry/ckpts/model.pt")
    ada = AdaFaceAdapter(MODEL_CACHE / "code/adaface", cfg["adaface"]["checkpoint_path"])
    assert not fx.model.training and not ada.model.training

    state = m2b.StateLog.load(runstate)
    log = m2b.StateLog(runstate / f"m2b_state.w{a.worker}.jsonl")
    progress_path = runstate / f"m2b_progress.w{a.worker}.json"
    counters = {"chunks_done": 0, "chunks_skipped": 0, "chunks_rebuilt": 0, "complete": 0, "failed": 0,
                "reused_face": 0, "regenerated_face": 0, "scrfd_no_face": 0, "frames_written": 0}
    t0 = time.time()

    def progress(current_shard, note=""):
        progress_path.write_text(json.dumps({
            "worker": a.worker, "shards_total_this_worker": len(my_shards), "current_shard": current_shard,
            "elapsed_s": round(time.time() - t0, 1), "free_gib": round(free_bytes(runtime_root) / 1024 ** 3, 2),
            "note": note, **counters}, indent=1) + "\n", encoding="utf-8")

    for n_done, shard_index in enumerate(my_shards):
        chunk = m2b.chunk_of(plan, shard_index)
        shards = m2b.ChunkShardSet(field_roots, shard_index)
        if shards.is_finalized() and all(
                state.get(r["sample_id"], {}).get("state") in ("COMPLETE", "FAILED") for r in chunk):
            # A finalized chunk is skipped only after its face artifacts still hash as recorded, so a
            # corrupted or replaced face can never be silently accepted on resume. Full per-block
            # verification of the shards themselves is the job of the cache-integrity audit.
            bad = [r["sample_id"] for r in chunk
                   if state.get(r["sample_id"], {}).get("state") == "COMPLETE"
                   and not _face_reusable(r, state, roots, counters)]
            if not bad:
                counters["chunks_skipped"] += 1
                continue
            log.write(event="RESUME_ARTIFACT_MISMATCH_REBUILD_CHUNK", shard=shard_index,
                      sample_ids=bad[:20], n_mismatched=len(bad))
            counters["chunks_rebuilt"] = counters.get("chunks_rebuilt", 0) + 1
            for sid in bad:                       # force regeneration rather than reuse
                state.pop(sid, None)

        avail = free_bytes(runtime_root)
        if avail < reserve:
            log.write(event="HALT_LOW_DISK", shard=shard_index, free_bytes=avail, reserve_bytes=reserve)
            print(json.dumps({"halt": "LOW_DISK", "shard": shard_index, "free_gib": round(avail / 1024 ** 3, 2)}), flush=True)
            progress(shard_index, "halted: free space below reserve")
            break

        # ---- group the chunk's video samples so each video is decoded once, sequentially ----
        by_video: dict = {}
        for r in chunk:
            by_video.setdefault((r["dataset"], r["video_id"]), []).append(r)

        shards.open()
        chunk_states, chunk_failed = [], []
        try:
            for (ds, _vid), rows in by_video.items():
                rows.sort(key=lambda r: r["_ordinal"])
                src_root = Path(data_cfg["datasets"][ds]["root"])
                decoded = None
                if C.scrfd_required(ds):
                    need = [r["frame_index"] for r in rows
                            if not _face_reusable(r, state, roots, counters)]
                    if need:
                        decoded = read_video_frames(src_root / rows[0]["source_path"], need)
                for r in rows:
                    rec = _process(r, ds, src_root, decoded, state, roots, det, fx, ada, shards, counters, log)
                    (chunk_failed if rec["state"] == "FAILED" else chunk_states).append(rec)
            finalized = shards.finalize()
        except BaseException:
            shards.abort()
            raise

        for rec in chunk_states:
            rec["shards"] = {f: v["shard"] for f, v in finalized.items()}
            rec["state"] = "COMPLETE"
            log.write(**rec)
            counters["complete"] += 1
        for rec in chunk_failed:
            log.write(**rec)
            counters["failed"] += 1
        counters["chunks_done"] += 1
        if (n_done + 1) % max(1, int(xcfg["execution"]["progress_every_n_samples"]) // shard_rows or 1) == 0 or n_done == 0:
            progress(shard_index)
            print(json.dumps({"shard": shard_index, "done": n_done + 1, "of": len(my_shards), **counters,
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    progress(None, "finished")
    log.close()
    print(json.dumps({"worker": a.worker, "finished": True, "elapsed_s": round(time.time() - t0, 1), **counters}), flush=True)
    return 0


def _face_path(roots, ds, sample_id):
    return roots["faces_256_root"] / ds / f"{sample_id}.png"


def _frame_path(roots, ds, sample_id):
    return roots["processed_frames_root"] / ds / f"{sample_id}.png"


def _face_reusable(r, state, roots, counters) -> bool:
    """A face may be reused only if a recorded hash still matches the file on disk."""
    prev = state.get(r["sample_id"])
    if not prev or prev.get("state") not in ("FACE_DONE", "COMPLETE") or not prev.get("face_png_sha256"):
        return False
    p = _face_path(roots, r["dataset"], r["sample_id"])
    if not p.is_file():
        return False
    return m2b.sha256_file(p) == prev["face_png_sha256"]


def _process(r, ds, src_root, decoded, state, roots, det, fx, ada, shards, counters, log) -> dict:
    sid = r["sample_id"]
    rec = {"sample_id": sid, "dataset": ds, "video_id": r["video_id"], "frame_index": r["frame_index"],
           "ordinal": r["_ordinal"], "shard": r["_shard"], "source_path": r["source_path"],
           "source_file_sha256": r["source_file_sha256"], "route": C.route(ds), "state": "PENDING"}
    try:
        reusable = _face_reusable(r, state, roots, counters)
        if reusable:
            prev = state[sid]
            counters["reused_face"] += 1
            import cv2
            face_png = _face_path(roots, ds, sid).read_bytes()
            rgb = cv2.imdecode(np.frombuffer(face_png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            rgb = np.ascontiguousarray(rgb)
            rec.update({k: prev.get(k) for k in (
                "frame_png_path", "frame_png_sha256", "scrfd_applied", "scrfd_status", "n_detections",
                "det_score", "bbox", "crop_requested", "crop_source", "pad_left", "pad_top", "pad_right",
                "pad_bottom", "crop_side_px", "crop_pre_resize_hw", "crop_touches_border", "frame_hw")})
            rec["face_png_path"] = str(_face_path(roots, ds, sid))
            rec["face_png_sha256"] = prev["face_png_sha256"]
            rec["face_reused"] = True
        else:
            if state.get(sid, {}).get("state") in ("FACE_DONE", "COMPLETE"):
                counters["regenerated_face"] += 1
                log.write(sample_id=sid, event="ARTIFACT_HASH_MISMATCH_REGENERATED", state="PENDING")
            rec["face_reused"] = False
            if ds == "casia_fasd":
                src = src_root / r["source_path"]
                got = m2b.sha256_file(src)
                if got != r["source_file_sha256"]:
                    raise RuntimeError(f"RAW_SOURCE_HASH_MISMATCH: {got} != {r['source_file_sha256']}")
                bgr = read_image_frame(src)
                # DEV-011: the M1-selected source IS the canonical lossless frame; nothing is duplicated.
                rec.update(frame_png_path=str(src), frame_png_sha256=got, frame_hw=f"{bgr.shape[0]}x{bgr.shape[1]}",
                           scrfd_applied=False, scrfd_status="N/A", n_detections="", det_score="", bbox="",
                           crop_requested="", crop_source="", pad_left="", pad_top="", pad_right="",
                           pad_bottom="", crop_side_px="", crop_pre_resize_hw="", crop_touches_border="")
                rgb = C.casia_to_canonical(bgr)
            else:
                bgr = decoded[r["frame_index"]]
                fp = _frame_path(roots, ds, sid)
                rec["frame_png_sha256"] = m2b.atomic_write_bytes(fp, C.encode_png_bgr(bgr))
                rec["frame_png_path"] = str(fp)
                rec["frame_hw"] = f"{bgr.shape[0]}x{bgr.shape[1]}"
                counters["frames_written"] += 1
                log.write(**{**rec, "state": "FRAME_DONE"})
                dets, _k = det.detect(bgr, C.SCRFD_THRESHOLD)
                rec.update(scrfd_applied=True, n_detections=int(len(dets)))
                outcome, i = C.detection_outcome(dets)
                if outcome == "SCRFD_NO_FACE":
                    counters["scrfd_no_face"] += 1
                    rec.update(scrfd_status="SCRFD_NO_FACE", state="FAILED", failure_reason="SCRFD_NO_FACE")
                    return rec
                d = dets[i]
                box = C.square_crop_box(d, bgr.shape[1], bgr.shape[0])
                crop = C.extract_square_crop(bgr, box)
                if crop.shape[0] != crop.shape[1] or crop.shape[0] != box.side_px:
                    raise RuntimeError("CROP_NOT_SQUARE")
                rec.update(scrfd_status="DETECTED", det_score=f"{float(d[4]):.6f}",
                           bbox=json.dumps([round(float(v), 3) for v in d[:4]]),
                           crop_requested=json.dumps(list(box.requested)), crop_source=json.dumps(list(box.source)),
                           pad_left=box.pad_left, pad_top=box.pad_top, pad_right=box.pad_right,
                           pad_bottom=box.pad_bottom, crop_side_px=box.side_px,
                           crop_pre_resize_hw=f"{crop.shape[0]}x{crop.shape[1]}",
                           crop_touches_border=bool(box.touches_border))
                rgb = C.to_canonical(crop)
            C.check_canonical(rgb)
            rec["face_png_sha256"] = m2b.atomic_write_bytes(_face_path(roots, ds, sid), C.encode_png_rgb(rgb))
            rec["face_png_path"] = str(_face_path(roots, ds, sid))
            log.write(**{**rec, "state": "FACE_DONE"})

        # Compute and validate BOTH models before appending any block, so a late failure can never
        # leave a sample with a geometry entry but no identity entry in the shards.
        g = fx(rgb)
        if not all(np.all(np.isfinite(v)) for v in g.values()):
            raise RuntimeError("FACEXFORMER_NON_FINITE")
        if g["parsing_logits"].dtype != np.float32 or g["parsing_logits"].shape != LS.PARSING_LOGITS_SHAPE:
            raise RuntimeError("FACEXFORMER_BAD_LOGITS_SHAPE_OR_DTYPE")
        if not np.array_equal(g["parsing_mask"], LS.mask_from_logits(g["parsing_logits"])):
            raise RuntimeError("MASK_NOT_ARGMAX_OF_LOGITS")
        e = ada(rgb)
        emb = e["embedding"]
        if emb.shape != (512,) or emb.dtype != np.float32:
            raise RuntimeError("ADAFACE_BAD_EMBEDDING_SHAPE_OR_DTYPE")
        if not np.all(np.isfinite(emb)) or abs(float(np.linalg.norm(emb)) - 1.0) > 1e-5:
            raise RuntimeError("ADAFACE_INVALID_EMBEDDING")

        shards.add("parsing_logits", sid, g["parsing_logits"])
        for f in ("parsing_mask", "landmarks_norm", "landmarks_px224", "landmarks_px256", "pose_pitch_yaw_roll_rad"):
            shards.add(f, sid, g[f])
        shards.add("embedding", sid, emb)
        rec.update(parsing_logits_shape=str(g["parsing_logits"].shape), fx_status="OK", ada_status="OK",
                   embedding_l2=f"{float(np.linalg.norm(emb)):.8f}", raw_norm=f"{e['raw_norm']:.6f}",
                   state="IDENTITY_DONE")
        return rec
    except Exception as ex:                       # recorded and classified, never silently replaced
        rec.update(state="FAILED", failure_reason=f"{type(ex).__name__}: {ex}")
        return rec


if __name__ == "__main__":
    sys.exit(main())
