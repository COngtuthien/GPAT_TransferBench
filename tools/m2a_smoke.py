"""M2A diagnostic smoke test (DIAGNOSTIC_ONLY; not M2 output; never reused as scientific cache).

  manifest : deterministic 24-sample subset -> outputs/audit/M2A_SMOKE_MANIFEST.csv
  run      : full M2 path on the subset with the OWNER-FROZEN contract
             (configs/frozen/preprocess_v1.yaml: Q-02 SCRFD 10G, Q-03 original AdaFace R50-WebFace4M,
             Q-18 BGR, Q-19 canonical-face 256->112 INTER_AREA, Q-22 square zero-padding,
             Q-23 lossless float32 logit shards)
             -> outputs/exploratory/m2a_smoke/<run_tag>/ (git-ignored) + <run_tag>_results.csv there
  compare  : byte/numeric comparison of two runs -> JSON on stdout

Run with the M2 environment:  /home/cong/.venvs/gpatbench-m2/bin/python tools/m2a_smoke.py <cmd> ...
"""
from __future__ import annotations

import csv
import hashlib
import json
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
from gpatbench.preprocess.frames import read_image_frame, read_video_frame  # noqa: E402

MANIFEST = ROOT / "outputs/audit/M2A_SMOKE_MANIFEST.csv"
OUTROOT = ROOT / "outputs/exploratory/m2a_smoke"
CACHE = Path("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache")
FROZEN = {   # configs/frozen/preprocess_v1.yaml — owner decisions, no provisional settings remain
    "scrfd_model": str(CACHE / "face_detectors/scrfd_10g_bnkps.onnx"),        # Q-02 SCRFD_10G_KPS
    "scrfd_sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "adaface_weights": str(CACHE / "face_identity/adaface_original/adaface_ir50_webface4m.ckpt"),   # Q-03
    "adaface_color_order": "BGR",                                             # Q-18
    "adaface_geometric": "RESIZE_256_TO_112_INTER_AREA",                      # Q-19
    "crop_border": "REQUESTED_SQUARE_ZERO_PAD",                               # Q-22
    "parsing_logits_storage": LS.CODEC_ID,                                    # Q-23
    "torch_threads": 4,
}
PER_DATASET = {"casia_fasd": (4, 4), "msu_mfsd": (4, 4), "siwmv2": (4, 4)}   # (live, spoof)
SALT = "gpatbench.m2a_smoke.v1|"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_manifest(path=MANIFEST) -> None:
    samples = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
    groups = {}
    with open(ROOT / "outputs/audit/siw_content_groups.csv") as f:
        for r in csv.DictReader(f):
            groups[r["video_id"]] = r["content_group_id"]
    rows = []
    for ds, (n_live, n_spoof) in PER_DATASET.items():
        cand = sorted((s for s in samples if s["dataset"] == ds), key=lambda s: sha((SALT + s["sample_id"]).encode()))
        chosen, videos, attacks = [], set(), set()
        for lab, need in ((0, n_live), (1, n_spoof)):
            for pass_ in (1, 2):   # pass 1: prefer unseen attack_raw (spoof); pass 2: relax
                for s in cand:
                    if sum(1 for c in chosen if c["label_binary"] == lab) >= need:
                        break
                    if s["label_binary"] != lab or s["video_id"] in videos:
                        continue
                    if lab == 1 and pass_ == 1 and s["attack_raw"] in attacks:
                        continue
                    chosen.append(s)
                    videos.add(s["video_id"])
                    attacks.add(s["attack_raw"])
        for s in chosen:
            rows.append({"sample_id": s["sample_id"], "dataset": ds, "video_id": s["video_id"],
                         "frame_index": s["frame_index"], "label_binary": s["label_binary"],
                         "attack_raw": s["attack_raw"] or "", "attack_macro": s["attack_macro"],
                         "source_path": s["source_path"], "source_file_sha256": s["source_file_sha256"],
                         "content_group_id": groups.get(s["video_id"], "") if ds == "siwmv2" else "",
                         "selection_rule": f"sort by sha256('{SALT}'+sample_id); per dataset {n_live} live + {n_spoof} spoof;"
                                           " distinct video_id; spoof prefers unseen attack_raw first"})
    rows.sort(key=lambda r: (r["dataset"], r["sample_id"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"rows": len(rows), "sha256": sha(Path(path).read_bytes())}))


def run(tag: str) -> None:
    import torch
    from gpatbench.preprocess.aux_models import AdaFaceAdapter, FaceXFormerAdapter, set_deterministic
    from gpatbench.preprocess.scrfd import SCRFD
    set_deterministic(FROZEN["torch_threads"])
    cfg = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
    out = OUTROOT / tag
    if out.exists():
        raise SystemExit(f"{out} exists; runs must start from clean diagnostic outputs")
    for d in ("frames", "faces", "geometry", "identity", "logit_shards"):
        (out / d).mkdir(parents=True)
    det = SCRFD(FROZEN["scrfd_model"], FROZEN["scrfd_sha256"], input_size=C.SCRFD_INPUT)
    fx = FaceXFormerAdapter(CACHE / "code/facexformer", CACHE / "face_geometry/ckpts/model.pt")
    ada = AdaFaceAdapter(CACHE / "code/adaface", FROZEN["adaface_weights"])   # frozen BGR / canonical-face adapter
    assert not fx.model.training and not ada.model.training
    shards = LS.ShardWriter(out / "logit_shards", prefix="parsing_logits")
    results = []
    with open(MANIFEST) as f:
        man = list(csv.DictReader(f))
    for m in man:
        r = {k: m[k] for k in ("sample_id", "dataset", "video_id", "frame_index", "label_binary", "attack_raw")}
        r.update(route=C.route(m["dataset"]), status="OK", error="")
        t0 = time.time()
        try:
            src = Path(cfg["datasets"][m["dataset"]]["root"]) / m["source_path"]
            if m["dataset"] == "casia_fasd":
                bgr = read_image_frame(src)
                r["frame_png_sha256"] = m["source_file_sha256"]          # source frame is already a PNG
                r.update(scrfd_applied=False, scrfd_status="N/A", n_detections="", det_score="", bbox="",
                         crop_requested="", crop_source="", pad_left="", pad_top="", pad_right="", pad_bottom="",
                         crop_side_px="", crop_pre_resize_hw="", crop_touches_border="")
                rgb = C.casia_to_canonical(bgr)
            else:
                bgr = read_video_frame(src, int(m["frame_index"]))
                png = C.encode_png_bgr(bgr)
                (out / "frames" / f"{m['sample_id']}.png").write_bytes(png)
                r["frame_png_sha256"] = sha(png)
                r["frame_hw"] = f"{bgr.shape[0]}x{bgr.shape[1]}"
                dets, _k = det.detect(bgr, C.SCRFD_THRESHOLD)
                r.update(scrfd_applied=True, n_detections=len(dets))
                outcome, i = C.detection_outcome(dets)
                if outcome == "SCRFD_NO_FACE":
                    r.update(scrfd_status="SCRFD_NO_FACE", status="SCRFD_NO_FACE")
                    results.append(r)
                    continue
                d = dets[i]
                box = C.square_crop_box(d, bgr.shape[1], bgr.shape[0])
                crop = C.extract_square_crop(bgr, box)                    # Q-22: requested square, zero-padded
                r.update(scrfd_status="DETECTED", det_score=f"{d[4]:.6f}", bbox=json.dumps([round(float(v), 3) for v in d[:4]]),
                         crop_requested=json.dumps(list(box.requested)), crop_source=json.dumps(list(box.source)),
                         pad_left=box.pad_left, pad_top=box.pad_top, pad_right=box.pad_right, pad_bottom=box.pad_bottom,
                         crop_side_px=box.side_px, crop_pre_resize_hw=f"{crop.shape[0]}x{crop.shape[1]}",
                         crop_touches_border=box.touches_border)
                if crop.shape[0] != crop.shape[1] or crop.shape[0] != box.side_px:
                    raise AssertionError(f"pre-resize crop is not the requested square: {crop.shape} vs {box.side_px}")
                rgb = C.to_canonical(crop)
            C.check_canonical(rgb)
            face = C.encode_png_rgb(rgb)
            (out / "faces" / f"{m['sample_id']}.png").write_bytes(face)
            r["face_png_sha256"] = sha(face)
            g = fx(rgb)
            for k, v in g.items():
                np.save(out / "geometry" / f"{m['sample_id']}__{k}.npy", v)
            # Q-23 lossless float32 logit storage: encode, shard, and verify exact reconstruction
            logits = g["parsing_logits"]
            raw_bytes = LS.npy_bytes(logits)
            t_enc = time.perf_counter(); blob = LS.encode_block(logits); t_enc = time.perf_counter() - t_enc
            t_dec = time.perf_counter(); back_npy = LS.decode_block_npy_bytes(blob); t_dec = time.perf_counter() - t_dec
            back = LS.npy_load(back_npy)
            r.update(logits_raw_bytes=logits.nbytes, logits_npy_bytes=len(raw_bytes), logits_compressed_bytes=len(blob),
                     logits_ratio=f"{len(raw_bytes) / len(blob):.6f}", logits_encode_s=f"{t_enc:.4f}", logits_decode_s=f"{t_dec:.4f}",
                     logits_bytes_equal=bool(back_npy == raw_bytes), logits_array_equal=bool(np.array_equal(logits, back)),
                     logits_dtype=str(logits.dtype),
                     mask_from_stored_logits=bool(np.array_equal(g["parsing_mask"], LS.mask_from_logits(back))))
            if not (r["logits_bytes_equal"] and r["logits_array_equal"] and r["mask_from_stored_logits"]):
                raise AssertionError("lossless logit round-trip failed")
            shards.add(m["sample_id"], logits)
            r.update(fx_status="OK", parsing_logits_shape=str(g["parsing_logits"].shape), landmarks_shape=str(g["landmarks_norm"].shape),
                     pose_deg=json.dumps([round(float(v) * 180 / np.pi, 3) for v in g["pose_pitch_yaw_roll_rad"]]),
                     fx_finite=bool(all(np.all(np.isfinite(v)) for v in g.values())))
            e = ada(rgb)
            np.save(out / "identity" / f"{m['sample_id']}__embedding.npy", e["embedding"])
            r.update(ada_status="OK", embedding_dim=e["embedding"].shape[0], embedding_l2=f"{float(np.linalg.norm(e['embedding'])):.8f}",
                     raw_norm=f"{e['raw_norm']:.6f}")
        except Exception as ex:   # recorded, never silently replaced
            r.update(status="ERROR", error=f"{type(ex).__name__}: {ex}")
        r["seconds"] = f"{time.time() - t0:.2f}"
        results.append(r)
    shards.close()
    idx_fields = ["sample_id", "shard", "byte_offset", "byte_length", "block_sha256", "npy_sha256",
                  "shape", "dtype", "codec_id", "row_in_shard"]
    with open(out / "logit_shards" / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=idx_fields, lineterminator="\n")
        w.writeheader()
        for row in shards.index:
            w.writerow({**row, "shape": json.dumps(row["shape"])})
    fields = sorted({k for r in results for k in r})
    with open(out / f"{tag}_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    print(json.dumps({"tag": tag, "n": len(results), "status": {s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}},
                      "torch": torch.__version__, "threads": torch.get_num_threads(),
                      "codec": LS.codec_provenance(), "logit_blocks": len(shards.index)}))


def compare(a: str, b: str) -> None:
    A, B = OUTROOT / a, OUTROOT / b
    files = sorted(p.relative_to(A).as_posix() for p in A.rglob("*") if p.is_file() and not p.name.endswith("_results.csv"))
    other = sorted(p.relative_to(B).as_posix() for p in B.rglob("*") if p.is_file() and not p.name.endswith("_results.csv"))
    rep = {"same_file_set": files == other, "n_files": len(files), "byte_identical": 0, "differs": [], "max_abs_diff": {}}
    for rel in files:
        x, y = (A / rel).read_bytes(), (B / rel).read_bytes()
        if x == y:
            rep["byte_identical"] += 1
        else:
            rep["differs"].append(rel)
            if rel.endswith(".npy"):
                d = float(np.max(np.abs(np.load(A / rel).astype(np.float64) - np.load(B / rel).astype(np.float64))))
                rep["max_abs_diff"][rel] = d
    drop = {"seconds", "logits_encode_s", "logits_decode_s"}   # wall-clock only; never a scientific value

    def _rows(path):
        # csv parsing (not naive splitting): JSON fields such as bbox/crop_requested contain commas
        with open(path, newline="", encoding="utf-8") as fh:
            return [{k: v for k, v in row.items() if k not in drop} for row in csv.DictReader(fh)]

    rows_a, rows_b = _rows(A / f"{a}_results.csv"), _rows(B / f"{b}_results.csv")
    rep["results_equal_except_timing"] = rows_a == rows_b
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"manifest": lambda: build_manifest(), "run": lambda: run(sys.argv[2]),
     "compare": lambda: compare(sys.argv[2], sys.argv[3])}[cmd]()
