"""Deterministic final validation of the finalized M2 artifacts.

Re-runs a deterministic sample set from the RAW sources into a clean temporary directory on the
external volume and compares the result against what the full M2 run actually stored.

    <m2 venv>/bin/python tools/m2b_validate.py \
        --exec-config configs/execution/m2b_laptop_external_storage.yaml [--extra 48]

The sample set is:
  * the 24 frozen M2A smoke samples (manifest sha256 138f5929..., never reselected); plus
  * `--extra` samples chosen deterministically from the full frozen inventory by
    sha256("gpatbench.m2b_validation.v1|" + sample_id) — a hash ordering, so nothing is hand-picked.

Compared per sample: frame PNG bytes (MSU/SiW), canonical face PNG bytes, every geometry tensor
read back out of the finalized shards, and the AdaFace embedding. Exact equality is required.
Writes outputs/audit/M2_DETERMINISTIC_VALIDATION.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
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
SALT = "gpatbench.m2b_validation.v1|"
AUDIT = ROOT / "outputs/audit"
GEOM_FIELDS = ("parsing_mask", "landmarks_norm", "landmarks_px224", "landmarks_px256", "pose_pitch_yaw_roll_rad")


def block_from_shard(field_root: Path, sample_id: str, compressed: bool):
    for idx in sorted(field_root.glob("*.index.json")):
        meta = json.loads(idx.read_text())
        for r in meta["rows"]:
            if r["sample_id"] == sample_id:
                with open(field_root / r["shard"], "rb") as f:
                    f.seek(r["byte_offset"])
                    blob = f.read(r["byte_length"])
                return LS.decode_block(blob) if compressed else np.load(io.BytesIO(blob), allow_pickle=False)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-config", required=True)
    ap.add_argument("--extra", type=int, default=48)
    a = ap.parse_args()
    xcfg = yaml.safe_load((ROOT / a.exec_config).read_text())
    cfg = yaml.safe_load((ROOT / xcfg["scientific_contract"]).read_text())
    roots = {k: Path(v) for k, v in xcfg["storage"]["roots"].items()}
    data_cfg = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())

    inv = {r["sample_id"]: r for r in pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()}
    smoke_ids = [r["sample_id"] for r in csv.DictReader(
        open(AUDIT / "M2A_SMOKE_MANIFEST.csv", newline="", encoding="utf-8"))]
    extra = sorted(inv, key=lambda s: hashlib.sha256((SALT + s).encode()).hexdigest())
    extra = [s for s in extra if s not in set(smoke_ids)][:a.extra]
    targets = sorted(set(smoke_ids) | set(extra))

    from gpatbench.preprocess.aux_models import AdaFaceAdapter, FaceXFormerAdapter, set_deterministic
    from gpatbench.preprocess.scrfd import SCRFD
    set_deterministic(int(xcfg["execution"]["torch_threads"]))
    det = SCRFD(cfg["scrfd"]["model_path"], cfg["scrfd"]["weight_sha256"], input_size=C.SCRFD_INPUT)
    fx = FaceXFormerAdapter(MODEL_CACHE / "code/facexformer", MODEL_CACHE / "face_geometry/ckpts/model.pt")
    ada = AdaFaceAdapter(MODEL_CACHE / "code/adaface", cfg["adaface"]["checkpoint_path"])

    tmp = Path(tempfile.mkdtemp(prefix="m2b_validate_", dir=xcfg["storage"]["tmp_root"]))
    results, mismatches = [], []
    try:
        by_video: dict = {}
        for sid in targets:
            r = inv[sid]
            by_video.setdefault((r["dataset"], r["video_id"]), []).append(r)
        for (ds, _v), rows in sorted(by_video.items()):
            src_root = Path(data_cfg["datasets"][ds]["root"])
            decoded = (read_video_frames(src_root / rows[0]["source_path"], [x["frame_index"] for x in rows])
                       if C.scrfd_required(ds) else None)
            for r in sorted(rows, key=lambda x: x["sample_id"]):
                sid = r["sample_id"]
                out = {"sample_id": sid, "dataset": ds, "in_smoke_manifest": sid in set(smoke_ids)}
                if ds == "casia_fasd":
                    bgr = read_image_frame(src_root / r["source_path"])
                    rgb = C.casia_to_canonical(bgr)
                    out["frame_png_equal"] = None                       # CASIA persists no new frame
                else:
                    bgr = decoded[r["frame_index"]]
                    png = C.encode_png_bgr(bgr)
                    stored_frame = roots["processed_frames_root"] / ds / f"{sid}.png"
                    out["frame_png_equal"] = stored_frame.is_file() and stored_frame.read_bytes() == png
                    dets, _k = det.detect(bgr, C.SCRFD_THRESHOLD)
                    outcome, i = C.detection_outcome(dets)
                    if outcome == "SCRFD_NO_FACE":
                        out.update(scrfd="SCRFD_NO_FACE", face_png_equal=None)
                        results.append(out)
                        continue
                    box = C.square_crop_box(dets[i], bgr.shape[1], bgr.shape[0])
                    rgb = C.to_canonical(C.extract_square_crop(bgr, box))
                face_png = C.encode_png_rgb(rgb)
                stored_face = roots["faces_256_root"] / ds / f"{sid}.png"
                out["face_png_equal"] = stored_face.is_file() and stored_face.read_bytes() == face_png
                g = fx(rgb)
                e = ada(rgb)
                stored_logits = block_from_shard(roots["geometry_cache_root"] / "parsing_logits", sid, True)
                out["parsing_logits_equal"] = stored_logits is not None and np.array_equal(stored_logits, g["parsing_logits"])
                out["parsing_logits_max_abs_diff"] = (
                    float(np.max(np.abs(stored_logits.astype(np.float64) - g["parsing_logits"].astype(np.float64))))
                    if stored_logits is not None else None)
                for f in GEOM_FIELDS:
                    st = block_from_shard(roots["geometry_cache_root"] / f, sid, False)
                    out[f"{f}_equal"] = st is not None and np.array_equal(st, g[f])
                st_emb = block_from_shard(roots["identity_cache_root"] / "embedding", sid, False)
                out["embedding_equal"] = st_emb is not None and np.array_equal(st_emb, e["embedding"])
                out["embedding_max_abs_diff"] = (
                    float(np.max(np.abs(st_emb.astype(np.float64) - e["embedding"].astype(np.float64))))
                    if st_emb is not None else None)
                bad = [k for k, v in out.items() if k.endswith("_equal") and v is False]
                if bad:
                    mismatches.append({"sample_id": sid, "fields": bad})
                results.append(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    checked = [r for r in results if r.get("face_png_equal") is not None]
    rep = {
        "purpose": "Deterministic final validation: re-run from RAW sources into a clean temporary external "
                   "directory and compare against the finalized full-M2 artifacts.",
        "selection": {"smoke_manifest_samples": len(smoke_ids), "hash_selected_extra": len(extra),
                      "salt": SALT, "hand_picked": False, "total": len(targets)},
        "samples_compared": len(checked),
        "max_abs_diff": {
            "parsing_logits": max([r.get("parsing_logits_max_abs_diff") or 0.0 for r in checked], default=0.0),
            "embedding": max([r.get("embedding_max_abs_diff") or 0.0 for r in checked], default=0.0)},
        "all_equal": not mismatches,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
        "per_sample": results,
    }
    (AUDIT / "M2_DETERMINISTIC_VALIDATION.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("status", "samples_compared", "selection", "max_abs_diff", "mismatches")},
                     indent=1, default=str))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
