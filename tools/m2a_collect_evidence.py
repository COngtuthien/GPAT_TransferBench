"""Collect the committed small-text evidence of an M2A smoke run.

    python tools/m2a_collect_evidence.py <run_tag>

Writes (from the run's own outputs only; no model is re-executed):
  outputs/audit/M2A_SMOKE_RESULTS.csv    — the run's per-sample table
  outputs/audit/M2A_SMOKE_SUMMARY.json   — aggregate counters
  outputs/audit/M2A_BORDER_CASES.csv     — Q-22 border evidence for every crop touching the frame edge
The run's images/tensors/shards stay under outputs/exploratory/ (git-ignored).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.preprocess import logit_store as LS  # noqa: E402

AUDIT = ROOT / "outputs/audit"
BORDER_FIELDS = ["sample_id", "dataset", "label_binary", "attack_raw", "video_id", "frame_index", "frame_hw",
                 "det_score", "bbox", "requested_square", "source_intersection",
                 "pad_left", "pad_top", "pad_right", "pad_bottom", "side_px",
                 "pre_resize_hw", "is_square_before_resize", "final_hw", "canonical_face_sha256"]


def main() -> int:
    tag = sys.argv[1]
    run = ROOT / "outputs/exploratory/m2a_smoke" / tag
    rows = list(csv.DictReader(open(run / f"{tag}_results.csv", newline="", encoding="utf-8")))

    # 1. per-sample results table (verbatim copy of the run's own output)
    (AUDIT / "M2A_SMOKE_RESULTS.csv").write_bytes((run / f"{tag}_results.csv").read_bytes())

    # 2. border-case evidence (Q-22)
    border = [r for r in rows if r.get("crop_touches_border") == "True"]
    with open(AUDIT / "M2A_BORDER_CASES.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BORDER_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in sorted(border, key=lambda r: (r["dataset"], r["sample_id"])):
            side = int(r["crop_side_px"])
            w.writerow({"sample_id": r["sample_id"], "dataset": r["dataset"], "label_binary": r["label_binary"],
                        "attack_raw": r["attack_raw"], "video_id": r["video_id"], "frame_index": r["frame_index"],
                        "frame_hw": r["frame_hw"], "det_score": r["det_score"], "bbox": r["bbox"],
                        "requested_square": r["crop_requested"], "source_intersection": r["crop_source"],
                        "pad_left": r["pad_left"], "pad_top": r["pad_top"], "pad_right": r["pad_right"],
                        "pad_bottom": r["pad_bottom"], "side_px": side,
                        "pre_resize_hw": r["crop_pre_resize_hw"],
                        "is_square_before_resize": str(r["crop_pre_resize_hw"] == f"{side}x{side}"),
                        "final_hw": "256x256", "canonical_face_sha256": r["face_png_sha256"]})

    # 3. aggregate summary
    ok = [r for r in rows if r["status"] == "OK"]
    det = [r for r in rows if r["scrfd_applied"] == "True"]
    summary = {
        "label": "DIAGNOSTIC_ONLY", "run_tag": tag, "n": len(rows),
        "status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "frozen_contract": {
            "Q-02": "SCRFD_10G_KPS scrfd_10g_bnkps.onnx (official byte hash confirmed)",
            "Q-03": "original AdaFace R50 / WebFace4M (adaface_ir50_webface4m.ckpt)",
            "Q-18": "BGR", "Q-19": "canonical 256 -> 112 INTER_AREA, no additional alignment",
            "Q-22": "requested square preserved, zero padding outside the image",
            "Q-23": f"full float32 11x224x224 logits, lossless {LS.CODEC_ID} level {LS.ZSTD_LEVEL}"},
        "scrfd_success_by_dataset": {
            ds: {"n": sum(1 for r in det if r["dataset"] == ds),
                 "detected": sum(1 for r in det if r["dataset"] == ds and r["scrfd_status"] == "DETECTED"),
                 "no_face": sum(1 for r in det if r["dataset"] == ds and r["scrfd_status"] == "SCRFD_NO_FACE")}
            for ds in sorted({r["dataset"] for r in det})},
        "casia_scrfd_status": sorted({r["scrfd_status"] for r in rows if r["dataset"] == "casia_fasd"}),
        "n_detections_max": max((int(r["n_detections"]) for r in det), default=0),
        "crop_touches_border_by_dataset": {
            ds: sum(1 for r in det if r["dataset"] == ds and r["crop_touches_border"] == "True")
            for ds in sorted({r["dataset"] for r in det})},
        "crop_square_before_resize_all": all(r["crop_pre_resize_hw"] == f"{r['crop_side_px']}x{r['crop_side_px']}" for r in det
                                             if r["scrfd_status"] == "DETECTED"),
        "fx_ok": sum(1 for r in rows if r.get("fx_status") == "OK"),
        "fx_finite_all": all(r.get("fx_finite") == "True" for r in ok),
        "ada_ok": sum(1 for r in rows if r.get("ada_status") == "OK"),
        "embedding_l2_min": min(float(r["embedding_l2"]) for r in ok),
        "embedding_l2_max": max(float(r["embedding_l2"]) for r in ok),
        "embedding_dim": sorted({int(r["embedding_dim"]) for r in ok}),
        "logits_dtype": sorted({r["logits_dtype"] for r in ok}),
        "logits_exact_reconstruction_all": all(r["logits_array_equal"] == "True" and r["logits_bytes_equal"] == "True" for r in ok),
        "mask_from_stored_logits_all": all(r["mask_from_stored_logits"] == "True" for r in ok),
        "logits_compression_ratio_min": min(float(r["logits_ratio"]) for r in ok),
        "logits_compression_ratio_max": max(float(r["logits_ratio"]) for r in ok),
    }
    (AUDIT / "M2A_SMOKE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"results": len(rows), "border_cases": len(border),
                      "square_all": summary["crop_square_before_resize_all"],
                      "lossless_all": summary["logits_exact_reconstruction_all"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
