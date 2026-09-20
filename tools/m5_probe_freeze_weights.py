"""Recompute the frozen ArtifactProbeNet class map and class weights from the authoritative split.

    <m2 venv>/bin/python tools/m5_probe_freeze_weights.py

Owner decisions D-M5-01 (LIVE_PLUS_SPOOF_ATTACK_MACRO, K=7) and D-M5-02
(RESOLVED_BY_OWNER_BALANCED_INVERSE_FREQUENCY). Every count is recomputed from
manifests/split_v1.parquet and asserted against the expected values; hand-written numbers are never
trusted. Writes outputs/audit/M5_ARTIFACT_PROBE_CLASS_WEIGHTS.csv. Trains nothing.
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
from gpatbench.pairs import execute as E  # noqa: E402

AUDIT = ROOT / "outputs/audit"
SPLIT = ROOT / "manifests/split_v1.parquet"

CLASSES = ["live", "makeup", "mask_2d", "mask_3d", "partial", "print", "replay"]   # lexical, K=7
K = 7
CLIP_LO, CLIP_HI = 0.5, 3.0

EXPECTED_TRAIN_TOTAL = 14467
EXPECTED_VAL_TOTAL = 3121
EXPECTED_TRAIN = {"live": 5629, "makeup": 759, "mask_2d": 96, "mask_3d": 1056,
                  "partial": 1911, "print": 2838, "replay": 2178}
EXPECTED_VAL = {"live": 1216, "makeup": 159, "mask_2d": 24, "mask_3d": 224,
                "partial": 406, "print": 623, "replay": 469}


def main() -> int:
    rows = pq.read_table(SPLIT).to_pylist()
    for r in rows:
        if r["split"] in ("TRAIN", "VAL") and r["m2_status"] != "COMPLETE":
            raise SystemExit(f"STOP: {r['sample_id']} is not M2 COMPLETE")

    tr = [r for r in rows if r["split"] == "TRAIN"]
    va = [r for r in rows if r["split"] == "VAL"]
    tc = collections.Counter(r["attack_macro"] for r in tr)
    vc = collections.Counter(r["attack_macro"] for r in va)

    problems = []
    if len(tr) != EXPECTED_TRAIN_TOTAL:
        problems.append(f"TRAIN total {len(tr)} != {EXPECTED_TRAIN_TOTAL}")
    if len(va) != EXPECTED_VAL_TOTAL:
        problems.append(f"VAL total {len(va)} != {EXPECTED_VAL_TOTAL}")
    if sorted(tc) != CLASSES:
        problems.append(f"TRAIN classes {sorted(tc)} != {CLASSES}")
    if sorted(vc) != CLASSES:
        problems.append(f"VAL classes {sorted(vc)} != {CLASSES}")
    for c in CLASSES:
        if tc.get(c, 0) != EXPECTED_TRAIN[c]:
            problems.append(f"TRAIN {c}: {tc.get(c, 0)} != {EXPECTED_TRAIN[c]}")
        if vc.get(c, 0) != EXPECTED_VAL[c]:
            problems.append(f"VAL {c}: {vc.get(c, 0)} != {EXPECTED_VAL[c]}")
    if problems:
        raise SystemExit("STOP: counts differ from the frozen expectation:\n  " +
                         "\n  ".join(problems))

    # D-M5-02: w_raw = N / (K * n_c); w = clip(w_raw, 0.5, 3.0). float64, TRAIN only.
    N = np.float64(len(tr))
    n = np.array([tc[c] for c in CLASSES], dtype=np.float64)
    w_raw = N / (np.float64(K) * n)
    w = np.minimum(np.float64(CLIP_HI), np.maximum(np.float64(CLIP_LO), w_raw))

    out = []
    for i, c in enumerate(CLASSES):
        out.append({
            "class_index": i, "class": c,
            "train_count": int(tc[c]), "val_count": int(vc[c]),
            "train_fraction": repr(float(n[i] / N)),
            "w_raw_balanced": repr(float(w_raw[i])),
            "w_clipped": repr(float(w[i])),
            "clipped_low": bool(w_raw[i] < CLIP_LO),
            "clipped_high": bool(w_raw[i] > CLIP_HI),
            "train_datasets": ";".join(sorted({r["dataset"] for r in tr
                                               if r["attack_macro"] == c})),
        })
    p = AUDIT / "M5_ARTIFACT_PROBE_CLASS_WEIGHTS.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wri.writeheader(); wri.writerows(out)

    doc = {"decision_d_m5_01": "LIVE_PLUS_SPOOF_ATTACK_MACRO",
           "decision_d_m5_02": "RESOLVED_BY_OWNER_BALANCED_INVERSE_FREQUENCY",
           "split_manifest_sha256": E.sha256_file(SPLIT),
           "classes": CLASSES, "K": K, "N_train": int(N), "N_val": len(va),
           "dtype": "float64", "clip": [CLIP_LO, CLIP_HI],
           "formula": "w_raw(c) = N / (K * n_c); w(c) = min(3.0, max(0.5, w_raw(c)))",
           "renormalized_after_clip": False,
           "train_counts": {c: int(tc[c]) for c in CLASSES},
           "val_counts": {c: int(vc[c]) for c in CLASSES},
           "w_raw": {c: float(w_raw[i]) for i, c in enumerate(CLASSES)},
           "w_clipped": {c: float(w[i]) for i, c in enumerate(CLASSES)},
           "counts_asserted_against_expected": True}
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
