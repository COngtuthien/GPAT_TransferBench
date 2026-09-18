"""CASIA 112->256 INTER_CUBIC resize frequency audit (DIAGNOSTIC ONLY; no training, no tuning).

Sample: every CASIA frame already selected by the M1 inventory (manifests/inventory.parquet),
a deterministic set fixed before this audit. For each frame, compare the original 112x112 image
("before") with its INTER_CUBIC 256x256 resize ("after"), the frozen upscaling interpolation of
spec §4. All measurements are on luminance Y (BT.601 from decoded BGR) in [0, 1].

Metrics (per image):
  mean_Y, std_Y
  radial power fractions, frequency f in cycles/pixel of the image's own grid (Nyquist = 0.5):
      band_0_0.1, band_0.1_0.2, band_0.2_0.3, band_0.3_0.5   (fraction of non-DC power)
  above_src_nyquist (after only): fraction of non-DC power above 0.5*112/256 = 0.21875 c/px,
      i.e. content that did not exist in the 112 source grid (interpolation-created)
  freqsub_region: fraction of non-DC power with normalized radius r in [0.20, 0.50]
      (r = f; the spec §8.2 FreqSub eligible region), before vs after
  haar_detail_frac: level-1 Haar (LH+HL+HH) energy / total energy of mean-subtracted Y
  hp_mean_abs: mean |Y - GaussianBlur(Y, 9x9, sigma=1.5)| (spec §4 high-pass operator)
  lap_var: variance of cv2.Laplacian(Y, CV_64F, ksize=3)

Outputs: outputs/audit/CASIA_RESIZE_FREQUENCY_AUDIT.csv (per image) and a JSON summary on stdout.
Run: .venv/bin/python tools/audit_casia_resize.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
CASIA = Path(CFG["datasets"]["casia_fasd"]["root"])
OUT = ROOT / "outputs/audit/CASIA_RESIZE_FREQUENCY_AUDIT.csv"
BANDS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
SRC_NYQ = 0.5 * 112 / 256


def luminance(bgr):
    b, g, r = [bgr[..., k].astype(np.float64) / 255.0 for k in range(3)]
    return 0.299 * r + 0.587 * g + 0.114 * b


def radial(y):
    h, w = y.shape
    F = np.fft.fftshift(np.fft.fft2(y - y.mean()))
    P = np.abs(F) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    rad = np.sqrt(fx ** 2 + fy ** 2)
    P[h // 2, w // 2] = 0.0
    tot = P.sum() or 1.0
    out = {f"band_{a}_{b}": float(P[(rad >= a) & (rad < b)].sum() / tot) for a, b in BANDS}
    out["freqsub_region"] = float(P[(rad >= 0.20) & (rad <= 0.50)].sum() / tot)
    out["above_src_nyquist"] = float(P[rad > SRC_NYQ].sum() / tot)
    return out


def haar_detail_frac(y):
    y = y - y.mean()
    a, b, c, d = y[0::2, 0::2], y[0::2, 1::2], y[1::2, 0::2], y[1::2, 1::2]
    ll, lh, hl, hh = (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2
    e = [float((x ** 2).sum()) for x in (ll, lh, hl, hh)]
    return (e[1] + e[2] + e[3]) / (sum(e) or 1.0)


def metrics(y):
    m = {"mean_Y": float(y.mean()), "std_Y": float(y.std())}
    m.update(radial(y))
    m["haar_detail_frac"] = haar_detail_frac(y)
    m["hp_mean_abs"] = float(np.abs(y - cv2.GaussianBlur(y, (9, 9), 1.5)).mean())
    m["lap_var"] = float(cv2.Laplacian(y, cv2.CV_64F, ksize=3).var())
    return m


def main() -> int:
    rows = [r for r in pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist() if r["dataset"] == "casia_fasd"]
    rows.sort(key=lambda r: r["sample_id"])
    out = []
    for r in rows:
        bgr = cv2.imdecode(np.fromfile(CASIA / r["source_path"], np.uint8), cv2.IMREAD_COLOR)
        up = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
        mb, ma = metrics(luminance(bgr)), metrics(luminance(up))
        row = {"sample_id": r["sample_id"], "label_binary": r["label_binary"], "attack_macro": r["attack_macro"],
               "src_h": bgr.shape[0], "src_w": bgr.shape[1]}
        for k in mb:
            row[f"{k}_112"] = round(mb[k], 8)
            row[f"{k}_256"] = round(ma[k], 8)
        out.append(row)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    keys = [k[:-4] for k in out[0] if k.endswith("_112")]
    summ = {"n": len(out), "cv2": cv2.__version__, "metrics": {}}
    for k in keys:
        b = np.array([r[f"{k}_112"] for r in out])
        a = np.array([r[f"{k}_256"] for r in out])
        summ["metrics"][k] = {"median_112": float(np.median(b)), "median_256": float(np.median(a)),
                              "iqr_112": [float(np.percentile(b, 25)), float(np.percentile(b, 75))],
                              "iqr_256": [float(np.percentile(a, 25)), float(np.percentile(a, 75))]}
        for lab in (0, 1):
            sel = [i for i, r in enumerate(out) if r["label_binary"] == lab]
            summ["metrics"][k][f"median_256_over_112_label{lab}"] = float(np.median(a[sel] / np.where(b[sel] == 0, np.nan, b[sel])))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
