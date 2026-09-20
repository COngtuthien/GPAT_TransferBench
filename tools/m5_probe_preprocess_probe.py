"""Measure how much each unresolved high-pass choice actually changes the probe input (spec §12.1).

    <m2 venv>/bin/python tools/m5_probe_preprocess_probe.py

READ-ONLY measurement on frozen M2 canonical faces. It builds NO model, trains NOTHING and writes
no checkpoint. Its only purpose is to turn "underspecified" into a number, so the owner can see
which of the open choices are execution-affecting and by how much.

Nothing here is frozen: every variant is a candidate until the owner decides.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E      # noqa: E402
from gpatbench.preprocess import m2b          # noqa: E402

AUDIT = ROOT / "outputs/audit"
SALT = "gpatbench.m5_hp_probe.v1|"
N_PER_DATASET = 12
K, SIGMA = 9, 1.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


def load_faces():
    import hashlib
    import pyarrow.parquet as pq
    cfg = yaml.safe_load((ROOT / "configs/execution/m2b_laptop_external_storage.yaml").read_text())
    roots = m2b.resolve_roots(cfg)["roots"]
    rows = [r for r in pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()
            if r["split"] in ("TRAIN", "VAL")]          # TEST is never opened
    out = []
    for ds in ("casia_fasd", "msu_mfsd", "siwmv2"):
        sub = sorted([r for r in rows if r["dataset"] == ds],
                     key=lambda r: hashlib.sha256((SALT + r["sample_id"]).encode()).hexdigest())
        for r in sub[:N_PER_DATASET]:
            p = roots["faces_256_root"] / ds / f"{r['sample_id']}.png"
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            out.append((r, np.ascontiguousarray(bgr[:, :, ::-1])))   # RGB uint8 256x256x3
    return out


# ---------------------------------------------------------------- Gaussian implementations
def hp_cv2(rgb01: np.ndarray, border=cv2.BORDER_REFLECT_101) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb01, (K, K), SIGMA, borderType=border)
    return rgb01 - blur


def hp_torchvision(rgb01: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(rgb01).permute(2, 0, 1).unsqueeze(0)
    blur = TF.gaussian_blur(t, [K, K], [SIGMA, SIGMA])
    return (t - blur).squeeze(0).permute(1, 2, 0).numpy()


def hp_conv_separable(rgb01: np.ndarray, pad_mode="reflect") -> np.ndarray:
    """Explicit separable convolution with a normalised Gaussian kernel."""
    ax = np.arange(K, dtype=np.float64) - (K - 1) / 2.0
    k1 = np.exp(-(ax ** 2) / (2 * SIGMA ** 2))
    k1 /= k1.sum()
    t = torch.from_numpy(rgb01).permute(2, 0, 1).unsqueeze(0).double()
    kt = torch.from_numpy(k1)
    pad = (K - 1) // 2
    x = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode=pad_mode)
    x = torch.nn.functional.conv2d(x, kt.view(1, 1, 1, K).repeat(3, 1, 1, 1), groups=3)
    x = torch.nn.functional.conv2d(x, kt.view(1, 1, K, 1).repeat(3, 1, 1, 1), groups=3)
    return (t - x).squeeze(0).permute(1, 2, 0).numpy()


def resize(rgb: np.ndarray, size: int, interp) -> np.ndarray:
    return cv2.resize(rgb, (size, size), interpolation=interp)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_to_signal_range": float(d.max() / max(1e-12, (a.max() - a.min())))}


def main() -> int:
    faces = load_faces()
    rep = {"read_only": True, "trained": False, "n_faces": len(faces),
           "per_dataset": N_PER_DATASET, "kernel": K, "sigma": SIGMA,
           "source_image": "frozen M2 canonical 256x256 RGB uint8 face",
           "note": "All variants are CANDIDATES. Nothing here is frozen."}

    acc = {k: [] for k in ("cv2_vs_torchvision", "cv2_vs_conv_reflect",
                           "torchvision_vs_conv_reflect", "border_reflect101_vs_replicate",
                           "border_reflect101_vs_constant0",
                           "resize_then_hp_vs_hp_then_resize",
                           "interp_area_vs_bilinear_resize_then_hp",
                           "interp_area_vs_bicubic_resize_then_hp",
                           "domain_uint8_scaled_vs_unit_interval")}
    ranges = {"hp_min": [], "hp_max": [], "hp_absmax": [], "hp_std": [],
              "hp_after_imagenet_norm_min": [], "hp_after_imagenet_norm_max": []}

    for _row, rgb_u8 in faces:
        r256 = rgb_u8.astype(np.float64) / 255.0

        # --- resize 256 -> 224 first (candidate order 1)
        r224_area = resize(rgb_u8, 224, cv2.INTER_AREA).astype(np.float64) / 255.0
        r224_bilin = resize(rgb_u8, 224, cv2.INTER_LINEAR).astype(np.float64) / 255.0
        r224_bicub = resize(rgb_u8, 224, cv2.INTER_CUBIC).astype(np.float64) / 255.0

        a = hp_cv2(r224_area.astype(np.float32)).astype(np.float64)
        b = hp_torchvision(r224_area.astype(np.float32)).astype(np.float64)
        c = hp_conv_separable(r224_area)
        acc["cv2_vs_torchvision"].append(stats(a, b))
        acc["cv2_vs_conv_reflect"].append(stats(a, c))
        acc["torchvision_vs_conv_reflect"].append(stats(b, c))

        rep_b = hp_cv2(r224_area.astype(np.float32), cv2.BORDER_REPLICATE).astype(np.float64)
        con_b = hp_cv2(r224_area.astype(np.float32), cv2.BORDER_CONSTANT).astype(np.float64)
        acc["border_reflect101_vs_replicate"].append(stats(a, rep_b))
        acc["border_reflect101_vs_constant0"].append(stats(a, con_b))

        # --- HP at 256 then resize the residual (candidate order 2)
        hp256 = hp_cv2(r256.astype(np.float32)).astype(np.float64)
        hp256_to224 = cv2.resize(hp256.astype(np.float32), (224, 224),
                                 interpolation=cv2.INTER_AREA).astype(np.float64)
        acc["resize_then_hp_vs_hp_then_resize"].append(stats(a, hp256_to224))

        acc["interp_area_vs_bilinear_resize_then_hp"].append(
            stats(a, hp_cv2(r224_bilin.astype(np.float32)).astype(np.float64)))
        acc["interp_area_vs_bicubic_resize_then_hp"].append(
            stats(a, hp_cv2(r224_bicub.astype(np.float32)).astype(np.float64)))

        # --- value domain: HP on uint8 then /255 vs HP on [0,1]
        hp_u8 = hp_cv2(resize(rgb_u8, 224, cv2.INTER_AREA).astype(np.float32)) / 255.0
        acc["domain_uint8_scaled_vs_unit_interval"].append(stats(a, hp_u8.astype(np.float64)))

        ranges["hp_min"].append(float(a.min()))
        ranges["hp_max"].append(float(a.max()))
        ranges["hp_absmax"].append(float(np.abs(a).max()))
        ranges["hp_std"].append(float(a.std()))
        norm = (a - IMAGENET_MEAN) / IMAGENET_STD
        ranges["hp_after_imagenet_norm_min"].append(float(norm.min()))
        ranges["hp_after_imagenet_norm_max"].append(float(norm.max()))

    rep["variant_differences"] = {
        k: {"max_abs_over_faces": max(s["max_abs"] for s in v),
            "mean_abs_over_faces": float(np.mean([s["mean_abs"] for s in v])),
            "identical": max(s["max_abs"] for s in v) == 0.0}
        for k, v in acc.items()}
    rep["residual_statistics_unit_interval"] = {
        k: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v))}
        for k, v in ranges.items()}
    rep["signed_residual_note"] = (
        "HP is signed: the measured residual spans roughly [-0.5, +0.5] in the unit-interval "
        "domain, so any implementation that clips to [0,1] or stores uint8 would destroy about "
        "half of the signal. Whether the residual stays signed is therefore execution-affecting.")
    rep["imagenet_normalization_note"] = (
        "The spec does not say whether ImageNet mean/std normalisation is applied AFTER the "
        "high-pass. Applying it to a zero-centred residual shifts the input far from the statistics "
        "the pretrained weights expect; not applying it leaves a near-zero-mean input. Both are "
        "defensible and they are not equivalent, so this is an owner decision, not a detail.")
    (AUDIT / "M5_HP_VARIANT_MEASUREMENTS.json").write_bytes(E.canonical_json_bytes(rep))
    print(json.dumps({"n_faces": rep["n_faces"],
                      "variant_differences": rep["variant_differences"],
                      "residual_range": rep["residual_statistics_unit_interval"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
