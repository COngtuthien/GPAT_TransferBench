"""M2 preprocessing geometry contracts (spec §4), dataset routing, canonical encoding.

Pure numpy/OpenCV; no model code. Every rule here is either verbatim spec §4 or an explicitly
labelled IMPLEMENTATION_DETAIL / PROVISIONAL choice recorded in outputs/audit/M2A_PREPROCESS_CONTRACT.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CANONICAL_SIZE = 256            # spec §4
CROP_SCALE = 1.25               # spec §4
SCRFD_INPUT = 320               # spec §4
SCRFD_THRESHOLD = 0.50          # spec §4
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 3]   # lossless; fixed level for byte determinism

ROUTES = {                      # dataset_protocol_policy_v1 + DEV-011
    "casia_fasd": "PRECROPPED_112_RGB_TO_256_INTER_CUBIC",
    "msu_mfsd": "SPEC_SECTION_4_SCRFD",
    "siwmv2": "SPEC_SECTION_4_SCRFD",
}


def route(dataset: str) -> str:
    return ROUTES[dataset]


def scrfd_required(dataset: str) -> bool:
    return route(dataset) == "SPEC_SECTION_4_SCRFD"


# ------------------------------------------------------------------ face selection
def select_largest_face(dets: np.ndarray) -> int | None:
    """Spec: 'choose largest detected face'. Area = (x2-x1)*(y2-y1) in original-frame pixels.
    Equal-area tie-break (IMPLEMENTATION_DETAIL): higher score, then smaller x1, then smaller y1,
    then lower row index. dets: (N, 5) [x1, y1, x2, y2, score] after threshold+NMS."""
    if dets is None or len(dets) == 0:
        return None
    keys = [(-(d[2] - d[0]) * (d[3] - d[1]), -d[4], d[0], d[1], i) for i, d in enumerate(dets)]
    return min(keys)[4]


def detection_outcome(dets: np.ndarray) -> tuple[str, int | None]:
    """No fallback (owner rule): no detection >= threshold -> SCRFD_NO_FACE and the sample stops.
    Never whole frame, centre crop, previous bbox, another detector or a lower threshold."""
    i = select_largest_face(dets)
    return ("SCRFD_NO_FACE", None) if i is None else ("DETECTED", i)


# ------------------------------------------------------------------ crop
@dataclass(frozen=True)
class CropBox:
    requested: tuple[int, int, int, int]   # square before clamping (x0, y0, x1, y1), half-open
    clamped: tuple[int, int, int, int]     # intersection with the image
    side_float: float
    was_clamped: bool


def square_crop_box(bbox, img_w: int, img_h: int, scale: float = CROP_SCALE) -> CropBox:
    """Spec §4: square centred on the SCRFD bbox, side = 1.25*max(w, h), clamp to image.
    Integer rounding (IMPLEMENTATION_DETAIL): side_px = round(side); x0 = round(cx - side_px/2)
    (Python round = half-to-even); box is half-open [x0, x0+side_px).
    Clamping (PROVISIONAL, Q-22): literal intersection with the image; no padding, no shifting."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w, h = x2 - x1, y2 - y1
    side = scale * max(w, h)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = int(round(side))
    X0, Y0 = int(round(cx - s / 2.0)), int(round(cy - s / 2.0))
    req = (X0, Y0, X0 + s, Y0 + s)
    cl = (max(0, X0), max(0, Y0), min(img_w, X0 + s), min(img_h, Y0 + s))
    return CropBox(req, cl, side, cl != req)


def resize_interpolation(src_h: int, src_w: int, size: int = CANONICAL_SIZE) -> int:
    """Spec §4: INTER_AREA when downscaling, INTER_CUBIC when upscaling. For a (possibly clamped,
    non-square) crop the decision uses the larger side (PROVISIONAL, part of Q-22); equal size ->
    INTER_AREA (no-op resize is never called)."""
    return cv2.INTER_AREA if max(src_h, src_w) >= size else cv2.INTER_CUBIC


def to_canonical(bgr_crop: np.ndarray, size: int = CANONICAL_SIZE) -> np.ndarray:
    """Crop (BGR uint8) -> canonical RGB uint8 size x size."""
    h, w = bgr_crop.shape[:2]
    if (h, w) != (size, size):
        bgr_crop = cv2.resize(bgr_crop, (size, size), interpolation=resize_interpolation(h, w, size))
    return np.ascontiguousarray(bgr_crop[:, :, ::-1])


def casia_to_canonical(bgr_112: np.ndarray) -> np.ndarray:
    """DEV-011: canonical 112x112 crop -> INTER_CUBIC 256x256 RGB uint8; no SCRFD, no extra crop."""
    if bgr_112.shape != (112, 112, 3) or bgr_112.dtype != np.uint8:
        raise ValueError(f"CASIA input must be 112x112x3 uint8, got {bgr_112.shape} {bgr_112.dtype}")
    out = cv2.resize(bgr_112, (CANONICAL_SIZE, CANONICAL_SIZE), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(out[:, :, ::-1])


def encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Lossless deterministic PNG of an RGB uint8 image (stored as RGB in the PNG)."""
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]), PNG_PARAMS)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def encode_png_bgr(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr, PNG_PARAMS)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def check_canonical(rgb: np.ndarray) -> None:
    if rgb.shape != (CANONICAL_SIZE, CANONICAL_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"canonical face must be {CANONICAL_SIZE}x{CANONICAL_SIZE}x3 uint8, got {rgb.shape} {rgb.dtype}")


def is_finite(x) -> bool:
    return bool(np.all(np.isfinite(x))) if isinstance(x, np.ndarray) else math.isfinite(x)
