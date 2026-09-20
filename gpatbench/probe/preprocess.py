"""High-pass input pipeline for ArtifactProbeNet — CANDIDATE, NOT FROZEN.

Spec §12.1 fixes two things and only two things about the transform:

    HP(x) = x - GaussianBlur(x, kernel=9, sigma=1.5)      # §4 table, §12.1
    the probe input is 224x224 high-pass RGB              # §12.1

Everything else that can change a pixel value is unresolved (see
`outputs/audit/M5_ARTIFACT_PROBE_PREPROCESS_ANALYSIS.md`), so this module refuses to pick:
`resize_order`, `interpolation`, `border_mode` and `post_normalization` are **required** keyword
arguments with no defaults. Measured on real faces, the border mode moves values by up to 0.26, the
resize order by up to 0.084 and the interpolation by up to 0.13 in the unit interval, against a
residual whose typical peak is ~0.245 -- these are not implementation details.

The Gaussian implementation itself is NOT in that list: OpenCV, torchvision and an explicit
separable convolution agree to ~6.6e-07, i.e. float32 rounding. One is still pinned for byte
reproducibility, but the scientific result does not depend on the choice.
"""
from __future__ import annotations

import numpy as np

KERNEL = 9            # spec §4 / §12.1
SIGMA = 1.5           # spec §4 / §12.1
PROBE_INPUT_SIZE = 224
CANONICAL_FACE_SIZE = 256

RESIZE_ORDERS = ("resize_then_highpass", "highpass_then_resize")
INTERPOLATIONS = ("area", "bilinear", "bicubic", "nearest", "lanczos4")
BORDER_MODES = ("reflect101", "reflect", "replicate", "constant0")
POST_NORMALIZATIONS = ("none", "imagenet")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ProbeContractError(RuntimeError):
    """An unresolved ArtifactProbeNet choice was left unspecified or given an unknown value."""


def _cv2():
    import cv2
    return cv2


def _interp(name: str):
    cv2 = _cv2()
    try:
        return {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST,
                "lanczos4": cv2.INTER_LANCZOS4}[name]
    except KeyError:
        raise ProbeContractError(f"unknown interpolation {name!r}; expected one of {INTERPOLATIONS}")


def _border(name: str):
    cv2 = _cv2()
    try:
        return {"reflect101": cv2.BORDER_REFLECT_101, "reflect": cv2.BORDER_REFLECT,
                "replicate": cv2.BORDER_REPLICATE, "constant0": cv2.BORDER_CONSTANT}[name]
    except KeyError:
        raise ProbeContractError(f"unknown border mode {name!r}; expected one of {BORDER_MODES}")


def to_unit_interval(rgb_uint8) -> np.ndarray:
    """Canonical M2 face (256x256x3 RGB uint8) -> float32 in [0, 1]. Channel order is preserved."""
    a = np.asarray(rgb_uint8)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ProbeContractError(f"expected an RGB uint8 image, got {a.shape} {a.dtype}")
    return a.astype(np.float32) / np.float32(255.0)


def gaussian_blur(img: np.ndarray, *, border_mode: str) -> np.ndarray:
    """GaussianBlur(x, kernel=9, sigma=1.5) with an explicit border mode."""
    cv2 = _cv2()
    return cv2.GaussianBlur(img, (KERNEL, KERNEL), SIGMA, borderType=_border(border_mode))


def high_pass(img: np.ndarray, *, border_mode: str) -> np.ndarray:
    """HP(x) = x - GaussianBlur(x). The residual is SIGNED and is never clipped or rescaled here."""
    return img - gaussian_blur(img, border_mode=border_mode)


def resize(img: np.ndarray, size: int, *, interpolation: str) -> np.ndarray:
    cv2 = _cv2()
    return cv2.resize(img, (size, size), interpolation=_interp(interpolation))


def normalize(img: np.ndarray, *, post_normalization: str) -> np.ndarray:
    if post_normalization == "none":
        return img
    if post_normalization == "imagenet":
        return (img - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / \
            np.asarray(IMAGENET_STD, dtype=np.float32)
    raise ProbeContractError(
        f"unknown post_normalization {post_normalization!r}; expected one of {POST_NORMALIZATIONS}")


def probe_input(rgb_uint8, *, resize_order: str, interpolation: str, border_mode: str,
                post_normalization: str, size: int = PROBE_INPUT_SIZE) -> np.ndarray:
    """Candidate probe input, CHW float32.

    Every contested choice is a required keyword argument: this function cannot be called without
    stating the four open decisions, which is deliberate while the contract is unfrozen.
    """
    if resize_order not in RESIZE_ORDERS:
        raise ProbeContractError(
            f"unknown resize_order {resize_order!r}; expected one of {RESIZE_ORDERS}")
    x = to_unit_interval(rgb_uint8)
    if resize_order == "resize_then_highpass":
        x = resize(x, size, interpolation=interpolation)
        x = high_pass(x, border_mode=border_mode)
    else:
        x = high_pass(x, border_mode=border_mode)
        x = resize(x, size, interpolation=interpolation)
    x = normalize(x, post_normalization=post_normalization)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)).astype(np.float32))
