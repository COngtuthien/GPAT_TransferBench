"""Lossless extraction of M1-selected frames (spec §3.4 'extract canonical PNGs losslessly').

DEV-006: the frame identity is the explicit loop index over [0, N_declared) with one sequential read()
per index using cv2.VideoCapture(path, cv2.CAP_FFMPEG); CAP_PROP_POS_FRAMES is never used as identity.
A failed read at the requested index is an error; there is no fallback to a neighbouring frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameReadError(RuntimeError):
    pass


def read_video_frame(path, frame_index: int) -> np.ndarray:
    """Return the BGR uint8 frame at original decode position `frame_index`."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FrameReadError(f"cannot open {path}")
    try:
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 <= frame_index < declared:
            raise FrameReadError(f"index {frame_index} outside [0, {declared})")
        frame = None
        for i in range(frame_index + 1):
            ok, fr = cap.read()
            if i == frame_index:
                if not ok or fr is None or fr.size == 0:
                    raise FrameReadError(f"decode failed at index {frame_index} (no fallback)")
                frame = fr
        return np.ascontiguousarray(frame)
    finally:
        cap.release()


def read_image_frame(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise FrameReadError(f"cannot decode {path}")
    return img


def decoder_versions() -> dict:
    info = {"cv2": cv2.__version__, "api": "cv2.VideoCapture(path, cv2.CAP_FFMPEG); sequential read(); loop index"}
    for line in cv2.getBuildInformation().splitlines():
        s = line.strip()
        for k in ("avcodec", "avformat", "avutil", "swscale"):
            if s.startswith(k + ":"):
                info["ffmpeg_" + k] = s.split(":", 1)[1].strip()
    return info
