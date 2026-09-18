"""Valid-frame probing (spec §3.4; Q-04 operational definition, see configs/frozen/data_v1.yaml).

Read-only: files are only opened for reading; nothing is written or extracted here.

image_sequence: frame i is valid iff its file decodes (cv2.imdecode) to a non-empty image.
video_file:     index-bounded decode over the container-declared frame range [0, N_declared):
                cv2.VideoCapture(path, cv2.CAP_FFMPEG); for each original index i in that range,
                one sequential read() is attempted; success with a non-empty array => valid,
                failure => invalid (index i is preserved; decoding continues with i+1).
                Evidence that one read() = one frame position, including failed reads: M1
                run A, MSU client008 127 good + 1 failed + 172 good = 300 declared; client023
                247 + 1 + 53 = 301. After the range, one extra read() is attempted as evidence
                only: success => frames_beyond_declared=True (never sampled).
                Fallback: if N_declared <= 0 the range is unknown -> decode_status
                DECLARED_COUNT_UNAVAILABLE, no valid frames (reported as ERROR, never guessed).
                FFmpeg may also return error-concealed frames (read() ok) while logging decode
                errors to stderr; those log lines are captured per video (fd 2 redirected
                inside the worker process) and counted as decoder_error_lines for review.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np



def decoder_info() -> dict:
    info = {"library": "opencv-python-headless", "cv2_version": cv2.__version__,
            "video_api": "cv2.VideoCapture(path, cv2.CAP_FFMPEG)", "image_api": "cv2.imdecode(IMREAD_COLOR)"}
    for line in cv2.getBuildInformation().splitlines():
        s = line.strip()
        for key in ("avcodec", "avformat", "avutil", "swscale"):
            if s.startswith(key + ":"):
                info[f"ffmpeg_{key}"] = s.split(":", 1)[1].strip()
    return info


def probe_image_sequence(root: str, frame_files: dict[int, str]) -> dict:
    valid, invalid, shapes = [], [], set()
    for idx in sorted(frame_files):
        buf = np.fromfile(Path(root) / frame_files[idx], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if img is not None and img.size > 0:
            valid.append(idx)
            shapes.add(f"{img.shape[1]}x{img.shape[0]}")
        else:
            invalid.append(idx)
    return {"declared_frames": len(frame_files), "valid_indices": valid, "invalid_indices": invalid,
            "decode_status": "OK" if not invalid else "SOME_FRAMES_UNDECODABLE",
            "frame_sizes": sorted(shapes), "fps": None, "resumed_after_failure": False,
            "frames_beyond_declared": False, "decoder_error_lines": 0, "decoder_error_sample": []}


def probe_video(root: str, rel: str) -> dict:
    """Wrapper capturing native (FFmpeg) stderr for one video; safe because each worker
    process decodes one video at a time."""
    with tempfile.TemporaryFile() as log:
        saved = os.dup(2)
        os.dup2(log.fileno(), 2)
        try:
            res = _probe_video(root, rel)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        log.seek(0)
        # strip run-specific pointer addresses so the evidence text is deterministic
        lines = [re.sub(r" @ 0x[0-9a-fA-F]+", "", l) for l in log.read().decode("utf-8", "replace").splitlines() if l.strip()]
    res["decoder_error_lines"] = len(lines)
    res["decoder_error_sample"] = sorted(set(lines))[:5]
    return res


def _probe_video(root: str, rel: str) -> dict:
    cap = cv2.VideoCapture(str(Path(root) / rel), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return {"declared_frames": None, "valid_indices": [], "invalid_indices": [], "decode_status": "OPEN_FAILED",
                "frame_sizes": [], "fps": None, "resumed_after_failure": False, "frames_beyond_declared": False}
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    valid, invalid, shapes = [], [], set()
    if declared <= 0:
        cap.release()
        return {"declared_frames": declared, "valid_indices": [], "invalid_indices": [],
                "decode_status": "DECLARED_COUNT_UNAVAILABLE", "frame_sizes": [], "fps": fps,
                "resumed_after_failure": False, "frames_beyond_declared": False}
    for i in range(declared):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            valid.append(i)
            shapes.add(f"{frame.shape[1]}x{frame.shape[0]}")
        else:
            invalid.append(i)
    beyond = bool(cap.read()[0])
    cap.release()
    # an invalid index followed by a valid one = failure inside the stream
    resumed = any(i < valid[-1] for i in invalid) if valid else False
    if not valid:
        status = "NO_DECODABLE_FRAMES"
    elif resumed:
        status = "SOME_FRAMES_UNDECODABLE"
    elif invalid:
        status = "TRAILING_FRAMES_UNDECODABLE"
    else:
        status = "OK"
    return {"declared_frames": declared, "valid_indices": valid, "invalid_indices": invalid,
            "decode_status": status, "frame_sizes": sorted(shapes), "fps": fps,
            "resumed_after_failure": bool(resumed), "frames_beyond_declared": beyond}
