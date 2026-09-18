"""Shared M1 inventory primitives: canonical records, stable IDs, frame selection.

Spec basis: §3.1 (inventory), §3.2 (canonical schema), §3.3 (attack harmonization),
§3.4 (8-frame sampling). No randomness anywhere; Python's salted hash() is never used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
ATTACK_MACROS = ("live", "print", "replay", "mask_2d", "mask_3d", "makeup", "partial", "other_spoof")

# File categories used by the raw file index ("indexed" != "used as a sample").
CANONICAL_VIDEO = "CANONICAL_VIDEO"                  # one raw video file = one canonical video
CANONICAL_FRAME_IMAGE = "CANONICAL_FRAME_IMAGE"      # one raw frame of an image-sequence video
DERIVED_AUGMENTATION_COPY = "DERIVED_AUGMENTATION_COPY"  # packager-made transform of a canonical frame
DATASET_METADATA = "DATASET_METADATA"                # protocol lists, per-video annotation files
DOCUMENTATION = "DOCUMENTATION"
AUXILIARY_TOOL = "AUXILIARY_TOOL"                    # scripts, binaries, licenses, presets shipped with data
FILE_CATEGORIES = (CANONICAL_VIDEO, CANONICAL_FRAME_IMAGE, DERIVED_AUGMENTATION_COPY,
                   DATASET_METADATA, DOCUMENTATION, AUXILIARY_TOOL)

SUBJECT_RECOVERED = "RECOVERED"
SUBJECT_MISSING = "MISSING_NOT_IN_LOCAL_METADATA"

SAMPLE_ID_NAMESPACE = "gpatbench.sample_id.v1"


class UnclassifiedFileError(ValueError):
    """Raised when a file under a selected dataset root matches no explicit adapter rule."""


@dataclass(frozen=True)
class FileClass:
    category: str
    video_key: str | None = None   # canonical video_id this file belongs to (if any)
    frame_index: int | None = None  # original frame index (image sequences only)


@dataclass
class VideoRecord:
    dataset: str
    video_id: str
    subject_id_raw: str | None
    subject_id_status: str
    subject_id_source: str
    label_binary: int
    label_source: str
    attack_raw: str | None
    native_protocol_split: str | None
    native_meta: dict
    source_kind: str                     # "video_file" | "image_sequence"
    source_path: str                     # dataset-root-relative file (video) or sequence pattern
    frame_files: dict[int, str] = field(default_factory=dict)  # image sequences: index -> rel path
    label_conflict: str | None = None

    @property
    def subject_id_global(self) -> str | None:
        return None if self.subject_id_raw is None else f"{self.dataset}::{self.subject_id_raw}"


def sample_id(dataset: str, video_id: str, frame_index: int) -> str:
    """Spec §3.2: stable hash of dataset + canonical_video_id + sampled_frame_index.

    Serialization: compact JSON array [namespace, dataset, video_id, frame_index] (UTF-8),
    hashed with SHA-256, lowercase hex (64 chars).
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    payload = json.dumps([SAMPLE_ID_NAMESPACE, dataset, video_id, int(frame_index)],
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_frame_indices(valid_indices, positions) -> tuple[list[int], bool]:
    """Spec §3.4 frame selection over the valid frame range.

    - n < len(positions): every valid frame once (never duplicated).
    - otherwise: target_k = lo + p_k * (hi - lo) over the valid index range [lo, hi];
      for each k in order, take the valid index nearest to target_k that is not already
      chosen; ties -> the smaller index. Returns (sorted indices, collision_resolution_used).
    """
    v = sorted(set(int(i) for i in valid_indices))
    if len(v) < len(positions):
        return v, False
    lo, hi = v[0], v[-1]
    chosen: list[int] = []
    used_fallback = False
    for p in positions:
        t = lo + float(p) * (hi - lo)
        best = min(v, key=lambda i: (abs(i - t), i))
        if best in chosen:
            used_fallback = True
            best = min((i for i in v if i not in chosen), key=lambda i: (abs(i - t), i))
        chosen.append(best)
    return sorted(chosen), used_fallback
