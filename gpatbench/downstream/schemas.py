"""Spec §22 schemas (synthetic manifest, downstream prediction file, run metadata) and canonical hashing."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Sequence

from .metrics import MetricError, aggregate_videos

FROZEN_SEEDS = (42, 1337, 2026)

SYNTHETIC_MANIFEST_COLUMNS = (            # §22.1, verbatim order
    "synthetic_id", "method_id", "track_id", "seed", "pair_id",
    "target_live_id", "source_spoof_id", "generated_image_path",
    "edit_map_path", "attack_macro_source", "style_id_source",
    "status", "failure_reason", "quality_q",
    "id_cosine", "landmark_nme", "parsing_dice", "lowfreq_rel_l1",
    "artifact_cosine", "attack_consistent", "radial_spectrum_l1",
    "generator_checkpoint_sha256", "method_config_sha256",
    "code_commit", "created_at")

PREDICTION_COLUMNS = (                    # §22.2, verbatim order
    "sample_id", "dataset", "subject_id_global", "video_id", "frame_index",
    "y_true", "attack_raw", "attack_macro",
    "p_frame", "threshold", "pred_frame",
    "video_score", "pred_video",
    "method_id", "track_id", "evaluator_id", "seed",
    "checkpoint_sha256", "split_manifest_sha256")

RUN_METADATA_KEYS = ("run_id", "method", "track", "seed", "git_commit", "config_sha256", "split_manifest_sha256",
                     "pair_manifest_sha256", "environment_lock_sha256", "gpu", "start_time", "end_time", "status",
                     "deviations")                                         # §22.3
RUN_STATUSES = ("COMPLETED", "FAILED", "BLOCKED")


class SchemaError(ValueError):
    pass


def _sha_like(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def validate_prediction_rows(rows: Sequence[dict]) -> dict:
    """Column set exactly §22.2; labels 0/1; probabilities finite in [0,1]; predictions 0/1; frozen seeds; sha256
    fields; one threshold, method, evaluator and seed per file; video_score equal to the recomputed mean of the
    video's p_frame. Decisions (pred_frame/pred_video) are not re-derived here because the comparison rule is an open
    contract item; they are checked for type only."""
    if not rows:
        raise SchemaError("empty prediction file")
    for i, r in enumerate(rows):
        if tuple(r) != PREDICTION_COLUMNS:
            raise SchemaError(f"row {i}: columns differ from §22.2 (order included)")
        if r["y_true"] not in (0, 1) or isinstance(r["y_true"], bool):
            raise SchemaError(f"row {i}: y_true")
        for k in ("p_frame", "video_score", "threshold"):
            v = r[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1:
                raise SchemaError(f"row {i}: {k} not a finite probability")
        for k in ("pred_frame", "pred_video"):
            if r[k] not in (0, 1) or isinstance(r[k], bool):
                raise SchemaError(f"row {i}: {k}")
        if r["seed"] not in FROZEN_SEEDS:
            raise SchemaError(f"row {i}: seed {r['seed']} not in {FROZEN_SEEDS}")
        for k in ("checkpoint_sha256", "split_manifest_sha256"):
            if not _sha_like(r[k]):
                raise SchemaError(f"row {i}: {k} is not a sha256 hex digest")
    for k in ("threshold", "method_id", "track_id", "evaluator_id", "seed", "checkpoint_sha256", "split_manifest_sha256"):
        if len({r[k] for r in rows}) != 1:
            raise SchemaError(f"{k} must be constant within one prediction file")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise SchemaError("duplicate sample_id")
    try:
        videos = {(v["dataset"], v["video_id"]): v for v in aggregate_videos(rows)}
    except MetricError as exc:
        raise SchemaError(str(exc)) from exc
    for i, r in enumerate(rows):
        v = videos[(r["dataset"], r["video_id"])]
        if r["video_score"] != v["video_score"]:
            raise SchemaError(f"row {i}: video_score is not the arithmetic mean of the video's p_frame")
    return {"rows": len(rows), "videos": len(videos)}


def validate_run_metadata(meta: dict) -> None:
    if tuple(sorted(meta)) != tuple(sorted(RUN_METADATA_KEYS)):
        raise SchemaError(f"run metadata keys differ from §22.3: {sorted(set(meta) ^ set(RUN_METADATA_KEYS))}")
    if meta["status"] not in RUN_STATUSES:
        raise SchemaError("status must be COMPLETED, FAILED or BLOCKED")
    if meta["seed"] not in FROZEN_SEEDS:
        raise SchemaError("seed")
    for k in ("config_sha256", "split_manifest_sha256", "environment_lock_sha256"):
        if not _sha_like(meta[k]):
            raise SchemaError(f"{k} is not a sha256 hex digest")
    if not isinstance(meta["deviations"], list):
        raise SchemaError("deviations must be a list")


def canonical_rows_sha256(rows: Iterable[dict], key: Sequence[str]) -> str:
    """Order-independent identity of a table: rows sorted by `key`, each serialized as sorted-key JSON (floats by
    repr, which round-trips exactly), newline-joined, SHA256."""
    def enc(r):
        for v in r.values():
            if isinstance(v, float) and not math.isfinite(v):
                raise SchemaError("non-finite value in a hashed table")
        return json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False)
    ordered = sorted(rows, key=lambda r: tuple(str(r[k]) for k in key))
    if len({tuple(str(r[k]) for k in key) for r in ordered}) != len(ordered):
        raise SchemaError("hash key is not unique")
    return hashlib.sha256("\n".join(enc(r) for r in ordered).encode("utf-8")).hexdigest()
