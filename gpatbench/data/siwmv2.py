"""SiW-Mv2 adapter for the LOCAL copy.

Local layout observed in M1:
    README.pdf, DRA.pdf                               documentation
    Live/Live_{n}.{mov|mp4|avi}                       live videos
    Spoof/{AttackFolder}/{Prefix}_{n}.{mov|mp4|avi}   spoof videos, 14 attack folders

Subject identity: NOT recoverable from local files. Filenames carry only a per-class sequence
number (785 live videos vs 493 live subjects per README.pdf, so the number is not a subject),
and README.pdf defers the per-sample naming to an external link that is not present locally.
subject_id_raw is therefore null with status MISSING_NOT_IN_LOCAL_METADATA; no heuristic is
applied. This blocks the M3 main split for SiW-Mv2 (BLOCKED_BY_MISSING_SUBJECT_ID).

attack_raw = the attack folder name, preserved verbatim.
"""
from __future__ import annotations

import re

from .base import (CANONICAL_VIDEO, DOCUMENTATION, SUBJECT_MISSING, FileClass,
                   UnclassifiedFileError, VideoRecord)

DATASET = "siwmv2"
LIVE_RE = re.compile(r"^Live/(?P<prefix>Live)_(?P<num>\d+)\.(?P<ext>mov|mp4|avi)$")
SPOOF_RE = re.compile(r"^Spoof/(?P<folder>[A-Za-z_]+)/(?P<prefix>[A-Za-z_]+?)_(?P<num>\d+)\.(?P<ext>mov|mp4|avi)$")
DOCS = {"README.pdf", "DRA.pdf"}


def classify(rel: str) -> FileClass:
    if LIVE_RE.match(rel) or SPOOF_RE.match(rel):
        return FileClass(CANONICAL_VIDEO, rel)
    if rel in DOCS:
        return FileClass(DOCUMENTATION)
    raise UnclassifiedFileError(f"{DATASET}: no explicit rule for {rel!r}")


def build_videos(classified: list[tuple[str, FileClass]]) -> list[VideoRecord]:
    out = []
    for rel, fc in sorted(classified):
        if fc.category != CANONICAL_VIDEO:
            continue
        m = LIVE_RE.match(rel)
        live = m is not None
        m = m or SPOOF_RE.match(rel)
        out.append(VideoRecord(
            dataset=DATASET, video_id=rel,
            subject_id_raw=None, subject_id_status=SUBJECT_MISSING,
            subject_id_source="none: filenames carry only a per-class sequence number; README.pdf defers naming to an external link",
            label_binary=0 if live else 1,
            label_source="top-level directory 'Live/' or 'Spoof/'",
            attack_raw=None if live else m["folder"],
            native_protocol_split=None,
            native_meta={"folder": "Live" if live else m["folder"], "file_prefix": m["prefix"],
                         "sequence_number": int(m["num"]), "container_ext": m["ext"]},
            source_kind="video_file", source_path=rel,
        ))
    return out
