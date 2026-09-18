"""MSU-MFSD adapter. Rules come verbatim from the local README.txt naming protocol.

    scene01/real/real_client{ID}_{camera}_{res}_scene01.{mov|mp4}
    scene01/attack/attack_client{ID}_{camera}_{res}_{attackType}_scene01.{mov|mp4}
    *.face                  PittPatt face/eye locations per video (DATASET_METADATA)
    train_sub_list.txt / test_sub_list.txt   native protocol subject lists (2-digit IDs)

video_id = the dataset-relative path without extension (unique; the .face file shares it).
subject_id_raw = the 3-digit clientID from the filename. The native lists are kept only as
native_protocol_split provenance; they are NOT the benchmark split (M3 owns that).
camera/resolution are kept in native_meta for audit only (spec §5.2 shortcut control).
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import (AUXILIARY_TOOL, CANONICAL_VIDEO, DATASET_METADATA, DOCUMENTATION,
                   SUBJECT_RECOVERED, FileClass, UnclassifiedFileError, VideoRecord)

DATASET = "msu_mfsd"
REAL_RE = re.compile(r"^scene01/real/real_client(?P<cid>\d{3})_(?P<cam>android|laptop)_(?P<res>SD|HD)_(?P<scn>scene01)\.(?P<ext>mov|mp4|face)$")
ATTACK_RE = re.compile(r"^scene01/attack/attack_client(?P<cid>\d{3})_(?P<cam>android|laptop)_(?P<res>SD|HD)_"
                       r"(?P<atk>ipad_video|iphone_video|printed_photo)_(?P<scn>scene01)\.(?P<ext>mov|mp4|face)$")
PROTOCOL_LISTS = {"train_sub_list.txt": "train", "test_sub_list.txt": "test"}


def classify(rel: str) -> FileClass:
    for rx in (REAL_RE, ATTACK_RE):
        m = rx.match(rel)
        if m:
            vid = rel.rsplit(".", 1)[0]
            return FileClass(DATASET_METADATA if m["ext"] == "face" else CANONICAL_VIDEO, vid)
    if rel == "README.txt" or rel == "ffmpeg/README.txt":
        return FileClass(DOCUMENTATION)
    if rel in PROTOCOL_LISTS:
        return FileClass(DATASET_METADATA)
    if re.fullmatch(r"DecFrames(_real_scene01|_attack_scene01)?\.m", rel) or rel.startswith("ffmpeg/"):
        return FileClass(AUXILIARY_TOOL)
    raise UnclassifiedFileError(f"{DATASET}: no explicit rule for {rel!r}")


def read_protocol_lists(root: Path) -> dict[int, str]:
    """clientID (int) -> native split, from the two list files. Overlap is a hard error."""
    out: dict[int, str] = {}
    for fname, split in PROTOCOL_LISTS.items():
        for tok in (root / fname).read_text(encoding="utf-8").split():
            if not re.fullmatch(r"\d{2,3}", tok):
                raise ValueError(f"{DATASET}: invalid subject token {tok!r} in {fname}")
            cid = int(tok)
            if cid in out:
                raise ValueError(f"{DATASET}: subject {tok} listed in both protocol lists")
            out[cid] = split
    return out


def build_videos(classified: list[tuple[str, FileClass]], root: Path) -> list[VideoRecord]:
    lists = read_protocol_lists(root)
    face_files = {fc.video_key for rel, fc in classified if fc.category == DATASET_METADATA and fc.video_key}
    out = []
    for rel, fc in sorted(classified):
        if fc.category != CANONICAL_VIDEO:
            continue
        m = REAL_RE.match(rel) or ATTACK_RE.match(rel)
        is_attack = rel.startswith("scene01/attack/")
        cid = m["cid"]
        out.append(VideoRecord(
            dataset=DATASET, video_id=fc.video_key,
            subject_id_raw=cid, subject_id_status=SUBJECT_RECOVERED,
            subject_id_source="filename clientID (README.txt naming protocol)",
            label_binary=1 if is_attack else 0,
            label_source="README.txt naming protocol: 'real'/'attack' directory and filename prefix",
            attack_raw=m["atk"] if is_attack else None,
            native_protocol_split=lists.get(int(cid)),
            native_meta={"camera": m["cam"], "resolution": m["res"], "scenario": m["scn"],
                         "container_ext": m["ext"], "face_annotation_present": fc.video_key in face_files},
            source_kind="video_file", source_path=rel,
        ))
    return out
