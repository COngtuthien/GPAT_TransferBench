"""CASIA-FASD adapter for the LOCAL copy (repackaged face-crop frame sequences).

Local layout observed in M1 (no documentation file ships with this copy):
    {train|test}/{live|spoof}/s{N}v{CODE}f{F}.png      canonical frames (112x112 face crops)
    train/live/{b|f}s{N}v{CODE}f{F}.png                 packager-derived copies (see below)
CODE is 1..8 or HR_1..HR_4 (CASIA-FASD native video code). F is the frame index.

Rules (evidence in outputs/audit/M1_DATASET_EVIDENCE.md):
- canonical video_id = "{partition}/{class_dir}/s{N}v{CODE}" (one image sequence).
- subject_id_raw = "{partition}_s{N}": subject numbers restart per native partition
  (train s1..s20, test s1..s30) and visually belong to different people, so the bare
  number is NOT a global identity.
- label_binary from the local class directory (live=0, spoof=1).
- attack_raw = CODE for spoof videos (dataset-native token, preserved verbatim).
- fs* files are exact horizontal flips and bs* files are brightened copies of the s* frame
  with the same name: DERIVED_AUGMENTATION_COPY, indexed but never used as samples.
- CODE HR_1 found under spoof/ is flagged: the published CASIA-FASD protocol (external
  context, not local documentation) lists HR_1 as a genuine high-quality video.
"""
from __future__ import annotations

import re

from .base import (CANONICAL_FRAME_IMAGE, DERIVED_AUGMENTATION_COPY, SUBJECT_RECOVERED,
                   FileClass, UnclassifiedFileError, VideoRecord)

DATASET = "casia_fasd"
_CODE = r"(?P<code>[1-8]|HR_[1-4])"
CANONICAL_RE = re.compile(rf"^(?P<part>train|test)/(?P<cls>live|spoof)/s(?P<subj>\d+)v{_CODE}f(?P<frame>\d+)\.png$")
DERIVED_RE = re.compile(rf"^(?P<part>train|test)/(?P<cls>live|spoof)/(?P<aug>[bf])s(?P<subj>\d+)v{_CODE}f(?P<frame>\d+)\.png$")
LABELS = {"live": 0, "spoof": 1}
# Published CASIA-FASD protocol codes that denote genuine accesses (external context only).
PUBLISHED_GENUINE_CODES = {"1", "2", "HR_1"}


def video_key(part: str, cls: str, subj: str, code: str) -> str:
    return f"{part}/{cls}/s{int(subj)}v{code}"


def classify(rel: str) -> FileClass:
    m = CANONICAL_RE.match(rel)
    if m:
        return FileClass(CANONICAL_FRAME_IMAGE, video_key(m["part"], m["cls"], m["subj"], m["code"]), int(m["frame"]))
    m = DERIVED_RE.match(rel)
    if m:
        return FileClass(DERIVED_AUGMENTATION_COPY, video_key(m["part"], m["cls"], m["subj"], m["code"]), int(m["frame"]))
    raise UnclassifiedFileError(f"{DATASET}: no explicit rule for {rel!r}")


def build_videos(classified: list[tuple[str, FileClass]]) -> list[VideoRecord]:
    seqs: dict[str, dict[int, str]] = {}
    for rel, fc in classified:
        if fc.category != CANONICAL_FRAME_IMAGE:
            continue
        frames = seqs.setdefault(fc.video_key, {})
        if fc.frame_index in frames:
            raise ValueError(f"{DATASET}: duplicate frame index {fc.frame_index} in {fc.video_key}")
        frames[fc.frame_index] = rel
    out = []
    for vid in sorted(seqs):
        part, cls, stem = vid.split("/")
        m = re.fullmatch(r"s(\d+)v(.+)", stem)
        subj, code = m.group(1), m.group(2)
        label = LABELS[cls]
        conflict = None
        if label == 1 and code in PUBLISHED_GENUINE_CODES:
            conflict = (f"CASIA_CODE_{code}_IN_SPOOF_DIR: local folder says spoof; published CASIA-FASD "
                        f"protocol lists code {code} as genuine. Owner decision required.")
        elif label == 0 and code not in PUBLISHED_GENUINE_CODES:
            conflict = (f"CASIA_CODE_{code}_IN_LIVE_DIR: local folder says live; published CASIA-FASD "
                        f"protocol lists code {code} as an attack. Owner decision required.")
        out.append(VideoRecord(
            dataset=DATASET, video_id=vid,
            subject_id_raw=f"{part}_s{int(subj)}", subject_id_status=SUBJECT_RECOVERED,
            subject_id_source="path: native partition dir + filename s{N} (numbers restart per partition)",
            label_binary=label, label_source=f"local class directory '{cls}/'",
            attack_raw=code if label == 1 else None,
            native_protocol_split=part,
            native_meta={"partition": part, "class_dir": cls, "subject_number": int(subj), "video_code": code},
            source_kind="image_sequence", source_path=f"{part}/{cls}/s{subj}v{code}f{{frame}}.png",
            frame_files=seqs[vid], label_conflict=conflict,
        ))
    return out
