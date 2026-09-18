"""CASIA-FASD adapter for the LOCAL copy (repackaged face-crop frame sequences).

Local layout observed in M1 (no documentation file ships with this copy):
    {train|test}/{live|spoof}/s{N}v{CODE}f{F}.png      canonical frames (112x112 face crops)
    train/live/{b|f}s{N}v{CODE}f{F}.png                 packager-derived copies (DEV-007, approved)
CODE is 1..8 or HR_1..HR_4 (CASIA-FASD native video code). F is the frame index.

Semantics (M1 correction pass; owner-provided external CASIA-FASD protocol evidence, recorded in
outputs/audit/deviation_report.md DEV-010 / Q-12 / Q-13):
- CODE semantics: 1 real_normal, 2 real_low, HR_1 real_high; 3 warped_normal, 4 warped_low,
  HR_2 warped_high; 5 cut_normal, 6 cut_low, HR_3 cut_high; 7 video_normal, 8 video_low, HR_4 video_high.
- label_binary is derived from CODE (live for 1, 2, HR_1). The local packager stored HR_1 under
  spoof/; that folder label is overridden and the disagreement is recorded per video.
- attack_raw = CODE for spoof videos only (live acquisition codes are not attack tokens).
- subject identity: canonical CASIA-FASD identities are train 1..20 and test 21..50, so
  train s{N} -> "N" and test s{N} -> str(20 + N). Numbers outside train 1..20 / test 1..30 are a
  hard error (the mapping would be undefined).
"""
from __future__ import annotations

import re

from .base import (CANONICAL_FRAME_IMAGE, DERIVED_AUGMENTATION_COPY, SUBJECT_RECOVERED,
                   FileClass, UnclassifiedFileError, VideoRecord)

DATASET = "casia_fasd"
_CODE = r"(?P<code>[1-8]|HR_[1-4])"
CANONICAL_RE = re.compile(rf"^(?P<part>train|test)/(?P<cls>live|spoof)/s(?P<subj>\d+)v{_CODE}f(?P<frame>\d+)\.png$")
DERIVED_RE = re.compile(rf"^(?P<part>train|test)/(?P<cls>live|spoof)/(?P<aug>[bf])s(?P<subj>\d+)v{_CODE}f(?P<frame>\d+)\.png$")

CODE_SEMANTICS = {
    "1": "real_normal", "2": "real_low", "HR_1": "real_high",
    "3": "warped_normal", "4": "warped_low", "HR_2": "warped_high",
    "5": "cut_normal", "6": "cut_low", "HR_3": "cut_high",
    "7": "video_normal", "8": "video_low", "HR_4": "video_high",
}
LIVE_CODES = {"1", "2", "HR_1"}
FOLDER_LABEL = {"live": 0, "spoof": 1}
PARTITION_SUBJECTS = {"train": (1, 20, 0), "test": (1, 30, 20)}   # (min N, max N, identity offset)


def video_key(part: str, cls: str, subj: str, code: str) -> str:
    return f"{part}/{cls}/s{int(subj)}v{code}"


def subject_id(part: str, n: int) -> str:
    lo, hi, offset = PARTITION_SUBJECTS[part]
    if not lo <= n <= hi:
        raise ValueError(f"{DATASET}: subject number {n} outside {part} range {lo}..{hi}; identity mapping undefined")
    return str(n + offset)


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
        n, code = int(m.group(1)), m.group(2)
        label = 0 if code in LIVE_CODES else 1
        folder_disagrees = FOLDER_LABEL[cls] != label
        out.append(VideoRecord(
            dataset=DATASET, video_id=vid,
            subject_id_raw=subject_id(part, n), subject_id_status=SUBJECT_RECOVERED,
            subject_id_source="path: native partition + s{N}; train N -> N, test N -> 20+N (canonical CASIA-FASD identities 1..50)",
            label_binary=label,
            label_source=f"CASIA-FASD video code {code} = {CODE_SEMANTICS[code]} (owner-provided protocol evidence, Q-12)"
                         + (f"; local folder '{cls}/' disagrees and is overridden" if folder_disagrees else ""),
            attack_raw=code if label == 1 else None,
            native_protocol_split=part,
            native_meta={"partition": part, "class_dir": cls, "subject_number": n, "video_code": code,
                         "code_semantics": CODE_SEMANTICS[code], "folder_label_overridden": folder_disagrees},
            source_kind="image_sequence", source_path=f"{part}/{cls}/s{m.group(1)}v{code}f{{frame}}.png",
            frame_files=seqs[vid], label_conflict=None,
        ))
    return out
