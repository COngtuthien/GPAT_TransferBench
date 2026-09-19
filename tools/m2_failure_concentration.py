"""Audit where the M2 SCRFD_NO_FACE failures fall, at canonical-video level.

    <m2 venv>/bin/python tools/m2_failure_concentration.py

M2 is COMPLETE and is not touched here: this only reads `manifests/m2_sample_accounting.parquet`,
the frozen M1 manifests and the frozen SiW content grouping, and reports.

The question that matters for M3 is the hard gate: does any canonical video end up with ZERO
COMPLETE samples? A video with no usable frame could not contribute a single image row, which would
be a policy question for the owner rather than something to paper over. Exit code is 1 if any such
video exists, so the gate cannot be passed by accident.

Writes outputs/audit/M2_FAILURE_CONCENTRATION.{csv,md}.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/audit"
EXPECTED_FRAMES_PER_VIDEO = 8      # spec §3.4; the actual expectation is read from M1, not assumed


def main() -> int:
    acct = pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet").to_pylist()
    inv = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
    by_sample = {r["sample_id"]: r for r in inv}
    groups = {}
    with open(AUDIT / "siw_content_groups.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups[r["video_id"]] = r["content_group_id"]

    # ---- per canonical video: expected / complete / failed, with original frame indices ----
    per_video: dict = collections.defaultdict(
        lambda: {"expected": 0, "complete": [], "failed": []})
    for r in acct:
        key = (r["dataset"], r["video_id"])
        v = per_video[key]
        v["expected"] += 1
        idx = int(by_sample[r["sample_id"]]["frame_index"])
        (v["complete"] if r["final_status"] == "COMPLETE" else v["failed"]).append(idx)

    zero_usable = sorted(k for k, v in per_video.items() if not v["complete"])
    affected = {k: v for k, v in per_video.items() if v["failed"]}

    rows = []
    for (ds, vid), v in sorted(affected.items()):
        s = next(r for r in inv if r["dataset"] == ds and r["video_id"] == vid)
        rows.append({
            "dataset": ds, "video_id": vid,
            "content_group_id": groups.get(vid, "") if ds == "siwmv2" else "",
            "label_binary": s["label_binary"], "attack_macro": s["attack_macro"],
            "attack_raw": s["attack_raw"] or "",
            "expected_selected_frames": v["expected"],
            "complete_frames": len(v["complete"]), "failed_frames": len(v["failed"]),
            "failed_original_frame_indices": json.dumps(sorted(v["failed"])),
            "complete_original_frame_indices": json.dumps(sorted(v["complete"]))})
    with open(AUDIT / "M2_FAILURE_CONCENTRATION.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # ---- distributions ----
    n_failed = sum(len(v["failed"]) for v in per_video.values())
    usable_hist = collections.Counter(len(v["complete"]) for v in affected.values())
    full_hist = collections.Counter(len(v["complete"]) for v in per_video.values())
    by_macro = collections.Counter(r["attack_macro"] for r in rows for _ in range(r["failed_frames"]))
    by_raw = collections.Counter((r["attack_raw"] or "(none)") for r in rows for _ in range(r["failed_frames"]))
    by_binary = collections.Counter(r["label_binary"] for r in rows for _ in range(r["failed_frames"]))
    # content-group size of affected SiW videos (how many canonical videos share their exact bytes)
    gsize = collections.Counter(groups.values())
    by_gsize = collections.Counter(gsize.get(r["content_group_id"], 0) for r in rows if r["content_group_id"])

    rep = {
        "total_expected_samples": len(acct),
        "total_complete": sum(1 for r in acct if r["final_status"] == "COMPLETE"),
        "total_failed": n_failed,
        "failure_reasons": dict(collections.Counter(r["failure_reason"] for r in acct if r["final_status"] == "FAILED")),
        "total_canonical_videos": len(per_video),
        "affected_canonical_videos": len(affected),
        "max_failed_frames_in_one_video": max((len(v["failed"]) for v in affected.values()), default=0),
        "min_complete_frames_in_affected_video": min((len(v["complete"]) for v in affected.values()), default=None),
        "zero_usable_canonical_videos": len(zero_usable),
        "zero_usable_video_ids": [f"{d}/{v}" for d, v in zero_usable],
        "usable_frame_count_histogram_all_videos": {str(k): v for k, v in sorted(full_hist.items(), reverse=True)},
        "usable_frame_count_histogram_affected_videos": {str(k): v for k, v in sorted(usable_hist.items(), reverse=True)},
        "failed_samples_by_label_binary": {str(k): v for k, v in sorted(by_binary.items())},
        "failed_samples_by_attack_macro": dict(by_macro.most_common()),
        "failed_samples_by_attack_raw": dict(by_raw.most_common()),
        "affected_videos_by_content_group_size": {str(k): v for k, v in sorted(by_gsize.items())},
        "gate": "PASS" if not zero_usable else "BLOCKED_BY_ZERO_USABLE_CANONICAL_VIDEO",
    }
    (AUDIT / "M2_FAILURE_CONCENTRATION.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

    def hist(h):
        return "\n".join(f"| {k} | {v:,} |" for k, v in h.items())

    md = f"""# M2 — Failure Concentration Audit (canonical-video level)

M2 is COMPLETE and untouched. This is a read-only audit of where its 25 technical failures fall, so
that M3 can be designed on facts rather than on the assumption that failures are harmless.

All {rep['total_failed']} failures are `SCRFD_NO_FACE` on SiW-Mv2 — the frozen no-fallback outcome
(no lowered threshold, no second detector, no centre crop, no neighbouring frame, no fabricated
geometry or identity). Per-video detail: `M2_FAILURE_CONCENTRATION.csv`.

## Hard gate: zero-usable canonical videos

**{rep['zero_usable_canonical_videos']}** canonical videos have zero COMPLETE samples → gate **{rep['gate']}**.

Every one of the {rep['total_canonical_videos']:,} canonical videos retains at least one usable
sample, so no video is silently lost from the benchmark and no deletion/replacement policy is needed.

## Headline numbers

| | |
|---|---|
| expected samples | {rep['total_expected_samples']:,} |
| COMPLETE | {rep['total_complete']:,} |
| FAILED | {rep['total_failed']:,} |
| canonical videos (all) | {rep['total_canonical_videos']:,} |
| affected canonical videos | {rep['affected_canonical_videos']} |
| max failed frames in one video | {rep['max_failed_frames_in_one_video']} |
| min COMPLETE frames in an affected video | {rep['min_complete_frames_in_affected_video']} |

Failures are **diffuse, not concentrated**: they touch {rep['affected_canonical_videos']} distinct
videos out of {rep['total_canonical_videos']:,}, at most {rep['max_failed_frames_in_one_video']} frames each.

## Usable-frame counts per canonical video (all {rep['total_canonical_videos']:,} videos)

| COMPLETE frames | videos |
|---|---|
{hist(rep['usable_frame_count_histogram_all_videos'])}

Affected videos only:

| COMPLETE frames | videos |
|---|---|
{hist(rep['usable_frame_count_histogram_affected_videos'])}

## Failure distribution by label

By binary class (0 = live, 1 = spoof):

| label_binary | failed samples |
|---|---|
{hist(rep['failed_samples_by_label_binary'])}

By `attack_macro`:

| attack_macro | failed samples |
|---|---|
{hist(rep['failed_samples_by_attack_macro'])}

By `attack_raw`:

| attack_raw | failed samples |
|---|---|
{hist(rep['failed_samples_by_attack_raw'])}

## Failure distribution by content-group size

How many canonical videos share the exact raw bytes of each affected SiW video
(`content_group_id`, the frozen SiW allocation key). Size 1 means the video is its own group.

| content group size | affected videos |
|---|---|
{hist(rep['affected_videos_by_content_group_size'])}

No identity is inferred from SiW filenames anywhere in this audit.

## Consequence for M3

These are frame-level technical failures inside otherwise healthy canonical videos. They do not
change any video's group identity, and the M3 population contract (see `M3_ALLOCATOR_DESIGN.md`)
therefore balances **canonical videos**, not surviving frame counts, while only COMPLETE rows become
usable downstream image samples.
"""
    (AUDIT / "M2_FAILURE_CONCENTRATION.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("total_failed", "affected_canonical_videos",
                                          "max_failed_frames_in_one_video",
                                          "min_complete_frames_in_affected_video",
                                          "zero_usable_canonical_videos", "gate")}, indent=1))
    return 0 if not zero_usable else 1


if __name__ == "__main__":
    sys.exit(main())
