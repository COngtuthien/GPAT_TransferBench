"""Real-data populations of the M10 scientific detector runs (TRAIN / VAL only; TEST is never parsed).

Source of truth: manifests/split_v1.parquet (hash-checked; TEST rows removed by the parquet row filter in
gpatbench/downstream/firewall.read_permitted_rows) and manifests/m2_sample_accounting.parquet (face_png_sha256,
final_status COMPLETE). Every frame is one sample (spec 3.4: the eight frames of a video are independent training
samples; video score = mean of its frame probabilities at evaluation). Pools, in ascending sample_id order:
  train_live        TRAIN label_binary 0 (5629 frames at split_v1 fb9aeb36)
  train_real_spoof  TRAIN label_binary 1 (8838 frames; the pair sources of every synthetic bank)
  val               VAL frames of both classes (3121 frames, 391 videos) -- checkpoint and threshold selection only
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import firewall

M2_PATH = "manifests/m2_sample_accounting.parquet"
M2_SHA256 = "65bb83e69ea7cebcb37373971e5e8f5248609b7963fd9d32aa1c0ba4377c0662"
EXPECTED = {"train_live": 5629, "train_real_spoof": 8838, "val": 3121, "val_videos": 391}
FRAME_KEYS = ("sample_id", "dataset", "subject_id_global", "video_id", "frame_index", "label_binary", "attack_raw",
              "attack_macro", "split", "content_group_id", "allocation_group_id")
# M3 grouping (outputs/audit/M3_LEAKAGE_AUDIT.json): CASIA-FASD / MSU-MFSD subject-disjoint (subject_id_global);
# SiW-Mv2 canonical-video / content-group-disjoint (content_group_id; subject identity unavailable, owner Q-14 --
# a subject-disjoint claim is FORBIDDEN for SiW-Mv2)
GROUP_KEY = {"casia_fasd": "subject_id_global", "msu_mfsd": "subject_id_global", "siwmv2": "content_group_id"}


class ScienceInputError(RuntimeError):
    pass


def build_populations(repo: str | Path) -> dict:
    import pyarrow.parquet as pq
    repo = Path(repo)
    rows = firewall.read_permitted_rows(repo, list(FRAME_KEYS))
    m2p = repo / M2_PATH
    if hashlib.sha256(m2p.read_bytes()).hexdigest() != M2_SHA256:
        raise ScienceInputError("M2 accounting sha256 differs from the frozen value")
    ids = sorted(r["sample_id"] for r in rows)
    m2 = {r["sample_id"]: r for r in pq.read_table(m2p, columns=["sample_id", "face_png_sha256", "final_status",
                                                               "dataset"],
                                                    filters=[("sample_id", "in", ids)]).to_pylist()}
    pools = {"train_live": [], "train_real_spoof": [], "val": []}
    for r in sorted(rows, key=lambda x: x["sample_id"]):
        a = m2.get(r["sample_id"])
        if a is None or a["final_status"] != "COMPLETE" or a["dataset"] != r["dataset"] or not a["face_png_sha256"]:
            raise ScienceInputError(f"{r['sample_id']} has no COMPLETE M2 face")
        f = {"sample_id": r["sample_id"], "dataset": r["dataset"], "subject_id_global": r["subject_id_global"],
             "video_id": r["video_id"], "frame_index": r["frame_index"], "y_true": int(r["label_binary"]),
             "attack_raw": r["attack_raw"], "attack_macro": r["attack_macro"], "split": r["split"],
             "face_rel_path": f"{r['dataset']}/{r['sample_id']}.png", "face_sha256": a["face_png_sha256"],
             "group": (r["dataset"], r[GROUP_KEY[r["dataset"]]]), "allocation_group_id": r["allocation_group_id"]}
        if f["group"][1] is None:
            raise ScienceInputError(f"{r['sample_id']} has no {GROUP_KEY[r['dataset']]}")
        if r["split"] == "TRAIN":
            pools["train_live" if f["y_true"] == 0 else "train_real_spoof"].append(f)
        elif r["split"] == "VAL":
            pools["val"].append(f)
        else:
            raise firewall.TestFirewallViolation("non-permitted split reached the population builder")
    counts = {k: len(v) for k, v in pools.items()}
    counts["val_videos"] = len({(f["dataset"], f["video_id"]) for f in pools["val"]})
    if counts != EXPECTED:
        raise ScienceInputError(f"population {counts} differs from the frozen split {EXPECTED}")
    train_vids = {(f["dataset"], f["video_id"]) for k in ("train_live", "train_real_spoof") for f in pools[k]}
    val_vids = {(f["dataset"], f["video_id"]) for f in pools["val"]}
    train_grp = {f["group"] for k in ("train_live", "train_real_spoof") for f in pools[k]}
    val_grp = {f["group"] for f in pools["val"]}
    train_alloc = {f["allocation_group_id"] for k in ("train_live", "train_real_spoof") for f in pools[k]}
    val_alloc = {f["allocation_group_id"] for f in pools["val"]}
    if train_vids & val_vids or train_grp & val_grp or train_alloc & val_alloc:
        raise ScienceInputError("TRAIN and VAL share a video, a grouping unit (subject / content group) or an "
                                "allocation group")
    for v in pools.values():
        for f in v:
            del f["group"]
    digest = {k: hashlib.sha256("\n".join(f"{f['sample_id']}|{f['face_sha256']}" for f in v).encode()).hexdigest()
              for k, v in pools.items()}
    return {"pools": pools, "counts": counts, "pool_sha256": digest,
            "split_manifest_sha256": firewall.SPLIT_MANIFEST_SHA256, "m2_sha256": M2_SHA256}
