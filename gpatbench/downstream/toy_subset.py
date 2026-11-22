"""M8 smoke toy-subset PLANNER (metadata only; reads no image, no TEST row).

Implements the owner-selected smoke directions S2 BINARY_LABEL_PER_DATASET, S3 CONTENT_GROUP_ID, S4 HASH_RANK,
S5 TRAIN_VAL_ONLY_TEST_PROXY and S6 FROZEN_PAIRS_TARGET_IN_TOY with the execution details pinned in
configs/amendments/m8_smoke_test_contract.FINAL_DRAFT.yaml. The plan is a feasibility artifact; it authorizes nothing
and no smoke run may bind it until that contract is owner-approved.

Units: casia_fasd / msu_mfsd use subject_id_global (verified non-null). siwmv2 uses content_group_id, the DEV-012
allocation unit; it is a video-content group and is NOT a subject identity (disclosure required wherever it appears).

Ranking (S4): within (dataset, split, label_binary), units ascend by
SHA256(UTF-8("gpatbench.smoke.v1|" + dataset + "|" + split + "|" + str(label_binary) + "|" + unit_id)) hex.
- TRAIN: first 2 eligible units per class. A live unit is eligible only if it is the target of >= 1 frozen
  pairs_train_v1 pair; ineligible units are skipped and recorded.
- VAL select (checkpoint/threshold): first 2 units per class.
- VAL holdout (TEST stand-in): first 2 units per class in the same order, excluding every unit used in VAL select for
  either class, so select and holdout share no unit (S5).
A selected unit contributes only the frames of the class it was selected for.

Pairs (S6): the frozen pairs whose target unit is a TRAIN live toy unit, each with its canonical 0-based global position
in pairs_train_v1 (sorted by pair_id), so E01 uses operator position % 8 and outputs equal future production outputs.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from .firewall import read_permitted_rows

RANK_PREFIX = "gpatbench.smoke.v1"
UNITS_PER_CLASS = 2
DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
SUBJECT_DATASETS = ("casia_fasd", "msu_mfsd")
PAIRS_PATH = "manifests/pairs_train_v1.parquet"
PAIRS_SHA256 = "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243"
BATCH = {"mixed": {"live": 32, "real_spoof": 16, "synthetic_spoof": 16},
         "real_only": {"live": 32, "real_spoof": 32}}          # frozen downstream_resnet18.yaml


class ToyPlanError(RuntimeError):
    pass


def rank_key(dataset: str, split: str, label: int, unit_id: str) -> str:
    return hashlib.sha256(f"{RANK_PREFIX}|{dataset}|{split}|{label}|{unit_id}".encode("utf-8")).hexdigest()


def _unit(row: dict) -> str:
    return row["content_group_id"] if row["dataset"] == "siwmv2" else row["subject_id_global"]


def plan_from_rows(split_rows: list[dict], pair_rows: list[dict]) -> dict:
    """Pure planner. `split_rows` must hold TRAIN/VAL rows only; `pair_rows` the frozen pair manifest in stored order."""
    if {r["split"] for r in split_rows} - {"TRAIN", "VAL"}:
        raise ToyPlanError("planner received a non-TRAIN/VAL row")
    for r in split_rows:
        if r["dataset"] in SUBJECT_DATASETS and not r["subject_id_global"]:
            raise ToyPlanError(f"{r['dataset']} row without subject_id_global")
        if r["dataset"] == "siwmv2" and not r["content_group_id"]:
            raise ToyPlanError("siwmv2 row without content_group_id")
    ids = [p["pair_id"] for p in pair_rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids) or {p["split"] for p in pair_rows} != {"TRAIN"}:
        raise ToyPlanError("pair manifest is not the sorted, unique, TRAIN-only frozen relation")
    targets = defaultdict(int)
    for p in pair_rows:
        tu = p["target_content_group_id"] if p["dataset"] == "siwmv2" else p["target_subject"]
        targets[(p["dataset"], tu)] += 1

    units = defaultdict(set)
    for r in split_rows:
        units[(r["dataset"], r["split"], r["label_binary"])].add(_unit(r))

    def ranked(ds, sp, c):
        return sorted(units[(ds, sp, c)], key=lambda u: rank_key(ds, sp, c, u))

    sel, skipped = {}, []
    for ds in DATASETS:
        for c in (0, 1):
            chosen = []
            for u in ranked(ds, "TRAIN", c):
                if c == 0 and targets[(ds, u)] == 0:
                    skipped.append({"dataset": ds, "split": "TRAIN", "label_binary": c, "unit": u,
                                    "reason": "not the target of any frozen pair"})
                    continue
                chosen.append(u)
                if len(chosen) == UNITS_PER_CLASS:
                    break
            sel[(ds, "TRAIN", "train", c)] = chosen
        for c in (0, 1):
            sel[(ds, "VAL", "select", c)] = ranked(ds, "VAL", c)[:UNITS_PER_CLASS]
        used = set(sel[(ds, "VAL", "select", 0)]) | set(sel[(ds, "VAL", "select", 1)])
        for c in (0, 1):
            sel[(ds, "VAL", "holdout", c)] = [u for u in ranked(ds, "VAL", c) if u not in used][:UNITS_PER_CLASS]

    short = {f"{k[0]}/{k[1]}/{k[2]}/{k[3]}": len(v) for k, v in sel.items() if len(v) != UNITS_PER_CLASS}
    members = defaultdict(list)
    for r in split_rows:
        for role in ("train", "select", "holdout"):
            key = (r["dataset"], r["split"], role, r["label_binary"])
            if key in sel and _unit(r) in sel[key]:
                members[(r["split"], role, r["label_binary"], r["dataset"])].append(r)

    def summary(rows):
        sids = sorted(r["sample_id"] for r in rows)
        return {"frames": len(rows), "videos": len({(r["dataset"], r["video_id"]) for r in rows}),
                "sample_ids_sha256": hashlib.sha256("\n".join(sids).encode()).hexdigest()}

    roles = {}
    for (sp, role, c, ds), rows in sorted(members.items(), key=lambda kv: tuple(map(str, kv[0]))):
        roles.setdefault(f"{sp}/{role}/{c}", {})[ds] = {"units": sorted(sel[(ds, sp, role, c)]), **summary(rows)}
    pooled = {k: {"frames": sum(v["frames"] for v in d.values()), "videos": sum(v["videos"] for v in d.values())}
              for k, d in roles.items()}

    live_toy = {(ds, u) for (ds, sp, role, c), v in sel.items() if role == "train" and c == 0 for u in v}
    toy_pairs = []
    for i, p in enumerate(pair_rows):
        tu = p["target_content_group_id"] if p["dataset"] == "siwmv2" else p["target_subject"]
        if (p["dataset"], tu) in live_toy:
            toy_pairs.append({"position": i, "pair_id": p["pair_id"], "dataset": p["dataset"],
                              "target_live_id": p["target_live_id"], "source_spoof_id": p["source_spoof_id"],
                              "e01_operator_index": i % 8})

    disjoint = {}
    for ds in DATASETS:
        a = set(sel[(ds, "VAL", "select", 0)]) | set(sel[(ds, "VAL", "select", 1)])
        b = set(sel[(ds, "VAL", "holdout", 0)]) | set(sel[(ds, "VAL", "holdout", 1)])
        disjoint[ds] = not (a & b)

    live = pooled.get("TRAIN/train/0", {}).get("frames", 0)
    spoof = pooled.get("TRAIN/train/1", {}).get("frames", 0)
    steps = live // BATCH["mixed"]["live"]
    feas = {
        "train_live_frames": live, "train_real_spoof_frames": spoof, "toy_pairs": len(toy_pairs),
        "live_pass_full_batches_per_epoch": steps, "live_pass_tail_frames_dropped": live - steps * BATCH["mixed"]["live"],
        "mixed_real_spoof_needed_per_epoch": steps * BATCH["mixed"]["real_spoof"],
        "mixed_synthetic_needed_per_epoch": steps * BATCH["mixed"]["synthetic_spoof"],
        "real_only_spoof_needed_per_epoch": steps * BATCH["real_only"]["real_spoof"],
        "full_batch_composition_satisfiable": steps >= 1 and spoof >= BATCH["real_only"]["real_spoof"]
        and len(toy_pairs) >= BATCH["mixed"]["synthetic_spoof"],
        "replacement_needed_within_epoch": {
            "real_spoof_mixed": steps * BATCH["mixed"]["real_spoof"] > spoof,
            "real_spoof_real_only": steps * BATCH["real_only"]["real_spoof"] > spoof,
            "synthetic_if_no_generation_failure": steps * BATCH["mixed"]["synthetic_spoof"] > len(toy_pairs)},
        "val_both_classes_per_dataset": {
            role: all(roles.get(f"VAL/{role}/{c}", {}).get(ds, {}).get("videos", 0) > 0 for ds in DATASETS for c in (0, 1))
            for role in ("select", "holdout")},
        "select_holdout_disjoint": disjoint,
        "e01_operator_coverage": {str(k): sum(1 for p in toy_pairs if p["e01_operator_index"] == k) for k in range(8)},
    }
    return {"rank_prefix": RANK_PREFIX, "units_per_class": UNITS_PER_CLASS, "shortfalls": short,
            "skipped_units": skipped, "roles": roles, "pooled": pooled, "toy_pairs": toy_pairs, "feasibility": feas,
            "siwmv2_unit_disclosure": "siwmv2 units are content_group_id (DEV-012 video-content groups), not subjects"}


def plan(root: str | Path) -> dict:
    """Hash-checked metadata read (TEST rows filtered by the reader), then plan_from_rows."""
    import pyarrow.parquet as pq
    root = Path(root)
    split_rows = read_permitted_rows(root, ["sample_id", "dataset", "subject_id_global", "content_group_id",
                                            "video_id", "label_binary"])
    pp = root / PAIRS_PATH
    if hashlib.sha256(pp.read_bytes()).hexdigest() != PAIRS_SHA256:
        raise ToyPlanError("pairs_train_v1 SHA256 differs from the frozen value")
    pair_rows = pq.read_table(pp, columns=["pair_id", "dataset", "split", "target_live_id", "source_spoof_id",
                                           "target_subject", "target_content_group_id"]).to_pylist()
    out = plan_from_rows(split_rows, pair_rows)
    out["inputs"] = {"split_rows_read": len(split_rows), "test_rows_read": 0, "pairs_sha256": PAIRS_SHA256}
    return out
