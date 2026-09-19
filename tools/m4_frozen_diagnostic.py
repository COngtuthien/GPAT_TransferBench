"""Diagnostic evaluation of the FROZEN pair contract on a deterministic real-data sample.

    <m2 venv>/bin/python tools/m4_frozen_diagnostic.py

Computes d_pose / d_scale / d_luma / d_pair under the frozen Q-24..Q-27 contract on a small
hash-selected sample, purely to confirm the formulas behave as the decisions say. **No pair
membership is persisted** and no formula is changed on the basis of what is observed. Results are
merged into outputs/audit/M4_PAIR_PREFLIGHT.json under `frozen_diagnostic`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P  # noqa: E402
from gpatbench.preprocess import m2b  # noqa: E402

AUDIT = ROOT / "outputs/audit"
DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
SALT = "gpatbench.m4_frozen_diag.v1|"
PER_DATASET_SPLIT = 40          # sampled sources per dataset x split


def main() -> int:
    rows = pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()
    acct = {r["sample_id"]: r for r in
            pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet").to_pylist()}
    xcfg = yaml.safe_load((ROOT / "configs/execution/m2b_laptop_external_storage.yaml").read_text())
    roots = m2b.resolve_roots(xcfg)["roots"]

    # ---- pose for every row (frozen M2 cache; FaceXFormer is never rerun) ----
    pose = {}
    d = roots["geometry_cache_root"] / "pose_pitch_yaw_roll_rad"
    for f in sorted(d.glob("*.index.json")):
        meta = json.loads(f.read_text())
        blob = (d / meta["shard"]).read_bytes()
        for r in meta["rows"]:
            pose[r["sample_id"]] = tuple(float(x) for x in np.load(
                io.BytesIO(blob[r["byte_offset"]:r["byte_offset"] + r["byte_length"]]), allow_pickle=False))

    def face_fraction(r):
        a = acct[r["sample_id"]]
        if r["dataset"] == "casia_fasd":
            return P.CASIA_FACE_AREA_FRACTION            # DEV-018
        h, w = (int(v) for v in a["frame_hw"].split("x"))
        return P.face_area_fraction(r["dataset"], json.loads(a["bbox"]), float(w), float(h))

    def sample(r, luma=None):
        return P.Sample(r["sample_id"], r["dataset"], r["split"], int(r["label_binary"]),
                        r["video_id"], r["subject_id_global"], r["content_group_id"],
                        r["attack_macro"], r["attack_raw"], r["sha256"],
                        pose.get(r["sample_id"]), face_fraction(r), luma)

    # ---- pose statistics: per dataset, TRAIN rows only ----
    all_samples = [sample(r) for r in rows]
    stats = {ds: P.fit_pose_stats(all_samples, ds) for ds in DATASETS}

    import cv2

    def luma_of(r):
        p = roots["faces_256_root"] / r["dataset"] / f"{r['sample_id']}.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return P.luma_mean_from_rgb(np.ascontiguousarray(bgr[:, :, ::-1]))

    out = {"salt": SALT, "sources_per_dataset_split": PER_DATASET_SPLIT,
           "pose_stats": {ds: {"n_train_complete": s.n_train_complete,
                               "mean": [round(x, 9) for x in s.mean],
                               "std_population": [round(x, 9) for x in s.std]}
                          for ds, s in stats.items()},
           "membership_persisted": False}
    for ds in DATASETS:
        entry = {}
        for split in P.SPLITS_WITH_PAIRS:
            pool = [r for r in rows if r["dataset"] == ds and r["split"] == split]
            spoof = sorted([r for r in pool if r["label_binary"] == 1],
                           key=lambda r: hashlib.sha256((SALT + r["sample_id"]).encode()).hexdigest())
            live = [r for r in pool if r["label_binary"] == 0]
            live_s = {r["sample_id"]: sample(r) for r in live}
            lum_cache = {}
            comps = {"d_pose": [], "d_scale": [], "d_luma": [], "d_pair": []}
            for r in spoof[:PER_DATASET_SPLIT]:
                src = sample(r, luma_of(r))
                cands = P.select_candidates(src, P.eligible_targets(src, list(live_s.values())))
                for t in cands:
                    if t.sample_id not in lum_cache:
                        lum_cache[t.sample_id] = luma_of(next(x for x in live if x["sample_id"] == t.sample_id))
                    tt = dataclasses.replace(t, luma_mean=lum_cache[t.sample_id])
                    dd = P.d_pair(src, tt, stats[ds])
                    for k in comps:
                        comps[k].append(dd[k])
            entry[split] = {k: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}
                            for k, v in comps.items()}
            entry[split]["all_finite"] = all(np.isfinite(v).all() for v in comps.values())
            entry[split]["all_non_negative"] = all(min(v) >= 0.0 for v in comps.values())
            if split == "TRAIN":
                entry["d_scale_unique_values"] = sorted({round(x, 15) for x in comps["d_scale"]})[:3] \
                    if ds == "casia_fasd" else None
        out[ds] = entry
    # CASIA: the frozen adaptation must make d_scale identically zero
    uniq = sorted({v for split in P.SPLITS_WITH_PAIRS
                   for v in (out["casia_fasd"][split].get("d_scale", {}).get("min"),
                             out["casia_fasd"][split].get("d_scale", {}).get("max")) if v is not None})
    out["casia_fasd"]["d_scale_unique_values"] = [0.0] if uniq == [0.0] else uniq

    rep_path = AUDIT / "M4_PAIR_PREFLIGHT.json"
    rep = json.loads(rep_path.read_text())
    rep["frozen_diagnostic"] = out
    rep_path.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({ds: {s: {k: round(out[ds][s][k]["median"], 5) for k in
                               ("d_pose", "d_scale", "d_luma", "d_pair")}
                           for s in P.SPLITS_WITH_PAIRS} for ds in DATASETS}, indent=1))
    print("CASIA d_scale unique:", out["casia_fasd"]["d_scale_unique_values"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
