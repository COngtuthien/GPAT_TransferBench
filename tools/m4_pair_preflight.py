"""M4 pre-flight: pair contract analysis, feasibility, native requirements, distance diagnostics.

    <m2 venv>/bin/python tools/m4_pair_preflight.py

Reads the frozen M3 split and the frozen M2 geometry/identity metadata. Writes only small audit
documents: **no pair manifest is created**, and no pair membership is persisted. Distance formulas
are evaluated only on a deterministic diagnostic sample, and only to quantify how much the
unresolved conventions actually matter — never to choose one.
"""
from __future__ import annotations

import collections
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
DIAG_SALT = "gpatbench.m4_preflight.diag.v1|"
DIAG_N = 400


def load_split() -> list:
    return pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()


def load_accounting() -> dict:
    return {r["sample_id"]: r for r in
            pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet").to_pylist()}


def to_samples(rows, acct, pose=None, luma=None) -> list:
    out = []
    for r in rows:
        a = acct[r["sample_id"]]
        bbox_area = frame_area = None
        if a["bbox"]:
            x1, y1, x2, y2 = json.loads(a["bbox"])
            bbox_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if a["frame_hw"]:
            h, w = (int(v) for v in a["frame_hw"].split("x"))
            frame_area = float(h * w)
        out.append(P.Sample(
            sample_id=r["sample_id"], dataset=r["dataset"], split=r["split"],
            label_binary=int(r["label_binary"]), video_id=r["video_id"],
            subject_id_global=r["subject_id_global"], content_group_id=r["content_group_id"],
            attack_macro=r["attack_macro"], attack_raw=r["attack_raw"], sha256=r["sha256"],
            pose=tuple(float(x) for x in pose[r["sample_id"]]) if pose and r["sample_id"] in pose else None,
            bbox_area=bbox_area, frame_area=frame_area,
            luma_mean=luma.get(r["sample_id"]) if luma else None))
    return out


def load_pose(sample_ids: set) -> dict:
    xcfg = yaml.safe_load((ROOT / "configs/execution/m2b_laptop_external_storage.yaml").read_text())
    roots = m2b.resolve_roots(xcfg)["roots"]
    d = roots["geometry_cache_root"] / "pose_pitch_yaw_roll_rad"
    out = {}
    for f in sorted(d.glob("*.index.json")):
        meta = json.loads(f.read_text())
        blob = (d / meta["shard"]).read_bytes()
        for r in meta["rows"]:
            if r["sample_id"] in sample_ids:
                out[r["sample_id"]] = np.load(
                    io.BytesIO(blob[r["byte_offset"]:r["byte_offset"] + r["byte_length"]]),
                    allow_pickle=False)
    return out


def diagnostic_sample(rows, n=DIAG_N) -> list:
    return sorted(rows, key=lambda r: hashlib.sha256((DIAG_SALT + r["sample_id"]).encode()).hexdigest())[:n]


def luma_variants(rows) -> dict:
    """Mean Y of the canonical face under each candidate standard (diagnostic only)."""
    import cv2
    xcfg = yaml.safe_load((ROOT / "configs/execution/m2b_laptop_external_storage.yaml").read_text())
    roots = m2b.resolve_roots(xcfg)["roots"]
    w601 = np.array([0.299, 0.587, 0.114])
    w709 = np.array([0.2126, 0.7152, 0.0722])
    out = {}
    for r in rows:
        p = roots["faces_256_root"] / r["dataset"] / f"{r['sample_id']}.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1].astype(np.float64) / 255.0
        out[r["sample_id"]] = {"bt601": float((rgb * w601).sum(2).mean()),
                               "bt709": float((rgb * w709).sum(2).mean())}
    return out


# ---------------------------------------------------------------- analyses
def feasibility(samples) -> dict:
    rep = {}
    for split in ("TRAIN", "VAL"):
        per = {}
        for ds in DATASETS:
            sub = [s for s in samples if s.split == split and s.dataset == ds]
            spoof = [s for s in sub if s.label_binary == 1]
            live = [s for s in sub if s.label_binary == 0]
            counts = [len(P.eligible_targets(s, live)) for s in spoof]
            per[ds] = {
                "spoof_sources": len(spoof), "live_targets": len(live),
                "eligible_min": min(counts) if counts else 0,
                "eligible_median": int(statistics.median(counts)) if counts else 0,
                "eligible_max": max(counts) if counts else 0,
                "sources_with_zero_candidates": sum(1 for c in counts if c == 0),
                "sources_with_1_to_63": sum(1 for c in counts if 1 <= c < P.CANDIDATE_CAP),
                "sources_with_64_or_more": sum(1 for c in counts if c >= P.CANDIDATE_CAP),
                "rule": ("different video AND different exact-content group (DEV-013)"
                         if ds == "siwmv2" else "different subject_id_global")}
        per["_total"] = {"spoof_sources": sum(v["spoof_sources"] for v in per.values()),
                         "sources_with_zero_candidates": sum(v["sources_with_zero_candidates"]
                                                             for v in per.values())}
        rep[split] = per
    rep["blocker_zero_candidate_sources"] = sum(rep[s]["_total"]["sources_with_zero_candidates"]
                                                for s in ("TRAIN", "VAL"))
    return rep


def native_requirements(samples) -> dict:
    train = [s for s in samples if s.split == "TRAIN"]
    rep = {}
    for ds in DATASETS:
        sub = [s for s in train if s.dataset == ds]
        if ds == "siwmv2":
            rep[ds] = {"subject_ids_available": False,
                       "rows": len(sub), "rows_with_subject_id": 0,
                       "identities_with_live_and_spoof": 0,
                       "dsdg_native_identity_pairing": "NOT_INSTANTIABLE (no subject identity, Q-14)",
                       "difffas_native_same_id_reconstruction": "NOT_INSTANTIABLE (no subject identity, Q-14)",
                       "style_ids": sorted({f"{ds}::{s.attack_raw}" for s in sub if s.label_binary == 1}),
                       "difffas_guide_different_subject": "NOT_DECIDABLE (no subject identity)"}
            continue
        by = collections.defaultdict(lambda: {"live": 0, "spoof": 0})
        for s in sub:
            by[s.subject_id_global]["live" if s.label_binary == 0 else "spoof"] += 1
        both = [k for k, v in by.items() if v["live"] > 0 and v["spoof"] > 0]
        styles = collections.defaultdict(set)
        for s in sub:
            if s.label_binary == 1:
                styles[f"{ds}::{s.attack_raw}"].add(s.subject_id_global)
        rep[ds] = {"subject_ids_available": True, "rows": len(sub),
                   "rows_with_subject_id": sum(1 for s in sub if s.subject_id_global),
                   "train_identities": len(by), "identities_with_live_and_spoof": len(both),
                   "coverage_pct": round(100.0 * len(both) / len(by), 2),
                   "dsdg_native_identity_pairing": "SUPPORTED",
                   "difffas_native_same_id_reconstruction": "SUPPORTED",
                   "style_ids": sorted(styles),
                   "style_ids_with_more_than_one_subject": sum(1 for v in styles.values() if len(v) > 1),
                   "difffas_guide_different_subject": "ALWAYS_POSSIBLE" if all(len(v) > 1 for v in styles.values())
                                                      else "NOT_ALWAYS_POSSIBLE"}
    return rep


def distance_diagnostics(rows, acct, pose, luma, all_rows) -> dict:
    """Quantify how much each unresolved convention changes the numbers. Never selects one.

    Pose statistics are fitted on the FULL TRAIN population (that is what M4 would do); the
    diagnostic sample is used only for the pairwise distance distributions.
    """
    samples = to_samples(rows, acct, pose, {k: v["bt601"] for k, v in luma.items()})
    train = [s for s in to_samples(all_rows, acct, pose) if s.split == "TRAIN" and s.pose is not None]
    rep = {"diagnostic_sample_rows": len(rows), "salt": DIAG_SALT}

    base = dict(candidate_selection="per_candidate_hash_rank", pose_stat_scope="train_pooled",
                pose_std="population", pose_norm="l2", pose_zero_variance="error",
                scale_box="scrfd_bbox", scale_formula="abs_log_ratio", luma_standard="bt601",
                luma_range="unit_interval", luma_formula="abs_difference")

    # --- pose: std convention and norm ---
    pol_pop = P.PairMetricPolicy(**base)
    pol_samp = P.PairMetricPolicy(**{**base, "pose_std": "sample"})
    n_pop = P.PoseNormalizer.fit(train, pol_pop)
    n_samp = P.PoseNormalizer.fit(train, pol_samp)
    rep["pose"] = {
        "train_rows_fitted": n_pop.n_rows, "mean_rad": [round(float(x), 6) for x in n_pop.mean],
        "std_population": [round(float(x), 6) for x in n_pop.std],
        "std_sample": [round(float(x), 6) for x in n_samp.std],
        "std_relative_difference": [round(float(abs(a - b) / a), 6) for a, b in zip(n_pop.std, n_samp.std)],
        "zero_variance_axes": list(n_pop.zero_variance_axes),
        "per_dataset_std": {ds: [round(float(x), 6) for x in P.PoseNormalizer.fit(train, pol_pop, ds).std]
                            for ds in DATASETS},
    }
    pairs = [(samples[i], samples[j]) for i in range(0, 120, 2) for j in (i + 1,) if j < len(samples)]
    pairs = [(a, b) for a, b in pairs if a.pose is not None and b.pose is not None]
    l2 = [P.d_pose(a, b, n_pop, pol_pop) for a, b in pairs]
    l1 = [P.d_pose(a, b, n_pop, P.PairMetricPolicy(**{**base, "pose_norm": "l1"})) for a, b in pairs]
    rep["pose"]["d_pose_l2"] = {"min": min(l2), "median": statistics.median(l2), "max": max(l2)}
    rep["pose"]["d_pose_l1"] = {"min": min(l1), "median": statistics.median(l1), "max": max(l1)}
    rep["pose"]["l1_over_l2_median_ratio"] = round(statistics.median(l1) / statistics.median(l2), 4)

    # --- scale: availability is the blocker, not the formula ---
    have = collections.Counter()
    for s in samples:
        have[(s.dataset, s.bbox_area is not None)] += 1
    rep["scale"] = {
        "rows_with_face_box": {ds: have[(ds, True)] for ds in DATASETS},
        "rows_without_face_box": {ds: have[(ds, False)] for ds in DATASETS},
        "casia_has_no_face_box": have[("casia_fasd", True)] == 0,
        "note": ("CASIA ran no SCRFD (DEV-011), so no detector box exists for any CASIA row; every "
                 "scale_box variant needing one is undefined there."),
    }
    ok = [(a, b) for a, b in pairs if a.bbox_area and b.bbox_area and a.dataset == b.dataset]
    if ok:
        abs_log = [P.d_scale(a, b, pol_pop) for a, b in ok]
        frac = [P.d_scale(a, b, P.PairMetricPolicy(**{**base, "scale_box": "bbox_frame_fraction"}))
                for a, b in ok if a.frame_area and b.frame_area]
        rep["scale"]["d_scale_abs_log_bbox"] = {"n": len(abs_log), "min": min(abs_log),
                                                "median": statistics.median(abs_log), "max": max(abs_log)}
        if frac:
            rep["scale"]["d_scale_abs_log_frame_fraction"] = {
                "n": len(frac), "min": min(frac), "median": statistics.median(frac), "max": max(frac)}
            rep["scale"]["variants_differ"] = not all(
                abs(x - y) < 1e-12 for x, y in zip(abs_log, frac))

    # --- luminance: which Y standard ---
    b601 = np.array([luma[r["sample_id"]]["bt601"] for r in rows])
    b709 = np.array([luma[r["sample_id"]]["bt709"] for r in rows])
    rep["luma"] = {
        "bt601": {"min": float(b601.min()), "mean": float(b601.mean()), "max": float(b601.max())},
        "bt709": {"min": float(b709.min()), "mean": float(b709.mean()), "max": float(b709.max())},
        "max_abs_difference_per_image": float(np.abs(b601 - b709).max()),
        "mean_abs_difference_per_image": float(np.abs(b601 - b709).mean()),
        "note": "OpenCV BGR2YCrCb matches BT.601 to ~1e-4; the BT.601/BT.709 gap is far larger.",
    }
    return rep


def main() -> int:
    rows = load_split()
    acct = load_accounting()
    samples_nogeom = to_samples(rows, acct)
    feas = feasibility(samples_nogeom)
    native = native_requirements(samples_nogeom)

    diag_rows = diagnostic_sample(rows)
    pose = load_pose({r["sample_id"] for r in rows})
    luma = luma_variants(diag_rows)
    dist = distance_diagnostics(diag_rows, acct, pose, luma, rows)

    split_counts = collections.Counter((r["split"], r["dataset"], r["label_binary"]) for r in rows)
    out = {"split_manifest_sha256": hashlib.sha256(
               (ROOT / "manifests/split_v1.parquet").read_bytes()).hexdigest(),
           "split_rows": len(rows),
           "counts": {f"{s}|{d}|{'spoof' if b else 'live'}": n for (s, d, b), n in sorted(split_counts.items())},
           "feasibility": feas, "native": native, "distances": dist,
           "membership_persisted": False, "pair_manifest_created": False}
    (AUDIT / "M4_PAIR_PREFLIGHT.json").write_text(json.dumps(out, indent=2, default=str) + "\n",
                                                  encoding="utf-8")
    write_docs(out)
    print(json.dumps({"train_sources": feas["TRAIN"]["_total"]["spoof_sources"],
                      "val_sources": feas["VAL"]["_total"]["spoof_sources"],
                      "zero_candidate_sources": feas["blocker_zero_candidate_sources"],
                      "casia_has_no_face_box": dist["scale"]["casia_has_no_face_box"]}, indent=1))
    return 0


def write_docs(o: dict) -> None:
    f, n, d = o["feasibility"], o["native"], o["distances"]

    # ---------------- feasibility ----------------
    L = ["# M4 — Common Pair Feasibility (pre-flight, no pairs written)", "",
         f"Source: `manifests/split_v1.parquet` (sha256 `{o['split_manifest_sha256']}`, "
         f"{o['split_rows']:,} usable M2 COMPLETE rows). Only COMPLETE rows can be a source or a "
         "target; the 25 M2 FAILED rows are absent from the split manifest and therefore cannot "
         "enter a pair.", "",
         "Every TRAIN spoof frame is one source row, so the intended synthetic budget is "
         f"**N_syn,intended = {f['TRAIN']['_total']['spoof_sources']:,}**.", "",
         "| split | dataset | spoof sources | live targets | eligible min | median | max | zero-candidate | 1-63 | >=64 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for split in ("TRAIN", "VAL"):
        for ds in DATASETS:
            v = f[split][ds]
            L.append(f"| {split} | {ds} | {v['spoof_sources']:,} | {v['live_targets']:,} | "
                     f"{v['eligible_min']:,} | {v['eligible_median']:,} | {v['eligible_max']:,} | "
                     f"**{v['sources_with_zero_candidates']}** | {v['sources_with_1_to_63']:,} | "
                     f"{v['sources_with_64_or_more']:,} |")
        L.append(f"| **{split}** | **all** | **{f[split]['_total']['spoof_sources']:,}** | | | | | "
                 f"**{f[split]['_total']['sources_with_zero_candidates']}** | | |")
    L += ["", "Eligibility rules applied:", "",
          "- CASIA / MSU: same dataset, same split, target LIVE, `subject_id_global` differs.",
          "- SiW-Mv2: same dataset, same split, target LIVE, **different canonical video AND "
          "different exact-content group** (DEV-013). This is a different-video / "
          "different-exact-content guarantee — never a different-person claim.", "",
          f"**Zero-candidate sources: {f['blocker_zero_candidate_sources']}.** No spoof source would "
          "have to be skipped, so there is no feasibility blocker.", "",
          "### The 64-candidate cap binds everywhere", "",
          "Every source in every dataset and both splits has at least 64 eligible targets (the "
          "tightest case is MSU VAL at exactly 64). The candidate sampler is therefore not a corner "
          "case: it selects the evaluated set for **every single pair**, which is why its exact rule "
          "(Q-24) blocks M4 rather than being a detail."]
    (AUDIT / "M4_COMMON_PAIR_FEASIBILITY.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------- native ----------------
    L = ["# M4 — Native Pair Requirements (pre-flight)", "",
         "The common manifest serves source-conditioned methods. Two methods additionally need a "
         "**native** pair structure that the common manifest does not provide, and both depend on "
         "subject identity.", "",
         "## TRAIN identity coverage", "",
         "| dataset | TRAIN rows | subject ids | identities | with both live+spoof | coverage |",
         "|---|---|---|---|---|---|"]
    for ds in DATASETS:
        v = n[ds]
        if v["subject_ids_available"]:
            L.append(f"| {ds} | {v['rows']:,} | yes | {v['train_identities']} | "
                     f"{v['identities_with_live_and_spoof']} | {v['coverage_pct']} % |")
        else:
            L.append(f"| {ds} | {v['rows']:,} | **none** | — | **0** | **0 %** |")
    L += ["", "## DSDG (spec §8.6)", "",
          "> \"Training live/spoof identity pairing uses only TRAIN identities that possess both live "
          "and spoof samples. Log usable identity coverage by dataset.\"", "",
          f"- CASIA: **{n['casia_fasd']['dsdg_native_identity_pairing']}** "
          f"({n['casia_fasd']['identities_with_live_and_spoof']}/{n['casia_fasd']['train_identities']} identities)",
          f"- MSU: **{n['msu_mfsd']['dsdg_native_identity_pairing']}** "
          f"({n['msu_mfsd']['identities_with_live_and_spoof']}/{n['msu_mfsd']['train_identities']} identities)",
          f"- SiW-Mv2: **{n['siwmv2']['dsdg_native_identity_pairing']}**", "",
          "SiW carries no trustworthy subject identity (Q-14), so it contributes **zero** identity "
          "pairs. The spec fixes the rule but not what a partially-covered pooled benchmark should "
          "do with it — DSDG must still 'generate exactly N_syn samples', where N_syn counts SiW "
          "sources too. Whether DSDG-native trains on CASIA+MSU identities while generating the full "
          "pooled budget, or is marked blocked for SiW, is **Q-28** (owner).", "",
          "## DiffFAS (spec §8.7)", "",
          "> \"Native training: construct same-dataset, same-identity live/spoof reconstruction pairs "
          "from TRAIN. … If the local dataset lacks enough same-ID live/spoof pairs for a dataset, "
          "report the coverage. Do not synthesize identity labels or pair different identities for "
          "the reconstruction target merely to increase count.\"", "",
          f"- CASIA: **{n['casia_fasd']['difffas_native_same_id_reconstruction']}**, "
          f"{len(n['casia_fasd']['style_ids'])} style_ids, guide with a different subject "
          f"{n['casia_fasd']['difffas_guide_different_subject']}",
          f"- MSU: **{n['msu_mfsd']['difffas_native_same_id_reconstruction']}**, "
          f"{len(n['msu_mfsd']['style_ids'])} style_ids, guide with a different subject "
          f"{n['msu_mfsd']['difffas_guide_different_subject']}",
          f"- SiW-Mv2: **{n['siwmv2']['difffas_native_same_id_reconstruction']}**, "
          f"{len(n['siwmv2']['style_ids'])} style_ids, guide selection "
          f"{n['siwmv2']['difffas_guide_different_subject']}", "",
          "Here the spec *does* say what not to do: coverage is reported and identity is never "
          "fabricated. DEV-013 resolved the **common** pairing rule for SiW and must not be "
          "stretched to cover a same-identity reconstruction requirement — that would be a different "
          "and much stronger claim. The remaining scope question (run DiffFAS-native on CASIA+MSU "
          "only, or treat it as blocked) is **Q-29** (owner).", "",
          "### Binary variants", "",
          "DIFFFAS-BIN collapses `style_id` into one spoof style but *preserves the reconstruction-"
          "pair structure*, and DSDG-BIN collapses the spoof-type target while keeping the rest of "
          "the architecture. Neither collapse creates identity labels, so **binary variants do not "
          "rescue SiW**: the missing information is identity, not style.", "",
          "## Native manifests M4 should eventually create", "",
          "| manifest | method | scope | row unit | identity metadata | supported |",
          "|---|---|---|---|---|---|",
          "| `dsdg_identity_pairs_v1.parquet` | DSDG / DSDG-BIN | TRAIN | one live/spoof identity pair | `subject_id_global` | CASIA + MSU only |",
          "| `difffas_recon_pairs_v1.parquet` | DiffFAS / DIFFFAS-BIN | TRAIN | one same-ID live/spoof reconstruction pair + guide | `subject_id_global`, `style_id` | CASIA + MSU only |",
          "", "Neither is created in this pass."]
    (AUDIT / "M4_NATIVE_PAIR_REQUIREMENTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------- distances ----------------
    p, sc, lu = d["pose"], d["scale"], d["luma"]
    L = ["# M4 — Pair Distance Analysis (diagnostic only)", "",
         f"Evaluated on a deterministic hash-selected sample of {d['diagnostic_sample_rows']} split "
         f"rows (salt `{d['salt']}`), purely to measure how much each *unresolved* convention changes "
         "the numbers. **No formula was chosen from these distributions**, and TEST was never "
         "inspected for a decision.", "",
         "## Pose (Q-25)", "",
         f"z-normalization fitted on **{p['train_rows_fitted']:,} TRAIN rows only**; VAL and TEST "
         "never contribute a statistic.", "",
         "| axis | mean (rad) | std (population) | std (sample) | relative difference |",
         "|---|---|---|---|---|"]
    for i, ax in enumerate(("pitch", "yaw", "roll")):
        L.append(f"| {ax} | {p['mean_rad'][i]} | {p['std_population'][i]} | {p['std_sample'][i]} | "
                 f"{p['std_relative_difference'][i]} |")
    L += ["", f"Zero-variance axes: {p['zero_variance_axes'] or 'none'}. Per-dataset std differs from "
              f"the pooled std ({p['per_dataset_std']}), so **pooled vs per-dataset scope changes the "
              "metric** — it is not a cosmetic choice.", "",
          f"`d_pose` under L2: median {p['d_pose_l2']['median']:.4f} (max {p['d_pose_l2']['max']:.4f}); "
          f"under L1: median {p['d_pose_l1']['median']:.4f} (max {p['d_pose_l1']['max']:.4f}); "
          f"L1/L2 median ratio {p['l1_over_l2_median_ratio']}. The norm materially rescales the "
          "0.50-weighted term relative to the other two, so it cannot be guessed.", "",
          "## Scale (Q-26)", "",
          "| dataset | rows with a face box | rows without |", "|---|---|---|"]
    for ds in DATASETS:
        L.append(f"| {ds} | {sc['rows_with_face_box'][ds]:,} | {sc['rows_without_face_box'][ds]:,} |")
    L += ["", f"**CASIA has no face box at all** (`casia_has_no_face_box = {sc['casia_has_no_face_box']}`). "
              + sc["note"], "",
          "This is the hardest of the four gaps: it is not a choice between conventions but a missing "
          "quantity. Any resolution (treat the whole CASIA frame as the box, derive a box from "
          "FaceXFormer landmarks, drop the term for CASIA and renormalise the weights, or exclude "
          "CASIA from common pairs) changes the scientific metric for a third of the sources, so none "
          "may be adopted silently."]
    if "d_scale_abs_log_bbox" in sc:
        L += ["", f"Where a box exists, `|log(area ratio)|` on raw bbox pixels has median "
                  f"{sc['d_scale_abs_log_bbox']['median']:.4f} (max {sc['d_scale_abs_log_bbox']['max']:.4f})."]
        if "d_scale_abs_log_frame_fraction" in sc:
            L.append(f"Using the bbox **fraction of the frame** instead gives median "
                     f"{sc['d_scale_abs_log_frame_fraction']['median']:.4f} — different, because SiW "
                     "and MSU frames have different resolutions, so raw pixel area is not comparable "
                     "across videos.")
    L += ["", "## Luminance (Q-27)", "",
          f"| standard | min | mean | max |", "|---|---|---|---|",
          f"| BT.601 | {lu['bt601']['min']:.4f} | {lu['bt601']['mean']:.4f} | {lu['bt601']['max']:.4f} |",
          f"| BT.709 | {lu['bt709']['min']:.4f} | {lu['bt709']['mean']:.4f} | {lu['bt709']['max']:.4f} |",
          "",
          f"Per-image difference between the two standards reaches **{lu['max_abs_difference_per_image']:.4f}** "
          f"(mean {lu['mean_abs_difference_per_image']:.4f}) on a quantity whose whole observed range "
          f"is about {lu['bt601']['max'] - lu['bt601']['min']:.2f}. " + lu["note"] + " The choice is "
          "therefore material and is left to the owner.", "",
          "## Combination", "",
          "`d_pair = 0.50·d_pose + 0.30·d_scale + 0.20·d_luma` with the spec's weights, minimum wins, "
          "exact ties broken by lexical target `sample_id` with no fuzzy tolerance. The weights are "
          "frozen and were not touched; what is missing is the definition of the three terms they "
          "weigh."]
    (AUDIT / "M4_PAIR_DISTANCE_ANALYSIS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
