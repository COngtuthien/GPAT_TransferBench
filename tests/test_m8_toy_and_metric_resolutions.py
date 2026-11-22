"""M8 toy-subset planner (metadata only) and the owner-selected metric directions D7 (EER), D8 (spoof F1), D9 (frame
rates at the video threshold), plus the TRAIN/VAL-only firewall read. Expected values are derived in comments.
"""
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import firewall as F  # noqa: E402
from gpatbench.downstream import metrics as M  # noqa: E402
from gpatbench.downstream import toy_subset as T  # noqa: E402

GE, GT = M.DECISION_RULES


class MetricResolutions(unittest.TestCase):
    def test_01_eer_linear_interpolation(self):
        # live {0.2,0.6,0.7}, spoof {0.5}; candidates 0, .35, .55, .65, 1. Under >=:
        # t=.35 -> APCER 0, BPCER 2/3 (d < 0); t=.55 -> APCER 1, BPCER 2/3 (d > 0).
        # crossing: alpha = (2/3 - 0) / ((1 - 0) - (2/3 - 2/3)) = 2/3 -> EER = 2/3
        self.assertEqual(M.eer_linear_interpolation([0, 0, 0, 1], [0.2, 0.6, 0.7, 0.5], rule=GE), Fraction(2, 3))
        # exact crossing: y [0,0,1,1], s [.1,.4,.35,.8]; t=.375 gives APCER = BPCER = 1/2
        self.assertEqual(M.eer_linear_interpolation([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], rule=GE), Fraction(1, 2))
        # perfectly separated: EER 0
        self.assertEqual(M.eer_linear_interpolation([0, 1], [0.3, 0.7], rule=GT), 0)
        with self.assertRaises(TypeError):
            M.eer_linear_interpolation([0, 1], [0.3, 0.7])

    def test_02_f1_spoof(self):
        # TP 1, FP 1, FN 1 -> 2 / (2 + 1 + 1) = 1/2
        self.assertEqual(M.f1_spoof([0, 0, 1, 1], [1, 0, 1, 0]), Fraction(1, 2))
        self.assertIsNone(M.f1_spoof([0, 0], [0, 0]))

    def test_03_frame_rates_use_video_threshold(self):
        frames = [{"y_true": 0, "p_frame": 0.2}, {"y_true": 0, "p_frame": 0.6},
                  {"y_true": 1, "p_frame": 0.55}, {"y_true": 1, "p_frame": 0.4}]
        r = M.frame_level_rates(frames, 0.5, rule=GE)
        # spoof 0.4 -> live: APCER 1/2; live 0.6 -> spoof: BPCER 1/2; TP 1 FP 1 FN 1 -> F1 1/2
        self.assertEqual((r["APCER"], r["BPCER"], r["ACER"], r["F1_spoof"]),
                         (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)))


class FirewallRead(unittest.TestCase):
    def test_04_permitted_rows_exclude_test(self):
        rows = F.read_permitted_rows(ROOT, ["sample_id"])
        self.assertEqual({r["split"] for r in rows}, {"TRAIN", "VAL"})
        lk = F.load_split_lookup(ROOT)
        self.assertEqual(set(lk.values()), {"TRAIN", "VAL"})
        with self.assertRaises(F.TestFirewallViolation):
            F.assert_allowed(["not-a-train-or-val-sample"], lk, {"TRAIN", "VAL"})


def synth_rows(n_val=6):
    rows = []
    for ds in ("casia_fasd", "msu_mfsd", "siwmv2"):
        for sp, n in (("TRAIN", 4), ("VAL", n_val)):
            for u in range(n):
                for c in (0, 1):
                    unit = f"{ds}::{sp}{u}" if ds != "siwmv2" else f"{ds}::g{sp}{u}c{c}"
                    for f in range(8):
                        rows.append({"sample_id": f"{unit}|{c}|{f}", "dataset": ds, "split": sp, "label_binary": c,
                                     "video_id": f"{unit}|{c}", "subject_id_global": None if ds == "siwmv2" else unit,
                                     "content_group_id": unit if ds == "siwmv2" else f"cg|{unit}|{c}"})
    return rows


def synth_pairs(rows):
    live = sorted({(r["dataset"], r["subject_id_global"] or r["content_group_id"]) for r in rows
                   if r["split"] == "TRAIN" and r["label_binary"] == 0})
    pairs = []
    for i, (ds, u) in enumerate(live * 6):
        pairs.append({"pair_id": f"PTR{i + 1:06d}", "dataset": ds, "split": "TRAIN", "target_live_id": f"t{i}",
                      "source_spoof_id": f"s{i}", "target_subject": None if ds == "siwmv2" else u,
                      "target_content_group_id": u if ds == "siwmv2" else None})
    return pairs


class ToyPlanner(unittest.TestCase):
    def test_05_synthetic_plan_rules(self):
        rows = synth_rows()
        p = T.plan_from_rows(rows, synth_pairs(rows))
        self.assertEqual(p["shortfalls"], {})
        for ds in T.DATASETS:
            sel = set(p["roles"]["VAL/select/0"][ds]["units"]) | set(p["roles"]["VAL/select/1"][ds]["units"])
            hold = set(p["roles"]["VAL/holdout/0"][ds]["units"]) | set(p["roles"]["VAL/holdout/1"][ds]["units"])
            self.assertFalse(sel & hold)
        # ranking is the pinned SHA256 key
        cas = sorted({r["subject_id_global"] for r in rows if r["dataset"] == "casia_fasd" and r["split"] == "TRAIN"},
                     key=lambda u: T.rank_key("casia_fasd", "TRAIN", 0, u))
        self.assertEqual(p["roles"]["TRAIN/train/0"]["casia_fasd"]["units"], sorted(cas[:2]))
        # positions are canonical and E01 operator = position % 8
        self.assertTrue(all(tp["e01_operator_index"] == tp["position"] % 8 for tp in p["toy_pairs"]))

    def test_06_ineligible_live_unit_is_skipped(self):
        rows = synth_rows()
        first = sorted({r["subject_id_global"] for r in rows if r["dataset"] == "msu_mfsd" and r["split"] == "TRAIN"},
                       key=lambda u: T.rank_key("msu_mfsd", "TRAIN", 0, u))[0]
        pairs = [p for p in synth_pairs(rows) if p["target_subject"] != first]
        pairs = [{**p, "pair_id": f"PTR{i + 1:06d}"} for i, p in enumerate(pairs)]
        plan = T.plan_from_rows(rows, pairs)
        self.assertNotIn(first, plan["roles"]["TRAIN/train/0"]["msu_mfsd"]["units"])
        self.assertEqual([s["unit"] for s in plan["skipped_units"]], [first])

    def test_06b_holdout_shortfall_is_reported_not_filled(self):
        # with 5 two-class VAL subjects, VAL select can use 4 distinct subjects (2 live + 2 spoof ranks), leaving 1 for
        # the disjoint holdout: the planner must report the shortfall, never reuse a select unit
        rows = synth_rows(n_val=5)
        p = T.plan_from_rows(rows, synth_pairs(rows))
        self.assertEqual(p["shortfalls"], {"casia_fasd/VAL/holdout/0": 1, "casia_fasd/VAL/holdout/1": 1})
        sel = set(p["roles"]["VAL/select/0"]["casia_fasd"]["units"]) | set(p["roles"]["VAL/select/1"]["casia_fasd"]["units"])
        self.assertEqual(len(sel), 4)
        self.assertFalse(sel & set(p["roles"]["VAL/holdout/0"]["casia_fasd"]["units"]))

    def test_07_refusals(self):
        rows = synth_rows()
        with self.assertRaises(T.ToyPlanError):
            T.plan_from_rows(rows + [{**rows[0], "split": "TEST"}], synth_pairs(rows))
        with self.assertRaises(T.ToyPlanError):
            T.plan_from_rows(rows, list(reversed(synth_pairs(rows))))
        bad = [dict(r) for r in rows]
        bad[0]["subject_id_global"] = None
        with self.assertRaises(T.ToyPlanError):
            T.plan_from_rows(bad, synth_pairs(rows))

    def test_08_frozen_metadata_plan_matches_record(self):
        rec_path = ROOT / "outputs/audit/M8_TOY_SUBSET_FEASIBILITY.json"
        p = T.plan(ROOT)
        self.assertEqual(p["inputs"]["test_rows_read"], 0)
        f = p["feasibility"]
        self.assertEqual((f["train_live_frames"], f["train_real_spoof_frames"], f["toy_pairs"],
                          f["live_pass_full_batches_per_epoch"]), (96, 256, 186, 3))
        self.assertTrue(f["full_batch_composition_satisfiable"])
        self.assertTrue(all(f["select_holdout_disjoint"].values()))
        if rec_path.exists():
            rec = json.loads(rec_path.read_text())
            for k in ("roles", "pooled", "toy_pairs", "feasibility", "shortfalls", "skipped_units"):
                self.assertEqual(rec[k], json.loads(json.dumps(p[k])), k)


if __name__ == "__main__":
    unittest.main()
