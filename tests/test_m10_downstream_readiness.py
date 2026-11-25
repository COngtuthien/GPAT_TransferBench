"""M10 downstream ResNet-18 scientific readiness (framework-free; laptop .venv with numpy + pyarrow, no torch).

- metrics (spec 15-16, contract D4/D5/D7/D8) against independent brute-force implementations and analytic fixtures;
- real populations: frozen counts, TRAIN/VAL disjointness under the M3 grouping, TEST never parsed or admitted;
- synthetic-bank bindings of all 21 banks and their refusals;
- reproducibility manifest schema;
- the science precision pin gate (D13) and the matrix invariants.
"""
import copy
import itertools
import json
import math
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import firewall, metrics as M, science_banks as SB, science_inputs as SI  # noqa: E402
from gpatbench.downstream import science_manifest as SM, science_runner as SR  # noqa: E402

GE = "SPOOF_IF_SCORE_GE_THRESHOLD"


# ----------------------------------------------------------------------------------------- independent references
def ref_rates(y, s, t):
    spoof = [si for yi, si in zip(y, s) if yi == 1]
    live = [si for yi, si in zip(y, s) if yi == 0]
    apcer = Fraction(sum(1 for x in spoof if not x >= t), len(spoof))
    bpcer = Fraction(sum(1 for x in live if x >= t), len(live))
    return apcer, bpcer, (apcer + bpcer) / 2


def ref_threshold(y, s):
    u = sorted(set(s))
    cands = sorted(set([(a + b) / 2 for a, b in zip(u, u[1:])] + [0.0, 1.0]))
    best = min(ref_rates(y, s, t)[2] for t in cands)
    tied = [t for t in cands if ref_rates(y, s, t)[2] == best]
    d = min(abs(t - 0.5) for t in tied)
    return [t for t in tied if abs(t - 0.5) == d], best


def ref_auc(y, s):
    pos = [b for a, b in zip(y, s) if a == 1]
    neg = [b for a, b in zip(y, s) if a == 0]
    tot = sum(Fraction(1) if p > n else Fraction(1, 2) if p == n else 0 for p in pos for n in neg)
    return tot / (len(pos) * len(neg))


class Metrics(unittest.TestCase):
    def test_perfect_and_analytic(self):
        y, s = [0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        sel = M.select_threshold(y, s, rule=GE)
        self.assertEqual((sel["threshold"], sel["val_acer"]), (0.5, 0))
        self.assertEqual(M.auc(y, s), 1)
        self.assertEqual(M.eer_linear_interpolation(y, s, rule=GE), 0)
        r = M.rates_from_counts(M.confusion_counts(y, M.predict_spoof(s, 0.5, rule=GE)))
        self.assertEqual((r["APCER"], r["BPCER"], r["ACER"], r["accuracy"]), (0, 0, 0, 1))

    def test_known_confusion(self):
        # 4 live, 4 spoof; at t = 0.5: one spoof below (APCER 1/4), two live at/above (BPCER 2/4)
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        s = [0.1, 0.2, 0.5, 0.6, 0.4, 0.7, 0.8, 0.9]
        r = M.rates_from_counts(M.confusion_counts(y, M.predict_spoof(s, 0.5, rule=GE)))
        self.assertEqual((r["APCER"], r["BPCER"], r["ACER"]), (Fraction(1, 4), Fraction(1, 2), Fraction(3, 8)))
        self.assertEqual(M.f1_spoof(y, M.predict_spoof(s, 0.5, rule=GE)), Fraction(6, 9))
        self.assertEqual(M.auc(y, s), ref_auc(y, s))

    def test_ge_operator_at_threshold(self):
        self.assertEqual(M.predict_spoof([0.5, 0.4999999], 0.5, rule=GE), [1, 0])

    def test_reversed_and_constant_scores_hit_the_documented_tie(self):
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        for s in ([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1], [0.5] * 8):
            with self.assertRaises(M.UnresolvedThresholdTie):
                M.select_threshold(y, s, rule=GE)
            self.assertEqual(ref_threshold(y, s), ([0.0, 1.0], Fraction(1, 2)))
        self.assertEqual(M.auc(y, [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]), 0)
        self.assertEqual(M.auc(y, [0.5] * 8), Fraction(1, 2))

    def test_tied_scores(self):
        y, s = [0, 0, 1, 1], [0.3, 0.6, 0.6, 0.9]
        self.assertEqual(M.auc(y, s), ref_auc(y, s))
        self.assertEqual(M.auc(y, s), Fraction(7, 8))

    def test_brute_force_agreement_random(self):
        import random
        rnd = random.Random(7)
        for trial in range(300):
            n = rnd.randint(2, 9)
            y = [0] * n + [1] * rnd.randint(1, 9)
            s = [round(rnd.random(), rnd.choice([1, 2, 3])) for _ in y]
            tied, best = ref_threshold(y, s)
            if len(tied) == 1:
                sel = M.select_threshold(y, s, rule=GE)
                self.assertEqual((sel["threshold"], sel["val_acer"]), (tied[0], best))
            else:
                with self.assertRaises(M.UnresolvedThresholdTie):
                    M.select_threshold(y, s, rule=GE)
            self.assertEqual(M.auc(y, s), ref_auc(y, s))
            eer = M.eer_linear_interpolation(y, s, rule=GE)
            self.assertTrue(0 <= eer <= 1)

    def test_eer_interpolation_analytic(self):
        # candidates 0, 0.3, 0.5, 0.7, 1 ; APCER-BPCER crosses between two operating points
        y, s = [0, 0, 1, 1], [0.2, 0.6, 0.4, 0.8]
        pts = M.curve_points(y, s, rule=GE)
        eer = M.eer_linear_interpolation(y, s, rule=GE)
        self.assertEqual(eer, Fraction(1, 2))
        self.assertTrue(any(p["APCER"] == p["BPCER"] == eer for p in pts))

    def test_tpr_at_fpr_no_interpolation(self):
        y, s = [0] * 10 + [1] * 10, [i / 20 for i in range(10)] + [0.3 + i / 20 for i in range(10)]
        v = M.tpr_at_fpr(y, s, "0.01", rule=GE)
        best = max(1 - p["APCER"] for p in M.curve_points(y, s, rule=GE) if p["BPCER"] <= Fraction(1, 100))
        self.assertEqual(v, best)

    def test_refusals(self):
        with self.assertRaises(M.MetricError):
            M.select_threshold([0, 0], [0.1, 0.2], rule=GE)            # empty spoof class
        with self.assertRaises(M.MetricError):
            M.auc([1, 1], [0.1, 0.2])
        with self.assertRaises(M.MetricError):
            M.predict_spoof([1.2], 0.5, rule=GE)                       # invalid score range
        with self.assertRaises(M.MetricError):
            M.predict_spoof([float("nan")], 0.5, rule=GE)              # non-finite
        with self.assertRaises(M.MetricError):
            M.sigmoid(float("inf"))
        with self.assertRaises(M.MetricError):
            M.confusion_counts([0, 2], [0, 1])                          # incorrect label
        with self.assertRaises(M.MetricError):
            M.confusion_counts([0, 1], [0])                             # missing prediction
        with self.assertRaises(M.MetricError):
            M.predict_spoof([0.5], 0.5, rule="SPOOF_IF_SCORE_ABOVE")   # no default operator
        with self.assertRaises(M.MetricError):
            M.aggregate_videos([{"dataset": "d", "video_id": "v", "y_true": 0, "p_frame": 0.1},
                                {"dataset": "d", "video_id": "v", "y_true": 1, "p_frame": 0.9}])

    def test_video_aggregation_is_mean_of_own_frames(self):
        fr = [{"dataset": "d", "video_id": "a", "y_true": 1, "p_frame": p} for p in (0.2, 0.4, 0.9)] + \
             [{"dataset": "d", "video_id": "b", "y_true": 0, "p_frame": 0.1}]
        v = {x["video_id"]: x for x in M.aggregate_videos(fr)}
        self.assertAlmostEqual(v["a"]["video_score"], 0.5, places=15)
        self.assertEqual((v["a"]["n_frames"], v["b"]["video_score"]), (3, 0.1))

    def test_seed_aggregation(self):
        r = M.seed_mean_std([0.1, 0.2, 0.3])
        self.assertAlmostEqual(r["mean"], 0.2)
        self.assertAlmostEqual(r["sample_std"], 0.1)
        with self.assertRaises(M.MetricError):
            M.seed_mean_std([0.1, float("nan")])

    def test_checkpoint_rank_min_acer_then_max_auc_then_earliest(self):
        es = [{"epoch": 0, "val_acer": "1/4", "val_auc": "9/10"}, {"epoch": 1, "val_acer": "1/8", "val_auc": "4/5"},
              {"epoch": 2, "val_acer": "1/8", "val_auc": "9/10"}, {"epoch": 3, "val_acer": "1/8", "val_auc": "9/10"}]
        self.assertEqual(min(es, key=SR._rank)["epoch"], 2)


# ----------------------------------------------------------------------------------------- data
class Populations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = SI.build_populations(ROOT)

    def test_frozen_counts_and_composition(self):
        self.assertEqual(self.pop["counts"], {"train_live": 5629, "train_real_spoof": 8838, "val": 3121,
                                              "val_videos": 391})
        p = self.pop["pools"]
        self.assertTrue(all(f["y_true"] == 0 and f["split"] == "TRAIN" for f in p["train_live"]))
        self.assertTrue(all(f["y_true"] == 1 and f["split"] == "TRAIN" for f in p["train_real_spoof"]))
        self.assertEqual({f["split"] for f in p["val"]}, {"VAL"})
        self.assertEqual(5629 // 32, 175)                               # D1 steps per epoch

    def test_pair_sources_are_the_train_spoof_frames(self):
        import pyarrow.parquet as pq
        pairs = pq.read_table(str(ROOT / "manifests/pairs_train_v1.parquet"),
                              columns=["target_live_id", "source_spoof_id"]).to_pylist()
        self.assertEqual({x["source_spoof_id"] for x in pairs}, {f["sample_id"] for f in self.pop["pools"]["train_real_spoof"]})
        self.assertTrue({x["target_live_id"] for x in pairs} <= {f["sample_id"] for f in self.pop["pools"]["train_live"]})

    def test_test_rows_never_admitted(self):
        lookup = firewall.load_split_lookup(ROOT)
        self.assertEqual(set(lookup.values()), {"TRAIN", "VAL"})
        with self.assertRaises(firewall.TestFirewallViolation):
            firewall.assert_allowed(["not-a-known-sample"], lookup, ["TRAIN"])
        with self.assertRaises(firewall.TestFirewallViolation):
            firewall.assert_allowed([], lookup, ["TEST"])
        with self.assertRaises(firewall.TestFirewallViolation):
            firewall.assert_allowed([self.pop["pools"]["val"][0]["sample_id"]], lookup, ["TRAIN"])


class Banks(unittest.TestCase):
    def test_all_21_bindings(self):
        seen = set()
        for m, s in itertools.product(SB.METHODS, SB.SEEDS):
            b = SB.bank_binding(ROOT, m, s)
            self.assertEqual((b["output_count"], b["failure_count"], b["seed"]), (8838, 0, s))
            self.assertEqual(b["family"], "bank_writer" if m in ("E01", "E02") else "learned_bank")
            if m in SB.LEARNED_METHODS:
                self.assertEqual(b["path"], "/media/ext_data/student20261_gpat_m8_banks/%s/seed_%d" % (m, s))
            seen.add(b["manifest_sha256"])
        self.assertEqual(len(seen), 21)

    def test_refusals(self):
        for m, s in (("E06a", 42), ("E00", 42), ("E04", 7), ("E04", "42"), ("E07a", 1337)):
            with self.assertRaises(SB.ScienceBankError):
                SB.bank_binding(ROOT, m, s)

    def test_tampered_or_failed_records_refused(self):
        with tempfile.TemporaryDirectory() as t:
            for rel in (SB.RECORD_LEARNED.format(M="E05", S=1337), SB.CLOSURE_LEARNED.format(M="E05"),
                        SB.CLOSURE_E0102.format(M="E02")):
                (Path(t) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, Path(t) / rel)
            SB.bank_binding(t, "E05", 1337)
            p = Path(t) / SB.RECORD_LEARNED.format(M="E05", S=1337)
            d = json.loads(p.read_text())
            d["production_audit_passed"], d["failed_gates"] = False, ["S.population"]
            p.write_text(json.dumps(d))
            with self.assertRaises(SB.ScienceBankError):
                SB.bank_binding(t, "E05", 1337)
            q = Path(t) / SB.CLOSURE_E0102.format(M="E02")
            d = json.loads(q.read_text())
            d["status"] = "E02_PRODUCTION_FAMILY_FAILED"
            q.write_text(json.dumps(d))
            with self.assertRaises(SB.ScienceBankError):
                SB.bank_binding(t, "E02", 42)
            with self.assertRaises(SB.ScienceBankError):
                SB.bank_binding(t, "E03", 42)                       # missing record

    def test_disclosures_preserved(self):
        e05 = json.loads((ROOT / "configs/amendments/e05_near_collapse_reporting_decision.yaml").read_text())
        self.assertEqual(e05["status"], "OWNER_DECIDED")
        e04 = json.loads((ROOT / SB.CLOSURE_LEARNED.format(M="E04")).read_text())
        self.assertEqual(e04["fidelity"], "CONTROLLED_ADAPTATION")
        self.assertIn("not an official native Physics-STD reproduction", " ".join(e04["disclosures"]))


# ----------------------------------------------------------------------------------------- manifest / precision
def valid_manifest(method="E03", seed=1337):
    bank = None if method == "E00" else {"method_id": method, "seed": seed, "manifest_sha256": "a" * 64}
    return {"schema": SM.SCHEMA, "label": "SCIENTIFIC", "method_id": method, "detector_seed": seed,
            "bank_seed": None if method == "E00" else seed, "source_commit": "1" * 40, "execution_commit": "2" * 40,
            "evaluator_contract_sha256": "b" * 64, "frozen_config_sha256": "c" * 64, "split_manifest_sha256": "d" * 64,
            "m2_accounting_sha256": "e" * 64, "pool_sha256": {}, "synthetic_bank": bank,
            "environment": {k: "x" for k in SM.ENV_KEYS}, "gpu": {k: "x" for k in SM.GPU_KEYS},
            "precision_policy": {}, "rng": {}, "command": "x", "working_directory": "/x", "start_utc": "t",
            "end_utc": "t", "epochs": 40, "warmup_epochs": 3, "steps_per_epoch": 175, "total_steps": 7000,
            "checkpoint": {"path": "/x/c.pt", "file_sha256": "f" * 64},
            "val_predictions": [{"epoch": e, "sha256": "9" * 64} for e in range(40)],
            "selection": {"selected_epoch": 12, "checkpoint_state_sha256": "8" * 64, "checkpoint_file_sha256": "f" * 64,
                          "val_acer": "1/10", "val_auc": "9/10", "frozen_val_threshold_hex": "0x1p-1",
                          "selection_rule": "D5", "threshold_rule": "s15", "selection_source": "VAL only"},
            "logs": {}, "audit": {}, "reproduction": [], "test_accessed": False,
            "backup": "no independent binary backup"}


class Manifest(unittest.TestCase):
    def test_valid(self):
        for m in ("E00", "E04"):
            self.assertTrue(SM.validate(valid_manifest(m))["ok"])

    def test_refusals(self):
        bad = {"E00 with bank": lambda d: d.update(bank_seed=42, synthetic_bank={"method_id": "E00", "seed": 42,
                                                                                  "manifest_sha256": "a" * 64}),
               "seed mismatch": lambda d: d.update(bank_seed=42), "test": lambda d: d.update(test_accessed=True),
               "schedule": lambda d: d.update(epochs=2, total_steps=350, val_predictions=d["val_predictions"][:2]),
               "missing": lambda d: d.pop("evaluator_contract_sha256"),
               "wrong method bank": lambda d: d["synthetic_bank"].update(method_id="E05"),
               "short commit": lambda d: d.update(source_commit="35964cf"),
               "selection": lambda d: d["selection"].pop("frozen_val_threshold_hex"),
               "unknown group": lambda d: d.update(method_id="E06a")}
        for name, fn in bad.items():
            d = valid_manifest()
            fn(d)
            with self.assertRaises(SM.ManifestError, msg=name):
                SM.validate(d)


class PrecisionPin(unittest.TestCase):
    def test_refused_without_owner_pin(self):
        if (ROOT / SR.PRECISION_PIN_PATH).exists():
            self.skipTest("owner pin activated")
        with self.assertRaises(SR.ScienceRunError):
            SR.load_science_precision(ROOT)

    def test_pin_gate(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / SR.PRECISION_PIN_PATH
            p.parent.mkdir(parents=True)
            for doc in ({"status": "DRAFT", "approved": False, "policy": "CUDA_AMP_BF16_DETERMINISTIC"},
                        {"status": "OWNER_APPROVED", "approved": True, "policy": "CUDA_FP32_DETERMINISTIC"},
                        {"status": "OWNER_APPROVED", "approved": True, "policy": "CPU_FP32_DETERMINISTIC"}):
                p.write_text(json.dumps(doc))
                with self.assertRaises(SR.ScienceRunError):
                    SR.load_science_precision(t)
            p.write_text(json.dumps({"status": "OWNER_APPROVED", "approved": True,
                                     "policy": "CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC"}))
            self.assertEqual(SR.load_science_precision(t).autocast_dtype, "float16")

    def test_tracks_follow_method_status(self):
        import csv
        st = {r["experiment_id"]: r for r in csv.DictReader(open(ROOT / "outputs/audit/method_status.csv"))}
        for m, tr in SR.TRACKS.items():
            if m != "E00":
                self.assertEqual(st[m]["track"], tr, m)
                self.assertEqual(st[m]["scope_status"], "ACTIVE_M6_BASELINE", m)
        self.assertEqual(st["E06a"]["scope_status"], "SUPERSEDED_BY_ADOPTED_AMENDMENT")
        self.assertEqual(st["E07c"]["scope_status"], "OWNER_EXCLUDED")

    def test_protocol_audit_rebuilds(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("pa", str(ROOT / "tools/m10_downstream_protocol_audit.py"))
        pa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pa)
        p = ROOT / (pa.OUT + ".json")
        if not p.exists():
            self.skipTest("protocol audit not built")
        d = pa.build()
        self.assertEqual(json.loads(json.dumps(d)), json.loads(p.read_text()))
        self.assertEqual(pa.report(d), (ROOT / (pa.OUT + ".md")).read_text())
        self.assertEqual(d["matrix"]["runs"], 24)

    def test_policies(self):
        self.assertEqual(SR.POLICIES["CPU_FP32_DETERMINISTIC"].record(),
                         {"name": "CPU_FP32_DETERMINISTIC", "device": "cpu", "autocast_dtype": None, "grad_scaler": False,
                          "deterministic": True})
        self.assertTrue(all(SR.POLICIES[p].device == "cuda" and SR.POLICIES[p].autocast_dtype for p in SR.SCIENCE_ELIGIBLE))


if __name__ == "__main__":
    unittest.main()
