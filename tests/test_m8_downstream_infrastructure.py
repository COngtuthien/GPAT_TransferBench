"""Non-executing downstream infrastructure (gpatbench/downstream, tools/m8_smoke.py) on synthetic test vectors.

Every expected value is derived by hand in the comments. No detector, no benchmark image, no TEST data: the only real
input is split_v1 read as (sample_id, split) metadata by the firewall test, and E01/E02 banks come from the synthetic
qualification fixture.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import metrics as M  # noqa: E402
from gpatbench.downstream import schemas as S  # noqa: E402
from gpatbench.downstream import firewall as F  # noqa: E402
from gpatbench.downstream import plots as P  # noqa: E402
from gpatbench.downstream import smoke_evidence as E  # noqa: E402

GE, GT = M.DECISION_RULES


class Metrics(unittest.TestCase):
    def test_01_sigmoid(self):
        self.assertEqual(M.sigmoid(0.0), 0.5)
        self.assertAlmostEqual(M.sigmoid(-800.0), 0.0)
        self.assertAlmostEqual(M.sigmoid(800.0), 1.0)
        with self.assertRaises(M.MetricError):
            M.sigmoid(float("nan"))

    def test_02_rates_from_confusion_counts(self):
        # 6 spoof, 3 predicted live -> APCER 1/2; 4 live, 1 predicted spoof -> BPCER 1/4; ACER 3/8;
        # accuracy (3 + 3)/10 = 3/5; balanced accuracy 1 - 3/8 = 5/8
        r = M.rates_from_counts({"n_live": 4, "n_spoof": 6, "live_pred_spoof": 1, "spoof_pred_live": 3})
        self.assertEqual((r["APCER"], r["BPCER"], r["ACER"], r["accuracy"], r["balanced_accuracy"]),
                         (Fraction(1, 2), Fraction(1, 4), Fraction(3, 8), Fraction(3, 5), Fraction(5, 8)))
        # spec 26: ACER from confusion counts equals ACER from the predictions themselves
        y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        pred = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        self.assertEqual(M.rates_from_counts(M.confusion_counts(y, pred))["ACER"], Fraction(3, 8))
        # an absent class is N/A, not 0
        self.assertIsNone(M.rates_from_counts({"n_live": 0, "n_spoof": 2, "live_pred_spoof": 0,
                                                "spoof_pred_live": 1})["ACER"])

    def test_03_decision_rule_required_and_distinct(self):
        with self.assertRaises(TypeError):
            M.predict_spoof([0.5], 0.5)                       # no default rule
        with self.assertRaises(M.MetricError):
            M.predict_spoof([0.5], 0.5, rule="SPOOF_IF_SCORE_ABOVE")
        self.assertEqual(M.predict_spoof([1.0, 0.5], 1.0, rule=GE), [1, 0])
        self.assertEqual(M.predict_spoof([1.0, 0.5], 1.0, rule=GT), [0, 0])

    def test_04_threshold_candidates(self):
        # unique [0.2, 0.6, 0.9] -> midpoints 0.4, 0.75; plus 0 and 1
        self.assertEqual(M.threshold_candidates([0.2, 0.6, 0.6, 0.9]), [0.0, 0.4, 0.75, 1.0])

    def test_05_select_threshold_min_acer_then_nearest_half(self):
        # y [0,0,1,1], s [0.1,0.4,0.35,0.8]; candidates 0, 0.225, 0.375, 0.6, 1 give ACER 1/2, 1/4, 1/2, 1/4, 1/2.
        # 0.225 and 0.6 tie at 1/4; |0.6 - 0.5| < |0.225 - 0.5| -> 0.6 (midpoints compared as computed floats)
        for rule in (GE, GT):
            r = M.select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], rule=rule)
            self.assertEqual((r["threshold"], r["val_acer"]), ((0.4 + 0.8) / 2, Fraction(1, 4)))
            self.assertEqual(r["min_acer_thresholds"], [(0.1 + 0.35) / 2, (0.4 + 0.8) / 2])

    def test_06_unbroken_tie_raises(self):
        # y [0,0,1,1], s [0.25,0.5,0.5,0.75]; candidates 0, 0.375, 0.625, 1; 0.375 and 0.625 both give ACER 1/4
        # and are both 0.125 from 0.5; the spec gives no third tie-break
        with self.assertRaises(M.UnresolvedThresholdTie):
            M.select_threshold([0, 0, 1, 1], [0.25, 0.5, 0.5, 0.75], rule=GE)
        with self.assertRaises(M.MetricError):
            M.select_threshold([1, 1], [0.2, 0.9], rule=GE)   # VAL needs both classes

    def test_07_auc_exact(self):
        # spoof {0.35, 0.8} vs live {0.1, 0.4}: 3 of 4 pairs ordered -> 3/4
        self.assertEqual(M.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), Fraction(3, 4))
        self.assertEqual(M.auc([0, 1], [0.5, 0.5]), Fraction(1, 2))          # a tie counts 1/2
        self.assertEqual(M.auc([0, 1, 1], [0.2, 0.2, 0.9]), Fraction(3, 4))  # (0.5 + 1) / 2

    def test_08_curve_points(self):
        pts = M.curve_points([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], rule=GE)
        self.assertEqual([p["threshold"] for p in pts], [0.0, (0.1 + 0.35) / 2, (0.35 + 0.4) / 2, (0.4 + 0.8) / 2, 1.0])
        self.assertEqual([(p["APCER"], p["BPCER"]) for p in pts],
                         [(0, 1), (0, Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 2), 0), (1, 0)])
        self.assertEqual([(p["FPR"], p["TPR"]) for p in pts][0], (1, 1))

    def test_09_video_aggregation(self):
        frames = [{"dataset": "d", "video_id": "v1", "y_true": 1, "p_frame": p, "attack_macro": "print"} for p in (0.2, 0.4, 0.9)]
        frames += [{"dataset": "d", "video_id": "v0", "y_true": 0, "p_frame": p, "attack_macro": "live"} for p in (0.1, 0.3)]
        v = M.aggregate_videos(frames)
        self.assertEqual([(x["video_id"], x["n_frames"], x["y_true"]) for x in v], [("v0", 2, 0), ("v1", 3, 1)])
        self.assertAlmostEqual(v[1]["video_score"], 0.5)
        self.assertEqual(M.aggregate_videos(list(reversed(frames))), v)       # frame order does not matter
        with self.assertRaises(M.MetricError):
            M.aggregate_videos(frames + [{"dataset": "d", "video_id": "v0", "y_true": 1, "p_frame": 0.5}])

    def test_10_frozen_threshold_subgroups(self):
        videos = [{"dataset": "a", "y_true": 0, "video_score": 0.2, "attack_macro": "live"},
                  {"dataset": "a", "y_true": 1, "video_score": 0.7, "attack_macro": "print"},
                  {"dataset": "b", "y_true": 0, "video_score": 0.6, "attack_macro": "live"},
                  {"dataset": "b", "y_true": 1, "video_score": 0.4, "attack_macro": "replay"},
                  {"dataset": "b", "y_true": 1, "video_score": 0.9, "attack_macro": "replay"}]
        r = M.apply_frozen_threshold(videos, 0.5, rule=GE)
        # overall: spoof 0.7, 0.4, 0.9 -> one live -> APCER 1/3; live 0.2, 0.6 -> one spoof -> BPCER 1/2
        self.assertEqual((r["overall"]["APCER"], r["overall"]["BPCER"]), (Fraction(1, 3), Fraction(1, 2)))
        self.assertEqual(r["by_dataset"]["a"]["ACER"], 0)
        self.assertEqual(r["by_dataset"]["b"]["AUC"], Fraction(1, 2))         # 0.4 < 0.6 < 0.9
        self.assertEqual(r["apcer_by_attack_macro"], {"print": 0, "replay": Fraction(1, 2)})

    def test_11_seed_mean_sample_std(self):
        self.assertEqual(M.seed_mean_std([1.0, 2.0, 3.0]), {"n": 3, "mean": 2.0, "sample_std": 1.0})
        self.assertIsNone(M.seed_mean_std([0.3])["sample_std"])


def prediction_rows():
    rows = []
    for vid, y, ps in (("v0", 0, (0.1, 0.3)), ("v1", 1, (0.6, 0.8))):
        for i, p in enumerate(ps):
            rows.append({"sample_id": f"{vid}_{i}", "dataset": "d", "subject_id_global": "d::1", "video_id": vid,
                         "frame_index": i, "y_true": y, "attack_raw": None, "attack_macro": "live" if y == 0 else "print",
                         "p_frame": p, "threshold": 0.5, "pred_frame": int(p >= 0.5),
                         "video_score": sum(ps) / len(ps), "pred_video": int(sum(ps) / len(ps) >= 0.5),
                         "method_id": "E01", "track_id": "binary", "evaluator_id": "resnet18", "seed": 42,
                         "checkpoint_sha256": "a" * 64, "split_manifest_sha256": F.SPLIT_MANIFEST_SHA256})
    return rows


class Schemas(unittest.TestCase):
    def test_12_prediction_schema(self):
        self.assertEqual(S.validate_prediction_rows(prediction_rows()), {"rows": 4, "videos": 2})
        bad = prediction_rows()
        bad[0] = dict(reversed(list(bad[0].items())))
        with self.assertRaises(S.SchemaError):
            S.validate_prediction_rows(bad)                                    # column order
        bad = prediction_rows()
        bad[0]["video_score"] = 0.25
        with self.assertRaises(S.SchemaError):
            S.validate_prediction_rows(bad)                                    # not the frame mean
        bad = prediction_rows()
        bad[1]["threshold"] = 0.4
        with self.assertRaises(S.SchemaError):
            S.validate_prediction_rows(bad)                                    # one frozen threshold per file
        bad = prediction_rows()
        bad[1]["seed"] = 7
        with self.assertRaises(S.SchemaError):
            S.validate_prediction_rows(bad)

    def test_13_run_metadata(self):
        meta = {"run_id": "r", "method": "E01", "track": "binary", "seed": 42, "git_commit": "c" * 40,
                "config_sha256": "1" * 64, "split_manifest_sha256": "2" * 64, "pair_manifest_sha256": "3" * 64,
                "environment_lock_sha256": "4" * 64, "gpu": None, "start_time": "t0", "end_time": "t1",
                "status": "COMPLETED", "deviations": []}
        S.validate_run_metadata(meta)
        with self.assertRaises(S.SchemaError):
            S.validate_run_metadata({**meta, "status": "PASSED"})
        with self.assertRaises(S.SchemaError):
            S.validate_run_metadata({k: v for k, v in meta.items() if k != "gpu"})

    def test_14_canonical_hash(self):
        rows = prediction_rows()
        h = S.canonical_rows_sha256(rows, ["sample_id"])
        self.assertEqual(h, S.canonical_rows_sha256(list(reversed(rows)), ["sample_id"]))
        rows[0]["p_frame"] = 0.1000000000000001
        self.assertNotEqual(h, S.canonical_rows_sha256(rows, ["sample_id"]))
        with self.assertRaises(S.SchemaError):
            S.canonical_rows_sha256(rows + [rows[0]], ["sample_id"])
        self.assertEqual(len(S.SYNTHETIC_MANIFEST_COLUMNS), 25)
        self.assertEqual(len(S.PREDICTION_COLUMNS), 19)


class Firewall(unittest.TestCase):
    def test_15_synthetic_lookup(self):
        lk = {"a": "TRAIN", "b": "VAL", "c": "TEST"}
        self.assertEqual(F.assert_allowed(["a", "b"], lk, {"TRAIN", "VAL"}), {"TRAIN": 1, "VAL": 1})
        for ids, allowed in ((["c"], {"TRAIN", "VAL"}), (["zz"], {"TRAIN"}), (["a"], {"TEST"}), (["b"], {"TRAIN"}),
                             (["a"], set())):
            with self.assertRaises(F.TestFirewallViolation):
                F.assert_allowed(ids, lk, allowed)

    def test_16_frozen_manifest_lookup(self):
        import pyarrow.parquet as pq
        lk = F.load_split_lookup(ROOT)
        p = pq.read_table(ROOT / "manifests/pairs_train_v1.parquet", columns=["target_live_id", "source_spoof_id"]).to_pydict()
        ids = p["target_live_id"][:50] + p["source_spoof_id"][:50]
        self.assertEqual(F.assert_allowed(ids, lk, {"TRAIN"}), {"TRAIN": 100})


class BankReader(unittest.TestCase):
    def test_17_reads_and_verifies_fixture_bank(self):
        from methods.common import bank_writer as bw
        from methods.common.bank_fixture import build_fixture
        from gpatbench.downstream.bank_reader import BankReadError, read_verified_bank
        with tempfile.TemporaryDirectory() as td:
            inputs = build_fixture(Path(td) / "faces", 8)
            bw.write_bank("E02", 42, inputs, bw.qualification_authorization("m8unittest"), Path(td) / "rt")
            rd = Path(td) / "rt" / "runs" / "m6" / "E02" / "seed_42"
            b = read_verified_bank(rd, "E02", 42)
            self.assertEqual((len(b["samples"]), b["failures"], b["rows"]), (8, [], 8))
            self.assertEqual({s.y_true for s in b["samples"]}, {1})
            with self.assertRaises(BankReadError):
                read_verified_bank(rd, "E02", 1337)
            victim = b["samples"][3].path
            os.chmod(victim, 0o644)
            victim.write_bytes(victim.read_bytes()[:-1] + b"\x00")
            with self.assertRaises(BankReadError):
                read_verified_bank(rd, "E02", 42)
            for p in Path(td).rglob("*"):
                if p.is_file():
                    os.chmod(p, 0o644)


class Plots(unittest.TestCase):
    def test_18_placeholder_rejection(self):
        rows = [{"method": "FAS-Aug", "ACER": 0.1, "AUC": "N/A"}, {"method": "FreqSub", "ACER": P.PLACEHOLDER, "AUC": 0.9}]
        with self.assertRaises(P.PlaceholderInPlotInput):
            P.validate_plot_table(rows, ["ACER", "AUC"], template_mode=False)
        self.assertEqual(P.validate_plot_table(rows, ["ACER", "AUC"], template_mode=True)["placeholder_cells"], 1)
        self.assertEqual(P.validate_plot_table(rows[:1], ["ACER", "AUC"], template_mode=False)["placeholder_cells"], 0)
        with self.assertRaises(P.PlaceholderInPlotInput):
            P.validate_plot_table([{"ACER": ""}], ["ACER"], template_mode=False)
        with self.assertRaises(TypeError):
            P.validate_plot_table(rows, ["ACER"], template_mode=None)


def passing_record(contract_sha):
    stage = {"executed": True, "simulated": False, "status": "PASS", "gates": {"g": True},
             "commands": [{"argv": ["x"], "exit_code": 0}]}
    return {"schema_version": E.SCHEMA, "contract_sha256": contract_sha,
            "stages": {s: json.loads(json.dumps(stage)) for s in E.REQUIRED_STAGES},
            "independent_replay": {"identical": True}, "test_firewall": {"violations": 0, "test_rows_used": 0},
            "source": {"worktree_dirty": False}, "scientific_results_produced": False, "smoke_test_passed": True}


class SmokeEvidence(unittest.TestCase):
    def test_19_validator_rules(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            c = Path(td) / E.CONTRACT_PATH
            c.parent.mkdir(parents=True)
            c.write_text(json.dumps({"status": "OWNER_APPROVED", "approved": True}))
            sha = hashlib.sha256(c.read_bytes()).hexdigest()
            self.assertTrue(E.validate_smoke_record(passing_record(sha), td)["passed"])
            for mutate in (lambda r: r["stages"].pop("plots"),
                           lambda r: r["stages"]["evaluator"].update(simulated=True),
                           lambda r: r["stages"]["metrics"]["commands"][0].update(exit_code=1),
                           lambda r: r["stages"]["bank"]["gates"].update(g=False),
                           lambda r: r.update(contract_sha256="0" * 64),
                           lambda r: r["test_firewall"].update(test_rows_used=3),
                           lambda r: r.update(scientific_results_produced=True)):
                rec = passing_record(sha)
                mutate(rec)
                with self.assertRaises(E.SmokeRecordInvalid):
                    E.validate_smoke_record(rec, td)
                rec["smoke_test_passed"] = False
                self.assertEqual(E.validate_smoke_record(rec, td)["admissible"], True)
        # the repository has no approved contract, so no record can pass here
        with self.assertRaises(E.SmokeRecordInvalid):
            E.validate_smoke_record(passing_record("0" * 64), ROOT)


class Cli(unittest.TestCase):
    def test_20_m8_smoke_check_is_blocked(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import m8_smoke
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = m8_smoke.main(["--check"])
        doc = json.loads(out.getvalue())
        self.assertEqual(rc, m8_smoke.EXIT_BLOCKED)
        self.assertTrue(doc["gates"]["S1_global_end_to_end_recorded"])
        if not (ROOT / m8_smoke.CONTRACT).exists():
            self.assertEqual(doc["status"], "M8_SMOKE_BLOCKED")
            self.assertIn("smoke_contract_S2_S10_approved", doc["missing"])


if __name__ == "__main__":
    unittest.main()
