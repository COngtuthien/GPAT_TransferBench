"""M5 pre-flight tests for ArtifactProbeNet (spec §12.1).

M5 has not started. These assert the analysis, the firewall and the guards that keep it that way;
nothing here trains, and the transform tests only check properties that hold for EVERY unresolved
variant, because the contract is not frozen.
"""
import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import execute as E        # noqa: E402
from gpatbench.probe import preprocess as PP     # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
import m5_test_env as H                            # noqa: E402

AUDIT = ROOT / "outputs/audit"
PROPOSED = ROOT / "configs/proposed/artifact_probe.proposed.yaml"
FROZEN = ROOT / "configs/frozen/artifact_probe.yaml"

FROZEN_M4 = {
    "manifests/pair_train_stats_v1.json":
        "a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50",
    "manifests/pairs_train_v1.parquet":
        "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243",
    "manifests/val_pairs_v1.parquet":
        "84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872",
    "manifests/split_v1.parquet":
        "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d",
}

_C = {}


def cfg():
    return _C.setdefault("cfg", yaml.safe_load(PROPOSED.read_text()))


def pre():
    return _C.setdefault("pre", json.loads((AUDIT / "M5_ARTIFACT_PROBE_PREFLIGHT.json").read_text()))


def stage():
    return _C.setdefault("stage",
                         json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"])


# ------------------------------------------------------------------ milestone state
class TestMilestoneState(unittest.TestCase):
    def test_m4_remains_finalized_under_amendment_a1(self):
        m4 = stage()["M4"]
        self.assertEqual(m4["status"], "COMPLETE")
        self.assertEqual(m4["phase"], "FINALIZED_UNDER_AMENDMENT_A1")
        a = m4["amendment_a1"]
        self.assertEqual(a["track_a"], "COMPLETE")
        self.assertEqual(a["track_b_native_pairing"], "DEFERRED_TO_M6_SECONDARY_TRACK")
        self.assertIs(a["original_native_manifests_completed"], False)

    def test_common_m4_hashes_are_unchanged(self):
        for rel, sha in FROZEN_M4.items():
            self.assertEqual(E.sha256_file(ROOT / rel), sha, rel)

    def test_m5_remains_not_started(self):
        self.assertEqual(stage()["M5"]["status"], "NOT_STARTED")
        self.assertEqual(stage()["M6"]["status"], "NOT_STARTED")
        self.assertEqual(cfg()["milestone_status"], "NOT_STARTED")

    def test_no_trained_probe_checkpoint_exists(self):
        self.assertFalse((ROOT / "models/artifact_probe").exists())
        for pat in ("*.pt", "*.pth", "*.ckpt", "artifact_probe*"):
            found = [p for p in ROOT.rglob(pat)
                     if p.is_file()
                     and not ({".git", ".venv", ".venv-audit"} & set(p.parts))
                     and "third_party/source_cache" not in p.as_posix()
                     and p.suffix in (".pt", ".pth", ".ckpt")]
            self.assertEqual(found, [], pat)

    def test_proposed_config_is_preserved_as_pre_decision_history(self):
        """The proposal keeps its open decisions on record even after the owner resolved them."""
        self.assertTrue(cfg()["blocking_decisions"])
        self.assertEqual(cfg()["status"], "PROPOSED_BLOCKED")
        self.assertTrue(PROPOSED.is_file())

    def test_a_frozen_config_may_only_exist_with_no_open_decisions(self):
        import yaml as _y
        if not FROZEN.exists():
            self.assertTrue(cfg()["blocking_decisions"])
            return
        frozen = _y.safe_load(FROZEN.read_text())
        self.assertEqual(frozen["status"], "FROZEN")
        self.assertEqual(frozen["blocking_decisions"], [])
        self.assertEqual(frozen["supersedes"], PROPOSED.relative_to(ROOT).as_posix())


# ------------------------------------------------------------------ scientific role / firewall
class TestProbeRoleAndFirewall(unittest.TestCase):
    def test_probe_is_not_gpat_e_art(self):
        s = cfg()["settled"]
        self.assertIs(s["is_gpat_e_art"], False)
        text = (AUDIT / "M5_ARTIFACT_PROBE_CONTRACT_ANALYSIS.md").read_text()
        self.assertIn("never", text)
        self.assertIn("E_art", text)

    def test_attack_macro_use_does_not_violate_amendment_a1(self):
        fw = cfg()["amendment_a1_firewall"]
        self.assertEqual(fw["status"], "COMPLIANT")
        for k in ("probe_labels_enter_generator_optimization",
                  "probe_predictions_select_or_modify_track_a_training_samples",
                  "probe_predictions_decide_common_pair_membership",
                  "probe_predictions_decide_generator_checkpoint_selection",
                  "probe_gradients_touch_generator_parameters",
                  "attack_macro_exported_as_generator_input"):
            self.assertIs(fw["conditions_that_must_hold"][k], False, k)
        self.assertIs(fw["conditions_that_must_hold"]["probe_frozen_before_synthetic_evaluation"],
                      True)

    def test_track_a_generator_ban_on_attack_type_is_untouched(self):
        fair = yaml.safe_load((ROOT / "configs/frozen/fair_track_v1.yaml").read_text())
        self.assertEqual(fair["track_a"]["attack_type_supervision"], "FORBIDDEN")
        self.assertEqual(fair["track_a"]["identity_supervision"], "FORBIDDEN")
        for f in ("attack_macro", "attack_raw"):
            self.assertIn(f, fair["track_a"]["forbidden_class_targets"])

    def test_test_split_is_forbidden_for_the_probe(self):
        s = cfg()["settled"]
        self.assertEqual(s["test_split_usage"], "NEVER")
        self.assertEqual(s["train_split"], "TRAIN")
        self.assertEqual(s["val_split"], "VAL")
        self.assertIn("never loaded", pre()["test_usage"])

    def test_synthetic_data_is_forbidden_during_probe_training(self):
        self.assertEqual(cfg()["settled"]["synthetic_data_during_probe_training"], "NONE")
        self.assertFalse((ROOT / "banks").exists() and
                         any((ROOT / "banks").rglob("*.png")))


# ------------------------------------------------------------------ population / classes
class TestClassInventory(unittest.TestCase):
    def test_inventory_is_deterministic_and_matches_the_split(self):
        import pyarrow.parquet as pq
        rows = pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()
        for key, pred in (("A_spoof_only", lambda r: int(r["label_binary"]) == 1),
                          ("B_live_plus_spoof", lambda r: True)):
            p = pre()["candidate_populations"][key]
            for split, field in (("TRAIN", "train"), ("VAL", "val")):
                for c, v in p["per_class"].items():
                    n = sum(1 for r in rows if r["split"] == split and pred(r)
                            and r["attack_macro"] == c)
                    self.assertEqual(v[field], n, f"{key}/{c}/{split}")

    def test_class_order_is_lexical_ascending(self):
        for key in ("A_spoof_only", "B_live_plus_spoof"):
            p = pre()["candidate_populations"][key]
            self.assertEqual(p["class_order"], sorted(p["class_order"]))
            self.assertEqual(p["class_order_rule"], "lexical class token ascending")
            self.assertEqual(len(set(p["class_order"])), len(p["class_order"]))

    def test_two_candidate_populations_differ_only_by_the_live_class(self):
        a = pre()["candidate_populations"]["A_spoof_only"]["class_order"]
        b = pre()["candidate_populations"]["B_live_plus_spoof"]["class_order"]
        self.assertEqual(set(b) - set(a), {"live"})
        self.assertEqual(set(a) - set(b), set())
        self.assertEqual(a, ["makeup", "mask_2d", "mask_3d", "partial", "print", "replay"])

    def test_live_token_is_exactly_the_live_rows(self):
        e = pre()["enum_coverage"]
        self.assertIs(e["live_token_present"], True)
        self.assertIs(e["live_token_only_on_live_rows"], True)
        self.assertEqual(e["data_tokens_absent_from_spec_enum"], [])
        self.assertEqual(e["spec_tokens_absent_from_data"], ["other_spoof"])

    def test_population_totals(self):
        p = pre()["population"]
        self.assertEqual((p["TRAIN"]["total"], p["TRAIN"]["live"], p["TRAIN"]["spoof"]),
                         (14467, 5629, 8838))
        self.assertEqual((p["VAL"]["total"], p["VAL"]["live"], p["VAL"]["spoof"]),
                         (3121, 1216, 1905))


# ------------------------------------------------------------------ class weights
class TestWeightAudit(unittest.TestCase):
    def test_raw_inverse_is_degenerate_and_recorded_as_such(self):
        for key in ("A_spoof_only", "B_live_plus_spoof"):
            v = pre()["candidate_populations"][key]["class_weights"]["variants"]["raw_inverse"]
            self.assertIs(v["all_clipped_to_lower_bound"], True, key)
            self.assertIs(v["clipping_destroys_all_information"], True, key)
            self.assertEqual(set(v["clipped"].values()), {0.5}, key)

    def test_candidate_weights_recompute_exactly(self):
        for key in ("A_spoof_only", "B_live_plus_spoof"):
            cw = pre()["candidate_populations"][key]["class_weights"]
            classes = cw["class_order"]
            n = np.array([cw["counts"][c] for c in classes], dtype=np.float64)
            N, K = n.sum(), len(classes)
            expect = {
                "raw_inverse": 1.0 / n,
                "normalized_inverse_N_over_Kn": N / (K * n),
                "mean_one_inverse": (1.0 / n) / (1.0 / n).mean(),
                "max_one_inverse": (1.0 / n) / (1.0 / n).max(),
                "median_frequency": np.median(n) / n,
            }
            for name, w in expect.items():
                got = cw["variants"][name]
                self.assertEqual([got["unclipped"][c] for c in classes], list(w), f"{key}/{name}")
                self.assertEqual([got["clipped"][c] for c in classes],
                                 list(np.clip(w, 0.5, 3.0)), f"{key}/{name}")

    def test_all_clipped_weights_are_inside_the_frozen_range(self):
        for key in ("A_spoof_only", "B_live_plus_spoof"):
            for name, v in pre()["candidate_populations"][key]["class_weights"]["variants"].items():
                for c, w in v["clipped"].items():
                    self.assertTrue(np.isfinite(w) and 0.5 <= w <= 3.0, f"{key}/{name}/{c}")

    def test_weight_normalization_is_not_silently_chosen(self):
        d = cfg()["open_decisions"]["D-M5-02"]
        self.assertEqual(d["status"], "OWNER_DECISION_REQUIRED")
        self.assertEqual(d["proposal"], "NONE_WITHOUT_OWNER_DECISION")
        self.assertIs(d["candidates"]["raw_inverse"]["degenerate"], True)


# ------------------------------------------------------------------ high-pass scaffolding
class TestHighPassInvariants(unittest.TestCase):
    """Only properties that hold for EVERY unresolved variant; the contract is not frozen."""
    VARIANTS = [dict(resize_order=o, interpolation=i, border_mode=b, post_normalization=p)
                for o in PP.RESIZE_ORDERS for i in ("area", "bilinear", "bicubic")
                for b in PP.BORDER_MODES for p in PP.POST_NORMALIZATIONS]

    def test_output_shape_and_dtype(self):
        img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        for v in self.VARIANTS:
            y = PP.probe_input(img, **v)
            self.assertEqual(y.shape, (3, 224, 224), v)
            self.assertEqual(y.dtype, np.float32, v)

    def test_constant_image_gives_zero_high_pass_interior(self):
        for val in (0, 37, 128, 255):
            img = np.full((256, 256, 3), val, np.uint8)
            for v in self.VARIANTS:
                if v["post_normalization"] != "none":
                    continue                     # normalisation adds a constant offset by design
                y = PP.probe_input(img, **v)
                # exactly zero for AREA/BILINEAR; bicubic resampling of a constant leaves ~1e-8
                self.assertLess(float(np.abs(y[:, 8:-8, 8:-8]).max()), 1e-6, (val, v))

    def test_signed_residual_is_preserved(self):
        img = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        for v in self.VARIANTS:
            if v["post_normalization"] != "none":
                continue
            y = PP.probe_input(img, **v)
            self.assertLess(y.min(), 0.0, v)
            self.assertGreater(y.max(), 0.0, v)

    def test_known_impulse_response_matches_an_explicit_gaussian(self):
        img = np.zeros((256, 256, 3), np.uint8)
        img[128, 128] = 255
        x = PP.to_unit_interval(img)
        got = PP.high_pass(x, border_mode="reflect101")
        ax = np.arange(PP.KERNEL, dtype=np.float64) - (PP.KERNEL - 1) / 2.0
        k1 = np.exp(-(ax ** 2) / (2 * PP.SIGMA ** 2))
        k1 /= k1.sum()
        k2 = np.outer(k1, k1)
        blur_centre = k2[PP.KERNEL // 2, PP.KERNEL // 2] * 1.0
        self.assertAlmostEqual(float(got[128, 128, 0]), 1.0 - blur_centre, places=5)
        off = k2[PP.KERNEL // 2, PP.KERNEL // 2 - 1] * 1.0
        self.assertAlmostEqual(float(got[128, 127, 0]), -off, places=5)

    def test_channel_order_is_preserved(self):
        img = np.zeros((256, 256, 3), np.uint8)
        img[100:140, 100:140, 0] = 255                      # red square only
        y = PP.probe_input(img, resize_order="resize_then_highpass", interpolation="area",
                           border_mode="reflect101", post_normalization="none")
        e = [float(np.abs(y[c]).sum()) for c in range(3)]
        self.assertGreater(e[0], 0.0)
        self.assertEqual(e[1], 0.0)
        self.assertEqual(e[2], 0.0)

    def test_transform_is_deterministic(self):
        img = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        for v in self.VARIANTS[:8]:
            a = PP.probe_input(img, **v)
            b = PP.probe_input(img, **v)
            self.assertTrue(np.array_equal(a, b), v)

    def test_unresolved_choices_have_no_defaults(self):
        """The module must not let a caller silently inherit an unfrozen decision."""
        import inspect
        sig = inspect.signature(PP.probe_input)
        for name in ("resize_order", "interpolation", "border_mode", "post_normalization"):
            self.assertIs(sig.parameters[name].default, inspect.Parameter.empty, name)
            self.assertEqual(sig.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY, name)

    def test_unknown_options_are_refused(self):
        img = np.zeros((256, 256, 3), np.uint8)
        base = dict(resize_order="resize_then_highpass", interpolation="area",
                    border_mode="reflect101", post_normalization="none")
        for k in base:
            bad = dict(base, **{k: "not_a_real_option"})
            with self.assertRaises(PP.ProbeContractError, msg=k):
                PP.probe_input(img, **bad)


# ------------------------------------------------------------------ measurements + config + CLI
class TestPreflightEvidence(unittest.TestCase):
    def test_hp_variant_measurements_separate_real_from_negligible(self):
        m = json.loads((AUDIT / "M5_HP_VARIANT_MEASUREMENTS.json").read_text())
        d = m["variant_differences"]
        for k in ("cv2_vs_torchvision", "cv2_vs_conv_reflect", "torchvision_vs_conv_reflect",
                  "domain_uint8_scaled_vs_unit_interval"):
            self.assertLess(d[k]["max_abs_over_faces"], 1e-5, k)
        for k in ("border_reflect101_vs_replicate", "resize_then_hp_vs_hp_then_resize",
                  "interp_area_vs_bicubic_resize_then_hp"):
            self.assertGreater(d[k]["max_abs_over_faces"], 1e-2, k)
        r = m["residual_statistics_unit_interval"]
        self.assertLess(r["hp_min"]["median"], 0.0)
        self.assertGreater(r["hp_max"]["median"], 0.0)

    def test_proposed_config_marks_every_unresolved_field(self):
        c = cfg()
        self.assertEqual(c["status"], "PROPOSED_BLOCKED")
        for d in c["blocking_decisions"]:
            entry = c["open_decisions"][d]
            self.assertIn(entry["status"], ("OWNER_DECISION_REQUIRED",))
            self.assertEqual(entry.get("proposal", "NONE_WITHOUT_OWNER_DECISION"),
                             "NONE_WITHOUT_OWNER_DECISION", d)
        self.assertEqual(c["open_environment"]["E-M5-01"]["status"], "BLOCKED_BY_ENVIRONMENT")

    def test_q07_is_an_existing_record_and_was_not_invented(self):
        self.assertIn("Q-07", cfg()["blocking_decisions"])
        dev = (AUDIT / "deviation_report.md").read_text()
        self.assertIn("| Q-07 | M5 | 12.1 |", dev)
        self.assertIn("Q-07", (ROOT / "configs/CONFIG_STATUS.md").read_text())
        self.assertIn("existing_record", cfg()["open_decisions"]["Q-07"])

    def test_no_augmentation_was_invented(self):
        d = cfg()["open_decisions"]["D-M5-08"]
        self.assertEqual(d["augmentation_finding"], "NO_AUGMENTATION_SPECIFIED")
        self.assertIn("apply_section_14_1_fas_safe_augmentation", d["candidates"])
        for f in d["forbidden"]:
            self.assertTrue(f)
        self.assertIn("WeightedRandomSampler", " ".join(d["forbidden"]))

    def test_backbone_weight_provenance_is_recorded(self):
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())
        r = reg["models"]["resnet18_imagenet1k_v1"]
        self.assertEqual(r["status"], "VERIFIED")
        self.assertEqual(r["weight_sha256"],
                         "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec")
        self.assertTrue(r["weight_sha256"].startswith("f37072fd"),
                        "torchvision filename hash must match the file digest")
        # the download URL lives in the audit evidence, not the registry (M0 no-invented-URL rule)
        self.assertIn("resnet18-f37072fd.pth", r["official_artifact"])
        self.assertNotIn("official_url", r)
        env = (AUDIT / "M5_ARTIFACT_PROBE_ENVIRONMENT_ANALYSIS.md").read_text()
        self.assertIn("https://download.pytorch.org/models/resnet18-f37072fd.pth", env)
        self.assertIs(r["m5_preflight_resolution"]["committed_to_git"], False)
        self.assertEqual(r["m5_preflight_resolution"]["penultimate_feature_dim"], 512)

    def test_cli_train_probe_refuses_to_train(self):
        for c in ("configs/frozen/artifact_probe.yaml",
                  "configs/proposed/artifact_probe.proposed.yaml"):
            with H.hermetic_exec_config() as cfg:
                r = subprocess.run([sys.executable, "-m", "gpatbench.cli", "train-probe",
                                    "--config", c], cwd=ROOT, capture_output=True, text=True,
                                   env=H.subprocess_env(cfg))
            self.assertNotEqual(r.returncode, 0, c)
            self.assertIn("STOP", r.stdout + r.stderr, c)

    def test_training_primitives_live_only_in_the_single_trainer_module(self):
        """The authoritative trainer now exists; it must still be the ONLY one.

        Training primitives (backward, optimizer.step, GradScaler, autocast) may appear in
        `train.py` and nowhere else in the probe package, so no second trainer can hide beside it.
        """
        import ast
        banned = ("backward", "step", "train_probe", "fit", "GradScaler", "autocast")
        for f in sorted((ROOT / "gpatbench/probe").rglob("*.py")):
            if f.name == "train.py":
                continue
            tree = ast.parse(f.read_text())
            names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            self.assertEqual(names & set(banned), set(), f"{f.name} looks like a trainer")
        self.assertTrue((ROOT / "gpatbench/probe/train.py").is_file())


if __name__ == "__main__":
    unittest.main()
