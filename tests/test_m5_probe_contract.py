"""Frozen ArtifactProbeNet contract (spec §12.1 + the 2026-09-20 owner resolutions).

M5 is still NOT_STARTED: these tests assert the frozen contract and the refusals that keep it that
way. Nothing here trains, and the one forward pass exists only to check shapes.
"""
import csv
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.probe import contract as C        # noqa: E402
from gpatbench.probe import data as D            # noqa: E402
from gpatbench.probe import metrics as MET       # noqa: E402
from gpatbench.probe import model as MODEL       # noqa: E402
from gpatbench.probe import preprocess as PP     # noqa: E402
from gpatbench.probe import train as T           # noqa: E402

AUDIT = ROOT / "outputs/audit"
CONFIG_SHA = "3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe"
EXPECT_TRAIN = {"live": 5629, "makeup": 759, "mask_2d": 96, "mask_3d": 1056,
                "partial": 1911, "print": 2838, "replay": 2178}
EXPECT_VAL = {"live": 1216, "makeup": 159, "mask_2d": 24, "mask_3d": 224,
              "partial": 406, "print": 623, "replay": 469}
EXPECT_W = {"live": 0.5, "makeup": 2.722943722943723, "mask_2d": 3.0,
            "mask_3d": 1.957115800865801, "partial": 1.0814831427076326,
            "print": 0.7282291352058794, "replay": 0.9489046307228125}

_C = {}


def cfg():
    return _C.setdefault("cfg", C.load_config())


def smoke():
    return _C.setdefault("smoke",
                         json.loads((AUDIT / "M5_ARTIFACT_PROBE_SMOKE.json").read_text()))


# ------------------------------------------------------------------ frozen config
class TestFrozenConfig(unittest.TestCase):
    def test_config_is_frozen_hashed_and_snapshotted(self):
        self.assertEqual(C.sha256_file(C.FROZEN_CONFIG), CONFIG_SHA)
        self.assertEqual(C.FROZEN_SNAPSHOT.read_bytes(), C.FROZEN_CONFIG.read_bytes())
        self.assertEqual(cfg()["status"], "FROZEN")
        self.assertEqual(cfg()["blocking_decisions"], [])
        rec = json.loads((AUDIT / "artifact_probe_v1_config.sha256").read_text())
        self.assertEqual(rec["sha256"], CONFIG_SHA)
        self.assertIs(rec["snapshot_byte_identical"], True)

    def test_no_unresolved_execution_field(self):
        bad = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif node is None or (isinstance(node, str) and node.strip().upper() in
                                  {"TODO", "TBD", "OWNER_DECISION_REQUIRED", "NULL"}):
                bad.append(path)

        walk(cfg(), "")
        self.assertEqual(bad, [], f"unresolved: {bad}")

    def test_decisions_are_not_claimed_as_spec_dictated(self):
        self.assertIn("did not uniquely determine", cfg()["decision_provenance"])
        doc = (AUDIT / "M5_ARTIFACT_PROBE_OWNER_DECISIONS.md").read_text()
        self.assertIn("OWNER RESOLUTIONS", doc)
        self.assertIn("did **not** uniquely dictate", doc)

    def test_proposed_config_is_preserved(self):
        p = ROOT / "configs/proposed/artifact_probe.proposed.yaml"
        self.assertTrue(p.is_file())
        self.assertEqual(cfg()["supersedes"], p.relative_to(ROOT).as_posix())


# ------------------------------------------------------------------ population and classes
class TestPopulationAndClasses(unittest.TestCase):
    def test_k_is_seven_and_order_is_lexical(self):
        self.assertEqual(C.K, 7)
        self.assertEqual(list(C.CLASSES),
                         ["live", "makeup", "mask_2d", "mask_3d", "partial", "print", "replay"])
        self.assertEqual(list(C.CLASSES), sorted(C.CLASSES))
        self.assertEqual(cfg()["classes"]["order"], list(C.CLASSES))
        self.assertEqual(cfg()["classes"]["index_map"],
                         {c: i for i, c in enumerate(C.CLASSES)})

    def test_other_spoof_is_excluded_and_no_unused_logit(self):
        self.assertEqual(cfg()["classes"]["excluded"], ["other_spoof"])
        m = MODEL.build(C.SEED, pretrained=False)
        self.assertEqual(m.fc.out_features, 7)

    def test_exact_counts_recomputed_from_the_manifest(self):
        self.assertEqual(C.class_counts("TRAIN"), EXPECT_TRAIN)
        self.assertEqual(C.class_counts("VAL"), EXPECT_VAL)
        self.assertEqual(sum(EXPECT_TRAIN.values()), 14467)
        self.assertEqual(sum(EXPECT_VAL.values()), 3121)
        C.verify_population(cfg())

    def test_population_excludes_test_and_synthetic(self):
        p = cfg()["population"]
        self.assertEqual(p["test_split_usage"], "NEVER")
        self.assertEqual(p["synthetic_samples"], "NONE")
        self.assertEqual(p["usable_rows_only"], "M2_COMPLETE")


# ------------------------------------------------------------------ class weights
class TestClassWeights(unittest.TestCase):
    def test_balanced_inverse_frequency_formula(self):
        w = C.compute_class_weights(EXPECT_TRAIN)
        n = np.array([EXPECT_TRAIN[c] for c in C.CLASSES], dtype=np.float64)
        expect = np.clip(np.float64(n.sum()) / (np.float64(7) * n), 0.5, 3.0)
        self.assertTrue(np.array_equal(w, expect))
        self.assertEqual(w.dtype, np.float64)

    def test_frozen_values_match_and_are_inside_the_clip(self):
        w = C.verify_class_weights(cfg(), EXPECT_TRAIN)
        for c, v in zip(C.CLASSES, w):
            self.assertEqual(float(v), EXPECT_W[c], c)
            self.assertTrue(0.5 <= float(v) <= 3.0, c)
            self.assertTrue(np.isfinite(v) and v > 0, c)

    def test_exactly_two_classes_are_pinned_by_the_clip(self):
        self.assertEqual(cfg()["class_weights"]["clipped_low"], ["live"])
        self.assertEqual(cfg()["class_weights"]["clipped_high"], ["mask_2d"])
        self.assertIs(cfg()["class_weights"]["renormalize_after_clip"], False)
        for f in ("raw_inverse_1_over_n", "renormalize_after_clip", "per_dataset_weights",
                  "weighted_random_sampler", "weights_from_val", "weights_from_test"):
            self.assertIn(f, cfg()["class_weights"]["forbidden"])

    def test_weight_csv_matches_the_config(self):
        rows = {r["class"]: r for r in
                csv.DictReader((AUDIT / "M5_ARTIFACT_PROBE_CLASS_WEIGHTS.csv").open())}
        self.assertEqual([r["class"] for r in
                          csv.DictReader((AUDIT / "M5_ARTIFACT_PROBE_CLASS_WEIGHTS.csv").open())],
                         list(C.CLASSES))
        for c in C.CLASSES:
            self.assertEqual(float(rows[c]["w_clipped"]), EXPECT_W[c], c)
            self.assertEqual(int(rows[c]["train_count"]), EXPECT_TRAIN[c], c)

    def test_loss_tensor_is_built_in_frozen_class_order(self):
        w = C.compute_class_weights(EXPECT_TRAIN)
        crit = T.build_loss(w)
        self.assertEqual(crit.reduction, "mean")
        got = crit.weight.detach().cpu().numpy()
        self.assertTrue(np.array_equal(got, np.asarray(w).astype(np.float32)))

    def test_count_mismatch_is_a_hard_error(self):
        bad = dict(EXPECT_TRAIN, mask_2d=95)
        with self.assertRaises(C.ProbeContractViolation):
            C.verify_class_weights(cfg(), bad)


# ------------------------------------------------------------------ input pipeline
class TestFrozenTransform(unittest.TestCase):
    def test_frozen_options(self):
        self.assertEqual(PP.FROZEN, {"resize_order": "resize_then_highpass",
                                     "interpolation": "area",
                                     "border_mode": "reflect101",
                                     "post_normalization": "none"})
        i = cfg()["input"]
        self.assertEqual(i["crop"], "NONE")
        self.assertEqual(i["resize"]["interpolation"], "INTER_AREA")
        self.assertEqual((i["resize"]["from"], i["resize"]["to"]), (256, 224))
        self.assertEqual(i["gaussian"]["kernel"], [9, 9])
        self.assertEqual((i["gaussian"]["sigma_x"], i["gaussian"]["sigma_y"]), (1.5, 1.5))
        self.assertEqual(i["gaussian"]["border"], "BORDER_REFLECT_101")
        self.assertIs(i["residual"]["signed"], True)
        self.assertIs(i["residual"]["clip_to_unit_interval"], False)
        self.assertEqual(i["output_tensor"]["shape"], [3, 224, 224])

    def test_no_post_hp_normalization(self):
        n = cfg()["input"]["post_hp_normalization"]
        for k in ("imagenet_mean_subtraction", "imagenet_std_division", "dataset_mean_std",
                  "min_max"):
            self.assertIs(n[k], False, k)
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        y = PP.frozen_probe_input(img)
        self.assertTrue(np.array_equal(y, PP.probe_input(img, **PP.FROZEN)))
        # ImageNet normalisation would shift the mean far from zero; the frozen path must not.
        self.assertLess(abs(float(y.mean())), 1e-2)

    def test_full_face_resize_then_highpass_and_signed_output(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        y = PP.frozen_probe_input(img)
        self.assertEqual(y.shape, (3, 224, 224))
        self.assertEqual(y.dtype, np.float32)
        self.assertLess(y.min(), 0.0)
        self.assertGreater(y.max(), 0.0)
        other = PP.probe_input(img, **dict(PP.FROZEN, resize_order="highpass_then_resize"))
        self.assertFalse(np.array_equal(y, other), "resize order must be the frozen one")

    def test_interpolation_and_border_are_the_frozen_ones(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        y = PP.frozen_probe_input(img)
        for k, v in (("interpolation", "bicubic"), ("border_mode", "replicate")):
            self.assertFalse(np.array_equal(y, PP.probe_input(img, **dict(PP.FROZEN, **{k: v}))),
                             k)

    def test_transform_is_deterministic_and_augmentation_free(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(PP.frozen_probe_input(img), PP.frozen_probe_input(img)))
        a = cfg()["augmentation"]
        self.assertIs(a["any"], False)
        for f in ("RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter", "RandomErasing",
                  "MixUp", "CutMix"):
            self.assertIn(f, a["forbidden"], f)

    def test_real_face_statistics_from_the_smoke(self):
        for split in ("TRAIN", "VAL"):
            s = smoke()["transform"][split]
            self.assertEqual(s["shape"], [3, 224, 224])
            self.assertEqual(s["dtype"], "float32")
            self.assertIs(s["signed"], True)
            self.assertLess(abs(s["mean"]), 1e-3)


# ------------------------------------------------------------------ model
class TestModel(unittest.TestCase):
    def test_full_fine_tune_seven_logits_and_embedding(self):
        m = MODEL.build(C.SEED, pretrained=False)
        rep = MODEL.trainable_report(m)
        self.assertIs(rep["fully_trainable"], True)
        self.assertEqual(rep["frozen_parameter_names"], [])
        self.assertEqual(rep["out_features"], 7)
        self.assertEqual(rep["in_features"], 512)
        x = torch.zeros(2, 3, 224, 224)
        m.eval()
        with torch.no_grad():
            self.assertEqual(list(m(x).shape), [2, 7])
            e = MODEL.embed(m, x)
        self.assertEqual(list(e.shape), [2, C.EMBED_DIM])
        self.assertEqual(C.EMBED_DIM, 512)
        self.assertEqual(C.L2_EPS, 1e-12)

    def test_embedding_is_unit_norm_and_not_logits(self):
        m = MODEL.build(C.SEED, pretrained=False)
        m.eval()
        x = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            e = MODEL.embed(m, x)
            logits = m(x)
        self.assertTrue(torch.allclose(e.norm(dim=1), torch.ones(3), atol=1e-5))
        self.assertNotEqual(e.shape[1], logits.shape[1])
        self.assertEqual(cfg()["embedding"]["dim"], 512)
        self.assertIs(cfg()["embedding"]["use_classifier_logits"], False)

    def test_backbone_config_forbids_freezing_and_differential_lr(self):
        b = cfg()["backbone"]
        self.assertEqual(b["frozen_parameters"], "NONE")
        self.assertIs(b["differential_lr"], False)
        self.assertIs(b["gradual_unfreeze"], False)
        self.assertIs(b["seed_set_before_classifier_construction"], True)
        self.assertEqual(b["weight_sha256"], C.RESNET18_WEIGHT_SHA256)


# ------------------------------------------------------------------ metric and selection
class TestMetricAndSelection(unittest.TestCase):
    def test_fixed_seven_class_macro_f1(self):
        cm = MET.confusion_matrix(list(range(7)), list(range(7)))
        self.assertEqual(cm.shape, (7, 7))
        self.assertEqual(MET.macro_f1(cm), 1.0)
        self.assertEqual(len(MET.per_class_f1(cm)), 7)

    def test_zero_division_gives_f1_zero_and_no_class_is_dropped(self):
        cm = MET.confusion_matrix([0, 0], [0, 0])          # only class 0 observed
        f1 = MET.per_class_f1(cm)
        self.assertEqual(len(f1), 7)
        self.assertEqual(f1[0], 1.0)
        self.assertEqual(list(f1[1:]), [0.0] * 6)
        self.assertAlmostEqual(MET.macro_f1(cm), 1.0 / 7.0, places=12)

    def test_macro_f1_is_not_weighted_or_micro(self):
        cm = MET.confusion_matrix([0] * 100 + [1], [0] * 100 + [0])
        self.assertLess(MET.macro_f1(cm), 0.5)             # a micro/weighted score would be ~0.99
        v = cfg()["validation"]["metric"]
        self.assertIs(v["drop_unpredicted_classes"], False)
        self.assertIs(v["depends_on_sklearn"], False)
        for f in ("weighted_f1", "micro_f1", "dataset_macro_averaging"):
            self.assertIn(f, v["forbidden"])

    def test_validation_runs_at_the_end_of_every_epoch_1_to_30(self):
        v = C.validation_epochs()
        self.assertEqual(v, tuple(range(1, 31)))
        self.assertEqual(len(v), 30)
        self.assertEqual(v[0], 1)
        self.assertEqual(v[-1], 30)
        self.assertIn(2, v)
        self.assertIn(29, v)
        self.assertTrue(all(b - a == 1 for a, b in zip(v, v[1:])))
        self.assertEqual(len(v), C.EPOCHS)

    def test_config_no_longer_encodes_the_range_as_a_two_element_list(self):
        ck = cfg()["validation"]["checkpoint"]
        self.assertNotIn("evaluate_epochs", ck)
        self.assertIs(ck["evaluate_every_epoch"], True)
        self.assertEqual(ck["epoch_start"], 1)
        self.assertEqual(ck["epoch_end"], 30)
        self.assertEqual(ck["evaluation_count"], 30)
        self.assertIs(ck["best_may_come_from_any_epoch_in_range"], True)
        self.assertEqual(ck["range_as_two_element_list"], "FORBIDDEN")
        self.assertEqual(ck["sequence_source"], "gpatbench.probe.contract.validation_epochs")

    def test_validation_epochs_rejects_a_malformed_contract(self):
        import copy
        for patch in ({"evaluate_every_epoch": False}, {"epoch_start": 0}, {"epoch_start": 2},
                      {"epoch_end": 29}, {"epoch_end": 31}):
            bad = copy.deepcopy(cfg())
            bad["validation"]["checkpoint"].update(patch)
            with self.assertRaises(C.ProbeContractViolation, msg=str(patch)):
                C.validation_epochs(bad)

    def test_the_sequence_has_exactly_one_source(self):
        """No second hard-coded copy of the 1..30 range may exist in the probe package."""
        import re
        for f in sorted((ROOT / "gpatbench/probe").rglob("*.py")):
            if f.name == "contract.py":
                continue
            src = f.read_text()
            self.assertNotIn("range(1, 31)", src, f.name)
            self.assertIsNone(re.search(r"\[\s*1\s*,\s*30\s*\]", src), f.name)

    def test_trainer_preflight_reports_thirty_validation_passes(self):
        out = T.run(dry_run=True)
        self.assertEqual(out["validation_passes"], 30)
        self.assertEqual(out["validation_epochs"], list(range(1, 31)))

    def test_an_intermediate_epoch_can_be_selected(self):
        """A maximum at epoch 17 must win; the old two-element reading could not express that."""
        scores = [(e, (e / 17.0 if e <= 17 else (34 - e) / 17.0)) for e in C.validation_epochs()]
        epoch, score = C.select_best_epoch(scores)
        self.assertEqual(epoch, 17)
        self.assertEqual(score, 1.0)

    def test_select_best_epoch_keeps_the_earlier_epoch_on_an_exact_tie(self):
        self.assertEqual(C.select_best_epoch([(1, 0.5), (2, 0.5), (3, 0.5)]), (1, 0.5))
        self.assertEqual(C.select_best_epoch([(1, 0.5), (2, 0.6), (3, 0.6)]), (2, 0.6))
        self.assertEqual(C.select_best_epoch([(1, 0.9), (2, 0.1)]), (1, 0.9))
        # inspect the EXECUTABLE body only: a comment saying "never >=" must not fail the check
        import ast
        fn = next(n for n in ast.walk(ast.parse(inspect.getsource(C).lstrip()))
                  if isinstance(n, ast.FunctionDef) and n.name == "select_best_epoch")
        body = ast.unparse(ast.Module(body=fn.body[1:], type_ignores=[]))   # drop the docstring
        self.assertIn("score > best_score", body)
        self.assertNotIn(">=", body)

    def test_checkpoint_rule_is_strictly_greater_so_ties_keep_the_earlier_epoch(self):
        self.assertFalse(MET.is_better(0.5, 0.5))
        self.assertTrue(MET.is_better(0.5 + 1e-15, 0.5))
        self.assertFalse(MET.is_better(0.5 - 1e-15, 0.5))
        src = inspect.getsource(MET.is_better)
        self.assertIn(">", src)
        self.assertNotIn(">=", src)
        ck = cfg()["validation"]["checkpoint"]
        self.assertEqual(ck["tie_break"], "earlier_epoch")
        self.assertEqual(ck["epsilon_tie_window"], "NONE")
        self.assertEqual(ck["test_involvement"], "NONE")


# ------------------------------------------------------------------ optimization
class TestOptimization(unittest.TestCase):
    def test_cosine_schedule_contract(self):
        s = cfg()["optimization"]["scheduler"]
        self.assertEqual(s["implementation"], "torch.optim.lr_scheduler.CosineAnnealingLR")
        self.assertEqual(s["T_max"], 30)
        self.assertEqual(s["eta_min"], 0.0)
        self.assertEqual(s["step_granularity"], "once_per_epoch")
        self.assertEqual(s["warmup"], "NONE")
        self.assertIs(s["per_batch_cosine"], False)
        self.assertIs(s["restarts"], False)
        self.assertIs(s["min_lr_1e_6"], False)

    def test_lr_sequence_matches_pytorch_and_starts_at_1e_4(self):
        m = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(m.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.T_MAX, eta_min=C.ETA_MIN)
        seq = []
        for _ in range(C.EPOCHS):
            seq.append(opt.param_groups[0]["lr"])
            sch.step()
        self.assertEqual(seq[0], 1e-4)
        self.assertEqual(opt.param_groups[0]["lr"], 0.0)
        self.assertEqual(len(seq), 30)
        self.assertTrue(all(a >= b for a, b in zip(seq, seq[1:])), "cosine must be monotone here")
        analytic = C.lr_schedule()
        for a, b in zip(seq, analytic):
            self.assertAlmostEqual(a, b, delta=max(1e-18, 1e-12 * max(a, b)))
        rows = list(csv.DictReader((AUDIT / "M5_ARTIFACT_PROBE_LR_SCHEDULE.csv").open()))
        self.assertEqual(len(rows), 30)
        self.assertEqual(float(rows[0]["lr_in_force_during_epoch_torch"]), 1e-4)

    def test_optimizer_and_amp_contract(self):
        o = cfg()["optimization"]
        self.assertEqual((o["optimizer"], o["lr"], o["weight_decay"]), ("AdamW", 1e-4, 1e-4))
        self.assertEqual((o["batch_size"], o["epochs"]), (64, 30))
        self.assertIs(o["amp"]["enabled"], True)
        self.assertEqual(o["amp"]["cpu_fallback"], "FORBIDDEN")
        self.assertEqual(o["amp"]["bfloat16_fallback"], "FORBIDDEN")
        self.assertEqual(o["amp"]["float32_fallback"], "FORBIDDEN")
        opt, sch = T.build_optimizer_and_scheduler(MODEL.build(C.SEED, pretrained=False))
        self.assertEqual(len(opt.param_groups), 1, "one group: no differential LR")
        self.assertEqual(opt.param_groups[0]["lr"], 1e-4)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 1e-4)


# ------------------------------------------------------------------ loaders and firewall
class TestLoadersAndFirewall(unittest.TestCase):
    def test_loader_contract(self):
        d = cfg()["dataloader"]
        self.assertIs(d["train"]["shuffle"], True)
        self.assertIs(d["val"]["shuffle"], False)
        self.assertIs(d["train"]["drop_last"], False)
        self.assertIs(d["val"]["drop_last"], False)
        self.assertEqual(d["train"]["batch_size"], 64)
        self.assertEqual(d["seed"], 42)
        self.assertEqual(d["gpu_defaults"], {"num_workers": 4, "pin_memory": True,
                                             "persistent_workers": True, "prefetch_factor": 2})
        self.assertIs(d["membership_depends_on_worker_scheduling"], False)

    def test_val_order_is_canonical_by_sample_id(self):
        rows = D.load_rows("VAL")
        self.assertEqual([r["sample_id"] for r in rows],
                         sorted(r["sample_id"] for r in rows))
        self.assertEqual(len(rows), 3121)

    def test_worker_seeding_is_deterministic(self):
        import random
        D.worker_init_fn(3)
        a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        D.worker_init_fn(3)
        b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(a, b)

    def test_test_split_is_refused_everywhere(self):
        with self.assertRaises(C.ProbeContractViolation):
            D.load_rows("TEST")
        with self.assertRaises(C.ProbeContractViolation):
            D.ProbeDataset("TEST")
        self.assertEqual(D.ALLOWED_SPLITS, ("TRAIN", "VAL"))

    def test_no_synthetic_code_path_exists(self):
        src = (ROOT / "gpatbench/probe/data.py").read_text()
        for bad in ("synthetic_root", "banks/", "generated_image", "bank_path"):
            self.assertNotIn(bad, src, bad)
        self.assertEqual(D.ProbeDataset("TRAIN", rows=[]).synthetic_samples, 0)

    def test_amendment_a1_firewall_is_recorded(self):
        f = cfg()["amendment_a1_firewall"]
        self.assertEqual(f["status"], "COMPLIANT")
        for k in ("probe_labels_enter_generator_optimization",
                  "probe_predictions_decide_generator_checkpoint_selection",
                  "probe_gradients_touch_generator_parameters",
                  "attack_macro_exported_as_generator_input"):
            self.assertIs(f[k], False, k)
        self.assertIs(cfg()["role"]["is_gpat_e_art"], False)


# ------------------------------------------------------------------ trainer refusals
class TestTrainerRefusals(unittest.TestCase):
    def test_cpu_authoritative_training_is_refused(self):
        if torch.cuda.is_available():
            self.skipTest("this guard is only meaningful without CUDA")
        with self.assertRaises(C.ProbeContractViolation):
            C.verify_environment(require_cuda=True)
        with self.assertRaises(C.ProbeContractViolation):
            T.run(dry_run=False)

    def test_non_frozen_config_is_refused(self):
        with self.assertRaises(C.ProbeContractViolation):
            C.load_config(ROOT / "configs/proposed/artifact_probe.proposed.yaml")

    def test_changed_config_hash_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "artifact_probe.yaml"
            p.write_text(C.FROZEN_CONFIG.read_text() + "\n# tampered\n")
            with self.assertRaises(C.ProbeContractViolation):
                C.load_config(p)

    def test_wrong_resnet_weight_hash_is_refused(self):
        self.assertEqual(T.verify_backbone_weights(), C.RESNET18_WEIGHT_SHA256)
        self.assertEqual(len(C.RESNET18_WEIGHT_SHA256), 64)
        self.assertTrue(C.RESNET18_WEIGHT_SHA256.startswith("f37072fd"))

    def test_dry_run_never_writes_a_checkpoint(self):
        out = T.run(dry_run=True)
        self.assertIs(out["dry_run"], True)
        self.assertIs(out["checkpoint_written"], False)
        self.assertFalse(T.CHECKPOINT_DIR.exists())
        self.assertFalse(T.CHECKPOINT_PATH.exists())

    def test_cli_is_the_only_training_entry_point(self):
        import ast
        fn = next(n for n in ast.walk(ast.parse((ROOT / "gpatbench/cli.py").read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "cmd_train_probe")
        called = {n.func.attr for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("run", called)
        for bad in ("build_loss", "build_optimizer_and_scheduler", "preflight", "seed_everything"):
            self.assertNotIn(bad, called, f"the CLI must not drive train.{bad} itself")

    def test_cli_refuses_authoritative_training_on_this_machine(self):
        r = subprocess.run([sys.executable, "-m", "gpatbench.cli", "train-probe", "--config",
                            "configs/frozen/artifact_probe.yaml"],
                           cwd=ROOT, capture_output=True, text=True)
        if torch.cuda.is_available():
            self.skipTest("CUDA present; the CPU guard cannot be exercised here")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("STOP", r.stdout + r.stderr)


# ------------------------------------------------------------------ milestone + smoke evidence
class TestMilestoneAndEvidence(unittest.TestCase):
    def test_m5_is_still_not_started(self):
        st = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        self.assertEqual(st["M5"]["status"], "NOT_STARTED")
        self.assertEqual(st["M6"]["status"], "NOT_STARTED")
        self.assertEqual(st["M5"]["preflight"]["phase"], "READY_FOR_GPU_EXECUTION_PREFLIGHT")
        o = st["M5"]["owner_resolution"]
        self.assertIs(o["trained"], False)
        self.assertIs(o["checkpoint_created"], False)
        self.assertIs(o["remote_facts_measured"], False)
        self.assertEqual(o["frozen_config_sha256"], CONFIG_SHA)
        self.assertEqual(cfg()["milestone_status"], "NOT_STARTED")
        self.assertEqual(cfg()["preflight_status"], "READY_FOR_GPU_EXECUTION_PREFLIGHT")

    def test_no_checkpoint_or_probe_model_exists(self):
        self.assertFalse((ROOT / "models/artifact_probe").exists())
        found = [p for p in ROOT.rglob("*")
                 if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt")
                 and not ({".git", ".venv", ".venv-audit"} & set(p.parts))
                 and "third_party/source_cache" not in p.as_posix()]
        self.assertEqual(found, [])

    def test_smoke_passed_without_training(self):
        s = smoke()
        self.assertEqual(s["status"], "PASS")
        self.assertEqual(s["failed_checks"], [])
        self.assertIs(s["authoritative_training"], False)
        self.assertIs(s["checkpoint_written"], False)
        self.assertIs(s["performance_claimed"], False)
        self.assertEqual(s["config_sha256"], CONFIG_SHA)
        for k in ("refuses_test_split", "refuses_test_dataset", "refuses_cpu_authoritative",
                  "refuses_non_frozen_config", "no_synthetic_code_path"):
            self.assertIs(s["checks"][k], True, k)

    def test_gpu_plan_states_no_remote_fact(self):
        t = (AUDIT / "M5_GPU_EXECUTION_PLAN.md").read_text()
        self.assertIn("sparc5090", t)
        self.assertIn("/home/sparc/workdir/longnm/GPAT_TransferBench", t)
        self.assertIn("PLAN ONLY", t)
        self.assertIn("rsync", t)
        self.assertIn("m2_sample_accounting.parquet", t)
        self.assertEqual(cfg()["execution"]["remote_host"], "sparc5090")
        self.assertIs(cfg()["execution"]["assume_remote_facts_before_measuring"], False)
        self.assertEqual(cfg()["execution"]["cpu_authoritative_training"], "FORBIDDEN")

    def test_disclosed_limitation_is_preserved_not_fixed(self):
        d = cfg()["disclosed_limitation"]
        for f in ("resampling", "dataset_balancing", "label_merging", "dropping_classes"):
            self.assertIn(f, d["must_not_be_fixed_by"], f)
        self.assertIn("SiW-Mv2", d["statement"])

    def test_frozen_m2_m3_m4_artifacts_unchanged(self):
        for rel, sha in (
            ("manifests/split_v1.parquet",
             "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"),
            ("manifests/pair_train_stats_v1.json",
             "a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50"),
            ("manifests/pairs_train_v1.parquet",
             "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243"),
            ("manifests/val_pairs_v1.parquet",
             "84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872"),
            ("manifests/difffas_bin_idfree_train_v1.parquet",
             "0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6"),
        ):
            self.assertEqual(C.sha256_file(ROOT / rel), sha, rel)


if __name__ == "__main__":
    unittest.main()
