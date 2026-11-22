"""M8 contract finalization package: both FINAL_DRAFTs are unapproved and carry exactly the owner-selected options, their
inputs are hash-pinned, the feasibility record re-derives and is feasible, the D10 support counts re-derive from
TRAIN/VAL metadata, the environment plan is a plan (no lock), and no approved contract existed at the commit that
added this test (pinned to that commit, B1). No image, no TEST row.
"""
import collections
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE = json.loads((ROOT / "configs/amendments/m8_smoke_test_contract.FINAL_DRAFT.yaml").read_text())
EVAL = json.loads((ROOT / "configs/amendments/m8_downstream_evaluator_contract.FINAL_DRAFT.yaml").read_text())
ENV = json.loads((ROOT / "environments/m8_downstream.planned.json").read_text())

OWNER_S = {"S1_scope": "A_GLOBAL_END_TO_END", "S2_class_definition": "BINARY_LABEL_PER_DATASET",
           "S3_siwmv2_subject_unit": "CONTENT_GROUP_ID", "S4_selection_rule": "HASH_RANK",
           "S5_splits_and_test": "TRAIN_VAL_ONLY_TEST_PROXY", "S6_bank_pair_population": "FROZEN_PAIRS_TARGET_IN_TOY",
           "S7_methods_covered": "ALL_METHODS_WITH_BANK_WRITERS", "S8_evaluator_budget_and_environment": "SMOKE_SCHEDULE_CPU",
           "S9_determinism_criterion": "BITWISE_ALL_STAGES", "S10_seeds": "ALL_THREE_BANK_SEED42_EVALUATOR"}
OWNER_D = {"D1_epoch_and_sampler": "LIVE_PASS", "D2_eval_preprocessing": "RESIZE_FULL_BICUBIC",
           "D3_target_and_score": "SPOOF_IS_ONE", "D4_decision_operator": "SPOOF_IF_SCORE_GE_THRESHOLD",
           "D5_checkpoint_metric": "EPOCH_OWN_VAL_THRESHOLD", "D6_lr_schedule_details": "PER_STEP_FROM_ZERO",
           "D7_eer_convention": "LINEAR_INTERPOLATION", "D8_positive_class_secondary": "SPOOF_POSITIVE",
           "D9_frame_level_threshold": "VIDEO_THRESHOLD", "D11_dinov3": "RESNET18_ONLY_IN_SMOKE",
           "D12_synthetic_input": "SEED_MATCHED_RAW", "D13_determinism_and_device": "CPU_NO_AMP_SMOKE_GPU_AMP_SCIENCE",
           "D14_augmentation_order": "CROP_FLIP_JITTER_NORMALIZE"}


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def exists_at_added_commit(path):
    c = subprocess.run(["git", "log", "--diff-filter=A", "-1", "--format=%H", "--", "tests/test_m8_contract_finalization.py"],
                       cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not c:
        return (ROOT / path).exists()
    return subprocess.run(["git", "cat-file", "-e", f"{c}:{path}"], cwd=ROOT, capture_output=True).returncode == 0


class M8ContractFinalization(unittest.TestCase):
    def test_01_final_drafts_unapproved_with_owner_options(self):
        for d in (SMOKE, EVAL):
            self.assertEqual((d["status"], d["approved"]), ("DRAFT_FOR_OWNER_FINAL_APPROVAL", False))
        self.assertEqual(SMOKE["decisions"], OWNER_S)
        self.assertEqual({k: v for k, v in EVAL["decisions"].items() if k != "D10_siwmv2_raw_coverage"}, OWNER_D)
        self.assertIn("requires owner approval", EVAL["decisions"]["D10_siwmv2_raw_coverage"])
        old_s = json.loads((ROOT / "configs/amendments/m8_smoke_test_contract.DRAFT.yaml").read_text())["questions"]
        old_d = json.loads((ROOT / "configs/amendments/m8_downstream_evaluator_contract.DRAFT.yaml").read_text())["questions"]
        for q, opt in OWNER_S.items():
            self.assertIn(opt, old_s[q]["options"])
        for q, opt in OWNER_D.items():
            self.assertIn(opt, old_d[q]["options"])

    def test_02_inputs_pinned(self):
        for d in (SMOKE, EVAL):
            for path, digest in d["based_on"].items():
                if path.endswith((".yaml", ".json")) and "/" in path:
                    self.assertEqual(sha(path), digest, path)
        self.assertIn("NOT a subject identity", SMOKE["pinned_execution_details"]["S3"])

    def test_03_no_activation_at_decision_commit(self):
        for p in ("configs/amendments/m8_smoke_test_contract.yaml", "configs/amendments/m8_downstream_evaluator_contract.yaml",
                  "environments/m8_downstream.lock.json", "outputs/audit/M8_SMOKE_TEST_RECORD.json"):
            self.assertFalse(exists_at_added_commit(p), p)
        self.assertEqual(ENV["status"], "PLANNED_NOT_CREATED")

    def test_04_feasibility_record(self):
        rec = json.loads((ROOT / "outputs/audit/M8_TOY_SUBSET_FEASIBILITY.json").read_text())
        self.assertEqual((rec["status"], rec["worktree_dirty_at_run"], rec["inputs"]["test_rows_read"]),
                         ("TOY_SUBSET_FEASIBLE", False, 0))
        f = rec["feasibility"]
        self.assertEqual(f["live_pass_full_batches_per_epoch"] * SMOKE["smoke_schedule_SMOKE_NON_SCIENTIFIC"]["epochs"], 6)
        self.assertEqual(SMOKE["smoke_schedule_SMOKE_NON_SCIENTIFIC"]["steps_per_epoch"], f["live_pass_full_batches_per_epoch"])
        self.assertFalse(any(f["replacement_needed_within_epoch"].values()))

    def test_05_d10_support_counts(self):
        from gpatbench.downstream.firewall import read_permitted_rows
        rows = [r for r in read_permitted_rows(ROOT, ["dataset", "video_id", "label_binary", "attack_raw"])
                if r["dataset"] == "siwmv2" and r["label_binary"] == 1]
        per_video = collections.defaultdict(set)
        for r in rows:
            per_video[(r["split"], r["video_id"])].add(r["attack_raw"])
        self.assertTrue(all(len(v) == 1 and None not in v for v in per_video.values()))
        counts = collections.Counter((sp, next(iter(v))) for (sp, _), v in per_video.items())
        got = {t: [counts[("TRAIN", t)], counts[("VAL", t)]] for t in {t for _, t in counts}}
        self.assertEqual(got, EVAL["D10_RAW_TYPE_COVERAGE_V1_proposed"]["support_counts_train_val_videos"])
        self.assertEqual(len(got), 14)

    def test_06_environment_plan(self):
        names = [w["name"] for w in ENV["added_wheels"]]
        self.assertEqual(names, ["matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing"])
        req = ENV["requirements_m8_downstream_txt"]
        for w, line in zip(ENV["added_wheels"], req):
            self.assertEqual(line, f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}")
        self.assertEqual(ENV["construction_options"]["recommended"], "A_VENV_OVERLAY")


if __name__ == "__main__":
    unittest.main()
