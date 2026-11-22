"""M8 smoke-test decision package: the DRAFT contract is never authority, its facts re-derive from manifest metadata
(TRAIN/VAL rows only), and at the commit that added this test no approved contract or smoke record existed and the
E01/E02 ruling still had smoke_test_record null. Historical claims are pinned to that commit (B1). No image is read.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402

DRAFT = "configs/amendments/m8_smoke_test_contract.DRAFT.yaml"
ASSESSMENT = "outputs/audit/M8_SMOKE_TEST_AUTHORITY_ASSESSMENT.json"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def added_commit():
    c = git("log", "--diff-filter=A", "-1", "--format=%H", "--", "tests/test_m8_smoke_decision_package.py").stdout.strip()
    return c or None   # None while uncommitted: fall back to the worktree


class M8SmokeDecisionPackage(unittest.TestCase):
    def test_01_draft_is_not_authority(self):
        d = json.loads((ROOT / DRAFT).read_text())
        self.assertEqual((d["status"], d["approved"]), ("DRAFT_PENDING_OWNER_DECISION", False))
        self.assertEqual(sorted(d["questions"]), sorted(f"S{i}_" + k for i, k in [
            (1, "scope"), (2, "class_definition"), (3, "siwmv2_subject_unit"), (4, "selection_rule"),
            (5, "splits_and_test"), (6, "bank_pair_population"), (7, "methods_covered"),
            (8, "evaluator_budget_and_environment"), (9, "determinism_criterion"), (10, "seeds")]))
        with self.assertRaises(bw.BankBlocked):
            bw.load_production_authorization("E01", ROOT / DRAFT)

    def test_02_state_at_decision_commit(self):
        c = added_commit()
        def exists(path):
            return git("cat-file", "-e", f"{c}:{path}").returncode == 0 if c else (ROOT / path).exists()
        def read(path):
            return git("show", f"{c}:{path}").stdout if c else (ROOT / path).read_text()
        self.assertFalse(exists("configs/amendments/m8_smoke_test_contract.yaml"))
        self.assertFalse(exists("outputs/audit/M8_SMOKE_TEST_RECORD.json"))
        self.assertIsNone(json.loads(read(bw.RULING_PATH))["smoke_test_record"])
        o3 = json.loads(read("configs/amendments/e07c_m6d6jr_science_launch_decision.yaml"))["O3_smoke_test"]
        self.assertIs(o3["smoke_test_completed"], False)
        self.assertEqual(o3["must_exercise"], ["generator", "bank", "evaluator", "metrics", "plots"])

    def test_03_assessment_facts_rederive_from_metadata(self):
        import pyarrow.parquet as pq
        a = json.loads((ROOT / ASSESSMENT).read_text())
        self.assertEqual(a["status"], "M8_SMOKE_TEST_BLOCKED_BY_OWNER_DECISION")
        self.assertEqual((a["smoke_test_executed"], a["real_images_read"], a["banks_generated"], a["test_rows_inspected"]),
                         (False, 0, 0, 0))
        rows = [r for r in pq.read_table(ROOT / "manifests/split_v1.parquet",
                                         columns=["dataset", "subject_id_global", "content_group_id", "label_binary",
                                                  "split"]).to_pylist() if r["split"] in ("TRAIN", "VAL")]
        for key, fact in a["toy_subset_facts_train_val_only"].items():
            sp, ds = key.split("/")
            sel = [r for r in rows if r["split"] == sp and r["dataset"] == ds]
            for lab, name in ((0, "live_units"), (1, "spoof_units")):
                self.assertEqual(len({r[fact["unit"]] for r in sel if r["label_binary"] == lab}), fact[name], key)
            self.assertEqual(sum(1 for r in sel if r["subject_id_global"] is None), fact["null_subject_id_rows"])

    def test_04_production_refused_while_unbound(self):
        if json.loads((ROOT / bw.RULING_PATH).read_text())["smoke_test_record"] is not None or \
                getattr(bw, "SMOKE_BINDING_PATH", None) and (ROOT / bw.SMOKE_BINDING_PATH).is_file():
            self.skipTest("smoke record bound by a later owner approval")
        for m in bw.METHODS:
            with self.assertRaises(bw.BankBlocked) as cm:
                bw.load_production_authorization(m)
            self.assertEqual(cm.exception.status, bw.M8_SMOKE_GATE_STATUS)


if __name__ == "__main__":
    unittest.main()
