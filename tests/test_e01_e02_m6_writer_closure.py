"""E01/E02 M6 writer-readiness closure: per-method statuses, evidence hashes pinned to the evidence commit, production
bank fields null (no fabricated counts or hashes), M8 smoke test still pending, method_status.csv not updated.
No image is read.
"""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402

EVIDENCE_COMMIT = "6f15f8a523eed40067162bb6e1c915a1ba46d9bf"


def blob(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def rec(m):
    return json.loads((ROOT / f"outputs/audit/{m}_M6_WRITER_READINESS_CLOSURE.json").read_text())


class E01E02M6WriterClosure(unittest.TestCase):
    def test_01_statuses(self):
        for m in bw.METHODS:
            r = rec(m)
            self.assertEqual(r["status"], f"{m}_M6_WRITER_QUALIFIED_M8_BANK_PENDING")
            self.assertFalse(r["is_bank_generation_completion_record"])
            self.assertFalse(r["milestone_states"]["real_production_banks_generated"])
            self.assertTrue(r["milestone_states"]["m8_smoke_test"].startswith("PENDING"))
            self.assertTrue(r["milestone_states"]["downstream_evaluation"].startswith("PENDING"))
            self.assertEqual(r["next_authorized_milestone"], "M8 deterministic smoke test")
            self.assertEqual(r["production_gate_status_now"], bw.M8_SMOKE_GATE_STATUS)

    def test_02_no_fabricated_bank_outputs(self):
        for m in bw.METHODS:
            pb = rec(m)["production_bank"]
            self.assertFalse(pb["generated"])
            for k in ("output_count", "failure_count", "bank_manifest_sha256", "output_sha256", "run_id"):
                self.assertIsNone(pb[k], k)

    def test_03_evidence_hashes_at_evidence_commit(self):
        for m in bw.METHODS:
            r = rec(m)
            self.assertEqual(r["authority"]["qualification_evidence_commit"], EVIDENCE_COMMIT)
            for path, digest in r["evidence_sha256"].items():
                self.assertEqual(hashlib.sha256(blob(EVIDENCE_COMMIT, path)).hexdigest(), digest, path)
            cfg = r["frozen_config"]
            self.assertEqual(hashlib.sha256((ROOT / cfg["path"]).read_bytes()).hexdigest(), cfg["sha256"])

    def test_04_qualification_passed_in_both_environments(self):
        for m in bw.METHODS:
            q = rec(m)["qualification"]
            for env in ("laptop_venv", "gpat_m5"):
                self.assertEqual(q[env]["status"], "E01_E02_BANK_WRITER_QUALIFIED_POST_RULING_SYNTHETIC")
                self.assertEqual(len(q[env]["gates"]), 10)
                self.assertTrue(all(q[env]["gates"].values()))
            self.assertEqual((q["gpat_m5"]["python"], q["gpat_m5"]["numpy"]), ("3.11.16", "2.4.6"))

    def test_05_method_specific(self):
        e01, e02 = rec("E01")["method_specific"], rec("E02")["method_specific"]
        self.assertEqual(e01["operator_order"][0], "Color_Diversity")
        self.assertEqual(sorted(e01["full_population_operator_counts_per_seed"].values()), [1104] * 2 + [1105] * 6)
        for v in e01["g8_full_population_checks"].values():
            self.assertEqual((v["operator_mismatch"], v["pair_id_suffix_agreements"], v["operator_seed_mismatch"]),
                             (0, 0, 0))
        self.assertEqual(rec("E01")["authority"]["owner_ruling_decision"]["Q2_e01_round_robin_ordinal"],
                         "MANIFEST_POSITION_ZERO_BASED")
        g9 = e02["g9_independent_recomputation"]
        self.assertEqual((g9["rows"], g9["pixel_mismatch_rows"], g9["edit_map_mismatch_rows"]), (72, 0, 0))
        self.assertLessEqual(e02["max_abs_imag_fixture"], e02["imag_tolerance"])
        self.assertEqual(rec("E02")["authority"]["owner_ruling_decision"]["Q3_e02_uint8_quantization"],
                         "CONFIRM_CLIP_THEN_RINT")

    def test_06_smoke_gate_and_registry_unchanged_at_closure(self):
        for m in bw.METHODS:
            r = rec(m)
            self.assertFalse(r["method_status_csv"]["updated"])
            o3 = json.loads(blob(EVIDENCE_COMMIT, "configs/amendments/e07c_m6d6jr_science_launch_decision.yaml"))
            self.assertIs(o3["O3_smoke_test"]["smoke_test_completed"], False)


if __name__ == "__main__":
    unittest.main()
