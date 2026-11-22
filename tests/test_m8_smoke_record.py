"""M8 smoke record and evidence: the committed record passes the strict validator against the approved contract, every
gate and stage passed, the copied artifacts match run A's hash listing, the independent re-derivation reproduces the
tables from the committed prediction files, the lock matches, and at the commit that added this test nothing was bound
(E01/E02 smoke_test_record null). Stdlib + pyarrow; no image is opened.
"""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

EV = ROOT / "outputs/audit/M8_SMOKE_EVIDENCE"
REC = json.loads((ROOT / "outputs/audit/M8_SMOKE_TEST_RECORD.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class M8SmokeRecord(unittest.TestCase):
    def test_01_validator_passes(self):
        from gpatbench.downstream.smoke_evidence import validate_smoke_record
        v = validate_smoke_record(REC, ROOT)
        self.assertEqual((v["admissible"], v["passed"], v["problems"]), (True, True, []))
        self.assertEqual((REC["mode"], REC["smoke_test_passed"], REC["scientific_results_produced"]), ("real", True, False))
        self.assertTrue(all(REC["gates"].values()))
        self.assertEqual({s: v["status"] for s, v in REC["stages"].items()},
                         {s: "PASS" for s in ("generator", "bank", "evaluator", "metrics", "plots")})
        self.assertEqual(REC["contract_sha256"], sha(ROOT / "configs/amendments/m8_smoke_test_contract.yaml"))

    def test_02_coverage(self):
        banks = REC["coverage"]["banks"]
        self.assertEqual(sorted(banks), sorted(f"{m}.seed_{s}" for m in ("E01", "E02") for s in (42, 1337, 2026)))
        self.assertTrue(all(v["rows"] == v["outputs"] == 186 and v["failures"] == 0 for v in banks.values()))
        self.assertEqual(sorted(REC["coverage"]["detectors"]), ["E00", "E01", "E02"])
        self.assertTrue(all(d["total_steps"] == 6 for d in REC["coverage"]["detectors"].values()))
        self.assertEqual(REC["toy_selection"]["toy_pairs"], 186)
        self.assertEqual((REC["independent_replay"]["identical"], REC["independent_replay"]["differences"]), (True, []))
        fw = REC["test_firewall"]
        self.assertEqual((fw["violations"], fw["listing_attempts"], fw["test_rows_used"], fw["atime_changed_files"]), (0, 0, 0, 0))

    def test_03_artifacts_match_hash_listing(self):
        hashes = json.loads((EV / "artifact_hashes_run_a.json").read_text())
        self.assertEqual(hashes, json.loads((EV / "artifact_hashes_run_b.json").read_text()))
        n = 0
        for rel, d in hashes.items():
            p = EV / "run_a" / rel
            if p.exists():
                n += 1
                self.assertEqual(sha(p), d, rel)
        self.assertGreaterEqual(n, 30)
        self.assertEqual(REC["toy_selection"]["toy_manifest_sha256"], hashes["select/toy_manifest.json"])

    def test_04_independent_rederivation(self):
        import m8_smoke_audit
        r = m8_smoke_audit.rederive(EV / "run_a" / "evaluator", EV / "run_a" / "metrics")
        self.assertTrue(r["ok"], r)

    def test_05_environment_lock(self):
        env = json.loads((EV / "environment_qualification.json").read_text())
        self.assertTrue(env["all_pass"])
        self.assertEqual(sha(ROOT / "environments/m8_downstream.lock.json"), env["lock_sha256"], REC["environment"]["lock_sha256"])
        self.assertTrue(env["gates"]["V4_cpu_training_bitwise_two_processes"])

    def test_06_nothing_bound_at_record_commit(self):
        c = subprocess.run(["git", "log", "--diff-filter=A", "-1", "--format=%H", "--", "tests/test_m8_smoke_record.py"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
        path = "configs/amendments/e01_e02_bank_milestone_ruling.yaml"
        raw = subprocess.run(["git", "show", f"{c}:{path}"], cwd=ROOT, capture_output=True).stdout if c else \
            (ROOT / path).read_bytes()
        self.assertIsNone(json.loads(raw)["smoke_test_record"])


if __name__ == "__main__":
    unittest.main()
