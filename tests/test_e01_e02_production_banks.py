"""Committed E01/E02 production bank audits and family closures re-derive from their committed evidence.

Each per-bank record must be PASS with every gate true and bound (collected_sha256) to the committed collected facts;
rebuilding it from those facts with tools/e01_e02_production_bank_audit.py build must give an identical record. Each
family closure must re-derive byte-identically from the three committed audits. Banks not yet committed are skipped.
No image is read (bank payloads live on the GPU host only).
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/audit"
BANKS = [(m, s) for m in ("E01", "E02") for s in (42, 1337, 2026)]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


class ProductionBankEvidence(unittest.TestCase):
    def test_01_bank_audits_pass_and_rederive(self):
        seen = 0
        for m, s in BANKS:
            rec_p = AUDIT / f"{m}_PRODUCTION_BANK_SEED{s}.json"
            if not rec_p.is_file():
                continue
            seen += 1
            rec = json.loads(rec_p.read_text())
            col = AUDIT / f"E01_E02_PRODUCTION/{m}_s{s}/audit_collected.json"
            self.assertEqual(rec["status"], f"{m}_SEED{s}_PRODUCTION_BANK_AUDIT_PASS")
            self.assertEqual(rec["failed_gates"], [])
            self.assertTrue(all(rec["gates"].values()))
            self.assertEqual((rec["output_count"], rec["failure_count"]), (8838, 0))
            self.assertEqual(hashlib.sha256(col.read_bytes()).hexdigest(), rec["collected_sha256"])
            with tempfile.TemporaryDirectory() as td:
                r = run("tools/e01_e02_production_bank_audit.py", "build", "--collected", str(col),
                        "--out-prefix", str(Path(td) / "x"))
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual(json.loads((Path(td) / "x.json").read_text()), rec)
        if not seen:
            self.skipTest("no production bank committed yet")

    def test_02_family_closures_rederive(self):
        for m in ("E01", "E02"):
            cl = AUDIT / f"{m}_PRODUCTION_FAMILY_CLOSURE.json"
            if not cl.is_file():
                continue
            with tempfile.TemporaryDirectory() as td:
                r = run("tools/e01_e02_production_family_closure.py", "--method", m, "--out-prefix", str(Path(td) / "c"))
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertEqual((Path(td) / "c.json").read_bytes(), cl.read_bytes())
                self.assertEqual((Path(td) / "c.md").read_bytes(), cl.with_suffix(".md").read_bytes())
            c = json.loads(cl.read_text())
            self.assertEqual(c["status"], f"{m}_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED")
            self.assertEqual(c["total_outputs"], 26514)
            self.assertFalse(c["is_downstream_evaluation_complete"])
            self.assertFalse(c["method_status_csv_updated"])

    def test_03_method_status_unchanged(self):
        self.assertEqual(hashlib.sha256((AUDIT / "method_status.csv").read_bytes()).hexdigest(),
                         "fd97e7cbac35af1ef02fd495d21ad3cfcebffb47702907ab01afd2d447003723")


if __name__ == "__main__":
    unittest.main()
