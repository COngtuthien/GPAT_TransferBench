"""M8 smoke-test S1 partial owner decision: S1 = A_GLOBAL_END_TO_END (the DRAFT's exact option id), the DRAFT is
byte-unchanged since 025d9e8, the full contract stays unapproved, nothing is authorized and E01/E02 production stays
refused. Stdlib + bank_writer only; no image is read.
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

RECORD = "configs/amendments/m8_smoke_test_s1_owner_decision.yaml"
DRAFT = "configs/amendments/m8_smoke_test_contract.DRAFT.yaml"
DRAFT_COMMIT = "025d9e8993ff18ca7b7b16e23a15411f45f44683"
R = json.loads((ROOT / RECORD).read_text())


def git_blob(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


class M8SmokeS1OwnerDecision(unittest.TestCase):
    def test_01_s1_recorded_with_draft_identifier(self):
        d = json.loads(git_blob(DRAFT_COMMIT, DRAFT))
        self.assertEqual((R["status"], R["approved"], R["full_contract_approved"]),
                         ("OWNER_APPROVED_PARTIAL_DECISION", True, False))
        self.assertEqual(R["decision"]["question"], "S1_scope")
        self.assertIn(R["decision"]["option"], d["questions"]["S1_scope"]["options"])
        self.assertEqual(R["decision"]["option"], "A_GLOBAL_END_TO_END")
        self.assertEqual(sorted(R["open_questions"]), sorted(q for q in d["questions"] if q != "S1_scope"))

    def test_02_draft_preserved(self):
        blob = git_blob(DRAFT_COMMIT, DRAFT)
        self.assertEqual(hashlib.sha256(blob).hexdigest(), R["draft"]["sha256"])
        self.assertEqual(hashlib.sha256((ROOT / DRAFT).read_bytes()).hexdigest(), R["draft"]["sha256"])
        self.assertFalse(json.loads(blob)["approved"])

    def test_03_authorizes_nothing(self):
        self.assertEqual(set(R["authorizes"].values()), {False})
        self.assertTrue((ROOT / R["document"]).is_file())

    def test_04_frozen_authorities(self):
        for key, digest in R["frozen_authorities"].items():
            p = ROOT / key.split(" (")[0]
            at = git_blob(DRAFT_COMMIT, key.split(" (")[0])
            self.assertEqual(hashlib.sha256(at).hexdigest(), digest, key)
            self.assertTrue(p.is_file())

    def test_05_e01_e02_production_refused_while_unbound(self):
        if json.loads((ROOT / bw.RULING_PATH).read_text())["smoke_test_record"] is not None or \
                getattr(bw, "SMOKE_BINDING_PATH", None) and (ROOT / bw.SMOKE_BINDING_PATH).is_file():
            self.skipTest("smoke record bound by a later owner approval")
        for m in bw.METHODS:
            with self.assertRaises(bw.BankBlocked) as cm:
                bw.load_production_authorization(m)
            self.assertEqual(cm.exception.status, bw.M8_SMOKE_GATE_STATUS)


if __name__ == "__main__":
    unittest.main()
