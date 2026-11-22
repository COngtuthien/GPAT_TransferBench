"""M8 owner approval: each approved contract equals its FINAL_DRAFT (60b50ef) except governance fields (and, for the
evaluator, the D10 decision value and the renamed D10 block whose content is unchanged). FINAL_DRAFTs and DRAFTs are
byte-unchanged. Stdlib only.
"""
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "60b50efea6c3f7a71ad7a5bf3e3cff7de1340614"
GOV = {"status", "approved", "approval_date", "approved_final_draft", "document", "owner_ruling", "confirmed_items",
       "items_requiring_explicit_confirmation_at_final_approval", "authorizes"}


def at(path):
    return subprocess.run(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


class M8ContractOwnerApproval(unittest.TestCase):
    def check(self, name, d10=False):
        draft_raw = at(f"configs/amendments/{name}.FINAL_DRAFT.yaml")
        self.assertEqual((ROOT / f"configs/amendments/{name}.FINAL_DRAFT.yaml").read_bytes(), draft_raw)
        d, a = json.loads(draft_raw), json.loads((ROOT / f"configs/amendments/{name}.yaml").read_text())
        self.assertEqual((a["status"], a["approved"]), ("OWNER_APPROVED", True))
        self.assertEqual(a["approved_final_draft"]["sha256"], hashlib.sha256(draft_raw).hexdigest())
        self.assertEqual(a["confirmed_items"], d["items_requiring_explicit_confirmation_at_final_approval"])
        self.assertTrue((ROOT / a["document"]).is_file())
        dd, aa = dict(d), dict(a)
        if d10:
            self.assertEqual(aa.pop("D10_RAW_TYPE_COVERAGE_V1"), dd.pop("D10_RAW_TYPE_COVERAGE_V1_proposed"))
            self.assertEqual(aa["decisions"].pop("D10_siwmv2_raw_coverage"), "RAW_TYPE_COVERAGE_V1")
            dd["decisions"] = {k: v for k, v in dd["decisions"].items() if k != "D10_siwmv2_raw_coverage"}
        for k in GOV:
            dd.pop(k, None)
            aa.pop(k, None)
        self.assertEqual(aa, dd)
        for flag in ("production_banks", "scientific_downstream_training", "test_access", "gpat", "production_binding"):
            self.assertIs(a["authorizes"][flag], False)

    def test_01_smoke_contract(self):
        self.check("m8_smoke_test_contract")

    def test_02_evaluator_contract(self):
        self.check("m8_downstream_evaluator_contract", d10=True)

    def test_03_earlier_drafts_unchanged(self):
        for p in ("configs/amendments/m8_smoke_test_contract.DRAFT.yaml",
                  "configs/amendments/m8_downstream_evaluator_contract.DRAFT.yaml",
                  "configs/amendments/m8_smoke_test_s1_owner_decision.yaml"):
            self.assertEqual((ROOT / p).read_bytes(), at(p), p)


if __name__ == "__main__":
    unittest.main()
