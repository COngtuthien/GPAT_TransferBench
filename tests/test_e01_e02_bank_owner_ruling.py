"""E01/E02 bank milestone owner ruling: the approved record answers Q1-Q4 with the DRAFT's recommended options, keeps
the DRAFT and the 028b5c1 qualification evidence byte-unchanged, pins the frozen authorities, authorizes no bank and
leaves production refused (M8 smoke-test record absent). No image is read.
"""
import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402
from methods.common.config import load_method_config  # noqa: E402

APPROVED = "configs/amendments/e01_e02_bank_milestone_ruling.yaml"
DRAFT = "configs/amendments/e01_e02_bank_milestone_ruling.DRAFT.yaml"
DRAFT_COMMIT = "01de91ac3d532d008a046cba4ade173d8fbe555f"
DRAFT_SHA256 = "e49df97fa2eefa451b26efae89077d155697c3bbafdd9a9c26981614bac13482"
DOC = "docs/spec/amendments/GPAT_TransferBench_v1_0_E01_E02_Bank_Milestone_Owner_Ruling_Addendum.md"
EVIDENCE_LIST = "outputs/audit/E01_E02_BANK_EVIDENCE_SHA256.txt"
DECISIONS = {"Q1_milestone_owner": "A_M8_SMOKE_GATED",
             "Q2_e01_round_robin_ordinal": "MANIFEST_POSITION_ZERO_BASED",
             "Q3_e02_uint8_quantization": "CONFIRM_CLIP_THEN_RINT",
             "Q4_execution_host": "SEPARATE_CLONE_CPU_CONCURRENT"}


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


A = json.loads((ROOT / APPROVED).read_text())
D = json.loads((ROOT / DRAFT).read_text())


class E01E02BankOwnerRuling(unittest.TestCase):
    def test_01_approved_governance_and_decisions(self):
        self.assertEqual((A["schema_version"], A["status"], A["approved"]),
                         (bw.RULING_SCHEMA, "OWNER_APPROVED", True))
        self.assertEqual(A["decisions"], DECISIONS)
        for q, choice in A["decisions"].items():
            self.assertIn(choice, D["questions"][q]["options"])
            self.assertEqual(choice, D["questions"][q]["recommended"], q)
            self.assertEqual(A["owner_ruling"][q]["decision"], choice)
        self.assertTrue((ROOT / DOC).is_file())

    def test_02_draft_preserved(self):
        self.assertEqual(sha(DRAFT), DRAFT_SHA256)
        at_commit = subprocess.run(["git", "show", f"{DRAFT_COMMIT}:{DRAFT}"], cwd=ROOT, capture_output=True,
                                   check=True).stdout
        self.assertEqual(hashlib.sha256(at_commit).hexdigest(), DRAFT_SHA256)
        self.assertEqual((A["approved_draft"]["sha256"], A["approved_draft"]["commit"]), (DRAFT_SHA256, DRAFT_COMMIT))
        self.assertEqual((D["status"], D["approved"]), ("DRAFT_PENDING_OWNER_DECISION", False))

    def test_03_historical_qualification_evidence_unchanged(self):
        # Every listed file matches its digest at the evidence commit; evidence records and the DRAFT are also
        # byte-unchanged now (source files may change in later, separately qualified commits).
        for line in (ROOT / EVIDENCE_LIST).read_text().splitlines():
            digest, path = line.split(None, 1)
            blob = subprocess.run(["git", "show", f"028b5c1d4b3e86270c22eb8db54360ece5daaaca:{path}"], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest, path)
            if path.startswith(("outputs/audit/", "configs/amendments/")):
                self.assertEqual(sha(path), digest, path)

    def test_04_frozen_authorities_pinned(self):
        for key, digest in A["frozen_authorities"].items():
            self.assertEqual(sha(key.split(" (")[0]), digest, key)

    def test_05_authorizes_nothing_and_smoke_record_absent(self):
        self.assertIsNone(A["smoke_test_record"])
        self.assertEqual(set(A["authorizes"].values()), {False})
        for m in bw.METHODS:
            with self.assertRaises(bw.BankBlocked):
                bw.load_production_authorization(m)

    def test_06_operator_order_pinned_to_config_and_official_source(self):
        order = A["owner_ruling"]["Q2_e01_round_robin_ordinal"]["pinned_operator_order"]
        self.assertEqual(order, load_method_config("E01")["generation"]["operator_set"])
        src = ROOT / "third_party/source_cache/fas_aug/data/FAS_Augmentations.py"
        if not src.is_file():
            self.skipTest("FAS-Aug source cache not present")
        body = src.read_text()
        al = body[body.index("self.al = ["):]
        al = al[:al.index("]\n")]
        official = re.findall(r'^\s*\("(\w+)",\s*[\d.]+,\s*[\d.]+,\s*(?:True|False)\),\s*#\s*(\d+)', al, re.M)
        self.assertEqual([n for n, _ in official], order)
        self.assertEqual([int(i) for _, i in official], list(range(8)))

    def test_07_population_split_matches_ruling(self):
        import pyarrow.parquet as pq
        ids = pq.read_table(ROOT / bw.PAIRS_PATH, columns=["pair_id"]).column("pair_id").to_pylist()
        self.assertEqual(ids, [f"PTR{i:06d}" for i in range(1, 8839)])
        counts = [sum(1 for p in range(len(ids)) if p % 8 == k) for k in range(8)]
        self.assertEqual(counts, [1105] * 6 + [1104] * 2)


if __name__ == "__main__":
    unittest.main()
