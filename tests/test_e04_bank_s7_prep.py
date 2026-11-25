"""E04 S7 plan, governance DRAFTs and reproduction manifest: byte-identical rebuild; DRAFTs never unlock anything."""
import json
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import e04_bank_s7_prep as prep  # noqa: E402
from methods.common import learned_bank as lb  # noqa: E402

FILES = (prep.S7_DRAFT, prep.STORAGE_DRAFT, prep.PRODUCTION_DRAFT, prep.REPRO, prep.PLAN + '.json', prep.PLAN + '.md')


class Prep(unittest.TestCase):
    def test_rebuild_byte_identical(self):
        with tempfile.TemporaryDirectory() as t:
            prep.build(ROOT, out_root=t)
            for rel in FILES:
                self.assertEqual(Path(t, rel).read_bytes(), (ROOT / rel).read_bytes(), rel)

    def test_drafts_not_approved_and_not_loadable(self):
        for rel in (prep.S7_DRAFT, prep.STORAGE_DRAFT, prep.PRODUCTION_DRAFT):
            d = json.loads((ROOT / rel).read_text())
            self.assertIs(d['approved'], False, rel)
            self.assertNotEqual(d['status'], 'OWNER_APPROVED', rel)
        # history scope (owner decision B1): neither activated record existed at this milestone (2caff0d); the owner
        # activated both in the 2026-11-25 production session; an activation keeps every DRAFT registry field
        for rel in (lb.E04_STORAGE_EXTENSION_PATH, 'configs/amendments/learned_bank_production_E04.yaml'):
            self.assertNotEqual(subprocess.run(['git', 'cat-file', '-e', '2caff0d:' + rel], cwd=str(ROOT),
                                               stderr=subprocess.DEVNULL).returncode, 0, rel)
        if (ROOT / lb.E04_STORAGE_EXTENSION_PATH).exists():
            a, d = (json.loads((ROOT / rel).read_text()) for rel in (lb.E04_STORAGE_EXTENSION_PATH, prep.STORAGE_DRAFT))
            for k in ('registry_extension', 'base_storage_authority', 'methods', 'schema_version'):
                self.assertEqual(a[k], d[k], k)
        # history scope (owner decision B1): no E04 S7 authority at this milestone (2caff0d); a later owner activation
        # must keep every DRAFT field unchanged
        self.assertNotEqual(subprocess.run(['git', 'cat-file', '-e', '2caff0d:' + lb.E04_S7_AUTHORITY_PATH],
                                           cwd=str(ROOT), stderr=subprocess.DEVNULL).returncode, 0)
        if (ROOT / lb.E04_S7_AUTHORITY_PATH).exists():
            a, d = (json.loads((ROOT / rel).read_text()) for rel in (lb.E04_S7_AUTHORITY_PATH, prep.S7_DRAFT))
            self.assertEqual((a['status'], a['approved']), ('OWNER_APPROVED', True))
            for k, v in d.items():
                if k not in ('status', 'approved', 'launch_preconditions'):
                    self.assertEqual(a[k], v, k)
        self.assertEqual(lb.registered_bank_methods(), lb.BANK_METHODS if (ROOT / lb.E04_STORAGE_EXTENSION_PATH).exists()
                         else lb.METHODS)

    def test_s7_draft_activates_only_when_owner_flips_status(self):
        """The DRAFT binds exactly what _load_e04_s7_authority checks: a copy with status OWNER_APPROVED is accepted,
        the DRAFT itself never is."""
        d = json.loads((ROOT / prep.S7_DRAFT).read_text())
        with tempfile.TemporaryDirectory() as t:
            for rel in (lb.E04_CONTRACT_PATH, lb.E04_DRAFT_PATH, lb.E04_FAMILY_CLOSURE_RECORD, prep.S7_DRAFT):
                Path(t, rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(ROOT / rel), str(Path(t, rel)))
            with self.assertRaises(lb.LearnedBankBlocked):
                lb.load_s7_authority('E04', t)
            d['status'], d['approved'] = 'OWNER_APPROVED', True
            Path(t, lb.E04_S7_AUTHORITY_PATH).write_text(json.dumps(d))
            a = lb.load_s7_authority('E04', t)
            self.assertEqual((a.mode, a.methods, a.seeds), ('S7_TOY', ('E04',), lb.SEEDS))

    def test_storage_draft_activates_only_when_owner_flips_status(self):
        d = json.loads((ROOT / prep.STORAGE_DRAFT).read_text())
        with tempfile.TemporaryDirectory() as t:
            Path(t, lb.STORAGE_AUTHORITY_PATH).parent.mkdir(parents=True)
            shutil.copyfile(str(ROOT / lb.STORAGE_AUTHORITY_PATH), str(Path(t, lb.STORAGE_AUTHORITY_PATH)))
            Path(t, lb.E04_STORAGE_EXTENSION_PATH).write_text(json.dumps(d))
            self.assertEqual(lb.registered_bank_methods(t), lb.METHODS)
            d['status'], d['approved'], d['activation_verification'] = 'OWNER_APPROVED', True, {'ok': True}
            Path(t, lb.E04_STORAGE_EXTENSION_PATH).write_text(json.dumps(d))
            self.assertEqual(lb.registered_bank_methods(t), lb.BANK_METHODS)

    def test_manifest_separates_executed_from_planned(self):
        r = json.loads((ROOT / prep.REPRO).read_text())
        self.assertEqual(r['training_reproduction']['status'], 'COMPLETED_AND_CLOSED')
        self.assertEqual(r['synthetic_fixture_qualification']['status'], 'EXECUTED_PASSED')
        self.assertEqual(r['future_s7_execution']['status'], 'PLANNED_NOT_EXECUTED')
        self.assertEqual(r['future_production_generation']['status'], 'PLANNED_NOT_EXECUTED')
        self.assertEqual(r['owner_ruling']['decisions'], lb.E04_APPROVED_DECISIONS)
        p = json.loads((ROOT / (prep.PLAN + '.json')).read_text())
        self.assertEqual((p['status'], p['population']['expected_images']), ('PREPARED_NOT_LAUNCHED', 1116))


if __name__ == '__main__':
    unittest.main()
