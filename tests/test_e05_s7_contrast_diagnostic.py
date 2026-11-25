"""The committed E05 S7 low-contrast review re-derives byte-identically from its host collection, its status is the
disclosed (non-violation) outcome, and it introduces no gate into the frozen bank pipeline."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC'


def tool():
    spec = importlib.util.spec_from_file_location('e05diag', ROOT / 'tools' / 'e05_s7_contrast_diagnostic.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Diagnostic(unittest.TestCase):
    def test_rebuild_and_sha(self):
        mod = tool()
        c = json.loads((ROOT / mod.EVIDENCE).read_text())
        doc = mod.build(c, ROOT)
        self.assertEqual(json.loads(json.dumps(doc)), json.loads(Path(str(OUT) + '.json').read_text()))
        self.assertEqual(mod.report(doc), Path(str(OUT) + '.md').read_text())
        for line in Path(str(OUT) + '_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_status_and_scope(self):
        d = json.loads(Path(str(OUT) + '.json').read_text())
        self.assertEqual(d['status'], 'E05_LOW_CONTRAST_DISCLOSED_NO_CONTRACT_VIOLATION')
        self.assertEqual(d['contract_violations'], [])
        self.assertTrue(all(d['checks'].values()))
        c = json.loads((ROOT / tool().EVIDENCE).read_text())
        self.assertEqual(c['reader']['splits_allowlisted'], ['TRAIN'])
        self.assertEqual({k: v['changed'] for k, v in c['atime']['by_split'].items()},
                         {'TRAIN': 0, 'VAL': 0, 'TEST': 0})
        # the review is not wired into the bank writer / reader / production authority (no hidden quality gate)
        for rel in ('methods/common/learned_bank.py', 'tools/learned_bank.py'):
            self.assertNotIn('contrast', (ROOT / rel).read_text().lower(), rel)

    def test_fail_closed_on_conversion_mismatch(self):
        mod = tool()
        c = json.loads((ROOT / mod.EVIDENCE).read_text())
        c['methods']['E05']['1337']['rows'][7]['conversion_max_consistent'] = False
        doc = mod.build(c, ROOT)
        self.assertEqual(doc['status'], 'E05_CONTRACT_VIOLATION_FOUND_PRODUCTION_BLOCKED')
        self.assertEqual(doc['contract_violations'], ['e05_conversion_consistent_with_tensor2im'])


if __name__ == '__main__':
    unittest.main()
