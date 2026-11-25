"""Seven previous-work bank families: the committed readiness record re-derives from the committed host verification
and audits, and fails closed when any bank's host evidence is missing or altered (framework-free)."""
import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('pwbr', str(ROOT / 'tools/previous_work_bank_readiness.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load()


class Readiness(unittest.TestCase):
    def setUp(self):
        if not (ROOT / (T.OUT + '.json')).exists():
            self.skipTest('seven-baseline readiness record not yet built')

    def test_rebuild_identical(self):
        doc = T.build(ROOT)
        self.assertEqual(json.loads(json.dumps(doc)), json.loads((ROOT / (T.OUT + '.json')).read_text()))
        self.assertEqual(T.report(doc), (ROOT / (T.OUT + '.md')).read_text())
        for line in (ROOT / (T.OUT + '_SHA256.txt')).read_text().splitlines():
            digest, name = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(doc['expected_images'], 21 * 8838)

    def _tampered(self, fn):
        v = json.loads((ROOT / T.VERIFIED).read_text())
        fn(v)
        orig = T.VERIFIED
        p = ROOT / 'outputs/audit/PREVIOUS_WORK_BANK_READINESS/_tampered_test.json'
        try:
            p.write_text(json.dumps(v))
            T.VERIFIED = str(p.relative_to(ROOT))
            return T.build(ROOT)
        finally:
            T.VERIFIED = orig
            p.unlink()

    def test_fail_closed(self):
        for name, fn in (('manifest changed', lambda v: v['banks']['E04|42'].update(manifest_sha256='0' * 64)),
                         ('reader failed', lambda v: v['banks']['E02|1337']['reader'].update(ok=False)),
                         ('png missing', lambda v: v['banks']['E05|2026'].update(png_files=8837)),
                         ('edit maps missing', lambda v: v['banks']['E02|42'].update(edit_maps=0)),
                         ('bank error', lambda v: v['banks']['E06c|42'].update(error='FileNotFoundError')),
                         ('bank absent', lambda v: v['banks'].pop('E03|2026')),
                         ('storage', lambda v: v['storage'].update(unexpected_entries=['E07']))):
            with self.subTest(case=name):
                d = self._tampered(fn)
                self.assertNotEqual(d['status'], 'ALL_SEVEN_PREVIOUS_WORK_BANK_FAMILIES_COMPLETED_AND_AUDITED')


if __name__ == '__main__':
    unittest.main()
