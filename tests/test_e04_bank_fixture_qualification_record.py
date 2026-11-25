"""The committed E04 fixture-qualification record rebuilds byte-identically and fails closed on tampered evidence."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import e04_bank_fixture_record as rec  # noqa: E402


class Record(unittest.TestCase):
    def test_rebuild_byte_identical(self):
        if not (ROOT / (rec.OUT + '.json')).exists():
            self.skipTest('record not yet built')
        with tempfile.TemporaryDirectory() as t:
            doc = rec.build(ROOT, out_root=t)
            self.assertEqual(doc['status'], rec.STATUS_OK)
            for ext in ('.json', '.md', '_SHA256.txt'):
                self.assertEqual(Path(t, rec.OUT + ext).read_bytes(), (ROOT / (rec.OUT + ext)).read_bytes(), ext)
            self.assertEqual(doc['real_device_qualification']['status'], 'PENDING_E04_S7')

    def test_fail_closed(self):
        if not (ROOT / rec.DIR / 'host_evidence.json').exists():
            self.skipTest('evidence not yet committed')
        with tempfile.TemporaryDirectory() as t:
            for rel in [rec.CONTRACT] + ['%s/%s' % (rec.DIR, n) for n in ('host_evidence.json', 'host_stdout.txt',
                                                                         'host_stderr.txt', 'laptop_tests.txt')]:
                Path(t, rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(ROOT / rel), str(Path(t, rel)))
            ev = json.loads(Path(t, rec.DIR, 'host_evidence.json').read_text())
            ev['failed_gates'] = ['G.bn_mode_is_the_only_difference']
            ev['gpu_bitwise_reproducibility_proven'] = True
            Path(t, rec.DIR, 'host_evidence.json').write_text(json.dumps(ev))
            doc = rec.build(t)
            self.assertIn('host_all_gates_passed', doc['failed_gates'])
            self.assertIn('no_gpu_claim', doc['failed_gates'])
            self.assertEqual(doc['status'], 'E04_BANK_ADAPTER_SYNTHETIC_QUALIFICATION_FAILED')


if __name__ == '__main__':
    unittest.main()
