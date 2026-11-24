"""M8 learned-bank S7 toy acceptance records: every committed outputs/audit/LEARNED_BANK_S7_<M>.json re-derives
byte-identically from its embedded host collection, its SHA list matches, and the gates fail closed on tampered
collections (non-bitwise replay, checkpoint change, VAL/TEST atime change, non-zero exit). Needs numpy (learned_bank)."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
RECORDS = sorted(p for p in AUDIT.glob('LEARNED_BANK_S7_*.json') if p.stem.split('_')[-1] in ('E03', 'E05', 'E06b', 'E06c'))


def tool():
    spec = importlib.util.spec_from_file_location('s7_audit', ROOT / 'tools' / 'learned_bank_s7_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class S7Records(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = tool()
        cls.docs = {p.stem.split('_')[-1]: json.loads(p.read_text()) for p in RECORDS}

    def test_01_records_exist(self):
        self.assertTrue(self.docs, 'no S7 record committed')

    def test_02_rebuild_and_sha(self):
        for m, d in self.docs.items():
            with self.subTest(method=m):
                doc = self.mod.build(d['collected'], ROOT)
                self.assertEqual(json.loads(json.dumps(doc)), d)
                self.assertEqual(self.mod.report(doc), (AUDIT / ('LEARNED_BANK_S7_%s.md' % m)).read_text())
                for line in (AUDIT / ('LEARNED_BANK_S7_%s_SHA256.txt' % m)).read_text().splitlines():
                    digest, name = line.split('  ')
                    self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_03_passed_records_shape(self):
        for m, d in self.docs.items():
            with self.subTest(method=m):
                self.assertEqual((d['method'], d['seeds']), (m, [42, 1337, 2026]))
                self.assertEqual(d['s7_acceptance_passed'], d['failed_gates'] == [])
                if d['s7_acceptance_passed']:
                    self.assertTrue(all(v['bitwise_identical'] for v in d['outputs'].values()))

    def tampered(self, m, fn):
        c = copy.deepcopy(self.docs[m]['collected'])
        fn(c)
        return self.mod.build(c, ROOT)['failed_gates']

    def test_04_fail_closed(self):
        for m in self.docs:
            with self.subTest(method=m):
                def replay(c):
                    c['compare']['1337']['bitwise_identical'] = False
                    c['compare']['1337']['files_differing'] = ['x.png']
                self.assertIn('S.seed1337.bitwise_replay_A_B', self.tampered(m, replay))

                def ckpt(c):
                    f = c['checkpoints_after_run_B']['2026']['files']
                    next(iter(f.values()))['sha256'] = '0' * 64
                self.assertIn('S.seed2026.checkpoint_after_run_B', self.tampered(m, ckpt))

                def val_atime(c):
                    c['atime']['by_split']['VAL']['atime_changed'] = 1
                self.assertIn('F.atime_ledger', self.tampered(m, val_atime))

                def exit_b(c):
                    c['markers']['B']['exit_raw'] = '1\n'
                self.assertIn('R.runB.exit_zero', self.tampered(m, exit_b))

                def nonfinite(c):
                    r = c['runs']['A']['42']
                    r['manifest_rows'][5][r['manifest_fields'].index('raw_output_finite')] = False
                self.assertIn('S.seed42.rows_finite_rgb_uint8', self.tampered(m, nonfinite))

                def wrong_checkout(c):
                    c['git']['head'] = '18ba00e94813f005876c8d6235b0b848226b2c1d'   # the GPU training checkout
                self.assertIn('P.execution_checkout', self.tampered(m, wrong_checkout))

                def dirty_checkout(c):
                    c['git']['status_porcelain'] = ' M methods/pcgan/training_qualification.py'
                self.assertIn('P.execution_checkout', self.tampered(m, dirty_checkout))

                def wrong_checkpoint(c):
                    c['runs']['A']['42']['provenance']['adapter']['checkpoint_identity_sha256'] = '1' * 64
                self.assertIn('S.seed42.runA.checkpoint_before', self.tampered(m, wrong_checkpoint))


if __name__ == '__main__':
    unittest.main()
