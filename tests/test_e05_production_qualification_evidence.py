"""E05 production-runner qualification: evidence checks (stdlib + git; any host).

Validates the committed E05_PRODUCTION_RUNNER_QUALIFICATION evidence, re-derives it from the embedded read-only host
collection, and proves the production code in this checkout is byte-identical to the GPU-qualified candidate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
EVIDENCE = AUDIT / 'E05_PRODUCTION_RUNNER_QUALIFICATION.json'
SHA_FILE = AUDIT / 'E05_PRODUCTION_RUNNER_EVIDENCE_SHA256.txt'
CANDIDATE = 'eede86819b49ec8b809ac5ce80612a86c6c28c78'
QUALIFIED_CODE = ('methods/pcgan/production_io.py', 'methods/pcgan/production_runner.py', 'tools/run_e05.py',
                  'methods/pcgan/production_qualification.py')


def tool():
    spec = importlib.util.spec_from_file_location('e05_prodq_evidence',
                                                  ROOT / 'tools' / 'e05_production_qualification_evidence.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE05ProductionQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(EVIDENCE.read_text())
        cls.c = cls.doc['collected']
        cls.r = cls.c['result']

    def test_01_status_and_blockers(self):
        self.assertEqual(self.doc['status'], 'E05_PRODUCTION_RUNNER_QUALIFIED')
        self.assertEqual(self.doc['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in self.doc['gates']))
        self.assertTrue(all(self.doc['blockers_closed'].values()))
        self.assertEqual(self.doc['method']['fidelity'], 'CONTROLLED_ADAPTATION')
        self.assertEqual(self.doc['candidate_commit'], CANDIDATE)

    def test_02_code_equals_qualified_candidate(self):
        for rel in QUALIFIED_CODE:
            r = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', CANDIDATE, '--', rel])
            self.assertEqual(r.returncode, 0, rel + ' differs from the qualified candidate')
        files = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in QUALIFIED_CODE[:3]}
        self.assertEqual(self.r['identities']['production_files_sha256'], files)

    def test_03_qualification_not_scientific(self):
        self.assertEqual((self.r['qualification_seed'], self.r['experiment_seed']), (60901, None))
        self.assertFalse(self.c['scientific_e05_root_exists'])
        for key in ('train_run', 'probe_run'):
            self.assertIn('/qualification/e05_production/E05/', self.r[key]['run_dir'])
            self.assertNotIn('/runs/', self.r[key]['run_dir'])
        self.assertEqual(self.c['train_root']['run_summary.json']['completion_status'], 'interrupted')
        self.assertEqual(self.c['probe_root']['run_summary.json']['completion_status'], 'interrupted')
        self.assertFalse(any(self.doc['confirmations'][k] for k in self.doc['confirmations']))

    def test_04_accounting_and_terminal_semantics(self):
        recs = self.r['records']
        self.assertEqual([x['optimizer_applications']['total'] for x in recs], [2, 4, 6])
        self.assertEqual([(w['global_step'], w['checkpoint_type'], w['selected_for_final']) for w in
                          self.r['train_written']], [(1, 'periodic', False)])
        p = self.r['probe_written']
        self.assertEqual((p['path'], p['checkpoint_type'], p['selected_for_final'], p['selection_reason'],
                          p['actual_benchmark_iteration']),
                         ('checkpoints/iteration_4000.pth', 'selected', True, 'BASELINE_FINAL_STATE_V1', 3))

    def test_05_firewall_val_test_zero(self):
        acc = self.r['access']
        self.assertEqual((acc['VAL_image_accesses'], acc['TEST_image_accesses'], acc['VAL_metadata_accesses'],
                          acc['TEST_metadata_accesses'], acc['denied_events']), (0, 0, 0, 0, 0))
        for cls, v in self.c['atime']['by_class'].items():
            if cls.split('|')[1] in ('TEST', 'VAL'):
                self.assertEqual((v['atime_changed'], v['sensitive_whole_window']), (0, v['files']), cls)

    def test_06_driver_disclosed(self):
        d = self.doc['gpu_driver']
        self.assertEqual((d['lock_driver_version'], d['live_driver_version'], d['driver_version_changed']),
                         ('595.84', '580.178.04', True))
        self.assertIn('580.178.04', (AUDIT / 'E05_PRODUCTION_RUNNER_QUALIFICATION.md').read_text())

    def test_07_rebuild_reproduces_evidence(self):
        self.assertEqual(json.loads(json.dumps(tool().build(self.c, ROOT))), self.doc)

    def test_08_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/E05_PRODUCTION_RUNNER_QUALIFICATION.json',
                                         'outputs/audit/E05_PRODUCTION_RUNNER_QUALIFICATION.md'])


if __name__ == '__main__':
    unittest.main()
