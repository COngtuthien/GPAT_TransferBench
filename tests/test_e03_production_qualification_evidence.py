"""E03 STDN production qualification: evidence checks (stdlib + numpy + git; any host).

Validates the committed E03_PRODUCTION_RUNNER_QUALIFICATION evidence, re-derives it from the embedded collection, and
proves the production code in this checkout is byte-identical to the GPU-qualified candidate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
EVIDENCE = AUDIT / 'E03_PRODUCTION_RUNNER_QUALIFICATION.json'
SHA_FILE = AUDIT / 'E03_PRODUCTION_RUNNER_EVIDENCE_SHA256.txt'
CANDIDATE = 'c07ecdd72c227c7b8fcace15e12dc23b28e9c8c8'
QUALIFIED_CODE = ('methods/stdn/production_io.py', 'methods/stdn/production_runner.py', 'methods/stdn/feed_pack.py',
                  'tools/run_e03.py', 'methods/stdn/production_qualification.py', 'tools/e03_feed_pack.py')


def tool():
    spec = importlib.util.spec_from_file_location('e03_prodq_evidence',
                                                  ROOT / 'tools' / 'e03_production_qualification_evidence.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE03ProductionQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(EVIDENCE.read_text())
        cls.c = cls.doc['collected']

    def test_01_status(self):
        self.assertEqual(self.doc['status'], 'E03_PRODUCTION_RUNNER_QUALIFIED')
        self.assertEqual(self.doc['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in self.doc['gates']))
        self.assertEqual(self.doc['method']['fidelity'], 'FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_'
                                                         'ADAPTATION_AND_RUNTIME_COMPATIBILITY')
        self.assertEqual(self.doc['candidate_commit'], CANDIDATE)

    def test_02_code_equals_qualified_candidate(self):
        for rel in QUALIFIED_CODE:
            r = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', CANDIDATE, '--', rel])
            self.assertEqual(r.returncode, 0, rel + ' differs from the qualified candidate')

    def test_03_qualification_only(self):
        for c in self.c['cases'].values():
            self.assertEqual((c['qualification_seed'], c['experiment_seed'], c['scientific_execution']), (60903, None, False))
            self.assertIn('/qualification/e03_production/E03/', c['run_dir'])
        self.assertNotIn('E03', self.c['host']['runs_m6'])

    def test_04_val_non_influence(self):
        g = {x['gate']: x['pass'] for x in self.doc['gates']}
        for name in ('G.in_process_all_variables_byte_identical', 'G.in_process_counters_unchanged',
                     'G.train_parser_untouched_by_val', 'G.val_parser_advanced_only',
                     'G.subsequent_train_decisions_identical', 'G.val_never_changes_global_step'):
            self.assertTrue(g[name], name)
        self.assertEqual(self.c['cases']['val_a']['val_variable_diff'], {'compared': 747, 'changed': [], 'missing': []})

    def test_05_schedule_and_checkpoints(self):
        a = self.c['cases']['val_a']
        self.assertEqual((a['final']['global_step'], a['final']['applications']), (314, {'G': 209, 'D': 105}))
        self.assertEqual(a['checkpoints'][0]['epoch'], 1)
        self.assertEqual(a['checkpoints'][0]['global_step'], 302)
        probe = self.c['cases']['terminal_probe']['checkpoints'][0]
        self.assertEqual((probe['epoch'], probe['selected_for_final'], probe['terminal_semantics_probe']), (50, True, True))

    def test_06_test_zero_and_firewall(self):
        test = {k: v for k, v in self.c['atime']['by_class'].items() if k.endswith('|TEST')}
        self.assertEqual(sum(v['files'] for v in test.values()), 3027)
        self.assertTrue(all(v['changed'] == 0 and v['sensitive'] == v['files'] for v in test.values()))
        for c in self.c['cases'].values():
            self.assertEqual(c['access']['denied'], [])

    def test_07_driver_disclosed(self):
        self.assertEqual(self.doc['owner_policy']['driver'], {'lock': '595.84', 'observed': '580.178.04',
                                                              'policy': 'OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_'
                                                                        'COMPATIBILITY'})
        self.assertIn('580.178.04', (AUDIT / 'E03_PRODUCTION_RUNNER_QUALIFICATION.md').read_text())

    def test_08_rebuild_reproduces_evidence(self):
        self.assertEqual(json.loads(json.dumps(tool().build(self.c, ROOT))), self.doc)

    def test_09_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(sorted(names), ['outputs/audit/E03_PRODUCTION_RUNNER_QUALIFICATION.json',
                                         'outputs/audit/E03_PRODUCTION_RUNNER_QUALIFICATION.md'])


if __name__ == '__main__':
    unittest.main()
