"""E06b/E06c retrospective runtime-driver provenance (2026-10-30): checks the committed disclosure, re-derives the
evidence record from its embedded host collection, and probes that contradicting evidence fails closed.

Stdlib only, any host.
"""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
RECORD = AUDIT / 'E06_RUNTIME_DRIVER_PROVENANCE.json'
DISCLOSURE = ROOT / 'configs' / 'amendments' / 'e06_runtime_driver_retrospective_disclosure.yaml'
DRAFT = ROOT / 'configs' / 'amendments' / 'e06_runtime_driver_acceptance.DRAFT.yaml'


def tool():
    spec = importlib.util.spec_from_file_location('e06_driver', ROOT / 'tools' / 'e06_runtime_driver_provenance_record.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class E06RuntimeDriverProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(RECORD.read_text())
        cls.disclosure = json.loads(DISCLOSURE.read_text())
        cls.mod = tool()

    def test_01_record_established(self):
        self.assertEqual(self.doc['status'], 'E06_HISTORICAL_DRIVER_ESTABLISHED')
        self.assertEqual(self.doc['failed_checks'], [])
        self.assertEqual((self.doc['lock_driver'], self.doc['historical_driver']), ('595.84', '580.178.04'))
        runs = self.doc['checks']['every_e06_run_bound_to_580_boot']['runs']
        self.assertEqual(len(runs), 6)
        self.assertTrue(all(r['historical_driver'] == '580.178.04' and r['kernel'] == '7.0.0-38-generic' for r in runs))

    def test_02_record_rederives(self):
        doc = self.mod.build(ROOT, self.doc['collected'])
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(self.mod.report(doc), (AUDIT / 'E06_RUNTIME_DRIVER_PROVENANCE.md').read_text())
        for line in (AUDIT / 'E06_RUNTIME_DRIVER_PROVENANCE_SHA256.txt').read_text().splitlines():
            digest, path = line.split('  ')
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)

    def test_03_disclosure_scope(self):
        d = self.disclosure
        self.assertEqual(d['status'], 'OWNER_AUTHORIZED_RETROSPECTIVE_DISCLOSURE_RECORDED')
        self.assertEqual((d['lock_driver_version'], d['historical_driver_version']), ('595.84', '580.178.04'))
        s = d['statements']
        self.assertTrue(s['disclosed_retrospectively'] and s['results_preserved_without_retraining_or_cherry_picking'])
        for k in ('lock_edited', 'scientific_source_training_data_checkpoints_run_outputs_modified',
                  'bitwise_runtime_equivalence_established', 'runs_presented_as_executed_under_locked_driver',
                  'e03_e05_driver_rulings_reused_as_prior_e06_approval'):
            self.assertIs(s[k], False, k)
        self.assertEqual(d['runtime_compatibility_acceptance']['status'], 'PENDING_OWNER_DECISION')
        draft = json.loads(DRAFT.read_text())
        self.assertIs(draft['approved'], False)
        self.assertFalse((ROOT / 'configs' / 'amendments' / 'e06_runtime_driver_acceptance.yaml').exists())

    def test_04_affected_runs_equal_closures(self):
        closed = set()
        for tag in ('E06C', 'E06B'):
            with open(AUDIT / ('%s_THREE_SEED_FINAL_CHECKPOINTS.csv' % tag)) as f:
                for r in csv.DictReader(f):
                    closed.add((tag.replace('E06C', 'E06c').replace('E06B', 'E06b'), int(r['seed']), r['run_id'],
                                r['seed_closure_commit']))
        listed = {(r['method_id'], r['seed'], r['run_id'], r['seed_closure']) for r in self.disclosure['affected_runs']}
        self.assertEqual(listed, closed)
        self.assertEqual({(m, s, i) for m, s, i, _ in listed}, set(self.mod.E06_RUNS))

    def test_05_lock_untouched(self):
        raw = (ROOT / 'environments' / 'e06c.lock.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95')

    def probe(self, mutate, failing):
        col = copy.deepcopy(self.doc['collected'])
        mutate(col)
        doc = self.mod.build(ROOT, col)
        self.assertEqual(doc['status'], 'E06_HISTORICAL_DRIVER_NOT_ESTABLISHED')
        self.assertIn(failing, doc['failed_checks'])

    def test_06_later_595_install_fails_closed(self):
        self.probe(lambda c: c['apt_nvidia_transactions'].append(
            {'start': '2026-10-15  10:00:00', 'end': '2026-10-15  10:00:10', 'file': 'history.log',
             'events': [['Install', 'nvidia-driver-595-open:amd64', '595.84-0ubuntu0.24.04.1']]}),
            'no_driver_change_after_switch')

    def test_07_other_kernel_fails_closed(self):
        self.probe(lambda c: c.__setitem__('wtmp_boots', c['wtmp_boots'].replace(
            '7.0.0-38-generic 2026-10-20T15:56:42', '7.0.0-31-generic 2026-10-20T15:56:42')), 'every_e06_run_bound_to_580_boot')

    def test_08_recorded_driver_string_fails_closed(self):
        self.probe(lambda c: c['e06_runs'][0].__setitem__('driver_strings_recorded', ['run_manifest.json']),
                   'every_e06_run_bound_to_580_boot')

    def test_09_missing_switch_transaction_fails_closed(self):
        self.probe(lambda c: c.__setitem__('apt_nvidia_transactions', [
            t for t in c['apt_nvidia_transactions'] if t['start'] != '2026-10-03  11:27:17']),
            'switch_transaction_installs_580_removes_595')


if __name__ == '__main__':
    unittest.main()
