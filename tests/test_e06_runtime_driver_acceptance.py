"""E06b/E06c retrospective runtime-driver acceptance (owner approval 2026-10-30): the approved record is the DRAFT
committed at 09b4524 with its terms approved as written, limited to the six completed runs and to the driver version;
the draft, the disclosure, its addendum, the evidence, the lock and the E06 closures are unchanged; nothing future is
authorized. Stdlib only.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRAFT_COMMIT = '09b452415988f03bae074a134890c6438639ff02'
APPROVED = 'configs/amendments/e06_runtime_driver_acceptance.yaml'
DRAFT = 'configs/amendments/e06_runtime_driver_acceptance.DRAFT.yaml'
DISCLOSURE = 'configs/amendments/e06_runtime_driver_retrospective_disclosure.yaml'
DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E06_Runtime_Driver_Acceptance_Addendum.md'
LOCK_SHA256 = '91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95'
CLOSURE_SHA = ('E06C_THREE_SEED_EVIDENCE_SHA256.txt', 'E06B_THREE_SEED_EVIDENCE_SHA256.txt')


def at_draft(path):
    return subprocess.run(['git', 'show', '%s:%s' % (DRAFT_COMMIT, path)], cwd=ROOT, capture_output=True,
                          check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


A = json.loads((ROOT / APPROVED).read_text())
D = json.loads(at_draft(DRAFT))


class E06RuntimeDriverAcceptance(unittest.TestCase):
    def test_01_approved_ruling(self):
        self.assertEqual((A['status'], A['approved']), ('OWNER_APPROVED', True))
        self.assertEqual(A['ruling'], D['proposed_ruling'])
        self.assertEqual(A['draft_terms_approved'], D['proposed_terms'])
        self.assertEqual((A['lock_driver_version'], A['historical_driver_version']), ('595.84', '580.178.04'))
        self.assertEqual(A['schema_version'], D['schema_version'])
        self.assertEqual(A['document'], DOC)
        self.assertIn('`OWNER_APPROVED`', (ROOT / DOC).read_text())

    def test_02_scope_is_the_six_disclosed_runs(self):
        disclosure = json.loads((ROOT / DISCLOSURE).read_text())
        self.assertEqual(A['affected_runs'], disclosure['affected_runs'])
        self.assertEqual(len(A['affected_runs']), 6)
        closed = set()
        for tag in ('E06C', 'E06B'):
            with open(ROOT / 'outputs' / 'audit' / ('%s_THREE_SEED_FINAL_CHECKPOINTS.csv' % tag)) as f:
                closed |= {(tag[:3] + tag[3].lower(), int(r['seed']), r['run_id']) for r in csv.DictReader(f)}
        self.assertEqual({(r['method_id'], r['seed'], r['run_id']) for r in A['affected_runs']}, closed)
        self.assertNotIn('d3c0ddd5dc82a4a1', {r['run_id'] for r in A['affected_runs']})

    def test_03_narrow_statements(self):
        s = A['statements']
        self.assertTrue(s['retrospective'] and s['results_preserved_without_retraining_or_cherry_picking'])
        for k in ('lock_edited', 'scientific_source_training_data_checkpoints_run_outputs_modified',
                  'bitwise_runtime_equivalence_established', 'runs_presented_as_executed_under_locked_driver',
                  'e03_e05_driver_rulings_reused_as_prior_e06_approval', 'authorizes_future_e06_execution',
                  'waives_future_runtime_qualification', 'authorizes_performance_driven_rerun'):
            self.assertIs(s[k], False, k)
        self.assertEqual(len(A['owner_scope_limits']), 9)

    def test_04_approved_draft_pins(self):
        p = A['approved_draft']
        self.assertEqual(p['commit'], DRAFT_COMMIT)
        for key, path in (('draft_sha256', 'draft'), ('disclosure_sha256', 'disclosure'),
                          ('disclosure_document_sha256', 'disclosure_document'), ('evidence_sha256', 'evidence')):
            self.assertEqual(p[key], sha(at_draft(p[path])), path)
            self.assertEqual(sha((ROOT / p[path]).read_bytes()), p[key], path)
        self.assertEqual(json.loads((ROOT / p['evidence']).read_text())['status'], p['evidence_status'])

    def test_05_lock_and_closures_unchanged(self):
        self.assertEqual(sha((ROOT / 'environments' / 'e06c.lock.json').read_bytes()), LOCK_SHA256)
        self.assertEqual(A['lock']['sha256'], LOCK_SHA256)
        for name in CLOSURE_SHA:
            path = 'outputs/audit/' + name
            self.assertEqual((ROOT / path).read_bytes(), at_draft(path), path)


if __name__ == '__main__':
    unittest.main()
