"""BASELINE-EXEC-P0: GPAT deferral backlog + previous-work execution roadmap evidence checks.

Stdlib + the GPAT runtime contract, any host. Planning-only milestone: these tests validate the committed records
against repository authority (frozen configs + snapshots, method_status.csv, A1 schedule, M7D1 closure evidence) and
the SHA-256 manifest. Nothing is trained or launched.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
BACKLOG = AUDIT / 'M7D1_GPAT_DEFERRED_FIX_BACKLOG.json'
BACKLOG_MD = AUDIT / 'M7D1_GPAT_DEFERRED_FIX_BACKLOG.md'
ROADMAP = AUDIT / 'PREVIOUS_WORK_EXECUTION_ROADMAP.json'
ROADMAP_MD = AUDIT / 'PREVIOUS_WORK_EXECUTION_ROADMAP.md'
SHA_FILE = AUDIT / 'BASELINE_EXEC_P0_EVIDENCE_SHA256.txt'
AUTHORITY = '03217143a766f0bbeae6fcf90c45c4226881a0fc'
ACTIVE = {'E01', 'E02', 'E03', 'E04', 'E05', 'E06b', 'E06c'}
EXCLUDED = {'E06a', 'E07a', 'E07b', 'E07c'}
SEEDS = [42, 1337, 2026]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_contract():
    spec = importlib.util.spec_from_file_location('gpat_runtime_contract_p0', ROOT / 'methods/gpat/runtime_contract.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BaselineExecP0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backlog = json.loads(BACKLOG.read_text())
        cls.roadmap = json.loads(ROADMAP.read_text())
        with (AUDIT / 'method_status.csv').open(newline='') as fh:
            cls.status = {r['experiment_id']: r for r in csv.DictReader(fh)}

    # ------------------------------------------------------------------ GPAT backlog
    def test_01_gpat_deferred(self):
        b = self.backlog
        self.assertEqual(b['authority_commit'], AUTHORITY)
        self.assertEqual(b['gpat_status'], 'DEFERRED_BY_OWNER_PRIORITY')
        self.assertEqual(b['owner_direction']['gpat_development'], 'PAUSED')

    def test_02_seed42_selection_matches_n6c(self):
        a = self.backlog['A_healthy_seed42_a1']
        sel = json.loads((AUDIT / 'M7D1_A1_E08_SEED42_SELECTION.json').read_text())
        self.assertIn(a['selected_sha256'], json.dumps(sel))
        self.assertEqual(a['selected_sha256'], '01fd718ae3672491f846f41609ad2de06c8c5b3b10f9717fc1d8014e100b1bb6')
        self.assertEqual((a['best_ema_seed42_epoch'], a['completed_epochs'], a['completed_updates']), (28, 60, 66300))
        self.assertEqual(a['n6c_selection_commit'], '13a0252fd097b7552f0875f346f83c7b676cef0f')
        self.assertFalse(a['test_used_for_selection'])

    def test_03_a1_schedule_matches_runtime_contract(self):
        rc = runtime_contract()
        for u in (1, 5525, 5526, 6630, 6631, 16575):
            self.assertEqual(rc.lambda_adv_a1(u), 0.0 if u <= 5526 else 0.05)
        self.assertEqual(rc.lambda_adv_a1(6078), 0.05 * ((6078 - 5526) / 1104))
        self.assertEqual((rc.lambda_adv_a1(16576), rc.lambda_adv_a1(66300)), (0.10, 0.10))
        self.assertEqual(len(self.backlog['C_m7d1_a1']['schedule']), 4)

    def test_04_seed1337_matches_closure(self):
        f2 = self.backlog['D_failure_mode_F2_seed1337_stage1_collapse']
        closure = json.loads((AUDIT / 'M7D1_A1_E08_SEED1337_STOPPED_COLLAPSE.json').read_text())
        self.assertEqual(f2['run_id'], closure['run_id'])
        self.assertEqual(f2['final_status'], closure['final_run_status'])
        self.assertEqual(f2['onset']['approximate_update'], closure['onset']['grad_spike_update'])
        self.assertEqual(f2['onset']['lambda_adv'], 0.0)
        sig = f2['historical_collapse_scalar_signature']
        self.assertEqual((sig['first_update'], sig['single_miss']), (2174, 2175))
        self.assertEqual(f2['last_complete_update'], 15107)
        self.assertEqual(f2['root_cause'], 'NOT_INFERRED')

    def test_05_rc1_and_no_approved_fix(self):
        rc1 = self.backlog['E_future_root_cause_milestone']
        self.assertEqual(rc1['name'], 'M7D1-SEED1337-RC1')
        self.assertEqual(rc1['replay']['snapshots'],
                         [1105, 2100, 2150, 2160, 2166, 2167, 2168, 2170, 2171, 2174, 2210])
        self.assertEqual(self.backlog['H_candidate_future_fixes_only_after_causality']['approved'], [])
        self.assertIn('NO BLIND HYPERPARAMETER TUNING', BACKLOG_MD.read_text())

    # ------------------------------------------------------------------ roadmap
    def test_06_scope_partition(self):
        active = {r['experiment_id'] for r in self.roadmap['active_previous_work']}
        excluded = {r['experiment_id'] for r in self.roadmap['excluded_or_superseded']}
        self.assertEqual(active, ACTIVE)
        self.assertEqual(excluded, EXCLUDED)
        for e in active:
            self.assertEqual(self.status[e]['scope_status'], 'ACTIVE_M6_BASELINE')
        for e in excluded:
            self.assertNotEqual(self.status[e]['scope_status'], 'ACTIVE_M6_BASELINE')

    def test_07_config_hashes_snapshots_and_fidelity(self):
        for r in self.roadmap['active_previous_work']:
            c = r['frozen_config']
            self.assertEqual(c['sha256'], sha(ROOT / c['path']), r['experiment_id'])
            self.assertEqual(sha(ROOT / c['snapshot']), c['sha256'], r['experiment_id'])
            self.assertTrue(r['fidelity'].startswith(self.status[r['experiment_id']]['fidelity_class']), r['experiment_id'])
            self.assertEqual(r['required_seeds'], SEEDS)

    def test_08_lifecycle_labels(self):
        rows = {r['experiment_id']: r for r in self.roadmap['active_previous_work']}
        for e in ('E01', 'E02'):
            self.assertTrue(rows[e]['lifecycle'].startswith('DIRECT_SYNTHETIC_TRANSFORMATION_BANK_GENERATION'))
            self.assertEqual(rows[e]['checkpoint_rule'], 'NOT_APPLICABLE_NO_TRAINED_GENERATOR')
        for e in ('E03', 'E04', 'E05', 'E06b', 'E06c'):
            self.assertTrue(rows[e]['lifecycle'].startswith('LEARNED_GENERATOR_TRAINING'))

    def test_09_queue_and_next_method(self):
        q = self.roadmap['recommended_execution_queue']
        self.assertEqual([x['rank'] for x in q], list(range(1, 8)))
        self.assertEqual({x['experiment_id'] for x in q}, ACTIVE)
        self.assertEqual(q[0]['experiment_id'], 'E06c')
        nxt = self.roadmap['next_method']
        self.assertEqual((nxt['experiment_id'], nxt['status']), ('E06c', 'STAGED_NOT_LAUNCHED'))
        self.assertEqual(nxt['seeds'], SEEDS)
        self.assertIn('tools/run_e06c.py', nxt['command_template'])
        self.assertTrue(nxt['checkpoint_rule'].startswith('OFFICIAL_GENERATOR_EPOCH_200'))
        for s in SEEDS:
            self.assertTrue(nxt['run_roots'][str(s)].endswith(f'/runs/m6/E06c/seed_{s}'))
        self.assertTrue((ROOT / 'tools/run_e06c.py').is_file())

    def test_10_sha_manifest(self):
        lines = [l.split('  ', 1) for l in SHA_FILE.read_text().splitlines() if l.strip()]
        paths = {p for _, p in lines}
        self.assertEqual(paths, {str(p.relative_to(ROOT)) for p in (BACKLOG, BACKLOG_MD, ROADMAP, ROADMAP_MD)})
        for digest, p in lines:
            self.assertEqual(digest, sha(ROOT / p), p)


if __name__ == '__main__':
    unittest.main()
