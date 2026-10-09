"""M7D1-N4: GPAT-B0 / E08 / seed 42 identity-collapse repair qualification (DIAGNOSTIC_ONLY) -- focused checks.

Stdlib on any host: exact R1/R2 schedules, curriculum isolation (only lambda_adv changes), the predeclared collapse
criterion, and the committed evidence (R0 exact-control parity, deterministic restart from the pre-u5526 snapshot,
TRAIN order, VAL/TEST firewall, byte-for-byte re-derivation of the derived files). No production config / method /
spec file differs from the authority commit in the commit that added this test (or the worktree, before commit).
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
PREFIX = 'M7D1_N4_SEED42_REPAIR_'
COLLECTED = AUDIT / f'{PREFIX}COLLECTED.json'
DERIVED = ('QUALIFICATION.json', 'SCREEN.csv', 'LONG.csv', 'GRADIENTS.csv')
AUTHORITY = 'd88563a46757d171a5e2af359eebc4fc18c305eb'
PROTECTED = ('configs', 'methods', 'docs', 'gpatbench', 'environments', 'outputs/audit/STAGE_STATE.json')


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_n4', ROOT / 'tools' / 'm7d1_n4_repair_qualification.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frozen_curriculum(u):
    """Mirror of runtime_contract.curriculum for stage 2 (the tool must leave everything but lambda_adv alone)."""
    if u < 5526:
        return {'stage': 1, 's_hf': 0.02 + 0.03 * (u - 1) / 5524, 'lambda_adv': 0.0, 'lambda_con': 0.5,
                'lambda_spec': 0.25}
    if u < 16576:
        return {'stage': 2, 's_hf': 0.10, 'lambda_adv': 0.05, 'lambda_con': 1.0, 'lambda_spec': 0.5}
    return {'stage': 3, 's_hf': 0.15, 'lambda_adv': 0.10, 'lambda_con': 1.0, 'lambda_spec': 0.5}


class ScheduleM7D1N4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tool()

    def _ramp(self, b, end):
        t = self.t
        vals = [t.lambda_adv(b, u) for u in range(5526, 16576)]
        self.assertEqual(t.lambda_adv(b, 5526), 0.0)
        self.assertEqual(t.lambda_adv(b, end), 0.05)              # exact in binary floating point
        self.assertTrue(all(t.lambda_adv(b, u) == 0.05 for u in range(end, 16576)))
        self.assertTrue(all(b2 >= a for a, b2 in zip(vals, vals[1:])), 'monotone non-decreasing')
        step = 0.05 / (end - 5526)
        for u in range(5527, end + 1):                             # continuity: constant increment, no jump
            self.assertAlmostEqual(t.lambda_adv(b, u) - t.lambda_adv(b, u - 1), step, delta=1e-15)
        for u in range(5526, end + 1):                             # equals the instruction's formula as reals
            self.assertAlmostEqual(t.lambda_adv(b, u), 0.05 * (u - 5526) / (end - 5526), delta=1e-17)
        for u in (5525, 16576, 1, 66300):
            with self.assertRaises(ValueError):
                t.lambda_adv(b, u)

    def test_01_r1_schedule(self):
        self._ramp('R1', 6630)
        self.assertEqual(self.t.BRANCHES['R1']['ramp_end'], 6630)

    def test_02_r2_schedule(self):
        self._ramp('R2', 11050)
        self.assertEqual(self.t.lambda_adv('R2', 8288), 0.025)                    # midpoint exact

    def test_03_control_and_fallbacks(self):
        t = self.t
        for u in (5526, 6630, 11050, 16575):
            self.assertEqual(t.lambda_adv('R0', u), 0.05)
            self.assertEqual(t.lambda_adv('R3', u), 0.01)
            self.assertEqual(t.lambda_adv('R4', u), 0.02)
        self.assertEqual(t.R5_STATUS['status'], 'R5_UNSUPPORTED_EXACT_STATE')
        self.assertEqual(sorted(t.BRANCHES), ['R0', 'R1', 'R2', 'R3', 'R4'])

    def test_04_only_lambda_adv_changes(self):
        t = self.t
        for b in t.BRANCHES:
            cur = t.branch_curriculum(b, frozen_curriculum)
            for u in (1, 2000, 5525, 5526, 5527, 6000, 6630, 6631, 11050, 11051, 16575):
                got, ref = cur(u), frozen_curriculum(u)
                self.assertEqual({k: v for k, v in got.items() if k != 'lambda_adv'},
                                 {k: v for k, v in ref.items() if k != 'lambda_adv'}, (b, u))
                self.assertEqual(got['lambda_adv'], ref['lambda_adv'] if u < 5526 else t.lambda_adv(b, u), (b, u))
            if b == 'R0':
                self.assertEqual(cur(16576), frozen_curriculum(16576))
            else:
                with self.assertRaises(AssertionError):
                    cur(16576)                                     # never leaves the frozen stage-2 window

    def test_05_collapse_criterion(self):
        ev = self.t.collapse_event
        self.assertEqual(self.t.COLLAPSE_RULE['threshold'], 0.01)
        self.assertEqual(self.t.COLLAPSE_RULE['consecutive_complete_updates'], 10)
        self.assertIsNone(ev([(u, 0.001) for u in range(100, 109)]))                    # 9 updates
        self.assertEqual(ev([(u, 0.001) for u in range(100, 110)]), {'onset_update': 100, 'event_update': 109})
        self.assertIsNone(ev([(u, 0.01) for u in range(100, 130)]))                     # strict '<'
        rows = [(u, 0.001) for u in range(100, 105)] + [(105, 0.5)] + [(u, 0.0) for u in range(106, 116)]
        self.assertEqual(ev(rows), {'onset_update': 106, 'event_update': 115})
        gap = [(u, 0.0) for u in range(100, 105)] + [(u, 0.0) for u in range(106, 111)]  # not consecutive
        self.assertIsNone(ev(gap))

    def test_06_firewall_and_roots(self):
        t = self.t
        self.assertEqual(t.N4_PARTS, ('diagnostics', 'm7', 'M7D1_N4_seed42'))
        self.assertEqual(t.n3.DIAG_PARTS, t.N4_PARTS)              # every N3 helper writes under N4 only
        self.assertNotEqual(t.N3_PARTS, t.N4_PARTS)
        src = (ROOT / 'tools' / 'm7d1_n4_repair_qualification.py').read_text()
        self.assertIn("forbidden = [rt + '/runs', str(n3_root()), str(ROOT / 'configs'), str(ROOT / 'docs')]", src)
        for token in ('val_split', "'val'", "'test'", 'VAL_ROOT', 'TEST_ROOT'):
            self.assertNotIn(token, src)

    def test_07_no_protected_file_changed(self):
        added = subprocess.run(['git', '-C', str(ROOT), 'log', '--diff-filter=A', '-1', '--format=%H', '--',
                                'tests/test_m7d1_n4_repair_qualification.py'], capture_output=True, text=True).stdout
        cmd = ['git', '-C', str(ROOT), 'diff', '--name-only', AUTHORITY]
        if added.strip():
            cmd.append(added.strip())
        changed = subprocess.run(cmd + ['--', *PROTECTED], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(changed, [])


@unittest.skipUnless(COLLECTED.exists(), 'N4 collected evidence not present')
class EvidenceM7D1N4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tool()
        cls.c = json.loads(COLLECTED.read_text())
        cls.q = json.loads((AUDIT / f'{PREFIX}QUALIFICATION.json').read_text())

    def test_10_identity(self):
        c = self.c
        self.assertEqual((c['method'], c['experiment'], c['seed'], c['scientific_run_id'], c['authority_commit']),
                         ('GPAT-B0', 'E08', 42, '7b799fbd6d0426da', AUTHORITY))
        self.assertIn('DIAGNOSTIC_REPAIR_VARIANT', c['labels'])
        self.assertIn('NOT_AN_APPROVED_PROTOCOL_CHANGE', c['labels'])
        self.assertEqual(c['base_snapshot']['sha256'], self.t.BASE_SHA256)
        self.assertEqual(c['collapse_rule'], self.t.COLLAPSE_RULE)
        self.assertEqual(c['branch_definitions'], self.t.BRANCHES)

    def test_11_derived_files_rederive(self):
        out = self.t.finalize(COLLECTED)
        for name in DERIVED:
            self.assertEqual((AUDIT / f'{PREFIX}{name}').read_text(), out[name], name)

    def test_12_r0_exact_control(self):
        p = self.q['r0_parity']
        self.assertEqual(p['status'], 'R0_PARITY_PASS')
        self.assertEqual((p['updates_compared'], p['bitwise_equal_vs_scientific'], p['bitwise_equal_vs_n3_fork_A'],
                          p['n3_mask_rows_bitwise_equal_fork_A']), (325, 325, 325, 325))
        self.assertEqual(p['amp_retry_updates'], [5527, 5529, 5758, 5759])
        self.assertIsNotNone(p['collapse'])

    def test_13_deterministic_restart_from_base(self):
        base = None
        for b, d in self.c['branches'].items():
            r = d['result']
            self.assertEqual(r['base_snapshot']['sha256'], self.t.BASE_SHA256, b)
            base = base or r['base_snapshot']['state_digest']
            self.assertEqual(r['base_snapshot']['state_digest'], base, b)
            for s in r['snapshots']:
                if s['global_update_attempted'] == 5526:            # captured live, before the first branch update
                    self.assertEqual(s['state_digest'], base, b)
        five = [b for b, d in self.c['branches'].items()
                if any(s['global_update_attempted'] == 5526 for s in d['result']['snapshots'])]
        self.assertGreaterEqual(len(five), 3)

    def test_14_schedules_and_train_order(self):
        for b, d in self.c['branches'].items():
            s = d['schedule']
            self.assertEqual(s['lambda_adv_exact_mismatch'], [], b)
            self.assertEqual(s['other_terms_vs_scientific_mismatch'], [], b)
            self.assertEqual(s['other_terms_vs_frozen_contract_mismatch'], [], b)
            self.assertTrue(s['contiguous_from_5526'], b)
            self.assertEqual(self.c['train_order'][b]['mismatch'], [], b)
            for u, v in s['lambda_adv_at'].items():
                self.assertEqual(v, self.t.lambda_adv(b, int(u)), (b, u))

    def test_15_firewall_and_access(self):
        for b, d in self.c['branches'].items():
            r = d['result']
            for k in ('val_images', 'val_metadata', 'test_images', 'test_metadata', 'non_train_images'):
                self.assertEqual(r['access'][k], 0, (b, k))
            self.assertEqual(r['write_firewall'], {'denied': [], 'outside_allowed': []}, b)
            last = r['last_completed_update']                  # intercepted (never written) epoch-end saves
            self.assertEqual(r['suppressed_recovery_saves'],
                             [u for u in range(5526, last) if (u - 5525) % 1105 == 0], b)
        self.assertEqual(self.c['scientific_metrics_sha256'],
                         '5f4aa9a004516fe2297b39e0a847f1d12d786e6f6c54f0bc0c401b0f476d632f')
        self.assertEqual(self.c['write_firewall_collect'], {'denied': [], 'outside_allowed': []})

    def test_16_schema_and_verdict(self):
        q = self.q
        for k in ('authority_commit', 'n3_root_cause', 'base_snapshot', 'branch_definitions', 'collapse_rule',
                  'artifact_rule', 'r0_parity', 'branches', 'qualified', 'primary_repair_candidate', 'verdict',
                  'gradient_summary', 'r5', 'train_order', 'source_collected_sha256'):
            self.assertIn(k, q)
        self.assertIn(q['verdict'], ('M7D1_N4_REPAIR_CANDIDATE_QUALIFIED', 'M7D1_N4_NO_REPAIR_QUALIFIED',
                                     'M7D1_N4_QUALIFICATION_BLOCKED'))
        if q['primary_repair_candidate']:
            b = q['primary_repair_candidate']
            self.assertEqual(q['verdict'], 'M7D1_N4_REPAIR_CANDIDATE_QUALIFIED')
            self.assertEqual(q['branches'][b]['qualification'], 'QUALIFIED')
            self.assertEqual(q['branches'][b]['hard_gate_failures'], [])
            self.assertEqual(q['branches'][b]['last_update'], 16575)
        self.assertEqual(q['branches']['R0']['phase_a']['result'], 'COLLAPSED')


if __name__ == '__main__':
    unittest.main()
