"""M7D1-N5: GPAT-B0 / E08 / seed 42 stage-3 adversarial-transition qualification (DIAGNOSTIC_ONLY) -- focused checks.

Stdlib on any host: exact S0..S4 stage-3 schedules (endpoints, continuity, isolation), unchanged lambda_con /
lambda_spec, s_hf behaviour, the predeclared collapse rule, the proposed complete schedule, and the committed evidence
(exact restart from the N4 R1 u16575 state via the bitwise-verified anchor, TRAIN-only access, firewall, unchanged
scientific root, byte-for-byte re-derivation of the derived files). No production config / method / spec file differs
from the authority commit in the commit that added this test (or the worktree, before commit).
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
PREFIX = 'M7D1_N5_SEED42_STAGE3_'
COLLECTED = AUDIT / f'{PREFIX}COLLECTED.json'
DERIVED = ('QUALIFICATION.json', 'SCREEN.csv', 'LONG.csv', 'GRADIENTS.csv')
AUTHORITY = '6fc875ef1c63f0d512e877643e8005879eb8c0cc'
PROTECTED = ('configs', 'methods', 'docs', 'gpatbench', 'environments', 'outputs/audit/STAGE_STATE.json',
             'tools/m7d1_n3_collapse_root_cause.py', 'tools/m7d1_n4_repair_qualification.py')
SCI_LISTING = 'ac64a06351fa8eb45e3efbef8fc64d297d6d3f0d78ee62f8b3cfcd26b6de893a'     # N4 pre-branch listing digest
SCI_METRICS = '5f4aa9a004516fe2297b39e0a847f1d12d786e6f6c54f0bc0c401b0f476d632f'


def tool():
    spec = importlib.util.spec_from_file_location('m7d1_n5', ROOT / 'tools' / 'm7d1_n5_stage3_qualification.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frozen_curriculum(u):
    """Mirror of runtime_contract.curriculum (the tool must leave everything but the listed stage-3 terms alone)."""
    if u < 5526:
        return {'stage': 1, 's_hf': 0.02 + 0.03 * (u - 1) / 5524, 'lambda_adv': 0.0, 'lambda_con': 0.5,
                'lambda_spec': 0.25}
    if u < 16576:
        return {'stage': 2, 's_hf': 0.10, 'lambda_adv': 0.05, 'lambda_con': 1.0, 'lambda_spec': 0.5}
    return {'stage': 3, 's_hf': 0.15, 'lambda_adv': 0.10, 'lambda_con': 1.0, 'lambda_spec': 0.5}


class ScheduleM7D1N5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tool()

    def test_01_s0_original_stage3(self):
        t = self.t
        for u in range(16576, 27626):
            self.assertEqual(t.lambda_adv('S0', u), 0.10)
            self.assertEqual(t.s_hf('S0', u), 0.15)
        self.assertEqual(t.branch_curriculum('S0', frozen_curriculum)(16576), frozen_curriculum(16576))
        self.assertEqual((t.LAMBDA2, t.LAMBDA3, t.S_HF2, t.S_HF3), (0.05, 0.10, 0.10, 0.15))

    def _ramp(self, b, end):
        t = self.t
        vals = [t.lambda_adv(b, u) for u in range(16576, 27626)]
        self.assertEqual(t.lambda_adv(b, 16576), 0.05)            # exact: continuous with the stage-2 value
        self.assertEqual(t.lambda_adv(b, end), 0.10)              # exact in binary floating point
        self.assertTrue(all(t.lambda_adv(b, u) == 0.10 for u in range(end, 27626)))
        self.assertTrue(all(y >= x for x, y in zip(vals, vals[1:])), 'monotone non-decreasing')
        step = 0.05 / (end - 16576)
        for u in range(16577, end + 1):                            # continuity: constant increment, no jump
            self.assertAlmostEqual(t.lambda_adv(b, u) - t.lambda_adv(b, u - 1), step, delta=1e-15)
        for u in range(16576, end + 1):                            # equals the instruction's formula as reals
            self.assertAlmostEqual(t.lambda_adv(b, u), 0.05 + 0.05 * ((u - 16576) / (end - 16576)), delta=1e-17)
        self.assertTrue(all(t.s_hf(b, u) == 0.15 for u in range(16576, 27626)))
        for u in (16575, 27626, 1, 66300):
            with self.assertRaises(ValueError):
                t.lambda_adv(b, u)

    def test_02_s1_ramp(self):
        self._ramp('S1', 17680)
        self.assertEqual(self.t.BRANCHES['S1']['ramp_end'], 17680)
        self.assertEqual(self.t.lambda_adv('S1', 17681), 0.10)

    def test_03_s2_ramp(self):
        self._ramp('S2', 22100)
        self.assertEqual(self.t.BRANCHES['S2']['ramp_end'], 22100)
        self.assertEqual(self.t.lambda_adv('S2', 22101), 0.10)

    def test_04_s3_s4_isolation(self):
        t = self.t
        for u in range(16576, 16901):
            self.assertEqual((t.s_hf('S3', u), t.lambda_adv('S3', u)), (0.15, 0.05))   # only s_hf changes
            self.assertEqual((t.s_hf('S4', u), t.lambda_adv('S4', u)), (0.10, 0.10))   # only lambda_adv changes
        for b in ('S3', 'S4'):
            self.assertFalse(t.BRANCHES[b]['protocol_candidate'])
            self.assertEqual(t.DEFAULT_END[b], 16900)
            with self.assertRaises(ValueError):
                t.lambda_adv(b, 16901)                             # short screen only
        self.assertEqual([b for b in t.BRANCHES if t.BRANCHES[b]['protocol_candidate']], ['S0', 'S1', 'S2'])
        self.assertEqual((t.START, t.PHASE_A_END, t.PHASE_B_END), (16576, 16900, 27625))

    def test_05_only_listed_terms_change(self):
        t = self.t
        for b in t.BRANCHES:
            cur = t.branch_curriculum(b, frozen_curriculum)
            for u in (1, 5526, 16575):                               # nothing before stage 3 is touched
                self.assertEqual(cur(u), frozen_curriculum(u), (b, u))
            for u in (16576, 16577, 16700, 16900) + ((17680, 17681, 22100, 22101, 27625) if b in t.LONG_BRANCHES
                                                     else ()):
                got, ref = cur(u), frozen_curriculum(u)
                self.assertEqual((got['stage'], got['lambda_con'], got['lambda_spec']), (3, 1.0, 0.5), (b, u))
                self.assertEqual({k: v for k, v in got.items() if k not in ('lambda_adv', 's_hf')},
                                 {k: v for k, v in ref.items() if k not in ('lambda_adv', 's_hf')}, (b, u))
                self.assertEqual((got['lambda_adv'], got['s_hf']), (t.lambda_adv(b, u), t.s_hf(b, u)), (b, u))

    def test_06_collapse_rule(self):
        t = self.t
        r = t.COLLAPSE_RULE
        self.assertEqual((r['name'], r['threshold'], r['consecutive_complete_updates']),
                         ('N5_IDENTITY_COLLAPSE_V1', 0.01, 10))
        ev = t.collapse_event
        self.assertIsNone(ev([(u, 0.001) for u in range(100, 109)]))
        self.assertEqual(ev([(u, 0.001) for u in range(100, 110)]), {'onset_update': 100, 'event_update': 109})
        self.assertIsNone(ev([(u, 0.01) for u in range(100, 130)]))                     # strict '<'
        gap = [(u, 0.0) for u in range(100, 105)] + [(u, 0.0) for u in range(106, 111)]
        self.assertIsNone(ev(gap))

    def test_06b_amp_retry_integrity(self):
        t = self.t
        ev = {'global_update_attempted': 2, 'offending_optimizers': ['G_OPT'], 'old_scale': {'G_OPT': 32768.0},
              'new_scale': {'G_OPT': 16384.0}}
        rows = [{'global_update': 1, 'amp_attempts': 1, 'G_scale_before': 32768.0, 'G_scale': 32768.0,
                 'D_scale_before': 1024.0, 'D_scale': 1024.0},
                {'global_update': 2, 'amp_attempts': 2, 'G_scale_before': 32768.0, 'G_scale': 16384.0,
                 'D_scale_before': 1024.0, 'D_scale': 1024.0},
                {'global_update': 3, 'amp_attempts': 1, 'G_scale_before': 16384.0, 'G_scale': 16384.0,
                 'D_scale_before': 1024.0, 'D_scale': 1024.0}]
        self.assertEqual(t.amp_retry_integrity(rows, [ev])['bad_updates'], [])
        self.assertEqual(t.amp_retry_integrity(rows, [dict(ev, new_scale={'G_OPT': 8192.0})])['bad_updates'], [2])
        self.assertEqual(t.amp_retry_integrity(rows, [])['bad_updates'], [2])       # unrecorded attempt / scale drop
        self.assertEqual(t.amp_retry_integrity(rows, [dict(ev, offending_optimizers=['D_OPT'])])['bad_updates'], [2])

    def test_07_proposed_complete_schedule(self):
        t = self.t
        for b, end in (('S0', None), ('S1', 17680), ('S2', 22100)):
            f = t.proposed_schedule(b)
            self.assertEqual([f(u) for u in (1, 5525, 5526, 6630, 6631, 16575)], [0.0, 0.0, 0.0, 0.05, 0.05, 0.05])
            self.assertEqual(f(16576), 0.10 if end is None else 0.05)
            self.assertEqual(f(66300), 0.10)
            if end:
                self.assertEqual((f(end), f(end + 1)), (0.10, 0.10))

    def test_08_roots_and_firewall(self):
        t = self.t
        self.assertEqual(t.N5_PARTS, ('diagnostics', 'm7', 'M7D1_N5_seed42'))
        self.assertEqual(t.n3.DIAG_PARTS, t.N5_PARTS)               # every N3 helper writes under N5 only
        self.assertEqual(len({t.N3_PARTS, t.N4_PARTS, t.N5_PARTS}), 3)
        src = (ROOT / 'tools' / 'm7d1_n5_stage3_qualification.py').read_text()
        self.assertIn("forbidden = [rt + '/runs', str(n3_root()), str(n4_root()), str(ROOT / 'configs'), "
                      "str(ROOT / 'docs'),", src)
        for token in ('val_split', "'val'", "'test'", 'VAL_ROOT', 'TEST_ROOT'):
            self.assertNotIn(token, src)
        self.assertEqual(t.N4_R1_SNAPSHOT, 'snapshots/branch_R1/pre_update_16575.pt')

    def test_09_no_protected_file_changed(self):
        added = subprocess.run(['git', '-C', str(ROOT), 'log', '--diff-filter=A', '-1', '--format=%H', '--',
                                'tests/test_m7d1_n5_stage3_qualification.py'], capture_output=True, text=True).stdout
        cmd = ['git', '-C', str(ROOT), 'diff', '--name-only', AUTHORITY]
        if added.strip():
            cmd.append(added.strip())
        changed = subprocess.run(cmd + ['--', *PROTECTED], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(changed, [])


@unittest.skipUnless(COLLECTED.exists(), 'N5 collected evidence not present')
class EvidenceM7D1N5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tool()
        cls.c = json.loads(COLLECTED.read_text())
        cls.q = json.loads((AUDIT / f'{PREFIX}QUALIFICATION.json').read_text())

    def test_10_identity(self):
        c = self.c
        self.assertEqual((c['method'], c['experiment'], c['seed'], c['scientific_run_id'], c['authority_commit']),
                         ('GPAT-B0', 'E08', 42, '7b799fbd6d0426da', AUTHORITY))
        for lab in ('DIAGNOSTIC_ONLY', 'NOT_AN_APPROVED_PROTOCOL_CHANGE', 'DIAGNOSTIC_REPAIR_VARIANT'):
            self.assertIn(lab, c['labels'])
        self.assertEqual(c['collapse_rule'], self.t.COLLAPSE_RULE)
        self.assertEqual(c['branch_definitions'], self.t.BRANCHES)
        self.assertEqual(c['stage_boundary']['u16575'], frozen_curriculum(16575))
        self.assertEqual(c['stage_boundary']['u16576'], frozen_curriculum(16576))

    def test_11_derived_files_rederive(self):
        out = self.t.finalize(COLLECTED)
        for name in DERIVED:
            self.assertEqual((AUDIT / f'{PREFIX}{name}').read_text(), out[name], name)

    def test_12_exact_r1_u16575_anchor(self):
        a = self.q['anchor']
        self.assertEqual(a['gate'], 'PASS')
        self.assertEqual(a['status'], 'ANCHOR_EXACT')
        self.assertTrue(a['all_bitwise'])
        self.assertEqual(a['record_field_mismatch'], [])
        self.assertEqual(a['n4_per_update_row_field_mismatch'], [])
        self.assertTrue(a['n3_mask_row_equal'] and a['amp_retry_events_equal'])
        b = a['base_n4_r1_snapshot']
        self.assertEqual(b['sha256'], self.t.N4_R1_SNAPSHOT_SHA256)
        self.assertEqual(b['n4_relative_path'], self.t.N4_R1_SNAPSHOT)
        self.assertEqual({k: b['position'][k] for k in self.t.N4_R1_POSITION}, self.t.N4_R1_POSITION)
        self.assertEqual(a['checks']['suppressed_recovery_saves'], [16575])
        self.assertEqual(a['checks']['suppressed_candidate_epochs'], [15])
        s = a['anchor_snapshot']
        self.assertEqual((s['path'], s['global_update_attempted'], s['code_head']),
                         (self.t.ANCHOR_SNAPSHOT, 16576, AUTHORITY))
        self.assertEqual({k: s['state_digest']['position'][k] for k in self.t.ANCHOR_POSITION},
                         self.t.ANCHOR_POSITION)

    def test_13_every_branch_restarts_from_the_anchor(self):
        anchor = self.q['anchor']['anchor_snapshot']
        self.assertEqual(sorted(self.c['branches']), ['S0', 'S1', 'S2', 'S3', 'S4'])
        for b, d in self.c['branches'].items():
            r = d['result']
            self.assertEqual(r['start_snapshot']['sha256'], anchor['sha256'], b)
            self.assertEqual(r['start_snapshot']['state_digest'], anchor['state_digest'], b)
            first = [s for s in r['snapshots'] if s['global_update_attempted'] == 16576]
            self.assertEqual(len(first), 1, b)                      # captured live, before the first branch update
            self.assertEqual(first[0]['state_digest'], anchor['state_digest'], b)
            self.assertTrue(self.c['train_order'][b]['u16576_equals_anchor_group'], b)

    def test_14_schedules_and_train_order(self):
        for b, d in self.c['branches'].items():
            s = d['schedule']
            for k in ('lambda_adv_exact_mismatch', 's_hf_exact_mismatch', 'lr_con_spec_vs_frozen_contract_mismatch',
                      'order_and_lr_vs_scientific_mismatch'):
                self.assertEqual(s[k], [], (b, k))
            self.assertTrue(s['contiguous_from_16576'], b)
            self.assertEqual(self.c['train_order'][b]['mismatch'], [], b)
            for u, v in s['lambda_adv_at'].items():
                self.assertEqual(v, self.t.lambda_adv(b, int(u)), (b, u))
            for u, v in s['s_hf_at'].items():
                self.assertEqual(v, self.t.s_hf(b, int(u)), (b, u))

    def test_15_train_only_firewall_scientific_root(self):
        for b, d in self.c['branches'].items():
            r = d['result']
            for k in ('val_images', 'val_metadata', 'test_images', 'test_metadata', 'non_train_images'):
                self.assertEqual(r['access'][k], 0, (b, k))
                if 'attribution_access' in d:
                    self.assertEqual(d['attribution_access'][k], 0, (b, k))
            self.assertEqual(r['write_firewall'], {'denied': [], 'outside_allowed': []}, b)
            last = r['last_completed_update']                  # intercepted (never written) epoch-end saves
            self.assertEqual(r['suppressed_recovery_saves'], [u for u in range(16576, last) if u % 1105 == 0], b)
        self.assertEqual(self.c['anchor']['access']['val_images'] + self.c['anchor']['access']['test_images'], 0)
        self.assertEqual(self.c['scientific_metrics_sha256'], SCI_METRICS)
        self.assertEqual(self.c['scientific_root_listing_pre_sha256'], SCI_LISTING)
        self.assertEqual(self.c['scientific_root_listing_sha256'], SCI_LISTING)
        self.assertEqual(self.c['write_firewall_collect'], {'denied': [], 'outside_allowed': []})

    def test_16_schema_and_verdict(self):
        q = self.q
        for k in ('authority_commit', 'history', 'anchor', 'branch_definitions', 'collapse_rule', 'artifact_rule',
                  'branches', 'phase_a_outcomes', 'qualified', 'preferred_stage3_branch', 'stage3_status', 'verdict',
                  'gradient_summary', 'train_order', 'source_collected_sha256'):
            self.assertIn(k, q)
        self.assertIn(q['verdict'], ('M7D1_N5_STAGE3_TRANSITION_QUALIFIED', 'M7D1_N5_NO_STAGE3_REPAIR_QUALIFIED',
                                     'M7D1_N5_QUALIFICATION_BLOCKED'))
        order = [b for b, _ in self.t.DECISION_ORDER]
        if q['preferred_stage3_branch']:
            b = q['preferred_stage3_branch']
            self.assertEqual(q['verdict'], 'M7D1_N5_STAGE3_TRANSITION_QUALIFIED')
            self.assertEqual(q['branches'][b]['qualification'], 'QUALIFIED')
            self.assertEqual(q['branches'][b]['hard_gate_failures'], [])
            self.assertEqual(q['branches'][b]['last_update'], 27625)
            for earlier in order[:order.index(b)]:                  # predeclared preference order
                self.assertNotEqual(q['branches'][earlier]['qualification'], 'QUALIFIED')
            self.assertTrue(q['proposed_lambda_adv_schedule']['status'].startswith('DIAGNOSTIC_RECOMMENDATION_ONLY'))
        for b in ('S3', 'S4'):
            self.assertEqual(q['branches'][b]['qualification'], 'NOT_A_CANDIDATE')


if __name__ == '__main__':
    unittest.main()
