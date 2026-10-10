"""M7D1-A1 (N6A): adversarial curriculum amendment -- focused production checks.

Stdlib on any host: exact lambda_adv_a1 values and float bits, the 1105-update ramp, continuity at u6630|u6631, the
unchanged stage-3 jump, every other curriculum / LR value equal to the pre-amendment runtime contract (162bdfd) for
all u, bitwise equality with the qualified N4 R1 and the N5 S0 proposal, one shared curriculum path for B0..B3,
the fresh run identity, and the amendment record.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'outputs' / 'audit' / 'M7D1_A1_ADVERSARIAL_CURRICULUM_AMENDMENT.json'
PRE_AMENDMENT = '162bdfd4cca8e83926ef6f64ff46da2bef1bca66'
SPEC_SHA256 = 'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e'
METHODS = ('GPAT-B0', 'GPAT-B1', 'GPAT-B2', 'GPAT-B3')
EXPECTED = {1: 0.0, 5525: 0.0, 5526: 0.0, 5527: 0.05 * (1 / 1104), 6078: 0.025, 6629: 0.05 * (1103 / 1104),
            6630: 0.05, 6631: 0.05, 16575: 0.05, 16576: 0.10, 66300: 0.10}
EXPECTED_HEX = {5527: '0x1.7beb3922e017cp-15', 6078: '0x1.999999999999ap-6', 6629: '0x1.993a9ecb50e1ap-5',
                6630: '0x1.999999999999ap-5', 16576: '0x1.999999999999ap-4'}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScheduleM7D1A1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.gpat import runtime_contract as rc
        cls.rc = rc
        src = subprocess.run(['git', '-C', str(ROOT), 'show', f'{PRE_AMENDMENT}:methods/gpat/runtime_contract.py'],
                             capture_output=True, text=True, check=True).stdout
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'rc_pre_a1.py').write_text(src)
            cls.pre = load('rc_pre_a1', Path(d) / 'rc_pre_a1.py')

    def test_01_production_is_a1(self):
        rc = self.rc
        from methods.gpat import schedule
        self.assertIs(rc.curriculum, rc.curriculum_a1)
        self.assertIs(schedule.curriculum, rc.curriculum_a1)
        self.assertEqual(rc.CURRICULUM_ID, 'GPAT-TransferBench-v1.0+M7D1-A1')
        self.assertEqual((rc.A1_RAMP_FIRST, rc.A1_RAMP_LAST, rc.STAGE2_FIRST, rc.STAGE3_FIRST), (5526, 6630, 5526, 16576))

    def test_02_exact_representative_values(self):
        rc = self.rc
        for u, v in EXPECTED.items():
            got = rc.curriculum(u)['lambda_adv']
            self.assertEqual(got, v, u)
            self.assertIs(type(got), float)
            self.assertEqual(got, rc.lambda_adv_a1(u))
        for u, h in EXPECTED_HEX.items():
            self.assertEqual(float.hex(rc.lambda_adv_a1(u)), h, u)
        self.assertEqual(float.hex(rc.lambda_adv_a1(5526)), '0x0.0p+0')         # exactly +0.0
        for u in (0, 66301, True, 5526.0):
            with self.assertRaises(ValueError):
                rc.lambda_adv_a1(u)

    def test_03_canonical_expression_all_updates(self):
        rc = self.rc
        for u in range(1, rc.TOTAL_UPDATES + 1):
            if u <= 5525:
                ref = 0.0
            elif u <= 6630:
                ref = 0.05 * ((u - 5526) / (6630 - 5526))
            elif u <= 16575:
                ref = 0.05
            else:
                ref = 0.10
            self.assertEqual(rc.lambda_adv_a1(u), ref, u)

    def test_04_ramp_length_monotone_and_continuous(self):
        rc = self.rc
        ramp = list(range(5526, 6631))
        self.assertEqual(len(ramp), 1105)
        self.assertEqual(len(ramp), rc.UPDATES_PER_EPOCH)                        # exactly one epoch (epoch 6)
        vals = [rc.lambda_adv_a1(u) for u in ramp]
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])), 'strictly increasing on the ramp')
        self.assertEqual((vals[0], vals[-1]), (0.0, 0.05))
        self.assertEqual([u for u in range(5525, 16576) if 0.0 < rc.lambda_adv_a1(u) < 0.05], ramp[1:-1])
        for u in ramp[1:]:
            self.assertAlmostEqual(vals[u - 5526] - vals[u - 5527], 0.05 / 1104, delta=1e-15)
        self.assertEqual(rc.lambda_adv_a1(6630), rc.lambda_adv_a1(6631))       # no discontinuity at u6630|u6631
        self.assertEqual(rc.lambda_adv_a1(5525), rc.lambda_adv_a1(5526))       # nor at the stage-1|2 boundary

    def test_05_stage3_original(self):
        rc = self.rc
        self.assertTrue(all(rc.curriculum(u)['lambda_adv'] == 0.10 for u in range(16576, 66301)))
        self.assertTrue(all(rc.curriculum(u)['lambda_adv'] == 0.05 for u in range(6630, 16576)))
        self.assertEqual(rc.lambda_adv_a1(16576) - rc.lambda_adv_a1(16575), 0.05)    # the original stage-3 jump

    def test_06_everything_else_unchanged_vs_pre_amendment(self):
        rc, pre = self.rc, self.pre
        for u in range(1, rc.TOTAL_UPDATES + 1):
            old, new = pre.curriculum(u), rc.curriculum(u)
            self.assertEqual(rc.curriculum_v1_0(u), old, u)                       # historical form preserved
            self.assertEqual(set(new), set(old))
            self.assertEqual({k: v for k, v in new.items() if k != 'lambda_adv'},
                             {k: v for k, v in old.items() if k != 'lambda_adv'}, u)   # s_hf, con, spec, stage
            if not 5526 <= u <= 6629:
                self.assertEqual(new['lambda_adv'], old['lambda_adv'], u)
            self.assertEqual(rc.main_lr(u), pre.main_lr(u), u)
        for s in range(1, rc.ATTACK_WARMUP_STEPS + 1):
            self.assertEqual(rc.attack_warmup_lr(s), pre.attack_warmup_lr(s))
        changed = [u for u in range(1, rc.TOTAL_UPDATES + 1) if rc.curriculum(u) != pre.curriculum(u)]
        self.assertEqual(changed, list(range(5526, 6630)))                     # 1104 updates differ; u6630 = 0.05
        for name in ('TOTAL_UPDATES', 'UPDATES_PER_EPOCH', 'WARMUP_UPDATES', 'PEAK_LR', 'MIN_LR', 'EMA_DECAY',
                     'EMA_START_UPDATE', 'D_FIRST_UPDATE', 'JOINT_UPDATE_MICROBATCH', 'JOINT_UPDATE_BOUNDARY'):
            self.assertEqual(getattr(rc, name), getattr(pre, name), name)

    def test_07_equals_qualified_n4_r1_and_n5_s0(self):
        rc = self.rc
        n4 = load('m7d1_n4', ROOT / 'tools' / 'm7d1_n4_repair_qualification.py')
        for u in range(5526, 16576):
            self.assertEqual(rc.lambda_adv_a1(u), n4.lambda_adv('R1', u), u)    # bitwise, the qualified trajectory
        n5 = load('m7d1_n5', ROOT / 'tools' / 'm7d1_n5_stage3_qualification.py')
        prop = n5.proposed_schedule('S0')
        for u in range(1, rc.TOTAL_UPDATES + 1):
            self.assertEqual(rc.lambda_adv_a1(u), prop(u), u)


class SharedPathM7D1A1(unittest.TestCase):
    def test_08_one_curriculum_for_b0_b3(self):
        src = (ROOT / 'methods' / 'gpat' / 'runner.py').read_text()
        self.assertEqual(src.count('rc.curriculum(u)'), 2)                       # Trainer.attempt + recovery position
        self.assertNotIn('cfg.lambda_adv', src)
        self.assertNotIn("cfg.loss['lambda_adv']", src)
        losses = (ROOT / 'methods' / 'gpat' / 'losses.py').read_text()
        self.assertIn("'gadv': 'lambda_adv'", losses)                          # weight read from curriculum dict
        from methods.gpat import runner_io as rio
        self.assertEqual(set(rio.VARIANTS), set(METHODS))

    def test_09_minimal_production_change(self):
        """Pinned to PRE_AMENDMENT..the commit that added this test (or the worktree, before commit)."""
        added = subprocess.run(['git', '-C', str(ROOT), 'log', '--diff-filter=A', '-1', '--format=%H', '--',
                                'tests/test_m7d1_a1_adversarial_curriculum.py'], capture_output=True, text=True).stdout
        cmd = ['git', '-C', str(ROOT), 'diff', '--name-only', PRE_AMENDMENT] + ([added.strip()] if added.strip() else [])
        guarded = ('configs', 'frozen_config_snapshot', 'docs', 'gpatbench', 'environments', 'methods', 'tools',
                   'outputs/audit')
        out = subprocess.run(cmd + ['--', *guarded], capture_output=True, text=True, check=True).stdout.split()
        out = [f for f in out if not f.startswith('outputs/audit/M7D1_A1_ADVERSARIAL_CURRICULUM_AMENDMENT.')
               and f != 'outputs/audit/ARTIFACT_INDEX.csv']                # rows of the modified files refreshed
        self.assertEqual(sorted(out), ['methods/gpat/runner.py', 'methods/gpat/runner_io.py',
                                       'methods/gpat/runtime_contract.py', 'methods/gpat/schedule.py'])


class IdentityM7D1A1(unittest.TestCase):
    def test_10_fresh_scientific_root_and_provenance(self):
        from methods.gpat import runner_io as rio
        rt = '/home/student20261/workdir/GPAT_TransferBench_runtime'
        new = str(rio.run_root(rt, rio.SCIENTIFIC, 'GPAT-B0', 42))
        self.assertEqual(new, rt + '/runs/m7_a1/E08/seed_42')
        self.assertNotEqual(new, rt + '/runs/m7/E08/seed_42')                   # historical run 7b799fbd6d0426da
        self.assertEqual(rio.run_id('GPAT-B0', 42, '0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a',
                                    '058e976e5a538c4d18e1177f0eb27727cfb735cd'), '7b799fbd6d0426da')
        src = (ROOT / 'methods' / 'gpat' / 'runner.py').read_text()
        self.assertIn("'curriculum_id': rc.CURRICULUM_ID", src)
        ck = (ROOT / 'methods' / 'gpat' / 'runner_checkpoint.py').read_text()
        self.assertIn("f'resume refused: provenance {key} differs'", ck)       # a v1.0 recovery cannot resume


class RecordM7D1A1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = json.loads(RECORD.read_text())

    def test_11_record(self):
        from methods.gpat import runtime_contract as rc
        r = self.rec
        self.assertEqual(r['amendment_id'], 'M7D1-A1')
        self.assertEqual(rc.A1_RECORD, str(RECORD.relative_to(ROOT)))
        self.assertEqual(r['original_frozen_specification']['sha256'], SPEC_SHA256)
        self.assertEqual(r['authority_chain']['pre_amendment_head'], PRE_AMENDMENT)
        self.assertEqual({k: v['commit'][:7] for k, v in r['authority_chain'].items() if k != 'pre_amendment_head'},
                         {'N3': 'd88563a', 'N4': '6fc875e', 'N5': '162bdfd'})
        self.assertEqual(r['changed_field'], 'lambda_adv(u) only')
        rep = r['float_definition']['representative_values']
        self.assertEqual(set(rep), {f'u{u}' for u in EXPECTED})
        for u, v in EXPECTED.items():
            self.assertEqual(rep[f'u{u}']['lambda_adv'], v)
            self.assertEqual(rep[f'u{u}']['hex'], float.hex(v))
            self.assertEqual(rep[f'u{u}']['original_v1_0'], rc.curriculum_v1_0(u)['lambda_adv'])
        self.assertEqual(r['run_identity']['curriculum_id'], rc.CURRICULUM_ID)
        n6b = r['planned_N6B_run']
        self.assertEqual((n6b['method'], n6b['seed'], n6b['not_created_in_N6A']), ('GPAT-B0', 42, True))
        self.assertTrue(n6b['run_root'].endswith('/runs/m7_a1/E08/seed_42'))
        self.assertEqual(n6b['historical_run_kept']['run_id'], '7b799fbd6d0426da')
        self.assertFalse(r['checkpoint_policy_N6B']['performed_in_N6A'])
        self.assertIn('mask saturates open (M_mean ~1)', r['limitations'])


if __name__ == '__main__':
    unittest.main()
