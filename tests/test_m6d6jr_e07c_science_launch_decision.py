"""M6D6jR: E07c science-launch OWNER DECISION (O3 smoke-test timing, O4 resume before science). Static only.

History assertions compare the state at the commit that ADDED this test (candidate: the worktree) with the M6D6j
authority, so later milestones never break them (no B1 HEAD lock).
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = 'ade5900db1dbff73917a2eb2280f8f2fe7afc418'
THIS = 'tests/test_m6d6jr_e07c_science_launch_decision.py'
RECORD = 'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml'
EVIDENCE = ROOT / 'outputs/audit/M6D6JR_E07C_SCIENCE_LAUNCH_DECISION.json'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6d6jr_preflight_under_test',
                                                  ROOT / 'tools/m6d6jr_e07c_science_launch_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6d6jr_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6d6jr(rel):
    """File bytes at the M6D6jR state: the commit that added this test, else the (candidate) worktree."""
    commit = m6d6jr_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM6D6jRDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.raw = at_m6d6jr(RECORD)
        cls.r = json.loads(cls.raw)
        cls.d = cls.pf.derive(at_m6d6jr)

    def rejects(self, mutate):
        bad = copy.deepcopy(self.r)
        mutate(bad)
        raw = json.dumps(bad, indent=2, ensure_ascii=False).encode() + b'\n'
        pf = load_preflight()
        pf.RECORD_SHA = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):
            pf.check_record(lambda rel: raw if rel == RECORD else at_m6d6jr(rel))

    def test_00_record_hash_bindings_and_authority(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), self.pf.RECORD_SHA)
        self.pf.check_record(at_m6d6jr)
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)
        for rel, digest in self.r['bound_authority_sha256'].items():
            self.assertEqual(hashlib.sha256(at_authority(rel)).hexdigest(), digest, rel)
        self.assertEqual(self.r['frozen_encoder']['sha256'],
                         '49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c')

    def test_01_m6_m7_precede_m8_m9_m10(self):
        pos = self.d['spec_milestone_paragraphs']
        self.assertTrue(pos['M6 Baselines'] < pos['M7 GPAT'] < pos['M8 Banks'] < pos['M9 Generator eval'] <
                        pos['M10 ResNet downstream'] < pos['M11 DINOv3 downstream'])

    def test_02_smoke_requirement_still_exists(self):
        self.assertTrue(self.d['smoke_clause_present'] and self.d['smoke_checklist_present'])
        self.assertIs(self.r['O3_smoke_test']['smoke_test_required'], True)
        self.rejects(lambda b: b['O3_smoke_test'].update(smoke_test_required=False))

    def test_03_smoke_not_claimed_completed(self):
        o3 = self.r['O3_smoke_test']
        self.assertEqual((o3['smoke_test_completed'], o3['m6d6j_is_the_smoke_test']), (False, False))
        self.rejects(lambda b: b['O3_smoke_test'].update(smoke_test_completed=True))

    def test_04_smoke_not_deleted_or_waived(self):
        o3 = self.r['O3_smoke_test']
        self.assertEqual((o3['smoke_test_waived'], o3['smoke_test_deleted']), (False, False))
        self.assertEqual(o3['must_exercise'], ['generator', 'bank', 'evaluator', 'metrics', 'plots'])
        self.rejects(lambda b: b['O3_smoke_test'].update(smoke_test_waived=True))

    def test_05_hard_gate_before_full_m8_plus(self):
        self.assertIs(self.r['O3_smoke_test']['blocks_full_M8_plus_execution_until_completed'], True)
        self.rejects(lambda b: b['O3_smoke_test'].update(blocks_full_M8_plus_execution_until_completed=False))

    def test_06_no_longer_blocks_e07c_m6_training(self):
        self.assertIs(self.r['O3_smoke_test']['blocks_E07c_M6_training'], False)
        self.assertEqual(self.r['launch']['FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION'], False)
        self.assertEqual(self.r['launch']['E07c_main_scientific_training'],
                         'AUTHORIZED_TO_LAUNCH_AFTER_THIS_DECISION_IS_COMMITTED')
        self.assertTrue(self.d['m6d6j_pending_blockers']['FULL_SCIENCE_BLOCKED_PENDING_SMOKE_TEST_DECISION'])  # historical

    def test_07_production_runner_remains_qualified(self):
        self.assertTrue(self.d['m6d6j_production_runner_qualified'])
        self.assertEqual(self.r['statuses']['remains_qualified'], ['MAIN_PRODUCTION_RUNNER'])

    def test_08_resume_remains_unqualified(self):
        o4 = self.r['O4_resume']
        self.assertEqual((o4['MAIN_CHECKPOINT_RESUME'], o4['resume_qualified'], o4['resume_required_before_science'],
                          o4['m6d6k_prerequisite_for_science'], self.d['m6d6j_resume']),
                         ('UNQUALIFIED', False, False, False, 'UNQUALIFIED'))
        self.rejects(lambda b: b['O4_resume'].update(resume_qualified=True))

    def test_09_cli_fresh_only(self):
        self.assertTrue(self.d['cli_fresh_only'])
        self.assertEqual(self.r['O4_resume']['scientific_runs'], 'FRESH_ONLY')

    def test_10_no_resume_path(self):
        self.assertEqual(self.d['cli_options'], ['--seed', '--execution-config', '--preflight-only'])
        self.assertTrue({'--resume', '--resume-state', '--resume-latest', '--auto-resume'} <=
                        set(self.d['cli_refused_resume_flags']))
        self.assertIs(self.r['O4_resume']['resume_flag_or_path_allowed'], False)

    def test_11_failure_policy_same_seed_fresh_rerun(self):
        o4 = self.r['O4_resume']
        self.assertEqual(o4['recovery_policy'], 'FRESH_RERUN_SAME_SEED_AFTER_TECHNICAL_FIX')
        self.assertTrue(self.d['failure_policy_same_seed_rerun'])
        self.assertTrue({'substitute another seed', 'rerun to obtain a better result'} <= set(o4['forbidden']))
        self.rejects(lambda b: b['O4_resume'].update(recovery_policy='RESUME_LATEST'))

    def test_12_seeds_exact(self):
        self.assertEqual(self.d['experiment_seeds'], [42, 1337, 2026])
        self.assertEqual(self.r['scientific_run_contract_unchanged']['experiment_seeds'], [42, 1337, 2026])
        self.rejects(lambda b: b['scientific_run_contract_unchanged'].update(experiment_seeds=[42, 1337]))

    def test_13_no_science_values_changed(self):
        c = self.r['scientific_run_contract_unchanged']
        self.assertEqual((c['epochs'], c['batch_size'], c['drop_last'], c['iterations_per_epoch'], c['total_iterations'],
                          c['periodic_checkpoint_every'], c['changed']), (400, 4, False, 2210, 884000, 10000, False))
        rc = self.d['run_contract']
        self.assertEqual((rc['epochs'], rc['batch_size'], rc['total_iterations'], rc['checkpoint_cadence'],
                          rc['terminal_step'], rc['visualization_cadence']), (400, 4, 884000, 10000, 884000, 1000))
        for rel in self.r['bound_authority_sha256']:
            self.assertEqual(at_m6d6jr(rel), at_authority(rel), rel)
        self.rejects(lambda b: b['scientific_run_contract_unchanged'].update(epochs=200))

    def test_14_no_a9(self):
        self.assertEqual((self.r['amendment_created'], self.r['a9_created']), (False, False))
        self.assertFalse([p for p in git('ls-files', 'docs', 'configs').stdout.decode().split() if 'A9' in p])

    def test_15_no_new_deviation_or_fidelity_class(self):
        self.assertEqual((self.r['fidelity_class'], self.r['deviation'], self.r['new_deviation'],
                          self.r['new_fidelity_class'], self.r['classification']),
                         ('CONTROLLED_ADAPTATION', 'DEV-021', False, False, 'DETERMINISTIC_IMPLEMENTATION_CLARIFICATION'))
        self.rejects(lambda b: b.update(new_deviation=True))

    def test_16_scientific_training_unqualified_until_runs(self):
        self.assertIn('MAIN_DIFFFAS_SCIENTIFIC_TRAINING', self.r['statuses']['not_qualified'])
        self.assertIs(self.r['launch']['scientific_training_completed'], False)
        self.assertEqual(self.r['statuses']['method_status'], 'IMPLEMENTED_NOT_EXECUTED')

    def test_17_m8_bank_unqualified(self):
        self.assertIn('M8_BANK', self.r['statuses']['not_qualified'])
        self.assertEqual(self.r['decision_operation'], self.pf.ZERO)

    def test_18_history_future_safe_and_preflight_static(self):
        import inspect
        self.assertIn('--diff-filter=A', inspect.getsource(m6d6jr_commit))
        own = ast.parse(Path(__file__).read_text())
        git_calls = [[ast.literal_eval(a) for a in n.args if isinstance(a, ast.Constant)] for n in ast.walk(own)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == 'git']
        self.assertFalse([c for c in git_calls if c[:1] in (['status'], ['diff'], ['rev-parse'])])
        src = inspect.getsource(self.pf.check_record) + inspect.getsource(self.pf.derive)
        for token in ('git(', 'worktree(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, src)
        tree_ = ast.parse((ROOT / 'tools/m6d6jr_e07c_science_launch_preflight.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        self.assertTrue(at_m6d6jr('outputs/audit/EXECUTION_LEDGER.jsonl').startswith(
            at_authority('outputs/audit/EXECUTION_LEDGER.jsonl')))
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6D6jR evidence not yet recorded')
class TestM6D6jREvidence(unittest.TestCase):
    def test_evidence_validates(self):
        load_preflight().check_evidence(at_m6d6jr)


if __name__ == '__main__':
    unittest.main()
