"""M6H: the historical M6D6iR/M6D6jR no-A9 assertions inspect their OWN milestone trees, not the current HEAD.

Proves (A) each historical milestone tree has no A9 path and the historical test passes, and (B) a later tree DOES
contain the legitimate Amendment A9 without invalidating the historical assertion. History comparisons use the
commit that ADDED this test (candidate: the worktree), so later milestones never break them.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = 'e936ec252de20ac9dd28bcc8bff0282b5b65d2da'
THIS = 'tests/test_m6h_historical_test_scope.py'
PREFLIGHT = 'tools/m6h_historical_test_scope_preflight.py'
A9_DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A9_DiffFAS_Family_Resource_Constrained_Scope_Exclusion.md'
MODULES = {'tests.test_m6d6ir_e07c_checkpoint_retention': ('m6d6ir', 'TestM6D6iRDecision', 'test_06_no_a9', '39508719'),
           'tests.test_m6d6jr_e07c_science_launch_decision': ('m6d6jr', 'TestM6D6jRDecision', 'test_14_no_a9', '8358d8b6')}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6h_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6h(rel):
    commit = m6h_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def tracked_at_m6h(*paths):
    commit = m6h_commit()
    args = ('ls-tree', '-r', '--name-only', commit, *paths) if commit else ('ls-files', *paths)
    return git(*args).stdout.decode().split()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM6HScope(unittest.TestCase):
    def test_A_historical_trees_have_no_a9_and_tests_pass(self):
        for name, (fn, cls, test, prefix) in MODULES.items():
            mod = importlib.import_module(name)
            commit = getattr(mod, f'{fn}_commit')()
            self.assertTrue(commit.startswith(prefix), name)
            listed = getattr(mod, f'tracked_at_{fn}')('docs', 'configs')
            self.assertEqual(listed, git('ls-tree', '-r', '--name-only', commit, 'docs', 'configs').stdout.decode().split())
            self.assertGreater(len(listed), 10)
            self.assertFalse([p for p in listed if 'A9' in p])
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromName(f'{name}.{cls}.{test}').run(result)
            self.assertTrue(result.wasSuccessful() and result.testsRun == 1, (name, result.failures, result.errors))

    def test_B_later_tree_contains_a9_without_invalidating_history(self):
        later = tracked_at_m6h('docs', 'configs')
        self.assertIn(A9_DOC, later)
        self.assertIn('configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml', later)
        for name, (fn, _, _, _) in MODULES.items():
            mod = importlib.import_module(name)
            self.assertFalse([p for p in getattr(mod, f'tracked_at_{fn}')('docs', 'configs') if 'A9' in p])

    def test_C_historical_records_still_say_no_a9(self):
        for rel in ('configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml',
                    'configs/amendments/e07c_m6d6jr_science_launch_decision.yaml'):
            r = json.loads(at_m6h(rel))
            self.assertEqual((r['amendment_created'], r['a9_created']), (False, False), rel)
            self.assertEqual(at_m6h(rel), at_authority(rel))

    def test_D_change_is_exactly_the_scope_correction(self):
        pf = load(PREFLIGHT, 'm6h_preflight_under_test')
        d = pf.derive(at_m6h, at_authority)
        self.assertEqual(sorted(d['fixed_assertions']), sorted(pf.FIXED))
        self.assertTrue(d['a9_document_unchanged'])
        if (ROOT / pf.EV_JSON).is_file():
            self.assertEqual(pf.check_evidence(at_m6h, at_authority)['correction'],
                             'CURRENT_HEAD_TREE -> OWN_HISTORICAL_MILESTONE_TREE')


if __name__ == '__main__':
    unittest.main()
