"""M6A9: Amendment A9 — DiffFAS family (E07c, E07b) resource-constrained scope exclusion. Static only.

A9 is not the M6 closure: M6 stays OPEN and E06b DSDG-NATIVE stays active M6 work. History assertions compare the
state at the commit that ADDED this test (candidate: the worktree) with the M6D6jF authority, so later milestones never
break them (no B1 HEAD lock).
"""
import ast
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = '632e274bde083fd175dc6a2645542d281a767063'
THIS = 'tests/test_m6a9_difffas_scope_exclusion.py'
PREFLIGHT = 'tools/m6a9_difffas_scope_exclusion_preflight.py'
RECORD = 'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'
EVIDENCE = ROOT / 'outputs/audit/M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION.json'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
HISTORY_PREFIXES = ('outputs/audit/', 'docs/', 'configs/', 'frozen_config_snapshot/', 'methods/', 'tools/', 'tests/',
                    'third_party/')
M6A9_MODIFIED = {'configs/CONFIG_STATUS.md', 'outputs/audit/ARTIFACT_INDEX.csv', LEDGER}


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6a9_preflight_under_test', ROOT / PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6a9_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6a9(rel):
    """File bytes at the M6A9 state: the commit that added this test, else the (candidate) worktree."""
    commit = m6a9_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def exists_at_m6a9(rel):
    commit = m6a9_commit()
    return git('cat-file', '-e', f'{commit}:{rel}').returncode == 0 if commit else (ROOT / rel).exists()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM6A9Decision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.raw = at_m6a9(RECORD)
        cls.r = json.loads(cls.raw)
        cls.d = cls.pf.derive(at_m6a9)

    def rejects(self, mutate):
        bad = copy.deepcopy(self.r)
        mutate(bad)
        raw = json.dumps(bad, indent=2, ensure_ascii=False).encode() + b'\n'
        pf = load_preflight()
        pf.RECORD_SHA = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):
            pf.check_record(lambda rel: raw if rel == RECORD else at_m6a9(rel))

    def test_00_record_hash_bindings_and_authority(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), self.pf.RECORD_SHA)
        self.pf.check_record(at_m6a9)
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)
        for rel, digest in self.r['bound_authority_sha256'].items():
            self.assertEqual(hashlib.sha256(at_authority(rel)).hexdigest(), digest, rel)

    def test_01_scope_is_difffas_family_only(self):
        self.assertEqual((self.r['scope']['methods'], self.r['scope']['family']), (['E07c', 'E07b'], 'DiffFAS'))
        self.assertEqual(self.r['classification'], 'OWNER_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION')
        self.assertTrue(self.r['scientific_scope_change'])
        self.assertIn('BLOCKED_BY_SOURCE_GAP', self.r['not_classified_as'])
        self.rejects(lambda b: b['scope']['methods'].append('E06b'))
        self.rejects(lambda b: b.update(classification='BLOCKED_BY_SOURCE_GAP'))

    def test_02_e06b_out_of_scope_and_active(self):
        e = self.r['E06b']
        self.assertEqual((e['in_a9_scope'], e['owner_excluded'], e['resource_blocked'], e['status']),
                         (False, False, False, 'ACTIVE_M6_WORK'))
        self.assertEqual(self.d['track_b_deferred_to_m6'], ['E06b', 'E07b'])
        self.rejects(lambda b: b['E06b'].update(owner_excluded=True))
        self.rejects(lambda b: b['E06b'].update(resource_blocked=True))
        self.rejects(lambda b: b['E06b'].update(status='OWNER_EXCLUDED_RESOURCE_CONSTRAINT'))
        doc = at_m6a9(self.pf.DOC).decode()
        self.assertNotIn('OWNER_EXCLUDED_SECONDARY_TRACK', doc)

    def test_03_e07c_excluded_fidelity_retained(self):
        e = self.r['E07c']
        self.assertEqual((e['m6_gate_status'], e['blocker_or_exclusion_reason'], e['fidelity_class'], e['deviation'],
                          e['new_fidelity_class'], e['new_deviation']),
                         ('blocked', 'OWNER_EXCLUDED_RESOURCE_CONSTRAINT', 'CONTROLLED_ADAPTATION', 'DEV-021', False,
                          False))
        self.assertEqual(self.d['e07c_fidelity_historical'], 'CONTROLLED_ADAPTATION / DEV-021')
        self.rejects(lambda b: b['E07c'].update(fidelity_class='BLOCKED'))
        self.rejects(lambda b: b['E07c'].update(deviation='NONE'))

    def test_04_e07c_seed_history(self):
        e = self.r['E07c']
        a1, a2 = e['attempts']
        self.assertEqual((a1['attempt'], a1['failure_phase'], a1['scientific_optimizer_steps']), (1, 'RUN_CONTEXT_OPEN', 0))
        self.assertTrue(self.d['attempt_1_preserved_by_m6d6jf'])
        self.assertEqual((a2['attempt'], a2['last_completed_global_step'], a2['interruption'], a2['completed_seed'],
                          a2['benchmark_result']), (2, 54836, 'OWNER_MANUAL_INTERRUPT', False, False))
        self.assertEqual((e['completed_scientific_seeds'], e['seeds_never_started']), (0, [1337, 2026]))
        self.rejects(lambda b: b['E07c']['attempts'].pop(0))
        self.rejects(lambda b: b['E07c'].update(completed_scientific_seeds=1))
        self.rejects(lambda b: b['E07c'].update(seeds_never_started=[2026]))

    def test_05_partial_checkpoint_never_selected_or_eligible(self):
        c = self.r['E07c']['partial_checkpoint']
        self.assertEqual((c['path'], c['sha256'], c['retention_decision']),
                         ('checkpoints/model_050000.pt',
                          'b4ae6325bd04fdf615db24ca0cfb883680401c494b9c9d39f4830dbc8d775cae', 'RETAIN'))
        self.assertEqual((c['selected'], c['final'], c['terminal'], c['selected_for_final'], c['eligible_for_M8'],
                          c['eligible_for_downstream_evaluation'], c['eligible_for_paper_numeric_reporting'],
                          c['new_pruning_permission']), (False,) * 8)
        self.assertIs(self.r['E07c']['eligible_for_future_bank'], False)
        self.rejects(lambda b: b['E07c']['partial_checkpoint'].update(final=True))
        self.rejects(lambda b: b['E07c'].update(eligible_for_future_bank=True))

    def test_06_e07b_excluded_not_assessed(self):
        e = self.r['E07b']
        self.assertEqual((e['m6_gate_status'], e['fidelity_class'], e['fidelity_assessed'], e['eligible_for_future_bank'],
                          e['downstream_evaluation'], e['numeric_result_reporting'], e['source_gap_claimed']),
                         ('blocked', 'NOT_ASSESSED_NOT_IMPLEMENTED', False, False, False, False, False))
        self.rejects(lambda b: b['E07b'].update(fidelity_assessed=True))
        self.rejects(lambda b: b['E07b'].update(eligible_for_future_bank=True))

    def test_07_m6_open_and_no_method_status(self):
        self.assertEqual((self.r['m6']['M6_CLOSED'], self.r['m6']['M6_status'], self.r['m6']['method_status_csv_created']),
                         (False, 'OPEN', False))
        for rel in self.pf.M6E_CLOSURE_ARTIFACTS:
            self.assertFalse(exists_at_m6a9(rel), rel)
        self.rejects(lambda b: b['m6'].update(M6_CLOSED=True))
        self.rejects(lambda b: b['m6'].update(method_status_csv_created=True))

    def test_08_ledger_prefix_byte_identical(self):
        prefix, now = at_authority(LEDGER), at_m6a9(LEDGER)
        self.assertEqual(hashlib.sha256(prefix).hexdigest(), self.pf.LEDGER_PREFIX_SHA)
        self.assertEqual(len(prefix.splitlines()), self.pf.LEDGER_PREFIX_ROWS)
        self.assertTrue(now.startswith(prefix))
        appended = now[len(prefix):].splitlines()
        self.assertLessEqual(len(appended), 1)
        if appended:
            row = json.loads(appended[0])
            for k, v in self.pf.LEDGER_EXPECTED.items():
                self.assertEqual(row[k], v, k)

    def test_09_historical_evidence_unchanged(self):
        paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY).stdout.decode().splitlines()
                 if p.startswith(HISTORY_PREFIXES) and p not in M6A9_MODIFIED]
        self.assertGreater(len(paths), 500)
        commit = m6a9_commit()
        if commit:
            self.assertFalse(set(git('diff', '--name-only', AUTHORITY, commit).stdout.decode().split()) & set(paths))
        else:
            index = {r['path']: r['sha256'] for r in csv.DictReader(io.StringIO(
                at_authority('outputs/audit/ARTIFACT_INDEX.csv').decode()))}
            for rel in paths:
                self.assertTrue((ROOT / rel).is_file(), rel)
                if rel in index:
                    self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), index[rel], rel)
        base, now = at_authority('configs/CONFIG_STATUS.md'), at_m6a9('configs/CONFIG_STATUS.md')
        self.assertTrue(now.startswith(base) and len(now) > len(base), 'CONFIG_STATUS append-only')

    def test_10_no_scientific_config_altered(self):
        for rel, digest in self.r['bound_authority_sha256'].items():
            self.assertEqual(hashlib.sha256(at_m6a9(rel)).hexdigest(), digest, rel)
        for rel in self.pf.METHOD_CONFIGS.values():
            self.assertEqual(at_m6a9(rel), at_m6a9('frozen_config_snapshot/' + rel), rel)

    def test_11_zero_runtime_activity_and_smoke_gate(self):
        self.assertEqual(self.r['decision_operation'], self.pf.ZERO)
        s = self.r['smoke_test']
        self.assertEqual((s['smoke_test_required'], s['smoke_test_completed'], s['blocks_full_M8_plus_execution']),
                         (True, False, True))
        self.rejects(lambda b: b['decision_operation'].update(optimizer_steps=1))
        self.rejects(lambda b: b['smoke_test'].update(smoke_test_completed=True))

    def test_12_stage_state_stale_fixture_byte_unchanged(self):
        rel = 'outputs/audit/STAGE_STATE.json'
        ss = self.r['stage_state_json']
        self.assertEqual((ss['status'], ss['current_milestone_authority'], ss['modified_by_m6a9']),
                         ('STALE_HISTORICAL_FIXTURE', False, False))
        self.assertEqual(at_m6a9(rel), at_authority(rel))

    def test_13_history_future_safe_and_preflight_static(self):
        import inspect
        self.assertIn('--diff-filter=A', inspect.getsource(m6a9_commit))
        own = ast.parse(Path(__file__).read_text())
        git_calls = [[ast.literal_eval(a) for a in n.args if isinstance(a, ast.Constant)] for n in ast.walk(own)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == 'git']
        self.assertFalse([c for c in git_calls if c[:1] in (['status'], ['rev-parse'])])
        src = ''.join(inspect.getsource(f) for f in (self.pf.check_record, self.pf.derive, self.pf.check_evidence))
        for token in ('git(', 'worktree(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, src)
        tree_ = ast.parse((ROOT / PREFLIGHT).read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6A9 evidence not yet recorded')
class TestM6A9Evidence(unittest.TestCase):
    def test_evidence_validates(self):
        ev = load_preflight().check_evidence(at_m6a9)
        self.assertEqual((ev['m6']['M6_CLOSED'], ev['E06b']['status'], ev['scope']['methods']),
                         (False, 'ACTIVE_M6_WORK', ['E07c', 'E07b']))


if __name__ == '__main__':
    unittest.main()
