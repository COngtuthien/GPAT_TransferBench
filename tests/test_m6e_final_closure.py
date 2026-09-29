"""M6E: final M6 baseline closure (M6_CLOSED = true at implementation/qualification level). Static only.

History assertions read the state at the commit that ADDED this test (candidate: the worktree) and compare it with the
M6F-D authority, so later milestones never break them (no B1 HEAD lock).
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
AUTHORITY = '57c6a99035c2d45f47882c2104751a4cc0b923b1'
THIS = 'tests/test_m6e_final_closure.py'
PREFLIGHT = 'tools/m6e_final_closure_preflight.py'
EVIDENCE = ROOT / 'outputs/audit/M6E_FINAL_M6_CLOSURE.json'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
INDEX = 'outputs/audit/ARTIFACT_INDEX.csv'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6e_preflight_under_test', ROOT / PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6e_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6e(rel):
    """File bytes at the M6E state: the commit that added this test, else the (candidate) worktree."""
    commit = m6e_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def exists_at_m6e(rel):
    commit = m6e_commit()
    return git('cat-file', '-e', f'{commit}:{rel}').returncode == 0 if commit else (ROOT / rel).exists()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM6EClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.d = cls.pf.derive(at_m6e)
        cls.rows = cls.pf.check_rows(at_m6e)

    def rejects(self, mutate):
        bad = copy.deepcopy(self.pf.ROWS)
        mutate({r['experiment_id']: r for r in bad}, bad)
        raw = self.pf.render(bad)
        with self.assertRaises(ValueError):
            self.pf.check_rows(lambda rel: raw if rel == self.pf.METHOD_STATUS else at_m6e(rel), rows=bad)

    def test_00_universe_derived_from_spec_a1_a9(self):
        d = self.d
        self.assertEqual(d['spec_17_ids'], ['E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06a', 'E06b', 'E07a', 'E07b',
                                            'E08', 'E09', 'E10', 'E11'])
        self.assertEqual(d['universe'], ['E01', 'E02', 'E03', 'E04', 'E05', 'E06a', 'E06b', 'E06c', 'E07a', 'E07b',
                                         'E07c'])
        self.assertEqual((d['row_count'], d['non_rows'], d['a1_track_a_additions'], d['superseded_by_a1'],
                          d['a9_owner_excluded']),
                         (11, ['E00', 'E08', 'E09', 'E10', 'E11'], ['E06c', 'E07c'], ['E06a', 'E07a'], ['E07b', 'E07c']))
        self.assertEqual(sorted(self.pf.NON_ROWS), d['non_rows'])
        self.assertEqual(self.pf.NON_ROWS['E00']['classification'], 'NOT_AN_M6_BASELINE_NO_SYNTHETIC_METHOD')
        for e in ('E08', 'E09', 'E10', 'E11'):
            self.assertEqual(self.pf.NON_ROWS[e]['classification'], 'NOT_AN_M6_BASELINE_PROPOSED_METHOD_M7')

    def test_01_rows_gates_and_no_silent_omission(self):
        self.assertEqual(self.rows['rows'], 11)
        self.assertEqual(self.rows['gates'], {'faithful': ['E01', 'E03', 'E06b'], 'adapted': ['E02', 'E04', 'E05', 'E06c'],
                                              'blocked': ['E06a', 'E07a', 'E07b', 'E07c']})
        self.assertEqual(self.pf.GATES, ('faithful', 'adapted', 'blocked'))
        self.assertEqual(hashlib.sha256(at_m6e(self.pf.METHOD_STATUS)).hexdigest(), self.rows['method_status_sha256'])
        self.rejects(lambda by, rows: rows.pop(5))                                    # silently drop E06a
        self.rejects(lambda by, rows: rows.append(dict(rows[0])))                     # duplicate id
        self.rejects(lambda by, rows: by['E06a'].update(gate='superseded'))           # a fourth gate label
        self.rejects(lambda by, rows: by['E07a'].update(gate='adapted'))

    def test_02_active_classifications(self):
        self.rejects(lambda by, rows: by['E02'].update(gate='faithful'))
        self.rejects(lambda by, rows: by['E02'].update(spec_intended_status='SPEC_DEFINED'))
        self.rejects(lambda by, rows: by['E03'].update(fidelity_class='FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION'))
        self.rejects(lambda by, rows: by['E04'].update(gate='faithful'))
        self.rejects(lambda by, rows: by['E06c'].update(deviation_or_authority='NONE'))
        for e in ('E01', 'E02', 'E03', 'E04', 'E05'):
            self.rejects(lambda by, rows, e=e: by[e].update(production_qualification_status='QUALIFIED_ON_REAL_TRAIN'))
            self.rejects(lambda by, rows, e=e: by[e].update(completed_scientific_seeds='3'))

    def test_03_e06b_faithful_not_excluded(self):
        self.rejects(lambda by, rows: by['E06b'].update(gate='blocked'))
        self.rejects(lambda by, rows: by['E06b'].update(scope_status='OWNER_EXCLUDED'))
        self.rejects(lambda by, rows: by['E06b'].update(fidelity_class='CONTROLLED_ADAPTATION'))
        self.rejects(lambda by, rows: by['E06b'].update(disclosures='none'))
        self.rejects(lambda by, rows: by['E06b'].update(block_or_supersession_reason='OWNER_EXCLUDED_RESOURCE_CONSTRAINT'))
        self.rejects(lambda by, rows: by['E06b'].update(scientific_execution_status='SCIENTIFIC_TRAINING_COMPLETED'))

    def test_04_a9_rows_and_e07c_history(self):
        self.rejects(lambda by, rows: by['E07c'].update(fidelity_class='NOT_ASSESSED_NOT_IMPLEMENTED'))
        self.rejects(lambda by, rows: by['E07c'].update(future_bank_eligible='true'))
        self.rejects(lambda by, rows: by['E07c'].update(completed_scientific_seeds='1'))
        self.rejects(lambda by, rows: by['E07c'].update(checkpoint_status='model_050000.pt SELECTED_FINAL'))
        self.rejects(lambda by, rows: by['E07c'].update(scientific_execution_status='SEED_42_COMPLETED'))
        self.rejects(lambda by, rows: by['E07b'].update(fidelity_class='FAITHFUL_OFFICIAL'))
        self.rejects(lambda by, rows: by['E07b'].update(block_or_supersession_reason='BLOCKED_BY_SOURCE_GAP'))

    def test_05_records_reconciled_additively(self):
        rec = self.pf.check_records(at_m6e, at_authority)
        self.assertEqual(rec['deviation_register']['new_dev_numbers'], [])
        for rel in (self.pf.DEVIATIONS, self.pf.CONFIG_STATUS):
            self.assertTrue(at_m6e(rel).startswith(at_authority(rel)), rel)
        reg = at_m6e(self.pf.REGISTRY).decode()
        self.assertEqual(reg.count('    implementation_status: NOT_STARTED\n'),
                         at_authority(self.pf.REGISTRY).decode().count('    implementation_status: NOT_STARTED\n'))
        self.assertIn('    m6_final_fidelity: FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY\n',
                      self.pf.registry_entry(reg, 'E06b'))

    def test_06_frozen_authority_and_stage_state_unchanged(self):
        for rel, digest in self.pf.BOUND.items():
            self.assertEqual(hashlib.sha256(at_m6e(rel)).hexdigest(), digest, rel)
            self.assertEqual(at_m6e(rel), at_authority(rel), rel)
        self.assertEqual(self.pf.BOUND['configs/methods/e06b_dsdg_native.yaml'],
                         '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f')
        for rel in self.pf.METHOD_CONFIGS.values():
            self.assertEqual(at_m6e(rel), at_m6e('frozen_config_snapshot/' + rel), rel)

    def test_07_only_closure_files_changed(self):
        allowed = set(self.pf.NEW) | set(self.pf.MODIFIED) | {LEDGER, INDEX}
        commit = m6e_commit()
        if commit:
            changed = set(git('diff', '--name-only', AUTHORITY, commit).stdout.decode().split())
            self.assertEqual(changed, allowed)
        for rel in (*self.pf.WITHDRAWN_M6E_ARTIFACTS, *self.pf.GPAT_ABSENT):
            self.assertFalse(exists_at_m6e(rel), rel)

    def test_08_ledger_prefix_and_single_append(self):
        prefix, now = at_authority(LEDGER), at_m6e(LEDGER)
        self.assertEqual(hashlib.sha256(prefix).hexdigest(), self.pf.LEDGER_PREFIX_SHA)
        self.assertEqual(len(prefix.splitlines()), self.pf.LEDGER_PREFIX_ROWS)
        self.assertTrue(now.startswith(prefix))
        appended = now[len(prefix):].splitlines()
        self.assertLessEqual(len(appended), 1)
        if appended:
            row = json.loads(appended[0])
            for k, v in self.pf.LEDGER_EXPECTED.items():
                self.assertEqual(row[k], v, k)

    def test_09_static_and_history_safe(self):
        import inspect
        self.assertIn('--diff-filter=A', inspect.getsource(m6e_commit))
        own = ast.parse(Path(__file__).read_text())
        git_calls = [[ast.literal_eval(a) for a in n.args if isinstance(a, ast.Constant)] for n in ast.walk(own)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == 'git']
        self.assertFalse([c for c in git_calls if c[:1] in (['status'], ['rev-parse'], ['ls-files'])])
        src = ''.join(inspect.getsource(f) for f in (self.pf.derive, self.pf.check_rows, self.pf.check_records,
                                                      self.pf.check_evidence))
        for token in ('git(', 'worktree(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, src)
        tree_ = ast.parse((ROOT / PREFLIGHT).read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6E evidence not yet recorded')
class TestM6EEvidence(unittest.TestCase):
    def test_evidence_validates(self):
        ev = load_preflight().check_evidence(at_m6e, at_authority)
        m6 = ev['m6']
        self.assertEqual((m6['M6_CLOSED'], m6['baseline_full_scientific_execution_complete'],
                          m6['scientific_baseline_training_required_for_M6_gate'], m6['method_status_rows']),
                         (True, False, False, 11))
        self.assertEqual((ev['E06b']['final_fidelity'], ev['a9_scope'], ev['m7']['status']),
                         ('FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', ['E07c', 'E07b'], 'NOT_STARTED'))
        self.assertEqual((ev['smoke_test']['status'], ev['smoke_test']['gate'], ev['smoke_test']['smoke_test_waived']),
                         ('REQUIRED_NOT_COMPLETED', 'HARD_GATE_BEFORE_FIRST_FULL_M8_PLUS_EXECUTION', False))
        self.assertEqual(ev['known_environmental_test_errors']['count'], 4)
        self.assertEqual(ev['decision_operation']['checkpoint_writes'] + ev['decision_operation']['banks_generated'], 0)


if __name__ == '__main__':
    unittest.main()
