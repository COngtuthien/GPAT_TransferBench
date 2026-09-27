"""M6D6iR: E07c MAIN checkpoint-retention OWNER DECISION (DETERMINISTIC_IMPLEMENTATION_CLARIFICATION).

Static only: no Torch, no data, no checkpoint. History assertions compare the state at the commit that ADDED this
test (candidate: the worktree) with the M6D6i authority, so later milestones never break them (no B1 HEAD lock).
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = 'a433b28f861631b1e7828ff247c32f3e3d37bcca'
THIS = 'tests/test_m6d6ir_e07c_checkpoint_retention.py'
RECORD = 'configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml'
DOC = 'docs/spec/amendments/GPAT_TransferBench_v1_0_E07c_Checkpoint_Retention_Decision_M6D6iR.md'
EVIDENCE = ROOT / 'outputs/audit/M6D6IR_E07C_CHECKPOINT_RETENTION_DECISION.json'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6d6ir_preflight_under_test',
                                                  ROOT / 'tools/m6d6ir_e07c_checkpoint_retention_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6d6ir_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6d6ir(rel):
    """File bytes at the M6D6iR state: the commit that added this test, else the (candidate) worktree."""
    commit = m6d6ir_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM6D6iRDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.raw = at_m6d6ir(RECORD)
        cls.r = json.loads(cls.raw)
        cls.d = cls.pf.derive(at_m6d6ir)

    def rejects(self, mutate):
        """check_record must refuse a mutated record (reader serves the mutation with a matching pinned hash)."""
        bad = copy.deepcopy(self.r)
        mutate(bad)
        raw = json.dumps(bad, indent=2).encode() + b'\n'
        pf = load_preflight()
        pf.RECORD_SHA = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):
            pf.check_record(lambda rel: raw if rel == RECORD else at_m6d6ir(rel))

    # 1-10 identity ---------------------------------------------------------------------------
    def test_01_authority_is_ancestor(self):
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)
        self.assertEqual(self.r['authority_commit'], AUTHORITY)

    def test_02_record_exact_hash(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), self.pf.RECORD_SHA)
        self.pf.check_record(at_m6d6ir)

    def test_03_document_exact_hash(self):
        self.assertEqual(self.r['record_document'],
                         {'path': DOC, 'sha256': hashlib.sha256(at_m6d6ir(DOC)).hexdigest()})

    def test_04_bound_authority_hashes(self):
        self.assertEqual(sorted(self.r['bound_authority_sha256']), sorted(self.pf.BOUND))
        for rel, digest in self.r['bound_authority_sha256'].items():
            if rel != self.pf.TRAIN_SCRIPT:
                self.assertEqual(hashlib.sha256(at_authority(rel)).hexdigest(), digest, rel)
        self.assertEqual(self.r['bound_authority_sha256']['docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx'],
                         'f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e')
        self.assertEqual(self.r['source']['commit'], '23f40519ec25a833ebc06842aa6fbab74fad4d15')

    def test_05_e07c_main_only_scope(self):
        s = self.r['scope']
        self.assertEqual((s['applies_to'], s['experiment_seeds'], s['run_root_template']),
                         ('E07c main scientific runs', [42, 1337, 2026], '<runtime_root>/runs/m6/E07c/seed_<seed>/'))
        self.rejects(lambda b: b['scope'].update(experiment_seeds=[42, 1337, 2026, 60607]))

    def test_06_no_a9(self):
        self.assertEqual((self.r['amendment_created'], self.r['a9_created']), (False, False))
        self.assertIn('AMENDMENT_A9', self.r['not_classified_as'])
        self.assertFalse([p for p in git('ls-files', 'docs', 'configs').stdout.decode().split() if 'A9' in p])

    def test_07_classification(self):
        self.assertEqual((self.r['classification'], self.r['record_kind'], self.r['milestone']),
                         ('DETERMINISTIC_IMPLEMENTATION_CLARIFICATION', 'OWNER_DECISION / CHECKPOINT_RETENTION', 'M6D6iR'))
        self.rejects(lambda b: b.update(classification='SCIENTIFIC_ADAPTATION'))

    def test_08_fidelity_unchanged(self):
        self.assertEqual((self.r['fidelity_class'], self.r['deviation']), ('CONTROLLED_ADAPTATION', 'DEV-021'))

    def test_09_new_deviation_false(self):
        self.assertIs(self.r['new_deviation'], False)
        self.rejects(lambda b: b.update(new_deviation=True))

    def test_10_new_fidelity_class_false(self):
        self.assertIs(self.r['new_fidelity_class'], False)

    # 11-19 unchanged execution facts, re-derived from source/authority ------------------------------
    def test_11_cadence_unchanged(self):
        self.assertEqual((self.d['cadence_iterations'], self.d['pinned_defaults']['save_checkpoints_every_iters']),
                         (10000, 10000))
        self.assertEqual(self.d['source_lines'], {'iters_init': 30, 'epoch_loop': 37, 'iters_increment': 42,
                                                  'save_test': 102, 'torch_save': 105, 'dataloader': 205})
        self.assertFalse(self.d['save_at_global_step_0'])
        self.rejects(lambda b: b['checkpoint_creation'].update(cadence_iterations=20000))

    def test_12_payload_unchanged(self):
        self.assertEqual(self.d['payload_keys'], ['model', 'ema', 'scheduler', 'optimizer', 'conf'])
        self.assertEqual(self.r['payload']['keys'], self.d['payload_keys'])

    def test_13_epochs_unchanged(self):
        self.assertEqual((self.d['epochs'], self.d['batch_size'], self.d['drop_last']), (400, 4, False))

    def test_14_seeds_unchanged(self):
        self.assertEqual(self.d['experiment_seeds'], [42, 1337, 2026])

    def test_15_iterations_per_epoch(self):
        self.assertEqual((self.d['train_rows'], math.ceil(8838 / 4), self.d['iterations_per_epoch'],
                          self.d['last_batch_size']), (8838, 2210, 2210, 2))

    def test_16_total_iterations(self):
        self.assertEqual((2210 * 400, self.d['total_iterations']), (884000, 884000))

    def test_17_periodic_count(self):
        self.assertEqual((self.d['periodic_checkpoint_count'], self.d['periodic_global_steps']),
                         (88, {'first': 10000, 'last': 880000, 'step': 10000}))

    def test_18_terminal(self):
        self.assertEqual((self.d['terminal_global_step'], self.d['terminal_remainder'], self.d['terminal_checkpoint_count']),
                         (884000, 4000, 1))

    def test_19_total_created_states(self):
        self.assertEqual(self.d['created_checkpoint_count'], 89)
        for k in ('periodic_checkpoint_count', 'terminal_checkpoint_count', 'created_checkpoint_count', 'total_iterations'):
            self.assertEqual(self.r['checkpoint_creation'][k], self.d[k])
        self.rejects(lambda b: b['checkpoint_creation'].update(created_checkpoint_count=88))

    # 20-34 decision semantics ----------------------------------------------------------------
    def test_20_successor_prune_gate(self):
        p = self.r['pruning']
        self.assertEqual(p['successor_gate'], self.pf.GATE)
        self.assertEqual(len(p['successor_gate']), 9)
        self.assertEqual((p['successor_gate_all_required'], p['gated_on']), (True, 'direct successor C_{k+1}'))
        self.assertEqual(p['terminal_transition']['last_periodic_global_step'], 880000)
        self.rejects(lambda b: b['pruning']['successor_gate'].pop())

    def test_21_terminal_protected(self):
        self.assertEqual(self.r['protected']['pruning'], 'FORBIDDEN')
        self.assertIn('884000', self.r['protected']['resolve_to'])
        self.rejects(lambda b: b['protected'].update(pruning='ALLOWED'))

    def test_22_final_selected_terminal_one_physical_state(self):
        pt = self.r['protected']
        self.assertEqual(set(pt['roles']), {'authoritative_final_checkpoint', 'selected_checkpoint',
                                            'officially_required_checkpoint', 'terminal'})
        self.assertEqual((pt['one_physical_file_may_satisfy_all_roles'], pt['duplicate_byte_copies_required']), (True, False))
        self.assertEqual((self.r['selection']['final_checkpoint'], self.r['selection']['selected_checkpoint']),
                         ('terminal', 'terminal'))

    def test_23_metadata_survives_pruning(self):
        m = self.r['metadata_retention']
        self.assertTrue(m['records_never_removed'])
        self.assertEqual(m['required_fields_unchanged'], self.d['index_required_fields'])
        self.assertEqual({k: m['pruned_fields'][k] for k in ('bytes_present', 'bytes_pruned', 'prune_reason')},
                         {'bytes_present': False, 'bytes_pruned': True,
                          'prune_reason': 'EXPLICIT_E07C_PERIODIC_RETENTION_POLICY'})
        self.rejects(lambda b: b['metadata_retention'].update(records_never_removed=False))

    def test_24_silent_pruning_forbidden(self):
        self.assertEqual(self.r['silent_pruning'], 'FORBIDDEN')
        self.assertIn('  silent_pruning: FORBIDDEN\n', at_authority('configs/run_logging_v1.yaml').decode())
        for f in ('glob deletion', 'background deletion', 'unlogged cleanup', 'removal of unknown paths'):
            self.assertIn(f, self.r['pruning']['forbidden'])

    def test_25_prune_before_verification_forbidden(self):
        p = self.r['pruning']
        self.assertEqual((p['prune_before_successor_verification'], p['prune_immediately_after_own_save']),
                         ('FORBIDDEN', 'FORBIDDEN'))
        f = p['failure_policy']
        self.assertEqual({f[k] for k in f if k.startswith('successor_')}, {'KEEP_PREDECESSOR'})
        self.assertTrue(f['prune_failure'].startswith('STOP_AND_REPORT'))
        self.assertTrue(f['insufficient_space_for_successor'].startswith('STOP_AND_REPORT'))
        self.assertIn('early predecessor deletion to make room', p['forbidden'])

    def test_26_no_skip_save_authority(self):
        c = self.r['checkpoint_creation']
        self.assertEqual((c['skip_save_authorized'], c['cadence_change_authorized']), (False, False))
        self.rejects(lambda b: b['checkpoint_creation'].update(skip_save_authorized=True))

    def test_27_no_payload_rewrite_authority(self):
        self.assertFalse(self.r['payload']['payload_rewrite_authorized'] or self.r['payload']['optimizer_strip_authorized'])
        self.rejects(lambda b: b['payload'].update(optimizer_strip_authorized=True))

    def test_28_no_weights_only_authority(self):
        self.assertFalse(self.r['payload']['weights_only_authorized'])

    def test_29_no_dtype_change(self):
        self.assertFalse(self.r['payload']['dtype_change_authorized'])

    def test_30_no_compression_authority(self):
        self.assertFalse(self.r['payload']['compression_authorized'])
        self.rejects(lambda b: b['payload'].update(compression_authorized=True))

    def test_31_no_other_methods(self):
        ex = set(self.r['scope']['excludes'])
        self.assertTrue({'E01', 'E02', 'E03', 'E04', 'E05', 'E06c', 'M7', 'M8', 'any other method'} <= ex)

    def test_32_no_aux_encoder(self):
        ex = set(self.r['scope']['excludes'])
        self.assertTrue({'M6D6g auxiliary encoder run', 'encoder_final.pkl', 'auxiliary resume sidecars',
                         'qualification artifacts'} <= ex)
        self.rejects(lambda b: b['scope']['excludes'].remove('encoder_final.pkl'))

    def test_33_main_checkpoint_resume_unqualified(self):
        self.assertEqual(self.r['resume']['MAIN_CHECKPOINT_RESUME'], 'UNQUALIFIED')
        self.assertIs(self.r['resume']['pruning_proves_resume'], False)
        self.assertIn('MAIN_CHECKPOINT_RESUME', self.r['statuses']['not_qualified'])

    def test_34_production_runner_unqualified(self):
        st = self.r['statuses']
        self.assertEqual(st['qualified'], ['E07c_MAIN_CHECKPOINT_RETENTION_POLICY_FROZEN',
                                           'MAIN_CHECKPOINT_RETENTION_POLICY_FROZEN'])
        self.assertEqual(st['not_qualified'], ['MAIN_PRODUCTION_RUNNER', 'MAIN_CHECKPOINT_RESUME',
                                               'MAIN_DIFFFAS_SCIENTIFIC_TRAINING', 'M8_BANK'])
        self.assertEqual(st['method_status'], 'IMPLEMENTED_NOT_EXECUTED')
        self.assertEqual(self.r['future_implementation']['milestone'], 'M6D6j')

    # 35-38 activity / history --------------------------------------------------------------
    def test_35_no_runtime_or_scientific_activity(self):
        self.assertEqual(self.r['decision_operation'], self.pf.ZERO_OPERATION)
        self.assertEqual(set(self.r['science_unchanged'].values()), {False})
        self.assertIs(self.r['storage_rationale']['binding'], False)
        self.assertEqual(self.r['storage_rationale']['label'], 'RESOURCE_PLANNING_ESTIMATE')

    def test_36_previous_evidence_and_code_byte_identical(self):
        paths = [p for p in git('ls-tree', '-r', '--name-only', AUTHORITY, 'outputs/audit').stdout.decode().splitlines()
                 if Path(p).name.startswith(tuple(f'M6D6{c}_' for c in 'ABCDEFGHI'))]
        self.assertGreaterEqual(len(paths), 45)
        for rel in (*paths, *self.pf.PROTECTED):
            self.assertEqual(at_m6d6ir(rel), at_authority(rel), rel)

    def test_37_ledger_prefix_integrity(self):
        self.assertTrue(at_m6d6ir('outputs/audit/EXECUTION_LEDGER.jsonl').startswith(
            at_authority('outputs/audit/EXECUTION_LEDGER.jsonl')))
        self.assertEqual(len(at_authority('outputs/audit/EXECUTION_LEDGER.jsonl').splitlines()), 124)

    def test_38_history_check_future_safe(self):
        import inspect
        self.assertIn('--diff-filter=A', inspect.getsource(m6d6ir_commit))
        own = ast.parse(Path(__file__).read_text())
        git_calls = [[ast.literal_eval(a) for a in n.args if isinstance(a, ast.Constant)] for n in ast.walk(own)
                     if isinstance(n, ast.Call) and ast.unparse(n.func) == 'git']
        self.assertFalse([c for c in git_calls if c[:1] in (['status'], ['diff'], ['rev-parse'])])
        src = inspect.getsource(self.pf.check_record) + inspect.getsource(self.pf.check_evidence) + \
            inspect.getsource(self.pf.derive)
        for token in ('git(', 'worktree(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, src)

    def test_39_preflight_is_static(self):
        tree_ = ast.parse((ROOT / 'tools/m6d6ir_e07c_checkpoint_retention_preflight.py').read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        with self.assertRaises(ValueError):
            self.pf.read('runs/m6/E07c/seed_42/checkpoints/model_010000.pt')
        self.assertEqual('torch' in sys.modules, self.torch_before)


@unittest.skipUnless(EVIDENCE.is_file(), 'M6D6iR evidence not yet recorded')
class TestM6D6iREvidence(unittest.TestCase):
    def test_evidence_validates(self):
        load_preflight().check_evidence(at_m6d6ir)

    def test_evidence_zero_activity_and_statuses(self):
        ev = json.loads(at_m6d6ir('outputs/audit/M6D6IR_E07C_CHECKPOINT_RETENTION_DECISION.json'))
        self.assertEqual(ev['decision_operation']['checkpoint_deletions'], 0)
        self.assertEqual(ev['static_derivation']['created_checkpoint_count'], 89)
        self.assertFalse(ev['m6d6i_statuses_requalified'])


if __name__ == '__main__':
    unittest.main()
