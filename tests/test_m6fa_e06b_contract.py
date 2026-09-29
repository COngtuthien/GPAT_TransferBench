"""M6FA: E06b DSDG-NATIVE contract resolution (authority-derived; no implementation, no training). Static only.

History assertions compare the state at the commit that ADDED this test (candidate: the worktree) with the M6A9
authority, so later milestones never break them (no B1 HEAD lock). The pinned DSDG source cache is git-ignored and is
read from disk, anchored by SHA256.
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
AUTHORITY = '8e9ccc371f6c53b75ec6b473d2306aefa9aa672b'
THIS = 'tests/test_m6fa_e06b_contract.py'
PREFLIGHT = 'tools/m6fa_e06b_contract_preflight.py'
CONTRACT = 'outputs/audit/M6FA_E06B_CONTRACT_RESOLUTION.json'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
SOURCE_CACHE = ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG'


def load_preflight():
    spec = importlib.util.spec_from_file_location('m6fa_preflight_under_test', ROOT / PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6fa_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6fa(rel):
    """File bytes at the M6FA state: the commit that added this test, else the (candidate) worktree."""
    commit = m6fa_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def exists_at_m6fa(rel):
    commit = m6fa_commit()
    return git('cat-file', '-e', f'{commit}:{rel}').returncode == 0 if commit else (ROOT / rel).exists()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


@unittest.skipUnless(SOURCE_CACHE.is_dir(), 'pinned FaceX-Zoo/DSDG source cache not present on this host')
class TestM6FAContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_before = 'torch' in sys.modules
        cls.pf = load_preflight()
        cls.raw = at_m6fa(CONTRACT)
        cls.c, cls.d = cls.pf.check_contract(at_m6fa)

    def rejects(self, mutate):
        bad = copy.deepcopy(self.c)
        mutate(bad)
        raw = json.dumps(bad, indent=2, ensure_ascii=False).encode() + b'\n'
        pf = load_preflight()
        pf.CONTRACT_SHA = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):
            pf.check_contract(lambda rel: raw if rel == CONTRACT else at_m6fa(rel))

    def test_00_contract_hash_and_authority(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), self.pf.CONTRACT_SHA)
        self.assertEqual(git('merge-base', '--is-ancestor', AUTHORITY, 'HEAD').returncode, 0)
        self.assertEqual((self.c['method_id'], self.c['method_name'], self.c['track']),
                         ('E06b', 'DSDG-NATIVE', 'B_NATIVE_FULL_SECONDARY'))

    def test_01_dataset_scope_casia_msu_siw_not_instantiable(self):
        cov = self.c['dataset_coverage']
        self.assertEqual(cov['train_datasets'], ['casia_fasd', 'msu_mfsd'])
        self.assertEqual(cov['non_instantiable']['siwmv2']['status'], 'NOT_INSTANTIABLE_MISSING_SUBJECT_ID')
        self.assertEqual(cov['totals']['train_spoof_frames'], 3720)
        self.assertEqual(cov['spoof_frames_without_same_subject_live'], 0)
        self.rejects(lambda b: b['dataset_coverage']['train_datasets'].append('siwmv2'))
        self.rejects(lambda b: b['dataset_coverage']['per_dataset']['casia_fasd'].update(train_spoof_frames=2521))

    def test_02_k2_attack_macro_data_derived(self):
        st = self.c['spoof_type_supervision']
        self.assertEqual((st['target'], st['vocabulary'], st['K'], st['class_index']),
                         ('attack_macro', ['print', 'replay'], 2, {'print': 0, 'replay': 1}))
        self.assertEqual(self.c['dataset_coverage']['totals']['spoof_by_macro'], {'print': 2080, 'replay': 1640})
        self.rejects(lambda b: b['spoof_type_supervision'].update(K=4))
        self.rejects(lambda b: b['spoof_type_supervision'].update(vocabulary=['print', 'replay', 'mask_3d']))
        self.rejects(lambda b: b['spoof_type_supervision'].update(class_index={'print': 1, 'replay': 0}))

    def test_03_lambda_pair_is_the_official_executed_value(self):
        self.assertEqual(self.c['losses']['lambda_pair'], 5)
        self.assertEqual(self.d['shell']['lambda_pair'], (17, '5'))
        self.assertEqual(self.c['field_provenance']['lambda_pair'], 'OFFICIAL_CODE train_generator.sh:17')
        self.rejects(lambda b: b['losses'].update(lambda_pair=0.5))
        self.rejects(lambda b: b['losses'].update(lambda_pair=0.0))

    def test_04_official_training_values(self):
        t = self.c['training']
        self.assertEqual((t['all_epochs'], t['effective_batch_size'], t['learning_rate'], t['hdim']), (200, 240, 2e-4, 128))
        self.assertEqual(self.c['checkpoint']['rule'], 'OFFICIAL_GENERATOR_EPOCH_200')
        self.assertEqual(self.c['losses']['removed_terms'], [])
        self.assertTrue(self.c['losses']['loss_cls_active'] and self.c['losses']['loss_pair_active'])
        self.rejects(lambda b: b['training'].update(all_epochs=100))
        self.rejects(lambda b: b['training'].update(learning_rate=1e-4))

    def test_05_native_relation_online_no_manifest(self):
        rel = self.c['native_relation']
        self.assertEqual((rel['semantics'], rel['materialized_pair_list'], rel['native_manifest_created']),
                         ('SAME_SUBJECT_ONLINE_RANDOM', False, False))
        self.assertFalse(exists_at_m6fa('manifests/dsdg_identity_pairs_v1.parquet'))
        self.rejects(lambda b: b['native_relation'].update(materialized_pair_list=True))

    def test_06_worker_randomness_contract(self):
        rc = self.c['randomness_contract']
        self.assertEqual((rc['num_workers_frozen'], rc['num_workers_affects_results'], rc['loader']['batch_size'],
                          rc['loader']['persistent_workers']), (8, True, 240, False))
        self.rejects(lambda b: b['randomness_contract'].update(num_workers_frozen=4))
        self.rejects(lambda b: b['randomness_contract']['loader'].update(num_workers=4))

    def test_07_microbatch_execution_compatibility(self):
        x, ep = self.c['execution_compatibility'], self.c['epoch_plan']
        self.assertEqual((ep['global_batches'], ep['tail_batch'], ep['total_optimizer_steps']), ([240] * 15 + [120], 120, 3200))
        self.assertEqual((x['microbatch'], x['tail_120_chunks'], x['classification'], x['scientific_method_change']),
                         (20, [20] * 6, 'EXECUTION_RUNTIME_COMPATIBILITY', False))
        self.rejects(lambda b: b['epoch_plan'].update(tail_batch=198))
        self.rejects(lambda b: b['execution_compatibility'].update(scientific_method_change=True))

    def test_08_zero_unresolved_and_m8_deferral(self):
        self.assertEqual(self.c['unresolved_scientific_fields'], [])
        m8 = self.c['deferred_non_m6']['M8_track_b_generation_budget']
        self.assertEqual((m8['status'], m8['blocks_m6']), ('GENUINELY_AMBIGUOUS_DEFERRED_TO_M8', False))
        self.rejects(lambda b: b.update(unresolved_scientific_fields=['K']))

    def test_09_fidelity_target_not_assessed(self):
        f = self.c['fidelity']
        self.assertEqual((f['target_fidelity'], f['fidelity_assessed'], f['new_deviation']), ('FAITHFUL_OFFICIAL', False, False))
        self.rejects(lambda b: b['fidelity'].update(fidelity_assessed=True))

    def test_10_nothing_frozen_or_executed_m6_open(self):
        self.assertFalse(exists_at_m6fa('configs/methods/e06b_dsdg_native.yaml'))
        self.assertFalse(exists_at_m6fa('outputs/audit/method_status.csv'))
        self.assertEqual(self.c['m6'], {'M6_CLOSED': False, 'M7_started': False})
        op = self.c['decision_operation']
        self.assertEqual((op['training_runs'], op['optimizer_steps'], op['GPU_contacted'], op['image_reads'],
                          op['TEST_rows_materialized']), (0, 0, False, 0, 0))
        self.rejects(lambda b: b['m6'].update(M6_CLOSED=True))

    def test_11_history_ledger_prefix_and_append_only_status(self):
        prefix, now = at_authority(LEDGER), at_m6fa(LEDGER)
        self.assertEqual(hashlib.sha256(prefix).hexdigest(), self.pf.LEDGER_PREFIX_SHA)
        self.assertTrue(now.startswith(prefix))
        appended = now[len(prefix):].splitlines()
        self.assertLessEqual(len(appended), 1)
        if appended:
            row = json.loads(appended[0])
            for k, v in self.pf.LEDGER_EXPECTED.items():
                self.assertEqual(row[k], v, k)
        base, cur = at_authority('configs/CONFIG_STATUS.md'), at_m6fa('configs/CONFIG_STATUS.md')
        self.assertTrue(cur.startswith(base) and len(cur) > len(base))
        commit = m6fa_commit()
        if commit:
            changed = set(git('diff', '--name-only', AUTHORITY, commit).stdout.decode().split())
            self.assertEqual(changed - set(self.pf.NEW), {'configs/CONFIG_STATUS.md', LEDGER,
                                                          'outputs/audit/ARTIFACT_INDEX.csv'})

    def test_12_preflight_static(self):
        import inspect
        self.assertIn('--diff-filter=A', inspect.getsource(m6fa_commit))
        src = inspect.getsource(self.pf.check_contract) + inspect.getsource(self.pf.derive)
        for token in ('git(', 'worktree(', 'authority()', "'HEAD'"):
            self.assertNotIn(token, src)
        tree_ = ast.parse((ROOT / PREFLIGHT).read_text())
        imported = {a.name.split('.')[0] for n in ast.walk(tree_) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'yaml', 'pyarrow', 'numpy', 'PIL'} & imported)
        self.assertEqual('torch' in sys.modules, self.torch_before)


if __name__ == '__main__':
    unittest.main()
