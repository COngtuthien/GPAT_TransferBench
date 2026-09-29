"""M6FD: E06b DSDG-NATIVE production runner (PRODUCTION_PATH_QUALIFICATION_ONLY evidence + production safety). Laptop.

GPU-measured values are checked through the M6FD preflight derivation from the GPU JSONs. The real-population checks
read only the TRAIN + CASIA/MSU filtered split metadata (allowlisted columns; no image, no TEST row materialized).
History comparisons use the commit that ADDED this test (candidate: the worktree) against the M6F-C authority.
"""
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
from methods.common.learned import PreparationError  # noqa: E402
from methods.dsdg import native  # noqa: E402
from methods.dsdg import native_runner as nr  # noqa: E402

AUTHORITY = '4b54280c53f87c391151a5436b18e5fc51db88fe'
THIS = 'tests/test_m6fd_e06b_production_qualification.py'
PREFLIGHT = 'tools/m6fd_e06b_production_qualification_preflight.py'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'
CFG = 'configs/execution/m5_gpu_3090.yaml'
ENV = {'PYTHONHASHSEED': '42', 'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6fd_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6fd(rel):
    commit = m6fd_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


def real_relation():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None
    if not (ROOT / 'manifests/split_v1.parquet').is_file():
        return None
    return nr.load_relation(nr.verify_contract())


class TestM6FDEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = load(PREFLIGHT, 'm6fd_preflight_under_test')
        cls.ev = cls.pf.check_evidence(at_m6fd)
        cls.d = cls.ev['derived']

    def test_00_label_and_no_science(self):
        e = self.ev
        self.assertEqual((e['label'], e['status'], e['qualification_seed']), ('PRODUCTION_PATH_QUALIFICATION_ONLY', 'PASS', 60801))
        self.assertEqual((e['scientific_training_completed'], e['scientific_seeds_completed'],
                          e['scientific_checkpoints_created'], e['scientific_bank_created'], e['VAL_access'],
                          e['TEST_access'], e['siw_access'], e['native_manifest_created']),
                         (False, 0, 0, False, False, False, False, False))
        self.assertEqual(e['m6'], {'M6_CLOSED': False, 'M7_started': False})

    def test_01_loader_real_relation(self):
        L = self.d['loader']
        self.assertTrue(all(L['gates'].values()))
        self.assertEqual((L['batch_sizes'], L['distinct_worker_ids']), ([240] * 15 + [120], list(range(8))))
        self.assertEqual(L['access']['denied_events'], 0)
        self.assertEqual(self.d['population']['total'], {'spoof': 3720, 'live': 1240, 'subjects': 60,
                                                         'print': 2080, 'replay': 1640})
        self.assertTrue(all(p['same_subject'] for p in L['image_pairs']))
        self.assertTrue(all(t['shape'] == [3, 256, 256] and t['dtype'] == 'float32' and 0 <= t['min'] < t['max'] <= 1
                            for p in L['image_pairs'] for t in p['tensors']))

    def test_02_batch_cases(self):
        for name, B in (('b240', 240), ('tail120', 120)):
            c = self.d['cases'][name]
            self.assertEqual((c['logical_batch_size'], c['chunks'], c['optimizer_applications'], c['backward_calls']),
                             (B, B // 20, 1, B // 20))
            self.assertTrue(all(c['gates'].values()))
            self.assertGreater(c['losses']['loss_cls']['weighted'], 0)
            self.assertGreater(c['losses']['loss_pair']['weighted'], 0)
            self.assertEqual(c['owned_gradients'], {'tensors': 55, 'non_none': 55, 'finite': 55})
            self.assertEqual(c['labels_count'], {'print_0': B // 2, 'replay_1': B // 2})
            self.assertTrue(c['independent_case'])
        self.assertEqual({self.d['cases'][n]['epoch_branch'] for n in ('b240', 'tail120')}, {1, 2})

    def test_03_cli_refusals(self):
        c = self.d['cli']
        self.assertTrue(c['all_refused'] and not c['run_dir_exists'] and not c['torch_imported'])
        self.assertEqual(c['resume'], 'NOT_QUALIFIED_FRESH_ONLY')
        self.assertEqual(set(c['refusals']), {'dirty_worktree', 'qualification_seed_60801', 'wrong_pythonhashseed',
                                              'tf32_not_disabled', 'resume_state_flag', 'qualification_limit_flag',
                                              'lambda_pair_override', 'include_siw_flag'})

    def test_04_final_fidelity_and_status(self):
        f = self.ev['fidelity']
        self.assertEqual((f['new_scientific_deviation_found'], f['runtime_fidelity_assessment'],
                          f['final_method_fidelity'], f['controlled_adaptation']),
                         (False, 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY',
                          'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', False))
        self.assertEqual(self.ev['statuses'], [
            'CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_QUALIFIED', 'EXECUTION_MAPPING_QUALIFIED',
            'WORKER_DETERMINISM_QUALIFIED', 'PRODUCTION_DATA_PATH_QUALIFIED', 'PRODUCTION_BATCH_PATH_QUALIFIED',
            'SCIENTIFIC_CLI_PREFLIGHT_QUALIFIED', 'SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED',
            'CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED', 'RESUME_UNQUALIFIED_FRESH_ONLY',
            'SCIENTIFIC_TRAINING_NOT_EXECUTED'])
        import yaml
        cfg = yaml.safe_load(at_m6fd('configs/methods/e06b_dsdg_native.yaml'))
        self.assertEqual(cfg['final_execution_fidelity'], 'PENDING_M6F_C_RUNTIME_QUALIFICATION')  # frozen, historical


class TestM6FDProductionSafety(unittest.TestCase):
    def test_05_code_bytes_pinned(self):
        pf = load(PREFLIGHT, 'pf2')
        for rel, digest in pf.CODE_SHA.items():
            self.assertEqual(hashlib.sha256(at_m6fd(rel)).hexdigest(), digest, rel)

    def test_06_modes_roots_and_refused_flags(self):
        self.assertEqual(str(nr.run_root('/rt', nr.SCIENTIFIC, 42)), '/rt/runs/m6/E06b/seed_42')
        self.assertEqual(str(nr.run_root('/rt', nr.QUALIFICATION, 60801, 'b240')), '/rt/qualification/m6fd/E06b/b240')
        for mode, seed in ((nr.SCIENTIFIC, 60801), (nr.QUALIFICATION, 42), (nr.SCIENTIFIC, 7)):
            with self.assertRaises(PreparationError):
                nr.validate_mode_seed(mode, seed)
        self.assertEqual(nr.refused_arguments(['--resume-state', 'x', '--lambda-pair=0', '--include-siw', '--seed', '42',
                                               '--qualification-max-logical-batches', '1', '--workers', '4']),
                         ['--include-siw', '--lambda-pair', '--qualification-max-logical-batches', '--resume-state',
                          '--workers'])

    def test_07_contract_gates(self):
        cfg = nr.verify_contract()
        self.assertEqual(cfg['_runtime']['config_sha256'], nr.CONFIG_SHA)
        ids = nr.identities(cfg)
        self.assertEqual((ids['lightcnn_sha256'], ids['environment_lock_sha256'], ids['runtime_fidelity_assessment']),
                         (nr.LIGHTCNN_SHA, nr.LOCK_SHA, 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY'))
        bad = copy.deepcopy(cfg)
        bad['losses']['lambda_pair'] = 0
        with self.assertRaises(PreparationError):
            nr.verify_contract(bad)
        bad = copy.deepcopy(cfg)
        bad['_runtime']['config_sha256'] = '0' * 64
        with self.assertRaises(PreparationError):
            nr.verify_contract(bad)

    def test_08_cli_refuses_before_any_import(self):
        cli = load('tools/run_e06b.py', 'run_e06b_under_test')
        cases = ((['--seed', '42', '--execution-config', CFG], ENV, True),
                 (['--seed', '60801', '--execution-config', CFG], dict(ENV, PYTHONHASHSEED='60801'), False),
                 (['--seed', '42', '--execution-config', CFG], dict(ENV, PYTHONHASHSEED='1'), False),
                 (['--seed', '42', '--execution-config', CFG], dict(ENV, NVIDIA_TF32_OVERRIDE='1'), False))
        for argv, env, dirty in cases:
            with self.assertRaises(SystemExit) as cm:
                cli.preconditions(cli.parse(argv), environ=env, dirty=dirty)
            self.assertEqual(cm.exception.code, 2)
        for extra in (['--resume-state', 'y'], ['--lambda-pair', '0'], ['--include-siw'],
                      ['--qualification-only'], ['--epochs', '1']):
            with self.assertRaises(SystemExit):
                cli.parse(['--seed', '42', '--execution-config', CFG] + extra)
        self.assertNotIn('torch', sys.modules)

    def test_09_trainer_uses_native_gates_not_e06c(self):
        src = at_m6fd('methods/dsdg/native_runner.py').decode()
        self.assertNotIn("losses['loss_pair'] == 0.0", src)
        self.assertNotIn('frozen_lambdas', src)
        self.assertIn('v2.run_global_batch_v2(', src)
        self.assertIn("'native CE index in {0 print, 1 replay} equals the relation class'", src)
        self.assertIn('fresh only', src)

    def test_10_real_population_and_row_assertions(self):
        got = real_relation()
        if got is None:
            self.skipTest('split metadata / pyarrow unavailable')
        rows, rel, sample_datasets, pop = got
        self.assertEqual(pop['total']['spoof'], 3720)
        self.assertEqual(len(nr.assert_native_rows(rel, rows, range(len(rel)))), 3720)
        self.assertEqual(set(sample_datasets.values()), {'casia_fasd', 'msu_mfsd'})
        tampered = [dict(r) for r in rows]
        victim = next(r for r in tampered if r['label_binary'] == 0)
        victim['subject_id_global'] = victim['dataset'] + '::999999'
        rel2 = native.NativeRelation(tampered)
        idx = next(i for i, s in enumerate(rel.spoof) if victim['sample_id'] in rel.pools[s[1]])
        self.assertNotIn(victim['sample_id'], rel2.pools[rel.spoof[idx][1]])

    def test_11_qualification_subsets_deterministic_disjoint(self):
        got = real_relation()
        if got is None:
            self.skipTest('split metadata / pyarrow unavailable')
        rows, rel, _, _ = got
        q = load('methods/dsdg/native_runner_qualification.py', 'm6fd_harness_under_test')
        a, ca = q.subset_relation(rel, rows, *q.CASES['b240'][:2])
        b, cb = q.subset_relation(rel, rows, *q.CASES['tail120'][:2])
        self.assertEqual((len(a), len(b)), (240, 120))
        self.assertFalse(set(ca) & set(cb))
        self.assertEqual((a.summary()['total']['print'], a.summary()['total']['replay']), (120, 120))
        self.assertEqual(ca, q.subset_relation(rel, rows, *q.CASES['b240'][:2])[1])

    def test_12_history_frozen_and_e06c_unchanged(self):
        for rel in ('configs/methods/e06b_dsdg_native.yaml', 'frozen_config_snapshot/configs/methods/e06b_dsdg_native.yaml',
                    'methods/dsdg/native.py', 'methods/dsdg/native_qualification.py', 'methods/dsdg/runner.py',
                    'methods/dsdg/runner_io.py', 'methods/dsdg/runner_qualification.py', 'tools/run_e06c.py',
                    'methods/common/config.py', 'methods/common/learned.py', 'environments/e06c.lock.json',
                    'outputs/audit/M6FC_E06B_GPU_QUALIFICATION.json',
                    'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'):
            self.assertEqual(at_m6fd(rel), at_authority(rel), rel)
        pf = load(PREFLIGHT, 'pf3')
        prefix, now = at_authority(LEDGER), at_m6fd(LEDGER)
        self.assertEqual(hashlib.sha256(prefix).hexdigest(), pf.LEDGER_PREFIX_SHA)
        self.assertTrue(now.startswith(prefix))
        appended = now[len(prefix):].splitlines()
        self.assertLessEqual(len(appended), 1)
        if appended:
            row = json.loads(appended[0])
            for k, v in pf.LEDGER_EXPECTED.items():
                self.assertEqual(row[k], v, k)


if __name__ == '__main__':
    unittest.main()
