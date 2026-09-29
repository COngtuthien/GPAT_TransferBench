"""M6FC: E06b DSDG-NATIVE GPU synthetic qualification evidence + harness safety. Static on the laptop.

The GPU-measured values are checked through the M6FC preflight derivation from the four GPU JSONs. History
comparisons use the commit that ADDED this test (candidate: the worktree) against the M6F-B authority.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUTHORITY = '91d28a72f78d8f424da23cf07a015782e532ba0e'
THIS = 'tests/test_m6fc_e06b_gpu_qualification.py'
PREFLIGHT = 'tools/m6fc_e06b_gpu_qualification_preflight.py'
HARNESS = 'methods/dsdg/native_qualification.py'
LEDGER = 'outputs/audit/EXECUTION_LEDGER.jsonl'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)


def m6fc_commit():
    return git('log', '--diff-filter=A', '--format=%H', '-1', '--', THIS).stdout.decode().strip() or None


def at_m6fc(rel):
    commit = m6fc_commit()
    return git('show', f'{commit}:{rel}').stdout if commit else (ROOT / rel).read_bytes()


def at_authority(rel):
    return git('show', f'{AUTHORITY}:{rel}').stdout


class TestM6FCEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf = load(PREFLIGHT, 'm6fc_preflight_under_test')
        cls.ev = cls.pf.check_evidence(at_m6fc)
        cls.d = cls.ev['derived']

    def test_00_identity_and_labels(self):
        self.assertEqual((self.ev['label'], self.ev['scientific'], self.ev['qualification_seed'], self.ev['status']),
                         ('QUALIFICATION_ONLY_SYNTHETIC', 'NON_SCIENTIFIC', 60701, 'PASS'))
        self.assertNotIn(self.ev['qualification_seed'], (42, 1337, 2026))
        self.assertEqual(self.ev['gpu_authority_commit'], AUTHORITY)

    def test_01_live_graph_contract(self):
        self.assertEqual(self.d['graph']['netCls_live'], {'bias_shape': [2], 'in_features': 128, 'out_features': 2,
                                                          'weight_shape': [2, 128]})
        self.assertEqual(self.d['graph']['parameters_initial']['netCls'], {'parameters': 258, 'parameter_tensors': 2})
        self.assertEqual(self.d['graph']['lambdas'], {'lambda_ip': 1000, 'lambda_mmd': 50, 'lambda_ort': 1,
                                                      'lambda_pair': 5, 'lambda_type': 10})
        self.assertEqual(self.d['lightcnn']['sha256'], 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964')
        self.assertEqual(self.d['graph']['batch_coupling']['batchnorm_modules'], 0)

    def test_02_native_terms_active(self):
        ti = self.d['term_isolation']
        self.assertTrue(all(ti['gates'].values()))
        self.assertGreater(ti['terms']['loss_cls']['value'], 0)
        self.assertGreater(ti['terms']['loss_pair']['value'], 0)
        self.assertEqual(ti['terms']['loss_cls']['netCls_grad_nonzero_elements'], 258)
        for c in self.d['reference_cases']:
            self.assertEqual(sorted(set(c['labels'])), [0, 1])

    def test_03_equivalence_within_pre_registered_gates(self):
        g, w = self.ev['pre_registered_gates'], self.ev['worst_case_reference']
        self.assertEqual(g, json.loads((ROOT / 'configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml')
                                       .read_text())['qualification']['reference_check']['gates'])
        self.assertLessEqual(w['total_relative_error'], g['total_loss_relative_error_max'])
        self.assertLessEqual(w['max_term_relative_error'], self.ev['per_term_relative_error_max'])
        self.assertGreaterEqual(w['gradient_cosine'], g['aggregate_gradient_cosine_min'])
        self.assertLessEqual(w['gradient_relative_l2'], g['aggregate_gradient_relative_l2_max'])
        self.assertGreaterEqual(w['update_cosine'], g['post_step_parameter_delta_cosine_min'])
        self.assertEqual([c['chunk_sizes'] for c in self.d['reference_cases']], [[2, 2], [4, 2], [2, 2]])
        self.assertEqual({c['epoch'] for c in self.d['reference_cases']}, {1, 2})

    def test_04_logical_240_and_tail_120(self):
        for name, B, chunks in (('b240', 240, 12), ('tail120', 120, 6)):
            r = self.d['logical'][name]
            self.assertEqual((r['logical_batch_size'], r['physical_microbatch'], r['chunks'], r['chunk_sizes']),
                             (B, 20, chunks, [20] * chunks))
            self.assertEqual((r['optimizer_applications'], r['backward_calls'], r['oom']), (1, chunks, False))
            self.assertTrue(all(r['gates'].values()))
            self.assertGreater(r['peak_allocated_bytes'], 0)
            self.assertLess(r['peak_reserved_bytes'], 24 * 1024 ** 3)
            self.assertEqual(min(r['labels_count'].values()), B // 2)

    def test_05_worker_determinism(self):
        w = self.d['workers']
        self.assertTrue(all(w['gates'].values()))
        self.assertEqual(w['loader_contract']['num_workers'], 8)
        self.assertEqual(w['relation_summary']['total'], {'spoof': 3720, 'live': 1240, 'subjects': 60,
                                                          'print': 2080, 'replay': 1640})
        self.assertEqual(len(w['epochs']), 2)

    def test_06_fidelity_decision_and_status(self):
        f = self.ev['fidelity']
        self.assertEqual((f['target_source_fidelity'], f['execution_classification'], f['runtime_fidelity_assessment'],
                          f['final_method_fidelity'], f['controlled_adaptation']),
                         ('FAITHFUL_OFFICIAL', 'EXECUTION_RUNTIME_COMPATIBILITY',
                          'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'PENDING_M6F_D_PRODUCTION_QUALIFICATION', False))
        self.assertEqual(self.ev['statuses'], ['CONFIG_FROZEN', 'STATIC_ADAPTER_IMPLEMENTED', 'GPU_GRAPH_QUALIFIED',
                                               'EXECUTION_MAPPING_QUALIFIED', 'WORKER_DETERMINISM_QUALIFIED',
                                               'PRODUCTION_RUNNER_NOT_YET_QUALIFIED', 'SCIENTIFIC_TRAINING_NOT_EXECUTED'])
        self.assertIs(self.ev['scientific_training_completed'], False)
        self.assertEqual(self.ev['m6'], {'M6_CLOSED': False, 'M7_started': False})
        a = self.ev['aborted_attempt']
        self.assertEqual((a['status'], a['classification'], a['e06b_model_or_loader_failure']),
                         ('STOP_WORKER_GATE', 'QUALIFICATION_HARNESS_ISSUE_ONLY', False))
        import yaml
        cfg = yaml.safe_load(at_m6fc('configs/methods/e06b_dsdg_native.yaml'))
        self.assertEqual(cfg['final_execution_fidelity'], 'PENDING_M6F_C_RUNTIME_QUALIFICATION')  # frozen, not rewritten


class TestM6FCHarnessAndHistory(unittest.TestCase):
    def test_07_harness_static_safety(self):
        src = at_m6fc(HARNESS).decode()
        tree = ast.parse(src)
        imported = {a.name.split('.')[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [ast.alias(n.module or '')])}
        self.assertFalse({'torch', 'numpy', 'yaml'} & imported, 'torch imported only inside the audited process')
        for token in ('frozen_lambdas', 'FROZEN_LAMBDAS', 'execution_guard(', 'torch.save(', 'faces_256', 'split_v1'):
            self.assertNotIn(token, src)
        for token in ("MICROBATCH = 20", "smaller_retry=False", "SEED = 60701", "K = 2",
                      "'runs' not in build.parts", "QUALIFICATION_ONLY_SYNTHETIC", "NON_SCIENTIFIC"):
            self.assertIn(token, src)
        self.assertEqual(hashlib.sha256(at_m6fc(HARNESS)).hexdigest(), load(PREFLIGHT, 'pf2').HARNESS_SHA)

    def test_08_mock_relation_static(self):
        q = load(HARNESS, 'm6fc_harness_under_test')
        from methods.dsdg import native
        a, b = native.NativeRelation(q.mock_rows()), native.NativeRelation(q.mock_rows(shuffle_seed=q.SEED))
        self.assertEqual((a.spoof, a.pools), (b.spoof, b.pools))
        self.assertEqual(a.summary()['total']['spoof'], 3720)
        self.assertTrue(all(r['sample_id'].startswith('mock-') for r in q.mock_rows()))

    def test_09_frozen_config_and_e06c_unchanged(self):
        for rel in ('configs/methods/e06b_dsdg_native.yaml', 'frozen_config_snapshot/configs/methods/e06b_dsdg_native.yaml',
                    'configs/methods/e06c_dsdg_bin_idfree.yaml', 'methods/dsdg/native.py', 'methods/dsdg/adapter.py',
                    'methods/dsdg/microbatch_execution.py', 'methods/dsdg/microbatch_execution_v2.py',
                    'methods/dsdg/microbatch_qualification.py', 'methods/dsdg/runtime.py',
                    'methods/dsdg/training_graph.py', 'methods/dsdg/training_qualification.py',
                    'environments/e06c.lock.json',
                    'configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml'):
            self.assertEqual(at_m6fc(rel), at_authority(rel), rel)
        self.assertEqual(hashlib.sha256(at_m6fc('configs/methods/e06b_dsdg_native.yaml')).hexdigest(),
                         '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f')

    def test_10_ledger_prefix(self):
        pf = load(PREFLIGHT, 'pf3')
        prefix, now = at_authority(LEDGER), at_m6fc(LEDGER)
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
