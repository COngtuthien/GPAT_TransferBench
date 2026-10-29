"""E04 production-readiness audit: checks the committed readiness record (status, stage table, decision register,
synthetic re-qualification comparison, SHA-256 list) and re-derives it from the embedded host collection and the
committed re-qualification copies when the frozen 3DDFA assets are available on this host.

Stdlib only (re-derivation additionally imports the E04 static adapter).
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
RECORD = AUDIT / 'E04_PRODUCTION_READINESS_AUDIT.json'
REQUAL = {'M6D3A_CPU': 'm6d3a_cpu.json', 'M6D3A_CUDA': 'm6d3a_cuda.json', 'M6D3E_PROCESS': 'm6d3e_process.json'}


class E04ReadinessAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(RECORD.read_text())

    def test_01_status_not_authorized(self):
        d = self.doc
        self.assertEqual(d['status'], 'E04_BLOCKED_BY_SOURCE_GAP')
        self.assertIn('E04_READINESS_AUDIT_COMPLETE', d['also'])
        self.assertFalse(d['scientific_execution_authorized'])
        self.assertIsNone(d['e04_scientific_authority'])
        self.assertFalse(any(v for k, v in d['confirmations'].items() if k != 'benchmark_images_decoded'))
        self.assertEqual(d['confirmations']['benchmark_images_decoded'], 0)

    def test_02_stages(self):
        s = self.doc['stages']
        self.assertEqual(s['A_static_source_and_contract'], 'PASS')
        self.assertTrue(s['B_environment_and_dependencies'].startswith('PASS_SYNTHETIC'))
        for k in ('E_deterministic_data_path', 'F_real_train_val_feed', 'G_depth_m0_integrity',
                  'J_real_data_production_qualification', 'K_independent_evidence_verification'):
            self.assertTrue(s[k].startswith('BLOCKED'), k)
        self.assertTrue(all(self.doc['host_checks'].values()))

    def test_03_decision_register(self):
        ds = self.doc['decisions_required']
        self.assertEqual([d['id'] for d in ds], ['D%d' % i for i in range(1, 10)])
        for d in ds:
            self.assertTrue(d['options'] and d['why_open'] and d['recommendation'] and d['blocks'], d['id'])
        self.assertEqual({d['id'] for d in ds if d['scientific']}, {'D1', 'D2', 'D4', 'D5', 'D7', 'D8'})

    def test_04_requalification_matches_committed_evidence(self):
        r = self.doc['requalification']
        self.assertEqual(r['M6D3A_CPU']['depth_sha256'], r['M6D3A_CUDA']['depth_sha256'])
        self.assertTrue(r['M6D3A_CPU']['depth_sha256_in_committed_m6d3a'])
        e = r['M6D3E_PROCESS']
        self.assertTrue(e['graph_def_equals_committed'])
        self.assertEqual((e['status'], e['optimizer_applications'], e['tests']['failures'], e['tests']['errors']),
                         ('PASS', 3, 0, 0))
        self.assertEqual(e['post_state_difference_networks_per_step'], [['D3'], ['D3'], ['D3']])
        self.assertEqual(e['committed_process_pair_differences_per_step'], [0, 51, 51])

    def test_05_contract_values(self):
        c = self.doc['contract']
        self.assertEqual(c['fidelity_class'], 'CONTROLLED_ADAPTATION')
        self.assertEqual(c['training']['iterations'], 150000)
        self.assertEqual(c['checkpoint']['rule'], 'BASELINE_FINAL_STATE_V1')
        self.assertEqual(c['checkpoint']['terminal_state'], 'iteration_150000')
        self.assertEqual(c['data']['passes'], [67, 7854])
        self.assertEqual(c['seeds'], [42, 1337, 2026])

    def test_06_evidence_sha_manifest(self):
        names = []
        for line in (AUDIT / 'E04_PRODUCTION_READINESS_AUDIT_SHA256.txt').read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(len(names), 5)
        for k, v in REQUAL.items():
            self.assertEqual(self.doc['host_collect']['requal_files_sha256'][v],
                             hashlib.sha256((AUDIT / ('E04_READINESS_REQUAL_%s.json' % k)).read_bytes()).hexdigest())

    def test_07_rederives(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('e04_ready', ROOT / 'tools' / 'e04_production_readiness_audit.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from methods.common.config import load_method_config
        root = Path(load_method_config('E04')['external_assets']['geometry_engine']['external_runtime_root'])
        if not root.is_dir():
            self.skipTest('frozen 3DDFA assets are not mounted on this host')
        with tempfile.TemporaryDirectory() as tmp:
            for k, v in REQUAL.items():
                shutil.copyfile(AUDIT / ('E04_READINESS_REQUAL_%s.json' % k), Path(tmp) / v)
            doc = mod.build(ROOT, self.doc['host_collect'], tmp)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)
        self.assertEqual(mod.report(doc), (AUDIT / 'E04_PRODUCTION_READINESS_AUDIT.md').read_text())


if __name__ == '__main__':
    unittest.main()
