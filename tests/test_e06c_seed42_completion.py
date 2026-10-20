"""E06c seed42 closure: evidence checks for the completed E06c DSDG-BIN-IDFREE / seed 42 scientific run
(run_id 27dcaf23aac18d7a, launch authority 84809868, CONTROLLED_ADAPTATION, DEV-020).

Stdlib only, any host. The run root lives on the GPU host; these tests validate the committed evidence (completion
JSON with the embedded read-only collected document, checkpoint manifest, SHA-256 manifest) and re-derive the
completion JSON and manifest from the collected document to prove the build is deterministic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs' / 'audit'
COMPLETION = AUDIT / 'E06C_SEED42_COMPLETION.json'
MANIFEST = AUDIT / 'E06C_SEED42_CHECKPOINT_MANIFEST.txt'
SHA_FILE = AUDIT / 'E06C_SEED42_EVIDENCE_SHA256.txt'
AUTHORITY = '84809868acf472af5ba8cf6ab47165ae038e1012'
RUN_ID = '27dcaf23aac18d7a'
CONFIG_SHA = '7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002'
EPOCHS = [1] + list(range(10, 201, 10))


def tool():
    spec = importlib.util.spec_from_file_location('e06c_s42', ROOT / 'tools' / 'e06c_seed42_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE06cSeed42(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(COMPLETION.read_text())
        cls.c = cls.doc['collected']
        cls.manifest = MANIFEST.read_text()
        cls.t = tool()

    def test_01_status_identity(self):
        d = self.doc
        self.assertEqual(d['status'], 'E06C_SEED42_COMPLETED_CLEANLY')
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual((d['method']['id'], d['method']['name'], d['method']['fidelity_class'],
                          d['method']['deviation']), ('E06c', 'DSDG-BIN-IDFREE', 'CONTROLLED_ADAPTATION', 'DEV-020'))
        self.assertEqual((d['seed'], d['run_id'], d['launch_authority_commit']), (42, RUN_ID, AUTHORITY))
        m = self.c['run_manifest']
        self.assertEqual((m['git_commit'], m['git_dirty'], m['runner_mode'], m['config_sha256']),
                         (AUTHORITY, False, 'SCIENTIFIC', CONFIG_SHA))
        self.assertEqual(hashlib.sha256(f'E06c|42|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16], RUN_ID)

    def test_02_completion_exact(self):
        t, m = self.doc['training'], self.c['metrics']
        self.assertEqual((t['completed_epoch'], t['global_step'], t['step_records'], t['optimizer_applications']),
                         (200, 7400, 7400, 7400))
        self.assertTrue(m['contiguous'])
        self.assertEqual((m['duplicates'], m['missing_count'], m['schema_violation_count']), ([], 0, 0))
        self.assertEqual(m['epochs_present'], list(range(1, 201)))
        self.assertEqual((m['epoch_sizes_ok'], m['epoch_rows']), (200, [8838]))
        o = t['order_prediction']
        self.assertEqual((o['predicted_steps'], o['batch_hash_match'], o['epoch_order_match'], o['permutations'],
                          o['epoch_complete_events_ok']), (7400, 7400, 200, 200, 200))
        self.assertEqual(self.c['resume_reconciliation_events'], 0)

    def test_03_numerical_health(self):
        n = self.doc['numerical_health']
        self.assertEqual((n['nonfinite_steps'], n['corrupted_lines'], n['sidecars_checked']), (0, 0, 21))
        self.assertEqual(n['loss_ranges']['loss_cls'], [0.0, 0.0])
        self.assertEqual(n['loss_ranges']['loss_pair'], [0.0, 0.0])
        for s in self.c['sidecars']:
            for model in ('netE_nir', 'netE_vis', 'netG', 'netCls'):
                self.assertEqual(s['models'][model]['nonfinite_elements'], 0, (s['epoch'], model))
            self.assertEqual(s['adam']['nonfinite_elements'], 0, s['epoch'])
            self.assertEqual(s['adam']['step_values'], [float(s['global_step'])])
        for v in n['final_sidecar']['official_storage_match'].values():
            self.assertTrue(v['equal_multiset'])

    def test_04_checkpoints(self):
        k = self.doc['checkpoints']
        self.assertEqual((k['events'], k['epochs'], k['official_files']), (21, EPOCHS, 63))
        f = k['final_netG']
        self.assertEqual((f['path'], f['epoch'], f['checkpoint_type'], f['selected_for_final'], f['selection_rule']),
                         ('checkpoints/netG_model_epoch_200_iter_0.pth', 200, 'selected', True,
                          'OFFICIAL_GENERATOR_EPOCH_200'))
        self.assertEqual(f['sha256'], self.c['run_summary']['final_or_selected_checkpoint_sha256'])
        selected = [r for r in self.c['official_checkpoints'] if r['selected_for_final']]
        self.assertEqual([r['path'] for r in selected], [f['path']])
        self.assertEqual([s['epoch'] for s in self.c['sidecars']], EPOCHS)

    def test_05_firewall(self):
        f = self.doc['firewall']
        self.assertEqual((f['test_benchmark_access'], f['val_checkpoint_selection_access']), (0, 0))
        self.assertFalse(f['val_split_accessed'] or f['test_split_accessed'])
        for cls, v in f['atime']['by_class'].items():
            if not cls.endswith('_IN_TRAIN_RELATION'):
                self.assertEqual(v['atime_changed'], 0, cls)
        self.assertFalse(any(self.doc['confirmations'].values()))

    def test_06_manifest_deterministic(self):
        self.assertEqual(self.t.manifest_text(self.c), self.manifest)
        self.assertEqual(len(self.manifest.splitlines()), 1 + 63 + 1 + 21)
        self.assertEqual(hashlib.sha256(self.manifest.encode()).hexdigest(), self.doc['checkpoints']['manifest_sha256'])

    def test_07_rebuild_reproduces_completion(self):
        doc, text = self.t.build(self.c, ROOT)
        self.assertEqual(text, self.manifest)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)

    def test_08_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(len(names), 3)


if __name__ == '__main__':
    unittest.main()
