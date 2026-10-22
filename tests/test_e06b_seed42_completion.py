"""E06b seed42 closure: evidence checks for the completed E06b DSDG-NATIVE / seed 42 scientific run, ATTEMPT 2
(run_id aa96129289d080ba, training authority d9ecbef, FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY).

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
COMPLETION = AUDIT / 'E06B_SEED42_COMPLETION.json'
MANIFEST = AUDIT / 'E06B_SEED42_CHECKPOINT_MANIFEST.txt'
SHA_FILE = AUDIT / 'E06B_SEED42_EVIDENCE_SHA256.txt'
AUTHORITY = 'd9ecbef7295fc6b41ced6a4966b49810d36a8c87'
RUN_ID = 'aa96129289d080ba'
ATTEMPT1_RUN_ID = 'd3c0ddd5dc82a4a1'
CONFIG_SHA = '9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f'
FINAL_NETG_SHA = 'd56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa'
EPOCHS = [1] + list(range(10, 201, 10))


def tool():
    spec = importlib.util.spec_from_file_location('e06b_s42', ROOT / 'tools' / 'e06b_seed42_completion_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StaticE06bSeed42(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(COMPLETION.read_text())
        cls.c = cls.doc['collected']
        cls.manifest = MANIFEST.read_text()
        cls.t = tool()

    def test_01_status_identity_attempt2(self):
        d = self.doc
        self.assertEqual(d['status'], 'E06B_SEED42_COMPLETED_CLEANLY')
        self.assertEqual(d['failed_gates'], [])
        self.assertTrue(all(g['pass'] for g in d['gates']))
        self.assertEqual((d['method']['id'], d['method']['fidelity'], d['method']['track']),
                         ('E06b', 'FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY', 'B_NATIVE_FULL_SECONDARY'))
        self.assertEqual((d['seed'], d['attempt'], d['run_id'], d['training_authority_commit']),
                         (42, 2, RUN_ID, AUTHORITY))
        m = self.c['run_manifest']
        self.assertEqual((m['run_id'], m['git_commit'], m['git_dirty'], m['config_sha256']),
                         (RUN_ID, AUTHORITY, False, CONFIG_SHA))
        self.assertEqual(hashlib.sha256(f'E06b|42|{CONFIG_SHA}|{AUTHORITY}'.encode()).hexdigest()[:16], RUN_ID)
        self.assertEqual(self.c['git']['head'], AUTHORITY)

    def test_02_completion_exact_no_resume(self):
        t, m = self.doc['training'], self.c['metrics']
        self.assertEqual((t['completed_epoch'], t['global_step'], t['step_records'], t['optimizer_applications']),
                         (200, 3200, 3200, 3200))
        self.assertTrue(m['contiguous'])
        self.assertEqual((m['duplicates'], m['missing_count'], m['schema_violation_count']), ([], 0, 0))
        self.assertEqual(m['epochs_present'], list(range(1, 201)))
        self.assertEqual((m['epoch_sizes_ok'], m['epoch_rows']), (200, [3720]))
        o = t['order_prediction']
        self.assertEqual((o['predicted_steps'], o['pair_hash_match'], o['epoch_order_match'], o['permutations'],
                          o['epoch_complete_events_ok']), (3200, 3200, 200, 200, 200))
        p = t['pairing_audit']
        self.assertEqual({k: p[k] for k in ('pairs', 'same_subject', 'partner_live', 'train_split', 'casia_msu')},
                         dict.fromkeys(('pairs', 'same_subject', 'partner_live', 'train_split', 'casia_msu'), 744000))
        self.assertEqual(self.c['run_start_events'], 1)
        self.assertNotIn('--resume', self.c['run_manifest']['command_line'])
        self.assertFalse(self.doc['confirmations']['resume_used'])

    def test_03_numerical_health_active_losses(self):
        n = self.doc['numerical_health']
        self.assertEqual((n['nonfinite_steps'], n['corrupted_lines']), (0, 0))
        self.assertGreater(n['loss_ranges']['loss_cls'][0], 0.0)
        self.assertGreater(n['loss_ranges']['loss_pair'][0], 0.0)
        self.assertGreater(n['netCls_grad_nonzero_min'], 0)
        for r in self.c['official_checkpoints']:
            self.assertEqual(r['storages']['nonfinite_elements'], 0, r['path'])
            self.assertGreater(r['storages']['float_elements'], 0, r['path'])
        hits = self.c['logs']['stdout.log']['pattern_hits']
        self.assertEqual((hits['LearnedRunContext'], hits['has no attribute'], hits['Traceback']), (0, 0, 0))
        self.assertEqual(self.c['logs']['stderr.log']['bytes'], 0)

    def test_04_checkpoints_63_no_sidecar(self):
        k = self.doc['checkpoints']
        self.assertEqual((k['events'], k['epochs'], k['official_files'], k['resume_sidecars']), (21, EPOCHS, 63, 0))
        self.assertEqual(len(self.c['checkpoint_events']), 21)
        self.assertTrue(all(e['resume_sidecar'] is None for e in self.c['checkpoint_events']))
        self.assertEqual(self.c['runner_state_files'], [])
        self.assertFalse(self.c['resume_index_present'])
        f = k['final_netG']
        self.assertEqual((f['path'], f['epoch'], f['global_step'], f['checkpoint_type'], f['selected_for_final'],
                          f['selection_rule']), ('checkpoints/netG_model_epoch_200_iter_0.pth', 200, 3200, 'selected',
                                                 True, 'OFFICIAL_GENERATOR_EPOCH_200'))
        self.assertEqual(f['sha256'], FINAL_NETG_SHA)
        self.assertEqual(f['sha256'], self.c['run_summary']['final_or_selected_checkpoint_sha256'])
        selected = [r for r in self.c['official_checkpoints'] if r['selected_for_final']]
        self.assertEqual([(r['path'], r['sha256_actual']) for r in selected], [(f['path'], FINAL_NETG_SHA)])
        for r in self.c['official_checkpoints']:
            if r['path'] != f['path']:
                self.assertEqual(r['checkpoint_type'], 'terminal' if r['epoch'] == 200 else 'periodic', r['path'])

    def test_05_attempt1_not_selectable(self):
        a1, c1 = self.doc['attempt1'], self.c['attempt1_archive']
        self.assertEqual((a1['run_id'], a1['classification'], a1['selectable'], a1['eligible_for_bank'],
                          a1['eligible_for_downstream']),
                         (ATTEMPT1_RUN_ID, 'FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG', False, False, False))
        self.assertTrue(c1['exists'])
        self.assertFalse(c1['under_runs'])
        self.assertEqual(c1['listing_digest_path_size_sha256'],
                         '8ce8d5a8be3ec6179ef4299a51ab7903383bb963307e809d50c7e2dd9d91f6ca')
        self.assertNotEqual(self.doc['run_id'], ATTEMPT1_RUN_ID)
        self.assertFalse(self.doc['confirmations']['attempt1_selected'])

    def test_06_firewall_test_val_siw_zero(self):
        f = self.doc['firewall']
        self.assertEqual((f['test_access'], f['val_access'], f['siw_access']), (0, 0, 0))
        self.assertFalse(f['test_split_accessed'])
        for cls, v in f['atime']['by_class'].items():
            if cls not in ('casia_fasd|TRAIN', 'msu_mfsd|TRAIN'):
                self.assertEqual(v['atime_changed'], 0, cls)
            if cls.endswith('|TEST') or cls.endswith('|VAL'):
                self.assertEqual(v['sensitive_whole_window'], v['files'], cls)
        self.assertFalse(any(self.doc['confirmations'].values()))

    def test_07_manifest_deterministic(self):
        self.assertEqual(self.t.manifest_text(self.c), self.manifest)
        self.assertEqual(len(self.manifest.splitlines()), 1 + 63 + 1)
        self.assertEqual(hashlib.sha256(self.manifest.encode()).hexdigest(), self.doc['checkpoints']['manifest_sha256'])

    def test_08_rebuild_reproduces_completion(self):
        doc, text = self.t.build(self.c, ROOT)
        self.assertEqual(text, self.manifest)
        self.assertEqual(json.loads(json.dumps(doc)), self.doc)

    def test_09_evidence_sha_manifest(self):
        names = []
        for line in SHA_FILE.read_text().splitlines():
            digest, name = line.split('  ')
            names.append(name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(len(names), 3)


if __name__ == '__main__':
    unittest.main()
