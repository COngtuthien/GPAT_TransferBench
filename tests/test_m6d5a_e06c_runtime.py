"""M6D5a E06c: retained GPU evidence and fail-closed runtime guards.

CPU/static checks never claim CUDA execution. The two separately launched
methods/dsdg/runtime.py processes produced the CUDA evidence validated here.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest

from methods.common.config import sha256_file
from methods.common.learned import PreparationError
from methods.dsdg import DSDGAdapter
from methods.dsdg.runtime import (EXPECTED_CLASSES, EXPECTED_PARAMETERS, Firewall, LIGHTCNN,
                                  OPTIMIZER_OWNED, ROOT, SEED, contract)

PIN = '16b793a7564a4b9308cf94e62bdb2ffacb3a725a'
TREE = '0d2216bdbdd9977130db1100641abf336f46ac9d'
LIGHTCNN_SHA = 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964'
PROCESSES = [ROOT / f'outputs/audit/M6D5A_E06C_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2)]


class TestE06cRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [json.loads(p.read_text()) for p in PROCESSES]
        cls.adapter = DSDGAdapter()

    def test_frozen_e06c_config_identity(self):
        cfg = self.adapter.config
        self.assertEqual(cfg['method_id'], 'E06c')
        self.assertEqual(cfg['status'], 'CONFIG_FROZEN')
        self.assertEqual(cfg['_runtime']['config_sha256'], sha256_file(ROOT / 'configs/methods/e06c_dsdg_bin_idfree.yaml'))

    def test_frozen_a1_config_identity(self):
        ref = self.adapter.config['adaptation_semantics']
        self.assertEqual(sha256_file(ROOT / ref['frozen_in']), ref['frozen_config_sha256'])
        self.assertEqual(ref['frozen_config_sha256'],
                         'a0f84fac2e893158410aab6d3cd11f464f6385edad2403c1f8ce4a66615b15de')

    def test_pinned_source_identity_and_immutability(self):
        local = self.adapter.validate_source()
        self.assertEqual((local['commit'], local['tree']), (PIN, TREE))
        for r in self.runs:
            for key in ('source_before', 'source_after'):
                s = r[key]
                self.assertEqual((s['commit'], s['tree'], s['repository']), (PIN, TREE, local['repository']))
                self.assertEqual(s['files_sha256'], local['files_sha256'])
                self.assertEqual(s['worktree_status'], 'D addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl')
            self.assertEqual(r['source_before'], r['source_after'])
        pins = json.loads((ROOT / 'third_party/source_pins.json').read_text())['sources']['dsdg']
        self.assertEqual(pins['pinned_commit'], PIN)
        self.assertEqual(pins['binary_weights_removed_after_checkout'],
                         ['addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl'])

    def test_hdim_attack_type_and_contract(self):
        c = contract(self.adapter)
        self.assertEqual((c['hdim'], c['attack_type'], c['input_resolution'], c['all_epochs']), (128, 1, 256, 200))
        self.assertEqual(c['losses'], {'lambda_mmd': 50, 'lambda_ip': 1000, 'lambda_type': 10,
                                       'lambda_ort': 1, 'lambda_pair': 0})
        self.assertEqual(c['optimizer']['learning_rate'], 2e-4)
        self.assertEqual(c['optimizer_contract_inspection'], 'STATIC_ONLY')
        for r in self.runs:
            self.assertEqual(r['contract'], c)

    def test_exact_architecture_class_binding(self):
        for r in self.runs:
            for name, cls in EXPECTED_CLASSES.items():
                b = r['binding'][name]
                self.assertEqual((b['class'], b['wrapper'], b['device_ids']), (cls, 'torch.nn.DataParallel', [0]))
                self.assertIn(b['module'], ('networks.generator', 'networks.light_cnn'))
                rel = 'addition_module/DSDG/' + b['module'].replace('.', '/') + '.py'
                self.assertEqual(b['file_sha256'], r['source_before']['files_sha256'][rel])

    def test_live_parameter_counts(self):
        for r in self.runs:
            for name, (count, tensors) in EXPECTED_PARAMETERS.items():
                p = r['parameters'][name]
                self.assertEqual((p['parameters'], p['parameter_tensors']), (count, tensors))
                self.assertEqual(sum(math.prod(s) for s in p['shapes'].values()), count)

    def test_optimizer_owned_total_excludes_netcls(self):
        self.assertEqual(OPTIMIZER_OWNED, ('netE_nir', 'netE_vis', 'netG'))
        for r in self.runs:
            o = r['optimizer_ownership']
            self.assertEqual((o['parameters'], o['parameter_tensors']), (45_370_182, 55))
            self.assertEqual(o['parameters'], sum(r['parameters'][n]['parameters'] for n in OPTIMIZER_OWNED))
            self.assertTrue(o['netCls_excluded'] and o['netIP_excluded'])
            self.assertNotIn('netCls', o['owned_modules'])
            self.assertFalse(o['optimizer_constructed'])
        self.assertFalse(self.adapter.config['optimizer']['netCls_in_optimizer'])

    def test_lightcnn_asset_identity(self):
        asset, = self.adapter.config['external_assets']
        self.assertEqual((asset['bytes'], asset['sha256']), (123844849, LIGHTCNN_SHA))
        for r in self.runs:
            for key in ('asset_before', 'asset_after'):
                self.assertEqual((r[key]['path'], r[key]['bytes'], r[key]['sha256']),
                                 (str(LIGHTCNN), 123844849, LIGHTCNN_SHA))
            self.assertFalse(Path(r['asset_before']['path']).is_relative_to(ROOT))

    def test_lightcnn_60_of_60_and_fc2_filtering(self):
        for r in self.runs:
            l = r['lightcnn']
            self.assertEqual((l['checkpoint_tensor_count'], l['expected_model_tensor_count'], l['matched']), (61, 60, 60))
            self.assertEqual((l['missing'], l['shape_mismatches']), ([], []))
            self.assertEqual(l['filtered_extra_keys'], {'module.fc2.weight': [80013, 256]})
            self.assertTrue(l['loaded_values_equal_checkpoint'] and l['torch_load_weights_only'])

    def test_netip_eval_and_frozen(self):
        for r in self.runs:
            self.assertFalse(r['modes']['netIP'])
            self.assertTrue(all(r['modes'][n] for n in ('netE_nir', 'netE_vis', 'netG')))
            self.assertEqual(r['parameters']['netIP']['requires_grad_parameters'], 0)

    def test_forward_interface_shapes(self):
        expected = {'x_spoof': [1, 3, 256, 256], 'x_live': [1, 3, 256, 256],
                    'mu_nir': [1, 128], 'logvar_nir': [1, 128], 'mu_a': [1, 128], 'logvar_a': [1, 128],
                    'mu_vis': [1, 128], 'logvar_vis': [1, 128],
                    'z_cls': [1, 128], 'z_nir': [1, 128], 'z_vis': [1, 128],
                    'pre_spoof': [1, 1], 'latent': [1, 384], 'rec': [1, 6, 256, 256],
                    'rec_spoof': [1, 3, 256, 256], 'rec_live': [1, 3, 256, 256]}
        for n in ('x_spoof', 'x_live', 'rec_spoof', 'rec_live'):
            expected.update({n + '_128': [1, 3, 128, 128], n + '_gray': [1, 1, 128, 128],
                             n + '_ip_feature': [1, 256]})
        for r in self.runs:
            self.assertEqual(r['shapes'], expected)
            self.assertEqual({k: v['shape'] for k, v in r['forward'].items()}, expected)

    def test_synthetic_inputs_distinct_unit_range(self):
        for r in self.runs:
            a, b = r['forward']['x_spoof'], r['forward']['x_live']
            self.assertNotEqual(a['sha256'], b['sha256'])
            for x in (a, b):
                self.assertGreaterEqual(x['min'], 0.0)
                self.assertLessEqual(x['max'], 1.0)

    def test_reparameterization_is_pinned(self):
        for r in self.runs:
            rp = r['reparameterization']
            self.assertEqual(rp['function'], 'misc/util.py::reparameterize')
            self.assertFalse(rp['altered'])
        source = (ROOT / 'third_party/source_cache/facexzoo/addition_module/DSDG/misc/util.py').read_text()
        self.assertIn('eps = torch.cuda.FloatTensor(std.size()).normal_()', source)

    def test_finite_outputs(self):
        for r in self.runs:
            self.assertTrue(r['finite_all'])
            self.assertTrue(all(v['finite'] and v['dtype'] == 'torch.float32' for v in r['forward'].values()))

    def test_one_logit_ce_degeneracy(self):
        for r in self.runs:
            c = r['one_logit_classifier']
            self.assertEqual(c['status'], 'EXPECTED_DEGENERACY_CONFIRMED')
            self.assertEqual((c['cross_entropy'], c['logit_shape'], c['target_class_index']), (0.0, [1, 1], 0))
            self.assertFalse(c['converted_to_two_logits'])
        self.assertEqual(self.adapter.config['training']['attack_type'], 1)

    def test_zero_optimizer_backward_checkpoint(self):
        for r in self.runs:
            self.assertEqual(r['counters'], {'optimizer_constructions': 0, 'optimizer_applications': 0,
                                             'backward_passes': 0, 'checkpoint_saves': 0})
            self.assertEqual((r['optimizer_applications'], r['backward_passes']), (0, 0))
            self.assertFalse(r['optimizer_constructed'] or r['checkpoint_created'] or r['training_launched'])
            self.assertEqual(r['parameters']['netG']['sha256'], self.runs[0]['parameters']['netG']['sha256'])

    def test_physical_batch_240_not_qualified(self):
        for r in self.runs:
            self.assertFalse(r['physical_batch_240_qualified'] or r['training_graph_qualified'])
            self.assertEqual(r['label'], 'DIAGNOSTIC_ARCHITECTURE_BATCH_ONLY')
            m = r['cuda_memory']
            self.assertEqual(m['label'], 'DIAGNOSTIC_BATCH1_ONLY')
            self.assertFalse(m['establishes_physical_batch_240_feasibility'])
            self.assertEqual(m['batch_240_extrapolation'], 'NOT_PERFORMED')
        for bad in ((120, 2), (60, 4), (1, 240)):
            with self.assertRaisesRegex(PreparationError, 'E06C_REQUIRES_PHYSICAL_BATCH_240'):
                self.adapter.validate_batch(physical_batch_size=bad[0], gradient_accumulation_steps=bad[1])

    def test_no_benchmark_data_no_test_no_bank(self):
        for r in self.runs:
            self.assertEqual(r['firewall']['denied'], [])
            self.assertGreater(r['firewall']['lightcnn_read_opens'], 0)
            self.assertFalse(r['benchmark_data_access'] or r['TEST_access'] or r['synthetic_bank'])
        test = self.adapter.config['data']['splits']['TEST']
        self.assertFalse(test['allowed'] or test['code_path_present'])

    def test_firewall_rejects_data_and_foreign_weights(self):
        fw = Firewall(Path('/nonexistent/builds/e06c_dsdg'))
        for path in (str(ROOT / 'manifests/split_v1.parquet'), str(ROOT) + '_runtime/data/processed/faces_256/x',
                     '/elsewhere/other.pth.tar', '/elsewhere/image.png'):
            with self.assertRaisesRegex(RuntimeError, 'FIREWALL'):
                fw('open', (path, 'r', os.O_RDONLY))
        with self.assertRaisesRegex(RuntimeError, 'FIREWALL'):
            fw('open', ('/tmp/x.json', 'w', os.O_WRONLY | os.O_CREAT))
        with self.assertRaisesRegex(RuntimeError, 'FIREWALL'):
            fw('open', (str(LIGHTCNN), 'wb', os.O_WRONLY))
        fw('open', (str(LIGHTCNN), 'rb', os.O_RDONLY))
        self.assertEqual(fw.lightcnn_opens, 1)

    def test_two_process_repeatability(self):
        a, b = self.runs
        self.assertEqual(a['diagnostic_seed'], SEED)
        self.assertNotIn(SEED, (42, 1337, 2026))
        for key in ('parameters', 'forward', 'lightcnn', 'binding', 'environment_before', 'cuda_memory'):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(hashlib.sha256(PROCESSES[0].read_bytes()).hexdigest(),
                         hashlib.sha256(PROCESSES[1].read_bytes()).hexdigest())

    def test_environment_identity(self):
        rt = json.loads((ROOT / 'environments/e06c.runtime.json').read_text())
        self.assertEqual(rt['compatibility']['compatibility_patch'], 'NONE')
        self.assertTrue(rt['created_by_E06c'] and rt['gpat_m5_unchanged'])
        for r in self.runs:
            self.assertEqual(r['environment_before'], r['environment_after'])
            self.assertIn('/gpat-m6-e06c/', r['environment_before']['executable'])
            self.assertEqual(r['compatibility_patch'], 'NONE')

    def test_controlled_adaptation_wording(self):
        cfg = self.adapter.config
        self.assertEqual(cfg['fidelity_class'], 'CONTROLLED_ADAPTATION')
        for r in self.runs:
            self.assertEqual(r['fidelity'], 'CONTROLLED_ADAPTATION')
        report = (ROOT / 'outputs/audit/M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION.md').read_text().lower()
        for word in ('trained e06c', 'faithful dsdg', 'native dsdg', 'official reproduction of'):
            self.assertNotIn(word, report)
        self.assertIn('controlled_adaptation', report)


if __name__ == '__main__':
    unittest.main()
