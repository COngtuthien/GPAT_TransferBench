"""M6D6a E07c: retained GPU evidence and fail-closed runtime guards.

CPU/static checks never claim CUDA execution. The two separately launched
methods/difffas/runtime_qualification.py processes produced the CUDA evidence
validated here. The optional CPU construction test runs only where Torch exists.
"""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import unittest

import yaml

from methods.common.config import sha256_file
from methods.difffas import DiffFASAdapter
from methods.difffas.contract import validate_contract
from methods.difffas.runtime_qualification import (ALLOWED_EXACT_ARGV, BATCH, EXPECTED_FEATURES, Firewall, K,
                                                   LABEL, ROOT, SEED, contract)

PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
CONSUMER_SHA = '127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963'
PROCESSES = [ROOT / f'outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_{i}.json' for i in (1, 2)]
REPORT = ROOT / 'outputs/audit/M6D6A_E07C_RUNTIME_QUALIFICATION.md'
HAS_TORCH = importlib.util.find_spec('torch') is not None


class TestE07cRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [json.loads(p.read_text()) for p in PROCESSES]
        cls.adapter = DiffFASAdapter()
        cls.contracts = validate_contract(cls.adapter.config)

    # 1 -------------------------------------------------------------------------------
    def test_pinned_source_identity(self):
        local = self.adapter.validate_source()
        self.assertEqual((local['commit'], local['tree']), (PIN, TREE))
        self.assertEqual(local['files_sha256']['models/custom_rn.py'], CUSTOM_RN_SHA)
        self.assertEqual(local['files_sha256']['models/unet_autoenc.py'], CONSUMER_SHA)
        pins = json.loads((ROOT / 'third_party/source_pins.json').read_text())['sources']['difffas']
        self.assertEqual((pins['pinned_commit'], pins['commit_tree']), (PIN, TREE))
        for r in self.runs:
            for key in ('source_before', 'source_after'):
                s = r[key]
                self.assertEqual((s['commit'], s['tree'], s['repository']), (PIN, TREE, local['repository']))
                self.assertEqual(s['files_sha256'], local['files_sha256'])
                self.assertEqual(s['worktree_status'], '')  # no edits, no bytecode, no untracked files
            self.assertEqual(r['source_before'], r['source_after'])
            self.assertEqual(r['compatibility_patch'], 'NONE')

    # 2 -------------------------------------------------------------------------------
    def test_a6_effective_interface_and_preserved_history(self):
        overlay = self.contracts['overlay']
        self.assertEqual(overlay['correction'], {'x32x32': [32, 32, 256], 'x16x16': [16, 16, 512],
                                                 'x8x8': [8, 8, 512], 'embg': ['B', 7]})
        self.assertFalse(overlay['projection_layers_added'] or overlay['architecture_substituted'])
        frozen = yaml.safe_load((ROOT / 'configs/methods/e07c_difffas_bin_idfree.yaml').read_text())
        historical = {k: v['shape'] for k, v in frozen['conditioning_encoder']['feature_interface_at_256'].items()}
        self.assertEqual(historical['x32x32'], [32, 32, 128])  # pre-A6 text retained, never rewritten
        self.assertEqual(historical['x16x16'], [16, 16, 256])
        for r in self.runs:
            self.assertEqual(r['feature_interface_live_CHW'], {'x32x32': [256, 32, 32], 'x16x16': [512, 16, 16],
                                                                'x8x8': [512, 8, 8], 'embg': [7]})
            self.assertEqual(r['contract']['feature_interface_historical_frozen_HWC']['x32x32'], [32, 32, 128])
            self.assertEqual(r['contract']['feature_interface_effective_A6_HWC']['x32x32'], [32, 32, 256])
        self.assertEqual(EXPECTED_FEATURES, {'x32x32': [256, 32, 32], 'x16x16': [512, 16, 16], 'x8x8': [512, 8, 8]})

    # 3, 4, 5 -------------------------------------------------------------------------
    def test_custom_rn_resnet18_is_used_with_a3_head(self):
        for r in self.runs:
            e = r['encoder']
            self.assertEqual((e['class'], e['module'], e['file_sha256']), ('ResNet', 'custom_rn', CUSTOM_RN_SHA))
            self.assertEqual(e['factory'], 'models/custom_rn.py::resnet18(pretrained=False)')
            self.assertEqual((e['topology'], e['block'], e['stage_widths']), ([3, 4, 6, 3], 'BasicBlock',
                                                                                [64, 256, 512, 512]))
            self.assertFalse(e['torchvision_substitute'] or e['projection_or_adapter_modules_added'])
            self.assertEqual(e['structural_difference_vs_pinned_resnet18'], ['fc'])
            self.assertEqual(e['head'], {'type': 'Linear', 'in_features': 512, 'out_features': 7})
            self.assertEqual(e['pinned_default_head']['out_features'], 17)
            self.assertEqual(e['state']['shapes']['fc.weight'], [7, 512])
            self.assertTrue(e['untrained_in_memory_only'])
            self.assertFalse(e['weights_loaded'])
        seam = (ROOT / 'methods/difffas/encoder.py').read_text()
        self.assertIn("modules['custom_rn'].resnet18(pretrained=False)", seam)
        self.assertNotIn('torchvision.models', seam)

    # 6 -------------------------------------------------------------------------------
    def test_live_encoder_shapes_at_256(self):
        for r in self.runs:
            f = r['forward']
            self.assertEqual(f['target_pose']['shape'], [BATCH, 3, 256, 256])
            for name, chw in EXPECTED_FEATURES.items():
                self.assertEqual(f['encoder_' + name]['shape'], [BATCH, *chw])
                self.assertEqual((f['encoder_' + name]['dtype'], f['encoder_' + name]['device']),
                                 ('torch.float32', 'cuda:0'))
            self.assertEqual(f['encoder_embg']['shape'], [BATCH, K])
            s = r['encoder']['state']
            self.assertEqual(s['parameters'], sum(math.prod(v) for v in s['shapes'].values()))
            self.assertEqual(s['trainable_parameters'], s['parameters'])
            self.assertEqual(s['devices'], ['cuda:0'])

    @unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host; CUDA evidence validated from process JSON')
    def test_cpu_construction_of_pinned_encoder(self):
        import torch
        from methods.difffas.encoder import encoder_model
        with torch.random.fork_rng(devices=[]), torch.no_grad(), encoder_model(self.adapter.config) as enc:
            torch.manual_seed(SEED)
            enc.eval()
            self.assertEqual(type(enc).__module__, 'custom_rn')
            outs = enc(torch.zeros(1, 3, 256, 256))
        self.assertEqual([list(o.shape) for o in outs], [[1, 256, 32, 32], [1, 512, 16, 16], [1, 512, 8, 8], [1, 7]])
        self.assertTrue(all(torch.isfinite(o).all() for o in outs))
        self.assertEqual(sum(p.numel() for p in enc.parameters()), self.runs[0]['encoder']['state']['parameters'])

    # 7, 8, 9 -------------------------------------------------------------------------
    def test_main_model_live_construction_use_pair_false_three_channels(self):
        for r in self.runs:
            m = r['main_model']
            self.assertEqual(m['class'], 'BeatGANsAutoencModel')
            self.assertEqual(m['module'], 'models.unet_autoenc')
            self.assertEqual(m['file_sha256'], CONSUMER_SHA)
            self.assertEqual((m['conf']['in_channels'], m['conf']['image_size'], m['conf']['out_channels']), (3, 256, 6))
            self.assertEqual((m['first_conv_in_channels'], m['use_pair']), (3, False))
            s = m['state_initial']
            self.assertEqual(s['parameters'], sum(math.prod(v) for v in s['shapes'].values()))
            self.assertEqual(s['devices'], ['cuda:0'])
            self.assertEqual(r['contract']['main_training']['use_pair'], False)
            self.assertEqual(r['forward_loss_path']['use_pair'], False)
            d = r['diffusion']
            self.assertEqual((d['num_timesteps'], d['model_mean_type'], d['model_var_type'], d['loss_type']),
                             (1000, 'EPSILON', 'LEARNED_RANGE', 'MSE'))
            self.assertEqual(d['beta_schedule_node'], {'__target': 'diffusion.make_beta_schedule', 'schedule': 'linear',
                             'n_timestep': 1000, 'linear_start': 0.0001, 'linear_end': 0.02})
            self.assertEqual((d['betas']['min'], d['betas']['max'], d['betas']['dtype']), (1e-4, 0.02, 'torch.float64'))
            self.assertEqual({a['channels'] for a in m['attention_blocks']}, {256, 512})

    # 10 ------------------------------------------------------------------------------
    def test_finite_outputs_and_zero_initialised_projection_disclosed(self):
        for r in self.runs:
            self.assertTrue(r['finite_all'])
            self.assertTrue(all(v['finite'] for v in r['forward'].values()))
            c = r['connectivity']
            self.assertTrue(all(v['finite'] for v in c['internal_activations'].values()))
            self.assertTrue(c['final_output_identically_zero'])
            self.assertIn('resnet_use_zero_module=True', c['final_output_zero_reason'])
            self.assertEqual(r['forward_loss_path']['label'], 'FORWARD_LOSS_PATH_ONLY')
            self.assertFalse(r['forward_loss_path']['backward_executed'] or r['forward_loss_path']['training_graph_qualified'])

    # 11 ------------------------------------------------------------------------------
    def test_fourth_output_discarded_at_pinned_consumer(self):
        src = (ROOT / 'third_party/source_cache/difffas/models/unet_autoenc.py').read_text()
        forward = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == 'forward'
                       and n.lineno == 138)
        self.assertIn('x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)', ast.unparse(forward))
        self.assertEqual(src.splitlines()[177].strip(), 'x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)')
        for r in self.runs:
            c = r['connectivity']
            self.assertEqual((c['fourth_output_consumers'], c['feature_consumer_blocks']['embg']), (0, []))
            self.assertEqual(sorted(c['feature_consumer_blocks']), ['embg', 'x16x16', 'x32x32', 'x8x8'])
            for name in EXPECTED_FEATURES:
                self.assertTrue(c['feature_consumer_blocks'][name])
                self.assertTrue(all(v > 0 for v in c['zeroing_sensitivity_max_abs_at_consumer_attention_blocks'][name].values()))
            for name, (ch, h, w) in EXPECTED_FEATURES.items():
                self.assertTrue(all(x == {'x_channels': ch, 'x_hw': [h, w]} for x in c['feature_consumers'][name]))
            self.assertIn('bitwise identical', c['fourth_output_discard_proof'])

    # 12, 13 --------------------------------------------------------------------------
    def test_zero_optimizer_backward_checkpoint(self):
        for r in self.runs:
            self.assertEqual(r['counters'], {'optimizer_constructions': 0, 'optimizer_applications': 0,
                                             'backward_passes': 0, 'autograd_grad_calls': 0,
                                             'checkpoint_saves': 0, 'checkpoint_loads': 0})
            self.assertEqual((r['optimizer_applications'], r['backward_passes']), (0, 0))
            for flag in ('optimizer_constructed', 'checkpoint_created', 'checkpoint_loaded', 'training_launched',
                         'auxiliary_encoder_trained', 'synthetic_bank', 'diffusion_sampling', 'training_graph_qualified'):
                self.assertFalse(r[flag], flag)
            self.assertTrue(r['main_model']['state_after_parameters_unchanged'])
            self.assertTrue(all(n.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
                                for n in r['main_model']['buffers_changed_by_train_mode_forward']))
            self.assertEqual(r['firewall']['attempts']['scientific_run_root'], 0)
        plan = self.adapter.build_auxiliary_plan()
        self.assertFalse(plan['trained'])
        self.assertEqual(plan['checkpoint']['sha256_status'], 'RECORDED_AFTER_TRAINING')

    # 14, 15 --------------------------------------------------------------------------
    def test_zero_benchmark_manifest_and_image_access(self):
        for r in self.runs:
            fw = r['firewall']
            self.assertEqual(fw['denied'], [])
            self.assertEqual(fw['attempts'], {k: 0 for k in Firewall.CLASSES})
            self.assertGreater(fw['event_counts']['open'], 0)
            self.assertFalse(r['benchmark_data_access'] or r['TRAIN_access'] or r['VAL_access'] or r['TEST_access'])
            self.assertTrue(all(Path(a[0]).name in ('git', 'nvidia-smi') or a in ALLOWED_EXACT_ARGV
                                for a in fw['subprocesses']))
        test = self.adapter.config['data']['splits']['TEST']
        self.assertFalse(test['allowed'] or test['code_path_present'])

    def test_firewall_rejects_data_images_manifests_runs_and_foreign_writes(self):
        fw = Firewall(Path('/nonexistent/builds/e07c_difffas'))
        runtime = str(ROOT) + '_runtime'
        cases = {str(ROOT / 'manifests/split_v1.parquet'): 'benchmark_manifest',
                 str(ROOT / 'manifests/difffas_bin_idfree_train_v1.parquet'): 'benchmark_manifest',
                 str(ROOT / 'manifests/pairs_train_v1.parquet'): 'benchmark_manifest',
                 runtime + '/data/processed/faces_256/CASIA/x.png': 'benchmark_image',
                 runtime + '/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl': 'scientific_run_root',
                 '/elsewhere/PADISI.pkl': 'blocked_weight_or_array'}
        for path, cls in cases.items():
            with self.assertRaisesRegex(RuntimeError, 'FIREWALL'):
                fw('open', (path, 'r', os.O_RDONLY))
            self.assertIn(cls, fw.denied[-1]['classes'])
        with self.assertRaisesRegex(RuntimeError, 'FIREWALL'):
            fw('os.listdir', (runtime + '/data/processed/faces_256',))
        with self.assertRaisesRegex(RuntimeError, 'FIREWALL'):
            fw('open', ('/tmp/x.json', 'w', os.O_WRONLY | os.O_CREAT))
        with self.assertRaisesRegex(RuntimeError, 'SUBPROCESS FIREWALL'):
            fw('subprocess.Popen', ('/usr/bin/python3', ['python3', '-c', 'pass'], None, None))
        fw('subprocess.Popen', (None, ['/sbin/ldconfig', '-p'], None, None))
        fw('open', ('/nonexistent/builds/e07c_difffas/process_1.json', 'w', os.O_WRONLY | os.O_CREAT))

    # 16 ------------------------------------------------------------------------------
    def test_environment_lock_identity(self):
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        runtime = json.loads((ROOT / 'environments/e07c.runtime.json').read_text())
        self.assertEqual(lock['environment_name'], 'gpat-m6-e07c')
        self.assertEqual(lock['runtime_json_sha256'], sha256_file(ROOT / 'environments/e07c.runtime.json'))
        self.assertEqual(lock['runtime_harness_sha256'], sha256_file(ROOT / 'methods/difffas/runtime_qualification.py'))
        for name, digest in lock['exports_sha256'].items():
            self.assertEqual(sha256_file(ROOT / f'environments/e07c.{name}'), digest)
        self.assertEqual(lock['compatibility_patch'], 'NONE')
        self.assertTrue(runtime['created_by_E07c'] and not runtime['reused_existing_environment'])
        self.assertTrue(runtime['gpat_m5_unchanged'] and runtime['protected_environment_fingerprints']['unchanged'])
        self.assertEqual(runtime['pip_check'], 'No broken requirements found.')
        added = {p['name']: p['version'] for p in runtime['packages_added_after_clone']}
        self.assertEqual(added['tensorfn'], '0.1.28')
        self.assertEqual(added['pydantic'], '1.9.2')
        for r in self.runs:
            self.assertEqual(r['environment_before'], lock['identity'])
            self.assertEqual(r['environment_before'], r['environment_after'])
            self.assertIn('/gpat-m6-e07c/', r['environment_before']['executable'])
            e = r['environment_before']
            self.assertEqual((e['torch'], e['cuda'], e['gpu_capability']), ('2.12.1+cu130', '13.0', [8, 6]))
            self.assertIsNone(e['runtime_packages']['scipy'])

    # contract / repeatability / wording ---------------------------------------------
    def test_frozen_contract_static_values(self):
        c = contract(self.adapter, self.contracts)
        self.assertEqual(c['experiment_seeds'], [42, 1337, 2026])
        self.assertEqual(c['sampler_timesteps_ascending'], list(range(0, 250, 10)))
        self.assertEqual(c['authoritative_generation_tensor_set'], 'model')
        self.assertEqual((c['auxiliary_seed'], c['auxiliary_runs'], c['reused_for_main_seeds']), (42, 1, [42, 1337, 2026]))
        self.assertEqual(c['contract_inspection'], 'STATIC_ONLY_NOT_EXECUTED')
        for r in self.runs:
            self.assertEqual(r['contract'], c)

    def test_qualification_seed_and_two_process_repeatability(self):
        a, b = self.runs
        self.assertEqual((a['qualification_seed'], a['experiment_seed'], a['label']), (SEED, None, LABEL))
        self.assertNotIn(SEED, (42, 1337, 2026))
        self.assertEqual(a['batch_size_used_for_qualification'], 4)
        self.assertEqual(hashlib.sha256(PROCESSES[0].read_bytes()).hexdigest(),
                         hashlib.sha256(PROCESSES[1].read_bytes()).hexdigest())
        self.assertEqual(a['main_model']['state_initial']['aggregate_parameter_sha256'],
                         b['main_model']['state_initial']['aggregate_parameter_sha256'])

    def test_memory_observation_is_not_training_memory(self):
        for r in self.runs:
            m = r['cuda_memory']
            self.assertEqual(m['label'], 'SYNTHETIC_FORWARD_MEMORY_OBSERVATION')
            self.assertFalse(m['qualifies_training_memory'] or m['qualifies_scientific_batch_size'] or
                             m['qualifies_400_epoch_feasibility'])

    def test_controlled_adaptation_wording(self):
        self.assertEqual((self.adapter.config['fidelity_class'], self.adapter.config['deviation']),
                         ('CONTROLLED_ADAPTATION', 'DEV-021'))
        for r in self.runs:
            self.assertEqual(r['fidelity'], 'CONTROLLED_ADAPTATION')
        report = REPORT.read_text().lower()
        for phrase in ('faithful difffas', 'native difffas', 'official reproduction of', 'trained e07c',
                       'auxiliary encoder trained', 'training_graph_qualified', 'm8_ready'):
            self.assertNotIn(phrase, report)
        self.assertIn('controlled_adaptation', report)
        self.assertIn('implemented_not_executed', report)


if __name__ == '__main__':
    unittest.main()
