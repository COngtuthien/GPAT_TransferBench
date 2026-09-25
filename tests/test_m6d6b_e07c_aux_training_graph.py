"""M6D6b E07c: auxiliary conditioning-encoder training-graph evidence and fail-closed guards.

CPU/static checks never claim CUDA execution. The two separately launched
methods/difffas/aux_training_qualification.py processes produced the evidence
validated here. The optional live Torch test constructs the encoder and SGD and
walks the autograd graph WITHOUT calling backward() or optimizer.step(), so the
qualification counts (one backward + one step per GPU process) stay exact.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

import yaml

from methods.common.config import sha256_file
from methods.difffas import DiffFASAdapter
from methods.difffas.aux_training_qualification import (BATCH, BUILD_PARTS, ELIGIBILITY, HYPER, K, LABEL, ROOT,
                                                        SCIENTIFIC_AUX_BATCH, SEED, SOURCE_DISCONNECTED_PARAMETERS,
                                                        SOURCE_UNUSED_MODULES, SYNTHETIC_LABELS,
                                                        upstream_training_semantics)
from methods.difffas.contract import validate_contract
from methods.difffas.runtime_qualification import ALLOWED_EXACT_ARGV, Firewall

PIN = '23f40519ec25a833ebc06842aa6fbab74fad4d15'
TREE = 'd190a5fb4a94cb9e423215f9a24a7863aa17ea65'
CUSTOM_RN_SHA = 'fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3'
PRETRAIN_SHA = '3444691f6f2c59b41a07f06db9388b134e849bb889c5f6b03489d47c7e037b16'
LOCK_SHA = '0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450'
HARNESS = ROOT / 'methods/difffas/aux_training_qualification.py'
PROCESSES = [ROOT / f'outputs/audit/M6D6B_E07C_AUX_TRAINING_PROCESS_{i}.json' for i in (1, 2)]
LOG = ROOT / 'outputs/audit/M6D6B_E07C_AUX_TRAINING_RUNTIME_LOG.txt'
LAUNCH_ONLY = ('PYTHONHASHSEED', 'TMPDIR')  # launch variables that legitimately differ from the M6D6a lock capture
HAS_TORCH = importlib.util.find_spec('torch') is not None


class TestE07cAuxTrainingGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = [p.read_bytes() for p in PROCESSES]
        cls.runs = [json.loads(r) for r in cls.raw]
        cls.adapter = DiffFASAdapter()
        cls.contracts = validate_contract(cls.adapter.config)
        cls.tc = cls.contracts['effective_encoder']['training_contract']

    # 1 source pin -------------------------------------------------------------------
    def test_source_pin_unchanged(self):
        local = self.adapter.validate_source()
        self.assertEqual((local['commit'], local['tree']), (PIN, TREE))
        self.assertEqual(local['files_sha256']['models/custom_rn.py'], CUSTOM_RN_SHA)
        self.assertEqual(local['files_sha256']['models/pretrain_classifier.py'], PRETRAIN_SHA)
        pins = json.loads((ROOT / 'third_party/source_pins.json').read_text())['sources']['difffas']
        self.assertEqual((pins['pinned_commit'], pins['commit_tree']), (PIN, TREE))
        for r in self.runs:
            self.assertEqual(r['source_before'], r['source_after'])
            self.assertEqual(r['source_before']['files_sha256'], local['files_sha256'])
            self.assertEqual(r['source_before']['worktree_status'], '')
            self.assertEqual(r['compatibility_patch'], 'NONE')

    # 2 environment lock --------------------------------------------------------------
    def test_environment_lock_unchanged(self):
        self.assertEqual(sha256_file(ROOT / 'environments/e07c.lock.json'), LOCK_SHA)
        lock = json.loads((ROOT / 'environments/e07c.lock.json').read_text())
        for r in self.runs:
            env = r['environment_before']
            self.assertEqual(env, r['environment_after'])
            self.assertEqual({k: v for k, v in env.items() if k != 'launch_environment'},
                             {k: v for k, v in lock['identity'].items() if k != 'launch_environment'})
            strip = lambda d: {k: v for k, v in d.items() if k not in LAUNCH_ONLY}
            self.assertEqual(strip(env['launch_environment']), strip(lock['identity']['launch_environment']))
            self.assertEqual(env['launch_environment']['PYTHONHASHSEED'], str(SEED))
            self.assertIn('/gpat-m6-e07c/', env['executable'])

    # 3, 4 encoder identity and K=7 head ---------------------------------------------
    def test_a3_a6_encoder_identity_and_k7_head(self):
        for r in self.runs:
            e = r['encoder']
            self.assertEqual((e['class'], e['module'], e['file_sha256']), ('ResNet', 'custom_rn', CUSTOM_RN_SHA))
            self.assertEqual((e['topology'], e['block'], e['stage_widths']),
                             ([3, 4, 6, 3], 'BasicBlock', [64, 256, 512, 512]))
            self.assertEqual(e['structural_difference_vs_pinned_resnet18'], ['fc'])
            self.assertFalse(e['torchvision_substitute'] or e['projection_or_adapter_modules_added'] or
                             e['weights_loaded'])
            self.assertEqual(e['head'], {'type': 'Linear', 'in_features': 512, 'out_features': 7, 'bias': True})
            self.assertEqual(e['state_initial']['shapes']['fc.weight'], [K, 512])
            self.assertEqual(r['forward']['feature_shapes'], [[BATCH, 256, 32, 32], [BATCH, 512, 16, 16],
                                                              [BATCH, 512, 8, 8]])
            self.assertEqual(r['forward']['logits_shape'], [BATCH, 7])
        obj = self.contracts['effective_encoder']['substitute_objective']
        self.assertEqual((obj['K'], obj['classes'], obj['excluded']), (7, ['live', 'makeup', 'mask_2d', 'mask_3d',
                         'partial', 'print', 'replay'], ['other_spoof']))

    # 5 exact CE ----------------------------------------------------------------------
    def test_exact_cross_entropy_on_fourth_output(self):
        self.assertEqual(self.tc['loss'], 'CrossEntropyLoss_on_fourth_forward_output')
        for r in self.runs:
            loss = r['loss']
            self.assertEqual((loss['criterion'], loss['reduction'], loss['label_smoothing'], loss['weight']),
                             ('torch.nn.CrossEntropyLoss()', 'mean', 0.0, None))
            self.assertEqual(r['forward']['loss_attached_to'], 'outputs[3] only')
            self.assertLess(loss['abs_deviation_from_reference'], 1e-5)
        up = self.runs[0]['upstream_training_semantics']
        self.assertEqual(up['criterion']['statement'], 'nn.CrossEntropyLoss()')
        self.assertIn('_, _, _, outputs = resnet18(inputs)', up['step_order'])

    # 6, 7 exact SGD, no scheduler ----------------------------------------------------
    def test_exact_sgd_hyperparameters_and_no_scheduler(self):
        self.assertEqual((self.tc['optimizer'], self.tc['learning_rate'], self.tc['momentum'], self.tc['weight_decay'],
                          self.tc['scheduler']), ('SGD', 0.002, 0.9, 5e-3, 'NONE'))
        self.assertEqual(HYPER, {'lr': 0.002, 'momentum': 0.9, 'weight_decay': 5e-3})
        for r in self.runs:
            o = r['optimizer']
            self.assertEqual(o['class'], 'torch.optim.SGD')
            h = o['hyperparameters']
            self.assertEqual((h['lr'], h['momentum'], h['weight_decay'], h['dampening'], h['nesterov'], h['maximize']),
                             (0.002, 0.9, 0.005, 0, False, False))
            self.assertEqual(o['param_groups'], 1)
            self.assertIsNone(o['scheduler'])
            self.assertIsNone(o['grad_clipping'])
            s = r['encoder']['state_initial']
            self.assertEqual((o['optimized_tensors'], o['optimized_parameters']), (s['parameter_tensors'], s['parameters']))
            self.assertEqual(s['parameters'], sum(math.prod(v) for v in s['shapes'].values()))
            self.assertEqual((r['counters']['scheduler_constructions'], r['counters']['grad_clipping_calls']), (0, 0))
            up = r['upstream_training_semantics']
            self.assertEqual((up['optimizer']['params'], up['optimizer']['hyperparameters'], up['scheduler']),
                             ('resnet18.parameters()', HYPER, 'NONE'))

    def test_upstream_semantics_recomputed_from_pinned_source(self):
        up = upstream_training_semantics(self.adapter.validate_source())
        self.assertEqual(up, self.runs[0]['upstream_training_semantics'])
        self.assertEqual(up['mode_calls'], [])
        self.assertEqual(up['precision_controls'], [])
        self.assertFalse(up['imported_or_executed'])
        order = list(up['step_order'].values())
        self.assertEqual(order, sorted(order))
        self.assertEqual(up['loader']['batch_size'], SCIENTIFIC_AUX_BATCH)

    # 8, 9, 10 batch and seeds --------------------------------------------------------
    def test_qualification_batch_and_seed_distinct_from_science(self):
        self.assertEqual((BATCH, SCIENTIFIC_AUX_BATCH, self.tc['batch_size']), (4, 256, 256))
        self.assertEqual(SEED, 60602)
        self.assertNotIn(SEED, (42, 1337, 2026))
        self.assertEqual(self.contracts['effective_encoder']['auxiliary_encoder_training_seed'], 42)
        for r in self.runs:
            self.assertEqual((r['qualification_seed'], r['qualification_batch_size'], r['scientific_aux_batch_size']),
                             (SEED, 4, 256))
            self.assertTrue(r['batch_256_training_memory_not_qualified'])
            self.assertIsNone(r['auxiliary_encoder_training_seed'])
            self.assertIsNone(r['experiment_seed'])
            self.assertFalse(r['scientific_seed_consumed'])
            self.assertEqual(r['label'], LABEL)
            self.assertEqual(r['synthetic_batch']['target_values'], list(SYNTHETIC_LABELS))
            self.assertTrue(all(0 <= v < K for v in SYNTHETIC_LABELS))
            self.assertEqual(r['synthetic_batch']['label'], 'SYNTHETIC_IN_MEMORY_NOT_BENCHMARK_SAMPLES')

    # 11, 12 one backward / one step --------------------------------------------------
    def test_exactly_one_backward_and_one_step_per_process(self):
        for r in self.runs:
            self.assertEqual(r['counters'], {'optimizer_constructions': 1, 'sgd_constructions': 1,
                                             'optimizer_step_calls': 1, 'backward_calls': 1, 'zero_grad_calls': 1,
                                             'autograd_grad_calls': 0, 'scheduler_constructions': 0,
                                             'grad_clipping_calls': 0, 'checkpoint_saves': 0, 'checkpoint_loads': 0})
            self.assertEqual((r['backward_calls'], r['optimizer_step_calls'], r['epochs']), (1, 1, 0))
        self.assertEqual(sum(r['backward_calls'] for r in self.runs), 2)
        self.assertEqual(sum(r['optimizer_step_calls'] for r in self.runs), 2)

    # 13, 14, 15, 16 loss and gradients -----------------------------------------------
    def test_finite_loss_and_gradients(self):
        for r in self.runs:
            self.assertTrue(r['loss']['finite'] and math.isfinite(r['loss']['value']))
            g = r['gradients']
            self.assertTrue(g['all_finite'])
            self.assertTrue(all(v is None or v['finite'] for v in g['per_tensor'].values()))
            self.assertEqual(g['none_grad_tensors'], sorted(SOURCE_DISCONNECTED_PARAMETERS))
            self.assertEqual(g['unexpected_disconnected_tensors'], [])
            self.assertEqual(g['tensors_with_grad'] + len(g['none_grad_tensors']), r['optimizer']['optimized_tensors'])
            self.assertEqual(g['checked'], 'after backward, before optimizer.step')

    def test_nonzero_fc_and_backbone_gradients(self):
        for r in self.runs:
            g = r['gradients']
            self.assertEqual(g['fc_weight']['nonzero_elements'], 7 * 512)
            self.assertGreater(g['fc_bias']['nonzero_elements'], 0)
            self.assertTrue(g['fc_bias']['finite'])
            for stage in ('conv1.weight', 'layer1.0.conv1.weight', 'layer2.0.conv1.weight', 'layer3.0.conv1.weight',
                          'layer4.0.conv1.weight'):
                self.assertGreater(g['per_tensor'][stage]['nonzero_elements'], 0, stage)
            self.assertEqual(g['tensors_with_nonzero_grad'], g['tensors_with_grad'])
            self.assertEqual(r['encoder']['source_unused_modules'], list(SOURCE_UNUSED_MODULES))

    # 17 parameter update -------------------------------------------------------------
    def test_parameter_update(self):
        for r in self.runs:
            u = r['parameter_update']
            self.assertTrue(u['fc_changed'] and u['all_finite_after'])
            self.assertGreater(u['backbone_changed_tensors'], 0)
            self.assertEqual(u['changed_tensors'], r['gradients']['tensors_with_nonzero_grad'])
            self.assertEqual(u['unchanged_tensors'], sorted(SOURCE_DISCONNECTED_PARAMETERS))
            self.assertEqual(u['changed_parameters_outside_optimizer_scope'], 0)
            self.assertLessEqual(u['sgd_first_step_formula_max_abs_deviation'], u['sgd_formula_tolerance'])
            self.assertNotEqual(u['aggregate_parameter_sha256_before'], u['aggregate_parameter_sha256_after'])
            self.assertEqual(u['buffers_changed_by_backward_or_step'], [])
            self.assertTrue(all(k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
                                for k in u['buffers_changed_by_train_mode_forward']))
            self.assertEqual(r['optimizer']['momentum_buffers_after_step'], u['changed_tensors'])
            e = r['encoder']
            self.assertTrue(e['mode']['default_training_all_modules'] and e['mode']['model_train_called'])
            self.assertFalse(e['mode']['eval_called'] or e['batchnorm']['frozen_bn_or_running_stat_workaround'])
            self.assertGreater(e['batchnorm']['min_values_per_channel_at_B4'], 1)

    def test_bitwise_two_process_repeatability(self):
        a, b = self.runs
        self.assertEqual(hashlib.sha256(self.raw[0]).hexdigest(), hashlib.sha256(self.raw[1]).hexdigest())
        for key in ('aggregate_parameter_sha256_before', 'aggregate_parameter_sha256_after', 'parameter_sha256_after'):
            self.assertEqual(a['parameter_update'][key], b['parameter_update'][key])
        self.assertEqual(a['synthetic_batch']['input']['sha256'], b['synthetic_batch']['input']['sha256'])
        self.assertEqual(a['synthetic_batch']['target_sha256'], b['synthetic_batch']['target_sha256'])
        self.assertEqual(a['loss']['value_float32_hex'], b['loss']['value_float32_hex'])
        self.assertEqual(a['gradients']['per_tensor'], b['gradients']['per_tensor'])

    # 18 no checkpoint ----------------------------------------------------------------
    def test_no_checkpoint_save_or_load(self):
        for r in self.runs:
            self.assertEqual((r['counters']['checkpoint_saves'], r['counters']['checkpoint_loads']), (0, 0))
            self.assertFalse(r['checkpoint_created'] or r['checkpoint_loaded'] or r['auxiliary_encoder_trained'])
        tree = ast.parse(HARNESS.read_text())
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertFalse({'torch.save', 'torch.load', 'save', 'load', 'load_state_dict', 'state_dict'} & calls)
        for name, flag in ELIGIBILITY.items():
            self.assertTrue(flag, name)
            for r in self.runs:
                self.assertIs(r[name], True)

    # 19 zero benchmark access --------------------------------------------------------
    def test_zero_benchmark_access(self):
        for r in self.runs:
            fw = r['firewall']
            self.assertEqual(fw['denied'], [])
            self.assertEqual(fw['attempts'], {k: 0 for k in Firewall.CLASSES})
            self.assertGreater(fw['event_counts']['open'], 0)
            self.assertFalse(r['benchmark_data_access'] or r['TRAIN_access'] or r['VAL_access'] or r['TEST_access'])
            self.assertTrue(all(Path(a[0]).name in ('git', 'nvidia-smi') or a in ALLOWED_EXACT_ARGV
                                for a in fw['subprocesses']))
        text = HARNESS.read_text()
        for token in ('manifests/', '.parquet', 'faces_256', 'ImageFolder', 'DataLoader(', 'artifact_probe_classes'):
            self.assertNotIn(token, text)

    # 20, 21 no main training, no scientific run root -------------------------------
    def test_no_main_training_and_no_scientific_run_root(self):
        for r in self.runs:
            self.assertFalse(r['main_difffas_model_constructed'] or r['main_difffas_training'] or
                             r['scientific_training'] or r['synthetic_bank'])
            self.assertEqual(r['firewall']['attempts']['scientific_run_root'], 0)
        text = HARNESS.read_text()
        for token in ('unet_autoenc', 'make_model', 'training_losses', 'FAS_train', 'runs/m6'):
            self.assertNotIn(token, text)
        self.assertEqual(BUILD_PARTS, ('builds', 'e07c_difffas', 'm6d6b'))
        log = LOG.read_text()
        after = [ln for ln in log.splitlines() if ln.startswith('scientific_paths_after:')]
        self.assertEqual(len(after), 2)
        self.assertTrue(all(ln == 'scientific_paths_after: runs=ABSENT e07c_root=ABSENT aux_seed_42=ABSENT '
                                  'encoder_final=ABSENT' for ln in after))

    def test_memory_observation_and_precision_labels(self):
        for r in self.runs:
            m = r['cuda_memory']
            self.assertEqual(m['label'], 'AUX_ENCODER_SYNTHETIC_TRAINING_STEP_MEMORY_OBSERVATION')
            self.assertFalse(m['qualifies_training_memory'] or m['qualifies_scientific_batch_size'])
            self.assertEqual(m['batch_size'], 4)
            p = r['precision']
            self.assertEqual(p['label'], 'ENGINEERING_QUALIFICATION_CONTROLS')
            self.assertFalse(p['freezes_scientific_precision_policy'])
            self.assertEqual(r['fidelity'], 'CONTROLLED_ADAPTATION')

    # live Torch (GPU host, CUDA hidden): construction + graph walk, NO backward / NO step
    @unittest.skipUnless(HAS_TORCH, 'Torch unavailable on this host; CUDA evidence validated from process JSON')
    def test_live_cpu_graph_connectivity_without_backward(self):
        import torch
        from methods.difffas.encoder import encoder_model
        with torch.random.fork_rng(devices=[]), encoder_model(self.adapter.config) as enc:
            torch.manual_seed(SEED)
            self.assertTrue(all(m.training for m in enc.modules()))
            sgd = torch.optim.SGD(enc.parameters(), **HYPER)
            self.assertEqual([id(p) for p in sgd.param_groups[0]['params']], [id(p) for p in enc.parameters()])
            outs = enc(torch.zeros(2, 3, 256, 256))
            self.assertEqual(list(outs[3].shape), [2, K])
            loss = torch.nn.CrossEntropyLoss()(outs[3], torch.tensor([0, 6]))
            # seen maps id -> node so every visited wrapper stays alive (ids of freed wrappers get reused)
            reached, stack, seen = set(), [loss.grad_fn], {}
            while stack:
                fn = stack.pop()
                if fn is None or id(fn) in seen:
                    continue
                seen[id(fn)] = fn
                if hasattr(fn, 'variable'):
                    reached.add(id(fn.variable))
                stack.extend(f for f, _ in fn.next_functions)
            names = {id(p): n for n, p in enc.named_parameters()}
            disconnected = sorted(n for i, n in names.items() if i not in reached)
            self.assertEqual(disconnected, sorted(SOURCE_DISCONNECTED_PARAMETERS))
            self.assertTrue(all(p.grad is None for p in enc.parameters()))  # no backward was run
            self.assertEqual(len(sgd.state), 0)                              # no step was run


if __name__ == '__main__':
    unittest.main()
