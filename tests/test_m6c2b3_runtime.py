"""E07c seed/plan/diffusion/checkpoint validation without model execution."""
import json
import os
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np

from methods.common.learned import PreparationError, checkpoint_metadata
from methods.difffas import DiffFASAdapter
from methods.difffas.seed_adapter import apply_seed


class TestDiffFASRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DiffFASAdapter()
        cls.plans = [cls.adapter.build_training_plan(seed) for seed in cls.adapter.config['seeds']['experiment_seeds']]
        cls.plan = cls.plans[0]

    def test_three_main_seeds_one_shared_encoder(self):
        self.assertEqual([p['seed'] for p in self.plans], [42,1337,2026])
        for p in self.plans:
            self.assertEqual(p['auxiliary_checkpoint'], self.plan['auxiliary_checkpoint'])
            self.assertEqual((p['auxiliary_plan']['seed'],p['auxiliary_plan']['runs']), (42,1))
            self.assertEqual(p['seed_adapter']['framework_seed'], p['seed'])
            json.dumps(p)

    def test_seed_adapter_complete_surface(self):
        p = self.plan['seed_adapter']
        self.assertEqual(p['replaces_hardcoded_seed'], 1)
        self.assertEqual(p['launch_environment'], {'PYTHONHASHSEED':'42'})
        self.assertEqual(p['cuda_seed_calls'], ['torch.cuda.manual_seed','torch.cuda.manual_seed_all'])
        self.assertFalse(p['cudnn_benchmark'])
        self.assertTrue(p['cudnn_deterministic'])
        self.assertFalse(p['force_deterministic_algorithms'])

    def test_seed_hook_dispatch_with_mock_no_gpu(self):
        module = Mock()
        samples = []
        with patch.dict(os.environ, {'PYTHONHASHSEED':'1337'}), patch(
                'importlib.import_module', return_value=module):
            for _ in range(2):
                apply_seed(1337, cuda=True)
                samples.append((random.random(), np.random.rand()))
        self.assertEqual(samples[0], samples[1])
        module.manual_seed.assert_called_with(1337)
        module.cuda.manual_seed.assert_called_with(1337)
        module.cuda.manual_seed_all.assert_called_with(1337)
        module.use_deterministic_algorithms.assert_not_called()

    def test_hash_seed_must_be_set_at_launch(self):
        with patch.dict(os.environ, {'PYTHONHASHSEED':'wrong'}), self.assertRaises(PreparationError):
            apply_seed(42, cuda=False)

    def test_auxiliary_seed_cannot_follow_main_seed(self):
        with self.assertRaisesRegex(PreparationError, 'one frozen seed'):
            apply_seed(1337, cuda=False, auxiliary=True)

    def test_main_invalid_seed_rejected(self):
        for seed in (1,True,42.0,'42'):
            with self.subTest(seed=seed), self.assertRaises(PreparationError):
                self.adapter.build_training_plan(seed)

    def test_main_training_values(self):
        t = self.plan['training']
        self.assertEqual((t['input_resolution'],t['max_epochs'],t['batch_size']), (256,400,4))
        self.assertEqual(t['normalize'], '[0.5,0.5,0.5] / [0.5,0.5,0.5]')
        self.assertEqual((t['guidance_prob'],t['means_size'],t['var_size']), (.2,5,3))
        self.assertFalse(t['use_pair'])
        self.assertEqual(t['model_in_channels'], 3)

    def test_diffusion_exact(self):
        self.assertEqual(self.plan['diffusion'], {'schedule':'linear','n_timestep':1000,
            'linear_start':1e-4,'linear_end':2e-2,'model_mean_type':'EPSILON','variance':'LEARNED_RANGE'})

    def test_optimizer_scheduler_ema(self):
        o = self.plan['optimizer']
        self.assertEqual((o['name'],o['learning_rate']), ('AdamW',1e-5))
        self.assertEqual(o['scheduler'], {'type':'cycle','lr':1e-5,'n_iter':2400000,'warmup':5000,'decay':['linear','flat']})
        self.assertEqual(o['ema'], {'decay':.9999,'before_warmup':0.0})

    def test_sampler_model_not_ema(self):
        s = self.plan['sampler']
        self.assertEqual((s['algorithm'],s['DDIM_skip'],s['sample_initial_noise'],s['effective_steps'],s['cond_scale']), ('ddim',10,250,25,2.0))
        self.assertEqual(s['timesteps_ascending'], list(range(0,250,10)))
        self.assertEqual(s['tensor_set_loaded'], 'model')
        self.assertEqual(self.plan['source_evidence']['sampler_tensor_set'], 'model')
        self.assertFalse(s['sampling_executed'])

    def test_terminal_checkpoint_rule(self):
        cp = self.plan['checkpoint']
        self.assertEqual((cp['rule'],cp['final_epoch']), ('BASELINE_FINAL_STATE_V1',400))
        self.assertEqual(cp['cadence'], 'save_checkpoints_every_10000_iters')
        self.assertTrue(cp['save_terminal_if_cadence_misses'])
        self.assertEqual(cp['extra_optimizer_steps'], 0)
        self.assertFalse(cp['checkpoint_written'])

    def test_checkpoint_events_and_zero_extra_steps(self):
        self.assertFalse(self.adapter.validate_checkpoint_event(42,completed_epochs=400,terminal=True)['checkpoint_written'])
        for kwargs in ({'completed_epochs':399,'selected':True}, {'completed_epochs':401},
                       {'completed_epochs':400,'extra_optimizer_steps':1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(PreparationError):
                self.adapter.validate_checkpoint_event(42,**kwargs)

    def test_common_logging_checkpoint_metadata(self):
        # Synthetic metadata only: no checkpoint file or scientific evidence.
        args=dict(epoch=400,global_step=10,path='metadata_only',file_size_bytes=1,
                  sha256='a'*64,checkpoint_type='terminal',selected_for_final=True)
        result=checkpoint_metadata(self.adapter.config,42,**args)
        self.assertEqual(result['selection_reason'],'BASELINE_FINAL_STATE_V1')
        args['epoch']=399
        with self.assertRaises(PreparationError):
            checkpoint_metadata(self.adapter.config,42,**args)

    def test_no_val_or_test_selection(self):
        for split in ('TRAIN','VAL','TEST'):
            with self.subTest(split=split), self.assertRaises(PreparationError):
                self.adapter.build_training_plan(42,selection_split=split)
        self.assertFalse(self.plan['data_firewall']['TEST']['allowed'])
        self.assertEqual(self.plan['data_firewall']['VAL']['used_for'],['diagnostics_only'])

    def test_identity_hashes_in_all_plans(self):
        for p in self.plans:
            self.assertEqual(p['config_sha256'],self.adapter.config['_runtime']['config_sha256'])
            self.assertEqual(p['adaptation_config_sha256'],self.adapter.config['adaptation_semantics']['frozen_config_sha256'])
            self.assertEqual(p['a6_overlay_sha256'],p['contracts']['overlay_sha256'])
            self.assertIn('A3',p['contracts']['documents'])
            self.assertIn('A6',p['contracts']['documents'])

    def test_static_no_execution_or_gpu_probe(self):
        for key in ('training_launched','model_constructed','checkpoint_created','diffusion_sampling','production_runtime_write'):
            self.assertFalse(self.plan[key])
        self.assertFalse(self.plan['environment']['gpu_probe_performed'])
        self.assertFalse(self.plan['environment']['final_execution_environment_frozen'])


if __name__ == '__main__':
    unittest.main()
