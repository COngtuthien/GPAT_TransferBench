"""E06c A1 contracts, source/weight checks and in-memory synthetic adaptation."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from methods.common.config import FrozenConfigError
from methods.common import learned
from methods.common.learned import PreparationError, effective_batch
from methods.dsdg import DSDGAdapter


class TestDSDG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = DSDGAdapter()
        cls.plan = cls.adapter.build_training_plan(42)

    def pair(self):
        return {'pair_id': 'SYNTHETIC_PAIR', 'dataset': 'casia_fasd',
                'source_spoof_id': 'SYNTHETIC_SPOOF', 'target_live_id': 'SYNTHETIC_LIVE',
                'split': 'TRAIN', 'seed': 42}

    def test_pin(self):
        self.assertEqual(self.plan['source']['commit'], '16b793a7564a4b9308cf94e62bdb2ffacb3a725a')
        self.assertIn('addition_module/DSDG/networks/light_cnn.py', self.plan['source']['files_sha256'])

    def test_pin_mismatch_rejected(self):
        original = learned.git_read
        def fake(root, *args):
            return 'bad' if args == ('rev-parse', 'HEAD') else original(root, *args)
        with patch.object(learned, 'git_read', side_effect=fake):
            with self.assertRaises(PreparationError):
                self.adapter.validate_source()

    def test_lightcnn_verified_without_deserialization(self):
        asset = self.plan['lightcnn']
        self.assertEqual(asset['size_bytes'], 123844849)
        self.assertEqual(asset['sha256'], 'd0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964')
        self.assertFalse(asset['deserialized'])
        self.assertEqual(asset['compatibility']['matched_key_count'], 60)
        self.assertEqual(asset['compatibility']['missing_key_count'], 0)
        self.assertEqual(asset['compatibility']['shape_mismatch_count'], 0)
        self.assertEqual(asset['compatibility']['unexpected_keys'], ['module.fc2.weight'])

    def test_lightcnn_wrong_hash_rejected(self):
        with patch.object(learned, 'sha256_file', return_value='0'*64):
            with self.assertRaisesRegex(PreparationError, 'SHA256 mismatch'):
                self.adapter.verify_lightcnn()

    def test_lightcnn_wrong_size_rejected(self):
        asset = dict(self.adapter.config['external_assets'][0], bytes=1)
        with self.assertRaisesRegex(PreparationError, 'size mismatch'):
            learned.verify_asset(asset)

    def test_lambda_pair_zero(self):
        self.assertEqual(self.plan['adaptation']['lambda_pair'], 0)
        cfg = copy.deepcopy(self.adapter.config)
        cfg['losses']['lambda_pair'] = 5
        with self.assertRaises(FrozenConfigError):
            DSDGAdapter(cfg)

    def test_identity_supervision_prohibited(self):
        cfg = copy.deepcopy(self.adapter.config)
        cfg['subject_id_global_consumed'] = True
        with self.assertRaises(FrozenConfigError):
            DSDGAdapter(cfg)

    def test_attack_class_override_rejected(self):
        cfg = copy.deepcopy(self.adapter.config)
        cfg['training']['attack_type'] = 4
        with self.assertRaises(FrozenConfigError):
            DSDGAdapter(cfg)

    def test_binary_label_mapping(self):
        self.assertEqual(self.adapter.binary_label('live'), 0)
        self.assertEqual(self.adapter.binary_label('spoof'), 1)
        for label in ('print', 'subject_1', 4, 'TEST'):
            with self.assertRaises(PreparationError):
                self.adapter.binary_label(label)

    def test_pair_projection_ignores_identity_and_attack(self):
        base = self.pair()
        poison = dict(base, target_subject=object(), source_subject=object(),
                      subject_id_global=object(), attack_macro=object(), attack_raw=object())
        self.assertEqual(self.adapter.project_pair(base), self.adapter.project_pair(poison))

    def test_pair_split_firewall(self):
        for split in ('VAL', 'TEST'):
            with self.assertRaises(PreparationError):
                self.adapter.project_pair(dict(self.pair(), split=split))

    def test_missing_pair_fields_rejected(self):
        with self.assertRaises(PreparationError):
            self.adapter.project_pair({'split': 'TRAIN'})

    def test_synthetic_sample_upstream_keys_and_single_ce_class(self):
        source = np.zeros((256,256,3), np.uint8)
        target = np.full_like(source, 255)
        sample = self.adapter.adapt_pair_arrays(self.pair(), source_rgb=source, target_rgb=target)
        self.assertEqual(set(sample), {'0', '1', 'type'})
        self.assertEqual(sample['0'].shape, (3,256,256))
        self.assertEqual(sample['type'], 0)
        self.assertEqual(sample['0'].dtype, np.float32)
        np.testing.assert_array_equal(sample['0'], 0)
        np.testing.assert_array_equal(sample['1'], 1)

    def test_no_implicit_image_resize(self):
        with self.assertRaises(PreparationError):
            self.adapter.adapt_pair_arrays(self.pair(), source_rgb=np.zeros((8,8,3)),
                                          target_rgb=np.zeros((8,8,3)))

    def test_native_effective_batch_no_preemptive_oom(self):
        b = self.plan['effective_batch']
        self.assertEqual((b['physical_batch_size'], b['gradient_accumulation_steps'], b['replica_factor']), (240,1,1))
        self.assertEqual(b['effective_batch_size'], 240)
        self.assertIsNone(b['oom_deviation_reason'])
        self.assertEqual(b['execution_policy'], 'E06C_REQUIRES_PHYSICAL_BATCH_240_FOR_OBJECTIVE_EQUIVALENCE')
        self.assertEqual(b['oom_behavior'], 'STOP_AND_RESOLVE')
        self.assertFalse(b['semantics_preserving_microbatch_implementation_demonstrated'])

    def test_generic_arithmetic_accounting_is_not_e06c_permission(self):
        for physical, steps in ((120,2), (60,4), (30,8)):
            b = effective_batch(self.adapter.config, physical_batch_size=physical,
                                gradient_accumulation_steps=steps, oom_reason='synthetic accounting example only')
            self.assertEqual(b['physical_batch_size']*b['gradient_accumulation_steps'], 240)

    def test_invalid_accumulation_rejected(self):
        for physical, steps in ((120,1), (30,7), (0,1), (True,240), (120,2.0)):
            with self.assertRaises(PreparationError):
                effective_batch(self.adapter.config, physical_batch_size=physical,
                                gradient_accumulation_steps=steps, oom_reason='synthetic example')

    def test_reduction_requires_oom_evidence(self):
        with self.assertRaises(PreparationError):
            effective_batch(self.adapter.config, physical_batch_size=120, gradient_accumulation_steps=2)

    def test_data_parallel_not_replica_multiplier(self):
        with self.assertRaises(PreparationError):
            effective_batch(self.adapter.config, physical_batch_size=60, replica_factor=4, oom_reason='example')

    def test_epoch_200_rule(self):
        policy = self.plan['checkpoint']
        self.assertEqual(policy['rule'], 'OFFICIAL_GENERATOR_EPOCH_200')
        self.assertEqual(policy['authoritative_path_basename'], 'netG_model_epoch_200_iter_0.pth')
        self.assertFalse(policy['checkpoint_written'])

    def test_test_cannot_select(self):
        with self.assertRaises(PreparationError):
            learned.checkpoint_plan(self.adapter.config, 42, selection_split='TEST')

    def test_no_val_best_override(self):
        cfg = copy.deepcopy(self.adapter.config)
        cfg['checkpoint']['selection_uses_val'] = True
        with self.assertRaises(FrozenConfigError):
            DSDGAdapter(cfg)

    def test_argument_translation(self):
        argv = self.plan['official_argument_vector']
        args = dict(zip(argv[::2], argv[1::2]))
        self.assertEqual(args['--batch_size'], '240')
        self.assertEqual(args['--all_epochs'], '200')
        self.assertEqual(args['--attack_type'], '1')
        self.assertEqual(args['--lambda_pair'], '0.0')
        self.assertEqual(args['--lr'], '0.0002')
        self.assertEqual(args['--save_epoch'], '10')
        self.assertNotIn('--train_list', args)  # never pass common pairs to identity-based GenDataset_s
        self.assertFalse(self.plan['argument_vector_is_launch_command'])

    def test_missing_framework_fails_closed(self):
        with patch.object(self.adapter, 'validate_environment', return_value={
                'missing_modules': ['torch'], 'reason': 'torch missing'}):
            with self.assertRaisesRegex(PreparationError, 'torch missing'):
                with self.adapter.official_components():
                    self.fail('binding must not succeed')

    def test_dataset_reader_gets_only_projected_sample_ids(self):
        from methods.dsdg.adapter import IDFreePairDataset
        calls = []
        def reader(sample_id):
            calls.append(sample_id)
            return np.zeros((256,256,3), np.uint8)
        records = [dict(self.pair(), source_subject=object(), target_subject=object())]
        dataset = IDFreePairDataset(self.adapter, records, reader)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['type'], 0)
        self.assertEqual(calls, ['SYNTHETIC_SPOOF', 'SYNTHETIC_LIVE'])
        self.assertNotIn('source_subject', dataset.records[0])

    def test_explicit_full_physical_batch_accepted(self):
        plan = self.adapter.build_training_plan(42, physical_batch_size=240,
                                               gradient_accumulation_steps=1, replica_factor=1)
        self.assertEqual(plan['effective_batch']['effective_batch_size'], 240)

    def test_e06c_rejects_120_times_2_even_with_oom_reason(self):
        with self.assertRaisesRegex(PreparationError,
                'gradient accumulation does not preserve the frozen batch-dependent objective automatically'):
            self.adapter.build_training_plan(42, physical_batch_size=120,
                gradient_accumulation_steps=2, replica_factor=1, oom_reason='synthetic OOM evidence')

    def test_e06c_rejects_60_times_4_even_with_oom_reason(self):
        with self.assertRaisesRegex(PreparationError, 'STOP_AND_RESOLVE'):
            self.adapter.build_training_plan(42, physical_batch_size=60,
                gradient_accumulation_steps=4, replica_factor=1, oom_reason='synthetic OOM evidence')

    def test_e06c_rejects_microbatches_without_oom_reason(self):
        with self.assertRaisesRegex(PreparationError, 'E06C_REQUIRES_PHYSICAL_BATCH_240'):
            self.adapter.build_training_plan(42, physical_batch_size=120, gradient_accumulation_steps=2)

    def test_e06c_rejects_wrong_effective_batch(self):
        with self.assertRaises(PreparationError):
            self.adapter.build_training_plan(42, physical_batch_size=120, gradient_accumulation_steps=1)

    def test_batch_loss_counterexample_on_synthetic_latents(self):
        # Pinned main(): abs(mean(z_nir)-mean(z_vis)) and
        # abs(mean(sum(z_cls*z_nir))). Opposing microbatch means cancel
        # in the full-batch objective, but not after per-microbatch abs.
        nir = np.concatenate((np.ones((120, 1)), -np.ones((120, 1))))
        vis, cls = np.zeros_like(nir), np.ones_like(nir)
        def mmd(a, b):
            return np.abs(a.mean(axis=0) - b.mean(axis=0)).mean()
        def ort(a, b):
            return np.abs((a*b).sum(axis=1).mean())
        for loss, a, b in ((mmd, nir, vis), (ort, cls, nir)):
            self.assertEqual(loss(a, b), 0)
            self.assertEqual((loss(a[:120], b[:120]) + loss(a[120:], b[120:]))/2, 1)
