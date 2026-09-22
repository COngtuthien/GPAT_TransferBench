"""STDN contract, source and synthetic landmark tests; no pixels read."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from methods.common.config import load_method_config, FrozenConfigError
from methods.common.learned import PreparationError
from methods.common import learned
from methods.stdn import STDNAdapter


class TestSTDN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = STDNAdapter().prepare()
        cls.plan = cls.adapter.build_training_plan(42)

    def test_pin(self):
        self.assertEqual(self.plan['source']['commit'], 'c79f1f8c615d2b8471b3df29da881bb18dd54c90')
        self.assertEqual(self.plan['source']['verification'], 'PASS')

    def test_config_sha_snapshot(self):
        cfg = load_method_config('E03')
        self.assertEqual(self.plan['config_sha256'], cfg['_runtime']['config_sha256'])
        self.assertTrue(self.plan['snapshot_verified'])
        self.assertEqual((learned.ROOT/cfg['_runtime']['config_path']).read_bytes(),
                         (learned.ROOT/cfg['_runtime']['snapshot_path']).read_bytes())

    def test_seeds(self):
        self.assertEqual(self.adapter.config['seeds']['experiment_seeds'], [42, 1337, 2026])

    def test_batch_and_schedule_mapping(self):
        settings = {x['target']: x['value'] for x in self.plan['settings_mapping']}
        self.assertEqual(settings['BATCH_SIZE'], 2)
        self.assertEqual(settings['MAX_EPOCH'], 50)
        self.assertEqual(settings['STEPS_PER_EPOCH'], 2000)
        self.assertEqual(settings['LEARNING_RATE'], 6e-5)
        self.assertEqual(settings['NUM_EPOCHS_PER_DECAY']*settings['STEPS_PER_EPOCH'], 20000)

    def test_official_graph_binding(self):
        self.assertEqual(self.plan['upstream_bindings']['graph'], 'train._step')
        self.assertIn('model/utils.py', self.plan['source']['files_sha256'])
        self.assertIn('model/loss.py', self.plan['source']['files_sha256'])
        self.assertFalse(self.plan['training_launched'])

    def test_landmarks_identity(self):
        xy = np.arange(136, dtype=np.float32).reshape(68, 2)
        result = self.adapter.landmarks.from_cache({'landmarks_px256': xy})
        np.testing.assert_array_equal(result, xy/256)
        self.assertEqual(result.shape, (68, 2))
        self.assertEqual(result.dtype, np.float32)

    def test_exact_flip_permutation(self):
        # Independent iBUG-68 semantic oracle, one-based -> zero-based.
        expected = np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,
            27,26,25,24,23,22,21,20,19,18,28,29,30,31,36,35,34,33,32,
            46,45,44,43,48,47,40,39,38,37,42,41,55,54,53,52,51,50,49,
            60,59,58,57,56,65,64,63,62,61,68,67,66])-1
        np.testing.assert_array_equal(self.adapter.landmarks.permutation, expected)
        xy = np.arange(136, dtype=np.float32).reshape(68, 2)
        result = self.adapter.landmarks.from_cache({'landmarks_px256': xy}, flip=True)
        target = xy/256
        target[:, 0] = 1-target[:, 0]
        np.testing.assert_array_equal(result, target[expected])
        self.assertEqual(self.plan['input_adapter']['horizontal_flip'], 'enabled')

    def test_double_flip_restores_semantic_order(self):
        # Binary fractions make coordinate arithmetic exact, including boundaries.
        xy = np.arange(136, dtype=np.float32).reshape(68, 2)/256
        f = self.adapter.landmarks.flip_normalized
        np.testing.assert_array_equal(f(f(xy)), xy)

    def test_missing_coordinate_convention_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.landmarks.from_cache({'landmarks_norm': np.zeros((68,2))})

    def test_bad_landmarks_rejected(self):
        for xy in (np.zeros((67,2)), np.full((68,2), np.nan)):
            with self.assertRaises(ValueError):
                self.adapter.landmarks.from_cache({'landmarks_px256': xy})

    def test_ckpt_50_only(self):
        rule = self.plan['checkpoint']
        self.assertEqual(rule['rule'], 'OFFICIAL_LATEST_FINAL_CKPT_50')
        self.assertEqual(rule['authoritative_path_basename'], 'ckpt-50')
        self.assertFalse(rule['selection_uses_val'])
        self.assertFalse(rule['selection_uses_test'])
        self.assertEqual(rule['selection_scope'], 'WITHIN_SEED')

    def test_test_cannot_select(self):
        with self.assertRaises(PreparationError):
            learned.checkpoint_plan(self.adapter.config, 42, selection_split='TEST')

    def test_val_best_override_rejected(self):
        cfg = copy.deepcopy(self.adapter.config)
        cfg['checkpoint']['rule'] = 'VAL_BEST'
        with self.assertRaises(FrozenConfigError):
            STDNAdapter(cfg)

    def test_seed_override_rejected(self):
        cfg = copy.deepcopy(self.adapter.config)
        cfg['seeds']['experiment_seeds'] = [42]
        with self.assertRaises(FrozenConfigError):
            STDNAdapter(cfg)

    def test_source_pin_mismatch(self):
        original = learned.git_read
        def fake(root, *args):
            return 'bad' if args == ('rev-parse', 'HEAD') else original(root, *args)
        with patch.object(learned, 'git_read', side_effect=fake):
            with self.assertRaises(PreparationError):
                self.adapter.validate_source()

    def test_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PreparationError):
                STDNAdapter(source_root=tmp).prepare()

    def test_required_source_missing(self):
        original = Path.is_file
        def missing(path):
            return False if str(path).endswith('stdn/model/warp.py') else original(path)
        with patch.object(Path, 'is_file', missing):
            with self.assertRaises(PreparationError):
                self.adapter.validate_source()

    def test_missing_framework_binding_fails_closed(self):
        with patch.object(self.adapter, 'validate_environment', return_value={
                'missing_modules': ['tensorflow'], 'reason': 'tensorflow missing'}):
            with self.assertRaisesRegex(PreparationError, 'tensorflow missing'):
                with self.adapter.official_components():
                    self.fail('binding must not succeed')
