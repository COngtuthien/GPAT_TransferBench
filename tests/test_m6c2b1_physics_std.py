"""Focused E04 contracts, source/assets and static plans; no benchmark input."""
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from methods.common import learned
from methods.common.config import load_method_config, FrozenConfigError, ROOT
from methods.common.learned import PreparationError
from methods.physics_std import PhysicsSTDAdapter
from methods.physics_std.assets import validate_assets, _NumpyOnlyUnpickler
from methods.physics_std.contract import validate_overlay, validate_q140
from methods.physics_std.geometry import GeometryAdapter
from methods.physics_std.source import validate_source


class TestE04Authority(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_method_config('E04')

    def test_base_snapshot_and_a4_binding(self):
        result = validate_overlay(self.cfg)
        self.assertTrue(self.cfg['_runtime']['snapshot_verified'])
        self.assertEqual(result['overlay']['base_config_sha256'], self.cfg['_runtime']['config_sha256'])
        self.assertEqual(result['A3']['sha256'], self.cfg['amendment_a3_sha256'])

    def test_changed_config_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['training']['batch_size'] = 1
        with self.assertRaises(FrozenConfigError):
            validate_overlay(cfg)

    def test_overlay_override_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'overlay.yaml'
            path.write_text('optimizer: {type: SGD}\n')
            with self.assertRaisesRegex(FrozenConfigError, 'changed immutable'):
                validate_overlay(self.cfg, overlay_path=path)

    def test_a4_depth_semantics(self):
        depth = validate_overlay(self.cfg)['overlay']['depth']
        self.assertEqual(depth['raster_output_dtype'], 'uint8')
        self.assertEqual(depth['resize_dtype'], 'uint8')
        self.assertEqual(depth['resize_interpolation'], 'cv2.INTER_AREA')
        self.assertEqual(depth['post_resize_cast'], 'float32')
        self.assertEqual(depth['post_resize_scale'], '1/255')

    def test_a4_adam_and_provenance(self):
        a4 = validate_overlay(self.cfg)['overlay']
        self.assertEqual(a4['optimizer'], {'type':'Adam', 'semantic_reference':'tf.train.AdamOptimizer',
                         'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8, 'weight_decay':0})
        self.assertEqual(a4['provenance']['optimizer_resolution'],
                         'BENCHMARK_DEFINED_NEAREST_OFFICIAL_PREDECESSOR_BEHAVIOR')

    def test_q140_length_unique_hash_anchor_order(self):
        q = validate_q140(self.cfg)
        frozen = json.loads((ROOT/self.cfg['controlled_reconstruction']['q140_vertex_set']['frozen_list']).read_text())
        self.assertEqual(len(q['vertex_indices']), 140)
        self.assertEqual(len(set(q['vertex_indices'])), 140)
        self.assertEqual(q['anchor_indices'], frozen['vertex_indices'][:68])
        self.assertEqual(len(q['fps_indices']), 72)
        self.assertEqual(q['vertex_indices_sha256'],
                         '1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f')
        self.assertTrue(q['image_independent'])
        self.assertFalse(q['rederived'])

    def test_anchor_permutation_rejected(self):
        q = json.loads((ROOT/self.cfg['controlled_reconstruction']['q140_vertex_set']['frozen_list']).read_text())
        q['vertex_indices'][0], q['vertex_indices'][1] = q['vertex_indices'][1], q['vertex_indices'][0]
        q['vertex_indices_sha256'] = hashlib.sha256(json.dumps(q['vertex_indices'], separators=(',',':')).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'q.json'
            path.write_text(json.dumps(q))
            with self.assertRaises(FrozenConfigError):
                validate_q140(self.cfg, path=path)


class TestE04SourcesAssets(unittest.TestCase):
    def setUp(self):
        self.cfg = load_method_config('E04')

    def test_source_valid(self):
        result = validate_source(self.cfg)
        self.assertEqual(result['commit'], self.cfg['external_assets']['geometry_engine']['pinned_commit'])
        self.assertEqual(result['verification'], 'PASS')

    def test_wrong_source_commit_rejected(self):
        original = learned.git_read
        def altered(root, *args):
            return '0'*40 if args == ('rev-parse','HEAD') else original(root,*args)
        with patch.object(learned, 'git_read', side_effect=altered):
            with self.assertRaisesRegex(PreparationError, 'commit mismatch'):
                validate_source(self.cfg)

    def test_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PreparationError, 'missing pinned source'):
                validate_source(self.cfg, source_root=tmp)

    def test_actual_assets_verified_without_deserialization(self):
        result = validate_assets(self.cfg)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(not a['deserialized'] for a in result.values()))

    def test_missing_asset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PreparationError, 'unavailable'):
                learned.verify_asset({'external_runtime_path':str(Path(tmp)/'absent')})

    def test_wrong_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'synthetic.txt'; p.write_text('test')
            with self.assertRaisesRegex(PreparationError, 'size mismatch'):
                learned.verify_asset({'external_runtime_path':str(p), 'bytes':1})

    def test_wrong_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'synthetic.txt'; p.write_text('test')
            with self.assertRaisesRegex(PreparationError, 'SHA256 mismatch'):
                learned.verify_asset({'external_runtime_path':str(p), 'sha256':'0'*64})

    def test_restricted_pickle_rejects_arbitrary_globals(self):
        with self.assertRaisesRegex(PreparationError, 'non-array pickle global'):
            _NumpyOnlyUnpickler(io.BytesIO(pickle.dumps(eval))).load()


class TestE04Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = GeometryAdapter()

    def test_native_anchor_order_verified(self):
        self.assertEqual(len(self.geometry.q['anchor_indices']), 68)
        self.assertEqual(self.geometry.triangles.shape, (76073,3))

    def test_official_reconstruction_on_synthetic_parameters(self):
        g = self.geometry
        param = np.zeros(g.inference['param_dim'], dtype=np.float32)
        param[:12] = np.eye(3,4, dtype=np.float32).ravel()
        result = g.reconstruct(param, [0,0,256,256])
        self.assertEqual(result['dense_vertices'].shape, (38365,3))
        self.assertEqual(result['q140_vertices'].shape, (140,3))
        self.assertEqual(result['dense_vertices'][:,2].min(), 0)
        np.testing.assert_array_equal(result['q140_vertices'],
            result['dense_vertices'][g.q['vertex_indices']])
        np.testing.assert_array_equal(result['dense_vertices'],
                                      g.reconstruct(param,[0,0,256,256])['dense_vertices'])

    def test_invalid_parameters_refused(self):
        with self.assertRaises(PreparationError):
            self.geometry.reconstruct(np.zeros(10, dtype=np.float32), [0,0,256,256])

    def test_invalid_roi_refused(self):
        with self.assertRaises(PreparationError):
            self.geometry.reconstruct(np.zeros(62, dtype=np.float32), [0,0,0,0])


class TestE04Plans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = PhysicsSTDAdapter()
        cls.plans = [cls.adapter.build_training_plan(seed) for seed in (42,1337,2026)]

    def test_three_seeds_identical_fixed_geometry(self):
        self.assertEqual([p['seed'] for p in self.plans], [42,1337,2026])
        for field in ('q140', 'external_assets', 'source'):
            self.assertTrue(all(p[field] == self.plans[0][field] for p in self.plans))

    def test_scientific_training_values(self):
        p = self.plans[0]
        self.assertEqual(p['fidelity_class'], 'CONTROLLED_ADAPTATION')
        self.assertEqual(p['training']['batch_size'], 8)
        self.assertEqual(p['optimizer']['learning_rate'], 5e-5)
        self.assertEqual(p['optimizer']['lr_schedule'],
                         {'type':'step_divide','factor':10,'every_iterations':45000})
        self.assertEqual(p['losses']['alpha_1_L_depth'], 100)
        self.assertEqual(p['depth_target']['A3']['K'], 32)

    def test_plan_contains_all_authority_hashes(self):
        p = self.plans[0]
        for digest in (p['base_config_sha256'], p['a4_overlay_sha256'],
                       p['contracts']['A3']['sha256'], p['contracts']['A4']['sha256']):
            self.assertEqual(len(digest), 64)

    def test_checkpoint_end_and_zero_extra_steps(self):
        p = self.plans[0]['checkpoint']
        self.assertEqual(p['terminal_iteration'], 150000)
        self.assertEqual(p['rule'], 'BASELINE_FINAL_STATE_V1')
        self.assertEqual(p['extra_optimizer_steps'], 0)
        self.assertTrue(p['save_terminal_if_cadence_misses'])
        self.adapter.validate_checkpoint_event(42, iteration=150000, terminal=True, selected=True)

    def test_early_checkpoint_cannot_be_selected(self):
        with self.assertRaises(PreparationError):
            self.adapter.validate_checkpoint_event(42, iteration=149999, selected=True)

    def test_extra_step_checkpoint_rejected(self):
        with self.assertRaises(PreparationError):
            self.adapter.validate_checkpoint_event(42, iteration=150000, terminal=True, extra_optimizer_steps=1)

    def test_test_and_val_cannot_select(self):
        for split in ('TEST','VAL'):
            with self.subTest(split=split), self.assertRaises(PreparationError):
                self.adapter.build_training_plan(42, selection_split=split)

    def test_test_firewall(self):
        p = self.plans[0]
        self.assertFalse(p['data_firewall']['TEST']['allowed'])
        self.assertEqual(p['data_firewall']['TEST']['used_for'], [])
        self.assertFalse(p['checkpoint']['selection_uses_val'])
        self.assertFalse(p['checkpoint']['selection_uses_test'])
        self.assertFalse(p['training_launched'])

    def test_unfrozen_seed_rejected(self):
        with self.assertRaises(PreparationError):
            self.adapter.build_training_plan(43)

    def test_optimizer_hook_mapping_without_framework_execution(self):
        tf = Mock()
        tf.train.exponential_decay.return_value = 'symbolic_lr'
        self.adapter.build_optimizer(tf, 'iteration')
        tf.train.exponential_decay.assert_called_once_with(5e-5,'iteration',45000,0.1,staircase=True)
        tf.train.AdamOptimizer.assert_called_once_with('symbolic_lr',beta1=0.9,beta2=0.999,epsilon=1e-8)


if __name__ == '__main__':
    unittest.main()
