"""Runtime identity rejection tests and production-extension synthetic semantics."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from methods.common.config import load_method_config
from methods.common.learned import PreparationError
from methods.physics_std.runtime import runtime_manifest, extension_root
from methods.physics_std.renderer import DepthRenderer, depth_target


class TestRuntimeIdentity(unittest.TestCase):
    def setUp(self):
        self.config = load_method_config('E04')
        engine = self.config['external_assets']['geometry_engine']
        self.record = {'source_pin':engine['pinned_commit'],
                       'qualification_scope':'E04_FIXED_GEOMETRY_DEPTH_ONLY',
                       'assets':{a['path']:{'sha256':a['sha256'],'bytes':a['bytes']} for a in engine['assets']}}

    def check_bad(self, record, message):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'runtime.json'
            path.write_text(json.dumps(record))
            with patch.dict(os.environ, {'GPAT_E04_GEOMETRY_RUNTIME':str(path)}):
                with self.assertRaisesRegex(PreparationError, message):
                    runtime_manifest(self.config)

    def test_wrong_pin_rejected(self):
        record = copy.deepcopy(self.record)
        record['source_pin'] = '0'*40
        self.check_bad(record, 'source/scope')

    def test_training_scope_rejected(self):
        record = copy.deepcopy(self.record)
        record['qualification_scope'] = 'FULL_TRAINING'
        self.check_bad(record, 'source/scope')

    def test_wrong_asset_hash_rejected(self):
        record = copy.deepcopy(self.record)
        record['assets']['configs/tri.pkl']['sha256'] = '0'*64
        self.check_bad(record, 'asset identity')

    def test_extra_asset_rejected(self):
        record = copy.deepcopy(self.record)
        record['assets']['weights/extra.pth'] = {'bytes':1,'sha256':'0'*64}
        self.check_bad(record, 'asset identity')


@unittest.skipUnless(os.environ.get('GPAT_E04_GEOMETRY_RUNTIME'), 'requires deployed qualified geometry runtime')
class TestProductionExtension(unittest.TestCase):
    def test_real_extension_larger_z_and_exact_conversion(self):
        import cv2
        from importlib.machinery import ExtensionFileLoader
        renderer = DepthRenderer()
        self.assertIsInstance(renderer._capture.extension.__loader__, ExtensionFileLoader)
        v = np.array([[16,16,1],[240,16,1],[16,240,1],
                      [16,16,3],[240,16,3],[16,240,3]],dtype=np.float32)
        t = np.array([[0,1,2],[3,4,5]],dtype=np.int32)
        a,b = renderer.render(v,t),renderer.render(v,t[::-1])
        self.assertTrue(a['scientific_output'])
        self.assertEqual(a['raster_uint8'][32,32],255)
        self.assertTrue(a['face_mask_256'][32,32])
        self.assertTrue(np.all(a['raster_uint8'][~a['face_mask_256']]==0))
        self.assertEqual(a['raster_uint8'].dtype,np.uint8)
        self.assertEqual(a['raster_uint8'].shape,(256,256))
        expected = np.clip(cv2.resize(a['raster_uint8'],(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)/255,0,1)
        np.testing.assert_array_equal(a['depth'],expected)
        for key in ('depth','raster_uint8','face_mask_256'):
            np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(a[key],renderer.render(v,t)[key])

    def test_tampered_extension_hash_rejected(self):
        record = json.loads(Path(os.environ['GPAT_E04_GEOMETRY_RUNTIME']).read_text())
        record['sim3dr']['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'runtime.json'; path.write_text(json.dumps(record))
            with patch.dict(os.environ,{'GPAT_E04_GEOMETRY_RUNTIME':str(path)}):
                with self.assertRaisesRegex(PreparationError,'compiled extension identity'):
                    DepthRenderer()

    def test_spoof_bypasses_all_geometry(self):
        from methods.physics_std.regressor import FixedGeometryRegressor
        from methods.physics_std.geometry import GeometryAdapter
        with patch.object(FixedGeometryRegressor,'fit',side_effect=AssertionError), \
             patch.object(GeometryAdapter,'reconstruct',side_effect=AssertionError), \
             patch.object(DepthRenderer,'render',side_effect=AssertionError):
            result = depth_target('spoof')['depth']
        self.assertEqual(result.dtype,np.float32)
        np.testing.assert_array_equal(result,np.zeros((32,32),np.float32))


if __name__ == '__main__':
    unittest.main()
