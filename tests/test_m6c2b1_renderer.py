"""Official kernel execution on synthetic meshes; never loads benchmark pixels."""
import shutil
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from methods.common.config import load_method_config
from methods.common.learned import PreparationError
from methods.physics_std.renderer import DepthRenderer, convert_live_depth, spoof_depth, depth_target
from tests.m6c2b1_kernel import SyntheticOfficialKernel


class TestDepthConversion(unittest.TestCase):
    def test_spoof_shape_dtype_exact_zeros(self):
        result = spoof_depth()
        self.assertEqual(result.shape, (32, 32))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(np.count_nonzero(result), 0)

    def test_spoof_never_calls_geometry(self):
        geometry = Mock(side_effect=AssertionError('must not reconstruct'))
        renderer = Mock(side_effect=AssertionError('must not render'))
        depth_target('spoof', reconstruct=geometry, renderer=renderer)
        geometry.assert_not_called()
        renderer.assert_not_called()

    def test_exact_uint8_resize_before_float(self):
        raster = np.zeros((256, 256), dtype=np.uint8)
        raster[:5, :8] = 1
        actual = convert_live_depth(raster)
        expected = cv2.resize(raster, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)/255
        wrong = cv2.resize(raster.astype(np.float32)/255, (32, 32), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(actual, expected)
        self.assertNotEqual(actual[0, 0], wrong[0, 0])

    def test_float_input_rejected(self):
        with self.assertRaisesRegex(PreparationError, 'uint8'):
            convert_live_depth(np.zeros((256, 256), dtype=np.float32))

    def test_wrong_dimensions_rejected(self):
        with self.assertRaises(PreparationError):
            convert_live_depth(np.zeros((32, 32), dtype=np.uint8))

    def test_non_depth_rgb_rejected(self):
        x = np.zeros((256, 256, 3), dtype=np.uint8)
        x[0, 0, 1] = 1
        with self.assertRaises(PreparationError):
            convert_live_depth(x)

    def test_full_range_and_channel_collapse(self):
        x = np.zeros((256, 256, 3), dtype=np.uint8)
        x[:128] = 255
        y = convert_live_depth(x)
        self.assertEqual(y.shape, (32, 32))
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual((float(y.min()), float(y.max())), (0, 1))

    def test_production_rejects_injected_kernel(self):
        with self.assertRaisesRegex(PreparationError, 'synthetic-only'):
            DepthRenderer(kernel=Mock())

    def test_live_rejects_synthetic_renderer(self):
        with self.assertRaisesRegex(PreparationError, 'official renderer'):
            depth_target('live', renderer=Mock(is_official=False), reconstruct=Mock())

    def test_unknown_label_rejected(self):
        with self.assertRaises(PreparationError):
            depth_target('TEST')


@unittest.skipUnless(shutil.which('c++'), 'c++ compiler required to execute pinned official rasterizer kernel')
class TestOfficialKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_method_config('E04')
        cls.kernel = SyntheticOfficialKernel(cls.config)
        cls.renderer = DepthRenderer(cls.config, synthetic_only=True, kernel=cls.kernel)

    @classmethod
    def tearDownClass(cls):
        cls.kernel.close()

    def mesh(self):
        # Coincident projected triangles with different z; one far, one near.
        return (np.array([[16,16,1],[240,16,1],[16,240,1],
                          [16,16,3],[240,16,3],[16,240,3]], dtype=np.float32),
                np.array([[0,1,2],[3,4,5]], dtype=np.int32))

    def test_triangle_projects_and_larger_z_wins(self):
        v, t = self.mesh()
        result = self.renderer.render(v, t)
        self.assertTrue(result['face_mask_256'][32,32])
        self.assertEqual(result['raster_uint8'][32,32], 255)
        self.assertFalse(result['scientific_output'])

    def test_order_invariance(self):
        v, t = self.mesh()
        a = self.renderer.render(v, t)
        b = self.renderer.render(v, t[::-1])
        for key in ('depth', 'raster_uint8', 'face_mask_256'):
            np.testing.assert_array_equal(a[key], b[key])

    def test_deterministic_repeat(self):
        v, t = self.mesh()
        a = self.renderer.render(v, t)
        b = self.renderer.render(v, t)
        self.assertEqual(a['depth'].tobytes(), b['depth'].tobytes())

    def test_background_exact_zero(self):
        v, t = self.mesh()
        result = self.renderer.render(v, t)
        self.assertFalse(result['face_mask_256'][0,0])
        self.assertEqual(result['raster_uint8'][0,0], 0)
        self.assertEqual(result['depth'][0,0], 0)

    def test_uint8_intermediate_shape_range_dtype(self):
        result = self.renderer.render(*self.mesh())
        self.assertEqual(result['raster_uint8'].dtype, np.uint8)
        self.assertEqual(result['raster_uint8'].shape, (256,256))
        self.assertEqual(result['depth'].dtype, np.float32)
        self.assertEqual(result['depth'].shape, (32,32))
        self.assertEqual(result['depth'].min(), 0)
        self.assertEqual(result['depth'].max(), 1)

    def test_interpolated_depth_uint8_quantization(self):
        v, t = self.mesh()
        v[3:,2] = [1,3,1]
        result = self.renderer.render(v, t[1:])
        # At (128,32), second barycentric weight = 112/224 = 1/2.
        self.assertEqual(result['raster_uint8'][32,128], 127)

    def test_zero_z_range_rejected(self):
        v, t = self.mesh()
        v[:,2] = 1
        with self.assertRaisesRegex(PreparationError, 'degenerate z range'):
            self.renderer.render(v, t)

    def test_degenerate_triangle_rejected(self):
        v, t = self.mesh()
        v[1] = v[0]
        with self.assertRaisesRegex(PreparationError, 'degenerate projected triangle'):
            self.renderer.render(v, t)

    def test_nonfinite_geometry_rejected(self):
        v, t = self.mesh()
        v[0,0] = np.nan
        with self.assertRaises(PreparationError):
            self.renderer.render(v, t)

    def test_out_of_bounds_indices_rejected(self):
        v, t = self.mesh()
        t[0,0] = len(v)
        with self.assertRaises(PreparationError):
            self.renderer.render(v, t)


class TestOfficialPythonBinding(unittest.TestCase):
    def test_official_extension_smoke_when_available(self):
        try:
            renderer = DepthRenderer()
        except PreparationError as exc:
            if 'Sim3DR_Cython unavailable' in str(exc):
                self.skipTest(str(exc) + '; C++ kernel tested separately, no model inference')
            raise
        v = np.array([[16,16,0],[240,16,1],[16,240,2]], dtype=np.float32)
        t = np.array([[0,1,2]], dtype=np.int32)
        first = renderer.render(v, t)
        self.assertTrue(first['scientific_output'])
        np.testing.assert_array_equal(first['depth'], renderer.render(v,t)['depth'])
