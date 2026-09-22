"""Architecture/blur contracts, synthetic dispatch; optional real Torch CPU test."""
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from methods.common.learned import PreparationError
from methods.pcgan import PCGANAdapter, blur_inputs


class TestArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = PCGANAdapter().build_training_plan(42)

    def test_canonical_spatial_and_output_shape(self):
        arch = self.plan['architecture']
        self.assertEqual(arch['resolution'], 256)
        self.assertEqual(arch['z_pat_shape'], [8, 128, 128])
        settings = {s['target']: s['value'] for s in arch['settings']}
        self.assertEqual(settings['opt.netE_num_downsampling_sp'], 1)
        self.assertEqual(settings['opt.spatial_code_ch'], 8)
        self.assertEqual(arch['output_shape_chw'], [3, 256, 256])

    def test_stylegan2_evidence_and_no_adain_substitution(self):
        arch = self.plan['architecture']
        self.assertEqual(arch['modulation'], 'StyleGAN2-style modulation/demodulation')
        refs = arch['source_evidence']
        self.assertEqual(refs['modulation_demodulation']['symbol'], 'ModulatedConv2d.forward')
        self.assertEqual(refs['discriminator']['symbol'], 'StyleGAN2Discriminator.__init__')
        for evidence in refs.values():
            self.assertEqual(evidence['sha256'], self.plan['source']['files_sha256'][evidence['file']])
            self.assertGreater(evidence['line'], 0)
        self.assertFalse(arch['stylegan_v1_adain_substitution'])
        self.assertIn('demodulate=False', arch['to_rgb'])

    def test_pinned_defaults_have_provenance(self):
        defaults = self.plan['architecture']['inherited_source_defaults']
        self.assertEqual(defaults['netE']['value'], 'StyleGAN2Resnet')
        self.assertEqual(defaults['netG']['value'], 'StyleGAN2Resnet')
        self.assertEqual(defaults['netD']['value'], 'StyleGAN2')
        for default in defaults.values():
            self.assertIn(default['source_file'], self.plan['source']['files_sha256'])
            self.assertEqual(default['provenance'], 'A2_PINNED_EXECUTABLE_ARCHITECTURE')

    def test_a5_blur_exact_mapping(self):
        op = self.plan['blur']
        self.assertEqual(op['framework_reference'], 'torch.nn.functional.avg_pool2d')
        self.assertEqual((op['kernel_size'], op['stride'], op['padding']), (2, 2, 0))
        self.assertIs(op['ceil_mode'], False)
        self.assertIs(op['count_include_pad'], False)
        self.assertEqual((op['input_resolution'], op['output_resolution']), (256, 128))
        self.assertEqual(op['application_branches'], ['x_tgt', 'G(z_pat_src, z_con_tgt)'])
        self.assertFalse(op['detach_due_to_blur'])
        self.assertEqual(op['authorized_fallbacks'], [])
        self.assertEqual(self.plan['contracts']['overlay']['provenance'],
                         'BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION')

    def test_two_branch_dispatch_without_detach_or_fallback(self):
        calls = []
        class Tensor:
            ndim = 4
            shape = (1, 3, 256, 256)
            def is_floating_point(self):
                return True
            def detach(self):
                raise AssertionError('blur must not detach')
        def pool(x, **kwargs):
            calls.append((x, kwargs))
            return ('pooled', x)
        module = SimpleNamespace(Tensor=Tensor, nn=SimpleNamespace(functional=SimpleNamespace(avg_pool2d=pool)))
        target, generated = Tensor(), Tensor()
        with patch.dict(sys.modules, {'torch': module}):
            result = blur_inputs(target, generated)
        self.assertEqual([call[0] for call in calls], [target, generated])
        self.assertEqual(calls[0][1], dict(kernel_size=2, stride=2, padding=0,
                                          ceil_mode=False, count_include_pad=False))
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(result, (('pooled', target), ('pooled', generated)))

    def test_invalid_blur_shape_rejected_before_pooling(self):
        class Tensor:
            ndim = 4
            shape = (1, 3, 128, 128)
        with patch.dict(sys.modules, {'torch': SimpleNamespace(Tensor=Tensor)}):
            with self.assertRaisesRegex(PreparationError, 'canonical resolution'):
                blur_inputs(Tensor(), Tensor())

    @unittest.skipUnless(importlib.util.find_spec('torch'),
                         'PyTorch unavailable: real CPU avg_pool2d/autograd execution requires torch; static dispatch verified separately')
    def test_real_torch_pooling_and_gradients(self):
        import torch
        x = torch.arange(256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256).requires_grad_()
        y = torch.ones_like(x, requires_grad=True)
        bx, by = blur_inputs(x, y)
        expected = x.detach().numpy().reshape(1, 1, 128, 2, 128, 2).mean(axis=(3, 5))
        np.testing.assert_array_equal(bx.detach().numpy(), expected)
        self.assertTrue(torch.equal(by, torch.ones_like(by)))
        self.assertTrue(torch.equal(bx, blur_inputs(x, y)[0]))
        (bx.sum() + by.sum()).backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, .25)))
        self.assertTrue(torch.equal(y.grad, torch.full_like(y, .25)))


if __name__ == '__main__':
    unittest.main()
