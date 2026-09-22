"""A5 contract tests using synthetic arrays; no PCGAN runtime implementation."""
import hashlib
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/methods/e05_pcgan.yaml"
SNAPSHOT = ROOT / "frozen_config_snapshot/configs/methods/e05_pcgan.yaml"
OVERLAY = ROOT / "configs/amendments/e05_a5_blur_operator_resolution.yaml"
SHA = "478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de"


def mathematical_blocks(x):
    """Test-only mathematical reference; not a framework/runtime implementation."""
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))


class TestM6A7E05BlurContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overlay = yaml.safe_load(OVERLAY.read_text())
        cls.op = cls.overlay["blur_operator"]
        cls.doc = (ROOT / cls.overlay["amendment_document"]).read_text()

    def test_original_yaml_unchanged(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), SHA)

    def test_original_snapshot_unchanged(self):
        self.assertEqual(hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(), SHA)
        self.assertEqual(BASE.read_bytes(), SNAPSHOT.read_bytes())

    def test_binding_and_identity(self):
        self.assertEqual(self.overlay["base_config_path"], str(BASE.relative_to(ROOT)))
        self.assertEqual(self.overlay["base_config_sha256"], SHA)
        self.assertEqual(self.overlay["amendment_id"], "A5_E05_BLUR_OPERATOR_RESOLUTION")
        self.assertEqual(self.overlay["status"], "OWNER_APPROVED_ADDITIVE_NON_DESTRUCTIVE")
        self.assertEqual(self.overlay["provenance"], "BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION")
        self.assertEqual(self.overlay["amendment_document"],
                         "docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A5_E05_Blur_Operator_Resolution.md")

    def test_only_blur_resolution_and_identity(self):
        self.assertEqual(set(self.overlay), {
            "amendment_id", "status", "base_config_path", "base_config_sha256",
            "amendment_document", "provenance", "blur_operator"})
        self.assertEqual(set(self.op), {
            "framework_reference", "kernel_size", "stride", "padding", "ceil_mode",
            "count_include_pad", "input_resolution", "output_resolution", "reduction",
            "learnable_parameters", "antialias_parameter", "align_corners",
            "boundary_extension", "application_branches", "application",
            "detach_due_to_blur", "authorized_fallbacks"})

    def test_exact_pooling_parameters(self):
        self.assertEqual(self.op["framework_reference"], "torch.nn.functional.avg_pool2d")
        self.assertEqual(self.op["kernel_size"], 2)
        self.assertEqual(self.op["stride"], 2)
        self.assertEqual(self.op["padding"], 0)
        self.assertIs(self.op["ceil_mode"], False)
        self.assertIs(self.op["count_include_pad"], False)
        self.assertEqual(self.op["reduction"], "arithmetic_mean_over_each_nonoverlapping_2x2_block")

    def test_canonical_geometry(self):
        self.assertEqual((self.op["input_resolution"], self.op["output_resolution"]), (256, 128))
        self.assertEqual(mathematical_blocks(np.zeros((2, 3, 256, 256))).shape,
                         (2, 3, 128, 128))

    def test_constant_preserved(self):
        x = np.full((2, 3, 256, 256), 0.375, dtype=np.float32)
        np.testing.assert_array_equal(mathematical_blocks(x),
                                      np.full((2, 3, 128, 128), 0.375, dtype=np.float32))

    def test_explicit_numeric_mean_without_rounding(self):
        x = np.array([[[[1., 2.], [3., 5.]]]])
        self.assertEqual(mathematical_blocks(x).item(), 2.75)

    def test_disjoint_windows_and_no_padding(self):
        x = np.zeros((1, 1, 4, 4))
        x[0, 0, 1, 1] = 8
        x[0, 0, 3, 3] = 4
        np.testing.assert_array_equal(mathematical_blocks(x), [[[[2., 0.], [0., 1.]]]])
        self.assertEqual(self.op["boundary_extension"], "NONE")

    def test_deterministic_and_channel_independent(self):
        x = np.arange(2 * 3 * 256 * 256, dtype=np.float64).reshape(2, 3, 256, 256)
        first = mathematical_blocks(x)
        np.testing.assert_array_equal(first, mathematical_blocks(x.copy()))
        expected = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2]
                    + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4
        np.testing.assert_array_equal(first, expected)

    def test_symmetric_loss_branch_contract(self):
        self.assertEqual(self.op["application_branches"], ["x_tgt", "G(z_pat_src, z_con_tgt)"])
        self.assertEqual(self.op["application"], "independently_before_existing_reconstruction_distance")
        self.assertIs(self.op["detach_due_to_blur"], False)
        self.assertIn("The SAME operator MUST be applied independently", self.doc)
        self.assertIn("Neither branch may be detached", self.doc)

    def test_no_fallback_or_extra_parameters(self):
        self.assertEqual(self.op["authorized_fallbacks"], [])
        self.assertEqual(self.op["learnable_parameters"], 0)
        self.assertEqual(self.op["antialias_parameter"], "NOT_APPLICABLE")
        self.assertEqual(self.op["align_corners"], "NOT_APPLICABLE")
        self.assertIn("No Gaussian, bilinear, bicubic, nearest-neighbor, StyleGAN2", self.doc)
        self.assertIn("fallback is authorized", self.doc)

    def test_other_loss_values_preserved(self):
        losses = yaml.safe_load(BASE.read_text())["losses"]
        self.assertEqual(losses["total"], "L_rec + L_recblur + L_advrec + L_advmix + L_pat")
        self.assertEqual(losses["weighting"], "ALL_UNIT_WEIGHTED")
        self.assertEqual(losses["alpha"], 0.2)
        self.assertEqual(losses["beta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
