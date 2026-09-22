"""A4 contract checks; no model or benchmark-data execution."""
import hashlib
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/methods/e04_physics_std.yaml"
SNAPSHOT = ROOT / "frozen_config_snapshot/configs/methods/e04_physics_std.yaml"
OVERLAY = ROOT / "configs/amendments/e04_a4_execution_resolution.yaml"
DOC = ROOT / "docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A4_E04_Execution_Gap_Resolution.md"
STDN = ROOT / "third_party/source_cache/stdn"


class TestM6A6E04Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_bytes = BASE.read_bytes()
        cls.overlay = yaml.safe_load(OVERLAY.read_text())

    def test_frozen_config_and_snapshot_unchanged_and_hash_bound(self):
        self.assertEqual(self.base_bytes, SNAPSHOT.read_bytes())
        self.assertEqual(hashlib.sha256(self.base_bytes).hexdigest(),
                         self.overlay["base_config_sha256"])

    def test_overlay_contains_only_a4_resolution_fields(self):
        self.assertEqual(self.overlay["amendment_id"], "A4_E04_EXECUTION_GAP_RESOLUTION")
        self.assertEqual(set(self.overlay), {
            "amendment_id", "status", "base_config_path", "base_config_sha256",
            "amendment_document", "provenance", "depth", "optimizer"})
        self.assertEqual(self.overlay["base_config_path"], "configs/methods/e04_physics_std.yaml")
        self.assertEqual(self.overlay["depth"], {
            "raster_output_dtype": "uint8", "resize_dtype": "uint8",
            "resize_interpolation": "cv2.INTER_AREA", "post_resize_cast": "float32",
            "post_resize_scale": "1/255", "final_clip": [0, 1],
            "output_shape": [32, 32], "spoof_m0": "zeros((32,32), float32)"})
        self.assertEqual(self.overlay["optimizer"], {
            "type": "Adam", "semantic_reference": "tf.train.AdamOptimizer",
            "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0})

    def test_predecessor_is_pinned_and_constructs_adam(self):
        commit = subprocess.check_output(
            ["git", "-C", str(STDN), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(commit, self.overlay["provenance"]["predecessor_commit"])
        source_path = STDN / "model/model.py"
        self.assertIn("tf.train.AdamOptimizer(lr)", source_path.read_text())
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(),
                         "a422834d59d24db684cc28f470fdd6eb8b1658bfb40ba113393df627d6fc3743")

    def test_document_and_overlay_paths(self):
        self.assertTrue(DOC.is_file())
        self.assertEqual(self.overlay["amendment_document"], str(DOC.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
