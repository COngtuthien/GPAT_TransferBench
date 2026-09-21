"""M6C1 — E01 FAS-Aug deterministic contract. Synthetic arrays only; no benchmark image is read."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.config import load_method_config  # noqa: E402
from methods.fas_aug.e01 import (  # noqa: E402
    AssetIndex, E01Generator, ToyOperatorBackend, enumerate_assets, magnitude_for_level,
    operator_seed, operator_seed_preimage, select_operator,
)


class TestE01Seed(unittest.TestCase):
    def test_preimage_has_no_separator(self):
        self.assertEqual(operator_seed_preimage("PTR000001", 42), b"PTR00000142")
        self.assertEqual(operator_seed_preimage("PTR000001", 1337), b"PTR0000011337")
        self.assertEqual(operator_seed_preimage("PTR000001", 2026), b"PTR0000012026")

    def test_known_seed_vectors(self):
        """Frozen worked example in configs/methods/e01_fas_aug.yaml and Amendment A2 §A2-05."""
        self.assertEqual(operator_seed("PTR000001", 42), 795981663)
        self.assertEqual(operator_seed("PTR000001", 1337), 924982993)
        self.assertEqual(operator_seed("PTR000001", 2026), 1531213468)

    def test_matches_frozen_config_worked_example(self):
        we = load_method_config("E01")["pair_seed"]["worked_example"]
        self.assertEqual(operator_seed_preimage(we["pair_id"], we["global_seed"]).decode(),
                         we["preimage_text"])
        self.assertEqual(operator_seed(we["pair_id"], we["global_seed"]), we["operator_seed"])

    def test_deterministic(self):
        for _ in range(5):
            self.assertEqual(operator_seed("PTR004242", 1337), operator_seed("PTR004242", 1337))

    def test_different_global_seeds_differ(self):
        seeds = {operator_seed("PTR000123", g) for g in (42, 1337, 2026)}
        self.assertEqual(len(seeds), 3)

    def test_different_pairs_differ(self):
        seeds = {operator_seed(f"PTR{i:06d}", 42) for i in range(1, 50)}
        self.assertEqual(len(seeds), 49)

    def test_range_is_mod_2_31(self):
        for i in range(1, 200):
            self.assertTrue(0 <= operator_seed(f"PTR{i:06d}", 2026) < 2 ** 31)

    def test_rejects_bad_types(self):
        with self.assertRaises(TypeError):
            operator_seed("PTR000001", True)
        with self.assertRaises(ValueError):
            operator_seed("", 42)


class TestE01AssetEnumeration(unittest.TestCase):
    def test_lexicographic_and_creation_order_independent(self):
        names = ["b10.png", "b2.png", "b1.png", "a_9.png", "A1.png"]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            for n in names:
                (Path(d1) / n).write_bytes(b"x")
            for n in reversed(names):
                (Path(d2) / n).write_bytes(b"x")
            a, b = enumerate_assets(d1), enumerate_assets(d2)
        self.assertEqual(a, b, "enumeration must not depend on filesystem creation order")
        self.assertEqual(a, sorted(names, key=lambda s: s.encode("utf-8")))
        self.assertEqual(a[0], "A1.png")            # bytewise: uppercase sorts before lowercase
        self.assertEqual(a.index("b1.png"), a.index("b10.png") - 1)

    def test_count_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "background").mkdir()
            (Path(d) / "background" / "only.png").write_bytes(b"x")
            with self.assertRaises(ValueError):
                AssetIndex(d, {"background": 90}).load(["background"])


class TestE01OperatorAndMagnitude(unittest.TestCase):
    def setUp(self):
        self.cfg = load_method_config("E01")
        self.ops = self.cfg["generation"]["operator_set"]

    def test_round_robin_is_deterministic_and_covers_the_set(self):
        seen = {select_operator(f"PTR{i:06d}", self.ops)[1] for i in range(len(self.ops))}
        self.assertEqual(seen, set(self.ops))
        self.assertEqual(select_operator("PTR000003", self.ops), select_operator("PTR000003", self.ops))
        self.assertEqual(select_operator("PTR000003", self.ops)[0], 3 % len(self.ops))

    def test_magnitude_quantisation(self):
        num = self.cfg["generation"]["num_mag"]
        lo, hi = 0.01, 0.9
        self.assertEqual(magnitude_for_level(0, num, lo, hi), (0.0, lo))
        level, mag = magnitude_for_level(num - 1, num, lo, hi)
        self.assertEqual(level, 1.0)
        self.assertAlmostEqual(mag, hi)
        for k in range(num):
            lv, _ = magnitude_for_level(k, num, lo, hi)
            self.assertAlmostEqual(lv, k / (num - 1))

    def test_level_zero_is_a_noop_and_preserved(self):
        """level 0 must remain reachable: it is the official grid end, never skipped."""
        gen = E01Generator(self.cfg, backend=ToyOperatorBackend(), synthetic_only=True).prepare()
        found = [i for i in range(1, 400) if gen.recipe(f"PTR{i:06d}", 42)["is_level_zero_noop"]]
        self.assertTrue(found, "level 0 must be reachable under the official magnitude grid")
        img = np.full((16, 16, 3), 7, dtype=np.uint8)
        r = gen.generate_one(f"PTR{found[0]:06d}", 42, img)
        self.assertTrue(r.success)
        np.testing.assert_array_equal(r.output, img)
        self.assertEqual(r.metadata["level_k"], 0)
        self.assertEqual(r.metadata["magnitude"], r.metadata["magnitude_range"][0])


class TestE01Generation(unittest.TestCase):
    def test_same_input_pair_seed_gives_byte_identical_output(self):
        cfg = load_method_config("E01")
        img = (np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3) % 256).astype(np.uint8)
        a = E01Generator(cfg, backend=ToyOperatorBackend(), synthetic_only=True).prepare().generate_one("PTR000011", 42, img)
        b = E01Generator(cfg, backend=ToyOperatorBackend(), synthetic_only=True).prepare().generate_one("PTR000011", 42, img)
        self.assertTrue(a.success and b.success)
        self.assertTrue(np.array_equal(a.output, b.output))
        self.assertEqual(a.metadata["operator_seed"], b.metadata["operator_seed"])

    def test_different_seed_changes_the_recipe(self):
        gen = E01Generator(load_method_config("E01"), backend=ToyOperatorBackend(), synthetic_only=True).prepare()
        recipes = [gen.recipe("PTR000011", g) for g in (42, 1337, 2026)]
        self.assertEqual(len({r["operator_seed"] for r in recipes}), 3)

    def test_record_carries_required_provenance(self):
        gen = E01Generator(load_method_config("E01"), backend=ToyOperatorBackend(), synthetic_only=True).prepare()
        rec = gen.generate_one("PTR000011", 42, np.zeros((8, 8, 3), np.uint8)).to_record()
        for key in ("method_id", "experiment_seed", "pair_id", "metadata",
                    "generation_status", "failure_reason", "elapsed_seconds"):
            self.assertIn(key, rec)
        for key in ("operator_seed", "operator", "level", "magnitude", "asset"):
            self.assertIn(key, rec["metadata"])

    def test_toy_backend_is_never_marked_official(self):
        self.assertFalse(ToyOperatorBackend().is_official)



class TestOfficialBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from methods.fas_aug.official import OfficialFASAugBackend
        cls.backend = OfficialFASAugBackend().prepare()

    def test_official_identity_and_exact_dispatch(self):
        self.assertTrue(self.backend.is_official)
        self.assertFalse(ToyOperatorBackend.is_official)
        aug = self.backend._augmenter(self.backend._namespace(None, None, None, None))
        for op in load_method_config("E01")["generation"]["operator_set"]:
            self.assertEqual(aug.get_augment(op)[0].__name__, op)

    def test_production_rejects_toy(self):
        with self.assertRaisesRegex(ValueError, "scientific"):
            E01Generator(backend=ToyOperatorBackend())

    def test_missing_source_fails_closed(self):
        from methods.fas_aug.official import OfficialFASAugBackend, OfficialBackendError
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(OfficialBackendError, "source unavailable"):
                OfficialFASAugBackend(source_root=d).prepare()

    def test_pin_mismatch_fails_closed(self):
        import copy
        from methods.fas_aug.official import OfficialFASAugBackend, OfficialBackendError
        cfg = copy.deepcopy(load_method_config("E01"))
        cfg["source"]["pinned_commit"] = "0" * 40
        with self.assertRaisesRegex(OfficialBackendError, "mismatch"):
            OfficialFASAugBackend(cfg).prepare()

    def test_missing_and_modified_asset_fail_closed(self):
        from unittest.mock import patch
        from methods.fas_aug.official import OfficialBackendError
        name = "data/background/" + self.backend.asset_index.dirs["background"][0]
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(OfficialBackendError, "asset unavailable"):
                self.backend._verified_bytes(name)
        with patch.object(Path, "read_bytes", return_value=b"altered"):
            with self.assertRaisesRegex(OfficialBackendError, "asset mismatch"):
                self.backend._verified_bytes(name)

    def test_missing_module_fails_closed(self):
        from unittest.mock import patch
        from methods.fas_aug.official import OfficialBackendError
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(OfficialBackendError):
                self.backend._verified_bytes("data/FAS_Augmentations.py")

    def test_upstream_level_zero_never_dispatches(self):
        aug = self.backend._augmenter(self.backend._namespace(None, None, None, None))
        img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        for op in load_method_config("E01")["generation"]["operator_set"]:
            out, changed = aug.apply_augment(img, op, 0, 0)
            np.testing.assert_array_equal(out, img)
            self.assertIsNot(out, img)
            self.assertFalse(changed)

    def test_assets_follow_git_tree_byte_order(self):
        for sub, names in self.backend.asset_index.dirs.items():
            expected = [Path(p).name for p in self.backend._blobs if p.startswith(f"data/{sub}/")]
            self.assertEqual(names, expected)

    def test_official_synthetic_execution_all_operators(self):
        import importlib.util
        missing = [m for m in ("PIL", "cv2") if importlib.util.find_spec(m) is None]
        if missing:
            self.skipTest("Official pixel smoke requires missing execution dependencies: " + ", ".join(missing))
        gen = E01Generator(backend=self.backend).prepare()
        image = np.random.Generator(np.random.PCG64(7)).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        for i, op in enumerate(gen.operator_set):
            pair = next(f"PTR{j:06d}" for j in range(1, 100) if gen.recipe(f"PTR{j:06d}", 42, index=i)["level_k"])
            a = gen.generate_one(pair, 42, image, index=i)
            b = gen.generate_one(pair, 42, image, index=i)
            self.assertTrue(a.success, a.failure_reason)
            self.assertTrue(b.success, b.failure_reason)
            np.testing.assert_array_equal(a.output, b.output)
            self.assertEqual(a.metadata, b.metadata)


if __name__ == "__main__":
    unittest.main()
