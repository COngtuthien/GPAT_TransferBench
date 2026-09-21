"""M6C1 — E02 Frequency Substitution contract. Synthetic arrays only; no benchmark image is read."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.config import load_method_config  # noqa: E402
from methods.freq_sub.e02 import (  # noqa: E402
    E02Generator, block_count, conjugate_index, conjugate_mask, eligible_blocks, pair_seed,
    pair_seed_preimage, select_block_indices, substitute,
)

#: Documented tolerance for the inverse-FFT imaginary residual. The substitution keeps the mixed
#: spectrum Hermitian by construction, so the residual is pure float64 round-off. Observed maxima
#: in M6C1 were ~1.2e-13 (white noise) and ~5.3e-12 (smooth image); 1e-8 is a safe bound that is
#: still ~4 orders of magnitude tighter than any perceptible pixel effect.
IMAG_TOL = 1e-8


class TestE02Rules(unittest.TestCase):
    def test_round_half_up_rule(self):
        self.assertEqual(block_count(156), 39)
        self.assertEqual(block_count(157), 39)
        self.assertEqual(block_count(158), 40)
        self.assertEqual(block_count(159), 40)
        self.assertEqual(block_count(160), 40)

    def test_159_gives_40(self):
        self.assertEqual(block_count(159), 40)

    def test_matches_frozen_worked_verification(self):
        wv = load_method_config("E02")["block_count_rule"]["worked_verification"]
        for n, k in wv.items():
            self.assertEqual(block_count(int(n)), int(k))

    def test_rule_equals_round_half_up(self):
        """(n + 2) // 4 must equal exact round-half-up of n/4 for every n."""
        from decimal import Decimal, ROUND_HALF_UP
        for n in range(0, 400):
            expect = int((Decimal(n) / 4).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            self.assertEqual(block_count(n), expect, n)


class TestE02Geometry(unittest.TestCase):
    def setUp(self):
        self.rc, self.grid = eligible_blocks()

    def test_eligible_count_is_159(self):
        self.assertEqual(self.rc.shape[0], 159)
        self.assertEqual(int(self.grid.sum()), 159)

    def test_matches_frozen_config(self):
        cfg = load_method_config("E02")
        self.assertEqual(self.rc.shape[0], cfg["generation"]["n_eligible_frozen_geometry"])
        self.assertEqual(block_count(self.rc.shape[0]), cfg["block_count_rule"]["k_frozen_geometry"])

    def test_canonical_row_major_order(self):
        rows, cols = self.rc[:, 0], self.rc[:, 1]
        for i in range(1, len(self.rc)):
            self.assertTrue((rows[i] > rows[i - 1]) or
                            (rows[i] == rows[i - 1] and cols[i] > cols[i - 1]),
                            f"not row-major at {i}: {self.rc[i-1]} -> {self.rc[i]}")
        self.assertEqual(tuple(self.rc[0]), (0, 6))

    def test_boundary_convention_does_not_change_the_count(self):
        n, b, g = 256, 16, 16
        f = (np.arange(n) - n // 2) / n
        fy, fx = np.meshgrid(f, f, indexing="ij")
        r = np.sqrt(fy ** 2 + fx ** 2)
        for m in ((r >= .20) & (r <= .50), (r >= .20) & (r < .50), (r > .20) & (r < .50)):
            pb = m.reshape(g, b, g, b).transpose(0, 2, 1, 3).reshape(g, g, b * b)
            self.assertEqual(int((pb.mean(axis=2) >= 0.75).sum()), 159)


class TestE02Seed(unittest.TestCase):
    def test_namespaced_preimage(self):
        self.assertEqual(pair_seed_preimage("PTR000001", 42),
                         b"gpatbench.freqsub.block.v1|PTR000001|42")

    def test_known_seed_vector(self):
        self.assertEqual(
            pair_seed("PTR000001", 42),
            19354000309184887456743517179028239418116648094833629759175866349341962868907)

    def test_matches_frozen_config_worked_example(self):
        we = load_method_config("E02")["pair_seed"]["worked_example"]
        self.assertEqual(pair_seed_preimage(we["pair_id"], 42).decode(), we["seed_42_preimage"])
        self.assertEqual(str(pair_seed(we["pair_id"], 42)), we["seed_42_pair_seed"])

    def test_full_digest_no_truncation_no_modulo(self):
        s = pair_seed("PTR000001", 42)
        self.assertLess(s, 2 ** 256)
        self.assertGreater(s.bit_length(), 200, "a truncated or mod-reduced seed would be small")

    def test_namespace_differs_from_e01(self):
        from methods.fas_aug.e01 import operator_seed_preimage
        self.assertNotEqual(pair_seed_preimage("PTR000001", 42),
                            operator_seed_preimage("PTR000001", 42))


class TestE02Selection(unittest.TestCase):
    def setUp(self):
        self.n, self.k = 159, 40

    def test_size_exactly_40_and_no_duplicates(self):
        sel = select_block_indices(self.n, self.k, pair_seed("PTR000001", 42))
        self.assertEqual(len(sel), 40)
        self.assertEqual(len(set(sel.tolist())), 40)
        self.assertTrue(((sel >= 0) & (sel < self.n)).all())

    def test_sorted_ascending(self):
        sel = select_block_indices(self.n, self.k, pair_seed("PTR000007", 1337))
        self.assertTrue((np.diff(sel) > 0).all())

    def test_same_pair_and_seed_same_set(self):
        a = select_block_indices(self.n, self.k, pair_seed("PTR000007", 42))
        b = select_block_indices(self.n, self.k, pair_seed("PTR000007", 42))
        self.assertTrue(np.array_equal(a, b))

    def test_different_seeds_different_set(self):
        sets = {tuple(select_block_indices(self.n, self.k, pair_seed("PTR000007", g)).tolist())
                for g in (42, 1337, 2026)}
        self.assertEqual(len(sets), 3)

    def test_known_selection_vector(self):
        """Matches the M6A3b audit vector for PTR000001 @ seed 42."""
        sel = select_block_indices(self.n, self.k, pair_seed("PTR000001", 42)).tolist()
        self.assertEqual(sel[:10], [2, 4, 16, 22, 25, 30, 33, 39, 42, 44])


class TestE02Hermitian(unittest.TestCase):
    def test_counterpart_mapping(self):
        self.assertEqual(conjugate_index(1), 255)
        self.assertEqual(conjugate_index(255), 1)
        self.assertEqual(conjugate_index(64), 192)

    def test_self_conjugate_positions(self):
        self.assertEqual(conjugate_index(0), 0)      # Nyquist
        self.assertEqual(conjugate_index(128), 128)  # DC
        cfg = load_method_config("E02")
        self.assertEqual(sorted(cfg["conjugate_symmetry"]["self_conjugate_bins"]), [0, 128])

    def test_involution(self):
        for p in range(256):
            self.assertEqual(conjugate_index(conjugate_index(p)), p)

    def test_conjugate_of_a_block_is_not_a_block(self):
        """A naive block-grid mirror is FORBIDDEN; the pixel-wise image of block (0,0) is split."""
        m = np.zeros((256, 256), bool)
        m[0:16, 0:16] = True
        cm = conjugate_mask(m)
        self.assertEqual(int(cm.sum()), 256)
        self.assertFalse(cm[0:16, 0:16].all())
        rows = sorted(set(np.nonzero(cm)[0].tolist()))
        self.assertIn(0, rows)
        self.assertIn(255, rows)


class TestE02Substitution(unittest.TestCase):
    def setUp(self):
        self.gen = E02Generator(load_method_config("E02")).prepare()
        rng = np.random.default_rng(7)
        self.tgt = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self.src = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def test_imaginary_residual_is_negligible(self):
        r = self.gen.generate_one("PTR000001", 42, self.tgt, self.src)
        self.assertTrue(r.success, r.failure_reason)
        self.assertLess(r.metadata["max_abs_imag"], IMAG_TOL)

    def test_imaginary_residual_on_smooth_image(self):
        yy, xx = np.mgrid[0:256, 0:256]
        smooth = np.stack([((np.sin(yy / 9.0) + np.cos(xx / 7.0) + 2) * 60).astype(np.uint8)] * 3, -1)
        _, diag = substitute(smooth, self.tgt, self.gen.eligible_rc[
            select_block_indices(159, 40, pair_seed("PTR000002", 42))])
        self.assertLess(diag["max_abs_imag"], IMAG_TOL)

    def test_deterministic_byte_identical_output(self):
        a = E02Generator(load_method_config("E02")).prepare().generate_one(
            "PTR000005", 1337, self.tgt, self.src)
        b = E02Generator(load_method_config("E02")).prepare().generate_one(
            "PTR000005", 1337, self.tgt, self.src)
        self.assertTrue(np.array_equal(a.output, b.output))

    def test_output_is_uint8_and_clipped(self):
        r = self.gen.generate_one("PTR000001", 42, self.tgt, self.src)
        self.assertEqual(r.output.dtype, np.uint8)
        self.assertEqual(r.output.shape, self.tgt.shape)

    def test_conjugate_pixels_are_also_edited(self):
        r = self.gen.generate_one("PTR000001", 42, self.tgt, self.src)
        self.assertGreater(r.metadata["conjugate_only_pixels"], 0)
        self.assertEqual(r.metadata["selected_pixels"], 40 * 16 * 16)

    def test_identical_target_and_source_is_a_noop(self):
        out, diag = substitute(self.tgt, self.tgt,
                               self.gen.eligible_rc[select_block_indices(159, 40,
                                                                         pair_seed("PTR000001", 42))])
        self.assertTrue(np.array_equal(out, self.tgt))
        self.assertLess(diag["max_abs_imag"], IMAG_TOL)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            substitute(self.tgt, self.src[:128], self.gen.eligible_rc[:1])


if __name__ == "__main__":
    unittest.main()
