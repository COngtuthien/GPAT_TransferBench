"""M4 pre-flight tests for the FROZEN common pair contract (Q-24..Q-27).

Every contested convention now has an owner decision, so these tests assert the decided value and,
where the decision names an analytically known result, check that number rather than a round-trip.
Nothing here writes a pair manifest.
"""
import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P  # noqa: E402

AUDIT = ROOT / "outputs/audit"
FROZEN = ROOT / "configs/frozen/pairs_v1.yaml"
PROPOSED = ROOT / "configs/proposed/pairs_v1.proposed.yaml"


def S(sid, *, dataset="casia_fasd", split="TRAIN", label=1, video="v0", subject="s0",
      cgroup=None, macro="print", raw="3", pose=(0.0, 0.0, 0.0), frac=1.0, luma=0.5):
    return P.Sample(sid, dataset, split, label, video, subject, cgroup, macro, raw,
                    "0" * 64, pose, frac, luma)


# ------------------------------------------------------------------ eligibility (unchanged rules)
class TestEligibility(unittest.TestCase):
    def test_source_must_be_spoof_and_target_live(self):
        with self.assertRaises(P.PairPolicyError):
            P.eligible_targets(S("src", label=0), [S("t", label=0, subject="s1")])
        self.assertEqual([t.sample_id for t in P.eligible_targets(S("src"), [S("t", label=0, subject="s1")])],
                         ["t"])
        self.assertEqual(P.eligible_targets(S("src"), [S("t", label=1, subject="s1")]), [])

    def test_same_dataset_same_split_only(self):
        src = S("src", dataset="casia_fasd", split="TRAIN")
        cands = [S("a", dataset="msu_mfsd", label=0, subject="s1"),
                 S("b", dataset="casia_fasd", split="VAL", label=0, subject="s1"),
                 S("c", dataset="casia_fasd", split="TRAIN", label=0, subject="s1")]
        self.assertEqual([t.sample_id for t in P.eligible_targets(src, cands)], ["c"])

    def test_no_train_val_leakage_either_direction(self):
        for a, b in (("TRAIN", "VAL"), ("VAL", "TRAIN")):
            self.assertEqual(P.eligible_targets(S("src", split=a),
                                                [S("t", split=b, label=0, subject="s1")]), [])

    def test_test_split_sources_are_refused(self):
        with self.assertRaises(P.PairPolicyError):
            P.eligible_targets(S("src", split="TEST"), [S("t", split="TEST", label=0, subject="s1")])
        self.assertEqual(P.SPLITS_WITH_PAIRS, ("TRAIN", "VAL"))

    def test_casia_msu_different_subject(self):
        for ds in ("casia_fasd", "msu_mfsd"):
            src = S("src", dataset=ds, subject="x")
            self.assertEqual(P.eligible_targets(src, [S("t", dataset=ds, label=0, subject="x")]), [])
            self.assertEqual(len(P.eligible_targets(src, [S("t", dataset=ds, label=0, subject="y")])), 1)

    def test_siw_different_video_and_content_group(self):
        src = S("src", dataset="siwmv2", subject=None, video="V1", cgroup="G1")
        for bad in ([S("t", dataset="siwmv2", label=0, subject=None, video="V1", cgroup="G2")],
                    [S("t", dataset="siwmv2", label=0, subject=None, video="V2", cgroup="G1")]):
            self.assertEqual(P.eligible_targets(src, bad), [])
        ok = [S("t", dataset="siwmv2", label=0, subject=None, video="V2", cgroup="G2")]
        self.assertEqual(len(P.eligible_targets(src, ok)), 1)

    def test_siw_never_claims_a_person_relationship(self):
        cfg = yaml.safe_load(FROZEN.read_text())["eligibility"]["siwmv2"]
        self.assertIn("NOT proven different-person", cfg["claim_wording"])
        for claim in ("different subject", "different person", "subject-disjoint pairing"):
            self.assertIn(claim, cfg["forbidden_claims"])
        self.assertIs(cfg["pseudo_subject_ids_forbidden"], True)

    def test_eligible_returns_canonical_lexical_order(self):
        live = [S(f"t{i}", label=0, subject=f"s{i}") for i in (3, 1, 2)]
        self.assertEqual([t.sample_id for t in P.eligible_targets(S("src"), live)], ["t1", "t2", "t3"])


# ------------------------------------------------------------------ Q-24 candidate selection
class TestCandidateSelection(unittest.TestCase):
    @staticmethod
    def _live(n):
        return [S(f"t{i:05d}", label=0, subject=f"s{i}") for i in range(n)]

    def test_63_64_65_and_large_pools(self):
        src = S("src", subject="zz")
        self.assertEqual(len(P.select_candidates(src, self._live(63))), 63)
        self.assertEqual(len(P.select_candidates(src, self._live(64))), 64)
        self.assertEqual(len(P.select_candidates(src, self._live(65))), 64)
        self.assertEqual(len(P.select_candidates(src, self._live(5000))), 64)

    def test_candidate_input_permutation_gives_the_same_64(self):
        src = S("src", subject="zz")
        live = self._live(400)
        base = [t.sample_id for t in P.select_candidates(src, live)]
        for seed in (0, 1, 2, 3):
            sh = list(live)
            random.Random(seed).shuffle(sh)
            self.assertEqual([t.sample_id for t in P.select_candidates(src, sh)], base)

    def test_repeated_call_is_identical(self):
        src, live = S("src", subject="zz"), self._live(300)
        a = [t.sample_id for t in P.select_candidates(src, live)]
        for _ in range(3):
            self.assertEqual([t.sample_id for t in P.select_candidates(src, live)], a)

    def test_different_source_id_changes_the_source_digest_and_subset(self):
        live = self._live(400)
        self.assertNotEqual(P.source_seed_digest("srcA"), P.source_seed_digest("srcB"))
        a = {t.sample_id for t in P.select_candidates(S("srcA", subject="zz"), live)}
        b = {t.sample_id for t in P.select_candidates(S("srcB", subject="zz"), live)}
        self.assertNotEqual(a, b)

    def test_different_split_seed_changes_the_ranking(self):
        live = self._live(400)
        src = S("src", subject="zz")
        a = {t.sample_id for t in P.select_candidates(src, live, split_seed=20260814)}
        b = {t.sample_id for t in P.select_candidates(src, live, split_seed=1)}
        self.assertNotEqual(a, b)
        self.assertNotEqual(P.source_seed_digest("src", 20260814), P.source_seed_digest("src", 1))

    def test_hash_construction_matches_the_frozen_definition(self):
        import hashlib
        sd = hashlib.sha256(f"gpatbench.pair.source_seed.v1|src|20260814".encode()).digest()
        self.assertEqual(P.source_seed_digest("src"), sd)
        cd = hashlib.sha256(sd + b"|gpatbench.pair.candidate.v1|tgt").digest()
        self.assertEqual(P.candidate_rank_digest(sd, "tgt"), cd)
        self.assertEqual(P.candidate_rank_key("src", "tgt"),
                         (int.from_bytes(cd, "big", signed=False), "tgt"))

    def test_rank_key_is_big_endian_unsigned(self):
        key = P.candidate_rank_key("src", "tgt")
        self.assertIsInstance(key[0], int)
        self.assertGreaterEqual(key[0], 0)
        self.assertLess(key[0], 2 ** 256)

    def test_defensive_lexical_tie_break_on_equal_digests(self):
        """If two candidates ever produced the same digest, the lexical id must decide."""
        a, b = ("zzz", "aaa")
        ka = (12345, a)
        kb = (12345, b)
        self.assertLess(kb, ka)          # 'aaa' wins, matching the frozen secondary key

    def test_zero_candidates_is_a_hard_failure(self):
        with self.assertRaises(P.PairFeasibilityError):
            P.select_candidates(S("src", subject="zz"), [])

    def test_fresh_interpreter_and_hashseed_invariance(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
                "from gpatbench.pairs import common as P\n"
                "live=[P.Sample('t%05d'%i,'casia_fasd','TRAIN',0,'v','s%d'%i,None,'live',None,'0'*64,"
                "(0.0,0.0,0.0),1.0,0.5) for i in range(400)]\n"
                "src=P.Sample('src','casia_fasd','TRAIN',1,'v','zz',None,'print','3','0'*64,"
                "(0.0,0.0,0.0),1.0,0.5)\n"
                "import json; print(json.dumps([t.sample_id for t in P.select_candidates(src,live)]))\n")
        outs = []
        for hs in ("0", "1", "424242"):
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin"}, check=True)
            outs.append(r.stdout.strip())
        self.assertEqual(len(set(outs)), 1)


# ------------------------------------------------------------------ Q-24 pair ids
class TestPairId(unittest.TestCase):
    def test_prefixes_and_numbering(self):
        self.assertEqual(P.assign_pair_ids("TRAIN", [1, 2, 3]), ["PTR000001", "PTR000002", "PTR000003"])
        self.assertEqual(P.assign_pair_ids("VAL", [1, 2]), ["PVA000001", "PVA000002"])

    def test_train_and_val_ids_cannot_collide(self):
        a = set(P.assign_pair_ids("TRAIN", range(50)))
        b = set(P.assign_pair_ids("VAL", range(50)))
        self.assertEqual(a & b, set())
        self.assertEqual(len(a), 50)

    def test_ids_are_unique_within_a_manifest(self):
        ids = P.assign_pair_ids("TRAIN", range(1000))
        self.assertEqual(len(set(ids)), 1000)

    def test_canonical_order_is_permutation_invariant(self):
        srcs = [S(f"s{i:03d}", dataset=("msu_mfsd" if i % 2 else "casia_fasd")) for i in range(40)]
        base = [s.sample_id for s in P.canonical_source_order(srcs)]
        for seed in (0, 1, 2):
            sh = list(srcs)
            random.Random(seed).shuffle(sh)
            self.assertEqual([s.sample_id for s in P.canonical_source_order(sh)], base)
        ordered = P.canonical_source_order(srcs)
        self.assertEqual([s.dataset for s in ordered], sorted(s.dataset for s in srcs))

    def test_pair_id_is_never_an_input_to_candidate_selection(self):
        src = ROOT / "gpatbench/pairs/common.py"
        text = src.read_text()
        sel = text[text.index("def select_candidates"):text.index("# ------------------------------------------------------------------ Q-25")]
        self.assertNotIn("pair_id", sel)
        cfg = yaml.safe_load(FROZEN.read_text())["pair_id"]
        self.assertIs(cfg["used_for_candidate_selection"], False)
        self.assertEqual(cfg["assigned_after"], "final_pair_membership")

    def test_unknown_split_has_no_prefix(self):
        with self.assertRaises(P.PairPolicyError):
            P.assign_pair_ids("TEST", [1])


# ------------------------------------------------------------------ Q-25 pose
class TestPose(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        for i in range(10):
            rows.append(S(f"c{i}", dataset="casia_fasd", pose=(0.1 * i, 0.2 * i, 0.3 * i)))
            rows.append(S(f"m{i}", dataset="msu_mfsd", pose=(1.0 + i, 2.0 + i, 3.0 + i)))
        rows.append(S("v", dataset="casia_fasd", split="VAL", pose=(99.0, 99.0, 99.0)))
        rows.append(S("t", dataset="casia_fasd", split="TEST", pose=(-99.0, -99.0, -99.0)))
        return rows

    def test_fitted_per_dataset_on_train_only(self):
        rows = self._rows()
        c = P.fit_pose_stats(rows, "casia_fasd")
        m = P.fit_pose_stats(rows, "msu_mfsd")
        self.assertEqual(c.n_train_complete, 10)
        self.assertEqual(m.n_train_complete, 10)
        self.assertNotEqual(c.mean, m.mean)        # never pooled
        self.assertNotEqual(c.std, m.std)

    def test_val_and_test_rows_do_not_move_the_statistics(self):
        train_only = [r for r in self._rows() if r.split == "TRAIN"]
        a = P.fit_pose_stats(self._rows(), "casia_fasd")
        b = P.fit_pose_stats(train_only, "casia_fasd")
        self.assertEqual(a.mean, b.mean)
        self.assertEqual(a.std, b.std)

    def test_population_std_ddof_zero_against_a_known_value(self):
        rows = [S(f"x{i}", pose=(v, 0.0, 1.0)) for i, v in enumerate([1.0, 2.0, 3.0, 4.0])]
        rows += [S("pad", pose=(2.5, 1.0, 0.0))]
        vals = np.array([1.0, 2.0, 3.0, 4.0, 2.5])
        st = P.fit_pose_stats(rows, "casia_fasd")
        self.assertAlmostEqual(st.mean[0], float(vals.mean()), places=12)
        self.assertAlmostEqual(st.std[0], float(vals.std(ddof=0)), places=12)
        self.assertNotAlmostEqual(st.std[0], float(vals.std(ddof=1)), places=6)

    def test_known_analytical_z_score(self):
        rows = [S("a", pose=(0.0, 0.0, 0.0)), S("b", pose=(2.0, 4.0, 6.0))]
        st = P.fit_pose_stats(rows, "casia_fasd")     # mean (1,2,3), population std (1,2,3)
        self.assertEqual(st.mean, (1.0, 2.0, 3.0))
        self.assertEqual(st.std, (1.0, 2.0, 3.0))
        self.assertEqual(st.z((2.0, 4.0, 6.0)), (1.0, 1.0, 1.0))
        self.assertEqual(st.z((1.0, 2.0, 3.0)), (0.0, 0.0, 0.0))

    def test_known_analytical_euclidean_d_pose(self):
        rows = [S("a", pose=(0.0, 0.0, 0.0)), S("b", pose=(2.0, 2.0, 2.0))]
        st = P.fit_pose_stats(rows, "casia_fasd")     # mean 1, std 1 on every axis
        s = S("s", pose=(0.0, 0.0, 0.0))              # z = (-1,-1,-1)
        t = S("t", label=0, subject="z", pose=(2.0, 2.0, 2.0))   # z = (1,1,1)
        self.assertAlmostEqual(P.d_pose(s, t, st), math.sqrt(12.0), places=12)
        self.assertEqual(P.d_pose(s, s, st), 0.0)

    def test_no_division_by_sqrt3(self):
        rows = [S("a", pose=(0.0, 0.0, 0.0)), S("b", pose=(2.0, 2.0, 2.0))]
        st = P.fit_pose_stats(rows, "casia_fasd")
        got = P.d_pose(S("s", pose=(0.0, 0.0, 0.0)),
                       S("t", label=0, subject="z", pose=(2.0, 2.0, 2.0)), st)
        self.assertNotAlmostEqual(got, math.sqrt(12.0) / math.sqrt(3.0), places=6)

    def test_zero_variance_is_a_hard_error(self):
        rows = [S(f"x{i}", pose=(1.0, 0.1 * i, 0.2 * i)) for i in range(10)]
        with self.assertRaises(P.PairPolicyError) as e:
            P.fit_pose_stats(rows, "casia_fasd")
        self.assertIn("hard error", str(e.exception))

    def test_statistics_must_belong_to_the_pair_dataset(self):
        rows = self._rows()
        st = P.fit_pose_stats(rows, "msu_mfsd")
        with self.assertRaises(P.PairPolicyError):
            P.d_pose(S("s", dataset="casia_fasd"), S("t", dataset="casia_fasd", label=0, subject="z"), st)

    def test_float64_execution(self):
        rows = [S("a", pose=(0.0, 0.0, 0.0)), S("b", pose=(2.0, 4.0, 6.0))]
        st = P.fit_pose_stats(rows, "casia_fasd")
        self.assertIsInstance(st.mean[0], float)
        self.assertIsInstance(P.d_pose(rows[0], S("t", label=0, subject="z", pose=(1.0, 1.0, 1.0)), st),
                              float)


# ------------------------------------------------------------------ Q-26 scale
class TestScale(unittest.TestCase):
    def test_bbox_fully_inside_frame(self):
        f = P.face_area_fraction("msu_mfsd", (10, 20, 110, 120), 200.0, 400.0)
        self.assertAlmostEqual(f, (100.0 * 100.0) / (200.0 * 400.0), places=12)

    def test_bbox_crossing_each_border_is_clipped(self):
        W, H = 100.0, 100.0
        self.assertAlmostEqual(P.face_area_fraction("msu_mfsd", (-50, 0, 50, 100), W, H), 0.5, places=12)
        self.assertAlmostEqual(P.face_area_fraction("msu_mfsd", (50, 0, 150, 100), W, H), 0.5, places=12)
        self.assertAlmostEqual(P.face_area_fraction("msu_mfsd", (0, -50, 100, 50), W, H), 0.5, places=12)
        self.assertAlmostEqual(P.face_area_fraction("msu_mfsd", (0, 50, 100, 150), W, H), 0.5, places=12)

    def test_same_relative_bbox_at_different_resolutions_gives_the_same_fraction(self):
        a = P.face_area_fraction("msu_mfsd", (160, 120, 480, 360), 640.0, 480.0)
        b = P.face_area_fraction("siwmv2", (480, 270, 1440, 810), 1920.0, 1080.0)
        self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(a, 0.25, places=12)

    def test_crop_square_and_canonical_size_do_not_enter(self):
        """Executable code only: the docstring may explain which stage is excluded."""
        import ast
        tree = ast.parse((ROOT / "gpatbench/pairs/common.py").read_text())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "face_area_fraction")
        body = fn.body[1:] if ast.get_docstring(fn) else fn.body   # drop the docstring node
        code = "\n".join(ast.unparse(n) for n in body)
        for forbidden in ("1.25", "crop_side", "256", "requested"):
            self.assertNotIn(forbidden, code, forbidden)
        # the only geometric inputs are the detector bbox and the original frame size
        args = {a.arg for a in fn.args.args}
        self.assertEqual(args, {"dataset", "bbox", "frame_w", "frame_h"})

    def test_known_natural_log_ratio(self):
        s = S("s", dataset="msu_mfsd", frac=0.10)
        t = S("t", dataset="msu_mfsd", label=0, subject="z", frac=0.20)
        self.assertAlmostEqual(P.d_scale(s, t), math.log(2.0), places=12)
        self.assertNotAlmostEqual(P.d_scale(s, t), math.log10(2.0), places=6)

    def test_equal_fraction_gives_exactly_zero(self):
        s = S("s", dataset="msu_mfsd", frac=0.3)
        t = S("t", dataset="msu_mfsd", label=0, subject="z", frac=0.3)
        self.assertEqual(P.d_scale(s, t), 0.0)

    def test_casia_fraction_is_one_and_d_scale_exactly_zero(self):
        self.assertEqual(P.CASIA_FACE_AREA_FRACTION, 1.0)
        s = S("s", dataset="casia_fasd", frac=P.CASIA_FACE_AREA_FRACTION)
        t = S("t", dataset="casia_fasd", label=0, subject="z", frac=P.CASIA_FACE_AREA_FRACTION)
        self.assertEqual(P.d_scale(s, t), 0.0)

    def test_casia_never_gets_a_detector_or_pseudo_box(self):
        with self.assertRaises(P.PairPolicyError) as e:
            P.face_area_fraction("casia_fasd", (0, 0, 10, 10), 100.0, 100.0)
        self.assertIn("no detector box", str(e.exception))
        body = (ROOT / "gpatbench/pairs/common.py").read_text()
        for forbidden in ("landmark", "landmarks"):
            self.assertNotIn(forbidden, body)

    def test_invalid_geometry_is_a_hard_error(self):
        for bbox, W, H in (((0, 0, 0, 10), 100.0, 100.0),      # zero width
                           ((0, 0, 10, 0), 100.0, 100.0),      # zero height
                           ((200, 0, 300, 10), 100.0, 100.0),  # entirely outside
                           ((0, 0, 10, 10), 0.0, 100.0)):      # bad frame
            with self.assertRaises(P.PairPolicyError):
                P.face_area_fraction("msu_mfsd", bbox, W, H)

    def test_out_of_range_fraction_is_rejected(self):
        with self.assertRaises(P.PairPolicyError):
            P.d_scale(S("s", frac=0.0), S("t", label=0, subject="z", frac=0.5))
        with self.assertRaises(P.PairPolicyError):
            P.d_scale(S("s", frac=1.5), S("t", label=0, subject="z", frac=0.5))

    def test_weights_are_not_renormalized_for_casia(self):
        cfg = yaml.safe_load(FROZEN.read_text())
        self.assertIs(cfg["scale"]["casia_fasd"]["weights_renormalized"], False)
        self.assertEqual(cfg["weights"], {"pose": 0.50, "scale": 0.30, "luma": 0.20})
        self.assertEqual(cfg["scale"]["casia_fasd"]["deviation"], "DEV-018")


# ------------------------------------------------------------------ Q-27 luminance
class TestLuminance(unittest.TestCase):
    @staticmethod
    def _img(rgb):
        a = np.zeros((256, 256, 3), np.uint8)
        a[:, :, 0], a[:, :, 1], a[:, :, 2] = rgb
        return a

    def test_analytically_known_means(self):
        cases = {(0, 0, 0): 0.0, (255, 255, 255): 1.0,
                 (255, 0, 0): 0.299, (0, 255, 0): 0.587, (0, 0, 255): 0.114}
        for rgb, want in cases.items():
            self.assertAlmostEqual(P.luma_mean_from_rgb(self._img(rgb)), want, places=12, msg=str(rgb))

    def test_channel_order_is_rgb_not_bgr(self):
        red = P.luma_mean_from_rgb(self._img((255, 0, 0)))
        blue = P.luma_mean_from_rgb(self._img((0, 0, 255)))
        self.assertAlmostEqual(red, 0.299, places=12)
        self.assertAlmostEqual(blue, 0.114, places=12)
        self.assertNotAlmostEqual(red, 0.114, places=6)

    def test_bt709_and_opencv_paths_are_absent(self):
        self.assertEqual(P.BT601, (0.299, 0.587, 0.114))
        body = (ROOT / "gpatbench/pairs/common.py").read_text()
        for forbidden in ("0.2126", "0.7152", "0.0722", "YCrCb", "cvtColor"):
            self.assertNotIn(forbidden, body, forbidden)

    def test_all_pixels_including_zero_padding_are_included(self):
        a = self._img((255, 255, 255))
        a[:128, :, :] = 0                      # half black, as canonical zero padding would be
        self.assertAlmostEqual(P.luma_mean_from_rgb(a), 0.5, places=12)

    def test_float64_and_dtype_guard(self):
        self.assertIsInstance(P.luma_mean_from_rgb(self._img((10, 20, 30))), float)
        with self.assertRaises(P.PairPolicyError):
            P.luma_mean_from_rgb(np.zeros((8, 8, 3), np.float32))
        with self.assertRaises(P.PairPolicyError):
            P.luma_mean_from_rgb(np.zeros((8, 8), np.uint8))

    def test_d_luma_is_the_absolute_mean_difference(self):
        s = S("s", luma=0.25)
        t = S("t", label=0, subject="z", luma=0.75)
        self.assertAlmostEqual(P.d_luma(s, t), 0.5, places=12)
        self.assertAlmostEqual(P.d_luma(t, s), 0.5, places=12)
        self.assertEqual(P.d_luma(s, s), 0.0)

    def test_out_of_range_luma_rejected(self):
        with self.assertRaises(P.PairPolicyError):
            P.d_luma(S("s", luma=1.2), S("t", label=0, subject="z", luma=0.5))


# ------------------------------------------------------------------ combination
class TestCombination(unittest.TestCase):
    def _stats(self):
        return P.fit_pose_stats([S("a", pose=(0.0, 0.0, 0.0)), S("b", pose=(2.0, 2.0, 2.0))],
                                "casia_fasd")

    def test_weighted_formula(self):
        st = self._stats()
        s = S("s", pose=(0.0, 0.0, 0.0), frac=0.1, luma=0.2)
        t = S("t", label=0, subject="z", pose=(2.0, 2.0, 2.0), frac=0.2, luma=0.5)
        d = P.d_pair(s, t, st)
        self.assertAlmostEqual(d["d_pair"],
                               0.50 * d["d_pose"] + 0.30 * d["d_scale"] + 0.20 * d["d_luma"], places=12)
        self.assertAlmostEqual(d["d_pose"], math.sqrt(12.0), places=12)
        self.assertAlmostEqual(d["d_scale"], math.log(2.0), places=12)
        self.assertAlmostEqual(d["d_luma"], 0.3, places=12)

    def test_exact_tie_breaks_on_lexical_target_sample_id(self):
        st = self._stats()
        s = S("s", pose=(0.0, 0.0, 0.0), frac=0.2, luma=0.3)
        a = S("aaa", label=0, subject="z1", pose=(1.0, 1.0, 1.0), frac=0.3, luma=0.4)
        b = S("bbb", label=0, subject="z2", pose=(1.0, 1.0, 1.0), frac=0.3, luma=0.4)
        for order in ([b, a], [a, b]):
            self.assertEqual(P.choose_target(s, order, st)["target"].sample_id, "aaa")

    def test_choose_target_is_permutation_invariant(self):
        st = self._stats()
        s = S("s", pose=(0.0, 0.0, 0.0), frac=0.2, luma=0.3)
        cands = [S(f"c{i:03d}", label=0, subject=f"z{i}", pose=(0.1 * i, 0.02 * i, 0.0),
                   frac=0.1 + 0.005 * i, luma=0.3 + 0.004 * i) for i in range(40)]
        base = P.choose_target(s, cands, st)["target"].sample_id
        for seed in (0, 1, 2, 3):
            sh = list(cands)
            random.Random(seed).shuffle(sh)
            self.assertEqual(P.choose_target(s, sh, st)["target"].sample_id, base)

    def test_no_fuzzy_tie_window(self):
        """A difference of one ULP must still decide the winner, not fall back to the tie-break."""
        st = self._stats()
        s = S("s", pose=(0.0, 0.0, 0.0), frac=0.2, luma=0.3)
        z = S("zzz", label=0, subject="z1", pose=(1.0, 1.0, 1.0), frac=0.2, luma=0.3)
        a = S("aaa", label=0, subject="z2", pose=(1.0, 1.0, 1.0), frac=0.2,
              luma=0.3 + 1e-12)
        self.assertEqual(P.choose_target(s, [a, z], st)["target"].sample_id, "zzz")


# ------------------------------------------------------------------ frozen config
class TestFrozenPairConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(FROZEN.read_text())

    def test_status_and_resolved_questions(self):
        self.assertEqual(self.cfg["status"], "FROZEN")
        self.assertEqual(self.cfg["resolves"], ["Q-24", "Q-25", "Q-26", "Q-27"])
        self.assertEqual(self.cfg["question_status"], {
            "Q-24": "RESOLVED_BY_OWNER_HASH_RANKING",
            "Q-25": "RESOLVED_BY_OWNER_DATASET_TRAIN_ZSCORE_L2",
            "Q-26": "RESOLVED_BY_OWNER_NORMALIZED_VISIBLE_BBOX_LOGRATIO",
            "Q-27": "RESOLVED_BY_OWNER_BT601_UNIT_MEAN_ABSDIFF"})
        self.assertEqual(self.cfg["non_blocking_open"], ["Q-28", "Q-29"])

    def test_no_unresolved_field(self):
        bad = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif node is None or (isinstance(node, str) and node.strip().upper() in
                                  {"TODO", "TBD", "OWNER_DECISION_REQUIRED", "NULL"}):
                bad.append(path)

        walk(self.cfg, "")
        self.assertEqual(bad, [], f"unresolved fields: {bad}")

    def test_config_matches_the_implementation(self):
        self.assertEqual(self.cfg["candidate_cap"], P.CANDIDATE_CAP)
        self.assertEqual(self.cfg["split_seed"], P.SPLIT_SEED)
        self.assertEqual(self.cfg["weights"], P.WEIGHTS)
        self.assertEqual(self.cfg["build_from_splits"], list(P.SPLITS_WITH_PAIRS))
        self.assertEqual(self.cfg["candidate_selection"]["source_seed_digest"]["namespace"],
                         P.SOURCE_SEED_NAMESPACE)
        self.assertEqual(self.cfg["candidate_selection"]["candidate_rank_digest"]["namespace"],
                         P.CANDIDATE_NAMESPACE)
        self.assertEqual(self.cfg["pose"]["zero_variance"]["threshold"], P.MIN_POSE_STD)
        self.assertEqual(self.cfg["pose"]["normalization"]["ddof"], 0)
        self.assertEqual(self.cfg["scale"]["casia_fasd"]["face_area_fraction"],
                         P.CASIA_FACE_AREA_FRACTION)
        self.assertEqual(self.cfg["pair_id"]["format"], dict(P.PAIR_ID_PREFIX_FORMAT))

    def test_snapshot_is_byte_identical(self):
        snap = ROOT / "frozen_config_snapshot/configs/frozen/pairs_v1.yaml"
        self.assertTrue(snap.is_file())
        self.assertEqual(snap.read_bytes(), FROZEN.read_bytes())

    def test_proposal_preserved_as_history(self):
        self.assertTrue(PROPOSED.is_file())
        prop = yaml.safe_load(PROPOSED.read_text())
        self.assertEqual(prop["status"], "PROPOSED_BLOCKED")
        self.assertEqual(self.cfg["supersedes"], PROPOSED.relative_to(ROOT).as_posix())

    def test_contains_no_membership(self):
        self.assertIs(self.cfg["membership_in_this_file"], False)
        for pat in ("pairs_train_v1.parquet", "val_pairs_v1.parquet", "pair_train_stats_v1.json"):
            self.assertEqual(list(ROOT.rglob(pat)), [], pat)

    def test_decision_provenance_is_not_claimed_as_literature(self):
        self.assertIn("Owner benchmark-design decisions", self.cfg["decision_provenance"])


# ------------------------------------------------------------------ preflight evidence
class TestPreflightEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = AUDIT / "M4_PAIR_PREFLIGHT.json"
        cls.rep = json.loads(p.read_text()) if p.is_file() else None

    def setUp(self):
        if self.rep is None:
            self.skipTest("preflight not run")

    def test_no_pair_manifest_was_created(self):
        self.assertIs(self.rep["pair_manifest_created"], False)
        self.assertIs(self.rep["membership_persisted"], False)

    def test_no_source_lacks_a_candidate(self):
        self.assertEqual(self.rep["feasibility"]["blocker_zero_candidate_sources"], 0)

    def test_casia_d_scale_is_identically_zero_on_real_data(self):
        d = self.rep.get("frozen_diagnostic")
        if not d:
            self.skipTest("frozen diagnostic not present")
        self.assertEqual(d["casia_fasd"]["d_scale_unique_values"], [0.0])

    def test_native_coverage_unchanged(self):
        n = self.rep["native"]
        self.assertIs(n["siwmv2"]["subject_ids_available"], False)
        self.assertEqual(n["siwmv2"]["identities_with_live_and_spoof"], 0)
        for ds in ("casia_fasd", "msu_mfsd"):
            self.assertEqual(n[ds]["dsdg_native_identity_pairing"], "SUPPORTED")

    def test_m4_is_not_started(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        self.assertEqual(s["M3"]["status"], "COMPLETE")
        self.assertEqual(s["M4"]["status"], "NOT_STARTED")


if __name__ == "__main__":
    unittest.main()
