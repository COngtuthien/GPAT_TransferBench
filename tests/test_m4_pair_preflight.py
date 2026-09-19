"""M4 pre-flight tests: the settled parts of the common pairing rule, and guards on the open parts.

Nothing here writes a pair manifest. Where the spec leaves a detail open (Q-24..Q-27), the test
asserts that the code *refuses to proceed without an explicit choice* rather than asserting a value.
"""
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P  # noqa: E402

AUDIT = ROOT / "outputs/audit"
PROPOSED = ROOT / "configs/proposed/pairs_v1.proposed.yaml"

POLICY = dict(candidate_selection="per_candidate_hash_rank", pose_stat_scope="train_pooled",
              pose_std="population", pose_norm="l2", pose_zero_variance="error",
              scale_box="scrfd_bbox", scale_formula="abs_log_ratio", luma_standard="bt601",
              luma_range="unit_interval", luma_formula="abs_difference")


def S(sid, *, dataset="casia_fasd", split="TRAIN", label=1, video="v0", subject="s0",
      cgroup=None, macro="print", raw="3", pose=(0.0, 0.0, 0.0), area=100.0, frame=10000.0,
      luma=0.5):
    return P.Sample(sid, dataset, split, label, video, subject, cgroup, macro, raw,
                    "0" * 64, pose, area, frame, luma)


def policy(**kw):
    return P.PairMetricPolicy(**{**POLICY, **kw})


class TestPolicyIsExplicit(unittest.TestCase):
    def test_every_contested_field_is_required(self):
        """A caller cannot build a policy without stating each unresolved convention."""
        with self.assertRaises(TypeError):
            P.PairMetricPolicy()
        for missing in ("pose_norm", "luma_standard", "scale_box", "candidate_selection"):
            kw = {k: v for k, v in POLICY.items() if k != missing}
            with self.assertRaises(TypeError, msg=missing):
                P.PairMetricPolicy(**kw)

    def test_policy_is_not_frozen(self):
        self.assertFalse(policy().frozen)

    def test_unsupported_values_are_rejected(self):
        for field, bad in (("pose_norm", "cosine"), ("luma_standard", "bt2020"),
                           ("scale_box", "made_up"), ("pose_std", "robust")):
            with self.assertRaises(P.PairPolicyError, msg=field):
                policy(**{field: bad}).validate()

    def test_weights_and_cap_match_the_spec(self):
        self.assertEqual(P.WEIGHTS, {"pose": 0.50, "scale": 0.30, "luma": 0.20})
        self.assertEqual(P.CANDIDATE_CAP, 64)
        self.assertEqual(P.SPLIT_SEED, 20260814)


class TestEligibility(unittest.TestCase):
    def test_source_must_be_spoof_and_target_live(self):
        live = [S("t1", label=0, subject="s1")]
        with self.assertRaises(P.PairPolicyError):
            P.eligible_targets(S("src", label=0), live)
        self.assertEqual([t.sample_id for t in P.eligible_targets(S("src", subject="s0"), live)], ["t1"])
        spoof_target = [S("t2", label=1, subject="s1")]
        self.assertEqual(P.eligible_targets(S("src", subject="s0"), spoof_target), [])

    def test_same_dataset_and_same_split_only(self):
        src = S("src", dataset="casia_fasd", split="TRAIN", subject="s0")
        cands = [S("a", dataset="msu_mfsd", label=0, subject="s1"),
                 S("b", dataset="casia_fasd", split="VAL", label=0, subject="s1"),
                 S("c", dataset="casia_fasd", split="TRAIN", label=0, subject="s1")]
        self.assertEqual([t.sample_id for t in P.eligible_targets(src, cands)], ["c"])

    def test_no_train_val_leakage_in_either_direction(self):
        for a, b in (("TRAIN", "VAL"), ("VAL", "TRAIN"), ("TRAIN", "TEST"), ("VAL", "TEST")):
            src = S("src", split=a, subject="s0")
            self.assertEqual(P.eligible_targets(src, [S("t", split=b, label=0, subject="s1")]), [])

    def test_casia_and_msu_require_a_different_subject(self):
        for ds in ("casia_fasd", "msu_mfsd"):
            src = S("src", dataset=ds, subject="x")
            same = [S("t", dataset=ds, label=0, subject="x")]
            diff = [S("t", dataset=ds, label=0, subject="y")]
            self.assertEqual(P.eligible_targets(src, same), [])
            self.assertEqual(len(P.eligible_targets(src, diff)), 1)

    def test_casia_msu_missing_subject_id_is_an_error(self):
        src = S("src", dataset="msu_mfsd", subject=None)
        with self.assertRaises(P.PairPolicyError):
            P.eligible_targets(src, [S("t", dataset="msu_mfsd", label=0, subject="y")])

    def test_siw_requires_different_video_and_different_content_group(self):
        src = S("src", dataset="siwmv2", subject=None, video="V1", cgroup="G1")
        same_video = [S("t", dataset="siwmv2", label=0, subject=None, video="V1", cgroup="G2")]
        same_group = [S("t", dataset="siwmv2", label=0, subject=None, video="V2", cgroup="G1")]
        ok = [S("t", dataset="siwmv2", label=0, subject=None, video="V2", cgroup="G2")]
        self.assertEqual(P.eligible_targets(src, same_video), [])
        self.assertEqual(P.eligible_targets(src, same_group), [])
        self.assertEqual(len(P.eligible_targets(src, ok)), 1)

    def test_siw_pairing_never_asserts_a_same_or_different_person_claim(self):
        src = ROOT / "gpatbench/pairs/common.py"
        text = src.read_text()
        self.assertIn("never be described as", text)
        cfg = yaml.safe_load(PROPOSED.read_text())
        siw = cfg["eligibility"]["siwmv2"]
        self.assertIn("NOT proven different-person", siw["claim_wording"])
        for claim in ("different subject", "different person", "subject-disjoint pairing"):
            self.assertIn(claim, siw["forbidden_claims"])
        self.assertIs(siw["pseudo_subject_ids_forbidden"], True)


class TestCandidateSelection(unittest.TestCase):
    @staticmethod
    def _live(n, ds="casia_fasd"):
        return [S(f"t{i:04d}", dataset=ds, label=0, subject=f"s{i}") for i in range(n)]

    def test_at_or_below_cap_uses_all(self):
        src = S("src", subject="zz")
        for n in (1, 10, 63, 64):
            got = P.select_candidates(src, self._live(n), policy())
            self.assertEqual(len(got), n)

    def test_above_cap_selects_exactly_64(self):
        src = S("src", subject="zz")
        got = P.select_candidates(src, self._live(500), policy())
        self.assertEqual(len(got), P.CANDIDATE_CAP)
        self.assertEqual(len({t.sample_id for t in got}), P.CANDIDATE_CAP)

    def test_candidate_permutation_invariance(self):
        src = S("src", subject="zz")
        live = self._live(300)
        base = [t.sample_id for t in P.select_candidates(src, live, policy())]
        for seed in (0, 1, 2):
            shuffled = list(live)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual([t.sample_id for t in P.select_candidates(src, shuffled, policy())], base)

    def test_different_sources_get_different_subsets(self):
        live = self._live(300)
        a = {t.sample_id for t in P.select_candidates(S("srcA", subject="zz"), live, policy())}
        b = {t.sample_id for t in P.select_candidates(S("srcB", subject="zz"), live, policy())}
        self.assertNotEqual(a, b)

    def test_seed_changes_the_subset(self):
        live = self._live(300)
        src = S("src", subject="zz")
        a = {t.sample_id for t in P.select_candidates(src, live, policy(), seed=20260814)}
        b = {t.sample_id for t in P.select_candidates(src, live, policy(), seed=1)}
        self.assertNotEqual(a, b)

    def test_zero_candidates_is_a_hard_failure(self):
        with self.assertRaises(P.PairFeasibilityError):
            P.select_candidates(S("src", subject="zz"), [], policy())

    def test_candidate_rank_is_a_stable_namespaced_hash(self):
        import hashlib
        want = hashlib.sha256(
            f"{P.CANDIDATE_NAMESPACE}|src|tgt|20260814".encode()).hexdigest()
        self.assertEqual(P.candidate_rank("src", "tgt"), want)
        self.assertEqual(want, want.lower())

    def test_selection_survives_a_fresh_interpreter_with_other_hash_seeds(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
                "from gpatbench.pairs import common as P\n"
                f"pol = P.PairMetricPolicy(**{POLICY!r})\n"
                "live=[P.Sample('t%04d'%i,'casia_fasd','TRAIN',0,'v','s%d'%i,None,'live',None,'0'*64)"
                " for i in range(300)]\n"
                "src=P.Sample('src','casia_fasd','TRAIN',1,'v','zz',None,'print','3','0'*64)\n"
                "import json; print(json.dumps([t.sample_id for t in P.select_candidates(src,live,pol)]))\n")
        outs = []
        for hs in ("0", "1", "98765"):
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin"}, check=True)
            outs.append(r.stdout.strip())
        self.assertEqual(len(set(outs)), 1)


class TestDistances(unittest.TestCase):
    @staticmethod
    def _train(n=50):
        return [S(f"x{i}", pose=(0.1 * i, 0.05 * i, -0.02 * i)) for i in range(n)]

    def test_pose_stats_are_fitted_on_train_rows_only(self):
        rows = self._train() + [S("v1", split="VAL", pose=(9.0, 9.0, 9.0)),
                                S("t1", split="TEST", pose=(-9.0, -9.0, -9.0))]
        n = P.PoseNormalizer.fit(rows, policy())
        self.assertEqual(n.n_rows, 50)
        ref = P.PoseNormalizer.fit(self._train(), policy())
        self.assertEqual(n.mean, ref.mean)
        self.assertEqual(n.std, ref.std)

    def test_pose_population_vs_sample_std_differ_as_declared(self):
        rows = self._train()
        a = P.PoseNormalizer.fit(rows, policy(pose_std="population"))
        b = P.PoseNormalizer.fit(rows, policy(pose_std="sample"))
        self.assertNotEqual(a.std, b.std)
        self.assertEqual(a.std_convention, "population")
        self.assertEqual(b.std_convention, "sample")

    def test_zero_variance_pose_axis_is_refused_by_default(self):
        rows = [S(f"x{i}", pose=(0.0, 0.1 * i, 0.2 * i)) for i in range(20)]
        with self.assertRaises(P.PairPolicyError):
            P.PoseNormalizer.fit(rows, policy(pose_zero_variance="error"))
        n = P.PoseNormalizer.fit(rows, policy(pose_zero_variance="treat_as_one"))
        self.assertEqual(n.zero_variance_axes, (0,))
        self.assertEqual(n.std[0], 1.0)

    def test_pose_distance_norms_differ_and_are_non_negative(self):
        rows = self._train()
        n = P.PoseNormalizer.fit(rows, policy())
        a, b = rows[0], rows[10]
        l2 = P.d_pose(a, b, n, policy(pose_norm="l2"))
        l1 = P.d_pose(a, b, n, policy(pose_norm="l1"))
        self.assertGreater(l1, l2)
        self.assertGreaterEqual(l2, 0.0)
        self.assertEqual(P.d_pose(a, a, n, policy()), 0.0)

    def test_scale_requires_a_face_box_and_refuses_casia(self):
        src = S("s", area=None)      # CASIA shape: no detector box at all
        tgt = S("t", area=200.0)
        with self.assertRaises(P.PairPolicyError) as e:
            P.d_scale(src, tgt, policy())
        self.assertIn("no face-box area", str(e.exception))

    def test_scale_formula_and_guards(self):
        a, b = S("a", area=100.0), S("b", area=200.0)
        import math
        self.assertAlmostEqual(P.d_scale(a, b, policy()), abs(math.log(2.0)))
        self.assertEqual(P.d_scale(a, a, policy()), 0.0)
        with self.assertRaises(P.PairPolicyError):
            P.d_scale(S("z", area=0.0), b, policy())

    def test_scale_frame_fraction_variant_differs(self):
        """Raw pixel area and area-as-fraction-of-frame are genuinely different metrics.

        Here the box doubles while the frame also doubles, so the fraction is unchanged (distance 0)
        although the raw pixel areas differ by a factor of two. This is exactly why the choice
        matters across datasets whose frames have different resolutions.
        """
        a = S("a", area=100.0, frame=10000.0)
        b = S("b", area=200.0, frame=20000.0)
        raw = P.d_scale(a, b, policy(scale_box="scrfd_bbox"))
        frac = P.d_scale(a, b, policy(scale_box="bbox_frame_fraction"))
        import math
        self.assertAlmostEqual(raw, abs(math.log(2.0)))
        self.assertAlmostEqual(frac, 0.0)
        self.assertNotAlmostEqual(raw, frac)

    def test_luma_distance_and_range_guard(self):
        self.assertAlmostEqual(P.d_luma(S("a", luma=0.2), S("b", luma=0.5), policy()), 0.3)
        self.assertEqual(P.d_luma(S("a", luma=0.4), S("b", luma=0.4), policy()), 0.0)
        with self.assertRaises(P.PairPolicyError):
            P.d_luma(S("a", luma=1.5), S("b", luma=0.4), policy())

    def test_d_pair_applies_the_frozen_weights(self):
        rows = self._train()
        n = P.PoseNormalizer.fit(rows, policy())
        a = S("a", pose=(0.0, 0.0, 0.0), area=100.0, luma=0.2)
        b = S("b", pose=(1.0, 0.0, 0.0), area=200.0, luma=0.5)
        d = P.d_pair(a, b, n, policy())
        self.assertAlmostEqual(d["d_pair"],
                               0.50 * d["d_pose"] + 0.30 * d["d_scale"] + 0.20 * d["d_luma"])
        for k in ("d_pose", "d_scale", "d_luma", "d_pair"):
            self.assertGreaterEqual(d[k], 0.0)

    def test_exact_tie_breaks_on_lexical_target_sample_id(self):
        rows = self._train()
        n = P.PoseNormalizer.fit(rows, policy())
        src = S("src", pose=(0.0, 0.0, 0.0), area=100.0, luma=0.3)
        # two identical candidates apart from their ids
        t_b = S("bbb", label=0, subject="z1", pose=(0.5, 0.0, 0.0), area=150.0, luma=0.4)
        t_a = S("aaa", label=0, subject="z2", pose=(0.5, 0.0, 0.0), area=150.0, luma=0.4)
        for order in ([t_b, t_a], [t_a, t_b]):
            self.assertEqual(P.choose_target(src, order, n, policy())["target"].sample_id, "aaa")

    def test_choose_target_is_permutation_invariant(self):
        rows = self._train()
        n = P.PoseNormalizer.fit(rows, policy())
        src = S("src", pose=(0.0, 0.0, 0.0), area=100.0, luma=0.3)
        cands = [S(f"c{i}", label=0, subject=f"z{i}", pose=(0.1 * i, 0.02 * i, 0.0),
                   area=100.0 + 7 * i, luma=0.3 + 0.01 * i) for i in range(30)]
        base = P.choose_target(src, cands, n, policy())["target"].sample_id
        for seed in (0, 1, 2, 3):
            sh = list(cands)
            random.Random(seed).shuffle(sh)
            self.assertEqual(P.choose_target(src, sh, n, policy())["target"].sample_id, base)


class TestPairId(unittest.TestCase):
    def test_format_and_determinism(self):
        self.assertEqual(P.pair_id(1), "P000001")
        self.assertEqual(P.pair_id(123456), "P123456")

    def test_canonical_source_order_is_order_independent(self):
        srcs = [S(f"s{i}", dataset=("msu_mfsd" if i % 2 else "casia_fasd")) for i in range(20)]
        base = [s.sample_id for s in P.canonical_source_order(srcs)]
        for seed in (0, 1, 2):
            sh = list(srcs)
            random.Random(seed).shuffle(sh)
            self.assertEqual([s.sample_id for s in P.canonical_source_order(sh)], base)


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
        for pat in ("pairs_train_v1.parquet", "val_pairs_v1.parquet", "pair_train_stats_v1.json"):
            self.assertEqual(list(ROOT.rglob(pat)), [], pat)

    def test_no_source_lacks_a_candidate(self):
        self.assertEqual(self.rep["feasibility"]["blocker_zero_candidate_sources"], 0)
        for split in ("TRAIN", "VAL"):
            for ds in ("casia_fasd", "msu_mfsd", "siwmv2"):
                self.assertEqual(self.rep["feasibility"][split][ds]["sources_with_zero_candidates"], 0)

    def test_source_counts_match_the_split_manifest(self):
        import pyarrow.parquet as pq
        rows = pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()
        for split in ("TRAIN", "VAL"):
            n = sum(1 for r in rows if r["split"] == split and r["label_binary"] == 1)
            self.assertEqual(self.rep["feasibility"][split]["_total"]["spoof_sources"], n)

    def test_casia_face_box_gap_is_recorded(self):
        self.assertIs(self.rep["distances"]["scale"]["casia_has_no_face_box"], True)
        self.assertEqual(self.rep["distances"]["scale"]["rows_with_face_box"]["casia_fasd"], 0)

    def test_native_coverage_recorded_without_fabricating_identity(self):
        n = self.rep["native"]
        self.assertIs(n["siwmv2"]["subject_ids_available"], False)
        self.assertEqual(n["siwmv2"]["identities_with_live_and_spoof"], 0)
        self.assertIn("NOT_INSTANTIABLE", n["siwmv2"]["dsdg_native_identity_pairing"])
        self.assertIn("NOT_INSTANTIABLE", n["siwmv2"]["difffas_native_same_id_reconstruction"])
        for ds in ("casia_fasd", "msu_mfsd"):
            self.assertEqual(n[ds]["dsdg_native_identity_pairing"], "SUPPORTED")
            self.assertEqual(n[ds]["identities_with_live_and_spoof"], n[ds]["train_identities"])


class TestProposedConfigIsNotFrozen(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(PROPOSED.read_text())

    def test_it_lives_in_proposed_and_is_blocked(self):
        self.assertFalse((ROOT / "configs/frozen/pairs_v1.yaml").exists())
        self.assertEqual(self.cfg["status"], "PROPOSED_BLOCKED")
        self.assertEqual(self.cfg["blocking_questions"], ["Q-24", "Q-25", "Q-26", "Q-27"])

    def test_every_open_question_is_marked_and_unresolved(self):
        for qid, q in self.cfg["open_questions"].items():
            self.assertEqual(q["status"], "OWNER_DECISION_REQUIRED", qid)
            self.assertIn("gap", q)

    def test_settled_fields_match_the_spec_and_the_code(self):
        s = self.cfg["settled"]
        self.assertEqual(s["candidate_cap"], P.CANDIDATE_CAP)
        self.assertEqual(s["split_seed"], P.SPLIT_SEED)
        self.assertEqual(s["weights"], P.WEIGHTS)
        self.assertEqual(s["build_from_splits"], ["TRAIN", "VAL"])
        self.assertEqual(s["test_pairs"], "NOT_REQUIRED")
        self.assertIs(s["one_pair_per_spoof_source"], True)
        self.assertIs(s["target_reuse_allowed"], True)
        self.assertEqual(s["usable_rows_only"], "M2_COMPLETE")

    def test_expected_row_counts_match_the_measured_sources(self):
        p = AUDIT / "M4_PAIR_PREFLIGHT.json"
        if not p.is_file():
            self.skipTest("preflight not run")
        rep = json.loads(p.read_text())
        self.assertEqual(self.cfg["manifests"]["expected_rows"]["train"],
                         rep["feasibility"]["TRAIN"]["_total"]["spoof_sources"])
        self.assertEqual(self.cfg["manifests"]["expected_rows"]["val"],
                         rep["feasibility"]["VAL"]["_total"]["spoof_sources"])

    def test_dev013_is_not_extended_to_native_same_identity(self):
        self.assertIn("must NOT be extended", self.cfg["native_manifests"]["dev013_scope"])
        self.assertEqual(self.cfg["native_manifests"]["difffas_recon_pairs_v1"]["unsupported"], ["siwmv2"])
        self.assertEqual(self.cfg["native_manifests"]["dsdg_identity_pairs_v1"]["unsupported"], ["siwmv2"])

    def test_m4_is_not_started(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        self.assertEqual(s["M3"]["status"], "COMPLETE")
        self.assertEqual(s["M4"]["status"], "NOT_STARTED")


if __name__ == "__main__":
    unittest.main()
