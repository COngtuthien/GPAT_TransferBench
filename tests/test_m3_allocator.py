"""Tests for the frozen Q-01 lexicographic split allocator.

Every case builds a small synthetic dataset and checks a property that the frozen contract claims.
Nothing here reads the real datasets, and nothing writes an allocation.
"""
import hashlib
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.split import allocator as A  # noqa: E402

AUDIT = ROOT / "outputs/audit"
CFG = ROOT / "configs/frozen/split_v1.yaml"


def V(vid, binary=0, macro="live", raw=None, complete=8, failed=0):
    return A.Video(vid, binary, macro, raw, complete, failed)


def G(gid, videos, dataset="ds"):
    return A.Group(dataset, gid, tuple(videos))


def uniform(n_groups, videos_per_group=1, dataset="ds", **kw):
    return [G(f"g{i:03d}", [V(f"g{i:03d}_v{k}", **kw) for k in range(videos_per_group)], dataset)
            for i in range(n_groups)]


class TestGroupIntegrity(unittest.TestCase):
    def test_01_one_group_goes_to_exactly_one_split(self):
        r = A.allocate(uniform(20))
        self.assertEqual(set(r.assignment.values()) <= set(A.SPLITS), True)
        self.assertEqual(len(r.assignment), 20)
        for g in uniform(20):
            self.assertIn(r.assignment[g.group_id], A.SPLITS)

    def test_02_multi_video_group_stays_intact(self):
        groups = uniform(10, videos_per_group=4)
        r = A.allocate(groups)
        for g in groups:
            splits = {r.assignment[g.group_id] for _ in g.videos}
            self.assertEqual(len(splits), 1, g.group_id)
        # and no canonical video is duplicated across splits
        A.verify_assignment(groups, r.assignment)

    def test_03_subject_disjoint_invariant(self):
        """CASIA/MSU shape: one subject = one group of many videos."""
        groups = [G(f"casia_fasd::{i}", [V(f"s{i}v{k}", k % 2, "live" if k % 2 == 0 else "print",
                                           None if k % 2 == 0 else "3") for k in range(12)])
                  for i in range(50)]
        r = A.allocate(groups)
        video_split = {}
        for g in groups:
            for v in g.videos:
                video_split.setdefault(r.assignment[g.group_id], set()).add(g.group_id)
        allsets = list(video_split.values())
        for i in range(len(allsets)):
            for j in range(i + 1, len(allsets)):
                self.assertEqual(allsets[i] & allsets[j], set(), "a subject appears in two splits")

    def test_04_siw_duplicate_content_group_stays_together(self):
        """Two exact-byte duplicate videos share one content group and may never be separated."""
        groups = uniform(40)
        dup = G("siwmv2::sha256:dup", [V("dupA"), V("dupB")])
        groups.append(dup)
        r = A.allocate(groups)
        self.assertIn(r.assignment["siwmv2::sha256:dup"], A.SPLITS)
        A.verify_assignment(groups, r.assignment)

    def test_19_m2_failed_rows_never_become_usable_samples(self):
        groups = [G("g0", [V("v0", complete=7, failed=1)]), *uniform(19)]
        r = A.allocate(groups)
        usable = sum(r.stats[s]["usable_samples"] for s in A.SPLITS)
        failed = sum(r.stats[s]["failed_samples_provenance_only"] for s in A.SPLITS)
        self.assertEqual(usable, 7 + 19 * 8)
        self.assertEqual(failed, 1)
        self.assertEqual(r.stats["_totals"]["usable_samples"], usable)

    def test_20_frame_counts_do_not_influence_video_balancing(self):
        """A 7-COMPLETE video and an 8-COMPLETE video both weigh exactly one canonical video."""
        healthy = uniform(30)
        degraded = [G(g.group_id, [V(v.video_id, v.label_binary, v.attack_macro, v.attack_raw,
                                     complete=7, failed=1) for v in g.videos]) for g in healthy]
        a, b = A.allocate(healthy), A.allocate(degraded)
        self.assertEqual(a.assignment, b.assignment)
        self.assertEqual(a.objectives, b.objectives)
        self.assertEqual({s: a.stats[s]["canonical_videos"] for s in A.SPLITS},
                         {s: b.stats[s]["canonical_videos"] for s in A.SPLITS})

    def test_zero_usable_video_raises_unresolved_policy(self):
        groups = [G("g0", [V("v0", complete=0, failed=8)]), *uniform(19)]
        with self.assertRaises(A.UnresolvedPolicyError) as e:
            A.allocate(groups)
        self.assertIn("BLOCKED_BY_ZERO_USABLE_CANONICAL_VIDEO", str(e.exception))

    def test_duplicate_group_id_and_duplicate_video_are_rejected(self):
        with self.assertRaises(A.AllocationError):
            A.allocate([G("g", [V("a")]), G("g", [V("b")])])
        with self.assertRaises(A.AllocationError):
            A.allocate([G("g1", [V("same")]), G("g2", [V("same")]), *uniform(5)])

    def test_mixed_datasets_rejected(self):
        with self.assertRaises(A.AllocationError):
            A.allocate([G("a", [V("v1")], "ds1"), G("b", [V("v2")], "ds2")])


class TestRatios(unittest.TestCase):
    def test_05_perfect_ratio_case(self):
        r = A.allocate(uniform(100))
        self.assertEqual(r.objectives["P1_total"], 0)
        self.assertEqual({s: r.stats[s]["canonical_videos"] for s in A.SPLITS},
                         {"TRAIN": 70, "VAL": 15, "TEST": 15})

    def test_06_indivisible_group_size_case(self):
        """7 groups of 3 videos: 70/15/15 of 21 videos is unreachable; deviation must be minimal."""
        groups = uniform(7, videos_per_group=3)
        r = A.allocate(groups)
        best = None
        for a in range(8):
            for b in range(8 - a):
                c = 7 - a - b
                n = (3 * a, 3 * b, 3 * c)
                e = sum(abs(100 * n[i] - p * 21) for i, p in enumerate((70, 15, 15)))
                best = e if best is None else min(best, e)
        self.assertEqual(r.objectives["P1_total"], best)

    def test_p1_matches_brute_force_on_a_single_class(self):
        """With one interchangeable class the optimum is enumerable; the solver must match it."""
        for n_groups, per_group in ((50, 12), (35, 8), (13, 1)):
            groups = uniform(n_groups, videos_per_group=per_group)
            r = A.allocate(groups)
            N = n_groups * per_group
            best = min(sum(abs(100 * per_group * k[i] - p * N) for i, p in enumerate((70, 15, 15)))
                       for k in ((a, b, n_groups - a - b) for a in range(n_groups + 1)
                                 for b in range(n_groups - a + 1)))
            self.assertEqual(r.objectives["P1_total"], best, f"{n_groups}x{per_group}")


class TestPriorityOrder(unittest.TestCase):
    """A later priority may never improve itself at the cost of an earlier one."""

    @staticmethod
    def _best_at_level(groups, level):
        """Brute-force the minimum of one level alone, over all assignments (small inputs only)."""
        ids = [g.group_id for g in groups]
        best = None
        for mask in range(3 ** len(ids)):
            m, asg = mask, {}
            for gid in ids:
                asg[gid] = A.SPLITS[m % 3]
                m //= 3
            v = A.objective_values(groups, asg)[level]
            best = v if best is None else min(best, v)
        return best

    def test_07_p1_beats_p2(self):
        # 6 live-only groups + 1 spoof group: improving the binary balance would require moving
        # groups away from the P1 optimum; P1 must win.
        groups = [*uniform(6, binary=0, macro="live"),
                  G("spoof", [V("sv", 1, "print", "p")])]
        r = A.allocate(groups)
        self.assertEqual(r.objectives["P1_total"], self._best_at_level(groups, "P1_total"))

    def test_08_p2_beats_p3(self):
        groups = [G("a", [V("a1", 0, "live", None)]), G("b", [V("b1", 0, "live", None)]),
                  G("c", [V("c1", 1, "print", "p1")]), G("d", [V("d1", 1, "replay", "r1")]),
                  G("e", [V("e1", 1, "print", "p2")]), G("f", [V("f1", 1, "replay", "r2")])]
        r = A.allocate(groups)
        # P1 and P2 must each equal their own achievable minimum given the previous levels
        self.assertEqual(r.objectives["P1_total"], self._best_at_level(groups, "P1_total"))
        p2_given_p1 = min(A.objective_values(groups, asg)["P2_binary"]
                          for asg in self._all_assignments(groups)
                          if A.objective_values(groups, asg)["P1_total"] == r.objectives["P1_total"])
        self.assertEqual(r.objectives["P2_binary"], p2_given_p1)

    def test_09_p3_beats_p4(self):
        groups = [G(f"g{i}", [V(f"v{i}", 1, "print" if i < 4 else "replay", f"raw{i}")]
                    ) for i in range(8)]
        r = A.allocate(groups)
        p3 = min(A.objective_values(groups, asg)["P3_attack_macro"]
                 for asg in self._all_assignments(groups)
                 if A.objective_values(groups, asg)["P1_total"] == r.objectives["P1_total"]
                 and A.objective_values(groups, asg)["P2_binary"] == r.objectives["P2_binary"])
        self.assertEqual(r.objectives["P3_attack_macro"], p3)
        p4 = min(A.objective_values(groups, asg)["P4_attack_raw"]
                 for asg in self._all_assignments(groups)
                 if A.objective_values(groups, asg)["P1_total"] == r.objectives["P1_total"]
                 and A.objective_values(groups, asg)["P2_binary"] == r.objectives["P2_binary"]
                 and A.objective_values(groups, asg)["P3_attack_macro"] == p3)
        self.assertEqual(r.objectives["P4_attack_raw"], p4)

    @staticmethod
    def _all_assignments(groups):
        ids = [g.group_id for g in groups]
        for mask in range(3 ** len(ids)):
            m, asg = mask, {}
            for gid in ids:
                asg[gid] = A.SPLITS[m % 3]
                m //= 3
            yield asg


class TestRareCategories(unittest.TestCase):
    def test_10_11_12_rare_categories_are_handled_not_forced(self):
        groups = [*uniform(20, binary=0, macro="live"),
                  G("rare_bin", [V("rb", 1, "print", "printed_photo")]),
                  G("rare_macro", [V("rm", 1, "mask_3d", "Mask_Silicone")]),
                  G("rare_raw", [V("rr", 1, "print", "unique_raw")])]
        r = A.allocate(groups)                     # must not raise
        rep = A.rare_category_report(groups)
        self.assertFalse(rep["P2_binary"]["1"]["can_occupy_three_splits"] is None)
        self.assertEqual(rep["P3_attack_macro"]["mask_3d"]["groups"], 1)
        self.assertFalse(rep["P3_attack_macro"]["mask_3d"]["can_occupy_three_splits"])
        self.assertEqual(rep["P4_attack_raw"]["unique_raw"]["groups"], 1)

    def test_13_category_impossible_in_all_three_splits(self):
        """A category carried by a single group can occupy at most one split; that is reported,
        never fixed by breaking a group."""
        groups = [*uniform(19, binary=0, macro="live"),
                  G("only", [V("o1", 1, "mask_2d", "Mask_Paper")])]
        r = A.allocate(groups)
        occupied = {s for s in A.SPLITS
                    if any(v.attack_macro == "mask_2d"
                           for g in groups if r.assignment[g.group_id] == s for v in g.videos)}
        self.assertEqual(len(occupied), 1)
        self.assertFalse(A.rare_category_report(groups)["P3_attack_macro"]["mask_2d"]["can_occupy_three_splits"])


class TestDeterminism(unittest.TestCase):
    def test_14_exact_tie_case(self):
        """Three interchangeable groups: (2,1,0) and (2,0,1) both score 110, an exact tie.

        70/15/15 of three videos does not put one group in each split — the optimum leaves one
        split empty — so the point of this case is that the tie is broken by the frozen seed and
        that the chosen solution really is one of the tied optima.
        """
        groups = uniform(3)
        r = A.allocate(groups)
        counts = tuple(sum(1 for v in r.assignment.values() if v == s) for s in A.SPLITS)
        self.assertIn(counts, {(2, 1, 0), (2, 0, 1)})
        best = min(sum(abs(100 * k[i] - p * 3) for i, p in enumerate((70, 15, 15)))
                   for k in ((a, b, 3 - a - b) for a in range(4) for b in range(4 - a)))
        self.assertEqual(r.objectives["P1_total"], best)
        self.assertEqual(A.allocate(list(reversed(groups))).assignment, r.assignment)

    def test_15_p5_tie_break_is_the_frozen_hash(self):
        want = hashlib.sha256(f"gpatbench.split_v1.tie.v1|ds|g001|20260814".encode()).hexdigest()
        self.assertEqual(A.stable_group_rank("ds", "g001"), want)
        self.assertEqual(len(want), 64)
        self.assertEqual(want, want.lower())
        self.assertNotEqual(A.stable_group_rank("ds", "g001"), A.stable_group_rank("ds2", "g001"))
        self.assertNotEqual(A.stable_group_rank("ds", "g001"), A.stable_group_rank("ds", "g001", 1))

    def test_16_permuted_input_gives_identical_assignment(self):
        base = uniform(37, videos_per_group=2)
        r = A.allocate(base)
        for seed in (0, 1, 2, 3):
            shuffled = list(base)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(A.allocate(shuffled).assignment, r.assignment, f"seed {seed}")

    def test_17_repeated_execution_is_identical(self):
        groups = uniform(23, videos_per_group=3)
        first = A.allocate(groups)
        for _ in range(3):
            again = A.allocate(groups)
            self.assertEqual(again.assignment, first.assignment)
            self.assertEqual(again.objectives, first.objectives)
            self.assertEqual(again.class_counts, first.class_counts)

    def test_18_hash_randomisation_cannot_affect_output(self):
        """Run the allocator in fresh interpreters with different PYTHONHASHSEED values."""
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from gpatbench.split import allocator as A\n"
            "gs=[A.Group('ds', 'g%%03d'%%i, tuple([A.Video('g%%03d_v%%d'%%(i,k), k%%2,"
            "  'live' if k%%2==0 else 'print', None if k%%2==0 else 'r%%d'%%(i%%3), 8)"
            "  for k in range(3)])) for i in range(17)]\n"
            "r=A.allocate(gs)\n"
            "import json; print(json.dumps(sorted(r.assignment.items())))\n" % str(ROOT)
        )
        outs = []
        for hs in ("0", "1", "12345"):
            env = {"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin"}
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env=env, check=True).stdout.strip()
            outs.append(out)
        self.assertEqual(len(set(outs)), 1, "PYTHONHASHSEED changed the allocation")


class TestObjectiveArithmetic(unittest.TestCase):
    def test_objective_is_integer_and_solver_agrees_with_recomputation(self):
        groups = uniform(41, videos_per_group=2, binary=1, macro="print", raw="p")
        r = A.allocate(groups)
        for v in r.objectives.values():
            self.assertIsInstance(v, int)
        self.assertEqual(A.objective_values(groups, r.assignment), r.objectives)

    def test_fixed_point_weight_definition(self):
        """W_c = SCALE // N_c is the frozen definition; check it on a hand-computed case."""
        groups = [G(f"g{i}", [V(f"v{i}", 0 if i < 3 else 1, "live" if i < 3 else "print",
                                None if i < 3 else "p")]) for i in range(10)]
        asg = {g.group_id: "TRAIN" for g in groups}
        got = A.objective_values(groups, asg)["P2_binary"]
        expect = 0
        for N_c, n_train in ((3, 3), (7, 7)):
            w = A.SCALE // N_c
            expect += w * (abs(100 * n_train - 70 * N_c) + abs(0 - 15 * N_c) + abs(0 - 15 * N_c))
        self.assertEqual(got, expect)

    def test_solver_reports_proven_global_optimum(self):
        r = A.allocate(uniform(29, videos_per_group=2))
        self.assertEqual(len(r.evidence), len(A.LEVELS))
        for e in r.evidence:
            self.assertEqual(e["status"], 0)
            self.assertEqual(e["optimality"], "GLOBAL_OPTIMUM_PROVEN")
            self.assertEqual(e["mip_gap"], 0.0)
        self.assertEqual(r.backend["kind"], "EXACT")
        self.assertEqual(r.backend["threads"], 1)

    def test_verify_assignment_rejects_a_split_crossing_video(self):
        groups = uniform(6)
        r = A.allocate(groups)
        broken = dict(r.assignment)
        broken[groups[0].group_id] = "TRAIN"
        broken[groups[1].group_id] = "VAL"
        bad = [G("x", [V("shared")]), G("y", [V("shared")])]
        with self.assertRaises(A.AllocationError):
            A.verify_assignment(bad, {"x": "TRAIN", "y": "VAL"})

    def test_tampered_objective_is_detected(self):
        groups = uniform(11)
        r = A.allocate(groups)
        with self.assertRaises(A.AllocationError):
            A.verify_assignment(groups, r.assignment, {**r.objectives, "P1_total": -1})


class TestFrozenSplitConfig(unittest.TestCase):
    def setUp(self):
        if not CFG.is_file():
            self.skipTest("split_v1.yaml not frozen yet")
        self.cfg = yaml.safe_load(CFG.read_text())

    def test_ratios_seed_and_priorities(self):
        self.assertEqual(self.cfg["ratios"], {"train": 0.70, "val": 0.15, "test": 0.15})
        self.assertEqual(self.cfg["split_seed"], A.SPLIT_SEED)
        self.assertEqual([p["id"] for p in self.cfg["priority_order"]],
                         ["P0", "P1", "P2", "P3", "P4", "P5"])
        self.assertEqual(self.cfg["normalization"]["scale"], A.SCALE)
        self.assertEqual(self.cfg["tie_break"]["namespace"], A.TIE_NAMESPACE)

    def test_allocation_keys_match_the_frozen_policy(self):
        policy = yaml.safe_load((ROOT / "configs/frozen/dataset_protocol_policy_v1.yaml").read_text())
        for ds, want in (("casia_fasd", "subject_id_global"), ("msu_mfsd", "subject_id_global"),
                         ("siwmv2", "content_group_id")):
            self.assertEqual(self.cfg["allocation_keys"][ds]["group_key"], want)
            self.assertEqual(policy["datasets"][ds]["group_key"], want)

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

    def test_contains_no_membership(self):
        """Structural check: the config may describe HOW to split, never WHO goes where.

        A membership table would show up as a mapping whose values are split names. Prose that
        merely mentions TRAIN/VAL/TEST (the ordering rules) is fine, so the check looks at parsed
        structure rather than at substrings.
        """
        self.assertIs(self.cfg["membership_in_this_file"], False)
        offenders = []

        def walk(node, path):
            if isinstance(node, dict):
                vals = [v for v in node.values() if isinstance(v, str)]
                if vals and all(v in A.SPLITS for v in vals) and len(vals) >= 2:
                    offenders.append(path)          # looks like group -> split
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(self.cfg, "")
        self.assertEqual(offenders, [], f"membership-like mappings: {offenders}")
        # the only place split names may be enumerated is the declared split list
        self.assertEqual(self.cfg["splits"], list(A.SPLITS))

    def test_snapshot_is_byte_identical(self):
        snap = ROOT / "frozen_config_snapshot/configs/frozen/split_v1.yaml"
        self.assertTrue(snap.is_file())
        self.assertEqual(snap.read_bytes(), CFG.read_bytes())


class TestNoRealSplitArtifact(unittest.TestCase):
    def test_split_artifacts_only_exist_once_m3_has_started(self):
        import stage_guard
        started = stage_guard.started("M3")
        for pat in ("split_v1.parquet", "split_v1.sha256"):
            found = list(ROOT.rglob(pat))
            if started:
                self.assertTrue(found, f"{pat} missing although M3 has started")
            else:
                self.assertEqual(found, [], pat)
        rel = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
               if p.is_file() and not ({".git", ".venv"} & set(p.parts))}
        self.assertEqual(stage_guard.forbidden_pair_artifacts(rel), [],
                         "a pair artifact exists that the current stage does not account for")

    def test_preflight_feasibility_report_persists_no_membership(self):
        p = AUDIT / "M3_ALLOCATOR_FEASIBILITY.md"
        if not p.is_file():
            self.skipTest("feasibility report not generated yet")
        j = AUDIT / "M3_ALLOCATOR_OBJECTIVE.json"
        if j.is_file():
            blob = json.loads(j.read_text())
            self.assertNotIn("assignment", json.dumps(blob))
            for ds in blob.get("datasets", {}).values():
                self.assertNotIn("class_counts", ds)
                self.assertNotIn("groups_by_split", ds)

    def test_stage_order_is_respected(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        self.assertEqual(s["M2"]["status"], "COMPLETE")
        self.assertIn(s["M3"]["status"], {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE"})
        # M4 may have started; COMPLETE is only legitimate once the native manifests are settled.
        self.assertIn(s["M4"]["status"], {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE"})
        # Amendment A1: M4 may be COMPLETE while the NATIVE (Track-B) manifests are still missing,
        # but only if the amendment record says so explicitly and does not claim the original rule
        # was met.
        if s["M4"]["status"] == "COMPLETE":
            native = (s["M4"].get("execution") or {}).get("native") or {}
            amend = s["M4"].get("amendment_a1")
            if native.get("status") == "BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP":
                self.assertIsNotNone(amend, "M4 COMPLETE with a native blocker needs Amendment A1")
                self.assertEqual(amend["completion_basis"], "AMENDMENT_A1_MAIN_FAIR_IDFREE_TRACK")
                self.assertEqual(amend["track_b_native_pairing"],
                                 "DEFERRED_TO_M6_SECONDARY_TRACK")
                self.assertIs(amend["original_native_manifests_completed"], False)
        self.assertEqual(s["M5"]["status"], "NOT_STARTED")


if __name__ == "__main__":
    unittest.main()
