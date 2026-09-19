"""M3 execution tests: the authoritative split manifest, its leakage properties and determinism.

These assert on the artifact that was actually written (`manifests/split_v1.parquet`) plus the
frozen metadata it came from, not on in-memory state.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.split import allocator as A  # noqa: E402
from gpatbench.split import execute as E  # noqa: E402

AUDIT = ROOT / "outputs/audit"
MANIFEST = ROOT / "manifests/split_v1.parquet"
GROUPS = ROOT / "manifests/split_groups_v1.parquet"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TestManifestExists(unittest.TestCase):
    def test_manifest_and_group_manifest_exist(self):
        self.assertTrue(MANIFEST.is_file())
        self.assertTrue(GROUPS.is_file())


@unittest.skipUnless(MANIFEST.is_file(), "split not executed")
class TestSchemaAndPopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pq.read_table(MANIFEST).to_pylist()
        cls.cols = [f.name for f in pq.read_schema(MANIFEST)]
        cls.acct = pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet").to_pylist()

    def test_required_spec_schema_present_and_first(self):
        for c in E.SPEC_COLUMNS:
            self.assertIn(c, self.cols, c)
        self.assertEqual(self.cols[:len(E.SPEC_COLUMNS)], E.SPEC_COLUMNS)

    def test_row_count_is_exactly_m2_complete(self):
        complete = {r["sample_id"] for r in self.acct if r["final_status"] == "COMPLETE"}
        self.assertEqual(len(complete), 20615)
        self.assertEqual(len(self.rows), 20615)
        self.assertEqual({r["sample_id"] for r in self.rows}, complete)

    def test_no_failed_row_present_and_none_missing(self):
        failed = {r["sample_id"] for r in self.acct if r["final_status"] == "FAILED"}
        self.assertEqual(len(failed), 25)
        ids = {r["sample_id"] for r in self.rows}
        self.assertEqual(failed & ids, set())
        complete = {r["sample_id"] for r in self.acct if r["final_status"] == "COMPLETE"}
        self.assertEqual(complete - ids, set())

    def test_every_row_assigned_exactly_once(self):
        ids = [r["sample_id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(r["split"] in A.SPLITS for r in self.rows))

    def test_m2_status_column_is_complete_only(self):
        self.assertEqual({r["m2_status"] for r in self.rows}, {"COMPLETE"})

    def test_sha256_lineage_follows_the_existing_convention(self):
        inv = {r["sample_id"]: r for r in pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()}
        acct = {r["sample_id"]: r for r in self.acct}
        kinds = {r["dataset"]: set() for r in self.rows}
        for r in self.rows:
            kinds[r["dataset"]].add(r["sha256_kind"])
            self.assertTrue(r["sha256"] and len(r["sha256"]) == 64, r["sample_id"])
            if r["dataset"] == "casia_fasd":
                self.assertEqual(r["sha256"], inv[r["sample_id"]]["sha256"])
            else:
                # M1's PENDING placeholder resolved with M2's canonical frame PNG hash
                self.assertIsNone(inv[r["sample_id"]]["sha256"])
                self.assertEqual(r["sha256"], acct[r["sample_id"]]["frame_png_sha256"])
        self.assertEqual(kinds["casia_fasd"], {"original_frame_bytes"})
        self.assertEqual(kinds["msu_mfsd"], {"m2_canonical_frame_png"})
        self.assertEqual(kinds["siwmv2"], {"m2_canonical_frame_png"})

    def test_siw_subject_id_is_not_manufactured(self):
        siw = [r for r in self.rows if r["dataset"] == "siwmv2"]
        self.assertTrue(siw)
        self.assertTrue(all(r["subject_id_global"] is None for r in siw))
        for ds in ("casia_fasd", "msu_mfsd"):
            self.assertTrue(all(r["subject_id_global"] for r in self.rows if r["dataset"] == ds))


@unittest.skipUnless(MANIFEST.is_file(), "split not executed")
class TestLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pq.read_table(MANIFEST).to_pylist()
        cls.rep = E.audit_split(ROOT)

    def _sets(self, ds, field):
        return {s: {r[field] for r in self.rows if r["dataset"] == ds and r["split"] == s
                    and r[field] is not None} for s in A.SPLITS}

    def _assert_disjoint(self, d, label):
        for i, a in enumerate(A.SPLITS):
            for b in A.SPLITS[i + 1:]:
                self.assertEqual(d[a] & d[b], set(), f"{label}: {a} & {b}")

    def test_casia_subject_leakage_zero(self):
        self._assert_disjoint(self._sets("casia_fasd", "subject_id_global"), "CASIA subject")
        self._assert_disjoint(self._sets("casia_fasd", "video_id"), "CASIA video")
        self._assert_disjoint(self._sets("casia_fasd", "sample_id"), "CASIA sample")

    def test_msu_subject_leakage_zero(self):
        self._assert_disjoint(self._sets("msu_mfsd", "subject_id_global"), "MSU subject")
        self._assert_disjoint(self._sets("msu_mfsd", "video_id"), "MSU video")
        self._assert_disjoint(self._sets("msu_mfsd", "sample_id"), "MSU sample")

    def test_siw_content_group_leakage_zero_and_subject_claim_forbidden(self):
        self._assert_disjoint(self._sets("siwmv2", "content_group_id"), "SiW content group")
        self._assert_disjoint(self._sets("siwmv2", "video_id"), "SiW video")
        self._assert_disjoint(self._sets("siwmv2", "sample_id"), "SiW sample")
        e = self.rep["leakage"]["siwmv2"]
        self.assertEqual(e["subject_leakage"], "N/A")
        self.assertEqual(e["subject_disjoint_claim"], "FORBIDDEN")
        self.assertTrue(e["subject_id_global_all_null"])
        self.assertIn("content-group-disjoint", e["fidelity"])

    def test_siw_exact_duplicate_videos_stay_together(self):
        by_group = {}
        for r in self.rows:
            if r["dataset"] != "siwmv2":
                continue
            by_group.setdefault(r["content_group_id"], set()).add(r["split"])
        multi = [g for g, s in by_group.items() if len(s) > 1]
        self.assertEqual(multi, [])
        # the six known size-2 content groups must each sit in exactly one split
        vids = {}
        for r in self.rows:
            if r["dataset"] == "siwmv2":
                vids.setdefault(r["content_group_id"], set()).add(r["video_id"])
        dup = {g: v for g, v in vids.items() if len(v) > 1}
        self.assertEqual(len(dup), 6, f"expected 6 duplicate groups, got {len(dup)}")
        for g in dup:
            self.assertEqual(len(by_group[g]), 1)

    def test_all_frames_of_a_video_share_its_split(self):
        vs = {}
        for r in self.rows:
            vs.setdefault((r["dataset"], r["video_id"]), set()).add(r["split"])
        self.assertEqual([k for k, v in vs.items() if len(v) > 1], [])

    def test_allocation_group_never_crosses_splits(self):
        gs = {}
        for r in self.rows:
            gs.setdefault((r["dataset"], r["allocation_group_id"]), set()).add(r["split"])
        self.assertEqual([k for k, v in gs.items() if len(v) > 1], [])

    def test_every_dataset_present_in_every_split(self):
        for ds in E.DATASETS:
            for s in A.SPLITS:
                n = sum(1 for r in self.rows if r["dataset"] == ds and r["split"] == s)
                self.assertGreater(n, 0, f"{ds}/{s}")

    def test_audit_reports_overall_pass(self):
        self.assertTrue(self.rep["ok"])
        for k in ("allocation_group_leakage_total", "video_leakage_total", "sample_leakage_total",
                  "subject_leakage_total_casia_msu", "content_group_leakage_total_siw"):
            self.assertEqual(self.rep["checks"][k], 0, k)


@unittest.skipUnless(MANIFEST.is_file(), "split not executed")
class TestDeterminismAndSerialization(unittest.TestCase):
    def test_canonical_row_order(self):
        rows = pq.read_table(MANIFEST).to_pylist()
        self.assertEqual(rows, sorted(rows, key=lambda r: (r["dataset"], r["video_id"],
                                                           r["frame_index"], r["sample_id"])))
        self.assertEqual(list(E.ROW_ORDER), ["dataset", "video_id", "frame_index", "sample_id"])

    def test_recorded_hash_matches_the_file(self):
        rec = json.loads((AUDIT / "split_v1.sha256").read_text())
        self.assertEqual(rec["sha256"], sha(MANIFEST))
        self.assertEqual(rec["group_sha256"], sha(GROUPS))
        self.assertEqual(rec["rows"], pq.read_metadata(MANIFEST).num_rows)

    def test_split_config_hash_linkage(self):
        rec = json.loads((AUDIT / "split_v1.sha256").read_text())
        self.assertEqual(rec["split_config_sha256"], sha(ROOT / "configs/frozen/split_v1.yaml"))
        self.assertEqual(rec["split_config_sha256"],
                         "9dc04edeedbde9b52f301e4648f6a365e96fc5ddf373ea6b85ef0a2096737499")
        for rel, h in rec["inputs"].items():
            self.assertEqual(sha(ROOT / rel), h, rel)

    def test_determinism_report_is_pass_and_byte_identical(self):
        d = json.loads((AUDIT / "M3_DETERMINISM_COMPARE.json").read_text())
        self.assertEqual(d["status"], "PASS")
        self.assertTrue(d["byte_identical"])
        self.assertTrue(d["all_group_hashes_identical"])
        self.assertEqual(d["authoritative_manifest_sha256"], sha(MANIFEST))
        self.assertGreaterEqual(sum(1 for r in d["runs"] if r["fresh_process"]), 2)
        self.assertTrue(any(r["shuffle_seed"] is not None for r in d["runs"]))
        self.assertEqual(len({r["manifest_sha256"] for r in d["runs"]}), 1)

    def test_rebuild_is_byte_identical_with_shuffled_input(self):
        """Re-run the real allocator with permuted metadata and require the same bytes."""
        with tempfile.TemporaryDirectory() as t:
            out = E.run_split(ROOT, write=True, manifest_path=Path(t) / "s.parquet",
                              group_manifest_path=Path(t) / "g.parquet", shuffle_seed=4242)
            self.assertEqual(out["manifest_sha256"], sha(MANIFEST))
            self.assertEqual(out["group_manifest_sha256"], sha(GROUPS))

    def test_parquet_writer_settings_are_frozen_and_recorded(self):
        rec = json.loads((AUDIT / "split_v1.sha256").read_text())
        self.assertEqual(rec["parquet_writer"], E.PARQUET_WRITER)
        self.assertEqual(E.PARQUET_WRITER["compression"], "zstd")
        self.assertIs(E.PARQUET_WRITER["use_dictionary"], False)


@unittest.skipUnless(MANIFEST.is_file(), "split not executed")
class TestObjectivesAndReduction(unittest.TestCase):
    def test_profile_class_equivalence_holds_on_real_groups(self):
        pop = E.load_population(ROOT)
        for ds, groups in pop.items():
            rep = E.verify_profile_class_equivalence(groups)
            self.assertTrue(rep["ok"], ds)
            self.assertGreaterEqual(rep["groups"], rep["classes"])

    def test_group_size_is_part_of_the_profile_signature(self):
        """Group size is objective-relevant (it enters P1), so it must separate classes."""
        g1 = A.Group("ds", "a", (A.Video("v1", 0, "live", None, 8),))
        g2 = A.Group("ds", "b", (A.Video("v2", 0, "live", None, 8), A.Video("v3", 0, "live", None, 8)))
        self.assertNotEqual(g1.class_signature, g2.class_signature)
        # differing categories must separate classes too
        g3 = A.Group("ds", "c", (A.Video("v4", 1, "print", "p", 8),))
        self.assertNotEqual(g1.class_signature, g3.class_signature)
        self.assertTrue(E.verify_profile_class_equivalence([g1, g2, g3])["ok"])

    def test_equivalence_checker_detects_an_injected_collision(self):
        """If two non-equivalent groups ever shared a signature, the checker must refuse."""
        class Colliding(A.Group):
            @property
            def class_signature(self):
                return ("FORCED_COLLISION",)

        a = Colliding("ds", "a", (A.Video("v1", 0, "live", None, 8),))
        b = Colliding("ds", "b", (A.Video("v2", 1, "print", "p", 8), A.Video("v3", 1, "print", "p", 8)))
        with self.assertRaises(A.AllocationError) as e:
            E.verify_profile_class_equivalence([a, b])
        self.assertIn("reduction is unsound", str(e.exception))

    def test_objectives_match_the_frozen_preflight_contract(self):
        pop = E.load_population(ROOT)
        rows = pq.read_table(MANIFEST).to_pylist()
        for ds, groups in pop.items():
            asg = {}
            for r in rows:
                if r["dataset"] == ds:
                    asg[r["allocation_group_id"]] = r["split"]
            recomputed = A.objective_values(groups, asg)
            for lvl in A.LEVELS:
                self.assertIsInstance(recomputed[lvl], int)
            A.verify_assignment(groups, asg, recomputed)

    def test_canonical_video_weight_is_independent_of_frame_survival(self):
        """The 23 partially-failed videos still count as one video each in the allocation."""
        pop = E.load_population(ROOT)
        siw = pop["siwmv2"]
        degraded = [v for g in siw for v in g.videos if v.n_failed > 0]
        self.assertEqual(len(degraded), 23)
        self.assertTrue(all(v.n_complete > 0 for v in degraded))
        rows = pq.read_table(MANIFEST).to_pylist()
        counts = {}
        for r in rows:
            if r["dataset"] == "siwmv2":
                counts.setdefault(r["split"], set()).add(r["video_id"])
        self.assertEqual(sum(len(v) for v in counts.values()), 1700)


@unittest.skipUnless(MANIFEST.is_file(), "split not executed")
class TestFailedSampleAccounting(unittest.TestCase):
    def test_all_25_failures_accounted_for_and_absent_from_the_split(self):
        rows = list(csv.DictReader(open(AUDIT / "M3_FAILED_SAMPLE_SPLIT_ACCOUNTING.csv",
                                        newline="", encoding="utf-8")))
        self.assertEqual(len(rows), 25)
        self.assertEqual({r["failure_reason"] for r in rows}, {"SCRFD_NO_FACE"})
        self.assertEqual({r["dataset"] for r in rows}, {"siwmv2"})
        self.assertEqual({r["in_split_manifest"] for r in rows}, {"False"})
        self.assertTrue(all(r["video_split"] in A.SPLITS for r in rows))
        ids = {r["sample_id"] for r in pq.read_table(MANIFEST).to_pylist()}
        self.assertEqual({r["sample_id"] for r in rows} & ids, set())


@unittest.skipUnless(MANIFEST.is_file(), "split not executed")
class TestCliEquivalence(unittest.TestCase):
    def test_cli_dry_run_reports_the_same_objectives_and_row_count(self):
        res = subprocess.run([sys.executable, "-m", "gpatbench.cli", "split",
                              "--config", "configs/frozen/split_v1.yaml", "--dry-run"],
                             capture_output=True, text=True, cwd=str(ROOT),
                             env=dict(os.environ, PYTHONHASHSEED="3"))
        self.assertEqual(res.returncode, 0, res.stderr[-800:])
        out = json.loads(res.stdout[res.stdout.index("{"):])
        self.assertEqual(out["rows"], 20615)
        self.assertEqual(out["config_sha256"], sha(ROOT / "configs/frozen/split_v1.yaml"))
        self.assertEqual(out["row_order"], list(E.ROW_ORDER))
        self.assertEqual(out["parquet_writer"], E.PARQUET_WRITER)

    def test_cli_refuses_a_non_frozen_config(self):
        with tempfile.TemporaryDirectory() as t:
            fake = Path(t) / "split_v1.yaml"
            fake.write_text((ROOT / "configs/frozen/split_v1.yaml").read_text())
            res = subprocess.run([sys.executable, "-m", "gpatbench.cli", "split",
                                  "--config", str(fake)], capture_output=True, text=True, cwd=str(ROOT))
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("frozen split config", res.stderr + res.stdout)

    def test_cli_audit_split_passes_on_the_authoritative_manifest(self):
        res = subprocess.run([sys.executable, "-m", "gpatbench.cli", "audit-split",
                              "--manifest", "manifests/split_v1.parquet"],
                             capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(res.returncode, 0, res.stderr[-800:])
        out = json.loads(res.stdout[res.stdout.index("{"):])
        self.assertTrue(out["ok"])
        self.assertEqual(out["rows"], 20615)


class TestStageAndNoLaterMilestone(unittest.TestCase):
    def test_no_pair_or_training_artifact(self):
        for pat in ("pairs_*.parquet", "*.ckpt", "*.pt", "probe_*.pt"):
            found = [p for p in ROOT.rglob(pat) if ".git" not in p.parts and ".venv" not in p.parts]
            self.assertEqual(found, [], pat)

    def test_stage_state(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        self.assertEqual(s["M2"]["status"], "COMPLETE")
        self.assertIn(s["M3"]["status"], {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE"})
        self.assertEqual(s["M4"]["status"], "NOT_STARTED")
        if s["M3"]["status"] == "COMPLETE":
            self.assertTrue(MANIFEST.is_file())
            self.assertEqual(json.loads((AUDIT / "M3_DETERMINISM_COMPARE.json").read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
