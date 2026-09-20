"""M4 authoritative pair-manifest execution tests.

These assert the written artifacts, not a re-implementation of the rule: the manifests are read back
and checked against the frozen contract, the frozen inputs and an independent recomputation.
"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P      # noqa: E402
from gpatbench.pairs import execute as E     # noqa: E402

AUDIT = ROOT / "outputs/audit"
MANIFESTS = ROOT / "manifests"
EXPECTED = {"TRAIN": 8838, "VAL": 1905}
EXPECTED_BY_DATASET = {"TRAIN": {"casia_fasd": 2520, "msu_mfsd": 1200, "siwmv2": 5118},
                       "VAL": {"casia_fasd": 576, "msu_mfsd": 240, "siwmv2": 1089}}

_CACHE = {}


def manifest(split):
    if ("m", split) not in _CACHE:
        _CACHE[("m", split)] = pq.read_table(E.MANIFEST_PATH[split]).to_pylist()
    return _CACHE[("m", split)]


def stats_doc():
    if "s" not in _CACHE:
        _CACHE["s"] = json.loads(E.STATS_PATH.read_text())
    return _CACHE["s"]


def split_rows():
    if "sp" not in _CACHE:
        _CACHE["sp"] = pq.read_table(E.SPLIT_MANIFEST).to_pylist()
    return _CACHE["sp"]


def population():
    """Full frozen population. Loaded at most once for the whole module (reads M2 artifacts)."""
    if "pop" not in _CACHE:
        _CACHE["pop"] = E.load_population()
    return _CACHE["pop"]


# ------------------------------------------------------------------ frozen inputs
class TestFrozenInputs(unittest.TestCase):
    def test_pairs_config_hash_is_the_frozen_one(self):
        self.assertEqual(E.sha256_file(E.PAIRS_CONFIG),
                         "f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985")
        self.assertEqual(E.sha256_file(E.PAIRS_CONFIG),
                         E.sha256_file(ROOT / "frozen_config_snapshot/configs/frozen/pairs_v1.yaml"))

    def test_split_manifest_is_unchanged(self):
        self.assertEqual(E.sha256_file(E.SPLIT_MANIFEST),
                         "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d")

    def test_every_row_carries_the_input_provenance(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                self.assertEqual(r["split_manifest_sha256"], E.sha256_file(E.SPLIT_MANIFEST))
                self.assertEqual(r["pairs_config_sha256"], E.sha256_file(E.PAIRS_CONFIG))


# ------------------------------------------------------------------ TRAIN statistics (Q-25)
class TestTrainStats(unittest.TestCase):
    def test_fitted_on_train_only(self):
        d = stats_doc()
        self.assertEqual(d["fitted_on"], "TRAIN")
        self.assertEqual(d["ddof"], 0)
        self.assertEqual(d["std_convention"], "population")
        self.assertEqual(d["dtype"], "float64")
        self.assertIn("none", d["val_contribution"])
        self.assertEqual(d["test_contribution"], "none")

    def test_counts_match_the_train_rows_of_each_dataset(self):
        rows = [r for r in split_rows() if r["split"] == "TRAIN"]
        for ds, e in stats_doc()["datasets"].items():
            self.assertEqual(e["n_train_complete"], sum(1 for r in rows if r["dataset"] == ds))

    def test_no_val_or_test_row_could_have_contributed(self):
        """n_train_complete must equal TRAIN only — never TRAIN+VAL and never the whole split."""
        for ds, e in stats_doc()["datasets"].items():
            all_rows = sum(1 for r in split_rows() if r["dataset"] == ds)
            trainval = sum(1 for r in split_rows()
                           if r["dataset"] == ds and r["split"] in ("TRAIN", "VAL"))
            self.assertNotEqual(e["n_train_complete"], all_rows)
            self.assertNotEqual(e["n_train_complete"], trainval)

    def test_std_is_positive_and_above_the_guard(self):
        for e in stats_doc()["datasets"].values():
            for v in e["pose_std_population_pitch_yaw_roll"]:
                self.assertGreater(v, P.MIN_POSE_STD)

    def test_required_provenance_fields_present(self):
        d = stats_doc()
        for k in ("schema_version", "split_manifest_sha256", "pairs_config_sha256", "split_seed",
                  "dtype", "ddof", "json_policy", "generator", "datasets", "axes"):
            self.assertIn(k, d)
        self.assertEqual(d["split_seed"], P.SPLIT_SEED)
        self.assertEqual(d["axes"], ["pitch", "yaw", "roll"])
        for k in ("common_module_sha256", "execute_module_sha256", "numpy_version",
                  "pyarrow_version"):
            self.assertIn(k, d["generator"])

    def test_json_serialisation_is_canonical_and_reproduces_the_file_bytes(self):
        self.assertEqual(E.canonical_json_bytes(stats_doc()), E.STATS_PATH.read_bytes())
        self.assertTrue(E.STATS_PATH.read_bytes().endswith(b"\n"))
        self.assertNotIn(b"\r", E.STATS_PATH.read_bytes())

    def test_stats_hash_matches_the_frozen_record(self):
        rec = json.loads((AUDIT / "pair_train_stats_v1.sha256").read_text())
        self.assertEqual(rec["sha256"], E.sha256_file(E.STATS_PATH))

    def test_fit_is_independent_of_row_order(self):
        """Floating-point accumulation is not associative; the fit must not expose that."""
        import random
        mk = lambda sid, pose: P.Sample(sid, "d", "TRAIN", 0, "v", "s", None, "live", None,
                                        "0" * 64, pose, 1.0, 0.5)
        rows = [mk("s%04d" % i, (i * 0.001, i * 0.002, i * 0.003)) for i in range(500)]
        ref = P.fit_pose_stats(rows, "d")
        for seed in (1, 2, 3):
            sh = list(rows)
            random.Random(seed).shuffle(sh)
            got = P.fit_pose_stats(sh, "d")
            self.assertEqual(got.mean, ref.mean)
            self.assertEqual(got.std, ref.std)


# ------------------------------------------------------------------ manifest shape
class TestManifestShape(unittest.TestCase):
    def test_row_counts(self):
        self.assertEqual(len(manifest("TRAIN")), EXPECTED["TRAIN"])
        self.assertEqual(len(manifest("VAL")), EXPECTED["VAL"])

    def test_row_counts_per_dataset(self):
        for split, want in EXPECTED_BY_DATASET.items():
            got = {}
            for r in manifest(split):
                got[r["dataset"]] = got.get(r["dataset"], 0) + 1
            self.assertEqual(got, want)

    def test_schema_is_the_frozen_contract(self):
        cfg = yaml.safe_load(E.PAIRS_CONFIG.read_text())["manifests"]
        for split in ("TRAIN", "VAL"):
            names = pq.read_schema(E.MANIFEST_PATH[split]).names
            self.assertEqual(names, list(E.COLUMNS))
            self.assertEqual(names[:len(cfg["spec_columns"])], cfg["spec_columns"])
            self.assertEqual(sorted(names[len(cfg["spec_columns"]):]), sorted(cfg["audit_columns"]))

    def test_column_types(self):
        for split in ("TRAIN", "VAL"):
            s = pq.read_schema(E.MANIFEST_PATH[split])
            for c in E.FLOAT_COLUMNS:
                self.assertEqual(s.field(c).type, pa.float64(), c)
            for c in E.INT_COLUMNS:
                self.assertEqual(s.field(c).type, pa.int64(), c)

    def test_one_pair_per_spoof_source_no_missing_no_duplicate(self):
        for split in ("TRAIN", "VAL"):
            want = {r["sample_id"] for r in split_rows()
                    if r["split"] == split and r["label_binary"] == 1}
            got = [r["source_spoof_id"] for r in manifest(split)]
            self.assertEqual(len(got), len(set(got)), "duplicate source")
            self.assertEqual(set(got), want, "source set != spoof set")

    def test_canonical_row_order(self):
        for split in ("TRAIN", "VAL"):
            rows = manifest(split)
            key = lambda r: tuple(r[c] for c in E.ROW_ORDER)
            self.assertEqual([key(r) for r in rows], sorted(key(r) for r in rows))

    def test_pair_id_sequencing_and_disjoint_namespaces(self):
        train = [r["pair_id"] for r in manifest("TRAIN")]
        val = [r["pair_id"] for r in manifest("VAL")]
        self.assertEqual(train, ["PTR%06d" % i for i in range(1, len(train) + 1)])
        self.assertEqual(val, ["PVA%06d" % i for i in range(1, len(val) + 1)])
        self.assertEqual(set(train) & set(val), set())

    def test_seed_column(self):
        for split in ("TRAIN", "VAL"):
            self.assertEqual({r["seed"] for r in manifest(split)}, {P.SPLIT_SEED})


# ------------------------------------------------------------------ eligibility and leakage
class TestPairValidity(unittest.TestCase):
    def setUp(self):
        self.by_id = {r["sample_id"]: r for r in split_rows()}

    def test_source_spoof_target_live_same_dataset_same_split(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                s, t = self.by_id[r["source_spoof_id"]], self.by_id[r["target_live_id"]]
                self.assertEqual(s["label_binary"], 1)
                self.assertEqual(t["label_binary"], 0)
                self.assertEqual(s["dataset"], t["dataset"])
                self.assertEqual(s["dataset"], r["dataset"])
                self.assertEqual(s["split"], split)
                self.assertEqual(t["split"], split)
                self.assertEqual(r["split"], split)

    def test_no_split_crossing_and_no_test_reference(self):
        test_ids = {r["sample_id"] for r in split_rows() if r["split"] == "TEST"}
        self.assertTrue(test_ids)
        for split in ("TRAIN", "VAL"):
            refs = {r["source_spoof_id"] for r in manifest(split)} | \
                   {r["target_live_id"] for r in manifest(split)}
            self.assertEqual(refs & test_ids, set())
            other = {r["sample_id"] for r in split_rows()
                     if r["split"] not in (split,)}
            self.assertEqual(refs & other, set())

    def test_no_m2_failed_sample_is_referenced(self):
        acct = pq.read_table(E.ACCOUNTING).to_pylist()
        failed = {r["sample_id"] for r in acct if r["final_status"] != "COMPLETE"}
        self.assertEqual(len(failed), 25)
        for split in ("TRAIN", "VAL"):
            refs = {r["source_spoof_id"] for r in manifest(split)} | \
                   {r["target_live_id"] for r in manifest(split)}
            self.assertEqual(refs & failed, set())
            for sid in refs:
                self.assertEqual(self.by_id[sid]["m2_status"], "COMPLETE")

    def test_casia_and_msu_pairs_use_different_subjects(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                if r["dataset"] in ("casia_fasd", "msu_mfsd"):
                    self.assertIsNotNone(r["source_subject"])
                    self.assertIsNotNone(r["target_subject"])
                    self.assertNotEqual(r["source_subject"], r["target_subject"])

    def test_siw_pairs_satisfy_dev013_and_never_claim_identity(self):
        n = 0
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                if r["dataset"] != "siwmv2":
                    continue
                n += 1
                self.assertIsNone(r["source_subject"])
                self.assertIsNone(r["target_subject"])
                self.assertNotEqual(r["source_video_id"], r["target_video_id"])
                self.assertNotEqual(r["source_content_group_id"], r["target_content_group_id"])
        self.assertEqual(n, 5118 + 1089)

    def test_attack_macro_is_the_sources(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                self.assertEqual(r["attack_macro"], self.by_id[r["source_spoof_id"]]["attack_macro"])

    def test_lineage_hashes_come_from_the_split_manifest(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                self.assertEqual(r["source_sha256"], self.by_id[r["source_spoof_id"]]["sha256"])
                self.assertEqual(r["target_sha256"], self.by_id[r["target_live_id"]]["sha256"])


# ------------------------------------------------------------------ distances
class TestDistances(unittest.TestCase):
    def test_all_finite_and_non_negative(self):
        import math
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                for k in ("d_pose", "d_scale", "d_luma", "d_pair"):
                    self.assertTrue(math.isfinite(r[k]))
                    self.assertGreaterEqual(r[k], 0.0)

    def test_d_pair_is_the_exact_frozen_combination(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                self.assertEqual(r["d_pair"],
                                 0.50 * r["d_pose"] + 0.30 * r["d_scale"] + 0.20 * r["d_luma"])

    def test_casia_d_scale_is_exactly_zero_and_fractions_are_one(self):
        n = 0
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                if r["dataset"] != "casia_fasd":
                    continue
                n += 1
                self.assertEqual(r["d_scale"], 0.0)
                self.assertEqual(r["face_area_fraction_source"], 1.0)
                self.assertEqual(r["face_area_fraction_target"], 1.0)
        self.assertEqual(n, 2520 + 576)

    def test_non_casia_fractions_are_in_the_open_unit_interval(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                if r["dataset"] == "casia_fasd":
                    continue
                for k in ("face_area_fraction_source", "face_area_fraction_target"):
                    self.assertGreater(r[k], 0.0)
                    self.assertLessEqual(r[k], 1.0)

    def test_d_scale_recomputes_from_the_stored_fractions(self):
        import math
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                self.assertEqual(r["d_scale"], abs(math.log(r["face_area_fraction_target"]) -
                                                   math.log(r["face_area_fraction_source"])))


# ------------------------------------------------------------------ candidate rule (Q-24)
class TestCandidateRule(unittest.TestCase):
    def test_cap_is_64_everywhere_and_eligible_never_below_it(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                self.assertEqual(r["candidate_count_evaluated"], P.CANDIDATE_CAP)
                self.assertGreaterEqual(r["candidate_count_eligible"], P.CANDIDATE_CAP)

    def test_optimised_selection_equals_the_reference_implementation(self):
        """execute.select_candidates is a faster spelling of common.select_candidates, not a variant."""
        live = [P.Sample("t%05d" % i, "casia_fasd", "TRAIN", 0, "v%d" % i, "s%d" % i, None,
                         "live", None, "0" * 64, (0.0, 0.0, 0.0), 1.0, 0.5) for i in range(500)]
        for sid in ("srcA", "srcB", "casia_fasd/x.avi#000001"):
            src = P.Sample(sid, "casia_fasd", "TRAIN", 1, "v", "zz", None, "print", "3",
                           "0" * 64, (0.0, 0.0, 0.0), 1.0, 0.5)
            a = [t.sample_id for t in P.select_candidates(src, live)]
            b = [t.sample_id for t in E.select_candidates(src, live)]
            self.assertEqual(a, b)
            self.assertEqual(len(b), P.CANDIDATE_CAP)

    def test_raw_byte_preimage_is_used(self):
        sd = hashlib.sha256(b"gpatbench.pair.source_seed.v1|src|20260814").digest()
        self.assertEqual(P.source_seed_digest("src"), sd)
        self.assertEqual(P.candidate_rank_digest(sd, "tgt"),
                         hashlib.sha256(sd + b"|gpatbench.pair.candidate.v1|tgt").digest())
        forbidden = hashlib.sha256(
            ("gpatbench.pair.candidate.v1|" + sd.hex() + "|tgt").encode()).digest()
        self.assertNotEqual(P.candidate_rank_digest(sd, "tgt"), forbidden)

    def test_candidate_selection_audit_evidence_is_consistent(self):
        rows = list(csv.DictReader((AUDIT / "M4_CANDIDATE_SELECTION_AUDIT.csv").open()))
        self.assertGreaterEqual(len(rows), 150)
        by_pair = {}
        for split in ("TRAIN", "VAL"):
            for r in manifest(split):
                by_pair[r["pair_id"]] = r
        for r in rows:
            self.assertEqual(r["winner_in_selected"], "True")
            self.assertEqual(int(r["selected_count"]), P.CANDIDATE_CAP)
            self.assertGreaterEqual(int(r["eligible_count"]), P.CANDIDATE_CAP)
            m = by_pair[r["pair_id"]]
            self.assertEqual(m["target_live_id"], r["winner_target_id"])
            self.assertEqual(repr(m["d_pair"]), r["winner_d_pair"])


# ------------------------------------------------------------------ independent recomputation
class TestRecomputation(unittest.TestCase):
    """Rebuilds real pairs from the frozen inputs and compares with what was written."""
    @classmethod
    def setUpClass(cls):
        cls.pop = population()
        cls.doc = E.build_train_stats(cls.pop)
        cls.stats = E.stats_to_posestats(cls.doc)
        cls.by_id = {s.sample_id: s for s in cls.pop["samples"]}

    def test_recomputed_stats_equal_the_frozen_file(self):
        self.assertEqual(self.doc["datasets"], stats_doc()["datasets"])

    def test_winner_is_the_minimum_of_the_evaluated_set_on_a_real_subset(self):
        salt = "gpatbench.m4_test_winner.v1|"
        for split in ("TRAIN", "VAL"):
            live = {}
            for s in self.pop["samples"]:
                if s.split == split and s.label_binary == 0:
                    live.setdefault(s.dataset, []).append(s)
            for ds in live:
                live[ds].sort(key=lambda t: t.sample_id)
            for ds in ("casia_fasd", "msu_mfsd", "siwmv2"):
                rows = sorted([r for r in manifest(split) if r["dataset"] == ds],
                              key=lambda r: hashlib.sha256(
                                  (salt + r["source_spoof_id"]).encode()).hexdigest())
                for r in rows[:3]:
                    src = self.by_id[r["source_spoof_id"]]
                    cands = E.select_candidates(src, P.eligible_targets(src, live[ds]))
                    best = min((P.d_pair(src, t, self.stats[ds])["d_pair"], t.sample_id)
                               for t in cands)
                    self.assertEqual(best[1], r["target_live_id"])
                    self.assertEqual(best[0], r["d_pair"])
                    self.assertEqual(len(cands), r["candidate_count_evaluated"])

    def test_components_recompute_exactly_on_a_real_subset(self):
        for split in ("TRAIN", "VAL"):
            for r in manifest(split)[:200]:
                src, tgt = self.by_id[r["source_spoof_id"]], self.by_id[r["target_live_id"]]
                d = P.d_pair(src, tgt, self.stats[r["dataset"]])
                for k in ("d_pose", "d_scale", "d_luma", "d_pair"):
                    self.assertEqual(d[k], r[k])

    def test_parquet_write_is_byte_deterministic(self):
        for split in ("TRAIN", "VAL"):
            rows = E.build_pairs(self.pop, split, self.stats)
            with tempfile.TemporaryDirectory() as td:
                a = Path(td) / "a.parquet"
                b = Path(td) / "b.parquet"
                E.write_manifest(rows, a)
                E.write_manifest(list(reversed(rows)) and rows, b)
                self.assertEqual(a.read_bytes(), b.read_bytes())
                self.assertEqual(E.sha256_file(a), E.sha256_file(E.MANIFEST_PATH[split]))

    def test_input_permutation_does_not_change_membership(self):
        import random
        pop2 = dict(self.pop)
        pop2["samples"] = list(self.pop["samples"])
        random.Random(99).shuffle(pop2["samples"])
        doc2 = E.build_train_stats(pop2)
        self.assertEqual(doc2["datasets"], self.doc["datasets"])
        stats2 = E.stats_to_posestats(doc2)
        for split in ("TRAIN", "VAL"):
            self.assertEqual(E.build_pairs(pop2, split, stats2), manifest(split))


# ------------------------------------------------------------------ determinism evidence
class TestDeterminismEvidence(unittest.TestCase):
    def setUp(self):
        self.rep = json.loads((AUDIT / "M4_DETERMINISM_COMPARE.json").read_text())

    def test_recorded_runs_reproduced_the_current_files(self):
        self.assertEqual(self.rep["status"], "PASS")
        self.assertEqual(self.rep["mismatches"], [])
        for name, path in (("pair_train_stats_v1.json", E.STATS_PATH),
                           ("pairs_train_v1.parquet", E.MANIFEST_PATH["TRAIN"]),
                           ("val_pairs_v1.parquet", E.MANIFEST_PATH["VAL"])):
            self.assertEqual(self.rep["authoritative_sha256"][name], E.sha256_file(path))

    def test_runs_cover_shuffled_input_and_a_fresh_hashseed(self):
        runs = self.rep["runs"]
        self.assertTrue(any(v["shuffle_seed"] is not None for v in runs.values()))
        self.assertGreater(len({v["pythonhashseed"] for v in runs.values()}), 1)
        for name, v in runs.items():
            self.assertTrue(v["byte_identical"], name)
            self.assertTrue(all(v["rows_identical"].values()), name)
            self.assertTrue(all(v["schema_identical"].values()), name)
            self.assertTrue(all(v["membership_identical"].values()), name)
            self.assertTrue(all(v["pair_id_identical"].values()), name)

    def test_common_pair_audit_passed_against_these_files(self):
        rep = json.loads((AUDIT / "M4_COMMON_PAIR_AUDIT.json").read_text())
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["failures"], [])
        for split in ("TRAIN", "VAL"):
            self.assertEqual(rep["splits"][split]["sha256"], E.sha256_file(E.MANIFEST_PATH[split]))
            self.assertEqual(rep["splits"][split]["rows"], EXPECTED[split])

    def test_frozen_sha_records_match_the_files(self):
        for name, path in (("pairs_train_v1.sha256", E.MANIFEST_PATH["TRAIN"]),
                           ("val_pairs_v1.sha256", E.MANIFEST_PATH["VAL"]),
                           ("pair_train_stats_v1.sha256", E.STATS_PATH)):
            rec = json.loads((AUDIT / name).read_text())
            self.assertEqual(rec["sha256"], E.sha256_file(path))
            self.assertEqual(rec["pairs_config_sha256"], E.sha256_file(E.PAIRS_CONFIG))
            self.assertEqual(rec["split_manifest_sha256"], E.sha256_file(E.SPLIT_MANIFEST))


# ------------------------------------------------------------------ CLI
class TestCli(unittest.TestCase):
    def test_cli_uses_the_library_path_and_reproduces_the_frozen_hash(self):
        r = subprocess.run([sys.executable, "-m", "gpatbench.cli", "build-pairs",
                            "--config", "configs/frozen/pairs_v1.yaml", "--audit-only"],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        out = json.loads(r.stdout)
        self.assertFalse(out["written"])
        self.assertEqual(out["rows"], EXPECTED)
        self.assertEqual(out["pairs_config_sha256"], E.sha256_file(E.PAIRS_CONFIG))
        self.assertEqual(out["split_manifest_sha256"], E.sha256_file(E.SPLIT_MANIFEST))
        self.assertEqual(out["schema_signature"], E.schema_signature())
        self.assertEqual(out["parquet_writer"], dict(E.PARQUET_WRITER))
        self.assertEqual(out["sha256"]["pair_train_stats_v1.json"], E.sha256_file(E.STATS_PATH))

    def test_cli_refuses_a_non_frozen_config(self):
        r = subprocess.run([sys.executable, "-m", "gpatbench.cli", "build-pairs",
                            "--config", "configs/proposed/pairs_v1.proposed.yaml"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("STOP", r.stderr + r.stdout)

    def test_there_is_no_second_builder(self):
        """The CLI must delegate to execute.run, never drive the internals itself."""
        import ast
        src = (ROOT / "gpatbench/cli.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "cmd_build_pairs")
        called = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and \
                    isinstance(n.func.value, ast.Name) and n.func.value.id == "E":
                called.add(n.func.attr)
        self.assertIn("run", called)
        for bad in ("build_pairs", "write_manifest", "write_train_stats", "select_candidates",
                    "build_train_stats", "load_population"):
            self.assertNotIn(bad, called, f"CLI must not drive execute.{bad} itself")


# ------------------------------------------------------------------ native manifests
class TestNativeManifests(unittest.TestCase):
    def test_no_native_manifest_was_materialized(self):
        for name in ("dsdg_identity_pairs_v1.parquet", "difffas_recon_pairs_v1.parquet"):
            self.assertFalse((MANIFESTS / name).exists(),
                             f"{name} must not exist while the source gap is open")

    def test_coverage_table_records_siw_as_zero_and_never_as_rows(self):
        rows = list(csv.DictReader((AUDIT / "M4_NATIVE_PAIR_COVERAGE.csv").open()))
        self.assertEqual({r["method"] for r in rows}, {"DSDG", "DiffFAS"})
        for r in rows:
            self.assertEqual(int(r["native_rows_materialized"]), 0)
            if r["dataset"] == "siwmv2":
                self.assertEqual(r["status"], "NOT_INSTANTIABLE_MISSING_SUBJECT_ID")
                self.assertEqual(int(r["eligible_identities_live_and_spoof"]), 0)
                self.assertEqual(int(r["total_train_identities"]), 0)
            else:
                self.assertGreater(int(r["eligible_identities_live_and_spoof"]), 0)

    def test_dsdg_and_difffas_require_a_trustworthy_identity(self):
        cfg = yaml.safe_load(E.PAIRS_CONFIG.read_text())["native_manifests"]
        for k in ("dsdg_identity_pairs_v1", "difffas_recon_pairs_v1"):
            self.assertEqual(cfg[k]["datasets"]["siwmv2"], "NOT_INSTANTIABLE_MISSING_SUBJECT_ID")
            self.assertEqual(cfg[k]["siw_representation"]["native_identity_pair_coverage"], 0)
            self.assertEqual(cfg[k]["siw_representation"]["fake_rows"], "forbidden")
        self.assertEqual(cfg["difffas_recon_pairs_v1"]["requires"],
                         ["same_dataset", "same_trustworthy_identity", "live_and_spoof"])
        self.assertEqual(cfg["difffas_recon_pairs_v1"]["style_id"], "dataset::attack_raw")

    def test_dev013_and_surrogates_are_forbidden_as_identity(self):
        cfg = yaml.safe_load(E.PAIRS_CONFIG.read_text())["native_manifests"]
        for item in ("DEV-013_as_a_same_identity_substitute",
                     "content_group_id_as_person_identity",
                     "video_id_as_person_identity",
                     "invented_or_pseudo_siw_subject_ids",
                     "pairing_arbitrary_live_and_spoof_rows_to_imitate_same_id_reconstruction"):
            self.assertIn(item, cfg["forbidden_identity_substitutes"])

    def test_siw_really_has_no_subject_id(self):
        siw = [r for r in split_rows() if r["dataset"] == "siwmv2"]
        self.assertTrue(siw)
        self.assertEqual({r["subject_id_global"] for r in siw}, {None})

    def test_source_evidence_guard_is_recorded(self):
        """A native manifest may not be written while the official construction is unpinned."""
        reg = {m["method_id"]: m for m in
               yaml.safe_load((ROOT / "third_party/registry.yaml").read_text())["methods"]}
        for mid in ("E06b", "E07b"):
            self.assertIsNone(reg[mid]["pinned_commit"])
        audit = (AUDIT / "M4_NATIVE_PAIR_SOURCE_AUDIT.md").read_text()
        self.assertIn("BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP", audit)
        for d in ("methods/dsdg", "methods/difffas"):
            self.assertEqual([p.name for p in (ROOT / d).iterdir()], [".gitkeep"])

    def test_m4_is_not_marked_complete_while_native_is_blocked(self):
        st = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]["M4"]
        self.assertNotEqual(st["status"], "COMPLETE")
        self.assertEqual(st["status"], "IN_PROGRESS")


if __name__ == "__main__":
    unittest.main()
