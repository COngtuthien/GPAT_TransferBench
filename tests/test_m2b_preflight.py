"""M2B tests: storage preflight arithmetic and the cache-integrity primitives it gates.

M2B full preprocessing did NOT run: the mandatory storage preflight returned
BLOCKED_BY_STORAGE_CAPACITY. These tests therefore cover what exists and what the gate asserts:
the preflight's arithmetic and safety reserve, the CASIA frame policy that the projection relies on,
the bounded/streaming shard construction, and the shard-index integrity audit that the M2B cache
audit will reuse. Tests for resume semantics, per-sample state transitions and full-cache
completeness are deliberately NOT written as stubs here: they belong with the M2B execution code,
which is correctly not implemented while the gate blocks.
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from gpatbench.preprocess import logit_store as LS  # noqa: E402

AUDIT = ROOT / "outputs/audit"
PREFLIGHT_JSON = AUDIT / "M2B_STORAGE_PREFLIGHT.json"
PREFLIGHT_MD = AUDIT / "M2B_STORAGE_PREFLIGHT.md"
GIB = 1024 ** 3


class TestPreflightEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = json.loads(PREFLIGHT_JSON.read_text())
        cls.d = cls.rep["decision"]
        cls.c = cls.rep["components"]

    def test_report_pair_exists_and_agrees(self):
        self.assertTrue(PREFLIGHT_MD.is_file())
        self.assertIn(self.d["gate"], PREFLIGHT_MD.read_text())

    def test_counts_match_the_frozen_m1_inventory(self):
        import pyarrow.parquet as pq
        import collections
        inv = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
        self.assertEqual(self.rep["total_samples"], len(inv))
        self.assertEqual(self.rep["sample_counts_by_dataset"],
                         dict(collections.Counter(r["dataset"] for r in inv)))
        for ds, classes in self.rep["sample_counts_by_resolution"].items():
            self.assertEqual(sum(classes.values()), self.rep["sample_counts_by_dataset"][ds], ds)

    def test_frame_projection_arithmetic(self):
        total = 0
        for key, v in self.c["frames"]["by_class"].items():
            self.assertEqual(v["pixels_per_frame"], v["width"] * v["height"])
            expect = v["samples"] * v["pixels_per_frame"] * v["bytes_per_pixel_max"]
            self.assertAlmostEqual(v["bytes_max"], expect, delta=1, msg=key)
            self.assertLessEqual(v["bytes_mean"], v["bytes_max"] + 1, key)
            total += v["bytes_max"]
        self.assertAlmostEqual(self.c["frames"]["bytes_max"], total, delta=1)

    def test_casia_contributes_no_frame_artifact(self):
        """The projection relies on CASIA needing no duplicated frame file; that must follow
        from the frozen contract, not from an assumption."""
        self.assertNotIn("casia_fasd", {k.split("|")[0] for k in self.c["frames"]["by_class"]})
        cfg = yaml.safe_load((ROOT / "configs/frozen/preprocess_v1.yaml").read_text())
        self.assertEqual(cfg["routes"]["casia_fasd"], "PRECROPPED_112_RGB_TO_256_INTER_CUBIC")
        self.assertIn("VideoCapture", cfg["frame_extraction"]["decoder"]["api"])   # video route only
        # and the smoke, which implements the frozen routes, wrote no CASIA frame file
        rows = list(csv.DictReader(open(AUDIT / "M2A_SMOKE_RESULTS.csv", newline="", encoding="utf-8")))
        casia = [r for r in rows if r["dataset"] == "casia_fasd"]
        self.assertTrue(casia)
        for r in casia:
            self.assertEqual((r["scrfd_applied"], r["scrfd_status"]), ("False", "N/A"))
            self.assertEqual(r["frame_png_sha256"], r["frame_png_sha256"].lower())
        self.assertEqual(self.c["frames"]["files"],
                         sum(n for ds, n in self.rep["sample_counts_by_dataset"].items() if ds != "casia_fasd"))

    def test_persistent_total_is_the_sum_of_its_components(self):
        parts = [v["bytes_max"] for k, v in self.c.items() if k != "persistent_total"]
        self.assertAlmostEqual(self.c["persistent_total"]["bytes_max"], sum(parts), delta=1)
        self.assertEqual(self.d["projected_persistent_bytes"], self.c["persistent_total"]["bytes_max"])

    def test_minimum_safety_reserve_is_ten_gib_and_is_applied(self):
        self.assertEqual(self.d["minimum_safety_reserve_bytes"], 10 * GIB)
        self.assertEqual(self.d["predicted_maximum_occupied_bytes"],
                         self.d["projected_persistent_bytes"] + self.d["temporary_peak_allowance_bytes"])
        self.assertEqual(self.d["required_bytes"],
                         self.d["predicted_maximum_occupied_bytes"] + self.d["minimum_safety_reserve_bytes"])

    def test_gate_decision_follows_from_the_numbers(self):
        expect = "PASS" if self.d["required_bytes"] <= self.d["free_bytes_before"] else "BLOCKED_BY_STORAGE_CAPACITY"
        self.assertEqual(self.d["gate"], expect)
        self.assertEqual(self.d["shortfall_bytes"], max(0, self.d["required_bytes"] - self.d["free_bytes_before"]))
        if self.d["gate"] != "PASS":
            self.assertGreater(self.d["shortfall_bytes"], 0)

    def test_gate_uses_conservative_per_sample_sizes(self):
        b = self.rep["measurement_basis"]
        self.assertGreaterEqual(b["parsing_logits_compressed_bytes"]["max"],
                                b["parsing_logits_compressed_bytes"]["mean"])
        self.assertEqual(self.c["geometry_parsing_logits"]["bytes_max"],
                         self.rep["total_samples"] * b["parsing_logits_compressed_bytes"]["max"])
        self.assertGreaterEqual(self.d["projected_persistent_bytes"],
                                self.d["projected_persistent_bytes_mean_basis"])

    def test_preconditions_recorded_and_passing(self):
        pre = self.rep["preconditions"]
        self.assertEqual(pre["model_hashes"], "PASS")
        self.assertEqual(pre["m1_integrity"], "PASS")
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]
        for key in ("scrfd", "facexformer", "adaface_ir50"):
            self.assertEqual(pre["frozen_models"][key]["expected_sha256"], reg[key]["weight_sha256"], key)
            self.assertTrue(pre["frozen_models"][key]["match"], key)
        for rel, v in pre["m1_frozen_inputs"].items():
            self.assertTrue(v["match"], rel)

    def test_output_roots_resolve_to_one_recorded_filesystem(self):
        fs = self.rep["filesystems"]
        for role in ("project_root", "data_processed", "cache"):
            self.assertEqual(fs[role]["device"], self.d["target_filesystem"]["device"], role)
            for field in ("realpath", "mountpoint", "fstype", "total_bytes", "used_bytes", "free_bytes"):
                self.assertIn(field, fs[role])
        self.assertTrue(self.d["all_outputs_on_one_filesystem"])


class TestBlockedGateLeavesNoOutputs(unittest.TestCase):
    """While the gate blocks, M2B must not have partially processed anything."""

    def test_no_m2_outputs_were_produced(self):
        rep = json.loads(PREFLIGHT_JSON.read_text())
        if rep["decision"]["gate"] == "PASS":
            self.skipTest("preflight passed; completeness is covered by the M2B run tests")
        for rel in ("data/processed", "cache"):
            stray = [p.relative_to(ROOT).as_posix() for p in (ROOT / rel).rglob("*")
                     if p.is_file() and p.name != ".gitkeep"]
            self.assertEqual(stray, [], f"{rel} must stay empty while M2B is blocked")


class TestBoundedShardConstruction(unittest.TestCase):
    """No second raw-logit cache may be persisted: the writer streams one compressed block at a time."""

    def setUp(self):
        self.arr = (np.random.default_rng(5).standard_normal(LS.PARSING_LOGITS_SHAPE) * 8).astype(np.float32)

    def test_shard_grows_by_exactly_one_compressed_block_per_sample(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            w = LS.ShardWriter(d, prefix="parsing_logits", rows_per_shard=4)
            sizes = []
            for i in range(3):
                row = w.add(f"s{i}", (self.arr + i).astype(np.float32))
                sizes.append(row["byte_length"])
                on_disk = (d / row["shard"]).stat().st_size
                self.assertEqual(on_disk, sum(sizes))                   # bounded: nothing else accumulates
                self.assertLess(row["byte_length"], self.arr.nbytes)    # compressed, never raw
            w.close()
            # the only persisted artifacts are the shard files themselves
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["parsing_logits-00000.bin"])

    def test_no_raw_npy_sidecar_is_left_behind(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            w = LS.ShardWriter(d, prefix="parsing_logits", rows_per_shard=2)
            for i in range(2):
                w.add(f"s{i}", self.arr)
            w.close()
            self.assertEqual([p.suffix for p in d.iterdir()], [".bin"] * 1)


class TestShardIndexIntegrityAudit(unittest.TestCase):
    """The reusable core of the M2B cache-integrity audit."""

    @staticmethod
    def _build(d: Path, n=5, rows_per_shard=2):
        rng = np.random.default_rng(6)
        w = LS.ShardWriter(d, prefix="parsing_logits", rows_per_shard=rows_per_shard)
        arrs = {}
        for i in range(n):
            a = (rng.standard_normal(LS.PARSING_LOGITS_SHAPE) * 4).astype(np.float32)
            arrs[f"s{i}"] = a
            w.add(f"s{i}", a)
        w.close()
        return w.index, arrs

    def test_clean_index_passes_every_check(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, _ = self._build(d)
            rep = LS.verify_shard_index(idx, d)
            self.assertTrue(rep["ok"], rep)
            self.assertEqual(rep["rows"], rep["unique_sample_ids"])

    def test_duplicate_sample_id_is_detected(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, _ = self._build(d)
            idx.append(dict(idx[0]))
            rep = LS.verify_shard_index(idx, d, check_blocks=False)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["duplicate_sample_ids"], ["s0"])

    def test_orphan_shard_file_is_detected(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, _ = self._build(d)
            (d / "parsing_logits-09999.bin").write_bytes(b"orphan")
            rep = LS.verify_shard_index(idx, d, check_blocks=False)
            self.assertFalse(rep["ok"])
            self.assertIn("parsing_logits-09999.bin", rep["orphan_shard_files"])

    def test_missing_shard_and_orphan_bytes_are_detected(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, _ = self._build(d)
            os.remove(d / idx[0]["shard"])
            self.assertFalse(LS.verify_shard_index(idx, d, check_blocks=False)["ok"])
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, _ = self._build(d)
            with open(d / idx[-1]["shard"], "ab") as f:
                f.write(b"\0" * 32)                       # bytes no index row accounts for
            rep = LS.verify_shard_index(idx, d, check_blocks=False)
            self.assertFalse(rep["ok"])
            self.assertTrue(any("orphan bytes" in m for m in rep["missing_or_short_shards"]), rep)

    def test_corrupted_block_is_detected(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, _ = self._build(d)
            p = d / idx[0]["shard"]
            b = bytearray(p.read_bytes())
            b[idx[0]["byte_offset"] + 40] ^= 0xFF
            p.write_bytes(bytes(b))
            rep = LS.verify_shard_index(idx, d)
            self.assertFalse(rep["ok"])
            self.assertIn("s0", rep["bad_block_hashes"] + rep["bad_shape_or_dtype"])

    def test_decoded_blocks_keep_dtype_shape_and_mask_consistency(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            idx, arrs = self._build(d, n=3)
            for r in idx:
                a = LS.read_block(d / r["shard"], r["byte_offset"], r["byte_length"])
                self.assertEqual(a.dtype, np.float32)
                self.assertEqual(a.shape, LS.PARSING_LOGITS_SHAPE)
                self.assertTrue(np.array_equal(a, arrs[r["sample_id"]]))
                self.assertTrue(np.all(np.isfinite(a)))
                np.testing.assert_array_equal(LS.mask_from_logits(a), a.argmax(axis=0).astype(np.uint8))

    def test_smoke_shards_pass_the_same_audit(self):
        d = ROOT / "outputs/exploratory/m2a_smoke/runC/logit_shards"
        if not (d / "index.csv").is_file():
            self.skipTest("diagnostic smoke shards are git-ignored and absent")
        rows = []
        for r in csv.DictReader(open(d / "index.csv", newline="", encoding="utf-8")):
            r["byte_offset"], r["byte_length"] = int(r["byte_offset"]), int(r["byte_length"])
            r["shape"] = json.loads(r["shape"])
            rows.append(r)
        rep = LS.verify_shard_index(rows, d)
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["rows"], 24)


if __name__ == "__main__":
    unittest.main()
