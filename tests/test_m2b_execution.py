"""M2B execution tests: external roots, atomic writes, shard finalization, resume and auditing.

These exercise the real engine (`gpatbench.preprocess.m2b` and `tools/m2b_audit.py`) on synthetic
data — no stubs and no always-true assertions. The model-dependent behaviour is covered separately
by the deterministic final validation, which compares re-computed tensors against the artifacts the
full run actually stored.
"""
import csv
import io
import json
import os
import subprocess
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
from gpatbench.preprocess import m2b  # noqa: E402

AUDIT = ROOT / "outputs/audit"
EXEC_CFG = ROOT / "configs/execution/m2b_laptop_external_storage.yaml"


def _logits(seed=0):
    return (np.random.default_rng(seed).standard_normal(LS.PARSING_LOGITS_SHAPE) * 6).astype(np.float32)


def _fake_cfg(runtime_root: Path, **override) -> dict:
    roots = {"processed_frames_root": str(runtime_root / "data/processed/frames"),
             "faces_256_root": str(runtime_root / "data/processed/faces_256"),
             "geometry_cache_root": str(runtime_root / "cache/geometry"),
             "identity_cache_root": str(runtime_root / "cache/identity")}
    roots.update(override)
    return {"storage": {"runtime_root": str(runtime_root), "roots": roots,
                        "runstate_root": str(runtime_root / "runstate"),
                        "tmp_root": str(runtime_root / "tmp")}}


class TestExecutionConfig(unittest.TestCase):
    def test_config_is_infrastructure_only(self):
        x = yaml.safe_load(EXEC_CFG.read_text())
        self.assertEqual(x["kind"], "EXECUTION_INFRASTRUCTURE")
        self.assertEqual(x["scientific_contract"], "configs/frozen/preprocess_v1.yaml")
        self.assertEqual(x["deviation"], "DEV-017")
        # it must not restate or override any scientific field of the frozen contract
        frozen = yaml.safe_load((ROOT / x["scientific_contract"]).read_text())
        scientific = set(frozen) - {"version", "status", "references"}
        self.assertEqual(scientific & set(x), set())
        self.assertIs(x["storage"]["symlinks_used"], False)
        self.assertEqual(x["storage"]["min_free_reserve_bytes"], 10 * 1024 ** 3)
        self.assertEqual(x["execution"]["shard_rows"], LS.SHARD_ROWS)
        self.assertEqual(x["execution"]["torch_threads"], frozen["determinism"]["torch_threads"])

    def test_recorded_contract_hash_matches_the_frozen_file(self):
        x = yaml.safe_load(EXEC_CFG.read_text())
        self.assertEqual(m2b.sha256_file(ROOT / x["scientific_contract"]), x["scientific_contract_sha256"])

    def test_no_project_directory_was_replaced_by_a_symlink(self):
        for rel in ("data", "data/processed", "cache", "outputs", "manifests", "configs"):
            self.assertFalse((ROOT / rel).is_symlink(), rel)


class TestRootContainment(unittest.TestCase):
    def test_approved_roots_resolve_inside_the_runtime_root(self):
        x = yaml.safe_load(EXEC_CFG.read_text())
        res = m2b.resolve_roots(x)
        for k, v in res["roots"].items():
            self.assertTrue(v.resolve().is_relative_to(res["runtime_root"]), k)

    def test_root_outside_the_runtime_root_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            rt = Path(t) / "runtime"
            (rt / "cache").mkdir(parents=True)
            outside = Path(t) / "elsewhere"
            outside.mkdir()
            with self.assertRaises(m2b.RootContainmentError):
                m2b.resolve_roots(_fake_cfg(rt, geometry_cache_root=str(outside)))

    def test_dotdot_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            rt = Path(t) / "runtime"
            (rt / "cache").mkdir(parents=True)
            with self.assertRaises(m2b.RootContainmentError):
                m2b.resolve_roots(_fake_cfg(rt, identity_cache_root=str(rt / ".." / "sneaky")))

    def test_symlink_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            rt = Path(t) / "runtime"
            (rt / "cache").mkdir(parents=True)
            outside = Path(t) / "outside"
            outside.mkdir()
            link = rt / "cache" / "identity"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(m2b.RootContainmentError):
                m2b.resolve_roots(_fake_cfg(rt, identity_cache_root=str(link)))

    def test_field_root_maps_roles_correctly(self):
        roots = {"geometry_cache_root": Path("/g"), "identity_cache_root": Path("/i")}
        self.assertEqual(m2b.field_root(roots, "parsing_logits"), Path("/g/parsing_logits"))
        self.assertEqual(m2b.field_root(roots, "embedding"), Path("/i/embedding"))


class TestAtomicWrites(unittest.TestCase):
    def test_atomic_write_leaves_no_tmp_and_returns_the_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "sub" / "x.png"
            data = os.urandom(4096)
            got = m2b.atomic_write_bytes(p, data)
            self.assertEqual(got, hashlib.sha256(data).hexdigest())
            self.assertEqual(p.read_bytes(), data)
            self.assertEqual(sorted(q.name for q in p.parent.iterdir()), ["x.png"])

    def test_a_partial_final_file_is_never_exposed(self):
        """If the write dies before the rename, the final path must not exist at all."""
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.bin"
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(b"half")                      # simulate a crash before os.replace
            self.assertFalse(p.exists())
            m2b.atomic_write_bytes(p, b"complete-content")
            self.assertEqual(p.read_bytes(), b"complete-content")

    def test_rewrite_replaces_content_atomically(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.bin"
            m2b.atomic_write_bytes(p, b"a" * 100)
            m2b.atomic_write_bytes(p, b"b" * 50)
            self.assertEqual(p.read_bytes(), b"b" * 50)


class TestWorkPlan(unittest.TestCase):
    @staticmethod
    def _rows(n=600):
        def ds(i):
            return "d" if i < n // 2 else "e"
        return [{"sample_id": f"s{i:04d}", "dataset": ds(i),
                 "video_id": f"{ds(i)}/v{i // 8}", "frame_index": i % 8} for i in range(n)]

    def test_plan_is_deterministic_and_order_independent(self):
        import random
        a = m2b.build_plan(self._rows(), 256)
        shuffled = self._rows()
        random.Random(0).shuffle(shuffled)
        b = m2b.build_plan(shuffled, 256)
        self.assertEqual([(r["sample_id"], r["_ordinal"], r["_shard"]) for r in a],
                         [(r["sample_id"], r["_ordinal"], r["_shard"]) for r in b])

    def test_shard_membership_is_a_pure_function_of_the_manifest(self):
        plan = m2b.build_plan(self._rows(), 256)
        self.assertEqual({r["_shard"] for r in plan}, {0, 1, 2})
        for s in (0, 1, 2):
            chunk = m2b.chunk_of(plan, s)
            self.assertLessEqual(len(chunk), 256)
            self.assertTrue(all(r["_ordinal"] // 256 == s for r in chunk))
        self.assertEqual(sum(len(m2b.chunk_of(plan, s)) for s in (0, 1, 2)), len(plan))

    def test_a_videos_samples_stay_together(self):
        plan = m2b.build_plan(self._rows(), 256)
        for v in {(r["dataset"], r["video_id"]) for r in plan}:
            ords = sorted(r["_ordinal"] for r in plan if (r["dataset"], r["video_id"]) == v)
            self.assertEqual(ords, list(range(ords[0], ords[0] + len(ords))))

    def test_real_inventory_plan_covers_every_sample_once(self):
        import pyarrow.parquet as pq
        inv = pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()
        plan = m2b.build_plan(inv, LS.SHARD_ROWS)
        self.assertEqual(len(plan), len(inv))
        self.assertEqual(len({r["sample_id"] for r in plan}), len(inv))
        self.assertEqual([r["_ordinal"] for r in plan], list(range(len(inv))))


class TestStateLog(unittest.TestCase):
    def test_last_record_per_sample_wins_across_worker_logs(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a, b = m2b.StateLog(d / "m2b_state.w0.jsonl"), m2b.StateLog(d / "m2b_state.w1.jsonl")
            a.write(sample_id="s1", state="FACE_DONE")
            a.write(sample_id="s1", state="COMPLETE")
            b.write(sample_id="s2", state="FAILED", failure_reason="SCRFD_NO_FACE")
            a.close()
            b.close()
            st = m2b.StateLog.load(d)
            self.assertEqual(st["s1"]["state"], "COMPLETE")
            self.assertEqual(st["s2"]["failure_reason"], "SCRFD_NO_FACE")

    def test_torn_final_line_after_a_crash_is_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            log = m2b.StateLog(d / "m2b_state.w0.jsonl")
            log.write(sample_id="s1", state="COMPLETE")
            log.close()
            with open(d / "m2b_state.w0.jsonl", "a") as f:
                f.write('{"sample_id": "s2", "state": "COMP')      # truncated by a crash
            st = m2b.StateLog.load(d)
            self.assertEqual(set(st), {"s1"})

    def test_every_recorded_state_is_a_declared_state(self):
        rs = Path(yaml.safe_load(EXEC_CFG.read_text())["storage"]["runstate_root"])
        if not rs.is_dir():
            self.skipTest("no run state on this machine")
        seen = {v.get("state") for v in m2b.StateLog.load(rs).values()}
        self.assertTrue(seen <= set(m2b.STATES), seen - set(m2b.STATES))


class TestShardFinalization(unittest.TestCase):
    def _roots(self, d: Path):
        return {f: d / f for f in ("parsing_logits", "parsing_mask", "embedding")}

    def test_finalize_publishes_shard_and_index_atomically(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cs = m2b.ChunkShardSet(self._roots(d), 3)
            cs.open()
            arrs = {}
            for i in range(3):
                lg = _logits(i)
                arrs[f"s{i}"] = lg
                cs.add("parsing_logits", f"s{i}", lg)
                cs.add("parsing_mask", f"s{i}", LS.mask_from_logits(lg))
                cs.add("embedding", f"s{i}", np.full(512, i, np.float32))
            self.assertTrue((d / "parsing_logits/parsing_logits-00003.bin.tmp").exists())
            out = cs.finalize()
            self.assertFalse((d / "parsing_logits/parsing_logits-00003.bin.tmp").exists())
            meta = json.loads((d / "parsing_logits/parsing_logits-00003.bin.index.json").read_text())
            self.assertEqual(meta["shard_sha256"], out["parsing_logits"]["sha256"])
            for r in meta["rows"]:
                got = LS.read_block(d / "parsing_logits" / r["shard"], r["byte_offset"], r["byte_length"])
                self.assertTrue(np.array_equal(got, arrs[r["sample_id"]]))

    def test_abort_removes_the_in_flight_shard(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cs = m2b.ChunkShardSet(self._roots(d), 0)
            cs.open()
            cs.add("parsing_logits", "s0", _logits())
            cs.abort()
            self.assertEqual(list((d / "parsing_logits").iterdir()), [])
            self.assertFalse(cs.is_finalized())

    def test_finalize_refuses_a_corrupted_in_flight_shard(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cs = m2b.ChunkShardSet(self._roots(d), 0)
            cs.open()
            cs.add("parsing_logits", "s0", _logits())
            cs.add("parsing_mask", "s0", np.zeros((224, 224), np.uint8))
            cs.add("embedding", "s0", np.zeros(512, np.float32))
            cs._fh["parsing_logits"].flush()
            p = d / "parsing_logits/parsing_logits-00000.bin.tmp"
            b = bytearray(p.read_bytes())
            b[len(b) // 2] ^= 0xFF
            p.write_bytes(bytes(b))
            with self.assertRaises(RuntimeError):
                cs.finalize()
            self.assertFalse((d / "parsing_logits/parsing_logits-00000.bin").exists())   # never published

    def test_is_finalized_requires_both_shard_and_index(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cs = m2b.ChunkShardSet(self._roots(d), 0)
            cs.open()
            for f, arr in (("parsing_logits", _logits()), ("parsing_mask", np.zeros((224, 224), np.uint8)),
                           ("embedding", np.zeros(512, np.float32))):
                cs.add(f, "s0", arr)
            cs.finalize()
            self.assertTrue(cs.is_finalized())
            (d / "embedding/embedding-00000.bin.index.json").unlink()
            self.assertFalse(m2b.ChunkShardSet(self._roots(d), 0).is_finalized())


class TestCacheAudit(unittest.TestCase):
    """The audit must actually detect the faults it claims to check."""

    def _build(self, d: Path, n=4):
        roots = {f: d / f for f in ("parsing_logits", "embedding")}
        cs = m2b.ChunkShardSet(roots, 0)
        cs.open()
        for i in range(n):
            cs.add("parsing_logits", f"s{i}", _logits(i))
            cs.add("embedding", f"s{i}", np.full(512, i, np.float32))
        cs.finalize()
        return roots

    def test_clean_cache_passes(self):
        import m2b_audit
        with tempfile.TemporaryDirectory() as t:
            roots = self._build(Path(t))
            rep, _ = m2b_audit.audit_field(roots["parsing_logits"], "parsing_logits", True)
            self.assertTrue(rep["ok"], rep)
            self.assertEqual(rep["entries"], 4)

    def test_duplicate_entry_is_detected(self):
        import m2b_audit
        with tempfile.TemporaryDirectory() as t:
            roots = self._build(Path(t))
            idxp = roots["parsing_logits"] / "parsing_logits-00000.bin.index.json"
            meta = json.loads(idxp.read_text())
            meta["rows"].append(dict(meta["rows"][0]))
            idxp.write_text(json.dumps(meta))
            rep, _ = m2b_audit.audit_field(roots["parsing_logits"], "parsing_logits", False)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["duplicate_sample_ids"], ["s0"])

    def test_orphan_shard_file_is_detected(self):
        import m2b_audit
        with tempfile.TemporaryDirectory() as t:
            roots = self._build(Path(t))
            (roots["embedding"] / "embedding-09999.bin").write_bytes(b"orphan")
            rep, _ = m2b_audit.audit_field(roots["embedding"], "embedding", False)
            self.assertFalse(rep["ok"])
            self.assertIn("embedding-09999.bin", rep["orphan_shard_files"])

    def test_corrupted_block_and_shard_hash_are_detected(self):
        import m2b_audit
        with tempfile.TemporaryDirectory() as t:
            roots = self._build(Path(t))
            p = roots["parsing_logits"] / "parsing_logits-00000.bin"
            b = bytearray(p.read_bytes())
            b[len(b) // 2] ^= 0xFF
            p.write_bytes(bytes(b))
            rep, _ = m2b_audit.audit_field(roots["parsing_logits"], "parsing_logits", True)
            self.assertFalse(rep["ok"])
            self.assertTrue(rep["shard_hash_failures"])
            self.assertTrue(rep["bad_block_hashes"])

    def test_truncated_shard_is_detected(self):
        import m2b_audit
        with tempfile.TemporaryDirectory() as t:
            roots = self._build(Path(t))
            p = roots["embedding"] / "embedding-00000.bin"
            p.write_bytes(p.read_bytes()[:-100])
            rep, _ = m2b_audit.audit_field(roots["embedding"], "embedding", False)
            self.assertFalse(rep["ok"])
            self.assertTrue(rep["missing_shard_files"])


class TestFinalRunEvidence(unittest.TestCase):
    """Assertions over the artifacts of the actual full run, when it has produced them."""

    @classmethod
    def setUpClass(cls):
        cls.integrity = json.loads((AUDIT / "M2_CACHE_INTEGRITY.json").read_text()) \
            if (AUDIT / "M2_CACHE_INTEGRITY.json").is_file() else None

    def setUp(self):
        if self.integrity is None:
            self.skipTest("the full M2 run has not produced its audit yet")

    def test_every_frozen_sample_is_accounted_for(self):
        import pyarrow.parquet as pq
        acc = self.integrity["accounting"]
        n = pq.read_table(ROOT / "manifests/inventory.parquet").num_rows
        self.assertEqual(acc["expected"], n)
        self.assertEqual(acc["missing_state"], 0)
        self.assertEqual(acc["complete"] + acc["failed"], acc["expected"])
        acct = pq.read_table(ROOT / "manifests/m2_sample_accounting.parquet").to_pylist()
        self.assertEqual(len(acct), n)
        self.assertEqual(len({r["sample_id"] for r in acct}), n)
        self.assertTrue(all(r["final_status"] in ("COMPLETE", "FAILED") for r in acct))
        # every COMPLETE row carries the provenance the contract requires
        for r in acct:
            if r["final_status"] != "COMPLETE":
                continue
            for k in ("source_file_sha256", "face_png_sha256", "face_png_path", "route", "shard"):
                self.assertTrue(r[k], f"{r['sample_id']}: empty {k}")
            if r["dataset"] != "casia_fasd":
                self.assertTrue(r["frame_png_sha256"] and r["frame_png_path"])

    def test_shard_hash_summary_matches_the_cache(self):
        p = AUDIT / "M2_CACHE_SHARD_INDEX.csv"
        rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        self.assertTrue(rows)
        by_field = {}
        for r in rows:
            by_field.setdefault(r["field"], []).append(r)
        for f, rep in self.integrity["fields"].items():
            self.assertEqual(len(by_field[f]), rep["shards"], f)
            self.assertEqual(sum(int(r["rows"]) for r in by_field[f]), rep["entries"], f)
            self.assertEqual(sum(int(r["bytes"]) for r in by_field[f]), rep["bytes"], f)
            self.assertTrue(all(len(r["shard_sha256"]) == 64 for r in by_field[f]), f)

    def test_no_duplicate_or_orphan_cache_entries(self):
        self.assertEqual(self.integrity["n_orphan_cache_rows"], 0)
        self.assertEqual(self.integrity["n_complete_samples_missing_cache_entry"], 0)
        self.assertEqual(self.integrity["n_missing_face_files"], 0)
        for f, rep in self.integrity["fields"].items():
            self.assertEqual(rep["duplicate_sample_ids"], [], f)
            self.assertEqual(rep["orphan_shard_files"], [], f)
            self.assertEqual(rep["overlapping_blocks"], [], f)
            self.assertTrue(rep["ok"], f)

    def test_every_complete_sample_has_exactly_one_entry_per_field(self):
        acc = self.integrity["accounting"]
        for f, rep in self.integrity["fields"].items():
            self.assertEqual(rep["entries"], acc["complete"], f)

    def test_casia_scrfd_is_not_counted_as_success(self):
        self.assertEqual(self.integrity["casia_scrfd"], "N/A")
        self.assertNotIn("casia_fasd", self.integrity["scrfd"])
        rows = list(csv.DictReader(open(AUDIT / "M2_SCRFD_SUCCESS_REPORT.csv", newline="", encoding="utf-8")))
        casia = [r for r in rows if r["dataset"] == "casia_fasd"]
        self.assertEqual(len(casia), 1)
        self.assertEqual(casia[0]["success"], "N/A")

    def test_deterministic_validation_passed(self):
        p = AUDIT / "M2_DETERMINISTIC_VALIDATION.json"
        if not p.is_file():
            self.skipTest("validation not run yet")
        v = json.loads(p.read_text())
        self.assertEqual(v["status"], "PASS")
        self.assertTrue(v["all_equal"])
        self.assertEqual(v["mismatches"], [])
        self.assertFalse(v["selection"]["hand_picked"])
        self.assertGreaterEqual(v["samples_compared"], 24)

    def test_external_smoke_equivalence_recorded(self):
        v = json.loads((AUDIT / "M2B_EXTERNAL_SMOKE_COMPARE.json").read_text())
        self.assertEqual(v["status"], "PASS")
        self.assertEqual(v["byte_identical"], v["n_files"])
        self.assertIs(v["manifest_reselected"], False)
        self.assertEqual(v["manifest_sha256"],
                         "138f5929f81313069e4a4cb74c4ed54e4957432fe9ebc93cc52fdd14100dbce7")


if __name__ == "__main__":
    unittest.main()
