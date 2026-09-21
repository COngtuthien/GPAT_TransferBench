"""M6C1 — common M6 runtime: frozen-config loading, run dirs, logging, resume safety."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common.config import (  # noqa: E402
    FrozenConfigError, METHOD_FILES, load_logging_contract, load_method_config, sha256_bytes,
)
from methods.common.runlog import (  # noqa: E402
    RunContext, RunDirectoryError, atomic_write_json, compute_run_id,
)
from methods.common.seeding import seed_everything  # noqa: E402


class TestFrozenConfigLoading(unittest.TestCase):
    def test_all_seven_load_and_match_their_snapshot(self):
        for mid in METHOD_FILES:
            cfg = load_method_config(mid)
            self.assertEqual(cfg["method_id"], mid)
            self.assertTrue(cfg["_runtime"]["snapshot_verified"])
            snap = ROOT / cfg["_runtime"]["snapshot_path"]
            live = ROOT / cfg["_runtime"]["config_path"]
            self.assertEqual(live.read_bytes(), snap.read_bytes(), mid)
            self.assertEqual(cfg["_runtime"]["config_sha256"], sha256_bytes(live.read_bytes()))

    def test_mismatched_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tampered = Path(d) / "e01_fas_aug.yaml"
            src = ROOT / "configs/methods/e01_fas_aug.yaml"
            tampered.write_bytes(src.read_bytes() + b"\n# tampered\n")
            with self.assertRaises(FrozenConfigError) as cm:
                load_method_config("E01", config_path=tampered)
            self.assertIn("differ from the frozen snapshot", str(cm.exception))

    def test_unknown_method_rejected(self):
        with self.assertRaises(FrozenConfigError):
            load_method_config("E99")

    def test_missing_required_field_is_refused_not_repaired(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            cfg = yaml.safe_load((ROOT / "configs/methods/e01_fas_aug.yaml").read_text())
            cfg.pop("checkpoint")
            p = Path(d) / "e01_fas_aug.yaml"
            body = yaml.safe_dump(cfg, sort_keys=False).encode()
            p.write_bytes(body)
            snap = Path(d) / "snap.yaml"
            snap.write_bytes(body)                       # snapshot matches: isolate the field check
            with self.assertRaises(FrozenConfigError) as cm:
                load_method_config("E01", config_path=p, snapshot_path=snap)
            self.assertIn("checkpoint", str(cm.exception))

    def test_test_split_firewall_enforced(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            cfg = yaml.safe_load((ROOT / "configs/methods/e02_freqsub.yaml").read_text())
            cfg["data"]["splits"]["TEST"]["allowed"] = True        # a TEST path is offered
            body = yaml.safe_dump(cfg, sort_keys=False).encode()
            p, snap = Path(d) / "c.yaml", Path(d) / "s.yaml"
            p.write_bytes(body); snap.write_bytes(body)
            with self.assertRaises(FrozenConfigError) as cm:
                load_method_config("E02", config_path=p, snapshot_path=snap)
            self.assertIn("TEST", str(cm.exception))

    def test_every_config_declares_the_three_seeds(self):
        for mid in METHOD_FILES:
            self.assertEqual(load_method_config(mid)["seeds"]["experiment_seeds"], [42, 1337, 2026])

    def test_logging_contract_matches_snapshot(self):
        self.assertEqual(load_logging_contract()["version"], "run_logging_v1")


class TestSeeding(unittest.TestCase):
    def test_seed_everything_is_reproducible(self):
        import random
        import numpy as np
        seed_everything(42); a = (random.random(), float(np.random.rand()))
        seed_everything(42); b = (random.random(), float(np.random.rand()))
        self.assertEqual(a, b)

    def test_rejects_bad_seed(self):
        with self.assertRaises(TypeError):
            seed_everything(True)
        with self.assertRaises(ValueError):
            seed_everything(-1)


class TestRunContext(unittest.TestCase):
    def setUp(self):
        self.cfg = load_method_config("E01")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _ctx(self, **kw):
        return RunContext(method_id="E01", seed=42, config=self.cfg,
                          runtime_root=self.root, command_line="unittest", **kw)

    def test_run_id_is_deterministic(self):
        a = compute_run_id("E01", 42, "a" * 64, "b" * 40)
        b = compute_run_id("E01", 42, "a" * 64, "b" * 40)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)
        self.assertNotEqual(a, compute_run_id("E01", 1337, "a" * 64, "b" * 40))
        self.assertNotEqual(a, compute_run_id("E02", 42, "a" * 64, "b" * 40))

    def test_layout_matches_frozen_contract(self):
        ctx = self._ctx().open()
        self.assertEqual(ctx.run_dir, self.root / "runs/m6/E01/seed_42")
        ctx.close()
        for f in ("resolved_config.yaml", "run_manifest.json", "metrics.jsonl",
                  "checkpoint_index.json", "run_summary.json"):
            self.assertTrue((ctx.run_dir / f).is_file(), f)
        self.assertTrue((ctx.run_dir / "checkpoints").is_dir())

    def test_resolved_config_has_no_scientific_override(self):
        ctx = self._ctx().open(); ctx.close()
        import yaml
        rc = yaml.safe_load(ctx.path("resolved_config").read_text())
        self.assertFalse(rc["_resolved"]["scientific_override_applied"])
        self.assertEqual(rc["_resolved"]["config_sha256"], self.cfg["_runtime"]["config_sha256"])
        self.assertEqual(rc["generation"], self.cfg["generation"])

    def test_manifest_has_required_identity_fields(self):
        ctx = self._ctx().open(); ctx.close()
        m = json.loads(ctx.path("run_manifest").read_text())
        for k in ("method_id", "experiment_seed", "run_id", "config_path", "config_sha256",
                  "git_commit", "source_commits", "start_utc", "end_utc", "command_line",
                  "environment", "completion_status"):
            self.assertIn(k, m)
        self.assertEqual(m["completion_status"], "completed")
        self.assertFalse(m["test_split_accessed"])

    def test_jsonl_append_and_record_type(self):
        ctx = self._ctx().open()
        ctx.log_event("a", {"i": 1})
        ctx.log_generation({"pair_id": "PTR000001", "generation_status": "success"})
        ctx.close()
        lines = [json.loads(l) for l in ctx.path("metrics").read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual([l["record_type"] for l in lines], ["event", "generation"])

    def test_existing_run_is_not_silently_overwritten(self):
        ctx = self._ctx().open(); ctx.log_event("first"); ctx.close()
        with self.assertRaises(RunDirectoryError):
            self._ctx().open()

    def test_resume_appends_and_preserves_history(self):
        ctx = self._ctx().open(); ctx.log_event("first"); ctx.close()
        before = ctx.path("metrics").read_text()
        ctx2 = self._ctx(resume=True).open(); ctx2.log_event("second"); ctx2.close()
        after = ctx2.path("metrics").read_text()
        self.assertTrue(after.startswith(before), "resume must never truncate earlier history")
        kinds = [json.loads(l).get("event") for l in after.splitlines()]
        self.assertIn("resume_reconciliation", kinds)
        self.assertIn("first", kinds)
        self.assertIn("second", kinds)

    def test_resume_refuses_a_different_config(self):
        ctx = self._ctx().open(); ctx.close()
        other = dict(self.cfg)
        other["_runtime"] = dict(self.cfg["_runtime"], config_sha256="0" * 64)
        with self.assertRaises(RunDirectoryError):
            RunContext(method_id="E01", seed=42, config=other, runtime_root=self.root,
                       resume=True).open()

    def test_atomic_write_leaves_no_temp_file(self):
        p = self.root / "x" / "a.json"
        atomic_write_json(p, {"k": 1})
        self.assertEqual(json.loads(p.read_text())["k"], 1)
        self.assertEqual([q.name for q in p.parent.iterdir() if q.name.startswith(".tmp-")], [])

    def test_failure_path_records_status(self):
        try:
            with self._ctx() as ctx:
                raise ValueError("boom")
        except ValueError:
            pass
        s = json.loads(ctx.path("run_summary").read_text())
        self.assertEqual(s["completion_status"], "failed")
        self.assertIn("boom", s["failure_reason"])

    def test_checkpoint_index_helper(self):
        ctx = self._ctx().open()
        ctx.record_checkpoint(path="c.pt", epoch=1, global_step=10, file_size_bytes=3,
                              sha256="d" * 64, checkpoint_type="terminal",
                              selected_for_final=True, selection_reason="frozen rule")
        ctx.close()
        idx = json.loads(ctx.path("checkpoint_index").read_text())
        self.assertEqual(len(idx["checkpoints"]), 1)
        self.assertTrue(idx["checkpoints"][0]["selected_for_final"])
        with self.assertRaises(ValueError):
            ctx.record_checkpoint(path="x", epoch=1, global_step=1, file_size_bytes=1,
                                  sha256="e" * 64, checkpoint_type="bogus",
                                  selected_for_final=False, selection_reason="")



class TestRuntimeRepairs(unittest.TestCase):
    setUp = TestRunContext.setUp
    tearDown = TestRunContext.tearDown
    _ctx = TestRunContext._ctx
    def test_resume_refuses_missing_manifest(self):
        ctx = self._ctx().open(); ctx.close()
        ctx.path("run_manifest").unlink()
        with self.assertRaisesRegex(RunDirectoryError, "incomplete"):
            self._ctx(resume=True).open()

    def test_resume_refuses_changed_identity_and_partial_jsonl(self):
        ctx = self._ctx().open(); ctx.close()
        manifest = json.loads(ctx.path("run_manifest").read_text())
        original = dict(manifest)
        for key, value in (("git_commit", "0" * 40), ("experiment_seed", 1337), ("method_id", "E02")):
            manifest = dict(original, **{key: value})
            atomic_write_json(ctx.path("run_manifest"), manifest)
            with self.assertRaisesRegex(RunDirectoryError, "identity"):
                self._ctx(resume=True).open()
        atomic_write_json(ctx.path("run_manifest"), original)
        ctx.path("metrics").write_text('{"partial":')
        with self.assertRaisesRegex(RunDirectoryError, "partial"):
            self._ctx(resume=True).open()

    def test_manifest_contract_and_close_idempotence(self):
        ctx = self._ctx().open()
        first = ctx.close(summary={"determinism": True})
        self.assertEqual(first, ctx.close())
        manifest = json.loads(ctx.path("run_manifest").read_text())
        self.assertTrue(set(load_logging_contract()["run_identity"]["required_fields"]) <= set(manifest))
        import uuid
        self.assertEqual(uuid.UUID(manifest["run_uuid"]).version, 4)
        resumed = self._ctx(resume=True).open(); resumed.close()
        self.assertEqual(json.loads(ctx.path("run_manifest").read_text())["run_uuid"], manifest["run_uuid"])

    def test_generation_contract_and_no_pixel_payload(self):
        import numpy as np
        ctx = self._ctx().open()
        ctx.log_generation({"pair_id": "PTR000001", "generation_status": "success"})
        with self.assertRaises((TypeError, ValueError)):
            ctx.log_event("payload", {"array": np.zeros((256, 256, 3))})
        with self.assertRaises(ValueError):
            ctx.log_event("payload", {"pixels": [[[0]]]})
        with self.assertRaises(ValueError):
            ctx.log_epoch({})
        ctx.close()
        record = json.loads(ctx.path("generation").read_text())
        contract = load_logging_contract()["non_learned_methods"]
        self.assertTrue(set(contract["required_fields"] + contract["e01_specific_required"]) <= set(record))
        self.assertIn("output_sha256", record["missing_field_reasons"])

    def test_invalid_runtime_method_seed_and_mutated_config(self):
        for method, seed in (("E02", 42), ("E01", 3), ("E01", 42.0)):
            with self.assertRaises(RunDirectoryError):
                RunContext(method_id=method, seed=seed, config=self.cfg, runtime_root=self.root)

    def test_active_run_cannot_be_opened_by_another_writer(self):
        ctx = self._ctx().open()
        try:
            with self.assertRaises(BlockingIOError):
                self._ctx(resume=True).open()
        finally:
            ctx.close()



class TestGenerationFirewall(unittest.TestCase):
    def test_test_split_and_nonfrozen_seed_refused(self):
        import numpy as np
        from methods.fas_aug.e01 import E01Generator, ToyOperatorBackend
        from methods.freq_sub.e02 import E02Generator
        img = np.zeros((256, 256, 3), np.uint8)
        e01 = E01Generator(backend=ToyOperatorBackend(), synthetic_only=True).prepare()
        e02 = E02Generator().prepare()
        for gen, images in ((e01, (img,)), (e02, (img, img))):
            for pair, seed, split in (("PTR000001", 42, "TEST"), ("PTE000001", 42, "TRAIN"), ("PTR000001", 7, "TRAIN")):
                result = gen.generate_one(pair, seed, *images, split=split)
                self.assertFalse(result.success)
                self.assertIsNone(result.output)

    def test_mutating_both_config_copies_cannot_bypass_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            original = (ROOT / "configs/methods/e01_fas_aug.yaml").read_bytes()
            a, b = Path(directory) / "config.yaml", Path(directory) / "snapshot.yaml"
            a.write_bytes(original + b"\n# changed\n")
            b.write_bytes(a.read_bytes())
            with self.assertRaisesRegex(FrozenConfigError, "SHA256"):
                load_method_config("E01", config_path=a, snapshot_path=b)


if __name__ == "__main__":
    unittest.main()
