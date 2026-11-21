"""E01/E02 bank writer: authorization gate, population integrity, determinism, lineage, collisions, failures, TEST firewall.

Synthetic fixtures only (methods/common/bank_fixture.py). The one production-population test reads the frozen
manifests as metadata against an EMPTY temporary faces root: no benchmark image is read.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402
from methods.common.bank_fixture import build_fixture  # noqa: E402
from methods.common.config import load_method_config  # noqa: E402
from methods.common.runlog import RunDirectoryError  # noqa: E402
from methods.freq_sub import e02  # noqa: E402

N = 16
QA = bw.qualification_authorization("unittest")


def run_dir(rt, method, seed=42):
    return Path(rt) / "runs" / "m6" / method / f"seed_{seed}"


def files_under(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.inputs = build_fixture(self.tmp / "faces", N)

    def tearDown(self):
        for p in self.tmp.rglob("*"):
            if p.is_file():
                os.chmod(p, 0o644)
        self._td.cleanup()


class TestAuthorizationGate(unittest.TestCase):
    def ruling(self, td, **over):
        doc = {"schema_version": bw.RULING_SCHEMA, "status": "OWNER_APPROVED", "approved": True,
               "decisions": {"Q1_milestone_owner": "B_M6_EXECUTION_EXTENSION",
                             "Q2_e01_round_robin_ordinal": "MANIFEST_POSITION_ZERO_BASED",
                             "Q3_e02_uint8_quantization": "CONFIRM_CLIP_THEN_RINT"}}
        for k, v in over.items():
            if k.startswith("Q"):
                doc["decisions"][k] = v
            else:
                doc[k] = v
        p = Path(td) / "e01_e02_bank_milestone_ruling.yaml"
        p.write_text(json.dumps(doc))
        return p

    def assertBlocked(self, method, path=None):
        with self.assertRaises(bw.BankBlocked) as cm:
            bw.load_production_authorization(method, path)
        self.assertIn(bw.BLOCKED_STATUS, str(cm.exception))

    def test_01_repository_has_no_approved_ruling(self):
        self.assertFalse((ROOT / bw.RULING_PATH).exists())
        for m in bw.METHODS:
            self.assertBlocked(m)

    def test_02_draft_is_never_authority(self):
        draft = ROOT / "configs/amendments/e01_e02_bank_milestone_ruling.DRAFT.yaml"
        self.assertTrue(draft.is_file())
        self.assertFalse(json.loads(draft.read_text())["approved"])
        self.assertBlocked("E01", draft)

    def test_03_incomplete_rulings_refused(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertBlocked("E01", self.ruling(td, approved=False))
            self.assertBlocked("E01", self.ruling(td, status="DRAFT_PENDING_OWNER_DECISION"))
            self.assertBlocked("E01", self.ruling(td, Q1_milestone_owner=None))
            self.assertBlocked("E01", self.ruling(td, Q2_e01_round_robin_ordinal=None))
            self.assertBlocked("E02", self.ruling(td, Q3_e02_uint8_quantization="OTHER"))
            self.assertBlocked("E01", self.ruling(td, Q1_milestone_owner="A_M8_SMOKE_GATED"))  # no smoke record

    def test_04_complete_ruling_yields_production_authorization(self):
        with tempfile.TemporaryDirectory() as td:
            a = bw.load_production_authorization("E01", self.ruling(td))
            self.assertEqual((a.mode, a.e01_round_robin_ordinal), ("PRODUCTION", "MANIFEST_POSITION_ZERO_BASED"))
            a2 = bw.load_production_authorization("E02", self.ruling(td, Q2_e01_round_robin_ordinal=None))
            self.assertIsNone(a2.e01_round_robin_ordinal)

    def test_05_cli_exits_blocked_without_touching_runtime_root(self):
        import tools.e01_e02_bank as cli
        with tempfile.TemporaryDirectory() as td:
            for m in bw.METHODS:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = cli.main(["--method", m, "--seed", "42", "--runtime-root", td])
                self.assertEqual(rc, cli.EXIT_BLOCKED)
                self.assertEqual(json.loads(out.getvalue())["status"], bw.BLOCKED_STATUS)
            self.assertEqual(os.listdir(td), [])

    def test_06_locked_environment_check(self):
        live = bw.environment_versions()
        if live["numpy"] == "2.4.6" and live["python_version"] == "3.11.16":
            self.assertEqual(bw.verify_locked_environment()["numpy"], "2.4.6")
        else:
            with self.assertRaises(bw.BankError):
                bw.verify_locked_environment()


class TestProductionPopulation(unittest.TestCase):
    """Frozen manifests as metadata; the faces root is an empty temp dir, so no image can be read."""

    def test_07_frozen_train_population(self):
        with tempfile.TemporaryDirectory() as td:
            faces = Path(td).resolve() / "data" / "processed" / "faces_256"
            faces.mkdir(parents=True)
            for m in bw.METHODS:
                inp = bw.production_inputs(Path(td).resolve(), load_method_config(m))
                self.assertEqual(len(inp.pairs), 8838)
                self.assertEqual((inp.pairs[0].pair_id, inp.pairs[-1].pair_id), ("PTR000001", "PTR008838"))
                self.assertEqual(inp.provenance["referenced_samples"], 12668)
                with self.assertRaises(bw.BankError):
                    inp.read(inp.pairs[0].target_live_id)
            self.assertEqual(files_under(td), [])

    def test_08_production_mode_refuses_fixture_inputs_and_unverified_env(self):
        with tempfile.TemporaryDirectory() as td:
            inp = build_fixture(Path(td) / "faces", 2)
            prod = bw.Authorization("PRODUCTION", "MANIFEST_POSITION_ZERO_BASED", "B_M6_EXECUTION_EXTENSION",
                                    None, None, "production")
            with self.assertRaises(bw.BankError):
                bw.write_bank("E01", 42, inp, prod, Path(td) / "rt")
            with self.assertRaises(bw.BankError):
                bw.write_bank("E01", 42, inp, prod, Path(td) / "rt", environment_lock={"verified": True})
            self.assertFalse((Path(td) / "rt").exists())


class TestE01Bank(_Tmp):
    def test_09_bank_counts_lineage_and_round_robin(self):
        s = bw.write_bank("E01", 42, self.inputs, QA, self.tmp / "rt")
        self.assertEqual((s["completion_status"], s["input_count"], s["output_count"], s["failure_count"]),
                         ("completed", N, N, 0))
        rows = bw.read_manifest(run_dir(self.tmp / "rt", "E01"))
        ops = load_method_config("E01")["generation"]["operator_set"]
        for r, p in zip(rows, self.inputs.pairs):
            self.assertEqual(r["pair_id"], p.pair_id)
            self.assertEqual(r["selected_operator_name"], ops[p.position % 8])
            self.assertEqual(r["target_face_sha256"], self.inputs.face_sha256[p.target_live_id])
            self.assertFalse(r["source_spoof_used"])
            self.assertIsNone(r["source_face_sha256"])
            self.assertEqual(r["synthetic_output_path"], f"bank/{p.pair_id}.png")
            self.assertIsNone(r["generator_checkpoint_sha256"])
        self.assertEqual(rows[0]["deterministic_pair_seed"], "795981663")   # spec/A2-05 worked example
        self.assertTrue(bw.verify_bank_files(run_dir(self.tmp / "rt", "E01"), "E01")["ok"])
        self.assertTrue(bw.replay_bank(run_dir(self.tmp / "rt", "E01"), "E01", 42, self.inputs, QA)["ok"])

    def test_10_pair_id_numeric_rule_shifts_operator(self):
        qa = bw.qualification_authorization("unittest", "PAIR_ID_NUMERIC")
        bw.write_bank("E01", 42, self.inputs, qa, self.tmp / "rt")
        rows = bw.read_manifest(run_dir(self.tmp / "rt", "E01"))
        ops = load_method_config("E01")["generation"]["operator_set"]
        self.assertEqual([r["selected_operator_name"] for r in rows], [ops[(i + 1) % 8] for i in range(N)])

    def test_11_output_equals_direct_generate_one(self):
        bw.write_bank("E01", 1337, self.inputs, QA, self.tmp / "rt")
        rd = run_dir(self.tmp / "rt", "E01", 1337)
        gen = bw.make_generator("E01", load_method_config("E01"))
        for r, p in zip(bw.read_manifest(rd), self.inputs.pairs):
            target, _ = self.inputs.read(p.target_live_id)
            res = gen.generate_one(p.pair_id, 1337, target, index=p.position)
            self.assertTrue(np.array_equal(bw.decode_png((rd / r["synthetic_output_path"]).read_bytes()), res.output))
            if r["is_level_zero_noop"]:
                self.assertTrue(np.array_equal(res.output, target))

    def test_12_determinism_across_roots(self):
        a = bw.write_bank("E01", 2026, self.inputs, QA, self.tmp / "rt_a")
        b = bw.write_bank("E01", 2026, self.inputs, QA, self.tmp / "rt_b")
        ra, rb = (bw.read_manifest(run_dir(self.tmp / r, "E01", 2026)) for r in ("rt_a", "rt_b"))
        self.assertEqual(ra, rb)
        self.assertEqual(a["bank_manifest_sha256"], b["bank_manifest_sha256"])
        self.assertEqual(a["run_id"], b["run_id"])

    def test_13_seeds_give_different_banks(self):
        hashes = {}
        for seed in (42, 1337, 2026):
            bw.write_bank("E01", seed, self.inputs, QA, self.tmp / "rt")
            hashes[seed] = [r["output_pixel_sha256"] for r in bw.read_manifest(run_dir(self.tmp / "rt", "E01", seed))]
        self.assertNotEqual(hashes[42], hashes[1337])
        self.assertNotEqual(hashes[1337], hashes[2026])


class TestE02Bank(_Tmp):
    def test_14_bank_edit_maps_and_numeric_invariants(self):
        s = bw.write_bank("E02", 42, self.inputs, QA, self.tmp / "rt")
        self.assertEqual((s["output_count"], s["failure_count"], s["k_values"], s["n_eligible_values"]), (N, 0, [40], [159]))
        self.assertLessEqual(s["max_abs_imag"], 1e-8)
        rd = run_dir(self.tmp / "rt", "E02")
        gen = bw.make_generator("E02", load_method_config("E02"))
        for r, p in zip(bw.read_manifest(rd), self.inputs.pairs):
            sel = gen.selection(p.pair_id, 42)
            self.assertEqual(json.loads(r["selected_block_indices_sorted_ascending"]), sel["selected_indices"])
            t, _ = self.inputs.read(p.target_live_id)
            src, ssha = self.inputs.read(p.source_spoof_id)
            self.assertEqual(r["source_face_sha256"], ssha)
            self.assertTrue(r["source_spoof_used"])
            img, diag = e02.substitute(t, src, np.asarray(sel["selected_blocks_rc"]))
            self.assertTrue(np.array_equal(bw.decode_png((rd / r["synthetic_output_path"]).read_bytes()), img))
            em = bw.decode_png((rd / r["edit_map_path"]).read_bytes())
            self.assertTrue(np.array_equal(em > 0, diag["edit_mask"]))
            self.assertTrue(np.array_equal(em > 0, e02.conjugate_mask(em > 0)))   # Hermitian-closed mask
        self.assertEqual(bw.read_manifest(rd)[0]["deterministic_pair_seed"],
                         "19354000309184887456743517179028239418116648094833629759175866349341962868907")
        self.assertTrue(bw.verify_bank_files(rd, "E02")["ok"])
        self.assertTrue(bw.replay_bank(rd, "E02", 42, self.inputs, QA)["ok"])

    def test_15_determinism_across_roots(self):
        a = bw.write_bank("E02", 1337, self.inputs, QA, self.tmp / "rt_a")
        b = bw.write_bank("E02", 1337, self.inputs, QA, self.tmp / "rt_b")
        self.assertEqual(a["bank_manifest_sha256"], b["bank_manifest_sha256"])


class TestSafety(_Tmp):
    def test_16_second_run_same_root_refused_and_outputs_unchanged(self):
        bw.write_bank("E02", 42, self.inputs, QA, self.tmp / "rt")
        rd = run_dir(self.tmp / "rt", "E02")
        before = {f: (rd / f).read_bytes() for f in files_under(rd)}
        with self.assertRaises(RunDirectoryError):
            bw.write_bank("E02", 42, self.inputs, QA, self.tmp / "rt")
        self.assertEqual({f: (rd / f).read_bytes() for f in files_under(rd)}, before)

    def test_17_exclusive_writes(self):
        p = self.tmp / "x.png"
        bw.write_exclusive(p, b"a")
        with self.assertRaises(FileExistsError):
            bw.write_exclusive(p, b"b")
        self.assertEqual(p.read_bytes(), b"a")
        m = self.tmp / bw.BANK_MANIFEST
        m.write_bytes(b"old")
        with self.assertRaises(bw.BankError):
            bw._write_manifest(m, [])
        self.assertEqual(m.read_bytes(), b"old")

    def test_18_input_corruption_aborts_run(self):
        victim = self.inputs.pairs[5].target_live_id
        p = self.inputs.face_path(victim)
        p.write_bytes(p.read_bytes() + b"\0")
        with self.assertRaises(bw.BankError):
            bw.write_bank("E01", 42, self.inputs, QA, self.tmp / "rt")
        rd = run_dir(self.tmp / "rt", "E01")
        summary = json.loads((rd / "run_summary.json").read_text())
        self.assertEqual(summary["completion_status"], "failed")
        self.assertEqual(summary["rows_before_abort"], 5)
        self.assertFalse((rd / bw.BANK_MANIFEST).exists())

    def test_19_method_failure_is_recorded_not_substituted(self):
        from methods.freq_sub.e02 import E02Generator
        real = E02Generator.generate_one

        def flaky(self, pair_id, *a, **k):
            if pair_id == "PTR000003":
                return real(self, pair_id, 7, *a[1:], **k)      # invalid seed -> generator-level failure
            return real(self, pair_id, *a, **k)
        with mock.patch.object(E02Generator, "generate_one", flaky):
            s = bw.write_bank("E02", 42, self.inputs, QA, self.tmp / "rt")
        self.assertEqual((s["completion_status"], s["output_count"], s["failure_count"]),
                         ("completed_with_failures", N - 1, 1))
        self.assertEqual(s["failures"][0]["pair_id"], "PTR000003")
        rd = run_dir(self.tmp / "rt", "E02")
        self.assertFalse((rd / "bank" / "PTR000003.png").exists())
        self.assertTrue(bw.verify_bank_files(rd, "E02")["ok"])

    def test_20_tampered_output_detected(self):
        bw.write_bank("E01", 42, self.inputs, QA, self.tmp / "rt")
        rd = run_dir(self.tmp / "rt", "E01")
        f = rd / "bank" / "PTR000002.png"
        os.chmod(f, 0o644)
        f.write_bytes(bw.encode_png(np.zeros((256, 256, 3), np.uint8), "RGB"))
        v = bw.verify_bank_files(rd, "E01")
        self.assertFalse(v["ok"])
        (rd / "bank" / "PTR999999.png").write_bytes(b"x")
        self.assertTrue(any("extra" in p for p in bw.verify_bank_files(rd, "E01")["problems"]))

    def test_21_writes_confined_to_run_dir(self):
        rt = self.tmp / "rt"
        bw.write_bank("E01", 42, self.inputs, QA, rt)
        bw.write_bank("E02", 42, self.inputs, QA, rt)
        for f in files_under(rt):
            self.assertRegex(f, r"^runs/m6/E0[12]/(seed_42/|\.seed_42\.lock$)")
        self.assertEqual(sorted(os.listdir(run_dir(rt, "E01"))),
                         sorted(["bank", "bank_manifest.parquet", "bank_manifest.sha256", "checkpoint_index.json",
                                 "checkpoints", "generation_log.jsonl", "metrics.jsonl", "resolved_config.yaml",
                                 "run_manifest.json", "run_summary.json", "stderr.log", "stdout.log"]))


class TestFirewall(_Tmp):
    def fixture_with(self, **override):
        p = self.inputs.pairs
        kw = dict(pairs=p, faces_root=self.inputs.faces_root, face_sha256=self.inputs.face_sha256,
                  sample_split={s: "TRAIN" for s in self.inputs.face_sha256}, sample_dataset=self.inputs.sample_dataset,
                  provenance=self.inputs.provenance)
        kw.update(override)
        return bw.BankInputs(**kw)

    def test_22_test_and_val_pairs_refused(self):
        p = list(self.inputs.pairs)
        for bad in ("PTE000001", "PVA000001", "TEST01"):
            p0 = bw.PairRow(0, bad, p[0].target_live_id, p[0].source_spoof_id, p[0].dataset)
            with self.assertRaises(bw.BankError):
                self.fixture_with(pairs=[p0])

    def test_23_non_train_sample_refused(self):
        split = {s: "TRAIN" for s in self.inputs.face_sha256}
        split[self.inputs.pairs[0].source_spoof_id] = "TEST"
        with self.assertRaises(bw.BankError):
            self.fixture_with(sample_split=split)

    def test_24_population_order_and_duplicates(self):
        p = list(self.inputs.pairs)
        with self.assertRaises(bw.BankError):
            self.fixture_with(pairs=[p[1], p[0]])
        with self.assertRaises(bw.BankError):
            self.fixture_with(pairs=[p[0], bw.PairRow(1, p[0].pair_id, p[1].target_live_id, p[1].source_spoof_id, p[1].dataset)])

    def test_25_qualification_refuses_runtime_root_with_benchmark_faces(self):
        rt = self.tmp / "rt"
        (rt / "data" / "processed" / "faces_256").mkdir(parents=True)
        with self.assertRaises(bw.BankError):
            bw.write_bank("E01", 42, self.inputs, QA, rt)
        self.assertFalse((rt / "runs").exists())

    def test_26_unread_samples_unreachable(self):
        with self.assertRaises(bw.BankError):
            self.inputs.read("0" * 64)

    def test_27_non_frozen_seed_refused(self):
        with self.assertRaises(RunDirectoryError):
            bw.write_bank("E01", 7, self.inputs, QA, self.tmp / "rt")


if __name__ == "__main__":
    unittest.main()
