"""E01/E02 bank-writer SMOKE mode (M8): approved-contract authorization, canonical global positions (E01 operator =
position % 8), O_NOATIME reads, separate smoke roots, verification/replay with global positions, and the production path
unchanged. Synthetic fixtures only.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402
from methods.common.bank_fixture import build_fixture, build_smoke_fixture  # noqa: E402
from methods.common.config import load_method_config  # noqa: E402

POS = [3, 10, 17, 18, 29, 40, 41, 66, 99, 120, 125, 131]


def rd(rt, m, s=42):
    return Path(rt) / "runs" / "m6" / m / f"seed_{s}"


class BankSmokeMode(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        for p in self.tmp.rglob("*"):
            if p.is_file():
                os.chmod(p, 0o644)
        self._td.cleanup()

    def test_01_authorization_requires_approved_contract(self):
        a = bw.smoke_authorization()
        self.assertEqual((a.mode, a.e01_round_robin_ordinal), ("SMOKE", "MANIFEST_POSITION_ZERO_BASED"))
        for p in (ROOT / "configs/amendments/m8_smoke_test_contract.FINAL_DRAFT.yaml",
                  ROOT / "configs/amendments/m8_smoke_test_contract.DRAFT.yaml", self.tmp / "missing.yaml"):
            with self.assertRaises(bw.BankBlocked):
                bw.smoke_authorization(p)
        bad = self.tmp / "m8_smoke_test_contract.yaml"
        c = json.loads((ROOT / bw.SMOKE_CONTRACT_PATH).read_text())
        bad.write_text(json.dumps({**c, "approved": False}))
        with self.assertRaises(bw.BankBlocked):
            bw.smoke_authorization(bad)

    def test_02_global_positions_drive_e01_operator(self):
        inp = build_smoke_fixture(self.tmp / "faces", POS)
        s = bw.write_bank("E01", 42, inp, bw.smoke_authorization(), self.tmp / "smoke")
        self.assertEqual((s["completion_status"], s["output_count"]), ("completed", len(POS)))
        rows = bw.read_manifest(rd(self.tmp / "smoke", "E01"))
        ops = load_method_config("E01")["generation"]["operator_set"]
        self.assertEqual([r["position"] for r in rows], POS)
        self.assertEqual([r["selected_operator_name"] for r in rows], [ops[p % 8] for p in POS])
        self.assertTrue(bw.verify_bank_files(rd(self.tmp / "smoke", "E01"), "E01", global_positions=True)["ok"])
        self.assertFalse(bw.verify_bank_files(rd(self.tmp / "smoke", "E01"), "E01")["ok"])   # 0..N-1 rule still enforced
        self.assertTrue(bw.replay_bank(rd(self.tmp / "smoke", "E01"), "E01", 42, inp, bw.smoke_authorization())["ok"])

    def test_03_e02_smoke_equals_qualification_pixels_per_pair(self):
        # same synthetic faces, same pair ids -> E02 output identical whether written in SMOKE or QUALIFICATION mode
        q = build_fixture(self.tmp / "qfaces", 4)
        sm = build_smoke_fixture(self.tmp / "sfaces", [0, 1, 2, 3])
        bw.write_bank("E02", 1337, q, bw.qualification_authorization("u"), self.tmp / "q")
        bw.write_bank("E02", 1337, sm, bw.smoke_authorization(), self.tmp / "s")
        a = [r["output_pixel_sha256"] for r in bw.read_manifest(rd(self.tmp / "q", "E02", 1337))]
        b = [r["output_pixel_sha256"] for r in bw.read_manifest(rd(self.tmp / "s", "E02", 1337))]
        self.assertEqual(a, b)

    def test_04_noatime_reads(self):
        inp = build_smoke_fixture(self.tmp / "faces", POS[:4])
        old = time.time() - 5 * 86400
        files = [inp.face_path(s) for s in inp.face_sha256]
        for f in files:
            os.utime(f, (old, f.stat().st_mtime))
        before = {f: f.stat().st_atime_ns for f in files}
        bw.write_bank("E02", 42, inp, bw.smoke_authorization(), self.tmp / "smoke")
        self.assertEqual({f: f.stat().st_atime_ns for f in files}, before)

    def test_05_root_separation_and_mode_mixing_refused(self):
        inp = build_smoke_fixture(self.tmp / "faces", POS[:3])
        with self.assertRaises(bw.BankError):
            bw.write_bank("E01", 42, inp, bw.smoke_authorization(), self.tmp / "faces" / "inside")
        rt = self.tmp / "rt_with_faces"
        (rt / "data" / "processed" / "faces_256").mkdir(parents=True)
        with self.assertRaises(bw.BankError):
            bw.write_bank("E01", 42, inp, bw.smoke_authorization(), rt)
        with self.assertRaises(bw.BankError):          # contiguous qualification fixture is not a smoke population
            bw.write_bank("E01", 42, build_fixture(self.tmp / "q", 3), bw.smoke_authorization(), self.tmp / "x")
        with self.assertRaises(bw.BankError):          # smoke fixture refused under QUALIFICATION authorization
            bw.write_bank("E01", 42, inp, bw.qualification_authorization("u"), self.tmp / "y")
        with self.assertRaises(bw.BankError):
            build_smoke_fixture(self.tmp / "faces2", [5, 3, 9])
        toy = dict(inp.provenance, kind="SMOKE_TOY_POPULATION")
        fake = bw.BankInputs(pairs=inp.pairs, faces_root=inp.faces_root, face_sha256=inp.face_sha256,
                             sample_split={k: "TRAIN" for k in inp.face_sha256}, sample_dataset=inp.sample_dataset,
                             provenance=toy)
        with self.assertRaises(bw.BankError):          # toy population requires the verified gpat-m5 lock
            bw.write_bank("E01", 42, fake, bw.smoke_authorization(), self.tmp / "z")

    def test_06_production_cli_never_smoke(self):
        import tools.e01_e02_bank as cli
        src = Path(cli.__file__).read_text()
        for token in ("smoke_authorization", "smoke_toy_inputs", '"SMOKE"', "SMOKE_TOY_POPULATION"):
            self.assertNotIn(token, src)
        with tempfile.TemporaryDirectory() as td:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["--method", "E01", "--seed", "42", "--runtime-root", td])
            self.assertEqual(rc, cli.EXIT_BLOCKED)
            self.assertEqual(os.listdir(td), [])


if __name__ == "__main__":
    unittest.main()
