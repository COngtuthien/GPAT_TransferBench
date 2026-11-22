"""M8 smoke pipeline units that need no torch: D8 TPR@FPR and D10 raw-type table vectors, the O_NOATIME face reader,
the audit-hook firewall (in a subprocess), the real toy manifest (TRAIN/VAL metadata; face existence stubbed), the
synthetic world, and the metrics tables on synthetic prediction files. No benchmark image is opened.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpatbench.downstream import metrics as M  # noqa: E402

GE = "SPOOF_IF_SCORE_GE_THRESHOLD"


class MetricVectors(unittest.TestCase):
    def test_01_tpr_at_fpr(self):
        # y [0,0,1,1], s [.1,.4,.35,.8]: (APCER, BPCER) at candidates 0,.225,.375,.6,1 = (0,1),(0,1/2),(1/2,1/2),(1/2,0),(1,0)
        # BPCER <= 0.05 -> candidates .6 and 1 -> TPR max(1/2, 0) = 1/2; BPCER <= 0.5 also allows .225 -> TPR 1
        self.assertEqual(M.tpr_at_fpr([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], "0.05", rule=GE), Fraction(1, 2))
        self.assertEqual(M.tpr_at_fpr([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], "0.5", rule=GE), 1)
        with self.assertRaises(TypeError):
            M.tpr_at_fpr([0, 1], [0.2, 0.8], "0.01")

    def test_02_d10_raw_type_table(self):
        tok = {"Replay": "replay", "Paper": "print", "Silicone": "mask_3d"}
        vids = [{"dataset": "siwmv2", "video_id": "a", "y_true": 1, "attack_raw": "Replay", "attack_macro": "replay", "video_score": 0.9},
                {"dataset": "siwmv2", "video_id": "b", "y_true": 1, "attack_raw": "Replay", "attack_macro": "replay", "video_score": 0.2},
                {"dataset": "siwmv2", "video_id": "c", "y_true": 1, "attack_raw": "Paper", "attack_macro": "print", "video_score": 0.7},
                {"dataset": "siwmv2", "video_id": "d", "y_true": 0, "attack_raw": None, "attack_macro": "live", "video_score": 0.1},
                {"dataset": "casia_fasd", "video_id": "e", "y_true": 1, "attack_raw": "x", "attack_macro": "print", "video_score": 0.1}]
        t = M.siw_raw_type_table(vids, 0.5, tok, rule=GE)
        rows = {r["attack_raw"]: r for r in t["rows"]}
        self.assertEqual(t["status"], "VALID")
        self.assertEqual(list(rows), ["Paper", "Replay", "Silicone"])                     # all tokens listed
        self.assertEqual((rows["Replay"]["APCER"], rows["Replay"]["numerator"], rows["Replay"]["denominator"]),
                         (Fraction(1, 2), 1, 2))
        self.assertEqual((rows["Paper"]["APCER"], rows["Silicone"]["APCER"], rows["Silicone"]["reason"]),
                         (0, None, "ABSENT_IN_SPLIT"))
        bad = vids + [{"dataset": "siwmv2", "video_id": "f", "y_true": 1, "attack_raw": "Replay", "attack_macro": "print",
                       "video_score": 0.3}]
        t2 = M.siw_raw_type_table(bad, 0.5, tok, rule=GE)
        self.assertEqual((t2["status"], t2["reason"]), ("N/A", "RAW_LABEL_COVERAGE_INVALID"))
        self.assertTrue(all(r["APCER"] is None for r in t2["rows"]))
        self.assertEqual(M.siw_raw_type_table(vids[:2] + [dict(vids[2], attack_raw=None)], 0.5, tok, rule=GE)["status"], "N/A")


class FaceReaderAndFirewall(unittest.TestCase):
    def test_03_face_reader_noatime_and_hash(self):
        from gpatbench.downstream.faceio import FaceReadError, FaceReader
        from methods.common.bank_writer import encode_png
        import hashlib
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "d").mkdir()
            raw = encode_png(np.zeros((256, 256, 3), dtype=np.uint8), "RGB")
            f = Path(td) / "d" / "x.png"
            f.write_bytes(raw)
            old = time.time() - 5 * 86400
            os.utime(f, (old, f.stat().st_mtime))
            a0 = f.stat().st_atime_ns
            r = FaceReader(td)
            im = r.read("d/x.png", hashlib.sha256(raw).hexdigest())
            self.assertEqual((im.mode, im.size), ("RGB", (256, 256)))
            self.assertEqual(f.stat().st_atime_ns, a0)
            self.assertEqual(r.summary(), {"files": 1, "reads": 1, "atime_changed_files": 0})
            with self.assertRaises(FaceReadError):
                r.read("d/x.png", "0" * 64)
            with self.assertRaises(FaceReadError):
                r.read("../x.png", "0" * 64)

    def test_04_audit_hook_firewall(self):
        code = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
from gpatbench.downstream import audit_hook
root = sys.argv[2]
audit_hook.install(root, ["a/ok.png"], os.path.join(sys.argv[3], "fw.json"))
open(os.path.join(root, "a/ok.png"), "rb").read()
out = {}
for name, fn in (("other", lambda: open(os.path.join(root, "a/other.png"), "rb")),
                 ("osopen", lambda: os.open(os.path.join(root, "a/other.png"), os.O_RDONLY)),
                 ("listdir", lambda: os.listdir(os.path.join(root, "a"))),
                 ("scandir", lambda: list(os.scandir(root)))):
    try:
        fn(); out[name] = "allowed"
    except PermissionError:
        out[name] = "denied"
open(os.path.join(sys.argv[3], "outside.txt"), "w").write("x")
r = audit_hook.write_report()
print(json.dumps({"out": out, "violations": r["violation_count"], "distinct": r["distinct_files_opened"]}))
'''
        with tempfile.TemporaryDirectory() as faces, tempfile.TemporaryDirectory() as work:
            (Path(faces) / "a").mkdir()
            for n in ("ok.png", "other.png"):
                (Path(faces) / "a" / n).write_bytes(b"x")
            p = subprocess.run([sys.executable, "-c", code, str(ROOT), faces, work], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertEqual(r["out"], {"other": "denied", "osopen": "denied", "listdir": "denied", "scandir": "denied"})
            self.assertEqual((r["violations"], r["distinct"]), (4, 1))


class ToyManifests(unittest.TestCase):
    def test_05_real_manifest_counts(self):
        from gpatbench.downstream import toy_manifest as TM
        with mock.patch.object(TM.os.path, "isfile", return_value=True):
            man = TM.build_real(ROOT, "/nonexistent/faces_256")
        n = {k: len(v) for k, v in man["roles"].items()}
        self.assertEqual(n, {"train_live": 96, "train_spoof": 256, "val_select_live": 96, "val_select_spoof": 256,
                             "val_holdout_live": 96, "val_holdout_spoof": 263})
        self.assertEqual((len(man["toy_pairs"]), len(man["pair_samples"]), man["test_rows_read"]), (186, 247, 0))
        sel = {f["video_id"] for k in ("val_select_live", "val_select_spoof") for f in man["roles"][k]}
        hold = {f["video_id"] for k in ("val_holdout_live", "val_holdout_spoof") for f in man["roles"][k]}
        self.assertFalse(sel & hold)
        self.assertEqual(len(man["allowlist"]), len(set(man["allowlist"])))
        self.assertTrue(all(len(f["face_sha256"]) == 64 for v in man["roles"].values() for f in v))

    def test_06_synthetic_world(self):
        from gpatbench.downstream import toy_manifest as TM
        with tempfile.TemporaryDirectory() as td:
            a = TM.build_synthetic_world(Path(td) / "a")
            b = TM.build_synthetic_world(Path(td) / "b")
            self.assertEqual(TM.canonical_sha256(a), TM.canonical_sha256(b))       # deterministic world
            self.assertEqual(len(a["roles"]["train_live"]) // 32, 3)
            self.assertLess(len(a["roles"]["train_spoof"]), 3 * 32)                  # E00 must reshuffle
            self.assertLess(len(a["toy_pairs"]), 3 * 16)                             # synthetic pool reshuffles
            pos = [p["position"] for p in a["toy_pairs"]]
            self.assertEqual(pos, sorted(set(pos)))


class Tables(unittest.TestCase):
    def test_07_tables_from_synthetic_predictions(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from gpatbench.downstream import tables as TB
        from gpatbench.downstream.schemas import PREDICTION_COLUMNS
        with tempfile.TemporaryDirectory() as td:
            ev = Path(td)
            for mi, m in enumerate(TB.METHODS):
                (ev / m).mkdir()
                rows = []
                for v, (ds, y, raw, am, ps) in enumerate((("casia_fasd", 0, None, "live", (0.1, 0.2)),
                                                          ("casia_fasd", 1, None, "print", (0.7, 0.9)),
                                                          ("siwmv2", 0, None, "live", (0.3, 0.4)),
                                                          ("siwmv2", 1, "Replay", "replay", (0.6, 0.2 + 0.1 * mi)))):
                    vs = sum(ps) / 2
                    for fi, p in enumerate(ps):
                        rows.append(dict(zip(PREDICTION_COLUMNS, (f"{m}{v}{fi}", ds, None, f"v{v}", fi, y, raw, am, p, 0.5,
                                                                  int(p >= 0.5), vs, int(vs >= 0.5), m, "binary", "resnet18",
                                                                  42, "a" * 64, "b" * 64))))
                pq.write_table(pa.Table.from_pylist(rows), ev / m / "predictions_holdout.parquet")
                (ev / m / "steps.jsonl").write_text(json.dumps({"step": 0, "epoch": 0, "loss_hex": (0.5).hex()}) + "\n")
                (ev / m / "epochs.json").write_text(json.dumps([{"epoch": 0, "val_acer": "1/4", "val_auc": "3/4"}]))
            out = TB.build_tables(ev, ROOT)
            self.assertEqual(len(out), 7)
            t06 = out["T06_smoke_downstream_resnet18_video.csv"].decode().splitlines()
            self.assertIn("SMOKE_NON_SCIENTIFIC", t06[1])
            raw = out["T13b_smoke_siwmv2_raw_type_apcer.csv"].decode().splitlines()
            self.assertEqual(len(raw), 1 + 3 * 14)                                  # 14 tokens always listed
            self.assertEqual(TB.build_tables(ev, ROOT), out)                       # deterministic bytes


if __name__ == "__main__":
    unittest.main()
