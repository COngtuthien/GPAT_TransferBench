"""M2A tests: preprocessing contracts, adapters, smoke evidence, no full-M2/M3 artifacts.

Requires the M2 environment (torch, onnxruntime) for the model tests:
    /home/cong/.venvs/gpatbench-m2/bin/python -m unittest discover -s tests
Under an environment without torch/onnxruntime the model-dependent classes are skipped with a reason.
"""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from gpatbench.preprocess import contracts as C  # noqa: E402
from gpatbench.preprocess.frames import FrameReadError, read_video_frame  # noqa: E402

AUDIT = ROOT / "outputs/audit"
CACHE = Path("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache")
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("onnxruntime") is not None
HAS_WEIGHTS = (CACHE / "face_geometry/ckpts/model.pt").exists()


class TestRoutingAndConstants(unittest.TestCase):
    def test_01_routing(self):
        self.assertFalse(C.scrfd_required("casia_fasd"))
        self.assertTrue(C.scrfd_required("msu_mfsd") and C.scrfd_required("siwmv2"))
        self.assertEqual(C.route("casia_fasd"), "PRECROPPED_112_RGB_TO_256_INTER_CUBIC")

    def test_02_03_threshold_and_input(self):
        self.assertEqual((C.SCRFD_THRESHOLD, C.SCRFD_INPUT, C.CROP_SCALE, C.CANONICAL_SIZE), (0.50, 320, 1.25, 256))


class TestFaceSelection(unittest.TestCase):
    def test_04_largest_face(self):
        d = np.array([[0, 0, 10, 10, 0.99], [5, 5, 45, 45, 0.6], [0, 0, 20, 20, 0.9]], np.float32)
        self.assertEqual(C.select_largest_face(d), 1)

    def test_05_equal_area_tie(self):
        d = np.array([[10, 10, 30, 30, 0.7], [0, 0, 20, 20, 0.9], [50, 0, 70, 20, 0.9]], np.float32)
        self.assertEqual(C.select_largest_face(d), 1)          # higher score, then smaller x1
        d2 = np.array([[50, 0, 70, 20, 0.9], [0, 0, 20, 20, 0.9]], np.float32)
        self.assertEqual(C.select_largest_face(d2), 1)         # same score -> smaller x1

    def test_15_no_fallback(self):
        self.assertEqual(C.detection_outcome(np.empty((0, 5), np.float32)), ("SCRFD_NO_FACE", None))
        self.assertEqual(C.detection_outcome(np.array([[0, 0, 5, 5, 0.9]], np.float32))[0], "DETECTED")


class TestCrop(unittest.TestCase):
    def test_06_square_geometry(self):
        b = C.square_crop_box([100, 100, 180, 140], 1000, 1000)   # w=80, h=40 -> side 100, centre (140,120)
        self.assertEqual(b.requested, (90, 70, 190, 170))
        self.assertFalse(b.was_clamped)
        self.assertAlmostEqual(b.side_float, 100.0)

    def test_07_boundary_clamp(self):
        b = C.square_crop_box([0, 0, 40, 40], 100, 100)           # side 50 centred at (20,20)
        self.assertEqual(b.requested, (-5, -5, 45, 45))
        self.assertEqual(b.clamped, (0, 0, 45, 45))
        self.assertTrue(b.was_clamped)
        b2 = C.square_crop_box([60, 60, 100, 100], 100, 100)
        self.assertEqual(b2.clamped[2:], (100, 100))

    def test_08_no_random_padding(self):
        boxes = {C.square_crop_box([10.3, 20.7, 61.1, 90.9], 64, 64) for _ in range(20)}
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1 = next(iter(boxes)).clamped
        self.assertTrue(0 <= x0 < x1 <= 64 and 0 <= y0 < y1 <= 64)

    def test_09_10_interpolation(self):
        self.assertEqual(C.resize_interpolation(400, 400), cv2.INTER_AREA)
        self.assertEqual(C.resize_interpolation(100, 100), cv2.INTER_CUBIC)

    def test_11_canonical_output(self):
        rgb = C.to_canonical(np.random.default_rng(0).integers(0, 256, (300, 280, 3), dtype=np.uint8))
        C.check_canonical(rgb)
        self.assertEqual((rgb.shape, rgb.dtype), ((256, 256, 3), np.uint8))

    def test_12_casia_exact(self):
        bgr = np.random.default_rng(1).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        exp = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
        np.testing.assert_array_equal(C.casia_to_canonical(bgr), exp)
        with self.assertRaises(ValueError):
            C.casia_to_canonical(np.zeros((100, 112, 3), np.uint8))

    def test_png_lossless_deterministic(self):
        rgb = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        a, b = C.encode_png_rgb(rgb), C.encode_png_rgb(rgb)
        self.assertEqual(a, b)
        dec = cv2.imdecode(np.frombuffer(a, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        np.testing.assert_array_equal(dec, rgb)


class TestFrames(unittest.TestCase):
    def test_14_frame_index_is_m1_index(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "v.avi"
            w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            for i in range(12):
                w.write(np.full((32, 32, 3), i * 20, np.uint8))
            w.release()
            for i in (0, 5, 11):
                self.assertAlmostEqual(float(read_video_frame(p, i).mean()), i * 20, delta=3)
            with self.assertRaises(FrameReadError):
                read_video_frame(p, 12)
        man = list(csv.DictReader(open(AUDIT / "M2A_SMOKE_MANIFEST.csv")))
        import pyarrow.parquet as pq
        inv = {r["sample_id"]: r for r in pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist()}
        for m in man:
            self.assertEqual(int(m["frame_index"]), inv[m["sample_id"]]["frame_index"])


class TestSmokeEvidence(unittest.TestCase):
    def test_13_casia_no_scrfd_metric(self):
        res = list(csv.DictReader(open(AUDIT / "M2A_SMOKE_RESULTS.csv")))
        for r in res:
            if r["dataset"] == "casia_fasd":
                self.assertEqual((r["scrfd_applied"], r["scrfd_status"]), ("False", "N/A"))
        self.assertNotIn("casia_fasd", json.loads((AUDIT / "M2A_SMOKE_SUMMARY.json").read_text())["scrfd_success_by_dataset"])

    def test_21_manifest_deterministic(self):
        import m2a_smoke
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.csv"
            m2a_smoke.build_manifest(p)
            self.assertEqual(p.read_bytes(), (AUDIT / "M2A_SMOKE_MANIFEST.csv").read_bytes())
        rows = list(csv.DictReader(open(AUDIT / "M2A_SMOKE_MANIFEST.csv")))
        self.assertEqual(len(rows), 24)
        for ds in ("casia_fasd", "msu_mfsd", "siwmv2"):
            sub = [r for r in rows if r["dataset"] == ds]
            self.assertEqual({r["label_binary"] for r in sub}, {"0", "1"})
            self.assertEqual(len({r["video_id"] for r in sub}), len(sub))

    def test_22_rerun_deterministic(self):
        d = json.loads((AUDIT / "M2A_DETERMINISM_COMPARE.json").read_text())
        self.assertTrue(d["same_file_set"] and d["results_equal_except_timing"])
        self.assertEqual(d["byte_identical"], d["n_files"])
        self.assertEqual(d["differs"], [])

    def test_19_embedding_norms_from_smoke(self):
        res = list(csv.DictReader(open(AUDIT / "M2A_SMOKE_RESULTS.csv")))
        ok = [r for r in res if r["status"] == "OK"]
        self.assertTrue(all(abs(float(r["embedding_l2"]) - 1.0) < 1e-5 for r in ok))


@unittest.skipUnless(HAS_TORCH and HAS_WEIGHTS, "requires the M2 environment (torch, onnxruntime) and local model weights")
class TestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpatbench.preprocess.aux_models import AdaFaceAdapter, FaceXFormerAdapter, set_deterministic
        set_deterministic(4)
        cls.fx = FaceXFormerAdapter(CACHE / "code/facexformer", CACHE / "face_geometry/ckpts/model.pt")
        cls.ada = AdaFaceAdapter(CACHE / "code/adaface", CACHE / "face_identity/pretrained_model/model.pt", "RGB",
                                 "RESIZE_256_TO_112_INTER_AREA")
        cls.rgb = np.random.default_rng(3).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def test_16_17_facexformer(self):
        x = self.fx.preprocess(self.rgb)
        self.assertEqual(tuple(x.shape), (1, 3, 224, 224))
        g = self.fx(self.rgb)
        self.assertEqual(g["parsing_logits"].shape, (11, 224, 224))
        self.assertEqual(g["parsing_mask"].shape, (224, 224))
        self.assertEqual(g["landmarks_norm"].shape, (68, 2))
        self.assertEqual(g["landmarks_px256"].shape, (68, 2))
        self.assertEqual(g["pose_pitch_yaw_roll_rad"].shape, (3,))
        self.assertTrue(all(np.all(np.isfinite(v)) for v in g.values()))

    def test_18_adaface_input(self):
        x = self.ada.preprocess(self.rgb)
        self.assertEqual(tuple(x.shape), (1, 3, 112, 112))
        self.assertTrue(float(x.min()) >= -1.0 and float(x.max()) <= 1.0)
        small = cv2.resize(self.rgb, (112, 112), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(x[0, 0].numpy(), small[:, :, 0] / 127.5 - 1.0, atol=1e-6)   # channel 0 = R (RGB order)

    def test_19_20_adaface_norm_and_eval(self):
        e = self.ada(self.rgb)["embedding"]
        self.assertEqual((e.shape, e.dtype), ((512,), np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(e)), 1.0, delta=1e-5)
        self.assertFalse(self.ada.model.training or self.fx.model.training)
        self.assertFalse(any(p.requires_grad and p.grad is not None for p in self.ada.model.parameters()))

    def test_scrfd_contract(self):
        from gpatbench.preprocess.scrfd import SCRFD
        det = SCRFD(CACHE / "face_detectors/scrfd_10g_bnkps.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91")
        img = np.zeros((480, 640, 3), np.uint8)
        d, _ = det.detect(img, C.SCRFD_THRESHOLD)
        self.assertEqual(d.shape[1], 5)
        lb, scale, (nw, nh) = det.letterbox(img)
        self.assertEqual((lb.shape, nw, nh), ((320, 320, 3), 320, 240))
        self.assertAlmostEqual(scale, 0.5)


class TestIntegrity(unittest.TestCase):
    def test_23_no_full_m2_outputs(self):
        for rel in ("data/processed", "cache"):
            self.assertEqual([p for p in (ROOT / rel).rglob("*") if p.is_file() and p.name != ".gitkeep"], [], rel)

    def test_24_no_m3_artifacts(self):
        names = {p.name for p in ROOT.rglob("*.parquet") if ".git" not in p.parts}
        self.assertFalse(any("split" in n or "pair" in n for n in names))

    def test_25_no_weights_tracked(self):
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
        self.assertFalse([t for t in tracked if t.endswith((".onnx", ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".npy"))])
        smoke = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "outputs/exploratory/m2a_smoke/x.png"], capture_output=True, text=True)
        self.assertEqual(smoke.returncode, 0)


if __name__ == "__main__":
    unittest.main()
