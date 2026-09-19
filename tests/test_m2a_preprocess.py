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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from gpatbench.preprocess import contracts as C  # noqa: E402
from gpatbench.preprocess.frames import FrameReadError, read_video_frame  # noqa: E402

AUDIT = ROOT / "outputs/audit"
CACHE = Path("/media/cong/Data/AI on IOT/Anti_spoofing/model_cache")
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("onnxruntime") is not None
SCRFD_MODEL = CACHE / "face_detectors/scrfd_10g_bnkps.onnx"
SCRFD_NOT_SELECTED = CACHE / "face_detectors/det_2.5g.onnx"
ADAFACE_CKPT = CACHE / "face_identity/adaface_original/adaface_ir50_webface4m.ckpt"
ADAFACE_CVLFACE = CACHE / "face_identity/pretrained_model/model.pt"
HAS_WEIGHTS = (CACHE / "face_geometry/ckpts/model.pt").exists() and ADAFACE_CKPT.exists()
SCRFD_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
ADAFACE_SHA256 = "52cca7c64808fea6f44f9b9aee2b0e091bf96c1ab4f6e31bedcdf5d77009b4f8"
CVLFACE_SHA256 = "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"
FROZEN_CFG = ROOT / "configs/frozen/preprocess_v1.yaml"


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

    def test_07_boundary_source_region(self):
        b = C.square_crop_box([0, 0, 40, 40], 100, 100)           # side 50 centred at (20,20)
        self.assertEqual(b.requested, (-5, -5, 45, 45))
        self.assertEqual(b.source, (0, 0, 45, 45))                # only the SOURCE READ region is clamped
        self.assertEqual((b.pad_left, b.pad_top, b.pad_right, b.pad_bottom), (5, 5, 0, 0))
        self.assertTrue(b.touches_border)
        b2 = C.square_crop_box([60, 60, 100, 100], 100, 100)
        self.assertEqual(b2.source[2:], (100, 100))

    def test_08_no_random_padding(self):
        boxes = {C.square_crop_box([10.3, 20.7, 61.1, 90.9], 64, 64) for _ in range(20)}
        self.assertEqual(len(boxes), 1)
        x0, y0, x1, y1 = next(iter(boxes)).source
        self.assertTrue(0 <= x0 < x1 <= 64 and 0 <= y0 < y1 <= 64)

    def test_09_10_interpolation(self):
        self.assertEqual(C.resize_interpolation(400, 400), cv2.INTER_AREA)
        self.assertEqual(C.resize_interpolation(100, 100), cv2.INTER_CUBIC)

    def test_11_canonical_output(self):
        # after Q-22 the crop handed to to_canonical is always square (extract_square_crop)
        for side in (300, 180):
            rgb = C.to_canonical(np.random.default_rng(0).integers(0, 256, (side, side, 3), dtype=np.uint8))
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
        cls.ada = AdaFaceAdapter(CACHE / "code/adaface", ADAFACE_CKPT)
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
        self.assertEqual(x.dtype.__str__(), "torch.float32")
        self.assertTrue(float(x.min()) >= -1.0 and float(x.max()) <= 1.0)
        small = cv2.resize(self.rgb, (112, 112), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(x[0, 0].numpy(), small[:, :, 2] / 127.5 - 1.0, atol=1e-6)   # channel 0 = B (Q-18 BGR)

    def test_19_20_adaface_norm_and_eval(self):
        e = self.ada(self.rgb)["embedding"]
        self.assertEqual((e.shape, e.dtype), ((512,), np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(e)), 1.0, delta=1e-5)
        self.assertFalse(self.ada.model.training or self.fx.model.training)
        self.assertFalse(any(p.requires_grad and p.grad is not None for p in self.ada.model.parameters()))

    def test_scrfd_contract(self):
        from gpatbench.preprocess.scrfd import SCRFD
        det = SCRFD(SCRFD_MODEL, "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91")
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



# ====================================================================== owner resolutions (2026-09-19)
class TestQ22SquareZeroPadding(unittest.TestCase):
    """Q-22 RESOLVED_BY_OWNER: the requested 1.25x square is preserved with deterministic zero fill."""

    @staticmethod
    def _img(h=100, w=120):
        return np.arange(1, h * w * 3 + 1, dtype=np.int64).reshape(h, w, 3).astype(np.uint8) | 1   # never 0

    def _check(self, bbox, w, h, expect_pads):
        img = self._img(h, w)
        box = C.square_crop_box(bbox, w, h)
        crop = C.extract_square_crop(img, box)
        self.assertEqual(crop.shape, (box.side_px, box.side_px, 3))           # a true square, never shrunk
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertEqual(box.requested[2] - box.requested[0], box.side_px)
        self.assertEqual(box.requested[3] - box.requested[1], box.side_px)
        self.assertEqual((box.pad_left, box.pad_top, box.pad_right, box.pad_bottom), expect_pads)
        sx0, sy0, sx1, sy1 = box.source
        # the copied region is the intersection, placed at the pad offset, unmodified
        np.testing.assert_array_equal(crop[box.pad_top:box.pad_top + (sy1 - sy0),
                                           box.pad_left:box.pad_left + (sx1 - sx0)], img[sy0:sy1, sx0:sx1])
        # every pixel outside the intersection is constant zero (no reflect/replicate/random padding)
        mask = np.ones(crop.shape[:2], bool)
        mask[box.pad_top:box.pad_top + (sy1 - sy0), box.pad_left:box.pad_left + (sx1 - sx0)] = False
        self.assertTrue(np.all(crop[mask] == 0))
        self.assertEqual(int(mask.sum()), box.side_px ** 2 - (sy1 - sy0) * (sx1 - sx0))
        return box, crop

    def test_q22_no_boundary_contact(self):
        box, _ = self._check([40, 30, 80, 70], 120, 100, (0, 0, 0, 0))
        self.assertFalse(box.touches_border)

    def test_q22_left_border(self):
        box, _ = self._check([0, 40, 20, 60], 120, 100, (2, 0, 0, 0))
        self.assertTrue(box.touches_border and box.pad_right == 0)

    def test_q22_right_border(self):
        box, _ = self._check([100, 40, 120, 60], 120, 100, (0, 0, 3, 0))
        self.assertTrue(box.touches_border and box.pad_left == 0)

    def test_q22_top_border(self):
        box, _ = self._check([50, 0, 70, 20], 120, 100, (0, 2, 0, 0))
        self.assertTrue(box.touches_border and box.pad_bottom == 0)

    def test_q22_bottom_border(self):
        box, _ = self._check([50, 80, 70, 100], 120, 100, (0, 0, 0, 3))
        self.assertTrue(box.touches_border and box.pad_top == 0)

    def test_q22_corner_crossing(self):
        box, _ = self._check([0, 0, 20, 20], 120, 100, (2, 2, 0, 0))
        self.assertTrue(box.pad_left and box.pad_top)
        box2, _ = self._check([100, 80, 120, 100], 120, 100, (0, 0, 3, 3))
        self.assertTrue(box2.pad_right and box2.pad_bottom)

    def test_q22_square_larger_than_one_source_dimension(self):
        # side = 1.25*160 = 200 > image height 100: padding on both top and bottom at once
        box, crop = self._check([0, 10, 160, 90], 120, 100, (20, 50, 60, 50))
        self.assertGreater(box.side_px, 100)
        self.assertEqual(box.source[1:4:2], (0, 100))
        self.assertEqual(crop.shape[0], crop.shape[1])

    def test_q22_never_shifted_or_shrunk(self):
        for bbox in ([0, 0, 20, 20], [110, 90, 120, 100], [40, 30, 80, 70]):
            box = C.square_crop_box(bbox, 120, 100)
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            self.assertAlmostEqual((box.requested[0] + box.requested[2]) / 2.0, cx, delta=0.5)
            self.assertAlmostEqual((box.requested[1] + box.requested[3]) / 2.0, cy, delta=0.5)
            self.assertEqual(box.side_px, int(round(1.25 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))))

    def test_q22_deterministic(self):
        img = self._img()
        crops = {C.extract_square_crop(img, C.square_crop_box([0, 0, 21, 33], 120, 100)).tobytes() for _ in range(10)}
        self.assertEqual(len(crops), 1)

    def test_q22_disjoint_square_is_an_error(self):
        with self.assertRaises(C.CropError):
            C.square_crop_box([500, 500, 520, 520], 120, 100)

    def test_q22_canonical_resize_requires_square(self):
        with self.assertRaises(ValueError):
            C.to_canonical(np.zeros((300, 280, 3), np.uint8))

    def test_q22_frozen_config(self):
        cfg = yaml.safe_load(FROZEN_CFG.read_text())
        self.assertEqual(cfg["crop"]["border"], "REQUESTED_SQUARE_ZERO_PAD")
        self.assertIs(cfg["crop"]["random_padding"], False)
        self.assertEqual(cfg["owner_decisions"]["Q-22"], "RESOLVED_BY_OWNER")


class TestQ02ScrfdSelection(unittest.TestCase):
    def test_active_registry_key_is_10g(self):
        from gpatbench.preprocess import scrfd as S
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]["scrfd"]
        cfg = yaml.safe_load(FROZEN_CFG.read_text())["scrfd"]
        self.assertEqual(reg["variant"], "SCRFD_10G_KPS")
        self.assertEqual(reg["weight_sha256"], SCRFD_SHA256)
        self.assertEqual(reg["status"], "VERIFIED")
        self.assertEqual(cfg["variant"], "SCRFD_10G_KPS")
        self.assertEqual(cfg["weight_sha256"], SCRFD_SHA256)
        self.assertEqual(cfg["model_file"], "scrfd_10g_bnkps.onnx")
        self.assertEqual(S.SELECTED_VARIANT, "SCRFD_10G_KPS")
        self.assertEqual(S.SELECTED_WEIGHT_SHA256, SCRFD_SHA256)

    def test_25g_is_preserved_as_unselected_candidate(self):
        from gpatbench.preprocess import scrfd as S
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]["scrfd"]
        sel = {c["candidate_sha256"]: c.get("selection") for c in reg["local_candidates"]}
        self.assertEqual(sel["041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9"], "NOT_SELECTED_FOR_FINAL_M2")
        self.assertIn("041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9", S.NOT_SELECTED_SHA256)
        self.assertEqual(reg["m2a_resolution"]["candidate_B"]["identified_as"], "SCRFD_2.5G_KPS")   # history preserved

    def test_official_byte_verification_recorded(self):
        v = json.loads((AUDIT / "M2A_OFFICIAL_VERIFICATION.json").read_text())["scrfd"]
        self.assertEqual(v["match"], "OFFICIAL_BYTE_HASH_CONFIRMED")
        self.assertEqual(v["local_sha256"], v["official_member_sha256"])
        self.assertEqual(v["official_member"], "antelopev2/scrfd_10g_bnkps.onnx")

    @unittest.skipUnless(HAS_TORCH and SCRFD_NOT_SELECTED.exists(), "requires the M2 environment and the local candidate")
    def test_25g_cannot_become_active(self):
        from gpatbench.preprocess.scrfd import SCRFD
        with self.assertRaises(ValueError) as e:      # refused even without an expected hash
            SCRFD(SCRFD_NOT_SELECTED)
        self.assertIn("did not select", str(e.exception))
        with self.assertRaises(ValueError):           # and refused when its own hash is offered
            SCRFD(SCRFD_NOT_SELECTED, "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9")


class TestQ03Q18Q19AdaFaceRegistry(unittest.TestCase):
    def test_selected_is_original_r50_webface4m(self):
        from gpatbench.preprocess import aux_models as A
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]["adaface_ir50"]
        cfg = yaml.safe_load(FROZEN_CFG.read_text())["adaface"]
        self.assertEqual(reg["weight_sha256"], ADAFACE_SHA256)
        self.assertEqual((reg["variant"], reg["training_dataset"], reg["release_line"]),
                         ("IR-50", "WebFace4M", "original_repository"))
        self.assertEqual(reg["status"], "VERIFIED")
        self.assertEqual(reg["code_commit"], "c60eaa786a42c03444f3df7096dbaf9d57ae010d")
        self.assertEqual((cfg["architecture"], cfg["training_dataset"], cfg["weight_sha256"]),
                         ("IR-50", "WebFace4M", ADAFACE_SHA256))
        self.assertEqual(cfg["color_order"], "BGR")
        self.assertEqual(cfg["geometric_adapter"], "RESIZE_256_TO_112_INTER_AREA")
        self.assertEqual(cfg["additional_alignment"], "none")
        self.assertEqual(A.ADAFACE_WEIGHT_SHA256, ADAFACE_SHA256)
        self.assertEqual((A.ADAFACE_COLOR_ORDER, A.ADAFACE_GEOMETRIC), ("BGR", "RESIZE_256_TO_112_INTER_AREA"))

    def test_cvlface_candidate_preserved_but_not_selected(self):
        from gpatbench.preprocess import aux_models as A
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]["adaface_ir50"]
        sel = {c.get("candidate_sha256"): c.get("selection") for c in reg["local_candidates"]}
        self.assertEqual(sel[CVLFACE_SHA256], "NOT_SELECTED_FOR_FINAL_M2")
        self.assertIn(CVLFACE_SHA256, A.ADAFACE_NOT_SELECTED_SHA256)
        self.assertEqual(reg["m2a_resolution"]["local_candidate_sha256"], CVLFACE_SHA256)   # history preserved
        self.assertTrue(reg["m2a_owner_resolution"]["not_selected"]["file_retained_on_disk"])

    def test_adapter_has_no_alignment_or_detector_code(self):
        """Executable code only: docstrings may (and do) *explain* that no aligner is used."""
        import ast
        src = (ROOT / "gpatbench/preprocess/aux_models.py").read_text()
        cls = next(n for n in ast.parse(src).body
                   if isinstance(n, ast.ClassDef) and n.name == "AdaFaceAdapter")
        ast.get_docstring(cls)   # present by construction; its prose is deliberately not scanned
        names = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node is not cls.body[0].value:
                names.add(node.value)
        joined = " ".join(names).lower()
        for forbidden in ("mtcnn", "facenet", "face_alignment", "get_aligned_face", "scrfd", "landmark", "insightface"):
            self.assertNotIn(forbidden, joined, forbidden)

    def test_official_adaface_verification_recorded(self):
        v = json.loads((AUDIT / "M2A_OFFICIAL_VERIFICATION.json").read_text())["adaface"]
        self.assertEqual(v["sha256"], ADAFACE_SHA256)
        self.assertEqual(v["strict_load"], "OK")
        self.assertEqual(v["training_dataset"], "WebFace4M")
        c = v["cvlface_comparison"]
        self.assertEqual(c["n_tensors_differing"], 1)
        self.assertTrue(c["first_conv_is_exact_channel_reversal"])


@unittest.skipUnless(HAS_TORCH and HAS_WEIGHTS, "requires the M2 environment and the official checkpoint")
class TestQ18Q19AdaFaceAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpatbench.preprocess.aux_models import AdaFaceAdapter, set_deterministic
        set_deterministic(4)
        cls.ada = AdaFaceAdapter(CACHE / "code/adaface", ADAFACE_CKPT)
        cls.rgb = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def test_refuses_the_cvlface_checkpoint(self):
        from gpatbench.preprocess.aux_models import AdaFaceAdapter
        if not ADAFACE_CVLFACE.exists():
            self.skipTest("CVLFace candidate file not present")
        with self.assertRaises(ValueError) as e:
            AdaFaceAdapter(CACHE / "code/adaface", ADAFACE_CVLFACE)
        self.assertIn("NOT_SELECTED_FOR_FINAL_M2", str(e.exception))

    def test_strict_load_and_output_contract(self):
        e = self.ada(self.rgb)
        self.assertEqual((e["embedding"].shape, e["embedding"].dtype), ((512,), np.float32))
        self.assertTrue(np.all(np.isfinite(e["embedding"])))
        self.assertAlmostEqual(float(np.linalg.norm(e["embedding"])), 1.0, delta=1e-5)
        self.assertEqual(self.ada.weight_sha256, ADAFACE_SHA256)

    def test_deterministic_embedding(self):
        a, b = self.ada(self.rgb)["embedding"], self.ada(self.rgb)["embedding"]
        np.testing.assert_array_equal(a, b)

    def test_256_to_112_uses_inter_area_once(self):
        import gpatbench.preprocess.aux_models as A
        calls = []
        real = A.__dict__.get("cv2")
        import cv2 as _cv2
        orig = _cv2.resize
        def spy(src, dsize, *a, **k):
            calls.append((dsize, k.get("interpolation", a[2] if len(a) > 2 else None)))
            return orig(src, dsize, *a, **k)
        _cv2.resize = spy
        try:
            self.ada.preprocess(self.rgb)
        finally:
            _cv2.resize = orig
        self.assertEqual(calls, [((112, 112), cv2.INTER_AREA)])
        self.assertIsNone(real)   # the module imports cv2 locally; no global rebinding happened

    def test_no_mtcnn_or_detector_module_is_loaded(self):
        self.ada(self.rgb)
        for mod in ("facenet_pytorch", "mtcnn", "face_alignment", "insightface"):
            self.assertNotIn(mod, sys.modules, mod)

    def test_bgr_reversal_with_synthetic_pattern(self):
        rgb = np.zeros((256, 256, 3), np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 128, 0          # R=255, G=128, B=0
        x = self.ada.preprocess(rgb)[0].numpy()
        self.assertAlmostEqual(float(x[0].mean()), (0 / 255 - 0.5) / 0.5, places=5)      # channel 0 = B
        self.assertAlmostEqual(float(x[1].mean()), (128 / 255 - 0.5) / 0.5, places=5)    # channel 1 = G
        self.assertAlmostEqual(float(x[2].mean()), (255 / 255 - 0.5) / 0.5, places=5)    # channel 2 = R

    def test_exact_official_to_input_arithmetic(self):
        """Reimplements the official inference.py to_input and requires exact equality."""
        import torch
        small = cv2.resize(self.rgb, (112, 112), interpolation=cv2.INTER_AREA)
        official = torch.tensor([(((np.array(small)[:, :, ::-1] / 255.0) - 0.5) / 0.5).transpose(2, 0, 1)]).float()
        got = self.ada.preprocess(self.rgb)
        self.assertTrue(torch.equal(official, got))
        self.assertEqual(str(got.dtype), "torch.float32")
        self.assertTrue(float(got.min()) >= -1.0 and float(got.max()) <= 1.0)

    def test_rejects_non_canonical_input(self):
        with self.assertRaises(ValueError):
            self.ada.preprocess(np.zeros((112, 112, 3), np.uint8))


class TestQ23LogitStorage(unittest.TestCase):
    def setUp(self):
        from gpatbench.preprocess import logit_store as LS
        self.LS = LS
        self.arr = (np.random.default_rng(11).standard_normal(LS.PARSING_LOGITS_SHAPE) * 8).astype(np.float32)

    def test_float32_full_logits_are_the_frozen_representation(self):
        cfg = yaml.safe_load(FROZEN_CFG.read_text())["cache"]["geometry_parsing_logits"]
        self.assertEqual(cfg["representation"], "FULL_FLOAT32_LOGITS")
        self.assertEqual(cfg["dtype"], "float32")
        self.assertEqual(cfg["shape"], [11, 224, 224])
        self.assertIs(cfg["lossy_reduction_forbidden"], True)
        self.assertEqual(yaml.safe_load(FROZEN_CFG.read_text())["owner_decisions"]["Q-23"],
                         "RESOLVED_BY_OWNER_FLOAT32_FULL_LOGITS")

    def test_float16_or_mask_only_cannot_silently_replace_the_logits(self):
        with self.assertRaises(ValueError):
            self.LS.npy_bytes(self.arr.astype(np.float16))
        with self.assertRaises(ValueError):
            self.LS.encode_block(self.arr.astype(np.float16))
        with self.assertRaises(ValueError):
            self.LS.mask_from_logits(self.arr.astype(np.float16))
        cfg = yaml.safe_load(FROZEN_CFG.read_text())["cache"]["geometry_parsing_logits"]
        self.assertNotIn("float16", json.dumps(cfg))

    def test_mask_derives_from_the_stored_logits(self):
        blob = self.LS.encode_block(self.arr)
        back = self.LS.decode_block(blob)
        np.testing.assert_array_equal(self.LS.mask_from_logits(self.arr), self.LS.mask_from_logits(back))
        self.assertEqual(self.LS.mask_from_logits(back).dtype, np.uint8)
        np.testing.assert_array_equal(self.LS.mask_from_logits(back), back.argmax(axis=0).astype(np.uint8))

    def test_compression_is_lossless_and_exact(self):
        npy = self.LS.npy_bytes(self.arr)
        blob = self.LS.encode_block(self.arr)
        self.assertEqual(self.LS.decode_block_npy_bytes(blob), npy)          # raw byte representation
        back = self.LS.decode_block(blob)
        self.assertTrue(np.array_equal(self.arr, back))                      # no tolerance
        self.assertEqual(back.dtype, np.float32)
        self.assertEqual(back.tobytes(), self.arr.tobytes())

    def test_shuffle_is_exactly_invertible(self):
        b = self.arr.tobytes()
        self.assertEqual(self.LS.unshuffle4(self.LS.shuffle4(b)), b)

    def test_encoding_is_deterministic(self):
        self.assertEqual(self.LS.encode_block(self.arr), self.LS.encode_block(self.arr))

    def test_shard_round_trip_and_random_access(self):
        with tempfile.TemporaryDirectory() as t:
            w = self.LS.ShardWriter(Path(t), prefix="parsing_logits", rows_per_shard=2)
            arrs = {f"s{i}": (self.arr + i).astype(np.float32) for i in range(3)}
            rows = [w.add(k, v) for k, v in arrs.items()]
            w.close()
            self.assertEqual(sorted({r["shard"] for r in rows}), ["parsing_logits-00000.bin", "parsing_logits-00001.bin"])
            for r in rows:
                got = self.LS.read_block(Path(t) / r["shard"], r["byte_offset"], r["byte_length"])
                self.assertTrue(np.array_equal(arrs[r["sample_id"]], got))
                self.assertEqual(r["dtype"], "<f4")
                self.assertEqual(r["codec_id"], self.LS.CODEC_ID)

    def test_codec_provenance_matches_the_frozen_config(self):
        prov = self.LS.codec_provenance()
        cfg = yaml.safe_load(FROZEN_CFG.read_text())["cache"]["geometry_parsing_logits"]
        self.assertIs(prov["lossy"], False)
        self.assertEqual(cfg["storage_codec_id"], prov["codec_id"])
        self.assertEqual(cfg["dtype_descr"], prov["dtype"])
        self.assertEqual(cfg["compressor"]["level"], prov["level"])
        self.assertEqual(cfg["compressor"]["library_version"], prov["library_version"])
        self.assertEqual(cfg["compressor"]["libzstd_version"], prov["libzstd_version"])
        self.assertEqual(cfg["shard"]["rows_per_shard"], prov["shard_rows"])

    def test_pilot_evidence_is_lossless(self):
        s = json.loads((AUDIT / "M2A_COMPRESSION_PILOT.json").read_text())
        self.assertTrue(s["exact_reconstruction_all"])
        self.assertTrue(s["deterministic_all"])
        rows = list(csv.DictReader(open(AUDIT / "M2A_COMPRESSION_PILOT.csv")))
        self.assertTrue(rows and all(r["array_equal"] == "True" and r["npy_bytes_equal"] == "True"
                                     and r["raw_bytes_equal"] == "True" for r in rows))


class TestFrozenPreprocessConfig(unittest.TestCase):
    BLOCKING_NULL_ALLOWED = {("facexformer", "parsing_class_names")}   # Q-21: non-blocking for M2B

    def test_no_m2b_blocking_null_field(self):
        cfg = yaml.safe_load(FROZEN_CFG.read_text())
        nulls = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, path + (str(k),))
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, path + (str(i),))
            elif node is None:
                nulls.append(path)

        walk(cfg, ())
        unexpected = [p for p in nulls if (p[0], p[-1]) not in self.BLOCKING_NULL_ALLOWED]
        self.assertEqual(unexpected, [], f"unresolved null fields in the frozen config: {unexpected}")
        self.assertEqual(cfg["status"], "FROZEN")
        for q, expected in (("Q-02", "RESOLVED_BY_OWNER_SELECTION"), ("Q-03", "RESOLVED_BY_OWNER_SELECTION"),
                            ("Q-18", "RESOLVED_BY_OWNER"), ("Q-19", "RESOLVED_BY_OWNER_CANONICAL_FACE_ADAPTER"),
                            ("Q-22", "RESOLVED_BY_OWNER"), ("Q-23", "RESOLVED_BY_OWNER_FLOAT32_FULL_LOGITS")):
            self.assertEqual(cfg["owner_decisions"][q], expected, q)

    def test_snapshot_identical_and_hashes_recorded(self):
        import hashlib
        snap = ROOT / "frozen_config_snapshot/configs/frozen/preprocess_v1.yaml"
        self.assertEqual(snap.read_bytes(), FROZEN_CFG.read_bytes())
        with open(AUDIT / "ARTIFACT_INDEX.csv", newline="") as f:
            idx = {r["path"]: r["sha256"] for r in csv.DictReader(f)}
        for rel in ("configs/frozen/preprocess_v1.yaml", "frozen_config_snapshot/configs/frozen/preprocess_v1.yaml",
                    "models/registry.yaml"):
            self.assertEqual(idx[rel], hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), rel)

    def test_proposal_history_preserved(self):
        prop = ROOT / "configs/proposed/preprocess_v1.proposed.yaml"
        self.assertTrue(prop.is_file())
        self.assertEqual(yaml.safe_load(prop.read_text())["status"], "PROPOSED_BLOCKED")   # unchanged history
        self.assertEqual(yaml.safe_load(FROZEN_CFG.read_text())["supersedes"], prop.relative_to(ROOT).as_posix())

    def test_config_references_exact_registry_entries(self):
        cfg = yaml.safe_load(FROZEN_CFG.read_text())
        reg = yaml.safe_load((ROOT / "models/registry.yaml").read_text())["models"]
        for section, key in (("scrfd", "scrfd"), ("facexformer", "facexformer"), ("adaface", "adaface_ir50")):
            self.assertEqual(cfg[section]["registry_key"], key)
            self.assertEqual(cfg[section]["weight_sha256"], reg[key]["weight_sha256"], key)
            self.assertEqual(cfg[section]["code_commit"], reg[key]["code_commit"], key)
            self.assertEqual(reg[key]["status"], "VERIFIED", key)


class TestBorderCaseSmokeEvidence(unittest.TestCase):
    def test_border_cases_recorded_and_square(self):
        rows = list(csv.DictReader(open(AUDIT / "M2A_BORDER_CASES.csv")))
        self.assertTrue(rows)
        for r in rows:
            side = int(r["side_px"])
            self.assertEqual(r["pre_resize_hw"], f"{side}x{side}")          # a true square before resizing
            self.assertEqual(r["final_hw"], "256x256")
            rq, src = json.loads(r["requested_square"]), json.loads(r["source_intersection"])
            self.assertEqual(rq[2] - rq[0], side)
            self.assertEqual(rq[3] - rq[1], side)
            self.assertEqual((int(r["pad_left"]), int(r["pad_top"])), (src[0] - rq[0], src[1] - rq[1]))
            self.assertEqual((int(r["pad_right"]), int(r["pad_bottom"])), (rq[2] - src[2], rq[3] - src[3]))
            self.assertEqual(r["is_square_before_resize"], "True")

    def test_smoke_results_carry_the_geometry_fields(self):
        res = list(csv.DictReader(open(AUDIT / "M2A_SMOKE_RESULTS.csv")))
        det = [r for r in res if r["scrfd_status"] == "DETECTED"]
        self.assertTrue(det)
        for r in det:
            side = int(r["crop_side_px"])
            self.assertEqual(r["crop_pre_resize_hw"], f"{side}x{side}")
        self.assertTrue(all(r["logits_array_equal"] == "True" and r["logits_bytes_equal"] == "True"
                            and r["mask_from_stored_logits"] == "True" for r in res if r["status"] == "OK"))
        self.assertTrue(all(r["logits_dtype"] == "float32" for r in res if r["status"] == "OK"))


if __name__ == "__main__":
    unittest.main()
