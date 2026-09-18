"""M1 inventory unit tests on small synthetic fixtures (run with the project venv).

Run: .venv/bin/python -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_fixtures  # noqa: E402
from gpatbench.data import casia_fasd, msu_mfsd, siwmv2  # noqa: E402
from gpatbench.data.base import (CANONICAL_FRAME_IMAGE, CANONICAL_VIDEO, DATASET_METADATA,  # noqa: E402
                                 DERIVED_AUGMENTATION_COPY, DOCUMENTATION, AUXILIARY_TOOL,
                                 UnclassifiedFileError, sample_id, select_frame_indices)
from gpatbench.data.inventory import (FILE_SCHEMA, SAMPLE_SCHEMA, VIDEO_SCHEMA, InventoryError,  # noqa: E402
                                      _write_parquet, run_inventory)

POS = [0.10, 0.2142857, 0.3285714, 0.4428571, 0.5571429, 0.6714286, 0.7857143, 0.90]
DETERMINISTIC_OUTPUTS = ["manifests/inventory.parquet", "manifests/inventory_videos.parquet",
                         "manifests/raw_file_index.parquet", "outputs/audit/dataset_inventory_summary.csv",
                         "outputs/audit/dataset_attack_coverage.csv", "outputs/audit/unmapped_attack_tokens.csv",
                         "outputs/audit/dataset_integrity_issues.csv", "outputs/audit/dataset_duplicate_hashes.csv",
                         "outputs/audit/dataset_report.html"]


def _tree_digest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class TestAdapters(unittest.TestCase):
    def test_casia_classify(self):
        fc = casia_fasd.classify("train/spoof/s12vHR_3f7.png")
        self.assertEqual((fc.category, fc.video_key, fc.frame_index), (CANONICAL_FRAME_IMAGE, "train/spoof/s12vHR_3", 7))
        self.assertEqual(casia_fasd.classify("train/live/fs3v1f0.png").category, DERIVED_AUGMENTATION_COPY)
        self.assertEqual(casia_fasd.classify("train/live/bs3v1f0.png").video_key, "train/live/s3v1")

    def test_msu_classify(self):
        rel = "scene01/attack/attack_client007_laptop_SD_iphone_video_scene01.mov"
        self.assertEqual(msu_mfsd.classify(rel), msu_mfsd.classify(rel))
        self.assertEqual(msu_mfsd.classify(rel).category, CANONICAL_VIDEO)
        self.assertEqual(msu_mfsd.classify(rel.replace(".mov", ".face")).category, DATASET_METADATA)
        self.assertEqual(msu_mfsd.classify(rel).video_key, msu_mfsd.classify(rel.replace(".mov", ".face")).video_key)
        self.assertEqual(msu_mfsd.classify("README.txt").category, DOCUMENTATION)
        self.assertEqual(msu_mfsd.classify("ffmpeg/licenses/x264.txt").category, AUXILIARY_TOOL)

    def test_siw_classify(self):
        self.assertEqual(siwmv2.classify("Spoof/Mannequin/Mask_Mann_10.mov").category, CANONICAL_VIDEO)
        self.assertEqual(siwmv2.classify("Live/Live_889.mp4").video_key, "Live/Live_889.mp4")

    def test_malformed_paths_raise(self):
        for mod, rel in ((casia_fasd, "train/live/s1v9f0.png"), (casia_fasd, "train/live/s1v1f0.jpg"),
                         (casia_fasd, "val/live/s1v1f0.png"), (msu_mfsd, "scene01/real/real_client1_laptop_SD_scene01.mov"),
                         (msu_mfsd, "scene02/real/real_client001_laptop_SD_scene02.mov"), (siwmv2, "Other/x_1.mov"),
                         (siwmv2, "Spoof/Replay/Replay_1.mkv")):
            with self.assertRaises(UnclassifiedFileError, msg=rel):
                mod.classify(rel)

    def test_siw_lowercase_dirs_not_silently_merged(self):
        with self.assertRaises(UnclassifiedFileError):
            siwmv2.classify("live/Live_1.mov")
        with self.assertRaises(UnclassifiedFileError):
            siwmv2.classify("spoof/Replay/Replay_1.mov")


class TestIds(unittest.TestCase):
    def test_sample_id_exact_serialization(self):
        exp = hashlib.sha256(b'["gpatbench.sample_id.v1","msu_mfsd","v/x",12]').hexdigest()
        self.assertEqual(sample_id("msu_mfsd", "v/x", 12), exp)
        self.assertNotEqual(sample_id("msu_mfsd", "v/x", 12), sample_id("msu_mfsd", "v/x", 13))
        self.assertNotEqual(sample_id("msu_mfsd", "v/x", 12), sample_id("siwmv2", "v/x", 12))
        with self.assertRaises(ValueError):
            sample_id("oulu", "v", 0)

    def test_frame_selection(self):
        self.assertEqual(select_frame_indices(range(5), POS), ([0, 1, 2, 3, 4], False))
        idx, fb = select_frame_indices(range(8), POS)
        self.assertEqual(idx, list(range(8)))
        idx, _ = select_frame_indices(range(9), POS)
        self.assertEqual(len(set(idx)), 8)
        idx, fb = select_frame_indices(range(0, 71), POS)   # hi-lo = 70
        self.assertEqual(idx, [7, 15, 23, 31, 39, 47, 55, 63])
        self.assertFalse(fb)
        idx, _ = select_frame_indices([0, 1, 2, 3, 50, 51, 52, 53, 100], POS)  # gaps: only valid indices chosen
        self.assertTrue(set(idx) <= {0, 1, 2, 3, 50, 51, 52, 53, 100})
        self.assertEqual(len(idx), 8)


class FixtureRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gpat_m1_"))
        cls.fx = m1_fixtures.build(cls.tmp)
        for p in [cls.fx["casia"], cls.fx["msu"], cls.fx["siw"]]:   # raw roots read-only during the run
            for q in sorted(p.rglob("*"), reverse=True):
                q.chmod(q.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        cls.before = {k: _tree_digest(cls.fx[k]) for k in ("casia", "msu", "siw")}
        cls.run_result = run_inventory(str(cls.fx["config"]), workers=2, project_root=cls.fx["project_root"])
        cls.samples = pq.read_table(cls.fx["project_root"] / "manifests/inventory.parquet")
        cls.videos = pq.read_table(cls.fx["project_root"] / "manifests/inventory_videos.parquet").to_pylist()

    @classmethod
    def tearDownClass(cls):
        for p in cls.tmp.rglob("*"):
            if p.is_dir():
                p.chmod(p.stat().st_mode | stat.S_IWUSR)
        shutil.rmtree(cls.tmp)

    def _v(self, ds, vid):
        return next(v for v in self.videos if v["dataset"] == ds and v["video_id"] == vid)

    def test_raw_unchanged_and_readonly(self):
        self.assertTrue(self.run_result["raw_data_unchanged"])
        self.assertEqual({k: _tree_digest(self.fx[k]) for k in ("casia", "msu", "siw")}, self.before)

    def test_subject_ids(self):
        tr, te = self._v("casia_fasd", "train/live/s1v1"), self._v("casia_fasd", "test/live/s1v1")
        self.assertEqual((tr["subject_id_raw"], te["subject_id_raw"]), ("train_s1", "test_s1"))
        self.assertNotEqual(tr["subject_id_global"], te["subject_id_global"])
        m = self._v("msu_mfsd", "scene01/real/real_client002_laptop_SD_scene01")
        self.assertEqual((m["subject_id_raw"], m["subject_id_global"], m["native_protocol_split"]),
                         ("002", "msu_mfsd::002", "test"))
        s = self._v("siwmv2", "Live/Live_1.mov")
        self.assertIsNone(s["subject_id_raw"])
        self.assertEqual(s["subject_id_status"], "MISSING_NOT_IN_LOCAL_METADATA")
        for v in self.videos:
            if v["subject_id_raw"] is not None:
                self.assertEqual(v["subject_id_global"], f"{v['dataset']}::{v['subject_id_raw']}")

    def test_labels_and_attacks(self):
        self.assertEqual(self._v("casia_fasd", "train/live/s1v1")["label_binary"], 0)
        c = self._v("casia_fasd", "train/spoof/s1v3")
        self.assertEqual((c["label_binary"], c["attack_raw"], c["attack_macro"], c["attack_map_status"], c["style_id"]),
                         (1, "3", "other_spoof", "UNMAPPED_OTHER_SPOOF", "casia_fasd::3"))
        self.assertIsNotNone(self._v("casia_fasd", "train/spoof/s1vHR_1")["label_conflict"])
        m = self._v("msu_mfsd", "scene01/attack/attack_client001_android_SD_printed_photo_scene01")
        self.assertEqual((m["attack_raw"], m["attack_macro"]), ("printed_photo", "print"))
        r = self._v("siwmv2", "Spoof/Replay/Replay_1.mov")
        self.assertEqual((r["attack_raw"], r["attack_macro"], r["attack_map_status"]), ("Replay", "replay", "MAPPED"))
        p = self._v("siwmv2", "Spoof/Paper/Paper_7.mov")
        self.assertEqual((p["attack_raw"], p["attack_macro"]), ("Paper", "other_spoof"))
        live = [v for v in self.videos if v["label_binary"] == 0]
        self.assertTrue(all(v["attack_macro"] == "live" and v["attack_raw"] is None for v in live))

    def test_unmapped_csv(self):
        text = (self.fx["project_root"] / "outputs/audit/unmapped_attack_tokens.csv").read_text()
        self.assertIn("siwmv2,Paper,1,other_spoof,print", text)
        self.assertIn("casia_fasd,HR_1", text)

    def test_samples(self):
        sids = self.samples.column("sample_id").to_pylist()
        self.assertEqual(len(sids), len(set(sids)))
        self.assertEqual(self._v("siwmv2", "Live/Live_2.mp4")["n_sampled"], 5)  # never padded to 8
        self.assertEqual(self._v("siwmv2", "Spoof/Replay/Replay_1.mov")["n_sampled"], 8)
        for row in self.samples.to_pylist():
            self.assertEqual(row["sample_id"], sample_id(row["dataset"], row["video_id"], row["frame_index"]))
            if row["dataset"] == "casia_fasd":
                self.assertEqual(row["sha256_kind"], "original_frame_bytes")
                self.assertEqual(len(row["sha256"]), 64)
            else:
                self.assertIsNone(row["sha256"])
                self.assertEqual(row["sha256_kind"], "PENDING_M2_CANONICAL_PNG")

    def test_derived_copies_excluded(self):
        paths = set(self.samples.column("source_path").to_pylist())
        self.assertFalse(any(p.rsplit("/", 1)[-1][0] in "bf" for p in paths if p.startswith("train/live/")))

    def test_schema_and_no_split(self):
        self.assertEqual(self.samples.schema, SAMPLE_SCHEMA)
        self.assertNotIn("split", SAMPLE_SCHEMA.names + VIDEO_SCHEMA.names + FILE_SCHEMA.names)
        with self.assertRaises(InventoryError):
            import pyarrow as pa
            _write_parquet([], pa.schema([("split", pa.string())]), self.tmp / "x.parquet")
        for col in ("sample_id", "dataset", "video_id", "frame_index", "label_binary", "attack_macro", "source_path"):
            self.assertEqual(self.samples.column(col).null_count, 0, col)

    def test_archive_crosscheck(self):
        text = (self.fx["project_root"] / "outputs/audit/dataset_archive_crosscheck.csv").read_text().splitlines()
        self.assertTrue(text[1].endswith(",6,6,0,0,0,0"), text[1])

    def test_rerun_byte_deterministic(self):
        other = self.tmp / "project2"
        (other / "manifests").mkdir(parents=True)
        (other / "outputs/audit").mkdir(parents=True)
        run_inventory(str(self.fx["config"]), workers=3, project_root=other)
        for rel in DETERMINISTIC_OUTPUTS:
            self.assertEqual((self.fx["project_root"] / rel).read_bytes(), (other / rel).read_bytes(), rel)


class TestFailLoudly(unittest.TestCase):
    def test_unclassified_file_fails_run(self):
        tmp = Path(tempfile.mkdtemp(prefix="gpat_m1_bad_"))
        try:
            fx = m1_fixtures.build(tmp, casia_frames=3)
            (fx["siw"] / "notes.txt").write_text("stray")
            with self.assertRaises(UnclassifiedFileError):
                run_inventory(str(fx["config"]), workers=1, project_root=fx["project_root"], make_report=False)
        finally:
            shutil.rmtree(tmp)

    def test_msu_protocol_overlap_fails(self):
        tmp = Path(tempfile.mkdtemp(prefix="gpat_m1_msu_"))
        try:
            (tmp / "train_sub_list.txt").write_text("01\n02\n")
            (tmp / "test_sub_list.txt").write_text("02\n")
            with self.assertRaises(ValueError):
                msu_mfsd.read_protocol_lists(tmp)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()


class TestVideoDecodeLoop(unittest.TestCase):
    """Q-04: a failed read inside the stream is one invalid position; decoding continues."""

    def _run(self, pattern, declared):
        import numpy as np
        from unittest import mock
        from gpatbench.data import frames

        class FakeCap:
            def __init__(self, *a):
                self.seq = list(pattern)
            def isOpened(self):
                return True
            def get(self, prop):
                return declared if prop == frames.cv2.CAP_PROP_FRAME_COUNT else 25.0
            def read(self):
                ok = self.seq.pop(0) if self.seq else False
                return (True, np.zeros((2, 2, 3), np.uint8)) if ok else (False, None)
            def release(self):
                pass
        with mock.patch.object(frames.cv2, "VideoCapture", FakeCap):
            return frames._probe_video("/nonexistent", "x.mov")

    def test_failure_inside_stream_continues(self):
        r = self._run([1, 1, 1, 0, 1, 1], declared=6)
        self.assertEqual(r["valid_indices"], [0, 1, 2, 4, 5])
        self.assertEqual(r["invalid_indices"], [3])
        self.assertEqual(r["decode_status"], "SOME_FRAMES_UNDECODABLE")

    def test_trailing_failures_are_end_of_stream(self):
        r = self._run([1, 1, 1], declared=5)
        self.assertEqual((r["valid_indices"], r["invalid_indices"], r["decode_status"]),
                         ([0, 1, 2], [], "DECODED_LT_DECLARED"))
