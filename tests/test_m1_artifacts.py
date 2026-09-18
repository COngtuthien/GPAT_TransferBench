"""Validate the committed M1 artifacts against the M1 hard gate (spec §25).

Checks the real manifests produced by `python -m gpatbench.cli inventory` (no dataset access).
"""
import collections
import csv
import json
import sys
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.audit.ledger import sha256_file  # noqa: E402
from gpatbench.data.base import ATTACK_MACROS, DATASETS, FILE_CATEGORIES, sample_id  # noqa: E402
from gpatbench.data.inventory import FILE_SCHEMA, SAMPLE_SCHEMA, VIDEO_SCHEMA  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
OUT = {k: ROOT / v for k, v in CFG["outputs"].items()}


def _csv(key):
    with open(OUT[key], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestM1Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = pq.read_table(OUT["inventory"])
        cls.samples = cls.s.to_pylist()
        cls.v = pq.read_table(OUT["inventory_videos"])
        cls.videos = cls.v.to_pylist()
        cls.files = pq.read_table(OUT["raw_file_index"]).to_pylist()

    def test_schemas(self):
        self.assertEqual(self.s.schema, SAMPLE_SCHEMA)
        self.assertEqual(self.v.schema, VIDEO_SCHEMA)
        self.assertEqual(pq.read_table(OUT["raw_file_index"]).schema, FILE_SCHEMA)
        for name in (SAMPLE_SCHEMA.names + VIDEO_SCHEMA.names):
            self.assertNotEqual(name, "split")

    def test_no_duplicate_sample_ids_and_stable(self):
        ids = [r["sample_id"] for r in self.samples]
        self.assertEqual(len(ids), len(set(ids)))
        for r in self.samples:
            self.assertEqual(r["sample_id"], sample_id(r["dataset"], r["video_id"], r["frame_index"]))

    def test_video_ids_unique_and_samples_consistent(self):
        keys = [(v["dataset"], v["video_id"]) for v in self.videos]
        self.assertEqual(len(keys), len(set(keys)))
        per = collections.defaultdict(list)
        for r in self.samples:
            per[(r["dataset"], r["video_id"])].append(r["frame_index"])
        for v in self.videos:
            chosen = json.loads(v["sampled_frame_indices"])
            self.assertEqual(sorted(per.get((v["dataset"], v["video_id"]), [])), chosen)
            self.assertEqual(len(chosen), min(8, v["n_valid_frames"]))
            self.assertEqual(len(set(chosen)), len(chosen))

    def test_all_files_indexed_and_classified(self):
        self.assertEqual({f["dataset"] for f in self.files}, set(DATASETS))
        for f in self.files:
            self.assertIn(f["category"], FILE_CATEGORIES)
            self.assertTrue(f["readable"], f["rel_path"])
            self.assertEqual(len(f["sha256"]), 64)
        canon = {(f["dataset"], f["video_id"]) for f in self.files if f["category"].startswith("CANONICAL")}
        self.assertEqual(canon, {(v["dataset"], v["video_id"]) for v in self.videos})

    def test_labels_and_attack_fields(self):
        for v in self.videos:
            self.assertIn(v["label_binary"], (0, 1))
            self.assertIn(v["attack_macro"], ATTACK_MACROS)
            if v["label_binary"] == 0:
                self.assertEqual((v["attack_macro"], v["attack_raw"]), ("live", None))
            else:
                self.assertIsNotNone(v["attack_raw"])
                self.assertEqual(v["style_id"], f"{v['dataset']}::{v['attack_raw']}")
            if v["subject_id_raw"] is not None:
                self.assertEqual(v["subject_id_global"], f"{v['dataset']}::{v['subject_id_raw']}")

    def test_unknown_labels_listed(self):
        observed = {(v["dataset"], v["attack_raw"]) for v in self.videos if v["attack_map_status"] == "UNMAPPED_OTHER_SPOOF"}
        listed = {(r["dataset"], r["attack_raw"]) for r in _csv("unmapped_csv")}
        self.assertEqual(observed, listed)

    def test_report_generated_from_inventory(self):
        html = OUT["report_html"].read_text(encoding="utf-8")
        self.assertIn(f"{len(self.samples)} sampled-frame rows", html)
        self.assertIn(f"{len(self.files)} indexed files", html)
        self.assertNotIn("<img", html)

    def test_run_record_matches_outputs(self):
        run = json.loads(OUT["run_json"].read_text())
        self.assertTrue(run["raw_data_unchanged"])
        for key, h in run["outputs_sha256"].items():
            if key in ("facts_json",):
                continue
            self.assertEqual(sha256_file(OUT[key]), h, key)


if __name__ == "__main__":
    unittest.main()
