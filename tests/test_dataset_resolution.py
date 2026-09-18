"""Dataset-resolution pass tests (DEV-005, DEV-006, CASIA, SiW Q-14/Q-16, readiness, no M2/M3).

Evidence files are produced by tools/audit_*.py and committed; these tests check their content and
their consistency with the inventory manifests. Run with the project venv.
"""
import csv
import json
import sys
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.data.base import select_frame_indices  # noqa: E402

AUDIT = ROOT / "outputs/audit"
SPEC_POSITIONS = [0.10, 0.2142857, 0.3285714, 0.4428571, 0.5571429, 0.6714286, 0.7857143, 0.90]


def _csv(name):
    with open(AUDIT / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _videos(ds=None):
    rows = pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist()
    return [r for r in rows if ds is None or r["dataset"] == ds]


class TestDEV005(unittest.TestCase):
    def test_frozen_positions_unchanged(self):
        cfg = yaml.safe_load((ROOT / "configs/frozen/data_v1.yaml").read_text())
        self.assertEqual(cfg["frame_sampling"]["positions"], SPEC_POSITIONS)
        self.assertIn("[0.10, 0.2142857, 0.3285714, 0.4428571, 0.5571429, 0.6714286, 0.7857143, 0.90]",
                      (ROOT / "docs/spec/spec_extracted_text.txt").read_text())

    def test_exact_tie_lower_index_wins(self):
        self.assertEqual(select_frame_indices([0, 1], [0.5]), ([0], False))

    def test_collision_earlier_position_keeps_frame(self):
        idx, fb = select_frame_indices([0, 10, 20], [0.5, 0.5])
        self.assertEqual((idx, fb), ([0, 10], True))   # 2nd target: 0 and 20 tie at distance 10 -> 0

    def test_short_sequence_uses_all_once(self):
        self.assertEqual(select_frame_indices([3, 5, 9], SPEC_POSITIONS), ([3, 5, 9], False))

    def test_missing_internal_frames(self):
        v = list(range(0, 40)) + list(range(60, 101))   # indices 40..59 missing
        idx, _ = select_frame_indices(v, SPEC_POSITIONS)
        self.assertEqual(len(idx), 8)
        self.assertTrue(all(i in v for i in idx))
        self.assertIn(39, idx)   # target 42.857 -> nearest valid is 39 (60 is farther)

    def test_boundaries(self):
        self.assertEqual(select_frame_indices(range(8), SPEC_POSITIONS)[0], list(range(8)))
        idx, _ = select_frame_indices(range(1000), SPEC_POSITIONS)
        self.assertTrue(min(idx) > 0 and max(idx) < 999)

    def test_measured_collision_usage_zero(self):
        self.assertFalse(any(r["collision_resolution_used"] for r in _videos()))


class TestDEV006(unittest.TestCase):
    def test_targeted_continuity(self):
        rows = _csv("decoder_index_audit.csv")
        summaries = [r for r in rows if r["row_type"] == "video_summary"]
        self.assertEqual(len(summaries), 8)
        self.assertTrue(all(r["interpretation"].startswith("CONTINUITY_PROVEN") for r in summaries))
        idx = {(r["video"].split("_")[1], r["declared_index"]): r for r in rows if r["row_type"] == "index"}
        for client, f in (("client008", "127"), ("client023", "247")):
            self.assertEqual(idx[(client, f)]["opencv_status"], "READ_FAILED")
            self.assertEqual(idx[(client, f)]["pyav_status"], "ERROR")
            nxt = idx[(client, str(int(f) + 1))]
            self.assertEqual((nxt["opencv_status"], nxt["best_offset"]), ("OK", "0"))
            self.assertEqual(nxt["reported_msec_after"], nxt["pyav_pts_msec"])
            self.assertEqual(float(nxt["reported_pos_after"]), float(f) + 1)   # POS lags: shows it is not trusted

    def test_global_all_consistent(self):
        rows = _csv("decoder_index_global.csv")
        self.assertEqual(len(rows), len([v for v in _videos() if v["source_kind"] == "video_file"]))
        self.assertTrue(all(r["verdict"] == "INDEX_TIMESTAMPS_CONSISTENT" for r in rows))
        for r in rows:
            if int(r["n_failed"]) and not int(r["failed_inside_stream"]):
                self.assertEqual(int(r["n_failed"]), int(r["declared"]) - int(r["pyav_packets"]))

    def test_problem_videos_never_sample_invalid_index(self):
        for v in _videos("msu_mfsd"):
            if v["decode_status"] != "OK":
                self.assertTrue(set(json.loads(v["sampled_frame_indices"])).isdisjoint({127, 247}))


class TestCasia(unittest.TestCase):
    def test_disk_reaudit(self):
        a = json.loads((AUDIT / "casia_source_audit.json").read_text())
        self.assertEqual(a["n_unparsed"], 0)
        self.assertEqual(a["n_manifest_mismatches"], 0)
        self.assertEqual(a["identity_ids"], list(range(1, len(a["train_subject_numbers"]) + len(a["test_subject_numbers"]) + 1)))
        self.assertTrue(a["all_subjects_have_12_codes"])
        n = len(a["identity_ids"])
        self.assertEqual((a["live_sequences"], a["spoof_sequences"]), (3 * n, 9 * n))
        self.assertEqual(set(a["spoof_macro_counts"]), {"print", "replay"})
        self.assertEqual(a["folder_disagreements"], a["folder_x_code"].get("spoof/HR_1"))
        self.assertTrue(all(k.startswith(("canonical:(112, 112, 8, 'rgb')", "derived:(112, 112, 8, 'rgb')"))
                            for k in a["png_header_summary"]))

    def test_hr1_live_everywhere(self):
        hr1 = [v for v in _videos("casia_fasd") if json.loads(v["native_meta_json"])["video_code"] == "HR_1"]
        self.assertTrue(hr1)
        self.assertTrue(all(v["label_binary"] == 0 and v["attack_raw"] is None for v in hr1))

    def test_resize_audit_present_and_not_tuning(self):
        rows = _csv("CASIA_RESIZE_FREQUENCY_AUDIT.csv")
        self.assertEqual(len(rows), sum(1 for v in _videos("casia_fasd")) * 8)
        self.assertTrue(all((r["src_h"], r["src_w"]) == ("112", "112") for r in rows))
        self.assertIn("DIAGNOSTIC ONLY", (AUDIT / "CASIA_RESIZE_FREQUENCY_AUDIT.md").read_text())


class TestSiW(unittest.TestCase):
    def test_paper_trace(self):
        amap = yaml.safe_load((ROOT / "configs/frozen/attack_map_v1.yaml").read_text())
        paper = amap["mapped"]["siwmv2"]["Paper"]
        self.assertEqual(paper["attack_macro"], "print")
        self.assertTrue(any("'Print': 'Paper'" in t and "8667dbc" in t for t in paper["trace"]))
        man = json.loads((AUDIT / "siw_subject_recovery/official_repo_manifest.json").read_text())
        self.assertEqual(man["pinned_commit"], "8667dbcd316b38141729c057adf7517fe0602608")
        self.assertTrue(man["files"]["source_SiW_Mv2/config_siwm.py"].startswith("ebe37e87"))
        cache = AUDIT / "siw_subject_recovery/official_repo_cache/source_SiW_Mv2/config_siwm.py"
        if cache.exists():   # cache is git-ignored; verify content when present
            self.assertIn("'Print': 'Paper'", cache.read_text())

    def test_candidate_mapping_no_pseudo_ids(self):
        rows = _csv("siw_subject_mapping_candidate.csv")
        vids = {v["video_id"] for v in _videos("siwmv2")}
        self.assertEqual({r["video_id"] for r in rows}, vids)
        allowed = {"AUTHORITATIVE", "SUPPORTED", "AMBIGUOUS", "CONTRADICTED", "NO_EVIDENCE"}
        for r in rows:
            self.assertIn(r["confidence_class"], allowed)
            if r["confidence_class"] == "NO_EVIDENCE":
                self.assertEqual(r["candidate_subject_id"], "")
            self.assertEqual(r["protocol_name"], Path(r["local_path"]).stem)
        self.assertTrue(all(v["subject_id_raw"] is None for v in _videos("siwmv2")))   # no AdaFace-made IDs

    def test_duplicates_preserved(self):
        rows = _csv("SIW_DUPLICATE_VIDEO_AUDIT.csv")
        files = {r["rel_path"]: r["sha256"] for r in pq.read_table(ROOT / "manifests/raw_file_index.parquet").to_pylist()
                 if r["dataset"] == "siwmv2"}
        vids = {v["video_id"] for v in _videos("siwmv2")}
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertIn(r["path_A"], vids)
            self.assertIn(r["path_B"], vids)
            self.assertEqual(files[r["path_A"]], files[r["path_B"]])
            self.assertEqual(files[r["path_A"]], r["sha256"])
            self.assertIn(r["classification"], {"KNOWN_REDUNDANT_COPY", "SAME_CONTENT_SAME_IDENTITY",
                                                "SAME_CONTENT_DIFFERENT_METADATA", "UNRESOLVED_DUPLICATE"})


class TestReadiness(unittest.TestCase):
    def test_matrix(self):
        rows = {r["dataset"]: r for r in _csv("DATASET_READINESS_MATRIX.csv")}
        self.assertEqual(set(rows), {"casia_fasd", "msu_mfsd", "siwmv2"})
        summary = {r["dataset"]: r for r in _csv("dataset_inventory_summary.csv")}
        for ds, r in rows.items():
            missing = int(summary[ds]["videos_missing_subject_id"])
            self.assertEqual(r["subject_id_ready"], "NO" if missing else "YES", ds)
            if missing:   # missing subjects: blocked unless the frozen policy + an APPROVED deviation allow it
                policy = yaml.safe_load((ROOT / "configs/frozen/dataset_protocol_policy_v1.yaml").read_text()) \
                    if (ROOT / "configs/frozen/dataset_protocol_policy_v1.yaml").exists() else None
                dev = (AUDIT / "deviation_report.md").read_text()
                allowed = (policy is not None and policy["datasets"][ds]["subject_required"] is False
                           and "| DEV-012 | APPROVED |" in dev)
                self.assertEqual(r["M3_ready"], "READY_WITH_APPROVED_VIDEO_DISJOINT_DEVIATION" if allowed
                                 else "BLOCKED_BY_MISSING_SUBJECT_ID")
            unmapped = int(summary[ds]["unmapped_attack_tokens"])
            self.assertEqual(r["attack_macro_ready"], "NO" if unmapped else "YES")
            for col in ("evidence_file",):
                for p in r[col].split(";"):
                    self.assertTrue((ROOT / p).exists(), p)
        self.assertNotEqual(rows["casia_fasd"]["M2_ready"], "READY")   # 112 px repack: adaptation pending


class TestNoM2M3Artifacts(unittest.TestCase):
    def test_nothing_generated(self):
        for rel in ("data/processed", "cache", "runs", "probes", "downstream"):
            files = [p for p in (ROOT / rel).rglob("*") if p.is_file() and p.name != ".gitkeep"]
            self.assertEqual(files, [], rel)
        parquet = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.parquet") if not ({".git", ".venv", ".venv-audit"} & set(p.parts))}
        self.assertFalse(any("split" in p or "pair" in p for p in parquet))
        diag = ROOT / "outputs/audit/siw_subject_recovery"
        self.assertFalse(any(p.suffix in (".npy", ".npz", ".pt") for p in diag.rglob("*")))


if __name__ == "__main__":
    unittest.main()
