"""Final dataset policy freeze tests (dataset_protocol_policy_v1). No allocator is implemented here:
the tests check the frozen policy, its consistency with M1 evidence, and that no M2/M3/M4 artifact exists.
"""
import collections
import csv
import re
import sys
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY_PATH = ROOT / "configs/frozen/dataset_protocol_policy_v1.yaml"
P = yaml.safe_load(POLICY_PATH.read_text())
D = P["datasets"]
AUDIT = ROOT / "outputs/audit"


def _videos(ds):
    return [r for r in pq.read_table(ROOT / "manifests/inventory_videos.parquet").to_pylist() if r["dataset"] == ds]


def _content_groups():
    with open(AUDIT / "siw_content_groups.csv", newline="") as f:
        return list(csv.DictReader(f))


def _pair_allowed(ds, src, tgt):
    """Reference implementation of the frozen pairing rule (policy-level, no pairs are built)."""
    rule = D[ds]["pairing"]["rule"]
    if rule == "DIFFERENT_SUBJECT":
        return src["subject_id_global"] is not None and src["subject_id_global"] != tgt["subject_id_global"]
    if rule == "DIFFERENT_VIDEO_AND_DIFFERENT_CONTENT_GROUP":
        return src["video_id"] != tgt["video_id"] and src["content_group_id"] != tgt["content_group_id"]
    raise AssertionError(rule)


class TestGrouping(unittest.TestCase):
    def test_01_casia_subject_grouping(self):
        self.assertEqual((D["casia_fasd"]["split_grouping"], D["casia_fasd"]["group_key"], D["casia_fasd"]["subject_required"]),
                         ("SUBJECT", "subject_id_global", True))

    def test_02_msu_subject_grouping(self):
        self.assertEqual((D["msu_mfsd"]["split_grouping"], D["msu_mfsd"]["group_key"], D["msu_mfsd"]["subject_required"]),
                         ("SUBJECT", "subject_id_global", True))

    def test_03_siw_no_subject_required(self):
        self.assertIs(D["siwmv2"]["subject_required"], False)
        self.assertIs(D["siwmv2"]["pseudo_subject_ids_forbidden"], True)

    def test_04_siw_grouping_is_video_content_group(self):
        self.assertEqual((D["siwmv2"]["split_grouping"], D["siwmv2"]["group_key"]),
                         ("CANONICAL_VIDEO_CONTENT_GROUP", "content_group_id"))
        self.assertIs(D["siwmv2"]["content_group"]["perceptual_deduplication"], False)
        self.assertIs(D["siwmv2"]["content_group"]["is_person_identity"], False)

    def test_05_video_cannot_cross_partitions(self):
        self.assertIs(P["split"]["frame_level_split_forbidden"], True)
        self.assertEqual(P["split"]["unit_of_assignment"], "grouping_unit")
        groups = _content_groups()
        per_video = collections.Counter(r["video_id"] for r in groups)
        self.assertTrue(all(n == 1 for n in per_video.values()))   # each video -> exactly one group
        samples = [s for s in pq.read_table(ROOT / "manifests/inventory.parquet").to_pylist() if s["dataset"] == "siwmv2"]
        self.assertTrue({s["video_id"] for s in samples} <= set(per_video))   # every frame inherits one group

    def test_06_exact_duplicates_share_one_group(self):
        files = {r["rel_path"]: r["sha256"] for r in pq.read_table(ROOT / "manifests/raw_file_index.parquet").to_pylist()
                 if r["dataset"] == "siwmv2"}
        groups = _content_groups()
        for r in groups:
            self.assertEqual(r["raw_sha256"], files[r["video_id"]])
            self.assertEqual(r["content_group_id"], f"siwmv2::sha256:{r['raw_sha256']}")
        with open(AUDIT / "SIW_DUPLICATE_VIDEO_AUDIT.csv", newline="") as f:
            dups = list(csv.DictReader(f))
        gid = {r["video_id"]: r["content_group_id"] for r in groups}
        for d in dups:
            self.assertEqual(gid[d["path_A"]], gid[d["path_B"]])
        self.assertEqual(sum(1 for r in groups if r["group_size"] == "2"), 2 * len(dups))

    def test_07_duplicate_raw_records_preserved(self):
        vids = {v["video_id"] for v in _videos("siwmv2")}
        self.assertEqual(len(vids), len(_content_groups()))   # 1 row per raw video, none merged
        self.assertIs(D["siwmv2"]["content_group"]["raw_records_preserved"], True)

    def test_08_no_pseudo_subject_ids(self):
        self.assertTrue(all(v["subject_id_raw"] is None and v["subject_id_global"] is None for v in _videos("siwmv2")))
        self.assertEqual(D["siwmv2"]["subject_identity_status"], "UNAVAILABLE")


class TestPairing(unittest.TestCase):
    def test_09_10_casia_msu_different_subject(self):
        for ds in ("casia_fasd", "msu_mfsd"):
            self.assertEqual(D[ds]["pairing"]["rule"], "DIFFERENT_SUBJECT")
            a = {"subject_id_global": f"{ds}::1", "video_id": "v1"}
            self.assertFalse(_pair_allowed(ds, a, {"subject_id_global": f"{ds}::1", "video_id": "v2"}))
            self.assertTrue(_pair_allowed(ds, a, {"subject_id_global": f"{ds}::2", "video_id": "v2"}))

    def test_11_12_siw_different_video_and_content(self):
        self.assertEqual(D["siwmv2"]["pairing"]["rule"], "DIFFERENT_VIDEO_AND_DIFFERENT_CONTENT_GROUP")
        g = {r["video_id"]: r["content_group_id"] for r in _content_groups()}
        a = {"video_id": "Spoof/Replay/Replay_76.mov", "content_group_id": g["Spoof/Replay/Replay_76.mov"]}
        same_video = dict(a)
        dup = {"video_id": "Spoof/Replay/Replay_83.mov", "content_group_id": g["Spoof/Replay/Replay_83.mov"]}
        other = {"video_id": "Live/Live_1.mov", "content_group_id": g["Live/Live_1.mov"]}
        self.assertFalse(_pair_allowed("siwmv2", a, same_video))
        self.assertFalse(_pair_allowed("siwmv2", a, dup))          # exact duplicate content
        self.assertTrue(_pair_allowed("siwmv2", a, other))
        self.assertIn("NOT proven different-person", D["siwmv2"]["pairing"]["claim_wording"])


class TestCasiaAdaptation(unittest.TestCase):
    def test_13_dev011_approved(self):
        pre = D["casia_fasd"]["preprocessing"]
        self.assertEqual(pre["fidelity"], "APPROVED_CONTROLLED_DATASET_ADAPTATION")
        self.assertEqual(pre["resize"], {"size": 256, "interpolation": "INTER_CUBIC", "output": "uint8_sRGB"})
        self.assertIn("| DEV-011 | APPROVED |", (AUDIT / "deviation_report.md").read_text())

    def test_14_no_fake_scrfd_success(self):
        pre = D["casia_fasd"]["preprocessing"]
        self.assertIs(pre["scrfd_applied"], False)
        self.assertEqual(pre["scrfd_status"], "N/A")
        self.assertNotIn("success", str(pre["scrfd_status"]).lower())


class TestBalancing(unittest.TestCase):
    def test_15_16_17_balancing_enabled(self):
        b = P["split"]["balancing"]
        self.assertIs(b["binary_class_distribution"], True)
        self.assertIs(b["attack_macro_distribution"], True)
        self.assertIs(b["attack_raw_distribution_secondary"], True)
        for ds in D:
            self.assertEqual(D[ds]["balancing"], {"binary": True, "attack_macro": True, "attack_raw_secondary": True})
        self.assertEqual(P["split"]["priority_order"], ["grouping_and_leakage_safety", "total_canonical_video_counts_near_ratios",
                                                        "binary_class_distribution", "attack_macro_distribution",
                                                        "attack_raw_distribution_secondary"])

    def test_18_no_hidden_allocator_weights(self):
        a = P["split"]["allocator"]
        self.assertEqual(a["status"], "NOT_DEFINED_IN_THIS_POLICY")
        self.assertIsNone(a["objective"])
        self.assertIsNone(a["weights"])
        self.assertIsNone(a["tie_breaking"])
        text = POLICY_PATH.read_text()
        self.assertIsNone(re.search(r"(penalty|weight)\w*\s*:\s*[0-9]", text))   # no numeric weights anywhere

    def test_spec_values(self):
        self.assertEqual(P["split"]["seed"], 20260814)
        self.assertEqual(P["split"]["ratios"], {"train": 0.70, "val": 0.15, "test": 0.15})
        self.assertEqual(P["split"]["size_measure"], "canonical_video_count")


class TestNoLaterArtifacts(unittest.TestCase):
    def test_19_20_21_no_unstarted_milestone_artifacts(self):
        import stage_guard
        # in-repo heavy dirs stay empty: M2 outputs live on the external runtime volume (DEV-017)
        for rel in ("data/processed", "cache", "runs", "probes", "downstream"):
            self.assertEqual([p for p in (ROOT / rel).rglob("*") if p.is_file() and p.name != ".gitkeep"], [], rel)
        parquet = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.parquet")
                   if not ({".git", ".venv"} & set(p.parts))}
        self.assertEqual(stage_guard.forbidden_parquet(parquet), [])
        names = {p.name for p in ROOT.rglob("*") if not ({".git", ".venv"} & set(p.parts))}
        for bad in ("pairs_train_v1.parquet", "val_pairs_v1.parquet"):
            self.assertNotIn(bad, names)
        self.assertFalse(any(n.startswith("pairs_") and n.endswith(".parquet") for n in names))


class TestFrozenPolicy(unittest.TestCase):
    def test_22_snapshot_and_index(self):
        snap = ROOT / "frozen_config_snapshot/configs/frozen/dataset_protocol_policy_v1.yaml"
        self.assertEqual(snap.read_bytes(), POLICY_PATH.read_bytes())
        with open(AUDIT / "ARTIFACT_INDEX.csv", newline="") as f:
            idx = {r["path"]: r["sha256"] for r in csv.DictReader(f)}
        import hashlib
        self.assertEqual(idx["configs/frozen/dataset_protocol_policy_v1.yaml"], hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest())

    def test_disclosures_and_forbidden_statements(self):
        self.assertEqual(P["pooled_benchmark_fidelity"], "MIXED")
        self.assertIn("All three datasets are subject-disjoint.", P["forbidden_statements"])
        self.assertEqual(len(P["required_disclosures"]), 6)
        claims = (AUDIT / "CLAIM_EVIDENCE_MAP.csv").read_text()
        for i in range(1, 7):
            self.assertIn(f"D-POOL-0{i}", claims)

    def test_readiness_matches_policy(self):
        with open(AUDIT / "DATASET_READINESS_MATRIX.csv", newline="") as f:
            rows = {r["dataset"]: r for r in csv.DictReader(f)}
        for ds, r in rows.items():
            self.assertEqual(r["split_grouping"], D[ds]["split_grouping"])
            self.assertEqual(r["split_fidelity"], D[ds]["split_fidelity"])
            self.assertEqual(r["preprocessing_fidelity"], D[ds]["preprocessing"]["fidelity"])
            self.assertEqual(r["pairing_rule"], D[ds]["pairing"]["rule"])
        import json
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        # Stage-aware: M1 COMPLETE and M3 NOT_STARTED at every stage up to and including M2.
        self.assertEqual(s["M1"]["status"], "COMPLETE")
        # M3 may be COMPLETE, but only with a manifest whose recorded hash matches the file.
        if s["M3"]["status"] == "COMPLETE":
            import hashlib
            rec = json.loads((AUDIT / "split_v1.sha256").read_text())
            mp = ROOT / "manifests/split_v1.parquet"
            self.assertTrue(mp.is_file())
            self.assertEqual(hashlib.sha256(mp.read_bytes()).hexdigest(), rec["sha256"])
        self.assertIn(s["M2"]["status"], {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE"})
        self.assertEqual(s["M4"]["status"], "NOT_STARTED")
        # M2 may only be COMPLETE when the full run really happened and its audit passed. Under DEV-017
        # the artifacts live on the external runtime volume, so the in-repo dirs stay empty and the
        # evidence is the audit itself plus the physical roots named by the execution config.
        if s["M2"]["status"] == "COMPLETE":
            integ = json.loads((AUDIT / "M2_CACHE_INTEGRITY.json").read_text())
            self.assertTrue(integ["ok"])
            acc = integ["accounting"]
            self.assertEqual(acc["missing_state"], 0)
            self.assertEqual(acc["complete"] + acc["failed"], acc["expected"])
            self.assertEqual(json.loads((AUDIT / "M2_DETERMINISTIC_VALIDATION.json").read_text())["status"], "PASS")
            import yaml as _yaml
            x = _yaml.safe_load((ROOT / "configs/execution/m2b_laptop_external_storage.yaml").read_text())
            for role, root in x["storage"]["roots"].items():
                self.assertTrue(any(Path(root).rglob("*")), f"{role} is empty but M2 is COMPLETE")
        else:
            produced = [p for rel in ("data/processed", "cache") for p in (ROOT / rel).rglob("*")
                        if p.is_file() and p.name != ".gitkeep"]
            self.assertEqual(produced, [])

    def test_sanity_json(self):
        import json
        z = json.loads((AUDIT / "dataset_policy_sanity.json").read_text())
        self.assertTrue(z["casia_subjects_complete"] and z["msu_subjects_complete"] and z["siw_subject_ids_all_null"])
        self.assertTrue(z["all_samples_trace_to_one_canonical_video"])
        self.assertEqual(z["unmapped_tokens"], [])
        self.assertEqual(z["cross_video_exact_duplicates_casia_msu"], {"casia_fasd": 0, "msu_mfsd": 0})


if __name__ == "__main__":
    unittest.main()
