"""Amendment A1 — Track-A fair identity-free main track.

Asserts the amendment, the frozen Track-A contracts, the two adapted methods' relations and the
identity firewall. Nothing here trains, generates or touches TEST.
"""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P      # noqa: E402
from gpatbench.pairs import execute as E     # noqa: E402
from gpatbench.pairs import track_a as T     # noqa: E402

AUDIT = ROOT / "outputs/audit"
AMENDMENT = ROOT / "docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A1_Fair_IDFree_Main_Track.md"
AMENDMENT_SHA = "03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472"

FROZEN_COMMON = {
    "manifests/pair_train_stats_v1.json":
        "a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50",
    "manifests/pairs_train_v1.parquet":
        "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243",
    "manifests/val_pairs_v1.parquet":
        "84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872",
    "manifests/split_v1.parquet":
        "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d",
}
DSDG_PIN = "16b793a7564a4b9308cf94e62bdb2ffacb3a725a"
DIFFFAS_PIN = "23f40519ec25a833ebc06842aa6fbab74fad4d15"

_C = {}


def fair():
    return _C.setdefault("fair", yaml.safe_load(T.FAIR_TRACK_CONFIG.read_text()))


def dsdg_cfg():
    return _C.setdefault("dsdg", yaml.safe_load(T.DSDG_CONFIG.read_text()))


def diff_cfg():
    return _C.setdefault("diff", yaml.safe_load(T.DIFFFAS_CONFIG.read_text()))


def pins():
    return _C.setdefault("pins", json.loads(T.SOURCE_PINS.read_text()))


def difffas_rows():
    return _C.setdefault("dfr", pq.read_table(T.DIFFFAS_MANIFEST).to_pylist())


def split_rows():
    return _C.setdefault("sp", pq.read_table(E.SPLIT_MANIFEST).to_pylist())


def population():
    return _C.setdefault("pop", T.load_track_a_population())


# ------------------------------------------------------------------ Amendment A1
class TestAmendment(unittest.TestCase):
    def test_amendment_exists_and_hash_is_frozen(self):
        self.assertTrue(AMENDMENT.is_file())
        self.assertEqual(E.sha256_file(AMENDMENT), AMENDMENT_SHA)
        rec = json.loads((AUDIT / "amendment_a1.sha256").read_text())
        self.assertEqual(rec["sha256"], AMENDMENT_SHA)
        self.assertEqual(fair()["amendment_sha256"], AMENDMENT_SHA)

    def test_original_specification_was_not_modified(self):
        spec = ROOT / "docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx"
        self.assertEqual(
            E.sha256_file(spec),
            "f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e")
        self.assertIs(fair()["original_spec_modified"], False)

    def test_amendment_states_the_original_and_amended_m4_rules(self):
        import re
        text = AMENDMENT.read_text()
        flat = re.sub(r"[*`\s]+", " ", text)           # markdown emphasis and wrapping are noise here
        self.assertIn("remains historically authoritative", flat)
        self.assertIn("Original M4 rule (spec): M4 requires the common pair manifest and the "
                      "native pair manifests", flat)
        self.assertIn("Amended main-track M4 rule (A1)", flat)
        self.assertIn("deferred to M6", flat)
        for phrase in ("controlled benchmark adaptation", "identity-free fair-track variant",
                       "source-based architecture/code-path adaptation"):
            self.assertIn(phrase, flat, phrase)

    def test_decision_was_taken_before_any_result(self):
        self.assertIn("before", fair()["decision_timing"].lower())
        for pat in ("*.ckpt", "*.pt", "*.pth"):
            found = [p for p in ROOT.rglob(pat)
                     if not ({".git", ".venv", ".venv-audit"} & set(p.parts))
                     and "third_party/source_cache" not in p.as_posix()]
            self.assertEqual(found, [], f"a trained artifact exists: {pat}")
        for d in ("runs", "probes", "downstream"):
            files = [p for p in (ROOT / d).rglob("*") if p.is_file() and p.name != ".gitkeep"]
            self.assertEqual(files, [], d)


# ------------------------------------------------------------------ Track-A contract
class TestTrackAContract(unittest.TestCase):
    def test_datasets_are_exactly_casia_msu_siw(self):
        self.assertEqual(fair()["track_a"]["datasets"], ["casia_fasd", "msu_mfsd", "siwmv2"])
        self.assertIs(fair()["track_a"]["dataset_dropping_allowed"], False)
        self.assertEqual(sorted(T.TRACK_A_DATASETS), sorted(fair()["track_a"]["datasets"]))

    def test_identity_supervision_is_forbidden(self):
        a = fair()["track_a"]
        self.assertEqual(a["identity_supervision"], "FORBIDDEN")
        for f in ("subject_id_global", "pseudo_subject_id", "person_label",
                  "identity_class_label_derived_from_dataset_metadata"):
            self.assertIn(f, a["forbidden_inputs"])
        for scope in ("model_input", "loss", "class_target", "sampling_rule"):
            self.assertIn(scope, a["forbidden_inputs_scope"])

    def test_attack_type_supervision_is_forbidden(self):
        a = fair()["track_a"]
        self.assertEqual(a["attack_type_supervision"], "FORBIDDEN")
        for f in ("attack_macro", "attack_raw", "dataset_id_as_semantic_class"):
            self.assertIn(f, a["forbidden_class_targets"])

    def test_synthetic_budget_is_identical_for_every_method(self):
        self.assertEqual(fair()["track_a"]["n_syn_intended"], 8838)
        rows = list(csv.DictReader((AUDIT / "M4_TRACK_A_METHOD_MATRIX.csv").open()))
        self.assertEqual(len(rows), 8)
        for r in rows:
            self.assertEqual(int(r["n_syn_intended"]), 8838, r["method_id"])
            self.assertEqual(r["train_datasets"], "casia_fasd+msu_mfsd+siwmv2", r["method_id"])
            self.assertEqual(r["val_datasets"], "casia_fasd+msu_mfsd+siwmv2", r["method_id"])
            self.assertEqual(r["uses_subject_id"], "NO", r["method_id"])
            self.assertEqual(r["uses_attack_type"], "NO", r["method_id"])
            self.assertIn(r["method_id"], fair()["track_a"]["methods"])

    def test_identity_is_never_fabricated(self):
        pol = fair()["identity_policy"]
        self.assertEqual(pol["fabrication"], "FORBIDDEN")
        for f in ("video_id_as_person_identity", "content_group_id_as_person_identity",
                  "pseudo_subject_id", "face_embedding_clustering_as_subject_id",
                  "filename_derived_identity", "manually_guessed_identity",
                  "DEV-013_as_a_same_person_guarantee"):
            self.assertIn(f, pol["forbidden_identity_substitutes"])
        self.assertEqual(pol["correct_resolution"], "REMOVE_THE_IDENTITY_DEPENDENT_ASSUMPTION")

    def test_claim_limits_are_frozen(self):
        a = fair()["track_a"]
        joined = " ".join(a["claims_forbidden"])
        self.assertIn("reproduces official DSDG training exactly", joined)
        self.assertIn("is native DiffFAS", joined)
        self.assertIn("same pooled", a["claim_supported"])

    def test_configs_have_no_unresolved_field(self):
        bad = []

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif node is None or (isinstance(node, str) and node.strip().upper() in
                                  {"TODO", "TBD", "OWNER_DECISION_REQUIRED", "NULL"}):
                bad.append(path)

        for name, cfg in (("fair", fair()), ("dsdg", dsdg_cfg()), ("difffas", diff_cfg())):
            walk(cfg, name)
        self.assertEqual(bad, [], f"unresolved fields: {bad}")

    def test_snapshots_are_byte_identical(self):
        for cfg in (T.FAIR_TRACK_CONFIG, T.DSDG_CONFIG, T.DIFFFAS_CONFIG):
            snap = ROOT / "frozen_config_snapshot" / cfg.relative_to(ROOT)
            self.assertTrue(snap.is_file(), snap)
            self.assertEqual(snap.read_bytes(), cfg.read_bytes())

    def test_m6_training_hyperparameters_are_not_frozen_here(self):
        for cfg in (dsdg_cfg(), diff_cfg()):
            self.assertIs(cfg["m6_training_hyperparameters_frozen_here"], False)


# ------------------------------------------------------------------ source pins
class TestSourcePins(unittest.TestCase):
    def test_dsdg_pin_is_exact(self):
        s = pins()["sources"]["dsdg"]
        self.assertEqual(s["pinned_commit"], DSDG_PIN)
        self.assertEqual(s["repository"], "https://github.com/JDAI-CV/FaceX-Zoo")
        self.assertIn("FaceX-Zoo", s["commit_verified_from_remote"])
        self.assertEqual(dsdg_cfg()["official_source"]["pinned_commit"], DSDG_PIN)

    def test_difffas_pin_is_exact(self):
        s = pins()["sources"]["difffas"]
        self.assertEqual(s["pinned_commit"], DIFFFAS_PIN)
        self.assertEqual(s["repository"], "https://github.com/murphytju/DiffFAS")
        self.assertIn("DiffFAS", s["commit_verified_from_remote"])
        self.assertEqual(diff_cfg()["official_source"]["pinned_commit"], DIFFFAS_PIN)

    def test_registry_records_the_same_pins(self):
        reg = {m["method_id"]: m for m in
               yaml.safe_load((ROOT / "third_party/registry.yaml").read_text())["methods"]}
        for mid in ("E06a", "E06b", "E06c"):
            self.assertEqual(reg[mid]["pinned_commit"], DSDG_PIN, mid)
        for mid in ("E07a", "E07b", "E07c"):
            self.assertEqual(reg[mid]["pinned_commit"], DIFFFAS_PIN, mid)

    def test_no_model_weights_were_downloaded(self):
        for s in pins()["sources"].values():
            self.assertIs(s["model_weights_downloaded"], False)
            self.assertEqual(s["binary_weight_files_present_now"], [])
        self.assertEqual(pins()["verification_failures"], [])

    def test_cited_files_carry_content_hashes(self):
        for key, s in pins()["sources"].items():
            self.assertTrue(s["cited_files"], key)
            for rel, meta in s["cited_files"].items():
                self.assertEqual(len(meta["sha256"]), 64, f"{key}/{rel}")
                self.assertGreater(meta["size_bytes"], 0)

    def test_source_checkouts_are_not_committed(self):
        gitignore = (ROOT / ".gitignore").read_text()
        self.assertIn("third_party/source_cache/", gitignore)


# ------------------------------------------------------------------ DSDG-BIN-IDFREE (DEV-020)
class TestDsdgIdFree(unittest.TestCase):
    def test_classification_is_controlled_adaptation(self):
        c = dsdg_cfg()
        self.assertEqual(c["classification"], "CONTROLLED_ADAPTATION")
        self.assertIn("FAITHFUL_OFFICIAL", c["not_classification"])
        self.assertEqual(c["deviation"], "DEV-020")

    def test_official_relation_is_recorded_as_same_subject_online_random(self):
        o = dsdg_cfg()["official_relation"]
        self.assertEqual(o["semantics"], "SAME_SUBJECT_ONLINE_RANDOM")
        self.assertIn("random.choice", o["live_selection"])
        self.assertIn("user id", o["label_meaning"])
        self.assertIs(o["materialized_pair_list"], False)

    def test_lambda_pair_is_zero(self):
        self.assertEqual(dsdg_cfg()["losses"]["lambda_pair"], 0.0)
        self.assertEqual(dsdg_cfg()["losses"]["lambda_pair_official_default"], 0.5)

    def test_only_the_identity_equality_loss_was_removed(self):
        L = dsdg_cfg()["losses"]
        self.assertEqual(L["removed"], ["loss_pair"])
        for term in ("loss_rec", "loss_kl", "loss_mmd", "loss_ip", "loss_cls", "loss_ort"):
            self.assertIn(term, L["retained"], term)
            self.assertEqual(L["retained"][term]["identity_requirement"], "NONE", term)

    def test_binary_degeneracy_is_disclosed_not_worked_around(self):
        b = dsdg_cfg()["binary_supervision"]
        self.assertIs(b["spoof_type_collapse"], True)
        self.assertEqual(b["attack_type_arg"], 1)
        self.assertEqual(b["attack_macro_as_target"], "FORBIDDEN")
        self.assertEqual(b["attack_raw_as_target"], "FORBIDDEN")
        self.assertIs(b["substitute_supervision_invented"], False)
        self.assertIn("zero gradient", b["degeneracy_disclosure"])

    def test_relation_is_the_common_pair_manifest_itself(self):
        r = dsdg_cfg()["training_relation"]
        self.assertEqual(r["decision"], "USE_COMMON_PAIR_MANIFEST_DIRECTLY")
        self.assertIs(r["separate_manifest_created"], False)
        self.assertFalse((ROOT / "manifests/dsdg_bin_idfree_train_v1.parquet").exists())
        self.assertEqual(r["authoritative_manifest_sha256"],
                         FROZEN_COMMON["manifests/pairs_train_v1.parquet"])

    def test_rows_and_dataset_coverage(self):
        rows = T.dsdg_training_relation(population())
        self.assertEqual(len(rows), 8838)
        counts = {}
        for r in rows:
            counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
        self.assertEqual(counts, T.EXPECTED_BY_DATASET)
        self.assertGreater(counts["siwmv2"], 0, "SiW must participate in Track A")

    def test_adapter_interface_has_no_identity_field(self):
        rows = T.dsdg_training_relation(population())
        for k in rows[0]:
            self.assertNotIn("subject", k)
            self.assertNotIn("identity", k)
            self.assertNotIn("person", k)

    def test_relation_is_unchanged_when_subject_ids_are_nulled(self):
        pop = population()
        ref = T.dsdg_training_relation(pop)
        nulled = dict(pop)
        nulled["common_pairs"] = [{**r, "source_subject": None, "target_subject": None}
                                  for r in pop["common_pairs"]]
        self.assertEqual(T.dsdg_training_relation(nulled), ref)

    def test_common_pair_inheritance_is_exact(self):
        pairs = {r["pair_id"]: r for r in pq.read_table(E.MANIFEST_PATH["TRAIN"]).to_pylist()}
        for r in T.dsdg_training_relation(population()):
            src = pairs[r["pair_id"]]
            for k in ("dataset", "source_spoof_id", "target_live_id", "split", "seed",
                      "source_sha256", "target_sha256"):
                self.assertEqual(r[k], src[k], f"{r['pair_id']}/{k}")


# ------------------------------------------------------------------ DIFFFAS-BIN-IDFREE (DEV-021)
class TestDiffFasIdFree(unittest.TestCase):
    def test_classification_uses_the_official_unpaired_code_path(self):
        c = diff_cfg()
        self.assertEqual(c["classification"],
                         "CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH")
        self.assertIn("FAITHFUL_NATIVE", c["not_classification"])
        self.assertEqual(c["deviation"], "DEV-021")
        self.assertIs(c["official_code_path"]["official_code_modified"], False)
        self.assertIs(c["official_code_path"]["use_pair"], False)

    def test_use_pair_false_branch_is_proven_numerically(self):
        rep = json.loads((AUDIT / "M4_DIFFFAS_CONTENT_INERTNESS.json").read_text())
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["pinned_commit"], DIFFFAS_PIN)
        self.assertEqual(rep["content_training_role"], "INERT_API_PLACEHOLDER")
        c = rep["checks"]
        self.assertTrue(c["unpaired_model_input_is_x_t_only"])
        self.assertTrue(c["paired_model_input_concatenates_content"])
        self.assertTrue(c["conditioning_is_style_not_content"])
        self.assertEqual(rep["cases"]["use_pair=False"]["model_input_channels"], 3)
        self.assertEqual(rep["cases"]["use_pair=True"]["model_input_channels"], 6)
        self.assertEqual(rep["diffusion_py_sha256"],
                         pins()["sources"]["difffas"]["cited_files"]["diffusion.py"]["sha256"])

    def test_content_is_independent_of_the_loss_under_use_pair_false(self):
        rep = json.loads((AUDIT / "M4_DIFFFAS_CONTENT_INERTNESS.json").read_text())
        off = rep["cases"]["use_pair=False"]
        self.assertTrue(off["model_input_identical"])
        self.assertTrue(off["loss_identical"])
        self.assertTrue(off["mse_identical"])
        self.assertEqual(off["max_abs_loss_diff"], 0.0)
        # negative control: the same probe must detect the dependency when it exists
        on = rep["cases"]["use_pair=True"]
        self.assertFalse(on["model_input_identical"])
        self.assertFalse(on["loss_identical"])
        self.assertTrue(rep["checks"]["test_can_detect_a_dependency"])

    def test_rows_and_dataset_coverage(self):
        rows = difffas_rows()
        self.assertEqual(len(rows), 8838)
        counts = {}
        for r in rows:
            counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
        self.assertEqual(counts, T.EXPECTED_BY_DATASET)
        self.assertGreater(counts["siwmv2"], 0, "SiW must participate in Track A")
        self.assertEqual(len({r["gt_spoof_id"] for r in rows}), 8838)

    def test_style_id_is_binary_and_attack_labels_never_appear(self):
        self.assertEqual({r["style_id"] for r in difffas_rows()}, {"SPOOF_BINARY"})
        self.assertEqual(diff_cfg()["style_supervision"]["attack_raw_as_class_target"], "FORBIDDEN")
        self.assertEqual(diff_cfg()["style_supervision"]["dataset_id_as_semantic_class"], "FORBIDDEN")
        for c in T.DIFFFAS_COLUMNS:
            self.assertNotIn("attack", c)

    def test_no_subject_dependency(self):
        for c in T.DIFFFAS_COLUMNS:
            self.assertNotIn("subject", c)
            self.assertNotIn("identity", c)
        f = diff_cfg()["identity_firewall"]
        for k in ("subject_id_global_consumed", "consumed_for_eligibility",
                  "consumed_for_selection", "consumed_for_loss", "consumed_for_model_input"):
            self.assertIs(f[k], False, k)
        self.assertIs(f["siw_participates_without_identity"], True)

    def test_guide_is_a_train_spoof_of_the_same_dataset(self):
        by_id = {r["sample_id"]: r for r in split_rows()}
        for r in difffas_rows():
            gt, guide = by_id[r["gt_spoof_id"]], by_id[r["guide_spoof_id"]]
            self.assertEqual(guide["split"], "TRAIN")
            self.assertEqual(int(guide["label_binary"]), 1)
            self.assertEqual(guide["dataset"], gt["dataset"])
            self.assertEqual(guide["dataset"], r["dataset"])

    def test_guide_ranking_uses_the_raw_byte_digest(self):
        sd = hashlib.sha256(
            b"gpatbench.trackA.difffas.guide.source.v1|gt|20260814").digest()
        self.assertEqual(T.guide_source_digest("gt"), sd)
        self.assertIsInstance(sd, bytes)
        self.assertEqual(len(sd), 32)
        cd = hashlib.sha256(
            sd + b"|gpatbench.trackA.difffas.guide.candidate.v1|g1").digest()
        self.assertEqual(T.guide_candidate_digest(sd, "g1"), cd)
        self.assertEqual(T.guide_rank_key("gt", "g1"),
                         (int.from_bytes(cd, "big", signed=False), "g1"))
        forbidden = hashlib.sha256(
            ("gpatbench.trackA.difffas.guide.candidate.v1|" + sd.hex() + "|g1").encode()).digest()
        self.assertNotEqual(cd, forbidden)

    def test_guide_selection_is_deterministic_on_real_rows(self):
        pop = population()
        pool = {}
        for r in pop["train_rows"]:
            if int(r["label_binary"]) == 1:
                pool.setdefault(r["dataset"], []).append(r)
        for ds in pool:
            pool[ds].sort(key=lambda r: r["sample_id"])
        by_id = {r["sample_id"]: r for r in pop["train_rows"]}
        salt = "gpatbench.m4_trackA_test.v1|"
        rows = difffas_rows()
        for ds in T.TRACK_A_DATASETS:
            sub = sorted([r for r in rows if r["dataset"] == ds],
                         key=lambda r: hashlib.sha256((salt + r["gt_spoof_id"]).encode()).hexdigest())
            for r in sub[:15]:
                gt = by_id[r["gt_spoof_id"]]
                elig = T.eligible_guides(gt, pool[ds])
                self.assertEqual(len(elig), r["eligible_guide_count"])
                self.assertEqual(T.choose_guide(gt, elig)["sample_id"], r["guide_spoof_id"])

    def test_self_guide_is_allowed_and_only_reported(self):
        cfg = diff_cfg()["guide_selection"]["self_guide"]
        self.assertIs(cfg["allowed"], True)
        self.assertEqual(cfg["post_hoc_adjustment"], "FORBIDDEN")
        rep = json.loads((AUDIT / "M4_IDFREE_MANIFEST_AUDIT.json").read_text())
        n = rep["difffas"]["self_guide_count"]
        self.assertEqual(n, sum(1 for r in difffas_rows() if r["self_guide"]))
        self.assertEqual(n, sum(1 for r in difffas_rows()
                                if r["guide_spoof_id"] == r["gt_spoof_id"]))

    def test_content_live_id_is_inherited_and_marked_inert(self):
        pairs = {r["source_spoof_id"]: r["target_live_id"]
                 for r in pq.read_table(E.MANIFEST_PATH["TRAIN"]).to_pylist()}
        for r in difffas_rows():
            self.assertEqual(r["content_live_id"], pairs[r["gt_spoof_id"]])
            self.assertEqual(r["content_training_role"], "INERT_WHEN_USE_PAIR_FALSE")
            self.assertIs(r["use_pair"], False)

    def test_manifest_hash_is_frozen(self):
        rec = json.loads((AUDIT / "difffas_bin_idfree_train_v1.sha256").read_text())
        self.assertEqual(rec["sha256"], E.sha256_file(T.DIFFFAS_MANIFEST))
        self.assertEqual(rec["rows"], 8838)
        self.assertEqual(rec["official_source_commit"], DIFFFAS_PIN)

    def test_generation_uses_the_common_pair_universe(self):
        g = diff_cfg()["generation"]
        self.assertEqual(g["pair_universe"], "manifests/pairs_train_v1.parquet")
        self.assertEqual(g["substitute_target_population"], "FORBIDDEN")
        self.assertIs(g["executed_in_this_pass"], False)


# ------------------------------------------------------------------ audits, Track B, integrity
class TestAuditsAndIntegrity(unittest.TestCase):
    def test_idfree_audit_passed_against_these_artifacts(self):
        rep = json.loads((AUDIT / "M4_IDFREE_MANIFEST_AUDIT.json").read_text())
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["failures"], [])
        self.assertEqual(rep["difffas"]["sha256"], E.sha256_file(T.DIFFFAS_MANIFEST))
        self.assertIs(rep["dsdg"]["unchanged_when_subject_ids_nulled"], True)

    def test_idfree_determinism_passed(self):
        rep = json.loads((AUDIT / "M4_IDFREE_DETERMINISM_COMPARE.json").read_text())
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["mismatches"], [])
        self.assertEqual(rep["authoritative_sha256"][T.DIFFFAS_MANIFEST.name],
                         E.sha256_file(T.DIFFFAS_MANIFEST))
        self.assertTrue(any(v["shuffle_seed"] is not None for v in rep["runs"].values()))
        self.assertGreater(len({v["pythonhashseed"] for v in rep["runs"].values()}), 1)
        for name, v in rep["runs"].items():
            for k in ("byte_identical", "rows_identical", "schema_identical",
                      "membership_identical", "track_pair_id_identical",
                      "dsdg_relation_identical"):
                self.assertTrue(v[k], f"{name}/{k}")

    def test_common_m4_artifacts_are_unchanged(self):
        for rel, sha in FROZEN_COMMON.items():
            self.assertEqual(E.sha256_file(ROOT / rel), sha, rel)

    def test_m2_and_m3_artifacts_are_unchanged(self):
        self.assertEqual(E.sha256_file(ROOT / "manifests/split_v1.parquet"),
                         FROZEN_COMMON["manifests/split_v1.parquet"])
        rec = json.loads((AUDIT / "split_v1.sha256").read_text())
        self.assertEqual(rec["sha256"], FROZEN_COMMON["manifests/split_v1.parquet"])
        self.assertEqual(E.sha256_file(ROOT / "configs/frozen/pairs_v1.yaml"),
                         "f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985")

    def test_track_b_still_exists_and_is_labelled_secondary(self):
        b = fair()["track_b"]
        self.assertEqual(b["role"], "SECONDARY")
        self.assertIs(b["is_primary_fairness_table"], False)
        self.assertIs(b["blocks_m5"], False)
        self.assertIs(b["blocks_main_comparison"], False)
        self.assertEqual(b["status"], "DEFERRED_TO_M6_SECONDARY_TRACK")
        for mid in ("E06b", "E07b"):
            self.assertIn(mid, b["methods"])
        self.assertEqual(b["dataset_coverage"]["siwmv2"], "NOT_INSTANTIABLE_MISSING_SUBJECT_ID")

    def test_track_b_is_never_mislabelled_as_the_main_fair_result(self):
        self.assertIn("track column", fair()["track_b"]["reporting_rule"])
        a, b = fair()["track_a"]["methods"], fair()["track_b"]["methods"]
        self.assertEqual(set(a) & set(b), set())
        matrix = list(csv.DictReader((AUDIT / "M4_TRACK_A_METHOD_MATRIX.csv").open()))
        for r in matrix:
            self.assertEqual(r["track"], "A_FAIR_COMMON_IDENTITY_FREE")
            self.assertNotIn(r["method_id"], b)

    def test_no_native_track_b_manifest_was_created(self):
        for name in ("dsdg_identity_pairs_v1.parquet", "difffas_recon_pairs_v1.parquet"):
            self.assertFalse((ROOT / "manifests" / name).exists(), name)

    def test_stage_state(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())["milestones"]
        self.assertEqual(s["M2"]["status"], "COMPLETE")
        self.assertEqual(s["M3"]["status"], "COMPLETE")
        self.assertEqual(s["M4"]["status"], "COMPLETE")
        self.assertEqual(s["M5"]["status"], "NOT_STARTED")
        self.assertEqual(s["M6"]["status"], "NOT_STARTED")
        amend = s["M4"]["amendment_a1"]
        self.assertEqual(amend["track_a"], "COMPLETE")
        self.assertEqual(amend["track_b_native_pairing"], "DEFERRED_TO_M6_SECONDARY_TRACK")
        self.assertEqual(amend["completion_basis"], "AMENDMENT_A1_MAIN_FAIR_IDFREE_TRACK")
        self.assertIs(amend["original_native_manifests_completed"], False)

    def test_deviations_are_recorded_without_overwriting_earlier_ones(self):
        dev = (AUDIT / "deviation_report.md").read_text()
        for d in ("DEV-018", "DEV-019", "DEV-020", "DEV-021"):
            self.assertIn(d, dev, d)
        self.assertIn("MAIN_FAIR_TRACK_IDENTITY_FREE_PROTOCOL_AMENDMENT", dev)
        self.assertIn("BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP", dev)


if __name__ == "__main__":
    unittest.main()
