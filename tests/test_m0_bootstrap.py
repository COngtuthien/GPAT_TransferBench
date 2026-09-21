"""M0 bootstrap acceptance tests (stdlib unittest + PyYAML).

Run: python3 -m unittest discover -s tests -v
"""
import csv
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.audit.ledger import sha256_file  # noqa: E402

SPEC = ROOT / "docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx"
SPEC_SHA256 = "f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e"
SPEC_TEXT = (ROOT / "docs/spec/spec_extracted_text.txt").read_text(encoding="utf-8")
AUDIT = ROOT / "outputs/audit"
URL_RE = re.compile(r"https?://[^\s\"'#)]+")

SKELETON = [
    "configs/frozen", "configs/methods", "configs/experiments",
    "data/raw/casia_fasd", "data/raw/msu_mfsd", "data/raw/siwmv2",
    "data/processed/frames", "data/processed/faces_256",
    "manifests", "cache/geometry", "cache/identity",
    *[f"methods/{m}" for m in ["fas_aug", "freq_sub", "stdn", "physics_std", "pcgan", "dsdg", "difffas", "gpat"]],
    "probes", "downstream", "third_party", "models", "environments", "runs",
    *[f"outputs/{o}" for o in ["tables", "plots", "qualitative", "exploratory", "paper_pack", "audit"]],
    "tests", "frozen_config_snapshot",
]
AUDIT_FILES = [
    "SPEC_PROVENANCE.md", "LAPTOP_ENVIRONMENT_AUDIT.md", "PRISM_REUSE_AUDIT.md", "deviation_report.md",
    "EXECUTION_LEDGER.jsonl", "EXECUTION_FLOW.md", "STAGE_STATE.json", "ARTIFACT_INDEX.csv",
    "CLAIM_EVIDENCE_MAP.csv",
]
METHOD_IDS = ["E00", "E01", "E02", "E03", "E04", "E05", "E06a", "E06b", "E07a", "E07b", "E08", "E09", "E10", "E11"]
VERBATIM_CONFIGS = ["configs/frozen/downstream_resnet18.yaml", "configs/methods/gpat_b0.yaml"]


class TestSpec(unittest.TestCase):
    def test_spec_copy_hash(self):
        self.assertEqual(sha256_file(SPEC), SPEC_SHA256)

    def test_provenance_records_hash(self):
        self.assertIn(SPEC_SHA256, (AUDIT / "SPEC_PROVENANCE.md").read_text(encoding="utf-8"))


class TestSkeleton(unittest.TestCase):
    def test_directories(self):
        missing = [d for d in SKELETON if not (ROOT / d).is_dir()]
        self.assertEqual(missing, [])

    def test_audit_files(self):
        missing = [f for f in AUDIT_FILES if not (AUDIT / f).is_file()]
        self.assertEqual(missing, [])


class TestVerbatimConfigs(unittest.TestCase):
    def test_reextract_from_docx_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            for sub in ("configs/frozen", "configs/methods"):
                Path(tmp, sub).mkdir(parents=True)
            subprocess.run([sys.executable, str(ROOT / "tools/extract_spec_yaml.py"), str(SPEC)],
                           cwd=tmp, check=True, capture_output=True)
            for rel in VERBATIM_CONFIGS:
                self.assertEqual(Path(tmp, rel).read_bytes(), (ROOT / rel).read_bytes(), rel)

    def test_snapshot_identical_and_parses(self):
        for rel in VERBATIM_CONFIGS:
            self.assertEqual((ROOT / rel).read_bytes(), (ROOT / "frozen_config_snapshot" / rel).read_bytes())
            self.assertIsInstance(yaml.safe_load((ROOT / rel).read_text()), dict)

    def test_every_frozen_config_is_snapshotted(self):
        # Stage-aware (M1+): every frozen/method config must have a byte-identical snapshot.
        present = sorted(p.relative_to(ROOT).as_posix()
                         for sub in ("configs/frozen", "configs/methods") for p in (ROOT / sub).glob("*.yaml"))
        self.assertTrue(set(VERBATIM_CONFIGS) <= set(present))
        for rel in present:
            snap = ROOT / "frozen_config_snapshot" / rel
            self.assertTrue(snap.is_file(), rel)
            self.assertEqual(snap.read_bytes(), (ROOT / rel).read_bytes(), rel)


def _source_pins():
    return json.loads((ROOT / "third_party/source_pins.json").read_text(encoding="utf-8"))["sources"]


#: method_id -> pinned commit, as third_party/source_pins.json actually records it.
PINNED_METHOD_COMMITS = {mid: src["pinned_commit"]
                         for src in _source_pins().values() for mid in src["method_ids"]}

#: URLs a registry may carry although they are not verbatim in the spec: a pinned repository (or a
#: recorded byte-identical mirror of it) whose `spec_code_link` IS verbatim in the spec.
SPEC_RESOLVED_URLS = {
    u for src in _source_pins().values()
    if src.get("spec_code_link") and src["spec_code_link"] in SPEC_TEXT
    for u in (src["repository"], src.get("byte_identical_mirror")) if u
}


class TestRegistries(unittest.TestCase):
    def test_third_party_registry(self):
        reg = yaml.safe_load((ROOT / "third_party/registry.yaml").read_text(encoding="utf-8"))
        ids = [m["method_id"] for m in reg["methods"]]
        # Amendment A1 (2026-09-20) added the two Track-A identity-free variants. The M0 set must
        # still be present and unchanged; only these two additions are allowed.
        self.assertEqual([i for i in ids if i not in ("E06c", "E07c")], METHOD_IDS)
        self.assertEqual(len(set(ids)), len(ids), "duplicate method_id")
        required = {"method_id", "method_name", "role", "source_status", "official_repo", "pinned_commit",
                    "paper", "environment_status", "implementation_status", "notes"}
        for m in reg["methods"]:
            self.assertTrue(required <= set(m), m["method_id"])
            # A1 authorized pinning DSDG and DiffFAS; M6A1 (2026-09-21) authorized pinning the newly
            # verified E01 and E03 official sources, which spec 8.1 requires anyway ("pin the
            # repository commit on first setup"). Other baseline sources remain unpinned -- E02 needs
            # none, and no official release has been located for E04 or E05 as of 2026-09-21.
            # Hard-coding the allowed method list is therefore stale. The invariant
            # that actually matters is traceability, so it is enforced directly instead: a pin must
            # be a full 40-char commit, must point at source_pins.json, and must genuinely be
            # recorded there under a source whose method_ids include this method. That is stricter
            # than the old list -- an unrecorded pin now fails for every method, not just new ones.
            if m["pinned_commit"] is not None:
                self.assertRegex(m["pinned_commit"], r"^[0-9a-f]{40}$", m["method_id"])
                self.assertEqual(m["source_pins"], "third_party/source_pins.json", m["method_id"])
                self.assertIn(m["method_id"], PINNED_METHOD_COMMITS, m["method_id"])
                self.assertEqual(PINNED_METHOD_COMMITS[m["method_id"]], m["pinned_commit"], m["method_id"])
            self.assertEqual(m["implementation_status"], "NOT_STARTED")

    def test_no_invented_urls(self):
        """Every URL in a registry must be the spec's, or provably resolved from the spec's.

        Spec section 30 R4 links `RizhaoCai/FAS_Aug`, which is a one-file redirect stub naming two
        other repositories (M6A1). The repository actually pinned is therefore not verbatim in the
        spec, and blanking it would hide which code E01 runs. A URL is accepted only when
        source_pins.json records it together with a `spec_code_link` that IS verbatim in the spec,
        so every registry URL still traces back to the frozen document.
        """
        for rel in ["third_party/registry.yaml", "models/registry.yaml"]:
            for url in URL_RE.findall((ROOT / rel).read_text(encoding="utf-8")):
                if url in SPEC_TEXT:
                    continue
                self.assertIn(url, SPEC_RESOLVED_URLS,
                              f"{rel}: {url} is neither in the spec nor resolved from a spec URL "
                              f"in third_party/source_pins.json")

    def test_data_source_registry(self):
        reg = yaml.safe_load((ROOT / "configs/data_source_registry.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(reg["datasets"]), {"casia_fasd", "msu_mfsd", "siwmv2"})
        data_cfg = ROOT / "configs/frozen/data_v1.yaml"
        roots = yaml.safe_load(data_cfg.read_text())["datasets"] if data_cfg.exists() else {}
        for name, d in reg["datasets"].items():
            self.assertIs(d["read_only"], True)
            self.assertIn(d["status"], {"NEEDS_M1_AUDIT", "SELECTED_M1_INVENTORIED", "DATA_BLOCK"})
            if d["selected_path"] is None:
                self.assertNotEqual(d["status"], "SELECTED_M1_INVENTORIED")
            else:  # a selected source must be the frozen data_v1 root
                self.assertEqual(d["selected_path"], roots[name]["root"])


class TestAuditState(unittest.TestCase):
    def test_stage_state(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())
        allowed = {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE"}
        ms = s["milestones"]
        self.assertEqual(list(ms), [f"M{i}" for i in range(15)])
        for k, v in ms.items():
            self.assertIn(v["status"], allowed, k)
        # Stage-aware: M0 COMPLETE; milestones advance in order, one at a time.
        self.assertEqual(ms["M0"]["status"], "COMPLETE")
        statuses = [ms[f"M{i}"]["status"] for i in range(15)]
        first_open = next((i for i, s in enumerate(statuses) if s != "COMPLETE"), 15)
        self.assertTrue(all(s == "NOT_STARTED" for s in statuses[first_open + 1:]), statuses)

    def test_ledger_jsonl(self):
        keys = {"timestamp_utc", "host", "user", "cwd", "milestone", "purpose", "command", "git_commit",
                "git_dirty", "input_artifacts", "output_artifacts", "status", "exit_code", "notes"}
        lines = (AUDIT / "EXECUTION_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(lines), 0)
        for i, line in enumerate(lines):
            self.assertTrue(keys <= set(json.loads(line)), f"line {i + 1}")

    def test_deviation_statuses(self):
        text = (AUDIT / "deviation_report.md").read_text(encoding="utf-8")
        statuses = re.findall(r"\*\*Status:\*\* (\w+)", text)
        self.assertGreater(len(statuses), 0)
        self.assertTrue(set(statuses) <= {"APPROVED", "UNAPPROVED"}, statuses)

    def test_artifact_index_hashes(self):
        with open(AUDIT / "ARTIFACT_INDEX.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0)
        for r in rows:
            self.assertEqual(sha256_file(ROOT / r["path"]), r["sha256"], r["path"])


class TestNoLaterMilestoneArtifacts(unittest.TestCase):
    """M0 must not produce inventory/split/pair/run/output artifacts."""

    def _nonplaceholder(self, rel):
        return [p for p in (ROOT / rel).rglob("*") if p.is_file() and p.name != ".gitkeep"]

    # Stage-aware since M1: manifests/ may hold only the M1 inventory manifests.
    M1_MANIFESTS = {"manifests/inventory.parquet", "manifests/inventory_videos.parquet",
                    "manifests/raw_file_index.parquet"}
    # Stage-aware: each milestone may contribute only its own manifests, and only once it has
    # actually started. M4 pair manifests stay forbidden at every stage reached so far.
    def _allowed_manifests(self):
        import stage_guard
        return stage_guard.allowed_manifests()

    def test_empty_runtime_dirs(self):
        # Under DEV-017 the M2 artifacts live on the external runtime volume, so these in-repo
        # directories must stay empty even after M2 completes: no heavy generated data in Git.
        for rel in ["runs", "data/raw", "data/processed", "cache", "probes", "downstream",
                    "outputs/tables", "outputs/plots", "outputs/qualitative", "outputs/paper_pack"]:
            self.assertEqual(self._nonplaceholder(rel), [], rel)

    def test_manifests_are_stage_appropriate(self):
        present = {p.relative_to(ROOT).as_posix() for p in self._nonplaceholder("manifests")}
        allowed = self._allowed_manifests()
        self.assertTrue(present <= allowed, present - allowed)
        self.assertTrue(self.M1_MANIFESTS <= present, self.M1_MANIFESTS - present)

    def test_no_unaccounted_or_pair_parquet(self):
        import stage_guard
        parquet = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.parquet")
                   if not ({".git", ".venv"} & set(p.parts))}
        self.assertEqual(stage_guard.forbidden_parquet(parquet), [])
        self.assertEqual(stage_guard.forbidden_pair_artifacts(parquet), [],
                         "a pair manifest exists that the current stage does not account for")


if __name__ == "__main__":
    unittest.main()
