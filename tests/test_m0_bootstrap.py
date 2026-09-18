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

    def test_no_unfrozen_configs_present(self):
        present = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").glob("*/*.yaml")}
        self.assertEqual(present, set(VERBATIM_CONFIGS))


class TestRegistries(unittest.TestCase):
    def test_third_party_registry(self):
        reg = yaml.safe_load((ROOT / "third_party/registry.yaml").read_text(encoding="utf-8"))
        ids = [m["method_id"] for m in reg["methods"]]
        self.assertEqual(ids, METHOD_IDS)
        required = {"method_id", "method_name", "role", "source_status", "official_repo", "pinned_commit",
                    "paper", "environment_status", "implementation_status", "notes"}
        for m in reg["methods"]:
            self.assertTrue(required <= set(m), m["method_id"])
            self.assertIsNone(m["pinned_commit"], "no commit may be pinned at M0")
            self.assertEqual(m["implementation_status"], "NOT_STARTED")

    def test_no_invented_urls(self):
        """Every URL in registries must appear verbatim in the frozen spec text."""
        for rel in ["third_party/registry.yaml", "models/registry.yaml"]:
            for url in URL_RE.findall((ROOT / rel).read_text(encoding="utf-8")):
                self.assertIn(url, SPEC_TEXT, f"{rel}: {url}")

    def test_data_source_registry(self):
        reg = yaml.safe_load((ROOT / "configs/data_source_registry.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(reg["datasets"]), {"casia_fasd", "msu_mfsd", "siwmv2"})
        for d in reg["datasets"].values():
            self.assertIs(d["read_only"], True)
            self.assertIsNone(d["selected_path"])
            self.assertEqual(d["status"], "NEEDS_M1_AUDIT")


class TestAuditState(unittest.TestCase):
    def test_stage_state(self):
        s = json.loads((AUDIT / "STAGE_STATE.json").read_text())
        allowed = {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE"}
        ms = s["milestones"]
        self.assertEqual(list(ms), [f"M{i}" for i in range(15)])
        for k, v in ms.items():
            self.assertIn(v["status"], allowed, k)
        for i in range(1, 15):
            self.assertEqual(ms[f"M{i}"]["status"], "NOT_STARTED")

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

    def test_empty_runtime_dirs(self):
        for rel in ["manifests", "runs", "data/raw", "data/processed", "cache", "probes", "downstream",
                    "outputs/tables", "outputs/plots", "outputs/qualitative", "outputs/paper_pack"]:
            self.assertEqual(self._nonplaceholder(rel), [], rel)

    def test_no_parquet_anywhere(self):
        self.assertEqual([p for p in ROOT.rglob("*.parquet") if ".git" not in p.parts], [])


if __name__ == "__main__":
    unittest.main()
