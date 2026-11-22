"""E01/E02 production authority record (two-stage) and the storage preflight evidence. No image is read."""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.common import bank_writer as bw  # noqa: E402
from methods.common import runlog  # noqa: E402

CANDIDATE = "d995f556a8adcc5a032ba266a022b470117d1c1c"
A = json.loads((ROOT / bw.PRODUCTION_AUTHORITY_PATH).read_text())
S = json.loads((ROOT / "outputs/audit/E01_E02_STORAGE_PREFLIGHT.json").read_text())
Q = json.loads((ROOT / "outputs/audit/E01_E02_PRODUCTION/read_path_qualification_d995f55.json").read_text())


def blob_sha(commit, rel):
    return hashlib.sha256(subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True,
                                         check=True).stdout).hexdigest()


class ProductionAuthority(unittest.TestCase):
    def test_01_record_names_candidate_and_pins_its_blobs(self):
        self.assertEqual((A["status"], A["approved"], A["candidate_commit"]), ("OWNER_APPROVED", True, CANDIDATE))
        self.assertEqual(set(A["authority_files"]), set(bw.AUTHORITY_FILES))
        for rel, digest in A["authority_files"].items():
            self.assertEqual(blob_sha(CANDIDATE, rel), digest, rel)
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", CANDIDATE, "HEAD"], cwd=ROOT).returncode, 0)
        self.assertEqual(subprocess.run(["git", "cat-file", "-e", f"{CANDIDATE}:{bw.PRODUCTION_AUTHORITY_PATH}"], cwd=ROOT,
                                        capture_output=True).returncode, 128)   # not in its own candidate (no cycle)

    def test_02_identities_match_binding_and_storage(self):
        self.assertEqual(A["binding_sha256"], blob_sha(CANDIDATE, bw.SMOKE_BINDING_PATH))
        self.assertEqual((A["ruling_sha256"], A["smoke_record_sha256"]),
                         (bw.APPROVED_RULING_SHA256, bw.APPROVED_SMOKE_RECORD_SHA256))
        st = A["storage"]
        self.assertEqual((st["filesystem_uuid"], st["mount_point"], st["runtime_root"]),
                         ("227ce2f6-373e-4f13-9cdb-84b911775706", "/",
                          "/home/student20261/workdir/GPAT_TransferBench_runtime"))
        self.assertEqual(st["filesystem_uuid"], S["decision"]["filesystem_uuid"])
        self.assertEqual(A["execution"]["machine_id"], S["machine_id"])
        self.assertEqual({k for k, v in A["authorizes"].items() if v}, {"e01_e02_production_bank_generation"})

    def test_03_storage_preflight(self):
        self.assertFalse(S["answer_approximately_5tb_ssd_present"])
        self.assertEqual(S["forbidden_actions_performed"], [])
        sel = [c for c in S["candidates"] if c["selected"]]
        self.assertEqual([c["device"] for c in sel], ["/dev/nvme0n1p3"])
        self.assertFalse(S["decision"]["path_contract_changed"])
        self.assertFalse(S["decision"]["symlinks_or_bind_mounts_introduced"])
        self.assertTrue(S["probe"]["roundtrip_identical"] and S["probe"]["deleted"])
        cp = S["capacity_plan"]
        self.assertGreater(cp["free_bytes_now"] - cp["estimated_bytes_six_banks"], cp["min_free_bytes_before_bank"])
        raw = (ROOT / "outputs/audit/E01_E02_PRODUCTION/storage_collection_raw.json").read_text()
        self.assertIn("names redacted", raw)

    def test_04_read_path_qualification_evidence(self):
        self.assertTrue(Q["all_pass"])
        self.assertEqual((Q["git_commit"], Q["worktree_dirty"]), (CANDIDATE, False))
        self.assertEqual(len(Q["banks"]), 6)
        for k, b in Q["banks"].items():
            self.assertEqual((b["rows"], b["difference_count"]), (186, 0), k)
            self.assertEqual(b["bank_manifest_sha256"], b["smoke_bank_manifest_sha256"], k)
        self.assertEqual((Q["firewall"]["violation_count"], Q["atime_changed_faces"]), (0, []))

    def test_05_launch_gate_passes_authority_stage_and_stops_at_host_preflight(self):
        auth = bw.load_production_authorization("E02", seed=2026)
        with mock.patch.object(runlog, "git_dirty", return_value=False):
            with self.assertRaises(bw.BankBlocked) as cm:
                bw.verify_production_launch("E02", 2026, auth, Path(A["storage"]["runtime_root"]))
        machine = Path("/etc/machine-id").read_text().strip() if Path("/etc/machine-id").is_file() else None
        if machine != A["execution"]["machine_id"]:
            self.assertEqual(cm.exception.status, bw.PRODUCTION_PREFLIGHT_STATUS, str(cm.exception))
            self.assertIn("approved execution host", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
