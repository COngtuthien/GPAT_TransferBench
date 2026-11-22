"""M8 downstream readiness package: both DRAFTs are unapproved and never authority, the S-matrix mirrors the smoke DRAFT,
the frozen pins (ResNet-18 weights, downstream config) match, the DINOv3 gap is real, and the toy facts re-derive from
TRAIN/VAL metadata. At the commit that added this test no approved contract existed (pinned to that commit, B1).
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

A = json.loads((ROOT / "outputs/audit/M8_DOWNSTREAM_READINESS_ASSESSMENT.json").read_text())
SMOKE_DRAFT = json.loads((ROOT / "configs/amendments/m8_smoke_test_contract.DRAFT.yaml").read_text())
EVAL_DRAFT = json.loads((ROOT / "configs/amendments/m8_downstream_evaluator_contract.DRAFT.yaml").read_text())


def at_added_commit(path):
    c = subprocess.run(["git", "log", "--diff-filter=A", "-1", "--format=%H", "--",
                        "tests/test_m8_downstream_readiness_package.py"], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
    if not c:
        return (ROOT / path).exists()
    return subprocess.run(["git", "cat-file", "-e", f"{c}:{path}"], cwd=ROOT).returncode == 0


class M8DownstreamReadiness(unittest.TestCase):
    def test_01_drafts_unapproved(self):
        for d in (SMOKE_DRAFT, EVAL_DRAFT):
            self.assertEqual((d["status"], d["approved"]), ("DRAFT_PENDING_OWNER_DECISION", False))
        for p in ("configs/amendments/m8_smoke_test_contract.yaml", "configs/amendments/m8_downstream_evaluator_contract.yaml",
                  "outputs/audit/M8_SMOKE_TEST_RECORD.json"):
            self.assertFalse(at_added_commit(p), p)

    def test_02_matrix_mirrors_smoke_draft(self):
        self.assertEqual(A["status"], "M8_SMOKE_CONTRACT_PARTIALLY_RESOLVED")
        self.assertEqual(sorted(A["s_matrix"]), sorted(q for q in SMOKE_DRAFT["questions"] if q != "S1_scope"))
        for q, row in A["s_matrix"].items():
            self.assertEqual(row["recommended"], SMOKE_DRAFT["questions"][q]["recommended"])
            self.assertEqual(row["options"], sorted(SMOKE_DRAFT["questions"][q]["options"]))
            self.assertFalse(row["resolvable_by_existing_authority"])
        self.assertEqual(A["evaluator_contract_draft"]["open_questions"], sorted(EVAL_DRAFT["questions"]))

    def test_03_frozen_pins(self):
        import hashlib
        self.assertEqual(hashlib.sha256((ROOT / "configs/frozen/downstream_resnet18.yaml").read_bytes()).hexdigest(),
                         EVAL_DRAFT["frozen_inputs"]["configs/frozen/downstream_resnet18.yaml"])
        reg = (ROOT / "models/registry.yaml").read_text()
        self.assertIn("weight_sha256: " + A["resnet18_weights"]["sha256"], reg)
        self.assertIn(A["resnet18_weights"]["sha256"], EVAL_DRAFT["frozen_inputs"]["resnet18_weights"])
        self.assertFalse(A["dinov3"]["frozen_config_present"])
        self.assertFalse(at_added_commit("configs/frozen/downstream_dinov3_vits16.yaml"))

    def test_04_toy_facts(self):
        import pyarrow.parquet as pq
        s = [r for r in pq.read_table(ROOT / "manifests/split_v1.parquet",
                                      columns=["dataset", "subject_id_global", "content_group_id", "label_binary", "split"]
                                      ).to_pylist() if r["split"] in ("TRAIN", "VAL")]
        p = pq.read_table(ROOT / "manifests/pairs_train_v1.parquet",
                          columns=["dataset", "target_subject", "target_content_group_id"]).to_pylist()
        tgt = {(r["dataset"], r["target_subject"]) for r in p}
        tgt_cg = {r["target_content_group_id"] for r in p if r["dataset"] == "siwmv2"}
        live = {(r["dataset"], r["subject_id_global"]) for r in s
                if r["split"] == "TRAIN" and r["dataset"] != "siwmv2" and r["label_binary"] == 0}
        siw = {r["content_group_id"] for r in s if r["split"] == "TRAIN" and r["dataset"] == "siwmv2" and r["label_binary"] == 0}
        f = A["toy_facts_train_val_only"]
        self.assertEqual([len(live & tgt), len(live)], f["casia_msu_train_live_subjects_pair_target"])
        self.assertEqual([len(siw & tgt_cg), len(siw)], f["siwmv2_train_live_groups_pair_target"])
        for sp in ("TRAIN", "VAL"):
            self.assertEqual(sum(1 for r in s if r["split"] == sp and r["dataset"] == "siwmv2" and r["subject_id_global"] is None),
                             f["siwmv2_null_subject_rows"][sp])
        self.assertEqual(len({r["subject_id_global"] for r in s if r["split"] == "VAL" and r["dataset"] == "msu_mfsd"}),
                         f["msu_val_subjects"])

    def test_05_nothing_executed(self):
        self.assertEqual(set(A["executed"].values()), {False})


if __name__ == "__main__":
    unittest.main()
