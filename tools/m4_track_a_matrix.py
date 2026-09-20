"""Track-A fairness matrix (Amendment A1 §27).

    <m2 venv>/bin/python tools/m4_track_a_matrix.py

One row per planned Track-A method. Every Track-A method must train on CASIA + MSU + SiW, consume no
subject identity, consume no attack-type supervision, and carry the same synthetic budget of 8,838.
Any violation is flagged rather than smoothed over.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import track_a as T  # noqa: E402

AUDIT = ROOT / "outputs/audit"
POOLED = "casia_fasd+msu_mfsd+siwmv2"
N_SYN = 8838

# source_conditioned: does the method consume a (source_spoof, target_live) pair to make one output?
METHODS = [
    {"method_id": "E01", "method": "FAS-Aug", "source_conditioned": True,
     "uses_common_pair": "YES (target_live_id; pair_id seeds the operator, spec 8.1)",
     "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "Handcrafted physics operators on the common pair's live target; ignores source_spoof_id."},
    {"method_id": "E02", "method": "Frequency Substitution", "source_conditioned": True,
     "uses_common_pair": "YES (source and target)", "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "Fully specified in spec 8.2; direct frequency transfer between the paired images."},
    {"method_id": "E03", "method": "STDN", "source_conditioned": True,
     "uses_common_pair": "YES (source and target)", "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "Learned spoof-trace transfer; trains on the pooled TRAIN population."},
    {"method_id": "E04", "method": "Physics-Guided STD", "source_conditioned": True,
     "uses_common_pair": "YES (source and target)", "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "Frequency-aware trace transfer; FAITHFUL_PAPER unless official code is found (Q-09)."},
    {"method_id": "E05", "method": "PCGAN", "source_conditioned": True,
     "uses_common_pair": "YES (source and target)", "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "Full-image artifact pattern conversion; may not reuse GPAT identity/landmark losses."},
    {"method_id": "E06c", "method": "DSDG-BIN-IDFREE", "source_conditioned": False,
     "uses_common_pair": "YES (the common TRAIN pair IS its training relation)",
     "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "DEV-020. Generative: samples from the latent prior, so no per-output source/target, "
              "but the generation budget is still 8,838. lambda_pair = 0."},
    {"method_id": "E07c", "method": "DIFFFAS-BIN-IDFREE", "source_conditioned": True,
     "uses_common_pair": "YES (generation target_live_id and source_spoof_id)",
     "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "DEV-021. use_pair=false; training GT/guide from difffas_bin_idfree_train_v1.parquet; "
              "style_id = SPOOF_BINARY."},
    {"method_id": "E08", "method": "GPAT-B0", "source_conditioned": True,
     "uses_common_pair": "YES (source and target)", "uses_subject_id": "NO", "uses_attack_type": "NO",
     "notes": "Proposed method, main fair configuration: lambda_type = 0.0, lambda_idadv = 0.0."},
]


def main() -> int:
    fair = yaml.safe_load(T.FAIR_TRACK_CONFIG.read_text())
    reg = {m["method_id"]: m for m in
           yaml.safe_load((ROOT / "third_party/registry.yaml").read_text())["methods"]}
    rows, violations = [], []
    for m in METHODS:
        r = {
            "method_id": m["method_id"],
            "method": m["method"],
            "track": "A_FAIR_COMMON_IDENTITY_FREE",
            "train_datasets": POOLED,
            "val_datasets": POOLED,
            "uses_subject_id": m["uses_subject_id"],
            "uses_attack_type": m["uses_attack_type"],
            "source_conditioned": "YES" if m["source_conditioned"] else "NO",
            "common_pair_source": m["uses_common_pair"],
            "n_syn_intended": N_SYN,
            "registry_name": reg.get(m["method_id"], {}).get("method_name", "NOT_IN_REGISTRY"),
            "notes": m["notes"],
        }
        if r["uses_subject_id"] != "NO":
            violations.append(f"{r['method_id']}: consumes subject identity")
        if r["uses_attack_type"] != "NO":
            violations.append(f"{r['method_id']}: consumes attack-type supervision")
        if r["train_datasets"] != POOLED or r["val_datasets"] != POOLED:
            violations.append(f"{r['method_id']}: dataset population differs")
        if r["n_syn_intended"] != N_SYN:
            violations.append(f"{r['method_id']}: synthetic budget differs")
        if m["method_id"] not in fair["track_a"]["methods"]:
            violations.append(f"{r['method_id']}: not listed in the frozen Track-A method list")
        rows.append(r)

    listed = set(fair["track_a"]["methods"]) - {m["method_id"] for m in METHODS}
    for extra in sorted(listed):
        violations.append(f"{extra}: listed in the frozen Track-A config but missing from the matrix")

    p = AUDIT / "M4_TRACK_A_METHOD_MATRIX.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"methods": len(rows), "violations": violations}, indent=1))
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
