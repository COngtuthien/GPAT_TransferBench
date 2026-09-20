"""Native (DSDG / DiffFAS) TRAIN identity coverage for M4 (prompt §35-§36).

    <m2 venv>/bin/python tools/m4_native_coverage.py

Measures what the datasets can actually support. Coverage is reported for every method x dataset
cell, including the cells that cannot be instantiated -- those are recorded here, never as manifest
rows. Identity is never fabricated: `content_group_id` and `video_id` are not identity, and DEV-013
is not same-identity evidence.

Amendment A1 moved DSDG-NATIVE and DiffFAS-NATIVE to Track B (SECONDARY). Their manifests are
deferred to M6 and no longer block the main comparison or M5. The Track-A identity-free variants
(E06c, E07c) cover all three datasets and are audited separately.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDIT = ROOT / "outputs/audit"
DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
METHODS = ("DSDG", "DiffFAS")
REGISTRY = ROOT / "third_party/registry.yaml"


def main() -> int:
    rows = [r for r in pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()
            if r["split"] == "TRAIN"]
    import yaml
    reg = {m["method_id"]: m for m in yaml.safe_load(REGISTRY.read_text())["methods"]}
    pins = {"DSDG": reg["E06b"], "DiffFAS": reg["E07b"]}

    out = []
    for method in METHODS:
        m = pins[method]
        for ds in DATASETS:
            sub = [r for r in rows if r["dataset"] == ds]
            ids = sorted({r["subject_id_global"] for r in sub} - {None})
            live = {s for s in ids if any(r["subject_id_global"] == s and r["label_binary"] == 0
                                          for r in sub)}
            spoof = {s for s in ids if any(r["subject_id_global"] == s and r["label_binary"] == 1
                                           for r in sub)}
            both = live & spoof
            styles = sorted({f"{ds}::{r['attack_raw']}" for r in sub if r["label_binary"] == 1})
            usable = ds != "siwmv2"
            out.append({
                "method": method,
                "dataset": ds,
                "track": "B_NATIVE_FULL_SECONDARY",
                "status": "SUPPORTED_DEFERRED_TO_M6_SECONDARY_TRACK" if usable
                          else "NOT_INSTANTIABLE_MISSING_SUBJECT_ID",
                "train_rows": len(sub),
                "train_live_rows": sum(1 for r in sub if r["label_binary"] == 0),
                "train_spoof_rows": sum(1 for r in sub if r["label_binary"] == 1),
                "total_train_identities": len(ids),
                "identities_with_live": len(live),
                "identities_with_spoof": len(spoof),
                "eligible_identities_live_and_spoof": len(both),
                "identity_coverage": (round(len(both) / len(ids), 6) if ids else 0.0),
                "style_ids": len(styles),
                "native_rows_materialized": 0,
                "reason": ("dataset supports the row semantics; the official source is now pinned "
                           "(Amendment A1), but Track B is the SECONDARY track and its native "
                           "manifests are deferred to M6 -- they no longer block the main comparison "
                           "or M5"
                           if usable else
                           "SiW-Mv2 carries no trustworthy subject identity (Q-14); identity is "
                           "never fabricated and DEV-013 is not same-identity evidence"),
                "official_repo": m["official_repo"],
                "pinned_commit": m["pinned_commit"] or "NONE",
                "blocks_main_comparison": False,
                "blocks_m5": False,
                "url_verification": m["url_verification"],
                "source_evidence": "third_party/registry.yaml;outputs/audit/M4_NATIVE_PAIR_SOURCE_AUDIT.md",
            })

    p = AUDIT / "M4_NATIVE_PAIR_COVERAGE.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(json.dumps([{k: r[k] for k in ("method", "dataset", "status",
                                         "eligible_identities_live_and_spoof",
                                         "native_rows_materialized")} for r in out], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
