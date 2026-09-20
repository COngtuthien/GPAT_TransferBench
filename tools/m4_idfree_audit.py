"""Exhaustive audit of the Track-A identity-free relations (Amendment A1 §35, §36).

    <m2 venv>/bin/python tools/m4_idfree_audit.py

Re-derives both Track-A relations from the frozen inputs and checks them against the frozen
contracts. Read-only: nothing here can change membership.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P      # noqa: E402
from gpatbench.pairs import execute as E     # noqa: E402
from gpatbench.pairs import track_a as T     # noqa: E402

AUDIT = ROOT / "outputs/audit"


def main() -> int:
    pop = T.load_track_a_population()
    split = {r["sample_id"]: r for r in pq.read_table(E.SPLIT_MANIFEST).to_pylist()}
    acct = {r["sample_id"]: r for r in pq.read_table(E.ACCOUNTING).to_pylist()}
    dsdg_cfg = yaml.safe_load(T.DSDG_CONFIG.read_text())
    diff_cfg = yaml.safe_load(T.DIFFFAS_CONFIG.read_text())
    fair_cfg = yaml.safe_load(T.FAIR_TRACK_CONFIG.read_text())

    rep = {"expected_rows": T.EXPECTED_ROWS, "expected_by_dataset": T.EXPECTED_BY_DATASET,
           "dsdg": {}, "difffas": {}, "failures": []}

    def fail(m):
        rep["failures"].append(m)

    # ---------------------------------------------------------------- DSDG-BIN-IDFREE (§35)
    rows = T.dsdg_training_relation(pop)
    d = rep["dsdg"]
    d["separate_manifest_created"] = (ROOT / "manifests/dsdg_bin_idfree_train_v1.parquet").exists()
    d["authoritative_relation"] = "manifests/pairs_train_v1.parquet"
    d["authoritative_relation_sha256"] = pop["common_pair_manifest_sha256"]
    d["rows"] = len(rows)
    d["rows_by_dataset"] = dict(collections.Counter(r["dataset"] for r in rows))
    d["lambda_pair"] = dsdg_cfg["losses"]["lambda_pair"]
    d["removed_losses"] = dsdg_cfg["losses"]["removed"]
    d["retained_losses"] = sorted(dsdg_cfg["losses"]["retained"])
    d["adapter_interface_fields"] = sorted(rows[0].keys())
    if d["separate_manifest_created"]:
        fail("dsdg: a separate manifest exists although the frozen decision is to reuse the common one")
    if d["rows"] != T.EXPECTED_ROWS:
        fail(f"dsdg: {d['rows']} rows, expected {T.EXPECTED_ROWS}")
    if d["rows_by_dataset"] != T.EXPECTED_BY_DATASET:
        fail(f"dsdg: per-dataset counts {d['rows_by_dataset']}")
    if d["lambda_pair"] != 0.0:
        fail(f"dsdg: lambda_pair is {d['lambda_pair']}, must be 0")
    if d["removed_losses"] != ["loss_pair"]:
        fail(f"dsdg: removed losses {d['removed_losses']}")
    for bad in ("subject", "identity_class", "person"):
        leak = [k for k in d["adapter_interface_fields"] if bad in k]
        if leak:
            fail(f"dsdg: identity field in the adapter interface: {leak}")
    seen = set()
    for r in rows:
        s, t = split[r["source_spoof_id"]], split[r["target_live_id"]]
        if s["split"] != "TRAIN" or t["split"] != "TRAIN":
            fail(f"dsdg {r['pair_id']}: not TRAIN")
        if acct[s["sample_id"]]["final_status"] != "COMPLETE" or \
           acct[t["sample_id"]]["final_status"] != "COMPLETE":
            fail(f"dsdg {r['pair_id']}: not M2 COMPLETE")
        if int(s["label_binary"]) != 1 or int(t["label_binary"]) != 0:
            fail(f"dsdg {r['pair_id']}: source must be SPOOF and target LIVE")
        if s["dataset"] != t["dataset"] or s["dataset"] != r["dataset"]:
            fail(f"dsdg {r['pair_id']}: dataset mismatch")
        if r["binary_spoof_label"] != 1:
            fail(f"dsdg {r['pair_id']}: binary_spoof_label")
        seen.add(r["source_spoof_id"])
    want_sources = {k for k, v in split.items()
                    if v["split"] == "TRAIN" and int(v["label_binary"]) == 1}
    if seen != want_sources:
        fail("dsdg: source set != TRAIN spoof set")
    d["subject_equality_required"] = False
    d["subject_id_consumed"] = False
    d["siw_included"] = d["rows_by_dataset"].get("siwmv2", 0) > 0
    # the relation must survive nulling every CASIA/MSU subject id
    nulled = {"train_rows": pop["train_rows"], "common_pairs": [
        {**r, "source_subject": None, "target_subject": None} for r in pop["common_pairs"]]}
    nulled.update({k: v for k, v in pop.items() if k not in nulled})
    d["unchanged_when_subject_ids_nulled"] = T.dsdg_training_relation(nulled) == rows
    if not d["unchanged_when_subject_ids_nulled"]:
        fail("dsdg: the Track-A relation changed when subject ids were nulled")

    # ---------------------------------------------------------------- DIFFFAS-BIN-IDFREE (§36)
    written = pq.read_table(T.DIFFFAS_MANIFEST).to_pylist()
    rebuilt = T.build_difffas_rows(pop)
    f = rep["difffas"]
    f["path"] = T.DIFFFAS_MANIFEST.relative_to(ROOT).as_posix()
    f["sha256"] = E.sha256_file(T.DIFFFAS_MANIFEST)
    f["rows"] = len(written)
    f["rows_by_dataset"] = dict(collections.Counter(r["dataset"] for r in written))
    f["schema"] = [f"{x.name}:{x.type}" for x in pq.read_schema(T.DIFFFAS_MANIFEST)]
    f["schema_signature"] = T.difffas_schema_signature()
    if written != rebuilt:
        fail("difffas: written manifest != independent rebuild")
    if f["rows"] != T.EXPECTED_ROWS:
        fail(f"difffas: {f['rows']} rows, expected {T.EXPECTED_ROWS}")
    if f["rows_by_dataset"] != T.EXPECTED_BY_DATASET:
        fail(f"difffas: per-dataset counts {f['rows_by_dataset']}")
    if [r["track_pair_id"] for r in written] != [T.TRACK_PAIR_ID_FORMAT % i
                                                 for i in range(1, len(written) + 1)]:
        fail("difffas: track_pair_id is not the canonical sequence")
    key = lambda r: tuple(r[c] for c in T.ROW_ORDER)
    if [key(r) for r in written] != sorted(key(r) for r in written):
        fail("difffas: rows are not in canonical (dataset, gt_spoof_id) order")
    gts, self_guides, wrong_guide = set(), 0, 0
    pool = {}
    for r in pop["train_rows"]:
        if int(r["label_binary"]) == 1:
            pool.setdefault(r["dataset"], []).append(r)
    for ds in pool:
        pool[ds].sort(key=lambda r: r["sample_id"])
    for r in written:
        gt, guide = split[r["gt_spoof_id"]], split[r["guide_spoof_id"]]
        if gt["split"] != "TRAIN" or guide["split"] != "TRAIN":
            fail(f"difffas {r['track_pair_id']}: not TRAIN")
        if int(gt["label_binary"]) != 1 or int(guide["label_binary"]) != 1:
            fail(f"difffas {r['track_pair_id']}: GT and guide must both be SPOOF")
        if gt["dataset"] != guide["dataset"] or gt["dataset"] != r["dataset"]:
            fail(f"difffas {r['track_pair_id']}: guide is from another dataset")
        for sid in (gt["sample_id"], guide["sample_id"]):
            if acct[sid]["final_status"] != "COMPLETE":
                fail(f"difffas {r['track_pair_id']}: {sid} not M2 COMPLETE")
        if r["style_id"] != T.STYLE_ID:
            fail(f"difffas {r['track_pair_id']}: style_id {r['style_id']}")
        if r["use_pair"] is not False:
            fail(f"difffas {r['track_pair_id']}: use_pair must be False")
        if r["content_training_role"] != T.CONTENT_TRAINING_ROLE:
            fail(f"difffas {r['track_pair_id']}: content_training_role")
        if r["seed"] != P.SPLIT_SEED:
            fail(f"difffas {r['track_pair_id']}: seed")
        if r["split_manifest_sha256"] != pop["split_manifest_sha256"] or \
           r["common_pair_manifest_sha256"] != pop["common_pair_manifest_sha256"] or \
           r["adaptation_config_sha256"] != pop["difffas_config_sha256"]:
            fail(f"difffas {r['track_pair_id']}: provenance hash mismatch")
        if r["official_source_commit"] != T.official_source_commit("difffas"):
            fail(f"difffas {r['track_pair_id']}: source commit")
        # deterministic guide, recomputed
        elig = T.eligible_guides(gt, pool[gt["dataset"]])
        if r["eligible_guide_count"] != len(elig):
            fail(f"difffas {r['track_pair_id']}: eligible_guide_count")
        if T.choose_guide(gt, elig)["sample_id"] != r["guide_spoof_id"]:
            wrong_guide += 1
        if r["self_guide"] != (r["guide_spoof_id"] == r["gt_spoof_id"]):
            fail(f"difffas {r['track_pair_id']}: self_guide flag")
        self_guides += bool(r["self_guide"])
        gts.add(r["gt_spoof_id"])
    if wrong_guide:
        fail(f"difffas: {wrong_guide} guides are not the deterministic minimum")
    if gts != want_sources:
        fail("difffas: GT set != TRAIN spoof set")
    f["unique_gt"] = len(gts)
    f["self_guide_count"] = self_guides
    f["unique_guides"] = len({r["guide_spoof_id"] for r in written})
    f["guide_reuse_max"] = max(collections.Counter(r["guide_spoof_id"] for r in written).values())
    f["eligible_guide_counts"] = sorted({r["eligible_guide_count"] for r in written})
    f["style_ids"] = sorted({r["style_id"] for r in written})
    f["attack_raw_in_schema"] = any("attack" in c for c in T.DIFFFAS_COLUMNS)
    f["subject_in_schema"] = any(("subject" in c or "identity" in c) for c in T.DIFFFAS_COLUMNS)
    f["siw_included"] = f["rows_by_dataset"].get("siwmv2", 0) > 0
    if f["attack_raw_in_schema"]:
        fail("difffas: an attack-type column leaked into the Track-A manifest")
    if f["subject_in_schema"]:
        fail("difffas: an identity column leaked into the Track-A manifest")

    # ---------------------------------------------------------------- shared Track-A invariants
    rep["track_a_datasets"] = fair_cfg["track_a"]["datasets"]
    rep["n_syn_intended"] = fair_cfg["track_a"]["n_syn_intended"]
    if sorted(rep["track_a_datasets"]) != sorted(T.TRACK_A_DATASETS):
        fail("track A: dataset list is not CASIA + MSU + SiW")
    if rep["n_syn_intended"] != T.EXPECTED_ROWS:
        fail("track A: n_syn_intended != 8838")
    rep["common_artifacts_unchanged"] = {
        "pair_train_stats_v1.json": E.sha256_file(E.STATS_PATH) ==
            "a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50",
        "pairs_train_v1.parquet": pop["common_pair_manifest_sha256"] ==
            "a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243",
        "val_pairs_v1.parquet": E.sha256_file(E.MANIFEST_PATH["VAL"]) ==
            "84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872",
        "split_v1.parquet": pop["split_manifest_sha256"] ==
            "fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d"}
    for k, v in rep["common_artifacts_unchanged"].items():
        if not v:
            fail(f"frozen artifact changed: {k}")

    rep["failures_count"] = len(rep["failures"])
    rep["status"] = "PASS" if not rep["failures"] else "FAIL"
    (AUDIT / "M4_IDFREE_MANIFEST_AUDIT.json").write_bytes(E.canonical_json_bytes(rep))
    print(json.dumps({"status": rep["status"], "failures": rep["failures"][:10],
                      "dsdg_rows": rep["dsdg"]["rows"], "difffas_rows": rep["difffas"]["rows"],
                      "self_guide": rep["difffas"]["self_guide_count"]}, indent=1))
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
