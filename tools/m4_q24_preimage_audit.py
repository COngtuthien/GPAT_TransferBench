"""Audit evidence for the Q-24 candidate-preimage correction pass.

    <m2 venv>/bin/python tools/m4_q24_preimage_audit.py

Answers two separate questions on the SAME deterministic diagnostic sample used by
tools/m4_frozen_diagnostic.py (same salt, same 40 sources per dataset x split):

  1. Did the candidate-64 sets change between the implementation committed at e4d167b and the
     final owner-approved raw-digest-byte implementation?
  2. Counterfactual: how much WOULD the sets have differed had the forbidden hex-string
     formulation been implemented instead of the frozen raw-byte one?

Audit evidence only. No pair membership is persisted, and nothing here may change the frozen rule.
Results are merged into outputs/audit/M4_PAIR_PREFLIGHT.json under `q24_preimage_correction`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.pairs import common as P  # noqa: E402

AUDIT = ROOT / "outputs/audit"
DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
SALT = "gpatbench.m4_frozen_diag.v1|"        # identical to the frozen diagnostic
PER_DATASET_SPLIT = 40
PRIOR_COMMIT = "e4d167b9ea3502cebbb97f730a5499060bb4ce26"


def load_prior_module():
    """Import gpatbench/pairs/common.py exactly as committed at PRIOR_COMMIT."""
    src = subprocess.run(["git", "show", f"{PRIOR_COMMIT}:gpatbench/pairs/common.py"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "prior_common.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("prior_common", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(src).hexdigest()


def hex_variant_key(source_sample_id: str, target_sample_id: str, split_seed: int) -> tuple:
    """The FORBIDDEN formulation (section 3): the source seed rendered as lowercase hex text."""
    seed_hex = hashlib.sha256(
        f"{P.SOURCE_SEED_NAMESPACE}|{source_sample_id}|{split_seed}".encode()).hexdigest()
    d = hashlib.sha256(f"{P.CANDIDATE_NAMESPACE}|{seed_hex}|{target_sample_id}".encode()).digest()
    return (int.from_bytes(d, "big", signed=False), target_sample_id)


def pick(eligible_ids, key, cap=P.CANDIDATE_CAP):
    ids = sorted(eligible_ids)
    if len(ids) <= cap:
        return tuple(ids)
    return tuple(sorted(sorted(ids, key=key)[:cap]))


def main() -> int:
    prior, prior_sha = load_prior_module()
    rows = pq.read_table(ROOT / "manifests/split_v1.parquet").to_pylist()

    def s(r):
        return P.Sample(r["sample_id"], r["dataset"], r["split"], int(r["label_binary"]),
                        r["video_id"], r["subject_id_global"], r["content_group_id"],
                        r["attack_macro"], r["attack_raw"], r["sha256"])

    out = {"prior_commit": PRIOR_COMMIT, "prior_common_py_sha256": prior_sha,
           "salt": SALT, "sources_per_dataset_split": PER_DATASET_SPLIT,
           "membership_persisted": False, "per_cell": {}}
    tot = {"sources": 0, "changed_vs_prior": 0, "changed_vs_hex_variant": 0}

    for ds in DATASETS:
        for split in P.SPLITS_WITH_PAIRS:
            pool = [r for r in rows if r["dataset"] == ds and r["split"] == split]
            spoof = sorted([r for r in pool if r["label_binary"] == 1],
                           key=lambda r: hashlib.sha256((SALT + r["sample_id"]).encode()).hexdigest())
            live = [s(r) for r in pool if r["label_binary"] == 0]
            n = chg_prior = chg_hex = 0
            for r in spoof[:PER_DATASET_SPLIT]:
                src = s(r)
                ids = [t.sample_id for t in P.eligible_targets(src, live)]
                frozen = pick(ids, lambda t: P.candidate_rank_key(src.sample_id, t, P.SPLIT_SEED))
                asprior = pick(ids, lambda t: prior.candidate_rank_key(src.sample_id, t, prior.SPLIT_SEED))
                ashex = pick(ids, lambda t: hex_variant_key(src.sample_id, t, P.SPLIT_SEED))
                n += 1
                chg_prior += frozen != asprior
                chg_hex += frozen != ashex
            out["per_cell"][f"{ds}|{split}"] = {
                "sources_compared": n, "candidate_sets_changed_vs_prior": chg_prior,
                "candidate_sets_changed_vs_hex_variant": chg_hex,
                "pct_changed_vs_prior": round(100.0 * chg_prior / n, 4) if n else None,
                "pct_changed_vs_hex_variant": round(100.0 * chg_hex / n, 4) if n else None}
            tot["sources"] += n
            tot["changed_vs_prior"] += chg_prior
            tot["changed_vs_hex_variant"] += chg_hex

    out["total"] = {
        "sources_compared": tot["sources"],
        "candidate_sets_changed_vs_prior": tot["changed_vs_prior"],
        "pct_changed_vs_prior": round(100.0 * tot["changed_vs_prior"] / tot["sources"], 4),
        "candidate_sets_changed_vs_hex_variant": tot["changed_vs_hex_variant"],
        "pct_changed_vs_hex_variant": round(100.0 * tot["changed_vs_hex_variant"] / tot["sources"], 4)}
    out["interpretation"] = (
        "changed_vs_prior is the real effect of this correction pass on candidate membership. "
        "changed_vs_hex_variant is a counterfactual on a formulation that was never implemented; "
        "it is reported to quantify what the ambiguous prose would have cost had it been coded.")

    rep_path = AUDIT / "M4_PAIR_PREFLIGHT.json"
    rep = json.loads(rep_path.read_text())
    rep["q24_preimage_correction"] = out
    rep_path.write_text(json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(out["total"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
