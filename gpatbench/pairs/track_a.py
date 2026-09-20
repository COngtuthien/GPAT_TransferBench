"""Track-A (fair / common / identity-free) training relations — Amendment A1.

Two Track-A adapted methods need a frozen training relation:

* **DSDG-BIN-IDFREE (E06c, DEV-020)** reuses `manifests/pairs_train_v1.parquet` directly. That is an
  explicit frozen decision in `configs/frozen/dsdg_bin_idfree_v1.yaml`: a separate manifest would be
  a redundant projection of the common one plus three constants, and a second copy of the relation
  could drift. This module only *validates* that relation for Track-A use.
* **DIFFFAS-BIN-IDFREE (E07c, DEV-021)** needs a new manifest, because the deterministic style guide
  is information the common manifest does not carry. This module builds it.

Nothing here reads, derives or accepts subject identity. The only identity-shaped columns that exist
upstream (`target_subject`, `source_subject`) are never consulted, which is what lets SiW-Mv2
participate on equal terms.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import common as P
from . import execute as E

ROOT = Path(__file__).resolve().parents[2]

FAIR_TRACK_CONFIG = ROOT / "configs/frozen/fair_track_v1.yaml"
DSDG_CONFIG = ROOT / "configs/frozen/dsdg_bin_idfree_v1.yaml"
DIFFFAS_CONFIG = ROOT / "configs/frozen/difffas_bin_idfree_v1.yaml"
SOURCE_PINS = ROOT / "third_party/source_pins.json"

DIFFFAS_MANIFEST = ROOT / "manifests/difffas_bin_idfree_train_v1.parquet"

TRACK_A_DATASETS = ("casia_fasd", "msu_mfsd", "siwmv2")
EXPECTED_ROWS = 8838
EXPECTED_BY_DATASET = {"casia_fasd": 2520, "msu_mfsd": 1200, "siwmv2": 5118}

# Q-24-style raw-byte ranking, with its own namespaces so a guide digest can never collide with a
# common-pair candidate digest.
GUIDE_SOURCE_NAMESPACE = "gpatbench.trackA.difffas.guide.source.v1"
GUIDE_CANDIDATE_NAMESPACE = "gpatbench.trackA.difffas.guide.candidate.v1"

STYLE_ID = "SPOOF_BINARY"
USE_PAIR = False
CONTENT_TRAINING_ROLE = "INERT_WHEN_USE_PAIR_FALSE"
SELECTION_POLICY = "DETERMINISTIC_RAW_BYTE_SHA256_RANKING"
TRACK_PAIR_ID_FORMAT = "DFA%06d"

DIFFFAS_COLUMNS = ("track_pair_id", "dataset", "gt_spoof_id", "guide_spoof_id", "style_id",
                   "split", "seed", "eligible_guide_count", "selection_policy", "use_pair",
                   "self_guide", "content_live_id", "content_training_role",
                   "gt_sha256", "guide_sha256", "split_manifest_sha256",
                   "common_pair_manifest_sha256", "adaptation_config_sha256",
                   "official_source_commit")
DIFFFAS_INT_COLUMNS = {"seed", "eligible_guide_count"}
DIFFFAS_BOOL_COLUMNS = {"use_pair", "self_guide"}
ROW_ORDER = ("dataset", "gt_spoof_id")


class TrackAPolicyError(RuntimeError):
    """A Track-A contract was violated (identity leak, wrong population, missing provenance)."""


# ------------------------------------------------------------------ deterministic guide (DEV-021)
def guide_source_digest(gt_spoof_sample_id: str, split_seed: int = P.SPLIT_SEED) -> bytes:
    """RAW 32-byte SHA-256; never rendered as hex before the candidate preimage is built."""
    return hashlib.sha256(
        f"{GUIDE_SOURCE_NAMESPACE}|{gt_spoof_sample_id}|{split_seed}".encode("utf-8")).digest()


def guide_candidate_digest(source_digest: bytes, guide_spoof_sample_id: str) -> bytes:
    return hashlib.sha256(
        source_digest + f"|{GUIDE_CANDIDATE_NAMESPACE}|{guide_spoof_sample_id}".encode("utf-8")
    ).digest()


def guide_rank_key(gt_spoof_sample_id: str, guide_spoof_sample_id: str,
                   split_seed: int = P.SPLIT_SEED) -> tuple:
    d = guide_candidate_digest(guide_source_digest(gt_spoof_sample_id, split_seed),
                               guide_spoof_sample_id)
    return (int.from_bytes(d, "big", signed=False), guide_spoof_sample_id)


def eligible_guides(gt_row: dict, spoof_pool: list) -> list:
    """M2 COMPLETE + TRAIN + SPOOF + same dataset, in canonical lexical order.

    Subject identity and `attack_raw` are deliberately NOT part of eligibility: all attacks collapse
    into one binary spoof style pool, which is what allows SiW-Mv2 to participate. The GT itself is
    inside the pool because the official `random.choice` over the style folder includes it.
    """
    if int(gt_row["label_binary"]) != 1:
        raise TrackAPolicyError(f"{gt_row['sample_id']}: GT must be a SPOOF row")
    if gt_row["split"] != "TRAIN":
        raise TrackAPolicyError(f"{gt_row['sample_id']}: Track-A DiffFAS trains on TRAIN only")
    out = [r for r in spoof_pool if r["dataset"] == gt_row["dataset"]]
    if not out:
        raise TrackAPolicyError(f"{gt_row['sample_id']}: empty guide pool")
    return sorted(out, key=lambda r: r["sample_id"])


def choose_guide(gt_row: dict, eligible: list, split_seed: int = P.SPLIT_SEED) -> dict:
    return min(eligible, key=lambda r: guide_rank_key(gt_row["sample_id"], r["sample_id"],
                                                      split_seed))


# ------------------------------------------------------------------ population
def load_track_a_population() -> dict:
    """TRAIN rows plus the frozen common pair relation. No image or cache access is needed."""
    rows = pq.read_table(E.SPLIT_MANIFEST).to_pylist()
    acct = {r["sample_id"]: r for r in pq.read_table(E.ACCOUNTING).to_pylist()}
    train = []
    for r in rows:
        if r["split"] != "TRAIN":
            continue
        if r["m2_status"] != "COMPLETE" or acct[r["sample_id"]]["final_status"] != "COMPLETE":
            raise TrackAPolicyError(f"{r['sample_id']}: not M2 COMPLETE")
        train.append(r)
    pairs = pq.read_table(E.MANIFEST_PATH["TRAIN"]).to_pylist()
    return {
        "train_rows": train,
        "common_pairs": pairs,
        "split_manifest_sha256": E.sha256_file(E.SPLIT_MANIFEST),
        "common_pair_manifest_sha256": E.sha256_file(E.MANIFEST_PATH["TRAIN"]),
        "difffas_config_sha256": E.sha256_file(DIFFFAS_CONFIG),
        "dsdg_config_sha256": E.sha256_file(DSDG_CONFIG),
        "fair_track_config_sha256": E.sha256_file(FAIR_TRACK_CONFIG),
    }


def official_source_commit(key: str) -> str:
    import json
    return json.loads(SOURCE_PINS.read_text())["sources"][key]["pinned_commit"]


# ------------------------------------------------------------------ DSDG-BIN-IDFREE (DEV-020)
def dsdg_training_relation(pop: dict) -> list:
    """The frozen Track-A relation for DSDG-BIN-IDFREE: the common TRAIN pairs, identity stripped.

    Returns the exact adapter interface — no subject column is present, so a DSDG adapter built on
    this cannot consume identity even by accident.
    """
    out = []
    for r in pop["common_pairs"]:
        out.append({"pair_id": r["pair_id"], "dataset": r["dataset"],
                    "source_spoof_id": r["source_spoof_id"],
                    "target_live_id": r["target_live_id"],
                    "split": r["split"], "seed": r["seed"],
                    "binary_spoof_label": 1,
                    "source_sha256": r["source_sha256"], "target_sha256": r["target_sha256"]})
    out.sort(key=lambda r: (r["dataset"], r["source_spoof_id"]))
    return out


# ------------------------------------------------------------------ DIFFFAS-BIN-IDFREE (DEV-021)
def build_difffas_rows(pop: dict, split_seed: int = P.SPLIT_SEED) -> list:
    content_of = {r["source_spoof_id"]: r["target_live_id"] for r in pop["common_pairs"]}
    sha_of = {r["sample_id"]: r["sha256"] for r in pop["train_rows"]}
    spoof = [r for r in pop["train_rows"] if int(r["label_binary"]) == 1]
    pool_by_dataset = {}
    for r in spoof:
        pool_by_dataset.setdefault(r["dataset"], []).append(r)
    for ds in pool_by_dataset:
        pool_by_dataset[ds].sort(key=lambda r: r["sample_id"])

    commit = official_source_commit("difffas")
    rows = []
    for gt in sorted(spoof, key=lambda r: (r["dataset"], r["sample_id"])):
        elig = eligible_guides(gt, pool_by_dataset[gt["dataset"]])
        guide = choose_guide(gt, elig, split_seed)
        rows.append({
            "track_pair_id": None,                       # assigned after membership is final
            "dataset": gt["dataset"],
            "gt_spoof_id": gt["sample_id"],
            "guide_spoof_id": guide["sample_id"],
            "style_id": STYLE_ID,
            "split": "TRAIN",
            "seed": int(split_seed),
            "eligible_guide_count": len(elig),
            "selection_policy": SELECTION_POLICY,
            "use_pair": USE_PAIR,
            "self_guide": guide["sample_id"] == gt["sample_id"],
            "content_live_id": content_of[gt["sample_id"]],
            "content_training_role": CONTENT_TRAINING_ROLE,
            "gt_sha256": sha_of[gt["sample_id"]],
            "guide_sha256": sha_of[guide["sample_id"]],
            "split_manifest_sha256": pop["split_manifest_sha256"],
            "common_pair_manifest_sha256": pop["common_pair_manifest_sha256"],
            "adaptation_config_sha256": pop["difffas_config_sha256"],
            "official_source_commit": commit,
        })
    rows.sort(key=lambda r: tuple(r[c] for c in ROW_ORDER))
    for i, r in enumerate(rows, start=1):
        r["track_pair_id"] = TRACK_PAIR_ID_FORMAT % i
    return rows


def difffas_schema() -> pa.Schema:
    def t(c):
        if c in DIFFFAS_INT_COLUMNS:
            return pa.int64()
        if c in DIFFFAS_BOOL_COLUMNS:
            return pa.bool_()
        return pa.string()
    return pa.schema([(c, t(c)) for c in DIFFFAS_COLUMNS])


def write_difffas_manifest(rows: list, path: Path = DIFFFAS_MANIFEST) -> str:
    schema = difffas_schema()
    table = pa.table({c: pa.array([r[c] for r in rows], schema.field(c).type)
                      for c in DIFFFAS_COLUMNS}, schema=schema)
    buf = io.BytesIO()
    pq.write_table(table, buf, **E.PARQUET_WRITER)
    return E.atomic_write_bytes(path, buf.getvalue())


def difffas_schema_signature() -> str:
    return E.sha256_text(";".join(f"{f.name}:{f.type}" for f in difffas_schema()))


# ------------------------------------------------------------------ orchestration
def run(write: bool = True, out_dir: Path | None = None, shuffle_seed: int | None = None) -> dict:
    pop = load_track_a_population()
    if shuffle_seed is not None:
        import random
        pop = dict(pop)
        pop["train_rows"] = list(pop["train_rows"])
        pop["common_pairs"] = list(pop["common_pairs"])
        random.Random(shuffle_seed).shuffle(pop["train_rows"])
        random.Random(shuffle_seed + 1).shuffle(pop["common_pairs"])
    rows = build_difffas_rows(pop)
    path = (out_dir / DIFFFAS_MANIFEST.name) if out_dir else DIFFFAS_MANIFEST
    out = {"pop": pop, "difffas_rows": rows, "dsdg_rows": dsdg_training_relation(pop), "sha256": {}}
    if write:
        out["sha256"][DIFFFAS_MANIFEST.name] = write_difffas_manifest(rows, path)
    return out
