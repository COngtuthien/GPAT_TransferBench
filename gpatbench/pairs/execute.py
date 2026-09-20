"""M4 authoritative pair-manifest execution (spec §6, App. A).

Builds, from the frozen inputs only:

    manifests/pair_train_stats_v1.json   TRAIN-fitted pose normalisers (Q-25)
    manifests/pairs_train_v1.parquet     one row per TRAIN spoof source
    manifests/val_pairs_v1.parquet       one row per VAL spoof source

`configs/frozen/pairs_v1.yaml` is the execution source of truth and Q-24..Q-29 are not reinterpreted
here. Every metric primitive lives in `common.py`; this module only loads the frozen M2/M3 inputs,
walks the sources in a canonical order, and serialises the result deterministically.

Determinism contract: nothing depends on input row order, filesystem order, `PYTHONHASHSEED`,
process count or wall-clock time. Candidate ranking uses the frozen two-stage SHA-256 over the RAW
32-byte source-seed digest (never its hex text).
"""
from __future__ import annotations

import hashlib
import heapq
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import common as P

ROOT = Path(__file__).resolve().parents[2]

SPLIT_MANIFEST = ROOT / "manifests/split_v1.parquet"
ACCOUNTING = ROOT / "manifests/m2_sample_accounting.parquet"
PAIRS_CONFIG = ROOT / "configs/frozen/pairs_v1.yaml"
EXEC_CONFIG = ROOT / "configs/execution/m2b_laptop_external_storage.yaml"

STATS_PATH = ROOT / "manifests/pair_train_stats_v1.json"
MANIFEST_PATH = {"TRAIN": ROOT / "manifests/pairs_train_v1.parquet",
                 "VAL": ROOT / "manifests/val_pairs_v1.parquet"}

SPEC_COLUMNS = ("pair_id", "target_live_id", "source_spoof_id", "dataset", "target_subject",
                "source_subject", "attack_macro", "d_pose", "d_scale", "d_luma", "seed")
AUDIT_COLUMNS = ("split", "d_pair", "source_video_id", "target_video_id",
                 "source_content_group_id", "target_content_group_id",
                 "candidate_count_eligible", "candidate_count_evaluated",
                 "source_sha256", "target_sha256", "split_manifest_sha256", "pairs_config_sha256",
                 "face_area_fraction_source", "face_area_fraction_target")
COLUMNS = SPEC_COLUMNS + AUDIT_COLUMNS

FLOAT_COLUMNS = {"d_pose", "d_scale", "d_luma", "d_pair",
                 "face_area_fraction_source", "face_area_fraction_target"}
INT_COLUMNS = {"seed", "candidate_count_eligible", "candidate_count_evaluated"}

# Frozen writer contract — proven byte-deterministic in M3 and re-proven here.
PARQUET_WRITER = {"compression": "zstd", "compression_level": 9, "write_statistics": True,
                  "version": "2.6", "use_dictionary": False, "row_group_size": 65536,
                  "store_schema": True, "write_page_index": False}
ROW_ORDER = ("dataset", "source_spoof_id")       # canonical; also the pair_id order

STATS_SCHEMA_VERSION = "pair_train_stats_v1"
# Canonical JSON: UTF-8, LF only, sorted keys, 2-space indent, ASCII-escaped, NaN/Inf rejected,
# floats via Python's shortest round-trip repr of the IEEE-754 double. One trailing newline.
JSON_POLICY = {"encoding": "utf-8", "newline": "LF", "sort_keys": True, "indent": 2,
               "ensure_ascii": True, "allow_nan": False, "separators": [",", ": "],
               "float_repr": "python_shortest_roundtrip_repr_of_ieee754_double",
               "trailing_newline": True}


# ------------------------------------------------------------------ helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_bytes(obj) -> bytes:
    """The frozen serialisation for every JSON artifact this module writes."""
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
                       separators=(",", ": ")) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """tmp -> fsync -> rename -> dir fsync. An authoritative manifest never appears half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


# ------------------------------------------------------------------ population
def _roots():
    cfg = yaml.safe_load(EXEC_CONFIG.read_text())
    from gpatbench.preprocess import m2b
    return m2b.resolve_roots(cfg)["roots"]


def load_pose(roots) -> dict:
    """Frozen FaceXFormer pose from the M2 geometry cache. The model is never re-run."""
    out = {}
    d = roots["geometry_cache_root"] / "pose_pitch_yaw_roll_rad"
    for f in sorted(d.glob("*.index.json")):
        meta = json.loads(f.read_text())
        blob = (d / meta["shard"]).read_bytes()
        for r in meta["rows"]:
            arr = np.load(io.BytesIO(blob[r["byte_offset"]:r["byte_offset"] + r["byte_length"]]),
                          allow_pickle=False)
            out[r["sample_id"]] = tuple(float(x) for x in arr)
    return out


def load_population(splits=P.SPLITS_WITH_PAIRS) -> dict:
    """Every frozen input a pair needs. Returns samples plus the provenance hashes they carry."""
    rows = pq.read_table(SPLIT_MANIFEST).to_pylist()
    acct = {r["sample_id"]: r for r in pq.read_table(ACCOUNTING).to_pylist()}
    roots = _roots()
    pose = load_pose(roots)

    import cv2

    def luma_of(r):
        p = roots["faces_256_root"] / r["dataset"] / f"{r['sample_id']}.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise P.PairPolicyError(f"{r['sample_id']}: canonical face missing at {p}")
        return P.luma_mean_from_rgb(np.ascontiguousarray(bgr[:, :, ::-1]))

    def fraction(r):
        a = acct[r["sample_id"]]
        if r["dataset"] == "casia_fasd":
            return P.CASIA_FACE_AREA_FRACTION                       # DEV-018
        h, w = (int(v) for v in a["frame_hw"].split("x"))
        return P.face_area_fraction(r["dataset"], json.loads(a["bbox"]), float(w), float(h))

    samples, meta = [], {}
    for r in rows:
        if r["split"] not in splits:
            continue                                                # TEST never enters M4
        a = acct[r["sample_id"]]
        if r["m2_status"] != "COMPLETE" or a["final_status"] != "COMPLETE":
            raise P.PairPolicyError(f"{r['sample_id']}: M2 status is not COMPLETE; "
                                    "a FAILED sample can never be a source or a target")
        samples.append(P.Sample(r["sample_id"], r["dataset"], r["split"], int(r["label_binary"]),
                                r["video_id"], r["subject_id_global"], r["content_group_id"],
                                r["attack_macro"], r["attack_raw"], r["sha256"],
                                pose[r["sample_id"]], fraction(r), luma_of(r)))
        meta[r["sample_id"]] = {"sha256": r["sha256"], "sha256_kind": r["sha256_kind"]}
    return {"samples": samples, "meta": meta,
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "pairs_config_sha256": sha256_file(PAIRS_CONFIG)}


# ------------------------------------------------------------------ TRAIN statistics (Q-25)
def build_train_stats(pop: dict) -> dict:
    datasets = sorted({s.dataset for s in pop["samples"]})
    stats = {ds: P.fit_pose_stats(pop["samples"], ds) for ds in datasets}
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "split_manifest_sha256": pop["split_manifest_sha256"],
        "pairs_config_sha256": pop["pairs_config_sha256"],
        "split_seed": P.SPLIT_SEED,
        "fitted_on": "TRAIN",
        "val_contribution": "none (VAL reuses these TRAIN statistics)",
        "test_contribution": "none",
        "dtype": "float64",
        "ddof": 0,
        "std_convention": "population",
        "axes": ["pitch", "yaw", "roll"],
        "units": "radians",
        "pose_source": "frozen M2 FaceXFormer geometry cache (pose_pitch_yaw_roll_rad)",
        "zero_variance_policy": f"std <= {P.MIN_POSE_STD} is a hard error",
        "json_policy": JSON_POLICY,
        "reproduction_note": (
            "This document records the SHA-256 of the two modules that produced it, so byte-exact "
            "reproduction requires those identical module bytes. The pose statistics themselves "
            "depend only on the TRAIN row set: rows are stacked in canonical sample_id order "
            "before the reduction, because floating-point accumulation is not associative."),
        "generator": {"module": "gpatbench.pairs.execute",
                      "function": "build_train_stats",
                      "common_module_sha256": sha256_file(ROOT / "gpatbench/pairs/common.py"),
                      "execute_module_sha256": sha256_file(ROOT / "gpatbench/pairs/execute.py"),
                      "numpy_version": np.__version__,
                      "pyarrow_version": pa.__version__},
        "datasets": {ds: {"n_train_complete": st.n_train_complete,
                          "pose_mean_pitch_yaw_roll": list(st.mean),
                          "pose_std_population_pitch_yaw_roll": list(st.std)}
                     for ds, st in stats.items()},
    }


def stats_to_posestats(doc: dict) -> dict:
    return {ds: P.PoseStats(ds, d["n_train_complete"],
                            tuple(d["pose_mean_pitch_yaw_roll"]),
                            tuple(d["pose_std_population_pitch_yaw_roll"]))
            for ds, d in doc["datasets"].items()}


def write_train_stats(doc: dict, path: Path = STATS_PATH) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(doc))


# ------------------------------------------------------------------ candidate selection (Q-24)
def select_candidates(source: P.Sample, eligible: list, split_seed: int = P.SPLIT_SEED,
                      cap: int = P.CANDIDATE_CAP) -> list:
    """Frozen Q-24 ranking. Identical results to `common.select_candidates`, one digest cheaper.

    The source-seed digest is computed once per source instead of once per candidate; the preimage
    bytes fed to SHA-256 are unchanged, and `tests/test_m4_execution.py` asserts equality against
    the reference implementation on real sources.
    """
    if not eligible:
        raise P.PairFeasibilityError(
            f"source {source.sample_id} ({source.dataset}/{source.split}) has no eligible live target")
    if len(eligible) <= cap:
        return sorted(eligible, key=lambda t: t.sample_id)
    seed = P.source_seed_digest(source.sample_id, split_seed)          # RAW 32 bytes

    def key(t):
        d = P.candidate_rank_digest(seed, t.sample_id)
        return (int.from_bytes(d, "big", signed=False), t.sample_id)

    return sorted(heapq.nsmallest(cap, eligible, key=key), key=lambda t: t.sample_id)


# ------------------------------------------------------------------ common pair construction
def build_pairs(pop: dict, split: str, stats: dict) -> list:
    """One row per spoof source of `split`, in canonical (dataset, source_spoof_id) order."""
    if split not in P.SPLITS_WITH_PAIRS:
        raise P.PairPolicyError(f"pairs are built for {P.SPLITS_WITH_PAIRS}, not {split}")
    here = [s for s in pop["samples"] if s.split == split]
    sources = P.canonical_source_order([s for s in here if s.label_binary == 1])
    live_by_dataset = {}
    for s in here:
        if s.label_binary == 0:
            live_by_dataset.setdefault(s.dataset, []).append(s)
    for ds in live_by_dataset:
        live_by_dataset[ds].sort(key=lambda t: t.sample_id)

    pair_ids = P.assign_pair_ids(split, sources)
    rows = []
    for pid, src in zip(pair_ids, sources):
        eligible = P.eligible_targets(src, live_by_dataset.get(src.dataset, []))
        cands = select_candidates(src, eligible)
        win = P.choose_target(src, cands, stats[src.dataset])
        tgt = win["target"]
        rows.append({
            "pair_id": pid,
            "target_live_id": tgt.sample_id,
            "source_spoof_id": src.sample_id,
            "dataset": src.dataset,
            "target_subject": tgt.subject_id_global,
            "source_subject": src.subject_id_global,
            "attack_macro": src.attack_macro,
            "d_pose": win["d_pose"], "d_scale": win["d_scale"], "d_luma": win["d_luma"],
            "seed": P.SPLIT_SEED,
            "split": split,
            "d_pair": win["d_pair"],
            "source_video_id": src.video_id, "target_video_id": tgt.video_id,
            "source_content_group_id": src.content_group_id,
            "target_content_group_id": tgt.content_group_id,
            "candidate_count_eligible": len(eligible),
            "candidate_count_evaluated": len(cands),
            "source_sha256": src.sha256, "target_sha256": tgt.sha256,
            "split_manifest_sha256": pop["split_manifest_sha256"],
            "pairs_config_sha256": pop["pairs_config_sha256"],
            "face_area_fraction_source": src.face_area_fraction,
            "face_area_fraction_target": tgt.face_area_fraction,
        })
    rows.sort(key=lambda r: tuple(r[c] for c in ROW_ORDER))
    return rows


def manifest_schema() -> pa.Schema:
    def t(c):
        if c in FLOAT_COLUMNS:
            return pa.float64()
        if c in INT_COLUMNS:
            return pa.int64()
        return pa.string()
    return pa.schema([(c, t(c)) for c in COLUMNS])


def write_manifest(rows: list, path: Path) -> str:
    """Deterministic Parquet, written atomically. Returns the file SHA-256."""
    schema = manifest_schema()
    table = pa.table({c: pa.array([r[c] for r in rows], schema.field(c).type) for c in COLUMNS},
                     schema=schema)
    buf = io.BytesIO()
    pq.write_table(table, buf, **PARQUET_WRITER)
    return atomic_write_bytes(path, buf.getvalue())


def schema_signature() -> str:
    sig = ";".join(f"{f.name}:{f.type}" for f in manifest_schema())
    return sha256_text(sig)


# ------------------------------------------------------------------ orchestration
def run(splits=P.SPLITS_WITH_PAIRS, write: bool = True, pop: dict | None = None,
        out_dir: Path | None = None, shuffle_seed: int | None = None) -> dict:
    """Build (and optionally write) every M4 common artifact.

    `out_dir` redirects the three files for a determinism rerun; `shuffle_seed` deliberately
    permutes the loaded population so the rerun proves order-independence rather than assuming it.
    Neither may change a single byte of the result.
    """
    pop = pop or load_population(splits)
    if shuffle_seed is not None:
        import random
        pop = dict(pop)
        pop["samples"] = list(pop["samples"])
        random.Random(shuffle_seed).shuffle(pop["samples"])
    doc = build_train_stats(pop)
    stats = stats_to_posestats(doc)
    stats_path = (out_dir / STATS_PATH.name) if out_dir else STATS_PATH
    out = {"pop": pop, "stats_doc": doc, "rows": {}, "sha256": {}}
    if write:
        out["sha256"][STATS_PATH.name] = write_train_stats(doc, stats_path)
    else:
        out["sha256"][STATS_PATH.name] = hashlib.sha256(canonical_json_bytes(doc)).hexdigest()
    for split in splits:
        rows = build_pairs(pop, split, stats)
        out["rows"][split] = rows
        path = (out_dir / MANIFEST_PATH[split].name) if out_dir else MANIFEST_PATH[split]
        if write:
            out["sha256"][MANIFEST_PATH[split].name] = write_manifest(rows, path)
    return out
