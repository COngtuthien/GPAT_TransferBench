"""M2B full preprocessing engine: resumable, atomic, bounded-memory, externally rooted.

Scientific behaviour is entirely defined by `configs/frozen/preprocess_v1.yaml` and the frozen
adapters in this package. This module only *drives* them: where to write, in what order, how to
resume, and how to prove what it wrote. Nothing here may change a pixel or a tensor value.

Layout of the work
------------------
Samples come from the frozen M1 inventory and are ordered deterministically by
(dataset, video_id, frame_index, sample_id). That order fixes a global ordinal per sample, and the
ordinal fixes the shard: `shard = ordinal // shard_rows`. Shard membership is therefore a pure
function of the frozen manifest and never depends on run history, worker count or failures.
Grouping by video also lets one sequential decode pass serve a video's eight samples
(`read_video_frames`), which is the same decoder semantics at ~4x less decode work.

A chunk (= one shard's worth of samples) is the unit of finalization:

    for each sample in chunk:  frame -> face -> FaceXFormer -> AdaFace, appending one
                               compressed/raw block per field to that field's `.tmp` shard
    at end of chunk:           fsync -> read back and verify every block hash -> atomic rename
                               -> write the shard index atomically -> mark the samples COMPLETE

So memory holds one sample's tensors at a time, disk holds at most one in-flight shard per field,
and no uncompressed whole-cache copy ever exists.

Resume
------
State is an append-only JSONL log per worker; the last record for a `sample_id` wins. A sample's
frame/face artifacts are reused only after their SHA-256 is recomputed and matches the recorded
value, so a truncated or foreign file forces regeneration rather than being trusted. Geometry and
identity live in shards, so they are only ever marked done once their whole shard is finalized: an
interrupted chunk is simply rebuilt, which is why a resume can never duplicate or half-write a
cache record.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import contracts as C
from . import logit_store as LS
from .frames import read_image_frame, read_video_frames

# Fields written as raw (uncompressed) deterministic .npy blocks, and their cache role.
RAW_FIELDS = {
    "parsing_mask": "geometry",
    "landmarks_norm": "geometry",
    "landmarks_px224": "geometry",
    "landmarks_px256": "geometry",
    "pose_pitch_yaw_roll_rad": "geometry",
    "embedding": "identity",
}
COMPRESSED_FIELDS = {"parsing_logits": "geometry"}      # frozen lossless codec (Q-23)
ALL_FIELDS = {**COMPRESSED_FIELDS, **RAW_FIELDS}

STATES = ("PENDING", "FRAME_DONE", "FACE_DONE", "GEOMETRY_DONE", "IDENTITY_DONE", "COMPLETE", "FAILED")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """tmp -> write -> flush -> fsync -> atomic rename -> fsync(dir). Returns the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
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
    return sha256_bytes(data)


def npy_bytes_any(arr: np.ndarray) -> bytes:
    """Deterministic .npy v1.0 bytes for any dtype (the float32 logits use `logit_store`)."""
    import io
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), version=LS.NPY_VERSION, allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------- shard construction
@dataclass
class ChunkShardSet:
    """One in-flight shard per field, for a single chunk. Bounded: blocks stream straight to disk."""
    roots: dict            # field -> Path of that field's directory
    shard_index: int
    _fh: dict = field(default_factory=dict)
    _index: dict = field(default_factory=dict)

    def _name(self, fieldname: str) -> str:
        return f"{fieldname}-{self.shard_index:05d}.bin"

    def open(self) -> None:
        for fieldname, root in self.roots.items():
            root.mkdir(parents=True, exist_ok=True)
            tmp = root / (self._name(fieldname) + ".tmp")
            self._fh[fieldname] = open(tmp, "wb")
            self._index[fieldname] = []

    def add(self, fieldname: str, sample_id: str, arr: np.ndarray) -> None:
        compressed = fieldname in COMPRESSED_FIELDS
        blob = LS.encode_block(arr) if compressed else npy_bytes_any(arr)
        fh = self._fh[fieldname]
        offset = fh.tell()
        fh.write(blob)
        self._index[fieldname].append({
            "sample_id": sample_id, "shard": self._name(fieldname), "byte_offset": offset,
            "byte_length": len(blob), "block_sha256": sha256_bytes(blob),
            "npy_sha256": sha256_bytes(npy_bytes_any(arr)), "shape": list(arr.shape),
            "dtype": arr.dtype.str, "codec_id": LS.CODEC_ID if compressed else "npy1",
            "row_in_shard": len(self._index[fieldname])})

    def abort(self) -> None:
        for fieldname, fh in self._fh.items():
            try:
                fh.close()
            except OSError:
                pass
            tmp = self.roots[fieldname] / (self._name(fieldname) + ".tmp")
            if tmp.exists():
                tmp.unlink()
        self._fh, self._index = {}, {}

    def finalize(self) -> dict:
        """fsync, verify every block by reading it back, then atomically publish shard + index.

        Any failure leaves the shard unpublished; all handles are closed either way, so a rejected
        chunk cannot leak descriptors into a long run.
        """
        out = {}
        try:
            return self._finalize_all(out)
        finally:
            for fh in self._fh.values():
                try:
                    fh.close()
                except (OSError, ValueError):
                    pass
            self._fh, self._index = {}, {}

    def _finalize_all(self, out: dict) -> dict:
        for fieldname, fh in self._fh.items():
            if not fh.closed:
                fh.flush()
                os.fsync(fh.fileno())
                fh.close()
            root = self.roots[fieldname]
            tmp, final = root / (self._name(fieldname) + ".tmp"), root / self._name(fieldname)
            rows = self._index[fieldname]
            with open(tmp, "rb") as f:
                blob_all = f.read()
            if len(blob_all) != sum(r["byte_length"] for r in rows):
                raise RuntimeError(f"{fieldname} shard {self.shard_index}: size does not match the index")
            for r in rows:                                   # verify before publishing
                blob = blob_all[r["byte_offset"]:r["byte_offset"] + r["byte_length"]]
                if sha256_bytes(blob) != r["block_sha256"]:
                    raise RuntimeError(f"{fieldname} shard {self.shard_index}: block hash mismatch for {r['sample_id']}")
                arr = LS.decode_block(blob) if fieldname in COMPRESSED_FIELDS else np.load(
                    __import__("io").BytesIO(blob), allow_pickle=False)
                if list(arr.shape) != r["shape"] or arr.dtype.str != r["dtype"]:
                    raise RuntimeError(f"{fieldname} shard {self.shard_index}: shape/dtype mismatch for {r['sample_id']}")
                if sha256_bytes(npy_bytes_any(arr)) != r["npy_sha256"]:
                    raise RuntimeError(f"{fieldname} shard {self.shard_index}: decoded payload mismatch for {r['sample_id']}")
            os.replace(tmp, final)
            fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            shard_sha = sha256_bytes(blob_all)
            atomic_write_bytes(root / f"{self._name(fieldname)}.index.json", (json.dumps(
                {"field": fieldname, "shard": self._name(fieldname), "shard_sha256": shard_sha,
                 "rows": rows, "codec_id": LS.CODEC_ID if fieldname in COMPRESSED_FIELDS else "npy1"},
                indent=1) + "\n").encode("utf-8"))
            out[fieldname] = {"shard": self._name(fieldname), "sha256": shard_sha, "rows": len(rows)}
        return out

    def is_finalized(self) -> bool:
        return all((root / f"{self._name(f)}.index.json").is_file() and (root / self._name(f)).is_file()
                   for f, root in self.roots.items())


# ---------------------------------------------------------------- state log
class StateLog:
    """Append-only, crash-safe per-worker JSONL. Last record per sample_id wins."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, **rec) -> None:
        rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def load(runstate_root: Path) -> dict:
        """Merge every worker log; the last record for a sample_id wins."""
        latest: dict = {}
        for p in sorted(Path(runstate_root).glob("m2b_state.*.jsonl")):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # a torn final line after a crash: ignore it
                    if "sample_id" in rec:
                        latest[rec["sample_id"]] = rec
        return latest


# ---------------------------------------------------------------- work plan
def build_plan(inventory_rows, shard_rows: int) -> list:
    """Deterministic ordinal/shard assignment. Pure function of the frozen manifest."""
    rows = sorted(inventory_rows, key=lambda r: (r["dataset"], r["video_id"], r["frame_index"], r["sample_id"]))
    for i, r in enumerate(rows):
        r["_ordinal"] = i
        r["_shard"] = i // shard_rows
    return rows


def chunk_of(plan, shard_index: int) -> list:
    return [r for r in plan if r["_shard"] == shard_index]


class RootContainmentError(RuntimeError):
    """An execution config points an output root outside the owner-approved runtime root."""


def resolve_roots(xcfg: dict) -> dict:
    """Resolve the physical output roots of an execution config and enforce containment.

    Storage relocation (DEV-017) is owner-approved for ONE named runtime root. Any role that
    resolves outside it — including via `..`, a symlink or an absolute path elsewhere — is refused
    here rather than silently writing scientific outputs to an unapproved filesystem.
    """
    runtime_root = Path(xcfg["storage"]["runtime_root"]).resolve()
    roots = {k: Path(v) for k, v in xcfg["storage"]["roots"].items()}
    escaping = sorted(k for k, v in roots.items() if not v.resolve().is_relative_to(runtime_root))
    if escaping:
        raise RootContainmentError(f"output roots escape {runtime_root}: {escaping}")
    return {"runtime_root": runtime_root, "roots": roots,
            "runstate_root": Path(xcfg["storage"]["runstate_root"]),
            "tmp_root": Path(xcfg["storage"]["tmp_root"])}


def field_root(roots: dict, field: str) -> Path:
    """Physical directory of one cache field, derived from its role (geometry or identity)."""
    role = ALL_FIELDS[field]
    base = roots["geometry_cache_root"] if role == "geometry" else roots["identity_cache_root"]
    return Path(base) / field
