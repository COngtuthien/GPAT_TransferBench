"""Lossless deterministic storage for the FaceXFormer parsing logits (Q-23, owner-resolved).

Owner decision (Q-23 = RESOLVED_BY_OWNER_FLOAT32_FULL_LOGITS):
  * the scientific representation is the FULL float32 11x224x224 parsing-logit tensor;
  * the parsing mask is uint8 argmax OF THE STORED LOGITS (never stored instead of them);
  * no lossy quantisation, no float16, no mask-only reduction;
  * the ~45.6 GB figure is the UNCOMPRESSED estimate; storage uses LOSSLESS compression.

Compression settings are an OPERATIONAL STORAGE CHOICE, not a scientific hyper-parameter, but
they are recorded exactly (see `codec_provenance()` and configs/frozen/preprocess_v1.yaml):

  block := ZSTD( SHUFFLE4( NPY_V1( array ) ) )

  * NPY_V1  : numpy .npy format version (1, 0), allow_pickle=False, C order, dtype '<f4'
              (little-endian float32). Self-describing: dtype, shape and order travel with
              the bytes, so a decoded block is exactly the original array, not an
              interpretation of a raw buffer.
  * SHUFFLE4: deterministic byte-plane transpose of the payload (element size 4). Pure numpy,
              exactly invertible, no library dependency. It is applied to the whole .npy byte
              string, whose length is a multiple of 4 for a float32 array with the 64-byte
              aligned v1.0 header (asserted).
  * ZSTD    : python-zstandard, fixed level, fixed frame parameters, single-threaded ->
              byte-deterministic output for a given (library, level, input).

Shards: blocks are concatenated, in a fixed sample order, into shard files of a fixed number of
rows. Each block is independently addressable (offset + length) and independently hashed, so
random/partial access stays practical and corruption is detectable per block and per shard.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np

CODEC_ID = "npy1+shuffle4+zstd"
NPY_VERSION = (1, 0)
ELEMENT_SIZE = 4                 # float32
DTYPE = "<f4"                    # little-endian float32, explicit byte order
ARRAY_ORDER = "C"
ZSTD_LEVEL = 10                  # fixed operational setting (pilot: M2A_COMPRESSION_PILOT.md)
ZSTD_WRITE_CHECKSUM = False      # block integrity is covered by the recorded sha256
ZSTD_WRITE_CONTENT_SIZE = True
ZSTD_THREADS = 0                 # single-threaded: deterministic frames
SHARD_ROWS = 256                 # rows per shard file (fixed; deterministic sharding)
PARSING_LOGITS_SHAPE = (11, 224, 224)


def _compressor():
    import zstandard as zstd
    params = zstd.ZstdCompressionParameters.from_level(
        ZSTD_LEVEL, write_checksum=ZSTD_WRITE_CHECKSUM, write_content_size=ZSTD_WRITE_CONTENT_SIZE,
        write_dict_id=False, threads=ZSTD_THREADS)
    return zstd.ZstdCompressor(compression_params=params)


def codec_provenance() -> dict:
    """Everything needed to reproduce the stored bytes (recorded in the frozen config/ledger)."""
    import zstandard as zstd
    return {
        "codec_id": CODEC_ID,
        "library": "zstandard",
        "library_version": zstd.__version__,
        "libzstd_version": ".".join(str(v) for v in zstd.ZSTD_VERSION),
        "backend": zstd.backend,
        "level": ZSTD_LEVEL,
        "write_checksum": ZSTD_WRITE_CHECKSUM,
        "write_content_size": ZSTD_WRITE_CONTENT_SIZE,
        "threads": ZSTD_THREADS,
        "byte_shuffle_element_size": ELEMENT_SIZE,
        "container": f"npy v{NPY_VERSION[0]}.{NPY_VERSION[1]} (allow_pickle=False)",
        "dtype": DTYPE,
        "order": ARRAY_ORDER,
        "shard_rows": SHARD_ROWS,
        "lossy": False,
    }


# ------------------------------------------------------------------ deterministic serialization
def npy_bytes(arr: np.ndarray) -> bytes:
    """Canonical .npy v1.0 serialization of a C-order little-endian float32 array."""
    if arr.dtype != np.dtype(DTYPE):
        raise ValueError(f"expected dtype {DTYPE}, got {arr.dtype.str}")
    a = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.lib.format.write_array(buf, a, version=NPY_VERSION, allow_pickle=False)
    b = buf.getvalue()
    if len(b) % ELEMENT_SIZE:
        raise ValueError("npy byte length is not a multiple of the element size")
    return b


def npy_load(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def shuffle4(b: bytes) -> bytes:
    """Deterministic byte-plane transpose (exactly invertible; no information is lost)."""
    return np.frombuffer(b, np.uint8).reshape(-1, ELEMENT_SIZE).T.copy().tobytes()


def unshuffle4(b: bytes) -> bytes:
    return np.frombuffer(b, np.uint8).reshape(ELEMENT_SIZE, -1).T.copy().tobytes()


def encode_block(arr: np.ndarray) -> bytes:
    return _compressor().compress(shuffle4(npy_bytes(arr)))


def decode_block(blob: bytes) -> np.ndarray:
    import zstandard as zstd
    return npy_load(unshuffle4(zstd.ZstdDecompressor().decompress(blob)))


def decode_block_npy_bytes(blob: bytes) -> bytes:
    """The exact .npy byte string that was compressed (for raw byte-level verification)."""
    import zstandard as zstd
    return unshuffle4(zstd.ZstdDecompressor().decompress(blob))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def mask_from_logits(logits: np.ndarray) -> np.ndarray:
    """Parsing mask = uint8 argmax OF THE STORED LOGITS (owner decision Q-23)."""
    if logits.ndim != 3 or logits.dtype != np.float32:
        raise ValueError(f"parsing logits must be 3-D float32, got {logits.shape} {logits.dtype}")
    if logits.shape[0] > 256:
        raise ValueError("argmax would not fit in uint8")
    return logits.argmax(axis=0).astype(np.uint8)


# ------------------------------------------------------------------ shards
@dataclass
class ShardWriter:
    """Append blocks to shard files of SHARD_ROWS rows, in the caller's (fixed) sample order."""
    out_dir: object
    prefix: str = "shard"
    rows_per_shard: int = SHARD_ROWS
    index: list = field(default_factory=list)
    _n: int = 0
    _fh: object = None
    _shard_name: str = ""
    _shard_hash: object = None

    def _roll(self) -> None:
        if self._n % self.rows_per_shard == 0:
            self.close()
            self._shard_name = f"{self.prefix}-{self._n // self.rows_per_shard:05d}.bin"
            self._fh = open(self.out_dir / self._shard_name, "wb")
            self._shard_hash = hashlib.sha256()

    def add(self, sample_id: str, arr: np.ndarray) -> dict:
        self._roll()
        blob = encode_block(arr)
        offset = self._fh.tell()
        self._fh.write(blob)
        self._shard_hash.update(blob)
        row = {"sample_id": sample_id, "shard": self._shard_name, "byte_offset": offset,
               "byte_length": len(blob), "block_sha256": sha256(blob),
               "npy_sha256": sha256(npy_bytes(arr)), "shape": list(arr.shape),
               "dtype": DTYPE, "codec_id": CODEC_ID, "row_in_shard": self._n % self.rows_per_shard}
        self.index.append(row)
        self._n += 1
        return row

    def close(self) -> dict | None:
        if self._fh is None:
            return None
        self._fh.close()
        done = {"shard": self._shard_name, "sha256": self._shard_hash.hexdigest()}
        self._fh = None
        return done


def read_block(shard_path, byte_offset: int, byte_length: int) -> np.ndarray:
    """Random access: read exactly one block and decode it, without touching the other rows."""
    with open(shard_path, "rb") as f:
        f.seek(byte_offset)
        blob = f.read(byte_length)
    if len(blob) != byte_length:
        raise ValueError(f"short read at {byte_offset} in {shard_path}")
    return decode_block(blob)


# ------------------------------------------------------------------ integrity audit
def verify_shard_index(index_rows, shard_dir, *, check_blocks: bool = True) -> dict:
    """Audit a logit-shard index: no duplicates, no orphans, every block hash verifies.

    `index_rows` are the rows produced by `ShardWriter.add` (or read back from the cache index).
    Returns a report; `ok` is True only when every check passes. This is the reusable core of the
    M2B cache-integrity audit, so the same code path is exercised by the tests and by the run.
    """
    from pathlib import Path as _Path
    shard_dir = _Path(shard_dir)
    rows = list(index_rows)
    seen, dup = set(), []
    missing_shards, bad_hash, bad_shape, overlaps = [], [], [], []
    by_shard: dict[str, list] = {}
    for r in rows:
        if r["sample_id"] in seen:
            dup.append(r["sample_id"])
        seen.add(r["sample_id"])
        by_shard.setdefault(r["shard"], []).append(r)

    for shard, rs in by_shard.items():
        path = shard_dir / shard
        if not path.is_file():
            missing_shards.append(shard)
            continue
        spans = sorted((r["byte_offset"], r["byte_offset"] + r["byte_length"], r["sample_id"]) for r in rs)
        for (a0, a1, sid_a), (b0, _b1, sid_b) in zip(spans, spans[1:]):
            if b0 < a1:
                overlaps.append((sid_a, sid_b))
        size = path.stat().st_size
        covered = sum(r["byte_length"] for r in rs)
        if covered != size:
            missing_shards.append(f"{shard}: index covers {covered} of {size} bytes (orphan bytes)")
        if not check_blocks:
            continue
        for r in rs:
            # A corrupt block must be REPORTED, never raised: this audit runs over a whole cache.
            with open(path, "rb") as f:
                f.seek(r["byte_offset"])
                blob = f.read(r["byte_length"])
            if sha256(blob) != r["block_sha256"]:
                bad_hash.append(r["sample_id"])
                continue
            try:
                arr = decode_block(blob)
            except Exception:
                bad_hash.append(r["sample_id"])
                continue
            if sha256(npy_bytes(arr)) != r["npy_sha256"]:
                bad_hash.append(r["sample_id"])
            if list(arr.shape) != list(r["shape"]) or arr.dtype != np.dtype(r["dtype"]):
                bad_shape.append(r["sample_id"])

    orphan_files = sorted(p.name for p in shard_dir.glob("*.bin") if p.name not in by_shard)
    rep = {"rows": len(rows), "unique_sample_ids": len(seen), "duplicate_sample_ids": sorted(set(dup)),
           "missing_or_short_shards": missing_shards, "orphan_shard_files": orphan_files,
           "overlapping_blocks": overlaps, "bad_block_hashes": sorted(set(bad_hash)),
           "bad_shape_or_dtype": sorted(set(bad_shape)), "blocks_verified": check_blocks}
    rep["ok"] = not (dup or missing_shards or orphan_files or overlaps or bad_hash or bad_shape)
    return rep
