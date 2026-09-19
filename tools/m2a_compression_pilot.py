"""Q-23 lossless compression pilot on the existing deterministic 24-sample smoke logits.

Measures, for the FaceXFormer parsing logits (float32 11x224x224):
  raw float32 bytes, compressed bytes, compression ratio, encode time, decode time, and EXACT
  reconstruction (np.array_equal on the arrays AND byte equality of the .npy representation).
No tolerance is allowed: the cache is lossless.

    python tools/m2a_compression_pilot.py <run_tag> [--sweep]

`--sweep` additionally measures a fixed set of zstd levels to justify the frozen level as an
operational choice. Nothing here selects a model or a scientific parameter, and no new benchmark
sample is processed: the pilot only re-reads logits that the smoke already produced.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpatbench.preprocess import logit_store as LS  # noqa: E402

N_FULL_SAMPLES = 20640          # M1 inventory sample count (manifests/inventory.parquet)
SWEEP_LEVELS = (1, 3, 10, 19)
CSV_OUT = ROOT / "outputs/audit/M2A_COMPRESSION_PILOT.csv"
JSON_OUT = ROOT / "outputs/audit/M2A_COMPRESSION_PILOT.json"


def _codec(level: int):
    import zstandard as zstd
    return zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(
        level, write_checksum=LS.ZSTD_WRITE_CHECKSUM, write_content_size=LS.ZSTD_WRITE_CONTENT_SIZE,
        write_dict_id=False, threads=LS.ZSTD_THREADS))


def measure(arr: np.ndarray, level: int) -> dict:
    import zstandard as zstd
    npy = LS.npy_bytes(arr)
    shuf = LS.shuffle4(npy)
    t = time.perf_counter(); blob = _codec(level).compress(shuf); enc = time.perf_counter() - t
    t = time.perf_counter(); back_npy = LS.unshuffle4(zstd.ZstdDecompressor().decompress(blob)); dec = time.perf_counter() - t
    back = LS.npy_load(back_npy)
    return {"level": level, "raw_float32_bytes": int(arr.nbytes), "npy_bytes": len(npy),
            "compressed_bytes": len(blob), "ratio": len(npy) / len(blob),
            "encode_s": enc, "decode_s": dec,
            "array_equal": bool(np.array_equal(arr, back)),
            "npy_bytes_equal": bool(back_npy == npy),
            "raw_bytes_equal": bool(back.tobytes() == arr.tobytes()),
            "deterministic_repeat": bool(bytes(_codec(level).compress(shuf)) == bytes(blob))}


def main() -> int:
    tag = sys.argv[1]
    sweep = "--sweep" in sys.argv[2:]
    files = sorted((ROOT / "outputs/exploratory/m2a_smoke" / tag / "geometry").glob("*__parsing_logits.npy"))
    if not files:
        raise SystemExit(f"no parsing logits under run {tag}")
    rows = []
    for f in files:
        a = np.load(f, allow_pickle=False)
        if a.dtype != np.float32 or a.shape != LS.PARSING_LOGITS_SHAPE:
            raise SystemExit(f"unexpected logits {a.shape} {a.dtype} in {f.name}")
        for lvl in (SWEEP_LEVELS if sweep else (LS.ZSTD_LEVEL,)):
            rows.append({"sample_id": f.name.split("__")[0], **measure(a, lvl)})
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    summary = {"run_tag": tag, "n_samples": len(files), "codec": LS.codec_provenance(),
               "exact_reconstruction_all": all(r["array_equal"] and r["npy_bytes_equal"] and r["raw_bytes_equal"] for r in rows),
               "deterministic_all": all(r["deterministic_repeat"] for r in rows),
               "n_full_samples_projection": N_FULL_SAMPLES, "by_level": {}}
    for lvl in sorted({r["level"] for r in rows}):
        sub = [r for r in rows if r["level"] == lvl]
        raw = sum(r["raw_float32_bytes"] for r in sub)
        npyb = sum(r["npy_bytes"] for r in sub)
        comp = sum(r["compressed_bytes"] for r in sub)
        summary["by_level"][str(lvl)] = {
            "raw_float32_bytes": raw, "npy_bytes": npyb, "compressed_bytes": comp,
            "ratio_npy_over_compressed": npyb / comp, "ratio_raw_over_compressed": raw / comp,
            "mean_encode_s": sum(r["encode_s"] for r in sub) / len(sub),
            "mean_decode_s": sum(r["decode_s"] for r in sub) / len(sub),
            "projected_full_compressed_gb": comp / len(sub) * N_FULL_SAMPLES / 1e9,
            "projected_full_raw_gb": raw / len(sub) * N_FULL_SAMPLES / 1e9,
            "projected_full_encode_hours": sum(r["encode_s"] for r in sub) / len(sub) * N_FULL_SAMPLES / 3600.0}
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
