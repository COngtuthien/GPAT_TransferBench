# M2A — Q-23 Lossless Compression Pilot (2026-09-19)

Input: **only** the existing deterministic 24-sample smoke outputs (run `runC`, the rerun of the
frozen `M2A_SMOKE_MANIFEST`). No additional benchmark sample was processed to improve the estimate.
Per-sample rows: `M2A_COMPRESSION_PILOT.csv`; machine-readable summary: `M2A_COMPRESSION_PILOT.json`.

## Codec under test

`npy1+shuffle4+zstd` = ZSTD( byte-shuffle(4) ( `.npy` v1.0 bytes ) ), all settings fixed:
library `zstandard 0.25.0` (libzstd 1.5.7, backend `cext`),
level **10**, `write_checksum=False`, `write_content_size=True`,
`threads=0`, dtype `<f4`, order `C`, 256 rows per shard, `lossy=False`.

## Measurements (24 samples, float32 11×224×224 = 2,207,744 B each)

- raw float32 bytes: **52,985,856** (24 × 2,207,744)
- `.npy` bytes (raw + 128-byte self-describing header): 52,988,928

| zstd level | compressed bytes (24) | ratio | encode ms/sample | decode ms/sample | projected 20,640 samples (GB) | projected encode (h) |
|---|---|---|---|---|---|---|
| 1 | 40,540,958 | 1.3070 | 1.05 | 2.35 | 34.87 | 0.006 |
| 3 | 40,415,986 | 1.3111 | 1.40 | 2.26 | 34.76 | 0.008 |
| 10 | 40,389,638 | 1.3119 | 6.85 | 2.19 | 34.74 | 0.039 |
| 19 | 40,031,140 | 1.3237 | 223.99 | 2.42 | 34.43 | 1.284 |

**Frozen level: 10.** Level 19 buys 0.9 % space for ~33× the encode time
(1.28 h vs 0.039 h over the full set, against a ≈4.7 h preprocessing budget);
level 1–3 is marginally faster for slightly worse ratio. Level 10 is the operational middle and is
recorded as such — it is not a scientific parameter and changing it cannot change any stored value.

## Exactness (no tolerance allowed)

- `np.array_equal(original_float32_logits, decoded_float32_logits)` → **TRUE for all 24/24 samples**
- decoded `.npy` byte string **equals** the encoded one, byte for byte → TRUE for all samples
- decoded `array.tobytes()` equals the original raw buffer → TRUE for all samples
- re-encoding the same input yields **byte-identical** output → TRUE for all samples
- summary flags: `exact_reconstruction_all = true`, `deterministic_all = true`

The same round-trip is also executed inside every smoke sample (`logits_array_equal`,
`logits_bytes_equal`, `mask_from_stored_logits` in `M2A_SMOKE_RESULTS.csv`), and a failure raises
rather than being recorded silently.

## Projection to the full dataset (20,640 samples)

| | Size |
|---|---|
| Uncompressed float32 logits (the ≈45.6 GB figure) | **45.57 GB** |
| Lossless compressed logits at level 10 | **34.74 GB** |
| Saving | 10.83 GB (23.8 %) |
| Parsing mask uint8 224×224 | ≈1.04 GB |
| Landmarks + pose + embeddings (float32) | < 0.1 GB |
| `faces_256` canonical PNGs (measured mean 82,558 B over the 24 smoke faces) | ≈1.70 GB |

Float32 logits are high-entropy, so a lossless codec cannot do much better than this; the byte-plane
shuffle is what lifts the ratio from ≈1.15 (plain zstd) to ≈1.31. **No float16 fallback was considered**
— the owner decision forbids it, and the pilot is reported as measured rather than adjusted.

**Operational assessment (for the owner, not a blocker):** ≈35 GB of logits plus ≈1 GB of masks
plus ≈1.7 GB of canonical faces must live on the volume holding the model cache, which currently has
**66 GB free** (`/media/cong/Data`, 363 GB, 83 % used). Total M2B footprint ≈37.5 GB, i.e. ≈57 % of the
free space. That fits, but the owner may want to confirm the target volume for the M2B cache before
the full run, and no other large job should share that volume meanwhile.
