# M8 learned-bank production: E04 seed 1337

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 43/43 gates passing. Run ID `65a3704819bc715a`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E04/seed_1337` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `83993dc73979b8bdfe714b9588ada874a9da7415f8e9f80405e77f75a64b6f34` |
| checkpoint | run `a42fdd2a2accaf0a`, identity `4f183d2143d27d1d…` |
| authority | `configs/amendments/learned_bank_production_E04.yaml` (`9d714c0978456d0a…`) |
| execution commit | `fabc1ef5069c05cd1d39e17057a90148fa296220` (source `cc2834ce15df6fff587da7c6e4c81d2328a57562`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E04.json` (`138e1ef988970a7c…`) |
| window | 2026-11-25T15:25:38Z → 2026-11-25T15:50:22Z (1466 s generation) |
| environment | `gpat-m6-e03`, tensorflow 1.15.5, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2989.6 GiB |

## Scientific content

- Population: all 8838 frozen pairs_train_v1 rows in canonical 0-based order.
- RNG: none (deterministic inference; no sampling or augmentation).
- Conversion: `(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8) float32 [E04-Q3 A_PREDECESSOR_CLIP_TRUNCATE; reference stdn model/utils.py:24,37]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 54.5 [p05 31.4, p95 73.9]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 0.

## Reproduction

- Command (`9c57e51e20838c73…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E04 1337 fabc1ef5069c05cd1d39e17057a90148fa296220`.
- E04-Q5: bitwise A/B replay is established by the passed S7 record; this audit requires the 186 S7 toy rows to equal their production rows byte for byte and pixel for pixel (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
## E04 specifics

- Reporting label: Physics-STD (controlled geometry/depth reconstruction). Fidelity: `CONTROLLED_ADAPTATION` (not an official native Physics-STD reproduction).
- Checkpoint iteration 150000, run `a42fdd2a2accaf0a`; component sha256 before = after; mtimes unchanged.
- S7-to-production (manifest-matched by global position and pair_id): {"A": {"byte_mismatch_count": 0, "field_mismatch_count": 0, "pixel_arrays_equal": 186, "pixel_mismatch_count": 0, "rows_matched": 186}, "B": {"byte_mismatch_count": 0, "field_mismatch_count": 0, "pixel_arrays_equal": 186, "pixel_mismatch_count": 0, "rows_matched": 186}}.
- Raw-output observation (no gate): 4401 rows with raw min < 0, 5357 rows with raw max > 1; extreme raw range [-1.0828304290771484, 2.8766257762908936]; clipped to [0, 1] by the approved conversion.
- GPU (nvidia-smi samples, new PIDs): {"1474231": 4378, "1475452": 582, "1493241": 4384, "1520737": 2484, "1563434": 5650, "1578293": 4680}.

## Disclosures

- CONTROLLED_ADAPTATION: Physics-STD (controlled geometry/depth reconstruction); not an official native Physics-STD reproduction
- bank inference uses the stored BN moving statistics (batch 1, E04-Q1); training-time synthesis used 8-image batch statistics
- float->uint8 clip-truncate (E04-Q3) follows the ECCV20-STDN predecessor (reference only); raw values outside [0, 1] are clipped, never rescaled; S7 observed one seed-42 pair with raw range about [-1.11, 1.68]
- RAW-MATCHED: every numerically valid output is included; no quality filter, replacement or threshold
- TRAIN allowlist from the pinned e04_feed_pack_v1 in the inference process (owner ruling); split_v1 / M2 cross-check performed by this independent auditor
