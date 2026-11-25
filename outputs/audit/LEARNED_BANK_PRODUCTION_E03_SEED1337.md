# M8 learned-bank production: E03 seed 1337

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 34/34 gates passing. Run ID `db67fcf77751b4ed`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E03/seed_1337` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `c414758d373de2461a805cc785a6bd0767fdd18071132039d6d8b725f49121a3` |
| checkpoint | run `e8c85cc378e778f5`, identity `cab2d359906c3680…` |
| authority | `configs/amendments/learned_bank_production_E03.yaml` (`235b1dd2b1c99f40…`) |
| execution commit | `4c43b5182b3a14211f8ac5fdefdbed093bd755af` (source `6549ff76e4567b89f4d598279e21fdff90c23acd`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E03.json` (`c2d089ce62c37101…`) |
| window | 2026-11-25T06:44:09Z → 2026-11-25T07:05:48Z (1277 s generation) |
| environment | `gpat-m6-e03`, tensorflow 1.15.5, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2994.4 GiB |

## Scientific content

- Population: all 8838 frozen pairs_train_v1 rows in canonical 0-based order.
- RNG: none (E03 inference is deterministic; no sampling).
- Conversion: `tf.cast(tf.clip_by_value(synth1, 0.0, 1.0) * 255, tf.uint8) [stdn model/utils.py:24,37]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 58.5 [p05 34.2, p95 77.3]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 0.

## Reproduction

- Command (`27208a166e76a26a…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E03 1337 4c43b5182b3a14211f8ac5fdefdbed093bd755af`.
- C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
