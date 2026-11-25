# M8 learned-bank production: E05 seed 2026

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 37/37 gates passing. Run ID `6674eba825de2426`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E05/seed_2026` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `e2407a9ce312e3493cdb991af8d879e5ddefaaf85077e2fca0c7d35d31af30f8` |
| checkpoint | run `b739351daf9779bc`, identity `ee10a3b4ee1c66ec…` |
| authority | `configs/amendments/learned_bank_production_E05.yaml` (`a95b86e9f546cc2a…`) |
| execution commit | `125a8b3d5a1553e3fabbbc54cf4c2fdfb55e29a3` (source `65a99cf716ccb3ad8749a58ae050ea408961c39c`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E05.json` (`295776e56814eef1…`) |
| window | 2026-11-25T10:37:17Z → 2026-11-25T10:55:17Z (1069 s generation) |
| environment | `gpat-m6-e05`, torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2991.4 GiB |

## Scientific content

- Population: all 8838 frozen pairs_train_v1 rows in canonical 0-based order.
- RNG: torch.manual_seed(e05_pair_seed(seed, pair_id)) immediately before the single mix forward of each pair.
- Conversion: `util.tensor2im(mix[0]): clip((x+1)/2*255, 0, 255).astype(uint8) [swapping_autoencoder util/util.py:125-131]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 7.8 [p05 6.9, p95 8.2]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 57.

## Reproduction

- Command (`e56d21c918ee5390…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E05 2026 125a8b3d5a1553e3fabbbc54cf4c2fdfb55e29a3`.
- C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
