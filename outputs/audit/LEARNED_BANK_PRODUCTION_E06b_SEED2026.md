# M8 learned-bank production: E06b seed 2026

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 34/34 gates passing. Run ID `e2d2ba62af40de21`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E06b/seed_2026` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `e1dd3a2eaa990b05c54fa6bf2939e40a77697eb58ae19e7964ffa9aba042f88c` |
| checkpoint | run `ec71778b28479de5`, identity `b9a0465c4759f744…` |
| authority | `configs/amendments/learned_bank_production_E06b.yaml` (`99c7494531217b25…`) |
| execution commit | `99ea8268fd0bbfd2cdeb51c1607307b9a1155ef6` (source `a1aef39a791cd25ae483c832b658e4f2e1d66c33`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E06b.json` (`9d7ec5a3d0115e34…`) |
| window | 2026-11-25T08:31:31Z → 2026-11-25T08:51:53Z (1204 s generation) |
| environment | `gpat-m6-e06c`, torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2991.9 GiB |

## Scientific content

- Population: synthetic stream indices 0..8837.
- RNG: torch.Generator().manual_seed(1539980013) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 31.0 [p05 22.8, p95 48.8]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 0.

## Reproduction

- Command (`e5c09c4438b44348…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E06b 2026 99ea8268fd0bbfd2cdeb51c1607307b9a1155ef6`.
- C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
