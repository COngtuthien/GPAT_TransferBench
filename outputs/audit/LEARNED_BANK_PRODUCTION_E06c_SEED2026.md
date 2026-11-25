# M8 learned-bank production: E06c seed 2026

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 34/34 gates passing. Run ID `bfc0e4a6a947c9c4`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E06c/seed_2026` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `d30ad186264074a670962e46d535e8a6c6b471004803017027b265fab4f18bee` |
| checkpoint | run `86b631403e3c4ee1`, identity `b8143768e7a96652…` |
| authority | `configs/amendments/learned_bank_production_E06c.yaml` (`34ddaa1dba95be2b…`) |
| execution commit | `4f5d13b3177748922470bf11599a4e7f677d7b04` (source `271fdecc096a8ccd7cafca5fcbb383bbf21818ee`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E06c.json` (`42b1aef1c6283eae…`) |
| window | 2026-11-25T05:53:46Z → 2026-11-25T06:13:36Z (1166 s generation) |
| environment | `gpat-m6-e06c`, torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2996.0 GiB |

## Scientific content

- Population: synthetic stream indices 0..8837.
- RNG: torch.Generator().manual_seed(1718284219) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 41.8 [p05 30.1, p95 54.8]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 0.

## Reproduction

- Command (`095c40004e4ebd65…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E06c 2026 4f5d13b3177748922470bf11599a4e7f677d7b04`.
- C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
