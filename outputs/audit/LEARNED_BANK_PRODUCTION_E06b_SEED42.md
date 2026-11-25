# M8 learned-bank production: E06b seed 42

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 34/34 gates passing. Run ID `07e64f33b19310b4`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E06b/seed_42` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `882017b8fadc532691c43699360147e036fa464e6b7bb0b6d8794e38f1568691` |
| checkpoint | run `aa96129289d080ba`, identity `7f805bf0f2da4f2f…` |
| authority | `configs/amendments/learned_bank_production_E06b.yaml` (`99c7494531217b25…`) |
| execution commit | `99ea8268fd0bbfd2cdeb51c1607307b9a1155ef6` (source `a1aef39a791cd25ae483c832b658e4f2e1d66c33`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E06b.json` (`9d7ec5a3d0115e34…`) |
| window | 2026-11-25T07:45:32Z → 2026-11-25T08:05:54Z (1203 s generation) |
| environment | `gpat-m6-e06c`, torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2993.0 GiB |

## Scientific content

- Population: synthetic stream indices 0..8837.
- RNG: torch.Generator().manual_seed(492027736) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 32.5 [p05 24.4, p95 49.4]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 0.

## Reproduction

- Command (`d80e3298995f3690…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E06b 42 99ea8268fd0bbfd2cdeb51c1607307b9a1155ef6`.
- C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
