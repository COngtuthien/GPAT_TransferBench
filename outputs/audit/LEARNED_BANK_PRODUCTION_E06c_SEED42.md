# M8 learned-bank production: E06c seed 42

**Status:** `LEARNED_BANK_PRODUCTION_PASSED`, 34/34 gates passing. Run ID `45e035b85c735361`.

Failed gates: none.

| | |
|---|---|
| bank | `/media/ext_data/student20261_gpat_m8_banks/E06c/seed_42` |
| rows / outputs / failures | 8838 / 8838 / 0 |
| manifest sha256 | `8da6d40458a2ee420330c92ef3a956d5e06be0560c96f86353d107d48a06ab62` |
| checkpoint | run `27dcaf23aac18d7a`, identity `1f7b03e23ffcbcb4…` |
| authority | `configs/amendments/learned_bank_production_E06c.yaml` (`34ddaa1dba95be2b…`) |
| execution commit | `4f5d13b3177748922470bf11599a4e7f677d7b04` (source `271fdecc096a8ccd7cafca5fcbb383bbf21818ee`) |
| S7 record | `outputs/audit/LEARNED_BANK_S7_E06c.json` (`42b1aef1c6283eae…`) |
| window | 2026-11-25T05:10:30Z → 2026-11-25T05:30:25Z (1177 s generation) |
| environment | `gpat-m6-e06c`, torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` driver 580.178.04 |
| storage | HDD UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data`; free after 2997.2 GiB |

## Scientific content

- Population: synthetic stream indices 0..8837.
- RNG: torch.Generator().manual_seed(1660457813) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- S7-to-production equivalence: the 186 approved S7 toy rows equal their production rows byte for byte (manifest fields other than `mode`, and PNG bytes), for S7 runs A and B.
- Descriptive statistics (no threshold): per-image std median 40.5 [p05 29.5, p95 53.0]; distinct pixel hashes 8838 of 8838; exact duplicate rows 0; rows sharing a coarse 8x8 luminance signature 0.

## Reproduction

- Command (`ea715d9ec044836e…`): see the JSON record; launch script `tools/learned_bank_production_scripts/m8prod_launch.sh E06c 42 4f5d13b3177748922470bf11599a4e7f677d7b04`.
- C-Q2/C-Q3: bitwise A/B replay is established by the passed S7 record; the production audit requires the 186 S7 toy rows to equal their production rows byte for byte (runs A and B). A second full production bank is not required by the contract and was not generated.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
