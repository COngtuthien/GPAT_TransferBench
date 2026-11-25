# M8 learned-bank production family closure: E06b

**Status:** `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`, 15/15 closure gates passing; 26514 outputs.

Failed gates: none.

| seed | run ID | outputs | manifest sha256 | audit gates | window |
|---|---|---|---|---|---|
| 42 | `07e64f33b19310b4` | 8838 | `882017b8fadc5326…` | 34/34 | 2026-11-25T07:45:32Z → 2026-11-25T08:05:54Z |
| 1337 | `2037be09ba25acdc` | 8838 | `40564cd58982eae4…` | 34/34 | 2026-11-25T08:08:28Z → 2026-11-25T08:28:54Z |
| 2026 | `e2d2ba62af40de21` | 8838 | `e1dd3a2eaa990b05…` | 34/34 | 2026-11-25T08:31:31Z → 2026-11-25T08:51:53Z |

- Authority `configs/amendments/learned_bank_production_E06b.yaml` (`99c7494531217b25…`); execution commit `99ea8268fd0bbfd2cdeb51c1607307b9a1155ef6`; source commit `a1aef39a791cd25ae483c832b658e4f2e1d66c33`.
- Config `configs/methods/e06b_dsdg_native.yaml` (`9d665dc2c909d421…`); environment `gpat-m6-e06c` (torch 2.12.1+cu130).
- Storage: HDD `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, layout `<bank_root>/E06b/seed_<seed>/`.
- Population: synthetic stream indices 0..8837.
- RNG per seed: seed 42: torch.Generator().manual_seed(492027736) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded; seed 1337: torch.Generator().manual_seed(825501606) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded; seed 2026: torch.Generator().manual_seed(1539980013) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- TEST firewall: TRAIN-only inputs; zero VAL/TEST face atime or mtime change in every bank window.
- Disclosures: DSDG-NATIVE trained on CASIA+MSU TRAIN spoof coverage (3720 rows; SiW-Mv2 NOT_INSTANTIABLE_MISSING_SUBJECT_ID); N_syn = 8838 pooled budget (E06b-Q0); spoof RGB half only; synthetic IDs without pair lineage.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
