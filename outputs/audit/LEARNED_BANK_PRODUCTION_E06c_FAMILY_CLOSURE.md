# M8 learned-bank production family closure: E06c

**Status:** `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`, 15/15 closure gates passing; 26514 outputs.

Failed gates: none.

| seed | run ID | outputs | manifest sha256 | audit gates | window |
|---|---|---|---|---|---|
| 42 | `45e035b85c735361` | 8838 | `8da6d40458a2ee42…` | 34/34 | 2026-11-25T05:10:30Z → 2026-11-25T05:30:25Z |
| 1337 | `219f27730e3025c0` | 8838 | `e88e1ca631bd5e63…` | 34/34 | 2026-11-25T05:32:20Z → 2026-11-25T05:52:05Z |
| 2026 | `bfc0e4a6a947c9c4` | 8838 | `d30ad186264074a6…` | 34/34 | 2026-11-25T05:53:46Z → 2026-11-25T06:13:36Z |

- Authority `configs/amendments/learned_bank_production_E06c.yaml` (`34ddaa1dba95be2b…`); execution commit `4f5d13b3177748922470bf11599a4e7f677d7b04`; source commit `271fdecc096a8ccd7cafca5fcbb383bbf21818ee`.
- Config `configs/methods/e06c_dsdg_bin_idfree.yaml` (`7176dd4cd4900732…`); environment `gpat-m6-e06c` (torch 2.12.1+cu130).
- Storage: HDD `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, layout `<bank_root>/E06c/seed_<seed>/`.
- Population: synthetic stream indices 0..8837.
- RNG per seed: seed 42: torch.Generator().manual_seed(1660457813) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded; seed 1337: torch.Generator().manual_seed(203456175) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded; seed 2026: torch.Generator().manual_seed(1718284219) (dsdg_bank_seed); per batch of 100: noise = randn(100,128), noise_s = randn(100,128), z = cat(noise_s, noise, noise); 89 batches, 8900 drawn, 8838 kept, last 62 of batch 88 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- TEST firewall: TRAIN-only inputs; zero VAL/TEST face atime or mtime change in every bank window.
- Disclosures: DSDG-BIN-IDFREE adapted binary variant (spoof-type labels collapsed); spoof RGB half only; synthetic IDs without pair lineage.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
