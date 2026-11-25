# M8 learned-bank production family closure: E05

**Status:** `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`, 15/15 closure gates passing; 26514 outputs.

Failed gates: none.

| seed | run ID | outputs | manifest sha256 | audit gates | window |
|---|---|---|---|---|---|
| 42 | `3351ac8fe9ea606b` | 8838 | `079077c86c12a562…` | 37/37 | 2026-11-25T09:00:13Z → 2026-11-25T09:18:12Z |
| 1337 | `d1105c1fbf6d7487` | 8838 | `f8e9ff6782058c74…` | 37/37 | 2026-11-25T10:17:11Z → 2026-11-25T10:34:54Z |
| 2026 | `6674eba825de2426` | 8838 | `e2407a9ce312e349…` | 37/37 | 2026-11-25T10:37:17Z → 2026-11-25T10:55:17Z |

- Authority `configs/amendments/learned_bank_production_E05.yaml` (`a95b86e9f546cc2a…`); execution commit `125a8b3d5a1553e3fabbbc54cf4c2fdfb55e29a3`; source commit `65a99cf716ccb3ad8749a58ae050ea408961c39c`.
- Config `configs/methods/e05_pcgan.yaml` (`478756e150c42783…`); environment `gpat-m6-e05` (torch 2.12.1+cu130).
- Storage: HDD `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, layout `<bank_root>/E05/seed_<seed>/`.
- Population: all 8838 frozen pairs_train_v1 rows in canonical 0-based order.
- RNG per seed: seed 42: torch.manual_seed(e05_pair_seed(seed, pair_id)) immediately before the single mix forward of each pair; seed 1337: torch.manual_seed(e05_pair_seed(seed, pair_id)) immediately before the single mix forward of each pair; seed 2026: torch.manual_seed(e05_pair_seed(seed, pair_id)) immediately before the single mix forward of each pair.
- Conversion: `util.tensor2im(mix[0]): clip((x+1)/2*255, 0, 255).astype(uint8) [swapping_autoencoder util/util.py:125-131]`.
- TEST firewall: TRAIN-only inputs; zero VAL/TEST face atime or mtime change in every bank window.
- Disclosures: PCGAN controlled architecture resolution (not a native PCGAN reproduction); low-contrast outputs; seed 1337 close to collapse onto a near-constant grey image (see outputs/audit/E05_S7_CONTRAST_DIAGNOSTIC.md); retained under RAW-MATCHED; frozen paper budget 4000 iterations, lr 1e-6, batch 1.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
