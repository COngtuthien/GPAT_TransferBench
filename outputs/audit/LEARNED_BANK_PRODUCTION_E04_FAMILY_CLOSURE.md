# M8 learned-bank production family closure: E04

**Status:** `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`, 24/24 closure gates passing; 26514 outputs.

Failed gates: none.

| seed | run ID | outputs | manifest sha256 | audit gates | window |
|---|---|---|---|---|---|
| 42 | `fca99d171787cee0` | 8838 | `2ffbf95f968d0acd…` | 43/43 | 2026-11-25T14:55:41Z → 2026-11-25T15:21:13Z |
| 1337 | `65a3704819bc715a` | 8838 | `83993dc73979b8bd…` | 43/43 | 2026-11-25T15:25:38Z → 2026-11-25T15:50:22Z |
| 2026 | `6dc85654ddaf14e2` | 8838 | `14569ec629b9dffe…` | 43/43 | 2026-11-25T15:53:14Z → 2026-11-25T16:16:37Z |

- Authority `configs/amendments/learned_bank_production_E04.yaml` (`9d714c0978456d0a…`); execution commit `fabc1ef5069c05cd1d39e17057a90148fa296220`; source commit `cc2834ce15df6fff587da7c6e4c81d2328a57562`.
- Config `configs/methods/e04_physics_std.yaml` (`418617942ebd87c4…`); environment `gpat-m6-e03` (tensorflow 1.15.5).
- Storage: HDD `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, layout `<bank_root>/E04/seed_<seed>/`.
- Population: all 8838 frozen pairs_train_v1 rows in canonical 0-based order.
- RNG per seed: seed 42: none (deterministic inference; no sampling or augmentation); seed 1337: none (deterministic inference; no sampling or augmentation); seed 2026: none (deterministic inference; no sampling or augmentation).
- Conversion: `(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8) float32 [E04-Q3 A_PREDECESSOR_CLIP_TRUNCATE; reference stdn model/utils.py:24,37]`.
- TEST firewall: TRAIN-only inputs; zero VAL/TEST face atime or mtime change in every bank window.
- Disclosures: CONTROLLED_ADAPTATION: Physics-STD (controlled geometry/depth reconstruction); not an official native Physics-STD reproduction; bank inference uses the stored BN moving statistics (batch 1, E04-Q1); training-time synthesis used 8-image batch statistics; float->uint8 clip-truncate (E04-Q3) follows the ECCV20-STDN predecessor (reference only); raw values outside [0, 1] are clipped, never rescaled; S7 observed one seed-42 pair with raw range about [-1.11, 1.68]; RAW-MATCHED: every numerically valid output is included; no quality filter, replacement or threshold; TRAIN allowlist from the pinned e04_feed_pack_v1 in the inference process (owner ruling); split_v1 / M2 cross-check performed by this independent auditor.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
