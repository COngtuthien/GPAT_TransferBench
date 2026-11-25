# M8 learned-bank production family closure: E03

**Status:** `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`, 15/15 closure gates passing; 26514 outputs.

Failed gates: none.

| seed | run ID | outputs | manifest sha256 | audit gates | window |
|---|---|---|---|---|---|
| 42 | `bacbbec2e367fa33` | 8838 | `5cc8b9e26c4f5679…` | 34/34 | 2026-11-25T06:20:45Z → 2026-11-25T06:42:18Z |
| 1337 | `db67fcf77751b4ed` | 8838 | `c414758d373de246…` | 34/34 | 2026-11-25T06:44:09Z → 2026-11-25T07:05:48Z |
| 2026 | `a58f890bb30eda7b` | 8838 | `51fa6bafdf7ea1fd…` | 34/34 | 2026-11-25T07:07:41Z → 2026-11-25T07:29:16Z |

- Authority `configs/amendments/learned_bank_production_E03.yaml` (`235b1dd2b1c99f40…`); execution commit `4c43b5182b3a14211f8ac5fdefdbed093bd755af`; source commit `6549ff76e4567b89f4d598279e21fdff90c23acd`.
- Config `configs/methods/e03_stdn.yaml` (`08dde851bc9e8ac6…`); environment `gpat-m6-e03` (tensorflow 1.15.5).
- Storage: HDD `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, layout `<bank_root>/E03/seed_<seed>/`.
- Population: all 8838 frozen pairs_train_v1 rows in canonical 0-based order.
- RNG per seed: seed 42: none (E03 inference is deterministic; no sampling); seed 1337: none (E03 inference is deterministic; no sampling); seed 2026: none (E03 inference is deterministic; no sampling).
- Conversion: `tf.cast(tf.clip_by_value(synth1, 0.0, 1.0) * 255, tf.uint8) [stdn model/utils.py:24,37]`.
- TEST firewall: TRAIN-only inputs; zero VAL/TEST face atime or mtime change in every bank window.
- Disclosures: official test-mode inference with raw (non-EMA) variables, no inference augmentation; saturation at 0 and 255 is the source writer's clip-then-cast.
- Backup: the image bank exists only on the approved HDD; git holds the manifest (gzip), per-image file and pixel SHA256 and this audit, not the image bytes; no independent binary backup exists.
