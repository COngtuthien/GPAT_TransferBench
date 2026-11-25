# M8 learned-bank production summary

**Status:** `FOUR_LEARNED_PRODUCTION_FAMILIES_COMPLETED_AND_AUDITED`. 106056 of 106056 expected outputs.

| Method | Seed | Run ID | Output count | Manifest SHA256 | Audit | Status |
|---|---|---|---|---|---|---|
| E06c | 42 | `45e035b85c735361` | 8838 | `8da6d40458a2ee42…` | 34/34 | PASSED |
| E06c | 1337 | `219f27730e3025c0` | 8838 | `e88e1ca631bd5e63…` | 34/34 | PASSED |
| E06c | 2026 | `bfc0e4a6a947c9c4` | 8838 | `d30ad186264074a6…` | 34/34 | PASSED |
| E03 | 42 | `bacbbec2e367fa33` | 8838 | `5cc8b9e26c4f5679…` | 34/34 | PASSED |
| E03 | 1337 | `db67fcf77751b4ed` | 8838 | `c414758d373de246…` | 34/34 | PASSED |
| E03 | 2026 | `a58f890bb30eda7b` | 8838 | `51fa6bafdf7ea1fd…` | 34/34 | PASSED |
| E06b | 42 | `07e64f33b19310b4` | 8838 | `882017b8fadc5326…` | 34/34 | PASSED |
| E06b | 1337 | `2037be09ba25acdc` | 8838 | `40564cd58982eae4…` | 34/34 | PASSED |
| E06b | 2026 | `e2d2ba62af40de21` | 8838 | `e1dd3a2eaa990b05…` | 34/34 | PASSED |
| E05 | 42 | `3351ac8fe9ea606b` | 8838 | `079077c86c12a562…` | 37/37 | PASSED |
| E05 | 1337 | `d1105c1fbf6d7487` | 8838 | `f8e9ff6782058c74…` | 37/37 | PASSED |
| E05 | 2026 | `6674eba825de2426` | 8838 | `e2407a9ce312e349…` | 37/37 | PASSED |

## Family closures

- E06c: `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED` (15/15), authority `configs/amendments/learned_bank_production_E06c.yaml` (`34ddaa1dba95…`), execution commit `4f5d13b`.
- E03: `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED` (15/15), authority `configs/amendments/learned_bank_production_E03.yaml` (`235b1dd2b1c9…`), execution commit `4c43b51`.
- E06b: `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED` (15/15), authority `configs/amendments/learned_bank_production_E06b.yaml` (`99c749453121…`), execution commit `99ea826`.
- E05: `LEARNED_BANK_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED` (15/15), authority `configs/amendments/learned_bank_production_E05.yaml` (`a95b86e9f546…`), execution commit `125a8b3`.

## Final host verification (2026-11-25T11:00:09Z)

- all_banks_present_and_unchanged: pass
- all_bank_audits_passed: pass
- all_families_closed: pass
- clones_clean_at_execution_commit: pass
- host_scripts_equal_committed: pass
- all_markers_present: pass
- storage_live_ok: pass
- e01_e02_unchanged: pass

## Storage

- HDD `/media/ext_data/student20261_gpat_m8_banks`, UUID `608647d2-9cb0-45b0-b999-73c080ba909b`, mount `/media/ext_data` (ext4, /dev/sda2), root mode 0o700.
- Learned-bank PNG bytes: 5.87 GiB. Filesystem used 488.6 GiB, free 2991.4 GiB of 3666.3 GiB; 243607238 free inodes.
- Backup: the 12 image banks exist only on the approved HDD; git holds manifests (gzip), per-image file and pixel SHA256 and the audits, not image bytes; no independent binary backup was made or verified.

## Not done (out of scope)

- E04 Physics-STD bank adapter, S7 qualification and production banks
- downstream ResNet-18 training and evaluation of all seven baselines
- TEST evaluation
- DINOv3
- GPAT
- outputs/audit/method_status.csv unchanged
