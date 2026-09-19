# M3 — Authoritative Split Report

`manifests/split_v1.parquet` · **20,615 rows** · sha256 `fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d`. Produced by the frozen Q-01 allocator; this pass made no scientific choice of its own.

## 1. Population

| | |
|---|---|
| M2 sampled samples | 20,640 |
| M2 COMPLETE (usable, in manifest) | 20,615 |
| M2 FAILED (provenance only, excluded) | 25 |

Every M2 COMPLETE row appears exactly once; no FAILED row is present. A canonical video weighs exactly one video in the objective regardless of how many of its frames survived, so the 23 partially-failed videos were balanced like any other.

## 2. Allocation and usable rows

| dataset | split | groups | canonical videos | video % | usable samples | live vids | spoof vids |
|---|---|---|---|---|---|---|---|
| casia_fasd | TRAIN | 35 | 420 | 70.000 % | 3,360 | 105 | 315 |
| casia_fasd | VAL | 8 | 96 | 16.000 % | 768 | 24 | 72 |
| casia_fasd | TEST | 7 | 84 | 14.000 % | 672 | 21 | 63 |
| **casia_fasd** | **all** | **50** | **600** | 100 % | **4,800** | | |
| msu_mfsd | TRAIN | 25 | 200 | 71.429 % | 1,600 | 50 | 150 |
| msu_mfsd | VAL | 5 | 40 | 14.286 % | 320 | 10 | 30 |
| msu_mfsd | TEST | 5 | 40 | 14.286 % | 320 | 10 | 30 |
| **msu_mfsd** | **all** | **35** | **280** | 100 % | **2,240** | | |
| siwmv2 | TRAIN | 1,190 | 1,190 | 70.000 % | 9,507 | 549 | 641 |
| siwmv2 | VAL | 249 | 255 | 15.000 % | 2,033 | 118 | 137 |
| siwmv2 | TEST | 255 | 255 | 15.000 % | 2,035 | 118 | 137 |
| **siwmv2** | **all** | **1,694** | **1,700** | 100 % | **13,575** | | |

Ratios are the allocator's objective unit — **canonical videos**, not frames. Frame ratios differ slightly because a few videos contribute 6 or 7 usable frames.

## 3. Distributions

Full tables: `M3_DISTRIBUTION_BINARY.csv`, `M3_DISTRIBUTION_ATTACK_MACRO.csv`, `M3_DISTRIBUTION_ATTACK_RAW.csv` (canonical-video counts with per-category deviation in percentage points). These are audit results: the split was not adjusted after reading them.

| dataset | category | videos | TRAIN % | VAL % | TEST % |
|---|---|---|---|---|---|
| casia_fasd | live | 150 | 70.00 | 16.00 | 14.00 |
| casia_fasd | spoof | 450 | 70.00 | 16.00 | 14.00 |
| msu_mfsd | live | 70 | 71.43 | 14.29 | 14.29 |
| msu_mfsd | spoof | 210 | 71.43 | 14.29 | 14.29 |
| siwmv2 | live | 785 | 69.94 | 15.03 | 15.03 |
| siwmv2 | spoof | 915 | 70.05 | 14.97 | 14.97 |

Largest per-category deviations at the attack_raw level (percentage points from target):

| dataset | attack_raw | videos | TRAIN pp | VAL pp | TEST pp |
|---|---|---|---|---|---|
| siwmv2 | Mask_PaperMask | 17 | +0.59 | +2.65 | -3.24 |
| siwmv2 | Silicone | 17 | +0.59 | -3.24 | +2.65 |
| siwmv2 | Makeup_Obfuscation | 22 | +2.73 | -1.36 | -1.36 |
| siwmv2 | Partial_Mouth | 29 | +2.41 | -1.21 | -1.21 |
| msu_mfsd | (none) | 70 | +1.43 | -0.71 | -0.71 |
| msu_mfsd | ipad_video | 70 | +1.43 | -0.71 | -0.71 |
| msu_mfsd | iphone_video | 70 | +1.43 | -0.71 | -0.71 |
| msu_mfsd | printed_photo | 70 | +1.43 | -0.71 | -0.71 |

Deviation here is driven by group granularity and by the lexicographic order (P1 and P2 are satisfied first and may not be worsened), not by a tuning choice.

## 4. Manifest

| | |
|---|---|
| rows | 20,615 |
| columns | `sample_id`, `dataset`, `subject_id_global`, `video_id`, `frame_index`, `label_binary`, `attack_raw`, `attack_macro`, `split`, `sha256`, `sha256_kind`, `content_group_id`, `allocation_group_id`, `m2_status` |
| canonical row order | `dataset > video_id > frame_index > sample_id` |
| writer | pyarrow `pq.write_table` {"compression": "zstd", "compression_level": 9, "write_statistics": true, "version": "2.6", "use_dictionary": false, "row_group_size": 65536, "store_schema": true, "write_page_index": false} |
| sha256 | `fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d` |

`sha256` follows the existing lineage: CASIA keeps M1's `original_frame_bytes`, and MSU/SiW resolve M1's `PENDING_M2_CANONICAL_PNG` placeholder with M2's canonical frame PNG hash. `sha256_kind` records which applies. `subject_id_global` is left null for SiW — no subject identity is manufactured.

## 5. M2 failed-sample accounting

All **25** M2 failures are accounted for in `M3_FAILED_SAMPLE_SPLIT_ACCOUNTING.csv`: none appears in the split manifest, and each is recorded together with the split its canonical video received. They are provenance only and were neither replaced nor resampled.

