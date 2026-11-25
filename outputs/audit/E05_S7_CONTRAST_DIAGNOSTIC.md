# E05 PCGAN S7 low-contrast review

**Status:** `E05_LOW_CONTRAST_DISCLOSED_NO_CONTRACT_VIOLATION`.

Contract violations: none.

This review is descriptive. It reads only the committed S7 TRAIN toy outputs (run A; run B is byte-identical) and the TRAIN faces of the 186 approved toy pairs. It loads no model, generates nothing and introduces no quality threshold.

## Mechanical checks against the frozen contract

| check | result |
|---|---|
| inputs_are_committed_s7_outputs | pass |
| file_and_pixel_sha256 | pass |
| e05_raw_finite | pass |
| e05_conversion_consistent_with_tensor2im | pass |
| e05_lineage_source_spoof_target_live | pass |
| e05_per_pair_noise_seed | pass |
| e05_checkpoint_identity | pass |
| e05_tensor2im_runtime_at_defining_module | pass |
| e05_s7_record_passed | pass |
| e05_outputs_distinct_within_seed | pass |
| no_val_test_face_access | pass |
| collector_clean_checkout | pass |

## Contrast compared with the other learned methods (S7 toy outputs, 3 seeds x 186)

| method | per-image std, median [p05, p95] | per-image min, median | per-image max, median | p01 to p99, median | saturated 0 / 255 px | distinct images per seed |
|---|---|---|---|---|---|---|
| E05 | 7.3 [5.0, 8.9] | 99.0 | 130.0 | 99.0 to 126.0 | 0 / 0 | 186/186/186 |
| E06c | 41.3 [30.0, 53.0] | 2.0 | 214.0 | 18.0 to 186.5 | 77085 / 0 | 186/186/186 |
| E03 | 49.0 [34.6, 72.4] | 0.0 | 255.0 | 12.0 to 232.0 | 887962 / 1473708 | 186/186/186 |
| E06b | 30.9 [23.3, 47.5] | 27.0 | 220.0 | 46.0 to 182.0 | 26263 / 0 | 186/186/186 |
| TRAIN input faces (247) | 41.6 [24.2, 84.4] | 7.0 | 255.0 | | | |

## E05 details

- Per-seed per-image std (median): seed 42 8.3, seed 1337 5.2, seed 2026 7.7.
- Per-seed raw output range [min, max]: seed 42 [-0.251, 0.036], seed 1337 [-0.106, 0.026], seed 2026 [-0.236, 0.037].
- Per-seed structure correlation with the source spoof (median [p05, p95]): seed 42 0.67 [0.35, 0.83], seed 1337 0.43 [-0.10, 0.86], seed 2026 0.78 [0.19, 0.91].
- Near-identical output pairs within one seed (mean absolute difference < 1 grey level on the 64 px grid; descriptive, not a filter): seed 42: 151 pairs over 102 rows (0 share the source spoof, 29 share the target live), seed 1337: 12689 pairs over 186 rows (0 share the source spoof, 386 share the target live), seed 2026: 286 pairs over 86 rows (0 share the source spoof, 29 share the target live).
- Visual inspection of the montage: seeds 42 and 2026 show a faint, desaturated face silhouette taken from the source spoof on a flat grey-beige field. Seed 1337 outputs look almost uniformly grey: per-image std about 5, with little visible structure.
- Channel means (R, G, B), averaged over images: 121.9, 110.8, 107.4; channel std: 2.3, 2.5, 3.2.
- Share of pixels within ±5 of the image mean: median 0.39.
- Raw generator output before conversion (float32, recorded per row): min -0.251 to -0.092, max 0.006 to 0.037. Under tensor2im, (x+1)/2*255 maps [-1, 1] to [0, 255]; these raw values stay inside about [-0.251, 0.037], so the outputs never reach the clipping range.
- Conversion: every row's uint8 min and max equal tensor2im applied to its recorded raw min and max.
- Mixing semantics: zero-mean luminance structure (64x64) correlates with the source spoof (median 0.66) more than with the target live (median 0.18) in 99% of rows; the colour mean is closer to the target live in 57% of rows.
- Distinctness: every seed has 186 distinct pixel hashes; the smallest mean absolute difference between two outputs of one seed (64 px grid) is 0.0 (seed 42), 0.0 (seed 1337), 0.1 (seed 2026).
- Same pair across the three generators (mean absolute pixel difference): 1337_vs_2026 7.4, 42_vs_1337 12.6, 42_vs_2026 5.7.
- Training budget (frozen, PAPER): Adam lr 1e-6 betas (0.9, 0.999), 4000 iterations, batch 1; configs/methods/e05_pcgan.yaml training/optimizer (PAPER, unchanged); no pretrained weights (weight_bytes_from_source_cache NONE).

## Representative examples (rows with the lowest, median and highest per-image std per seed)

| seed | pair | min | max | mean | std | raw min | raw max | corr source | corr target |
|---|---|---|---|---|---|---|---|---|---|
| 42 | PTR002174 | 97 | 130 | 103.1 | 6.6 | -0.236 | 0.025 | 0.19 | -0.36 |
| 42 | PTR002544 | 96 | 130 | 107.0 | 8.3 | -0.242 | 0.027 | 0.82 | 0.11 |
| 42 | PTR002085 | 97 | 132 | 108.9 | 9.3 | -0.233 | 0.036 | 0.86 | 0.27 |
| 1337 | PTR000946 | 115 | 130 | 120.9 | 4.9 | -0.094 | 0.023 | 0.86 | 0.25 |
| 1337 | PTR002429 | 115 | 129 | 119.8 | 5.2 | -0.096 | 0.019 | 0.20 | -0.14 |
| 1337 | PTR000062 | 114 | 129 | 119.4 | 5.5 | -0.103 | 0.013 | -0.31 | -0.34 |
| 2026 | PTR006825 | 98 | 130 | 115.7 | 6.1 | -0.231 | 0.027 | 0.83 | 0.07 |
| 2026 | PTR000747 | 102 | 131 | 112.7 | 7.7 | -0.197 | 0.028 | 0.77 | 0.10 |
| 2026 | PTR000161 | 101 | 130 | 115.5 | 8.4 | -0.205 | 0.027 | 0.85 | 0.60 |

A montage of these rows (source spoof, target live, raw output, unmodified) is on the GPU host at `/home/student20261/workdir/m8_learned_banks/diagnostics/e05_s7_contrast_montage_v2`. It is not committed because it contains dataset faces.

## Interpretation

Every mechanical check of the frozen contract passes, and every output is distinct by pixel hash. The outputs are low-contrast, and the low contrast comes from the generator, not the conversion: the raw generator output already lies in a narrow band just below zero, so tensor2im maps it to mid-grey without clipping. The source-spoof structure is visible in seeds 42 and 2026. Seed 1337 is close to collapse onto one near-constant grey image: 12689 of 17205 within-seed output pairs differ by less than one grey level on average, compared with 151 and 286 for seeds 42 and 2026, and none for E06c, E03 or E06b. This is consistent with the frozen paper training budget (4000 iterations at batch 1, learning rate 1e-6, from random initialisation), under which the generator moves little from its initial state.

The frozen specification (RAW-MATCHED view) retains every numerically valid output and excludes only NaN/Inf, decode failure or missing output. Near-constant outputs are therefore not a contract violation, and low contrast alone is not evidence of an invalid checkpoint. The RAW-MATCHED protocol is retained: no enhancement, filtering, quality threshold, checkpoint substitution or retuning. This is a scientific limitation that must be disclosed with every E05 result. Seed 1337 in particular is likely to give a weak or near-degenerate downstream bank. Whether E05 results are interpreted with that caveat, or whether the training budget is revisited, is an owner decision outside this session.
