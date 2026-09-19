# M3 — Allocator Feasibility (PRE-M3, no split created)

The frozen Q-01 allocator was run against the real metadata **only** to measure problem size, feasibility and optimality. The resulting group-to-split membership was discarded and is not written, printed or hashed anywhere; `manifests/split_v1.parquet` does not exist. M3 itself will produce the authoritative split.

Backend: **HiGHS via scipy.optimize.milp** (scipy 1.18.1), EXACT, threads 1, options `{'mip_rel_gap': 0.0, 'presolve': True, 'time_limit': 600}`.

## casia_fasd

| dimension | value |
|---|---|
| canonical videos | 600 |
| allocation groups | 50 |
| group size distribution | {'12': 50} |
| profile classes after reduction | **1** |
| usable samples (M2 COMPLETE) | 4,800 |
| failed samples (provenance only) | 0 |
| integer variables (reduced) | 3 |
| deviation variables | 48 |
| total columns | 51 |
| constraint rows | 97 |
| variables without the class reduction | 150 |
| solve time (all levels + canonicalisation) | 0.392 s |

Category counts:

- **P2_binary** (2 categories): {'1': 450, '0': 150}
- **P3_attack_macro** (3 categories): {'print': 300, 'live': 150, 'replay': 150}
- **P4_attack_raw** (10 categories): {'(none)': 150, '3': 50, '4': 50, '5': 50, '6': 50, '7': 50, '8': 50, 'HR_2': 50, 'HR_3': 50, 'HR_4': 50}

Objective values reached (all levels proven globally optimal):

| level | objective | solver status | mip gap |
|---|---|---|---|
| P1_total | 1,200 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P2_binary | 3,999,600 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P3_attack_macro | 5,999,400 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P4_attack_raw | 19,999,800 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |

Achieved split shape (aggregate feasibility evidence; **not** a membership table):

| split | canonical videos | % | usable samples | live | spoof |
|---|---|---|---|---|---|
| TRAIN | 420 | 70.0000 % | 3,360 | 105 | 315 |
| VAL | 96 | 16.0000 % | 768 | 24 | 72 |
| TEST | 84 | 14.0000 % | 672 | 21 | 63 |

Categories that cannot occupy all three splits (fewer than 3 independent groups): **none**

## msu_mfsd

| dimension | value |
|---|---|
| canonical videos | 280 |
| allocation groups | 35 |
| group size distribution | {'8': 35} |
| profile classes after reduction | **1** |
| usable samples (M2 COMPLETE) | 2,240 |
| failed samples (provenance only) | 0 |
| integer variables (reduced) | 3 |
| deviation variables | 30 |
| total columns | 33 |
| constraint rows | 61 |
| variables without the class reduction | 105 |
| solve time (all levels + canonicalisation) | 0.068 s |

Category counts:

- **P2_binary** (2 categories): {'1': 210, '0': 70}
- **P3_attack_macro** (3 categories): {'replay': 140, 'print': 70, 'live': 70}
- **P4_attack_raw** (4 categories): {'ipad_video': 70, 'iphone_video': 70, 'printed_photo': 70, '(none)': 70}

Objective values reached (all levels proven globally optimal):

| level | objective | solver status | mip gap |
|---|---|---|---|
| P1_total | 800 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P2_binary | 5,713,600 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P3_attack_macro | 8,570,800 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P4_attack_raw | 11,428,000 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |

Achieved split shape (aggregate feasibility evidence; **not** a membership table):

| split | canonical videos | % | usable samples | live | spoof |
|---|---|---|---|---|---|
| TRAIN | 200 | 71.4286 % | 1,600 | 50 | 150 |
| VAL | 40 | 14.2857 % | 320 | 10 | 30 |
| TEST | 40 | 14.2857 % | 320 | 10 | 30 |

Categories that cannot occupy all three splits (fewer than 3 independent groups): **none**

## siwmv2

| dimension | value |
|---|---|
| canonical videos | 1,700 |
| allocation groups | 1,694 |
| group size distribution | {'1': 1688, '2': 6} |
| profile classes after reduction | **16** |
| usable samples (M2 COMPLETE) | 13,575 |
| failed samples (provenance only) | 25 |
| integer variables (reduced) | 48 |
| deviation variables | 75 |
| total columns | 123 |
| constraint rows | 166 |
| variables without the class reduction | 5,082 |
| solve time (all levels + canonicalisation) | 2.48 s |

Category counts:

- **P2_binary** (2 categories): {'1': 915, '0': 785}
- **P3_attack_macro** (7 categories): {'live': 785, 'partial': 341, 'mask_3d': 189, 'print': 135, 'makeup': 135, 'replay': 98, 'mask_2d': 17}
- **P4_attack_raw** (15 categories): {'(none)': 785, 'Partial_FunnyeyeGlasses': 179, 'Paper': 135, 'Replay': 98, 'Partial_PaperGlasses': 76, 'Mask_HalfMask': 72, 'Makeup_Impersonation': 61, 'Mask_TransparentMask': 60, 'Partial_Eye': 57, 'Makeup_Cosmetic': 52, 'Mannequin': 40, 'Partial_Mouth': 29, 'Makeup_Obfuscation': 22, 'Mask_PaperMask': 17, 'Silicone': 17}

Objective values reached (all levels proven globally optimal):

| level | objective | solver status | mip gap |
|---|---|---|---|
| P1_total | 0 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P2_binary | 236,500 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P3_attack_macro | 10,167,460 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |
| P4_attack_raw | 32,792,250 | Optimization terminated successfully. (HiGHS Status 7: Optimal) | 0.0 |

Achieved split shape (aggregate feasibility evidence; **not** a membership table):

| split | canonical videos | % | usable samples | live | spoof |
|---|---|---|---|---|---|
| TRAIN | 1,190 | 70.0000 % | 9,507 | 549 | 641 |
| VAL | 255 | 15.0000 % | 2,033 | 118 | 137 |
| TEST | 255 | 15.0000 % | 2,035 | 118 | 137 |

Categories that cannot occupy all three splits (fewer than 3 independent groups): **none**

## Why this is exact

Groups carrying an identical multiset of (label_binary, attack_macro, attack_raw) are interchangeable for every P1-P4 term, so the allocation collapses to a per-class count vector. That reduction is what makes a proven global optimum cheap here: the real problems shrink to a handful of integer variables, and every level terminates with `mip_gap = 0`. Which concrete group receives which split is then fixed by the frozen seeded hash, not by the solver.

