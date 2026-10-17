# M7D1-N6C — VAL-only best-EMA selection, M7D1-A1 GPAT-B0 / E08 / seed 42

Status: **M7D1_N6C_A1_SEED42_SELECTED**. Run `92e91ddb78f1e242` (launch authority `4466e12`, closure authority `ffffa9b`), curriculum `GPAT-TransferBench-v1.0+M7D1-A1`.

## What was done

All 51 EMA candidates (epochs 10..60, N6B manifest digest `51f9648ebefa0d41…`) were evaluated on the full VAL relation `manifests/val_pairs_v1.parquet` (1905 pairs, SHA-256 `84d124919a52cd8d…`). Each candidate was hash-checked against the manifest before loading. `recovery/latest.pt` was never opened as a candidate. TEST was never readable: an audit-hook firewall denies any non-VAL canonical face and any manifest other than `val_pairs_v1`.

Owner evaluation decisions for N6C (2026-10-17): the residual scale `s_hf = 0.15` (spec 9.3 / `delta_scale_hf`) for every candidate, and the M7B R-06 fp16 autocast for the E_art/G_res forward. Every other convention comes from existing authority: A10 D09/D13/D14, M7B R-03/R-06, M7C2a N-06/N-08, and the M7C3 evaluator adapters.

## Rule (frozen spec 10.6, A10 D14)

`score = 0.25·p(ID) + 0.15·p(Dice) + 0.25·p(ArtSim) + 0.20·p(−NME) + 0.15·p(−LFErr)`, where `p = (rank − 1)/(N − 1)` and ties get the average rank. Ties on score go to the higher raw ArtSim, then to the earlier epoch. Arithmetic is exact (rationals).

## Raw VAL metric ranges over the 51 candidates

| metric | min (epoch) | max (epoch) | distinct values |
|---|---|---|---|
| ID | 0.998777 (13) | 0.999447 (26) | 51 |
| Dice | 0.935276 (60) | 0.953507 (17) | 51 |
| ArtSim | 0.42252 (10) | 0.461861 (60) | 51 |
| NME | 0.00435444 (33) | 0.00646685 (14) | 51 |
| LFErr | 0 (10) | 0 (10) | 1 |

LFErr is exactly 0 for every candidate (γ = 0, M7C2a N-08). All 51 candidates therefore tie on it, and p(−LFErr) = 0.5 for each of them.

## Top 5

| pos | epoch | score | ID | Dice | ArtSim | NME | LFErr | checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 0.720000 | 0.999441 | 0.945683 | 0.447601 | 0.004412 | 0 | `01fd718ae3672491…` |
| 2 | 35 | 0.719000 | 0.999438 | 0.943797 | 0.450339 | 0.004371 | 0 | `4eb8105f1e988c00…` |
| 3 | 26 | 0.699000 | 0.999447 | 0.947393 | 0.446671 | 0.004488 | 0 | `2cc4c1ed8b4d0691…` |
| 4 | 34 | 0.697000 | 0.999437 | 0.943674 | 0.449790 | 0.004363 | 0 | `fa1946e7ce1991c3…` |
| 5 | 29 | 0.692000 | 0.999440 | 0.944634 | 0.446923 | 0.004416 | 0 | `c4b1280f733de22b…` |

## BEST_EMA_SEED42

- Epoch **28**: `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m7_a1/E08/seed_42/checkpoints/ema_candidates/ema_epoch_28.pt`
- SHA-256 `01fd718ae3672491f846f41609ad2de06c8c5b3b10f9717fc1d8014e100b1bb6`
- Raw metrics: ID 0.999441, Dice 0.945683, ArtSim 0.447601, NME 0.00441195, LFErr 0
- Percentiles: ID 0.9600, Dice 0.8400, ArtSim 0.3800, neg_NME 0.9200, neg_LFErr 0.5000
- Score 0.720000 (`18/25`)
- Runner-up: epoch 35, score 0.719000; margin 0.001000
- Tie-break required: False

## Reproducibility

- An independent re-ranking from the committed CSV gives the same winner.
- A full re-evaluation of the winner on VAL reproduces all five metrics bitwise: both the means and every per-pair value.

## Post-selection sanity (POST_SELECTION_SANITY_ONLY; does not affect the selection)

- A_mean: mean 0.0178553, min 0.00711005, median 0.0159231, max 0.0499463
- M_frac_gt_0_5: mean 1, min 0.999939, median 1, max 1
- M_max: mean 1, min 0.999993, median 1, max 1
- M_mean: mean 0.999658, min 0.998464, median 0.999778, max 0.99995
- M_min: mean 0.947477, min 0.453988, median 0.978052, max 0.997613
- hf_residual_mean_abs: mean 0.0552937, min 0.0220062, median 0.0491457, max 0.156777
- hf_target_mean_abs: mean 0.0619702, min 0.0257011, median 0.0572979, max 0.214862
- max_abs_xhat_minus_xt: mean 0.197761, min 0.125292, median 0.204225, max 0.224868
- mean_abs_xhat_minus_xt: mean 0.0161325, min 0.00643768, median 0.0145216, max 0.0430105
- u8_identical_image: mean 0, min 0, median 0, max 0
- u8_identical_pixel_fraction: mean 0.0463382, min 0.00402832, median 0.0416412, max 0.178726
- x_hat_out_of_range_fraction: mean 0.0189371, min 0, median 0.00376892, max 0.207514

Interpretation: the catastrophic v1.0 identity-collapse signature is NOT reproduced by the selected EMA. In v1.0 (M7D1-N2 F01), every EMA candidate gave u8 x_hat == x_t, with A ~ 0 (budget pinned at the 0.01 hinge) and a vanishing background/landmark change. Here, 0 of 1905 VAL outputs are u8-identical to their target; on median only 4.2% of pixels are u8-identical. The mean |x_hat - x_t| is 0.016 (max 0.22). The artifact map A has mean 0.018 (median 0.016; the 5th percentile, 0.0082, lies below the 0.01 budget floor). The HF residual magnitude, 0.055, is comparable to the target's own HF magnitude, 0.062. Observation (non-gating, consistent with N5): the mask stays saturated open (mean M 0.9997; M > 0.5 on effectively every pixel), and ID cosine stays very high (0.9994), so the edit preserves identity while injecting HF artifact energy. No post-hoc threshold was applied, and the frozen-rule winner stands.

## Firewall

Across the smoke, full and sanity sessions: TEST images 0, TEST metadata 0, TRAIN images 0, 0 denied opens, and 0 writes into the scientific run root. The run-root listing digest is `9b1a85eaeaf63545…` both before and after every phase.

Not started: seed 1337/2026, B1/B2/B3, bank generation, downstream FAS, TEST evaluation and any training.
