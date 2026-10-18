# M7D1 — GPAT deferred-fix backlog (BASELINE-EXEC-P0)

Status: **GPAT DEFERRED_BY_OWNER_PRIORITY**. Date 2026-10-18. Authority `03217143a766f0bbeae6fcf90c45c4226881a0fc`
(branch `m6-baselines`). Machine-readable record: `M7D1_GPAT_DEFERRED_FIX_BACKLOG.json`.

This is a current-state deferral record. It is not an amendment and not a fix, and it changes no GPAT code, config or
schedule. Its purpose is to let a future session resume GPAT without redoing the investigation.

## Owner direction

The owner has paused GPAT development. The priority is now to **execute all in-scope previous-work baselines first**
(see `PREVIOUS_WORK_EXECUTION_ROADMAP.md`). GPAT resumes only when that program is complete or the owner changes the
priority.

Forbidden until then:
- replaying, investigating (u2167) or resuming seed1337;
- implementing another GPAT fix;
- starting seed2026, or B1/B2/B3;
- any GPAT bank or downstream work.

## A. Healthy seed42 (M7D1-A1)

| Field | Value |
|---|---|
| Run | GPAT-B0 / E08 / seed 42, run `92e91ddb78f1e242`, curriculum `v1.0+M7D1-A1`, launch `4466e12` |
| Completion | 60 epochs / 66300 updates, `completed` (closure `ffffa9bd3a2f7d94f0d7fa228764dc425d2ff563`) |
| BEST_EMA_SEED42 | **epoch 28**, `ema_epoch_28.pt`, sha256 `01fd718ae3672491f846f41609ad2de06c8c5b3b10f9717fc1d8014e100b1bb6` |
| Selection | VAL-only spec 10.6 / A10 D14 percentile rank over 51 EMA candidates; score 18/25 = 0.72; TEST not used |
| N6C selection commit | `13a0252fd097b7552f0875f346f83c7b676cef0f` |

## B. Failure mode F1: seed42 historical stage-2 collapse (v1.0)

Root cause localized (M7D1-N3, `d88563a`): **ADVERSARIAL_IMBALANCE / CURRICULUM_TRANSITION_INSTABILITY**.
1. The original schedule jumped lambda_adv from 0 to 0.05 at u5526.
2. A discriminator trained for 5525 updates without adversarial pressure was overconfident, so gadv dominated the
   G gradient (68–100 %).
3. Through the shared G_res trunk, gadv drove the mask logit from about +14 to −45 (u5777–u5781), closing the mask.
4. Sigmoid saturation made the closed mask an absorbing state: the budget hinge had no gradient path back.

The M7D1-A1 one-epoch ramp repaired this mechanism for seed42 (N4, N5, N6B).

## C. M7D1-A1 (changes only lambda_adv)

| Updates | lambda_adv |
|---|---|
| u1..u5525 | 0 |
| u5526..u6630 | `0.05 * ((u - 5526) / 1104)` |
| u6631..u16575 | 0.05 |
| u16576..u66300 | 0.10 |

M7D1-A1 remains historical/active evidence for F1 and **must not be removed without evidence**.

## D. Failure mode F2: seed1337 stage-1 collapse

| Field | Value |
|---|---|
| Run | `62d7cfd98022a95a` (A1, launch `4466e12`, closure `0321714`) |
| Final status | **OWNER_STOPPED_CONFIRMED_GENERATOR_COLLAPSE** |
| Last COMPLETE | **u15107** (epoch 14 group 742; last complete epoch 13) |
| Onset | ~**u2167**, epoch 2, stage 1, **lambda_adv = 0**. G grad norm (pre-clip) went from 0.0206 to 2.45, with no AMP event (nearest earlier retry u2132) |
| Scalar signature | first at **u2174**, single miss **u2175**, continuous **u2176..u15107** |
| Recovery states (not selectable) | u15107 `c1b4295e…`; u14365 epoch-13 copy `0f845ec5…`. No pre-onset state survives |

**THIS IS NOT EXPLAINED BY THE M7D1-A1 ADVERSARIAL-TRANSITION FAILURE.** The adversarial term is zero throughout
stage 1. The root cause is **not inferred**.

Direct output confirmation used a fixed set of 32 TRAIN pairs (M7D1-N2 rule):
- the uint8 `x_hat` equals `x_target` for 32/32 pairs in every state;
- mean |x_hat − x_target| ≈ 6e-8;
- the mask M and the artifact A are effectively zero;
- the HF residual is saturated but masked out.

## E. Future milestone M7D1-SEED1337-RC1 (not started)

- Deterministic fresh replay **u1 → u2210** at `4466e12` / seed 1337, in a diagnostic root only.
- Snapshots: u1105, u2100, u2150, u2160, u2166, u2167, u2168, u2170, u2171, u2174, u2210.
- **Parity gate:** the replay must match the original scientific seed1337 trajectory before any causal interpretation.

## F. Required attribution (at and around u2167)

- Decompose: id, lm, parse, low, budget, tv, bg, artcon, spec and optimizer momentum.
- gadv must be zero in stage 1.
- Measure effects on: the shared G_res trunk, raw mask logits, mask M, HF residual, M · delta_HF, the x_hat − target
  residual, and artifact A.

## G. Hypotheses to test (not conclusions)

1. A pathological scientific group or batch near u2167.
2. An artcon/spec/fidelity interaction.
3. An E_art / shared-trunk / BN feature excursion.
4. Adam momentum or clipping overshoot.
5. A generic sigmoid-mask trap.
6. A budget gradient unable to reopen a closed mask.
7. A seed-specific initialization / data-order interaction.

Also inspect the exact u2167 group: datasets, pair IDs, source/target IDs, attack types, microbatch weights,
image/input statistics, and teacher/artifact features.

## H. Candidate fixes (only after causality; none approved)

- A mask anti-trap parameterization or bounded floor.
- Direct mask-state regularization.
- A stronger or earlier anti-collapse budget path, if the gradients support it.
- Per-loss or shared-trunk gradient balancing.
- A stage-1 artcon/spec/fidelity curriculum modification.
- Optimizer/momentum mitigation, if shown to be causal.
- Pathological-sample handling, **only** if a specific data defect is proven.

**NO BLIND HYPERPARAMETER TUNING.**

## GPAT inventory at deferral

| Item | State |
|---|---|
| E08 seed42 (A1) | completed and VAL-selected (epoch 28) |
| E08 seed1337 (A1) | owner-stopped generator collapse (not completed, not selectable) |
| E08 seed2026 | not started |
| E09–E11 (B1/B2/B3) | not started |
| Banks / downstream | none |

This record did no training, replay or resume, accessed no VAL or TEST data, and created no checkpoint, bank or result.
