# M7D1-N4 — GPAT-B0 / E08 / seed 42 identity-collapse repair qualification

Labels: DIAGNOSTIC_ONLY · NOT_SCIENTIFIC · NOT_A_REPLACEMENT_RUN · NOT_ELIGIBLE_FOR_VAL_SELECTION ·
NOT_ELIGIBLE_FOR_BANK · NOT_ELIGIBLE_FOR_PAPER_RESULT · DIAGNOSTIC_REPAIR_VARIANT · NOT_AN_APPROVED_PROTOCOL_CHANGE.

Nothing here changes a production config, the frozen specification, or the scientific run. The candidate named below
is a *qualified diagnostic candidate* for an owner decision. It is not an official, native, or frozen repair.

**Scope**
- Scientific run: `7b799fbd6d0426da`. N3 root cause: d88563a.
- Every branch restarts from the exact N3 pre-u5526 state, N3 `snapshots/replay/pre_update_5526.pt`, sha256
  `3b77d738…af59`. This is the state from which N3 fork A reproduced u5526..u5850 325/325 bitwise.
- Diagnostic root: `<rt>/diagnostics/m7/M7D1_N4_seed42/` on the GPU host.
- Only lambda_adv differs between branches. LR, s_hf, lambda_con, lambda_spec, D updates, AMP policy, optimizers,
  clipping, EMA and TRAIN order are the frozen scientific ones.

**Evidence** (SHA-256 in `M7D1_N4_SEED42_REPAIR_EVIDENCE_SHA256.txt`):
- `M7D1_N4_SEED42_REPAIR_COLLECTED.json` comes from `tools/m7d1_n4_repair_qualification.py --scenario collect`. It re-reads
  the raw branch artifacts, the read-only scientific `metrics.jsonl` and the attribution results.
- `finalize()` in the same tool derives these files from it with no new measurements. The tests re-derive them byte for byte:
  - `…_QUALIFICATION.json`: hard gates, formal decisions, verdict;
  - `…_SCREEN.csv`: every update u5526..u5850 for R0–R4;
  - `…_LONG.csv`: per-epoch distributions, R1/R2 epochs 6–15;
  - `…_GRADIENTS.csv`: per-snapshot N3/N4 loss attribution and Adam-step split.

## 1. Branches

| Branch | lambda_adv on u5526..u16575 | Window run |
|---|---|---|
| R0 CONTROL | 0.05 (frozen stage 2) | u5526..u5850 |
| R1 ADV_RAMP_1E | `0.05 * ((u-5526)/(6630-5526))` for u ≤ 6630, then 0.05 | u5526..u16575 |
| R2 ADV_RAMP_5E | `0.05 * ((u-5526)/(11050-5526))` for u ≤ 11050, then 0.05 | u5526..u16575 |
| R3 LOW_ADV_001 | 0.01 | u5526..u5850 |
| R4 LOW_ADV_002 | 0.02 | u5526..u5850 |
| R5 D reset | `R5_UNSUPPORTED_EXACT_STATE`: no recorded initial-D authority | not run |

**Floating-point form of the ramps.** The ramps are coded as `0.05 * (ratio)`, not in the literal algebraic form
`(0.05 * n) / d`. The two are equal as real numbers. In binary floating point the literal form gives
`(0.05 * 5524) / 5524 = 0.049999999999999996` at the R2 endpoint. The coded form gives exactly 0.0 at u5526 and exactly
0.05 at the endpoint.

Intermediate values can differ from the literal form in the last ulp; for example R2 u6630 is 0.009992758870383782. Collect
verified every recorded lambda_adv bit-exactly against `lambda_adv(branch, u)` (0 mismatches over 11050 rows per long branch):
- R1: lambda_adv(6630) = 0.05 exactly, and every u6630..u16575 equals 0.05.
- R2: lambda_adv(11050) = 0.05 exactly, and every u11050..u16575 equals 0.05. Midpoint u8288 = 0.025 exactly.

## 2. Measured facts

### 2.1 Hard-gate audit

Collect re-derived these from the raw artifacts, and read-only supplementary checks extended them. Results are identical for R0–R4 unless stated.

| Gate | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| status | COMPLETED_TO_END | COMPLETED_TO_END | COMPLETED_TO_END | COMPLETED_TO_END | COMPLETED_TO_END |
| COMPLETE updates (contiguous from u5526) | 325 | 11050 (→u16575) | 11050 (→u16575) | 325 | 325 |
| exact start state (base sha + state digest; u5526 live digest) | ✓ | ✓ | ✓ | ✓ | ✓ |
| provenance (code_head d88563a, base sha) | ✓ | ✓ | ✓ | ✓ | ✓ |
| TRAIN group digests across branches | ✓ | ✓ | ✓ | ✓ | ✓ |
| `group_samples` / microbatch sizes / sample weights == scientific record of same u (supplementary) | 325/325 | 11050/11050 | 11050/11050 | 325/325 | 325/325 |
| LR, s_hf, lambda_con, lambda_spec, epoch/group == scientific and frozen contract | ✓ | ✓ | ✓ | ✓ | ✓ |
| branch lambda_adv bit-exact | ✓ | ✓ | ✓ | ✓ | ✓ |
| AMP retry integrity: attempts = retries+1, each backoff halves, scale breaks only at retry updates (supplementary) | ✓ | ✓ | ✓ | ✓ | ✓ |
| non-finite losses / M / A / grad norms | 0 | 0 | 0 | 0 | 0 |
| non-finite tensors in snapshots ≥u6630 (supplementary; 6 per long branch) | – | 0 | 0 | – | – |
| non-finite attribution output | 0 | 0 | 0 | 0 | 0 |
| write firewall (denied / outside allowed) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| VAL / TEST / non-TRAIN access | 0 | 0 | 0 | 0 | 0 |
| formal collapse event | onset u5781 (expected) | none | none | none | onset u5535 |

**Other integrity checks**
- Epoch-end recovery saves inside the long windows (u6630…u15470) and EMA-candidate writes (epochs 10–14) were
  intercepted by the harness and never written.
- The collect firewall recorded 0 denied writes and 0 writes outside the allowed roots.
- The scientific run root is unchanged:
  - `metrics.jsonl` sha256 is `5f4aa9a0…632f`;
  - the path/size/mtime listing digest `ac64a063…893a`, recorded before any N4 branch started
    (`results/scientific_root_pre.txt`), reproduces exactly after N4.

### 2.2 R0 exact control

`R0_PARITY_PASS`:
- u5526..u5850 is 325/325 bitwise against the scientific records and against N3 fork A, including the N3 mask rows.
- AMP retries fall at u5527/5529/5758/5759, identical to the scientific run.
- The formal collapse (onset u5781, event u5790) equals N3 fork A.

### 2.3 Formal collapse rule

`N4_IDENTITY_COLLAPSE_V1`: mean(M) < 0.01 for 10 consecutive COMPLETE updates. This was declared before any branch ran and has not been changed.

| Branch | Formal event | Lowest mean(M) (update) | Severe transient excursions not meeting the rule |
|---|---|---|---|
| R0 | onset u5781, event u5790 | 6.7e-09 (u5836) | – |
| R1 | none to u16575 | 0.892 (u5576) | u5576: M_p10 0.445, 6.3 % of mask pixels < 0.01. the only update with mean(M) < 0.9; lowest after u6630 is 0.982 (u6640) |
| R2 | none to u16575 | 0.942 (u5616) | u5616: M_p10 0.885, 1.3 % of pixels < 0.01; lowest after u6630 is 0.990 (u6680) |
| R3 | none to u5850 | 0.488 (u5539) | **severe partial closure u5535–u5546**: M_p10 = 0, up to 51 % of pixels < 0.01, mean logit down to −1.8e3; recovered at u5547–5549 (logit +25, M ≈ 1) |
| R4 | onset u5535, event u5544 | 0.0 | absorbing within 10 updates; logit ≈ −2.2e4 at u5850 |

R4 collapses *earlier* than R0 at a *smaller* fixed lambda_adv. Mask safety is not monotonic in the fixed lambda_adv
magnitude.

### 2.4 R1 and R2 long qualification, per epoch

Values are medians over the 1105 updates of each epoch, from `…_LONG.csv`. "clip" is the share of updates with G grad norm > 1.

| Epoch | R1 λ | R1 M | R1 σ′ | R1 A | R1 |x̂−x_t| | R1 budget-active | R1 gadv | R1 D_total | R1 clip | R2 λ | R2 M | R2 σ′ | R2 A | R2 budget-active | R2 gadv | R2 D_total | R2 clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0→.05 | .997 | 3.2e-3 | .0127 | .0115 | .66 | 1.82 | .515 | .06 | 0→.010 | .999 | 7.3e-4 | .0138 | .47 | 3.31 | .384 | .03 |
| 7 | .05 | .998 | 2.4e-3 | .0117 | .0105 | .79 | 2.52 | .329 | .19 | .010→.020 | .999 | 6.3e-4 | .0117 | .91 | 2.89 | .344 | .09 |
| 8 | .05 | .998 | 1.6e-3 | .0124 | .0112 | .71 | 3.49 | .239 | .26 | .020→.030 | .999 | 6.1e-4 | .0117 | .87 | 3.19 | .282 | .09 |
| 9 | .05 | .999 | 8.3e-4 | .0127 | .0114 | .70 | 4.26 | .185 | .31 | .030→.040 | .999 | 6.7e-4 | .0122 | .86 | 3.63 | .235 | .07 |
| 10 | .05 | .999 | 5.8e-4 | .0136 | .0122 | .62 | 4.94 | .148 | .24 | .040→.05 | .999 | 5.1e-4 | .0132 | .71 | 4.16 | .194 | .18 |
| 11 | .05 | 1.0 | 3.7e-4 | .0140 | .0125 | .65 | 5.57 | .123 | .32 | .05 | 1.0 | 3.2e-4 | .0132 | .81 | 4.88 | .159 | .26 |
| 12 | .05 | 1.0 | 3.0e-4 | .0146 | .0131 | .59 | 6.25 | .097 | .41 | .05 | 1.0 | 1.6e-4 | .0138 | .80 | 5.40 | .125 | .35 |
| 13 | .05 | 1.0 | 2.5e-4 | .0156 | .0139 | .42 | 6.88 | .080 | .58 | .05 | 1.0 | 1.4e-4 | .0141 | .78 | 6.00 | .106 | .37 |
| 14 | .05 | 1.0 | 1.4e-4 | .0160 | .0142 | .52 | 7.48 | .073 | .71 | .05 | 1.0 | 8.6e-5 | .0146 | .77 | 6.65 | .093 | .40 |
| 15 | .05 | 1.0 | 1.1e-4 | .0157 | .0140 | .55 | 8.10 | .057 | .60 | .05 | 1.0 | 9.6e-5 | .0149 | .78 | 7.19 | .071 | .51 |

Column notes:
- σ′ = mean sigmoid derivative of the mask.
- A = artifact map, group mean.
- budget-active = share of updates with budget hinge > 0.

HF residual activity, as the median of mean |tanh(HF raw)| per epoch, rises in R1 from 0.122 (ep7) to 0.165 (ep14) and
in R2 from 0.121 (ep7) to 0.154 (ep15). M·|ΔHF| follows the same pattern: R1 0.036→0.050, R2 0.036→0.046.

AMP and clipping over u5526..u16575:

| | AMP retry events (updates) | lowest G scale | lowest D scale during run | G clipped | D clipped |
|---|---|---|---|---|---|
| R1 | 7 (7): G×6, D×1 (u15235) | 32768 | 1024 (initial) → 16384 at end | 36.7 % | 97.0 % |
| R2 | 9 (8): G×7, D×2 (u13306, u15970) | 32768 | 1024 (initial) → 8192 at end | 23.6 % | 98.1 % |
| R0 (u5526..u5850) | 4 (4) | 16384 | 1024 | 52.9 % | 45.2 % |

Every retry was accepted. No `FAIL_CLOSED_AMP_OVERFLOW` and no numerical stop occurred.

### 2.5 Attribution

The table uses the N3 per-loss attribution plus the validated Adam-step decomposition. Columns:
- **gadv grad share**: projection of the gadv term on the total G gradient.
- **gadv trunk-step share**: projection of gadv's part of the actual Adam step on the shared G_res trunk, which excludes the
  mask-head row.
- **gadv Δlogit**: linearized change in mean mask logit from gadv's part of the step.
- **TOTAL Δlogit**: actual change in mean mask logit over the step.

Positive Δlogit opens the mask.

| u | R0 λ / grad / trunk / Δlogit(gadv) / Δlogit(TOTAL) | R1 λ / grad / trunk / Δlogit(gadv) / Δlogit(TOTAL) | R2 λ / grad / trunk / Δlogit(gadv) / Δlogit(TOTAL) |
|---|---|---|---|
| 5526 | .05 / .68 / .71 / −3.45 / −0.35 | 0 / 0 / – / – / +1.02 | 0 / 0 / – / – / +1.02 |
| 5600 | .05 / 1.03 / .19 / −4.79 / −0.72 | .0034 / .45 / .013 / −0.24 / +0.04 | .0007 / .11 / .005 / −0.04 / +0.53 |
| 5700 | .05 / .94 / .38 / −0.50 / −0.59 | .0079 / .26 / .030 / +0.006 / −0.22 | .0016 / .18 / .010 / −0.015 / −0.11 |
| 5750 | .05 / .88 / .56 / −0.14 / −0.61 | .0101 / .15 / .024 / −0.010 / +0.03 | .0020 / .13 / .001 / −0.024 / +0.13 |
| 5781 | .05 / closed (M=2e-6; step is 100 % momentum) | .0115 / .68 / −.010 / −0.003 / +0.02 | .0023 / .20 / .007 / −0.010 / +0.10 |
| 5850 | .05 / closed (M=1e-8) | .0147 / .53 / .11 / −0.12 / −0.05 | .0029 / .35 / .021 / +0.004 / +0.03 |
| 6630 | – | .05 / .30 / .032 / −0.047 / +0.04 | .0100 / .11 / −.026 / +0.0001 / −0.11 |
| 6631 / 11051 | – | .05 / .65 / .24 / −0.18 / −0.18 (u6631) | .05 / .72 / .12 / −0.14 / −0.11 (u11051) |
| 8840 | – | .05 / .59 / .20 / −0.09 / −0.11 | .03 / .70 / .10 / −0.07 / +0.06 |
| 11050 | – | .05 / .37 / .031 / −0.08 / +0.04 | .05 / .14 / .065 / −0.09 / +0.15 |
| 13800 | – | .05 / .61 / .090 / −0.29 / −0.17 | .05 / .56 / .12 / −0.25 / +0.19 |
| 16575 | – | .05 / .07 / −.001 / −0.19 / +0.26 | .05 / .17 / −.078 / −0.22 / +0.74 |

The remaining columns are in `…_GRADIENTS.csv`: artcon, spec, budget and tv shares, the fidelity group
(id+lm+parse+low+bg), momentum, and the effects on M, A and residual. In R1 and R2 at every point:
- artcon carries most of the trunk step (≈0.50–0.96);
- momentum carries 0.03–0.39;
- spec and budget are below 0.01;
- the fidelity group is below 0.08 of the trunk step.

Mask-logit levels:
- At u5526 the mask logit is +13.9 and the D logits are +11.5 (real) and −10.3 (fake).
- In R1/R2 the logit dips to about +6 during the ramp and climbs back to +10.4 (R1) and +11.8 (R2) by u16575.
- In R0 it fell from +14 to −45 by u5781.

Attribution check: attribution matches the recorded G grad norm within 0.4 % relative at every point. The Adam-step
prediction matches the next snapshot within 0.2 % (u6630 R1: 0.09 %, u11050 R2: 0.19 %).

## 3. Formal qualification decisions

The rules were predeclared in the tool before any branch ran:
- **Hard gates**: §2.1.
- **Artifact retention**: every complete post-ramp stage-2 epoch has median group-mean A ≥ 0.01, the frozen
  ARTIFACT_MIN, and no collapse occurred.
- **Preference order**: hard gates → prevents collapse to u16575 → artifact retained → eventual λ = 0.05 → smallest
  deviation → no architecture change → no optimizer/D reset.

| Branch | Decision | Basis |
|---|---|---|
| R0 | control; `R0_PARITY_PASS` | collapses at u5781 as required |
| **R1** | **QUALIFIED** (`ADV_RAMP_1E_QUALIFIED`) | all gates pass; reaches u16575; no event; post-ramp epochs 7–15 have median A 0.0117–0.0160 (all ≥ 0.01); eventual λ = 0.05 |
| **R2** | **QUALIFIED** (`ADV_RAMP_5E_QUALIFIED`) | all gates pass; reaches u16575; no event; post-ramp epochs 11–15 have median A 0.0132–0.0149 (all ≥ 0.01); eventual λ = 0.05 |
| R3 | NOT_LONG_QUALIFIED | Phase A only. It changes the final objective (λ = 0.01 ≠ 0.05) and had a severe transient closure |
| R4 | COLLAPSED_IN_PHASE_A (REJECTED) | formal collapse onset u5535 |

- QUALIFIED_REPAIR_CANDIDATES: **R1, R2**.
- PRIMARY_REPAIR_CANDIDATE: **R1**. Both are otherwise acceptable and both restore λ = 0.05. R1 is the smaller temporal
  deviation (1 epoch instead of 5) and reaches the frozen adversarial objective at u6630 instead of u11050. The choice
  is not based on TRAIN loss.

Proposed repair schedule, for owner decision only. It is not applied to any config:

```
stage 2 (u5526..u16575): lambda_adv(u) = 0.05 * ((u - 5526) / (6630 - 5526))   for 5526 <= u <= 6630
                         lambda_adv(u) = 0.05                                    for 6630 <  u <= 16575
all other curriculum terms, stage 1 and stage 3 unchanged
```

Verdict: `M7D1_N4_REPAIR_CANDIDATE_QUALIFIED`.

## 4. Interpretation

These are readings of the facts above, not gates.

1. **The ramp prevents the abrupt gadv monopolization.** In R0 the gadv term takes 0.68–1.0 of the G gradient from the
   first stage-2 update. It carries 0.19–0.71 of the shared-trunk Adam step and moves the mean mask logit by −3.5 to
   −4.8 per update; with momentum, that drives +14 → −45 by u5781.

   In R1 and R2, during u5526..u5850, gadv's trunk-step share is ≤ 0.11 and its linearized logit effect is ≤ 0.24 in
   magnitude. artcon and momentum dominate, and the net logit change stays between −0.22 and +1.02 per update. The
   stage-1-overconfident D (|logit| ≈ 11) is re-equilibrated while λ is small. By u5700 the D logits are about ±1 in R1
   and the gadv BCE has fallen from 10.3 to about 1–3.
2. **After λ reaches 0.05, the catastrophic mechanism stays controlled through u16575, but its pressure grows.**
   - gadv still pushes the mask closed at every post-ramp point: −0.05 to −0.29 linearized Δlogit per update. Its share
     of the G gradient is often 0.55–0.72.
   - Other terms counter it: artcon and Adam momentum open the mask.
   - The mask logit holds a margin of +6 to +12, compared with the roughly −60 cumulative linearized change that
     produced the R0 flip.
   - After λ reaches 0.05, the lowest mean(M) is 0.982 (R1) and 0.990 (R2).
3. **Warning: D is strengthening.** In both branches, epochs 6→15:
   - D_total falls about 9× (R1 0.52→0.057, R2 0.38→0.071);
   - L_gadv rises about 4× (R1 1.8→8.1, R2 3.3→7.2);
   - the D real/fake logit gap widens (u13800: R1 +8.5/−8.4, R2 +7.9/−5.9);
   - G clipping becomes common, 51–71 % of updates in epochs 14–15.

   This is the start of the D-saturation trend seen in the scientific run (OBS-01). Through u16575 it does not reproduce
   the R0 mask closure. It is the main open risk for stage 3, where the frozen λ = 0.10, and for longer horizons.
4. **The mask is saturating open.** M ≈ 1 everywhere, and the sigmoid derivative falls from about 3e-3 to about 1e-4 by
   epoch 15. The spatial mask is therefore close to a pass-through: artifact control comes from A and the HF residual,
   which grow slowly, not from spatial gating. This is the open-side mirror of the R0 saturation. It does not violate
   any predeclared rule. Whether that output is scientifically meaningful cannot be judged without VAL, which was not run.
5. **Artifacts are only modestly above the floor.** Median A stays 1.2–1.6× ARTIFACT_MIN, and the budget hinge is active
   on 42–91 % of updates, so the artifact is held near the floor rather than freely expressed.
6. **R1 vs R2.**
   - R2 keeps M slightly more open early: σ′ is lower from epoch 6, and its minimum mean(M) is higher (0.942 vs 0.892).
   - R2 has slightly lower gadv and higher D_total at equal epochs.
   - R2 has fewer G clips (23.6 % vs 36.7 %) but more AMP retries, including 2 D retries.
   - R1 has slightly larger A and HF activity from epoch 11.

   No gate separates them. The preference rule selects R1.
7. **Why not R3 or R4.**
   - Both change the final objective (λ ≠ 0.05) for the whole stage.
   - R4 collapses at u5535, faster than R0, so a smaller fixed λ is not monotonically safer.
   - R3 survived to u5850 only after a severe transient closure: half the pixels below 0.01 and logits near −1.8e3. Its
     recovery shows recoverability, not robustness, and it was not long-qualified.

## 5. Limitations

1. Only seed 42 was tested, from one exact pre-u5526 state.
2. Long qualification ends at u16575, the end of stage 2. Stage 3 (u16576..u66300, frozen λ = 0.10, s_hf = 0.15) was not run.
3. R1 and R2 both show D strengthening and rising gadv: D_total ↓ about 9×, L_gadv ↑ about 4×, clipping up to 71 % of
   updates in an epoch.
4. No collapse through u16575 does not prove stability to u66300.
5. R3's transient recovery does not establish robustness.
6. R4 shows that fixed-λ behavior is not monotonic; ramp behavior was tested only at two durations.
7. No VAL/TEST scientific-quality comparison was made. "Artifact retained" is a TRAIN-side characterization, A ≥
   ARTIFACT_MIN, and not a quality claim.
8. No official protocol amendment has been approved. R1 is a diagnostic candidate only.
9. The mask saturates open (σ′ ≈ 1e-4) in both qualified branches. A second saturation mode may matter for later stages.
10. Attribution is a per-snapshot linearization at 12 points per long branch, not a continuous trace.
11. The R5 D-reset branch could not be defined exactly (`R5_UNSUPPORTED_EXACT_STATE`).

Not performed: VAL or TEST access, checkpoint selection, bank generation, downstream FAS, seeds 1337/2026, B1/B2/B3,
scientific restart, production/frozen config change. N3 and N4 raw diagnostic evidence is preserved.
