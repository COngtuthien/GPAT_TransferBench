# M7D1-N5 — GPAT-B0 / E08 / seed 42 stage-3 adversarial-transition qualification

Labels: DIAGNOSTIC_ONLY · NOT_SCIENTIFIC · NOT_A_REPLACEMENT_RUN · NOT_ELIGIBLE_FOR_VAL_SELECTION ·
NOT_ELIGIBLE_FOR_BANK · NOT_ELIGIBLE_FOR_PAPER_RESULT · DIAGNOSTIC_REPAIR_VARIANT · NOT_AN_APPROVED_PROTOCOL_CHANGE ·
STAGE3_TRANSITION_QUALIFICATION.

Nothing here changes a production config, the frozen specification, the completed scientific run, or any N3/N4
artifact. The policy named below is a *diagnostic recommendation* for an owner decision. It is not an official, native
or frozen protocol.

Every statement is tagged with one of four labels:
- **[MEASURED]**: read from the raw branch artifacts by `--scenario collect`.
- **[FORMAL_DECISION]**: the mechanical result of the rules predeclared in the tool before any branch ran.
- **[INTERPRETATION]**: a reading of the measured facts; not a gate.
- **[LIMITATION]**: a stated limit of the evidence.

**Scope**
- Scientific run: `7b799fbd6d0426da`. Authority commit: `6fc875e`, the N4 qualification.
- Diagnostic root on the GPU host: `<rt>/diagnostics/m7/M7D1_N5_seed42/`, where `<rt>` is
  `/home/student20261/workdir/GPAT_TransferBench_runtime`.
- Branch windows:
  - S0, S1, S2: u16576..u27625, 11050 updates, epochs 16–25.
  - S3, S4: u16576..u16900, 325 updates.
- Branches differ only in the listed stage-3 terms. LR, lambda_con, lambda_spec, D updates, AMP policy, optimizers,
  clipping, EMA and TRAIN order are the frozen scientific ones.

**Evidence** (SHA-256 in `M7D1_N5_SEED42_STAGE3_EVIDENCE_SHA256.txt`):
- `M7D1_N5_SEED42_STAGE3_COLLECTED.json` comes from `tools/m7d1_n5_stage3_qualification.py --scenario collect`. It
  re-reads:
  - the raw branch artifacts;
  - the read-only scientific `metrics.jsonl`;
  - the anchor result;
  - the attribution results.
- `finalize()` in the same tool derives the following files with no new measurement. The tests re-derive them byte
  for byte:
  - `…_QUALIFICATION.json`: hard gates, formal decisions, verdict and proposed schedule;
  - `…_SCREEN.csv`: every update u16576..u16900 for S0–S4;
  - `…_LONG.csv`: per-epoch distributions for S0/S1/S2, epochs 16–25;
  - `…_GRADIENTS.csv`: per-snapshot N3/N4 loss attribution and the Adam-step split, 45 rows.

## 1. Exact start state [MEASURED]

N4 saved only *pre-update* snapshots, so the latest R1 state it holds is the one before u16575. The exact
post-u16575 state was produced as follows.

**Step 1: verified base.**
- File: `<rt>/diagnostics/m7/M7D1_N4_seed42/snapshots/branch_R1/pre_update_16575.pt`.
- sha256 `ea26d463fa67b1adbfe70d56f997da1e32dea394a536abe2d9156e22f741a612`, matching the N4 snapshot meta and the N4
  `branch_R1.json` snapshot listing.
- N4 capture code head `d88563a`.
- Embedded position: stage generator, epoch 15, next_group 1105, global_update 16574.
- `n3.apply_snapshot` checked the state digest on load. The digest covers G_res, E_art, D, EMA, both optimizers, both
  AMP scalers, all RNG streams, buffers, curriculum and data-order position.

**Step 2: anchor replay (`--scenario anchor`).**
- The harness re-ran u16575 with the R1 schedule (λ = 0.05) and compared it with N4 R1. Result: `ANCHOR_EXACT`.
  - The optimizer record is bitwise equal: field mismatch [].
  - AMP retry events are equal; there are none.
  - The N3 mask row is equal.
  - The N4 per-update row is equal: field mismatch [].
- The epoch-15 end then ran exactly as in the scientific runner. The recovery save at u16575 and the EMA candidate for
  epoch 15 were intercepted and never written.
- The harness captured the pre-u16576 state without stepping u16576:
  - file `snapshots/anchor/pre_update_16576.pt`;
  - sha256 `99d068341632f05e8cb253baa81f9785bc9ea99c2b1efc054c1e49b8c25f2a76`, 733267493 bytes;
  - position epoch 16 / next_group 1 / global_update 16575;
  - code head `6fc875e`;
  - u16576 group `[[7241,3691,2016,8470],[3661,4713,5657,2457]]`.

**Step 3: every branch starts there.**
- Every branch S0–S4 loaded this snapshot, with the sha and state digest checked.
- Each branch then captured its own u16576 pre-update snapshot live. That snapshot has the same state digest as the
  anchor.
- No update before u16575 was re-run. The single replayed update, u16575, is a bitwise-verified replay, not a new
  result.

## 2. Branches

| Branch | λ_adv on u16576..u27625 | s_hf | Window | Role |
|---|---|---|---|---|
| S0 ORIGINAL | 0.10 (frozen stage 3) | 0.15 | u16576..u27625 | control / candidate |
| S1 ADV_RAMP_1E | `0.05 + 0.05 * ((u-16576)/(17680-16576))` for u ≤ 17680, then 0.10 | 0.15 | u16576..u27625 | candidate |
| S2 ADV_RAMP_5E | `0.05 + 0.05 * ((u-16576)/(22100-16576))` for u ≤ 22100, then 0.10 | 0.15 | u16576..u27625 | candidate |
| S3 ONLY_S_HF | 0.05 held | 0.15 | u16576..u16900 | causal diagnostic, not a candidate |
| S4 ONLY_ADV_JUMP | 0.10 | 0.10 held | u16576..u16900 | causal diagnostic, not a candidate |

**[MEASURED] Schedule checks.** Collect checked every recorded λ_adv and s_hf bit-exactly against the branch
definition, with 0 mismatches in every branch:
- S1 gives 0.05 at u16576 and 0.10 at u17680, exactly.
- S2 gives 0.05 at u16576 and 0.10 at u22100, exactly.
- u17681 and u22101 equal 0.10.

The ramp is coded as `0.05 + 0.05 * ratio`, which gives exact endpoints. LR, lambda_con and lambda_spec equal both the
frozen `runtime_contract` and the scientific record of the same update, at every row.

## 3. Hard-gate audit [MEASURED]

Identical for S0–S4 unless stated.

| Gate | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| status | COMPLETED_TO_END | COMPLETED_TO_END | COMPLETED_TO_END | COMPLETED_TO_END | COMPLETED_TO_END |
| COMPLETE updates, contiguous from u16576 | 11050 (→u27625) | 11050 (→u27625) | 11050 (→u27625) | 325 | 325 |
| exact start (anchor sha + state digest; live u16576 digest) | ✓ | ✓ | ✓ | ✓ | ✓ |
| provenance (code_head 6fc875e, tool 7fa04825 for every run) | ✓ | ✓ | ✓ | ✓ | ✓ |
| TRAIN group digests equal across branches; u16576 = anchor group | ✓ | ✓ | ✓ | ✓ | ✓ |
| epoch/group/sizes/weights/LR/con/spec == scientific record | ✓ | ✓ | ✓ | ✓ | ✓ |
| λ_adv / s_hf bit-exact | ✓ | ✓ | ✓ | ✓ | ✓ |
| AMP retry integrity (attempts = retries+1; each backoff halves the offending scaler; scales fall only at retries) | ✓ | ✓ | ✓ | ✓ | ✓ |
| non-finite losses / M / A / grad norms / end parameters | 0 | 0 | 0 | 0 | 0 |
| terminal AMP overflow | none | none | none | none | none |
| write firewall (denied / outside allowed) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| VAL / TEST / non-TRAIN access (incl. attribution runs) | 0 | 0 | 0 | 0 | 0 |
| formal collapse event | none | none | none | none | none |

**Other integrity checks**
- Long branches opened 176760 TRAIN images each, covering 12668 unique ids.
- Epoch-end recovery saves inside the windows (u17680, u18785 … u26520) and EMA-candidate writes (epochs 16–24) were
  intercepted and never written.
- The collect firewall recorded 0 denied writes and 0 writes outside the allowed roots.
- The scientific run root is unchanged:
  - `metrics.jsonl` sha256 is `5f4aa9a0…d632f`;
  - the path/size/mtime listing digest is `ac64a063…893a`.

  Both values are identical before any N5 run (`results/scientific_root_pre.txt`) and after the last one
  (`results/scientific_root_post.txt`). They are also identical to N4.

## 4. Phase A — transition screen u16576..u16900 [MEASURED]

Values come from `…_SCREEN.csv`, 325 COMPLETE updates per branch. "med" is the median over the window.

| | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| λ_adv first → last | .10 → .10 | .05 → .0647 | .05 → .0529 | .05 → .05 | .10 → .10 |
| s_hf | .15 | .15 | .15 | .15 | .10 |
| min mean(M) (update) | 0.9974 (u16617) | 0.9976 (u16814) | 0.9978 (u16617) | 0.9987 (u16617) | 0.9975 (u16686) |
| min M_p10 | 0.996 | 0.994 | 0.997 | 0.997 | 0.996 |
| lowest pixel M | 0.156 | 0.154 | 0.085 | 0.060 | 0.042 |
| max frac M < 0.1 / < 0.01 / < 0.001 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1e-4 / 0 / 0 | 0 / 0 / 0 |
| min frac M > 0.5 | 0.9999 | 0.9999 | 0.9997 | 0.9997 | 0.9995 |
| med frac M > 0.99 | 0.9992 | 0.9992 | 0.9994 | 0.9991 | 0.9991 |
| mask logit mean: med / min | 10.2 / 7.95 | 10.4 / 7.34 | 10.7 / 8.13 | 10.4 / 8.30 | 11.0 / 8.45 |
| mask logit std (med) | 2.80 | 2.65 | 2.70 | 2.66 | 3.02 |
| sigmoid derivative M(1−M), med | 2.5e-4 | 2.0e-4 | 1.6e-4 | 2.1e-4 | 1.7e-4 |
| mask spatial variance, med | 6.5e-7 | 7.5e-7 | 4.3e-7 | 7.7e-7 | 9.4e-7 |
| A_mean med / A_p90 med | .0197 / .0412 | .0189 / .0390 | .0192 / .0395 | .0189 / .0396 | .0166 / .0335 |
| \|x̂ − x_t\| med | .0176 | .0170 | .0172 | .0170 | .0148 |
| HF residual mean \|tanh\|, med | .136 | .130 | .133 | .130 | .172 |
| M·\|ΔHF\|, med | .061 | .059 | .060 | .059 | .052 |
| gadv: u16576 / med / p90 | 7.73 / 8.22 / 11.07 | 7.73 / 8.27 / 11.09 | 7.73 / 8.38 / 11.14 | 7.73 / 8.34 / 10.91 | 7.40 / 8.18 / 10.98 |
| D_total: u16576 / med / p90 | .022 / .056 / .209 | .022 / .051 / .216 | .022 / .044 / .211 | .022 / .046 / .207 | .027 / .053 / .217 |
| D logits real / fake (med) | +8.5 / −8.2 | +8.8 / −8.3 | +8.8 / −8.3 | +8.7 / −8.3 | +8.7 / −8.2 |
| G grad norm med / D grad norm med | 2.55 / 6.17 | 1.44 / 5.31 | 1.37 / 5.23 | 1.32 / 5.26 | 1.80 / 5.59 |
| share G clipped / D clipped | 1.00 / 0.90 | 0.99 / 0.89 | 0.97 / 0.87 | 0.94 / 0.90 | 1.00 / 0.89 |
| AMP retries (updates) | 1 (u16612, G) | 0 | 0 | 0 | 0 |
| lowest G / D scale | 16384 / 16384 | 32768 / 16384 | 32768 / 16384 | 32768 / 16384 | 32768 / 16384 |
| G_total med | 1.38 | 1.02 | 0.98 | 0.97 | 1.38 |

Every S0–S4 Phase-A row, including LR, budget, artcon, spec, id, lm, parse, low, bg, tv, logit quantiles and D
confidences, is in `…_SCREEN.csv`.

**[FORMAL_DECISION] Phase-A outcome letters (spec §9): `['A']`.**
- S0 stays healthy.
- B, C and E require S0 to fail, so they do not apply.
- D requires S3 to fail, and it did not.

## 5. Formal collapse events [FORMAL_DECISION]

`N5_IDENTITY_COLLAPSE_V1`: mean(M) < 0.01 for 10 consecutive COMPLETE updates. It was declared in the tool before any
branch ran and has not been changed. **No branch produced an event.**

[MEASURED] Lowest mean(M) per branch:

| Branch | Window | Lowest mean(M) | Updates with mean(M) < 0.01 |
|---|---|---|---|
| S0 | to u27625 | 0.9945 (epoch 16) | 0 |
| S1 | to u27625 | 0.9941 (epoch 17) | 0 |
| S2 | to u27625 | 0.9963 (epoch 17) | 0 |
| S3 | to u16900 | 0.9987 | 0 |
| S4 | to u16900 | 0.9975 | 0 |

## 6. Gradient attribution [MEASURED]

The method is the validated N3/N4 per-loss attribution plus the Adam-step decomposition on the shared G_res trunk, which
excludes the mask-head row. Snapshots analysed:
- S0, S1 and S2 at u16576, 16577, 16580, 16600, 16700, 16900, 17680, 17681, 22100, 22101 and 27625;
- S3 and S4 at the six screen points.

Columns:
- **grad**: projection of the gadv term on the total G gradient.
- **trunk**: gadv's share of the actual Adam step on the trunk.
- **Δlg(gadv)**: linearized change in mean mask logit from gadv's part of the step.
- **Δlg(TOT)**: the same quantity for the whole step.

Positive values open the mask.

| u | S0 λ / grad / trunk / Δlg(gadv) / Δlg(TOT) | S1 λ / grad / trunk / Δlg(gadv) / Δlg(TOT) | S2 λ / grad / trunk / Δlg(gadv) / Δlg(TOT) |
|---|---|---|---|
| 16576 | .10 / .92 / .16 / +0.08 / +0.18 | .05 / .72 / .07 / +0.07 / +0.18 | .05 / .72 / .07 / +0.07 / +0.18 |
| 16577 | .10 / .84 / .32 / −0.35 / −0.19 | .05 / .64 / .16 / −0.30 / −0.13 | .05 / .64 / .16 / −0.30 / −0.13 |
| 16580 | .10 / .96 / .37 / −0.70 / −0.45 | .0502 / .79 / .19 / −0.76 / −0.29 | .05 / .84 / .20 / −0.78 / −0.36 |
| 16600 | .10 / .94 / .53 / −0.03 / −0.15 | .0511 / .60 / .30 / −0.02 / −0.11 | .0502 / .36 / .03 / −0.42 / −0.05 |
| 16700 | .10 / .96 / .62 / −0.88 / −1.03 | .0556 / .81 / .16 / −0.44 / −0.28 | .0511 / .71 / .11 / −0.13 / +0.02 |
| 16900 | .10 / .81 / .20 / +0.03 / +0.29 | .0647 / .53 / .09 / −0.19 / −0.15 | .0529 / .66 / .09 / −0.27 / −0.10 |
| 17680 | .10 / .90 / .29 / −0.40 / +0.08 | .10 / .65 / .09 / −0.21 / +0.35 | .06 / .45 / .07 / −0.24 / +0.33 |
| 17681 | .10 / .97 / .10 / −0.45 / −0.14 | .10 / .82 / .15 / −0.64 / −0.11 | .06 / .66 / .04 / −0.21 / +0.32 |
| 22100 | .10 / .77 / .18 / −0.16 / +0.43 | .10 / .81 / .19 / −0.06 / +0.67 | .10 / .98 / .32 / −0.47 / −0.18 |
| 22101 | .10 / .98 / .21 / −0.58 / −0.43 | .10 / .93 / .32 / −0.74 / −0.55 | .10 / .90 / .38 / −0.88 / −0.81 |
| 27625 | .10 / .84 / .22 / −1.02 / −0.25 | .10 / .89 / .27 / −0.47 / −0.13 | .10 / .87 / .34 / +1.39 / +4.30 |

S3 and S4 at the screen points, as grad / trunk / Δlg(gadv) / Δlg(TOT):
- **S3**, λ 0.05 and s_hf 0.15: grad 0.49–0.82, trunk 0.03–0.21, Δlg(gadv) −0.77…+0.07, Δlg(TOT) −0.30…+0.41.
- **S4**, λ 0.10 and s_hf 0.10: grad 0.67–0.90, trunk 0.08–0.37, Δlg(gadv) −0.92…+0.08, Δlg(TOT) −0.64…+0.47.

**Other columns** (in `…_GRADIENTS.csv`): artcon, spec, budget and tv shares, the fidelity group
(id+lm+parse+low+bg), momentum, and the effects on M, A and the residual. At every point:
- artcon carries 0.16–0.85 of the trunk step and opens the mask, with linearized Δlogit between −0.18 and +3.36;
- momentum carries 0.06–0.66;
- the fidelity group carries below 0.19, mostly below 0.05.

**Mask-logit and D-logit levels.**
- At u16576 the mask logit is +11.0 and the D logits are +6.3 (real) and −7.7 (fake).
- At u27625 the mask logit is +8.6 (S0), +8.9 (S1) and +9.4 (S2).
- At u27625 the D real/fake logits are +18.8/−8.0 (S0), +16.3/−12.4 (S1) and +13.6/−14.0 (S2).

**Attribution checks.**
- Attribution matches the recorded G grad norm within 0.25 % relative at every point; the worst is 2.5e-3 at S2 u17680.
- The S2 u27625 total of +4.3 is dominated by artcon (+3.36) at a single snapshot.

## 7. Phase B — long qualification to u27625 [MEASURED]

Values are medians over the 1105 updates of each epoch, from `…_LONG.csv`:
- λ = λ_adv at the first → last update of the epoch.
- M = median mean(M); Mmin = epoch minimum of mean(M).
- σ′ = mean sigmoid derivative.
- A = group-mean artifact.
- bud = share of updates with the budget hinge active.
- gadv and D_total are given as median / p90 and median / max.
- Dr / Df = D logits, real / fake.
- Gc / Dc = share of updates with grad norm > 1.

### S0 (original stage 3)

| Ep | λ | M | Mmin | σ′ | A | bud | gadv | D_total | Dr / Df | Gc | Dc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | .10 | .9995 | .9945 | 4.5e-4 | .0190 | .27 | 8.31 / 11.2 | .055 / .91 | +8.7 / −8.3 | 1.00 | .88 |
| 17 | .10 | .9996 | .9965 | 4.2e-4 | .0192 | .19 | 8.78 / 11.8 | .045 / 1.04 | +9.2 / −8.7 | 1.00 | .87 |
| 18 | .10 | .9995 | .9948 | 5.3e-4 | .0196 | .14 | 9.22 / 12.4 | .038 / .95 | +9.8 / −9.2 | 1.00 | .85 |
| 19 | .10 | .9995 | .9968 | 4.7e-4 | .0192 | .16 | 9.76 / 12.9 | .033 / .86 | +10.1 / −9.7 | 1.00 | .83 |
| 20 | .10 | .9995 | .9963 | 5.1e-4 | .0194 | .15 | 10.27 / 13.5 | .027 / 1.01 | +10.6 / −10.2 | 1.00 | .79 |
| 21 | .10 | .9995 | .9978 | 4.7e-4 | .0198 | .14 | 10.74 / 13.9 | .020 / 1.06 | +11.2 / −10.7 | 1.00 | .75 |
| 22 | .10 | .9995 | .9978 | 4.8e-4 | .0206 | .19 | 11.36 / 14.5 | .015 / 1.10 | +11.8 / −11.3 | 1.00 | .70 |
| 23 | .10 | .9995 | .9968 | 5.1e-4 | .0205 | .22 | 11.46 / 14.8 | .014 / 1.67 | +12.3 / −11.5 | 1.00 | .71 |
| 24 | .10 | .9996 | .9975 | 3.9e-4 | .0202 | .16 | 12.04 / 15.4 | .012 / 1.27 | +12.8 / −12.0 | 1.00 | .67 |
| 25 | .10 | .9997 | .9987 | 3.3e-4 | .0205 | .15 | 12.43 / 15.9 | .0085 / 1.22 | +13.2 / −12.4 | 1.00 | .60 |

### S1 (1-epoch ramp, λ = 0.10 from u17680)

| Ep | λ | M | Mmin | σ′ | A | bud | gadv | D_total | Dr / Df | Gc | Dc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | .05→.10 | .9997 | .9975 | 2.6e-4 | .0186 | .28 | 8.36 / 11.2 | .052 / .85 | +8.8 / −8.3 | 1.00 | .90 |
| 17 | .10 | .9996 | .9941 | 3.9e-4 | .0194 | .27 | 8.74 / 11.6 | .043 / 1.02 | +9.3 / −8.7 | 1.00 | .86 |
| 18 | .10 | .9995 | .9957 | 4.9e-4 | .0194 | .19 | 9.34 / 12.4 | .036 / .88 | +9.9 / −9.3 | 1.00 | .84 |
| 19 | .10 | .9996 | .9967 | 4.2e-4 | .0196 | .15 | 9.82 / 12.9 | .031 / .88 | +10.3 / −9.8 | 1.00 | .82 |
| 20 | .10 | .9996 | .9972 | 3.6e-4 | .0200 | .18 | 10.34 / 13.7 | .025 / 1.28 | +10.8 / −10.3 | 1.00 | .78 |
| 21 | .10 | .9997 | .9984 | 2.6e-4 | .0198 | .10 | 10.82 / 14.0 | .019 / 1.02 | +11.3 / −10.8 | 1.00 | .74 |
| 22 | .10 | .9998 | .9976 | 2.2e-4 | .0206 | .12 | 11.45 / 14.8 | .016 / .89 | +11.8 / −11.4 | 1.00 | .72 |
| 23 | .10 | .9997 | .9985 | 3.4e-4 | .0207 | .15 | 11.59 / 14.8 | .014 / 1.06 | +12.4 / −11.6 | 1.00 | .69 |
| 24 | .10 | .9997 | .9983 | 2.5e-4 | .0211 | .13 | 11.96 / 15.6 | .012 / 1.23 | +12.9 / −11.9 | 1.00 | .67 |
| 25 | .10 | .9997 | .9988 | 2.5e-4 | .0218 | .13 | 12.53 / 16.0 | .0083 / 1.29 | +13.3 / −12.5 | 1.00 | .61 |

### S2 (5-epoch ramp, λ = 0.10 from u22100)

| Ep | λ | M | Mmin | σ′ | A | bud | gadv | D_total | Dr / Df | Gc | Dc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | .05→.06 | .9998 | .9975 | 2.1e-4 | .0184 | .33 | 8.42 / 11.2 | .051 / .90 | +8.8 / −8.4 | .95 | .89 |
| 17 | .06→.07 | .9997 | .9963 | 3.0e-4 | .0189 | .33 | 9.00 / 11.9 | .042 / 1.18 | +9.3 / −9.0 | .99 | .87 |
| 18 | .07→.08 | .9996 | .9966 | 3.6e-4 | .0186 | .29 | 9.44 / 12.4 | .034 / .97 | +9.9 / −9.4 | 1.00 | .82 |
| 19 | .08→.09 | .9997 | .9981 | 2.9e-4 | .0190 | .18 | 10.08 / 13.1 | .031 / .77 | +10.4 / −10.0 | 1.00 | .82 |
| 20 | .09→.10 | .9997 | .9970 | 2.8e-4 | .0207 | .18 | 10.32 / 13.5 | .024 / .99 | +10.7 / −10.3 | 1.00 | .78 |
| 21 | .10 | .9996 | .9966 | 4.4e-4 | .0207 | .19 | 10.87 / 14.2 | .020 / .99 | +11.4 / −10.8 | 1.00 | .75 |
| 22 | .10 | .9996 | .9981 | 4.0e-4 | .0205 | .17 | 11.31 / 14.7 | .017 / .89 | +11.9 / −11.3 | 1.00 | .71 |
| 23 | .10 | .9996 | .9984 | 3.4e-4 | .0218 | .17 | 11.57 / 14.9 | .016 / 1.06 | +12.4 / −11.5 | 1.00 | .70 |
| 24 | .10 | .9996 | .9977 | 3.5e-4 | .0212 | .17 | 12.10 / 15.6 | .012 / 1.39 | +12.8 / −12.1 | 1.00 | .67 |
| 25 | .10 | .9996 | .9979 | 3.7e-4 | .0216 | .15 | 12.42 / 16.2 | .0077 / 1.09 | +13.3 / −12.4 | 1.00 | .59 |

The S2 epoch-20 row spans u20996..u22100. Its last update is the first at exactly 0.10.

### Mask health, open side [MEASURED]

Across all S0/S1/S2 epochs 16–25:
- median mean(M) is 0.9995–0.9998;
- median share of pixels with M > 0.99 is 0.995–0.999;
- epoch minimum M_p01 is 0.930–0.979;
- epoch-maximum share of pixels with M < 0.01 is at most 6e-4 (S1 epoch 17), and 0 in most epochs;
- median σ′ is 2.1e-4 to 5.3e-4;
- median mask spatial variance is about 1e-6, at 4e-7 to 9e-7 in Phase A.

**Mask logit.** Median mean logit is +9.5 to +10.8 in every epoch. The epoch minimum is +6.9 to +8.9.

**Image level.** The lowest per-image mean M in any Phase-A update is 0.991.

The mask therefore remains spatially almost uniform and almost fully open in all three branches. No artifact
localization statistic is available without adding a new evaluator, so none was computed.

### Artifact retention [MEASURED → FORMAL_DECISION]

Median group-mean A by epoch, 16 → 25:

| Branch | Median A, epoch 16 → 25 | Min ≥ ARTIFACT_MIN 0.01 |
|---|---|---|
| S0 | 0.0190, 0.0192, 0.0196, 0.0192, 0.0194, 0.0198, 0.0206, 0.0205, 0.0202, 0.0205 | ✓ (min 0.0190) |
| S1 | 0.0186, 0.0194, 0.0194, 0.0196, 0.0200, 0.0198, 0.0206, 0.0207, 0.0211, 0.0218 | ✓ (min 0.0186) |
| S2 | 0.0184, 0.0189, 0.0186, 0.0190, 0.0207, 0.0207, 0.0205, 0.0218, 0.0212, 0.0216 | ✓ (min 0.0184) |

**Other artifact signals** [MEASURED]:
- \|x̂ − x_t\| median is 0.0165–0.0195.
- HF residual median \|tanh\| is 0.127–0.151.
- M·\|ΔHF\| is 0.057–0.068.

All of these rise slowly over the window in all three branches. Budget-hinge activity falls from 0.27–0.33 in epoch 16
to 0.13–0.15 in epoch 25.

**[FORMAL_DECISION]** `N5_ARTIFACT_RETENTION_CHARACTERIZATION_V1` (all 10 complete epochs have median A ≥ 0.01, and no
collapse occurred) is met by S0, S1 and S2.

### AMP and clipping over u16576..u27625 [MEASURED]

| | AMP retry events (updates) | G / D events | lowest G scale | lowest D scale | G clipped (epochs) | D clipped, epoch 16 → 25 |
|---|---|---|---|---|---|---|
| S0 | 8 (7) | 4 / 4 (u20736: two G halvings) | 4096 | 8192 | 100 % every epoch | 88 % → 60 % |
| S1 | 10 (8) | 5 / 5 (u21233: three G halvings) | 2048 | 8192 | 100 % every epoch | 90 % → 61 % |
| S2 | 10 (10) | 5 / 5 | 8192 | 8192 | 95 % / 99 % ep 16–17, then 100 % | 89 % → 59 % |

Every retry was accepted. No `FAIL_CLOSED_AMP_OVERFLOW` and no numerical stop occurred. Every retry update is in
`…_QUALIFICATION.json` (`amp_retry_updates`).

## 8. Formal qualification decisions [FORMAL_DECISION]

The rules were predeclared in the tool before any branch ran:
- **Hard gates**: §3.
- **Artifact retention**: §7.
- **Preference order**: hard gates → no closed-mask collapse → meaningful artifact → eventual λ = 0.10 → smallest
  deviation from the frozen curriculum → no architecture change.
- **Decision order**: S0, then S1, then S2. No weighted score is used.

| Branch | Decision | Basis |
|---|---|---|
| **S0** | **QUALIFIED**, `STAGE3_ORIGINAL_TRANSITION_QUALIFIED` | all gates pass; reaches u27625; no event; epochs 16–25 median A 0.0190–0.0206; eventual λ = 0.10; zero deviation from frozen stage 3 |
| S1 | QUALIFIED | all gates pass; reaches u27625; no event; median A 0.0186–0.0218 |
| S2 | QUALIFIED | all gates pass; reaches u27625; no event; median A 0.0184–0.0218 |
| S3 | NOT_A_CANDIDATE | causal diagnostic; no collapse in Phase A |
| S4 | NOT_A_CANDIDATE | causal diagnostic; no collapse in Phase A |

- Qualified: S0, S1, S2.
- **Preferred stage-3 policy: S0**, keeping the original frozen stage-3 transition. Selection is by the predeclared
  order, not by TRAIN loss.
- The anchor gate passes.

**Proposed complete λ_adv schedule** (`DIAGNOSTIC_RECOMMENDATION_ONLY`; owner approval required; no config changed):

```
u1     .. u5525 : 0.0                                        (frozen stage 1)
u5526  .. u6630 : 0.05 * ((u - 5526) / (6630 - 5526))        (N4 R1 stage-2 ramp)
u6631  .. u16575: 0.05                                       (frozen stage-2 value)
u16576 .. u66300: 0.10                                       (frozen stage 3, unchanged)
s_hf, lambda_con, lambda_spec, LR: frozen schedule unchanged
qualified only on seed 42 through u27625; not evaluated beyond u27625
```

Probes, also in `…_QUALIFICATION.json`: λ(5526) = 0.0, λ(6078) = 0.025, λ(6630) = 0.05, λ(16575) = 0.05,
λ(16576) = 0.10, λ(66300) = 0.10.

Verdict: `M7D1_N5_STAGE3_TRANSITION_QUALIFIED`.

## 9. Interpretation [INTERPRETATION]

These are readings of the facts above, not gates.

1. **S0 causal reading: the 0.05 → 0.10 jump does not repeat the N3 monopolization.**
   - At the stage-2 switch-on in N3/N4 R0, gadv carried 0.19–0.71 of the trunk step and moved the mean mask logit by
     −3.5 to −4.8 per update. That drove +14 → −45 by u5781.
   - At the S0 stage-3 jump, gadv dominates the *gradient direction*: 0.81–0.96 of the G gradient.
   - But its linearized logit effect is +0.08 to −0.88 per update, and the net logit change is −1.03 to +0.29.
   - artcon and momentum counter it, and the mask logit stays at +8 to +13 throughout Phase A.
   - The pre-transition D is already confident (D_total 0.022 at u16576), unlike the stage-1 D at u5526.

   The jump is absorbed because:
   - the G gradient is already clipped on essentially every update, with clip coefficient 0.25–0.58;
   - the Adam step is normalized;
   - the mask starts deep in the open-saturated region with a margin of about +10 logits.

   So doubling λ changes the clipped direction only modestly.
2. **S3/S4 isolation.** Neither diagnostic collapses.
   - S4 (adv jump only) behaves like S0 in mask terms.
   - S3 (s_hf only) shows the smallest gadv trunk shares.
   - S4's lower A and \|x̂ − x_t\| (0.0166 / 0.0148 vs 0.019 / 0.017) and higher HF |tanh| (0.172) follow from its lower
     s_hf scale, not from mask closure.

   With S0 healthy (outcome A), neither the λ_adv jump nor the s_hf jump is shown to be harmful on this horizon.
3. **Ramps (S1/S2) are not needed on this evidence.** They lower gadv's gradient and trunk shares while λ is small,
   S2 most clearly in epochs 16–18. After each ramp ends, attribution at u17681 (S1) and u22101 (S2) looks like S0.
   All three end in nearly identical states. The predeclared order therefore selects S0, the smallest deviation.
4. **Warning: D keeps strengthening, as in N4.** In all three branches, epochs 16 → 25:
   - D_total median falls about 6.5× (0.052–0.055 → 0.0077–0.0085);
   - gadv median rises about 1.5× (8.3–8.4 → 12.4–12.5);
   - the D real/fake logit gap widens from ±8.5 to ±13.3;
   - G clipping is near-total.

   The first links of the N3 chain are present: D confidence, then gadv dominance of the gradient direction. The later
   links do not appear through u27625: gadv's trunk-step share stays at 0.04–0.38 at the long-window points, and there is no
   closure trend. Mask logit and mean(M) are flat, and Mmin per epoch even rises slightly in late epochs.

   D clipping falls (about 90 % → 60 %) as D gets more confident. The risk grows with time, and the window covers
   only 11050 of the 49725 stage-3 updates.
5. **The mask is saturated open.**
   - M ≈ 1 everywhere, with spatial variance about 1e-6.
   - σ′ is about 2–5e-4, roughly constant, not falling further as it did in N4 stage 2.

   Artifact control comes from A and the HF residual. These grow slowly: median A rises from about 0.019 to about
   0.021, HF |tanh| from about 0.13 to 0.15. Spatial gating plays no part. This is the open-side mirror of the N3
   saturation. It violates no frozen gate, and no new threshold was introduced.
6. **Artifacts are non-trivial in the TRAIN sense.** Median A is 1.8–2.2× ARTIFACT_MIN, above the 1.2–1.6× seen in N4
   stage 2. The budget hinge is active on only 10–33 % of updates.

## 10. Limitations [LIMITATION]

1. Only seed 42 was tested, from one exact R1 state (u16575 → anchor u16576).
2. The long qualification ends at u27625, 11050 of 49725 stage-3 updates. Behaviour from u27626 to u66300 is not
   qualified.
3. D strengthening is still under way at u27625. A late recurrence of the N3 chain cannot be excluded.
4. No VAL/TEST quality comparison was made. "Artifact retained" is a TRAIN-side characterization (A ≥ ARTIFACT_MIN),
   not a quality claim.
5. The mask is saturated open in all branches, so the spatial mask is close to a pass-through. Whether that is
   scientifically acceptable is an owner and science decision.
6. Attribution is a per-snapshot linearization at 11 (long) or 6 (short) points, not a continuous trace.
7. S3 and S4 were screened only to u16900.
8. The exact start state needed a one-update anchor replay (u16575), because N4 kept only pre-update snapshots. The
   replay is bitwise-equal to N4 R1, but the method itself is an N5 construction, disclosed here for owner acceptance.
9. A shared GPU and another user's job (about 5.5–7 GB) slowed the long branches from about 4.4 to about 7.5 s/update.
   This is a warning only; no other user's job was touched.
10. The first parallel `analyze_S0` attempt ran out of GPU memory under this contention. Its second attempt was refused
    because the first attempt's run directory existed. Both are preserved:
    - `analyze_S0_attempt1_OOM/` with log `results/analyze_S0_attempt1_OOM.log`;
    - `results/analyze_S0_attempt2_root_exists.log`.

    The third attempt ran alone and is the one used. Attribution only reads snapshots and does not change any branch
    state.
11. The first `collect` failed with a `TypeError`: the AMP-retry check divided per-optimizer scale dicts as if they
    were numbers. That path had not run during the smoke tests. The fix (`_halved`) is the only change between the
    run-time tool (`7fa04825…`, used by anchor, every branch and every analyze run) and the committed tool
    (`4b24f3d3…`, used by collect). The failed log is preserved as `results/collect_attempt1_TypeError.log`.

Not performed: VAL or TEST access, checkpoint selection, bank generation, downstream FAS, seeds 1337/2026, B1/B2/B3,
scientific restart, production/frozen config change. N3 and N4 raw diagnostic evidence is untouched.
