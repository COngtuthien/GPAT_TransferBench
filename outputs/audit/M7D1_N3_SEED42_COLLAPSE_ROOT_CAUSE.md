# M7D1-N3 — GPAT-B0 / E08 / seed 42 generator identity collapse: root-cause localization

Labels: DIAGNOSTIC_ONLY · NOT_SCIENTIFIC · NOT_A_REPLACEMENT_RUN · NOT_ELIGIBLE_FOR_VAL_SELECTION ·
NOT_ELIGIBLE_FOR_BANK · NOT_ELIGIBLE_FOR_PAPER_RESULT. No fix is implemented here.

Scope: scientific run `7b799fbd6d0426da` (code d4c0af5 lineage, finding M7D1-N2-F01). Diagnostic root
`<rt>/diagnostics/m7/M7D1_N3_seed42/` on the GPU host. The scientific run root was only read.
Its `metrics.jsonl` sha256 `5f4aa9a0…632f` equals the value in `M7D1_E08_SEED42_COMPLETION.json`, and every file in
that root keeps its 2026-10-07 completion mtime. The audit-hook write firewall recorded 0 denied writes and 0 writes
outside the diagnostic root in every process. TRAIN access only: VAL/TEST images and metadata = 0. No bank, no seeds
1337/2026, no B1/B2/B3.

Evidence: `M7D1_N3_SEED42_COLLAPSE_COLLECTED.json`, produced by `tools/m7d1_n3_collapse_root_cause.py --scenario collect`.
Derived from it, with no new measurements (`finalize()` in the same tool):
- `M7D1_E08_SEED42_COLLAPSE_ROOT_CAUSE.json` (final machine-readable root cause);
- `M7D1_E08_SEED42_COLLAPSE_GRADIENTS.csv` (per-snapshot / per-loss attribution and Adam-step parts);
- `M7D1_E08_SEED42_COLLAPSE_FORKS.csv` (forks A–F and BDE).

SHA-256 values are in `M7D1_E08_SEED42_COLLAPSE_EVIDENCE_SHA256.txt`. `tests/test_m7d1_n3_collapse_root_cause.py`
re-derives the diagnosis and the three derived files byte for byte, and checks the Adam-step decomposition against
`torch.optim.Adam`.

Classification (final JSON): primary `ADVERSARIAL_IMBALANCE / CURRICULUM_TRANSITION_INSTABILITY`; mechanism
`MASK_SIGMOID_SATURATION through shared G_res trunk`.

## Verdict

`ROOT_CAUSE_LOCALIZED` — primary cause
`LAMBDA_ADV_STAGE2_ACTIVATION__ADVERSARIAL_G_GRADIENT_CLOSES_MASK_VIA_SHARED_TRUNK`.

The stage-2 switch at u5526 turns on the generator adversarial term (lambda_adv 0 → 0.05). It meets a discriminator
that was trained for 5525 updates while lambda_adv = 0, and is therefore extremely confident: D_real ≈ +11.5,
D_fake ≈ −10.3, L_gadv ≈ 10.

From that point the gadv gradient is 68–100 % of the generator gradient projection. Adam (with clipping) normalizes the
step magnitude, so the gadv direction sets almost all of the update. Its effect on the shared G_res trunk drives the
mask logit, channel 12 of `ending`, from about +14 (M ≈ 1, saturated open) to −45 in five updates (u5777–5781):
x_hat = x_t and A = 0.

At logit −45 the sigmoid derivative is ~1e-21. The closed mask is an absorbing state: the budget hinge (pinned at
0.01) has no gradient path back, and Adam momentum plus TV keep pushing the logit down (−67 by u5850).

## 1. Replay parity (u1..u5850) — `PARITY_PASS_TRAINING_STATE`

- 5850/5850 optimizer-group records are compared field by field (volatile fields excluded) against the scientific
  `metrics.jsonl`. Every training field is bitwise equal: losses, grad norms, scales, lr and curriculum.
- The only differing field is `ema_updated`, at u5526..u5850 (325 records): False in the replay, True in the
  scientific run.
  - Cause: a harness defect. The diagnostic step wrapper forwarded only `on_retry`, so `Trainer.activate_ema()`
    assigned `step.ema` on the wrapper and the replay EMA stayed at its epoch-5 initialization.
  - The EMA is write-only in `GeneratorStep`, so G/D/optimizer/scaler states are unaffected. The snapshot EMA digest
    stays constant while the g_res digest changes.
  - The pre-u5526 fork base is exact: its EMA has zero updates by construction.
  - The defect is fixed: the wrapper now forwards `ema`.
- AMP retry events are identical: 1971, 1985, 2587, 3002, 5031, 5234, 5527, 5529, 5758, 5759.
- Representative updates are all bitwise equal on training fields: 1, 1105, 1971, 5525, 5526, 5700, 5750, 5758,
  5759, 5777, 5778, 5779, 5780, 5781, 5800.
- Independent confirmation: fork A (frozen stage-2 values from the pre-u5526 snapshot, EMA live) reproduces the
  scientific records **325/325 bitwise, including `ema_updated`**, with identical retries (5527, 5529, 5758, 5759).
  Fork A's 31 snapshots (dense u5757–u5782) are therefore exact scientific states.

## 2. Mask / M / artifact / residual trajectory (replay = scientific)

| u | mask logit mean | M mean | frac M<1e-2 | sigmoid′ mean | tanh\|raw_HF\| | \|x_hat−x_t\| | A mean | budget |
|---|---|---|---|---|---|---|---|---|
| 1105 | 18.5 | 0.9989 | 0 | 1.0e-3 | 0.45 | – | – | 0 |
| 5525 | 13.7 | 1.0000 | 0 | ~0 | 0.34 | 0.0137 | 0.0161 | 0 |
| 5700 | 12.4 | 0.9999 | 0 | 1e-4 | 0.81 | 0.0614 | 0.0763 | 0 |
| 5775 | 9.0 | 0.9977 | 0 | 2.2e-3 | 0.88 | – | – | 0 |
| 5777 | 4.3 | 0.867 | 0 | 0.083 | 0.90 | 0.0581 | 0.0710 | 0 |
| 5778 | −4.2 | 0.165 | 0.685 | 0.020 | 0.91 | 0.0110 | 0.0134 | 0 |
| 5779 | −10.2 | 0.091 | 0.863 | 0.010 | 0.93 | 0.0062 | 0.0076 | 0.0024 |
| 5780 | −20.0 | 0.029 | 0.951 | 0.0037 | 0.93 | 0.0020 | 0.0024 | 0.0076 |
| 5781 | −45.5 | 0.0000 | 1.000 | ~1e-21 (p50) | 0.94 | 0.0000 | 0.0000 | 0.0100 |
| 5850 | −66.7 | 0.0000 | 1.000 | ~1e-30 (p50) | 0.997 | 0.0000 | 0.0000 | 0.0100 |

The HF residual stays alive (tanh ≈ 0.94–0.99). The collapse is a **mask closure**, not a dead HF branch.

Stage-1 precedent: over u≈1751–2900, with λ_adv = 0, 15–22 % of the mask pixels were already fully closed (logit min
−35 to −105; M mean ≈ 0.78–0.85). The mask recovered by u≈2950.

Regional closure can therefore happen without gadv, and it is recoverable while the open pixels still carry
gradient. The u5781 closure is image-wide (100 % of pixels, logit max −2.9 at u5781). With M = 0 everywhere, every
G gradient path through the residual vanishes (G grad norm ≈ 1e-3 afterwards), and this is what makes it absorbing.
The stage-1 episode is not analyzed further here.

The mask head row barely moves: ‖W_mask‖ 1.30 → 1.27, bias 0.186 → 0.173. The logit change is carried by the trunk
features. In the Adam-step split, the trunk (g_res except the mask row) carries −6.75/−5.83/−12.0/−34.9 of the
linearized logit change at u5777/78/79/80. The mask-head row carries −0.6 to −1.1 and E_art about 0.

## 3. Per-loss gradient attribution

Method:
- Each weighted term's gradient is taken separately, with the trained weight, at the live G scale and then unscaled.
- The real G Adam step (betas 0.5/0.999, no weight decay, clip coefficient of the full gradient, v_new of the full
  gradient) is split exactly into a momentum part plus one linear part per term.
- Each part's effect on the mask logit, M, |x_hat − x_t| and A is measured by fp32 forwards on the update's own group,
  both full-step and first-order (ε = 1/64; additive).

Self-checks:
- The summed per-term pre-clip norm matches the recorded `G_grad_norm` within ≤ 0.21 % on every snapshot.
- The predicted Δθ matches the next snapshot within ≤ 0.47 %.
- The linearized parts sum to the linearized total.

Cumulative linearized Δ(mask logit), fork A = scientific trajectory:

| window | total | gadv | momentum | artcon | bg | lm+id+parse | tv | budget |
|---|---|---|---|---|---|---|---|---|
| 5757–5769 (pre-drift) | −5.3 | **+4.3** | −1.3 | **−8.1** | −0.3 | 0.0 | +0.1 | 0 |
| 5770–5776 (onset) | −11.7 | **−5.5** | −4.8 | −1.2 | −0.2 | −0.1 | +0.1 | 0 |
| 5777–5780 (flip) | −64.3 | **−56.6** | −10.5 | +4.6 | −1.1 | −0.6 | −0.3 | +0.4 |

During the flip, gadv is 82 % of all closing contributions. Momentum is an echo of earlier gradients, and gadv is
≥ 95 % of the gradient projection on most of those updates.

The *direct* per-term dL/dlogit is tiny and sign-mixed for every term (|mean| ≤ 2e-7; tendency `MIXED_OR_NEGLIGIBLE`).
No loss pulls the mask logit closed directly. The closure is a side effect of the gadv-driven trunk update.

OPEN_MASK / CLOSE_MASK per term (first-order step effect):
- **gadv** is CLOSE in the onset and flip windows: −6.1, −6.2, −11, −34 at u5777–80. It is mixed earlier (OPEN in
  5760–5762 and 5765–5767).
- **artcon** is the only term that is mostly OPEN in the flip (+5.2 at u5778), but it closed in the pre-drift window.
- **bg, lm, id, parse** are weak CLOSE: fidelity-to-x_t terms whose optimum is M → 0.
- **budget** is OPEN but only from u5779, once mean A < 0.01: +0.16 and +0.26 against gadv's −11 and −34. A failed
  safeguard.
- **tv** is mixed, then CLOSE after the collapse.
- **spec** and **low** are negligible.

Adversarial path: at u5778 the half-closed generator briefly fools D (D_fake +2.8, L_gadv 0.36). D then re-adapts
(D_fake −3.1 at u5779) and gadv's push resumes until the mask is shut.

Interpretation, not separately tested: the generator's own HF residual is D's "fake" signature, so removing it
(M → 0) is the cheapest descent direction for the non-saturating gadv against an over-confident D.

## 4. AMP retry and grad-clipping roles around u5758 / u5759

- Both are G_OPT overflows at the output conv `g_res.ending.weight`, from gadv-dominated gradients (pre-clip norm 14.4
  / 23.7; gadv share 1.01 / 0.99).
- The retries are atomic: no optimizer step on the overflowed attempt.
- Scaler probe on the exact snapshots: 65536 / 32768 give a non-finite gradient, while 32768 / 16384 (and 16384 /
  8192) give gradients equal within 5.9e-5 / 4.3e-5 relative. The accepted gradient does not depend on the scale.
- The logit effect of those two updates is small (−0.28, −1.65, on a logit of ≈ 21). The flip starts about 15 updates
  later.
- Fork C collapses with retries at different updates (5527 ×2, 5663 ×2), and never at 5758/5759.
- **AMP retry: ruled out as cause.** The overflows are a symptom of the large gadv gradient.
- **Clipping:** active on most stage-2 updates (coefficient 0.04–0.65). It shrinks the gradient norm, but Adam
  normalizes per coordinate, so per-update ‖Δθ‖ stays 0.3–1.7 (stage 1, u5525: 0.86).
  - Clipping neither triggers nor prevents the closure: not causal, no protective effect.
  - A no-clip counterfactual was not run.

## 5. Causal forks from the exact pre-u5526 state (u5526–u5850, curriculum override in the diagnostic process only)

| fork | s_hf | λ_adv | λ_con | λ_spec | first mean-M < 0.01 | min mean M | median / max G norm | AMP retries |
|---|---|---|---|---|---|---|---|---|
| A exact stage 2 | 0.10 | 0.05 | 1.0 | 0.5 | **5781** | 0.0 | 1.18 / 23.7 | 5527, 5529, 5758, 5759 |
| B only s_hf | 0.10 | 0 | 0.5 | 0.25 | none | 0.962 | 0.24 / 0.86 | – |
| C only λ_adv | 0.05 | 0.05 | 0.5 | 0.25 | **5716** | 0.0 | 0.78 / 28.3 | 5527 ×2, 5663 ×2 |
| D only λ_con | 0.05 | 0 | 1.0 | 0.25 | none | 0.946 | 0.39 / 1.43 | – |
| E only λ_spec | 0.05 | 0 | 0.5 | 0.5 | none | 0.991 | 0.21 / 0.72 | – |
| F hold stage 1 | 0.05 | 0 | 0.5 | 0.25 | none | 0.999 | 0.20 / 0.52 | – |
| BDE stage 2 minus λ_adv | 0.10 | 0 | 1.0 | 0.5 | none | 0.948 | 0.48 / 1.79 | – |

- λ_adv alone is **sufficient**: in fork C, the logit goes 10.6 → −19.5 → −428 at u5714–5716, then −2700.
- λ_adv is **necessary within the window**: BDE has every other stage-2 change and does not collapse.
- s_hf, λ_con and λ_spec are not sufficient alone or together.

## 6. Localization

**Primary root cause:** activation of the generator adversarial loss (λ_adv 0 → 0.05 at the stage-2 boundary u5526)
against a pre-saturated discriminator. The gadv gradient dominates the G update direction. Through the shared G_res
trunk it drives the mask logit into negative saturation, where M = sigmoid(logit) has no gradient: an absorbing
identity solution, x_hat = x_t.

**Secondary contributors:**
1. **Stage-1 D over-confidence:** D trains from u1 while λ_adv = 0 (N-05). At switch-on, |D logits| ≈ 10, so the
   non-saturating L_gadv ≈ 10 and its gradient is maximal; gadv is 68 % of the G gradient at u5526 and ≥ 94 % from
   u5700.
2. **Mask parametrization without a recovery path:** M = sigmoid(logit) with no floor.
   - The mask sat saturated-open (logit +12…+21, sigmoid′ ≈ 0) over u≈2950–5776, so no loss held it in a responsive
     range.
   - Once closed, sigmoid′ ≈ 1e-21, and the budget hinge on A = clip(M·…) cannot reopen it.
   - The budget term activates only when A < 0.01, by which time M is already ≤ 0.09.
3. **Optimizer amplification:** Adam momentum (β1 = 0.5) carries the closing direction through and beyond the flip:
   −10.5 during the flip, and −9.1 at u5781 when the current gradient is ≈ 0. Per-coordinate normalization makes
   clipping ineffective as a brake.
4. **Weak closing preference of the fidelity losses** (bg, lm, id, parse; their optimum is x_hat = x_t): cumulative
   ≈ −2.4 over u5757–5780. **artcon** closed in the pre-drift window (−8.1, u5757–5769).

**Ruled out:**
- **s_hf, λ_con, λ_spec curriculum changes:** forks B, D, E and their combination BDE do not collapse.
- **AMP retry / loss scaling:** retries are atomic, gradients are scale-invariant (≤ 6e-5), and the collapse occurs
  with different retry points.
- **Grad clipping as trigger.**
- **EMA:** write-only.
- **Replay/numerics:** bitwise parity.
- **Dead HF branch:** tanh|raw_HF| 0.9+.
- **Direct mask-head drift:** the head row is a minor share of the closing contribution.
- **Data order:** identical groups in every fork.
- **Stage-3 change:** collapse at u5781 is inside stage 2.

## 7. Limitations and disclosures

- **Window:** forks end at u5850, 69 updates after A's collapse and 134 after C's. "No collapse" for B, D, E, F and
  BDE holds only within 325 updates; it is not a proof that these never collapse. One seed (42) only.
- **Attribution scope:**
  - The attribution is first-order per update.
  - Full-step effects are reported too and agree in sign with the first-order ones for every flip-window total.
  - The "why gadv closes the mask" mechanism (D keys on G's HF residual) is an interpretation and was not
    separately tested.
- **Classification rules:** the diagnose rules and the collapse threshold (mean M < 0.01) were written after the
  replay trajectory had been seen.
- **Tool versions:** the tool was modified during the milestone, after the replay. The replay ran with sha `1610d939…`
  (copy kept at `results/tool_at_replay_1610d939.py`). The later edits added:
  - the `ema` forwarding;
  - the Adam split and logit-effect probe;
  - CPU storage of the attribution vectors;
  - the fork-A parity and dense snapshots;
  - collect/diagnose.

  The G/D update computation was not changed by any edit (the `ema` fix only restores EMA updates). Result JSONs record the tool sha read at process end (`c38a6057…`
  for the analyses and forks), not at launch.
  Fork A was launched under the intermediate version (CPU-storage fix), whose fork path is identical to `c38a6057…`.
  The committed tool adds only the CPU-only collect/diagnose/finalize derivation on top of `c38a6057…`.
- **Moved aside, not evidence:**
  - Smoke runs: `smoke_fork_A_5528` (3/3 bitwise parity) and `analyze_smoke_fork_A_5528`.
  - Aborted runs: `analyze_smoke_fork_A_5528.aborted-attempt1-oom` (an fp16 overflow from unit-weight per-term
    backward, then CUDA OOM; fixed by backward at the trained weight and CPU storage) and
    `fork_C.aborted-attempt1-reprioritized` (stopped after 4 records so the fork A attribution could run first;
    re-run from scratch).
- **Disk:** the diagnostic root occupies ≈ 62 GB on the GPU host, mostly snapshots.
- **Owner decision required:** any fix (for example a D policy before the λ_adv switch-on, a λ_adv ramp, a
  mask-logit floor or parametrization, a budget on M) is a protocol change and is not implemented here. The
  scientific E08/seed42 run stays as is. VAL selection, bank and seeds 1337/2026 remain blocked on that decision.
