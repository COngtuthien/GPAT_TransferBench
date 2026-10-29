# E04 D10 — Numerical Gradient Reproducibility Investigation: Decision Report

**Status: `E04_PRODUCTION_QUALIFICATION_INCOMPLETE` (unchanged).**
**Decision under the pre-registered protocol: `STOP_AND_REPORT`. D10 is not recorded and no requalification was run.**

There is no E04 scientific authority. No scientific seed, VAL, TEST, E01/E02, GPAT, synthetic bank or downstream evaluation was run. No scientific source, graph, optimizer, geometry, data, seed, environment or device placement was changed. The original 41/42 qualification record is unchanged (`outputs/audit/E04_PRODUCTION_QUALIFICATION.json`, sha256 `d158a1ca…8843`).

Machine-readable record: `E04_D10_INVESTIGATION.json`. Full analysis output: `E04_D10_ANALYSIS.json`.

## 1. Origin

Production qualification 1d02625 failed one gate, `K.real_data_gradient_repeatability` (41/42). With identical iteration-0 state, inputs and losses, the L_G per-loss gradient was not bit-repeatable either within or across processes. Every other per-loss gradient was bit-repeatable.

## 2. Protocol (fixed before execution, commit a6bb459)

- **Protocol file:** `E04_D10_INVESTIGATION_PROTOCOL.json`. Tools: `tools/e04_d10_numerical_probe.py` and `tools/e04_d10_analysis.py`.
- **Seed:** qualification seed 60904. The owner text names 314159, which is the M6D3e reference-construction seed; this is noted in the protocol.
- **Processes:** three independent clean processes P1, P2 and P3, run sequentially from the committed GPU checkout a6bb459 (clean worktree). Each ran 1000 iterations of the frozen production update sequence on real TRAIN data.
- **Gradient probes:** at 12 iterations, with 3 repeats each.
- **Minibatch reference:** the gradient at the same state on the next batch.
- **Snapshots:** 18 snapshot points.
- **Operation level:** a FULL_TRACE device record and a first-divergence localization (7083 tensors, 3 repeats).
- **Criteria:** C1–C6. The rule is D10_SUPPORTED only if all six pass, and STOP_AND_REPORT if any of C1–C3 fails. No post-hoc changes are allowed.

All three processes finished (rc=0, 1000 iterations each, about 1.58 s per iteration) with no firewall denials and no runtime writes. The construction was 61f1ed58… with 11315 ops, and the initial state sha256 was 951dad42… in every process.

## 3. Results by criterion

| Criterion | Result | Key evidence |
|---|---|---|
| C1 contract identity | **FAIL** | Per-iteration checks all equal: pair rows and passes, hard draws, input sha256, counters, call log, Adam beta powers, look-ahead batches; initial state, construction and diagnostics-never-change-state also equal. **The executed graph-op signature set at iteration 0 is not equal:** P1 has 3935 distinct signatures; P2 and P3 have 3932 (identical sets, and identical to the committed smoke run). |
| C2 finite | PASS | Every loss, gradient and state value is finite. |
| C3 localization | PASS | At iteration 0 all losses and all non-L_G gradients are bit-identical across processes. In-process repeats differ only in L_G. The 32 origins are all ScatterNd ops on GPU:0, with 0 undetermined. Forward GatherNd and ResizeBilinear run on CPU. |
| C4 fixed-state magnitude | PASS | Numerical repeat ‖Δg‖ divided by minibatch ‖Δg‖ has a maximum of 7.5e-7 (median 3.2e-7) against a threshold of 1e-3, over 72 points. |
| C5 trajectory parameters | **FAIL** | 63 of 102 points fail. The first failure is at snapshot 50 (G: 1.09 updates). At t=1000 the deviation is 8.5–12.1 single-iteration updates for G and 16.7–44.9 for D. |
| C6 trajectory losses | **FAIL** | 31 of 36 points fail. Over the final window, median \|ΔL\| divided by the median iteration-to-iteration change ranges from 0.0085 to 0.36, against a threshold of 0.01. The passing points are the three identically-zero P0 priors and P2–P3 L_P / P_positive. |

### C1 detail

The executed op-type/device counts differ only for the following:

| Op type and device | P1 | P2 | P3 |
|---|---|---|---|
| Const on GPU | 107 | 102 | 97 |
| Identity on GPU | 740 | 740 | 741 |
| RealDiv on GPU | 48 | 48 | 49 |
| Reshape on GPU | 352 | 352 | 354 |
| Tile on GPU | 45 | 45 | 47 |

Executed node counts were 4082, 4077 and 4078. These op types are graph-rewrite products (constant folding and CSE representatives); no compute op differs. The iteration-0 numerics outside L_G were bit-identical regardless. The protocol still classifies a differing executed op set as a reproducibility-contract violation. The decision follows that classification, and no criterion is reinterpreted.

## 4. Operation-level findings

- **Variation origins.** The origins are 32 ScatterNd ops. Each is the gradient of one of the trace-warp `GatherNd` lookups for warps B, C, P and T: four per warp, in each of the two L_G gradient graphs `gradients_1` and `gradients_7`. All 32 appear in all three processes and all execute on GPU:0. An op is counted as an origin when its outputs differ across repeats while all of its inputs are bit-identical.
- **ResizeBilinearGrad.** All 40 executed ResizeBilinearGrad ops run on GPU. Their outputs vary only downstream of the ScatterNd origins; none is an origin. The original record's list of "known nondeterministic GPU ops" (ResizeBilinearGrad and ScatterNd) is therefore refined: within the observable closure, only ScatterNd originates variation.
- **Downstream propagation.** Variation spreads to 431–432 tensors, produced by AddN, Conv2D, Conv2DBackprop*, FusedBatchNormGradV3, LeakyReluGrad, Reshape, Slice, StridedSliceGrad, Sum and other op types. Every one of these traces back to the ScatterNd origins.
- **Placement gap (disclosed).** `warp()` and `resize()` place their forward ops on `/cpu:0`, and the code comments give deterministic gradients as the reason. `tf.gradients` does not colocate gradient ops, so the gradients (ScatterNd and ResizeBilinearGrad) are requested on and executed on `/device:GPU:0`. Placement was not changed.
- **Kernel cause.** No precise GPU kernel cause is claimed. The evidence shows only that ScatterNd outputs on GPU are not bit-repeatable given bit-identical inputs.

## 5. Error distributions (descriptive)

**Fixed-state repeats** (3000 comparisons per gradient set):

| Measure | L_G | Step 1 |
|---|---|---|
| Concatenated relative L2, median | 7.9e-8 | 7.8e-8 |
| Concatenated relative L2, p99 | 2.8e-7 | 1.5e-7 |
| Concatenated relative L2, max | 4.0e-5 | 2.1e-5 |
| Concatenated cosine, minimum | 0.99999999923 | 0.99999999979 |

- The per-variable tails are large: relative L2 p90 is about 0.84, the maximum scale-aware max-abs is about 4.6, and the maximum ULP is about 2e9. These tails come from near-zero-gradient variables, chiefly the BN-cancelled biases (see §7).

**Cross-process gradient on each process's own trajectory** (relative L2):

| Iteration | L_G | Step 1 |
|---|---|---|
| 0 | 3.4e-7 | 1.8e-7 |
| 1 | 2.2e-3 | 7.4e-4 |
| 100 | 1.4e-2 | 6.1e-3 |
| 500 | 0.33 | 0.18 |
| 999 | 0.41 | 0.36 |

## 6. Parameter, Adam and BN deviation (normalized ‖Δ‖/‖ref‖)

| Group | t=1 | t=100 | t=1000 (P1–P2) |
|---|---|---|---|
| G functional | 5e-5 | 9.7e-4 | 5.1e-3 |
| D functional | 1e-5 | 2.7e-4 | 6.5e-3 |
| Adam m (G) | 4e-5 | 1.3e-2 | 0.106 |
| Adam m (D) | 1.6e-4 | 1.6e-2 | 0.360 |
| Adam v (G) | 1e-5 | 1.8e-3 | 6.7e-3 |
| Adam v (D) | 1.6e-4 | 4.9e-3 | 8.1e-2 |
| BN moving mean | 1.8e-4 | 5.0e-3 | 5.3e-2 |
| BN moving variance | ~0 | 6.4e-3 | 4.6e-2 |

- The Adam beta-power accumulators are bit-identical in every process at every iteration, so the optimizer application count is identical.
- The log-log growth slope of parameter deviation for t ≥ 10 is about 0.54–0.61 for G and 0.79–1.01 for D. The deviation grows steadily, with no sign of saturating within 1000 iterations.

## 7. Disclosed tooling defect (does not affect the decision)

**The defect.** The structural BN-cancelled-bias detector (`bn_cancelled_biases` in the probe) returned an empty set in P1–P3 and in the committed smoke run. As a result, the 15 G and 16 D convolution biases that feed training-mode BN were counted inside the functional groups and the per-variable distributions. The pre-registration had required them to be reported separately.

**Sensitivity check.** A descriptive check (`tools/e04_d10_bias_sensitivity.py`, run read-only) excluded the BN-sibling biases:

- Early deviation is almost entirely bias. At t=1 the G deviation is 0.00205 in total and 0.00011 without the biases.
- At t=1000, excluding biases still leaves a deviation of 7.0–10.9 updates for G and 15.1–44.4 for D.
- C6 is computed from losses only and is unaffected.
- The C1 failure alone already decides STOP_AND_REPORT.

The criteria were evaluated exactly as executed; nothing was recomputed into a decision.

## 8. TEST isolation and authority checks

- **atime** (stat-only, from e04_d10_atime_pre 12:54:52Z to post 14:20:38Z, 20627 files):

  | Split | Files | atime changed | mtime changed |
  |---|---|---|---|
  | TEST | 3027 | 0 | 0 |
  | VAL | 3121 | 0 | 0 |
  | TRAIN | 14467 | 6474 | 0 |

  All 12 manifests are unchanged (mtime and atime).
- **Run roots.** `runs/m6` contains no E04 run root.
- **Code and authority files.** The GPU checkout is at a6bb459 with a clean worktree. Scientific source, configs, manifests, D1–D9 rulings, the pre-pass manifest and the feed-pack manifest are unchanged at this commit.
- **Raw outputs.** 1345 files, 2.0 GB, stored read-only at `<runtime>/qualification/e04_production/d10_probe_a6bb459/`. Their SHA256SUMS are copied to `E04_D10_RAW_SHA256SUMS.txt`. Smoke outputs are kept under `smoke_not_interpreted/` and were never interpreted.

## 9. D10 decision and rationale

D10 is **not supported**, under the decision rule fixed before execution:

- **C1 failed.** The executed graph-operation set is not identical across independently started processes. The protocol treats this as a reproducibility-contract violation, so the outcome is STOP_AND_REPORT.
- **C5 and C6 also failed, independently of C1.** The fixed-state numerical noise is tiny (about 1e-7 relative, C4). Even so, over 1000 iterations the three trajectories separated by about 10–45 single-iteration updates in parameters. In 31 of 36 comparisons, their final-window loss differences were 1.07–36 % of the iteration-to-iteration sampling change.

The bounded evidence therefore does not show that the L_G GPU-gradient variation is trajectory-negligible. Per §5 of the directive, this is a decision report: no D10 amendment was written, no amended gate was introduced, no requalification was run, and no gate was weakened.

## 10. Remaining blockers

1. `K.real_data_gradient_repeatability` is still failed. E04 production qualification remains 41/42 and **INCOMPLETE**.
2. D10 is not supported by the pre-registered bounded evidence.
3. An owner decision is required on how E04 proceeds. This report does not create, approve or imply any amended reproducibility contract.
