# E04 D11 — Targeted CPU Gradient Placement Investigation: Decision Report

**D11 status: `E04_D11_UNSUPPORTED`.**
**E04 status: `E04_PRODUCTION_QUALIFICATION_INCOMPLETE` (unchanged, 41/42).**

The pre-registered rule (protocol ba625b8) is "any S criterion fails → UNSUPPORTED, no L_bounded run". Two criteria failed:
- S4 forward identity;
- S5 CPU bitwise reproducibility.

So the 1000-iteration bounded stage was not run, and no D11 amendment is proposed. Nothing was approved, requalified or relaxed.

There is no E04 scientific authority. None of the following was run: a scientific seed (42, 1337 or 2026), VAL, TEST, E01, E02, GPAT, a synthetic bank or a downstream evaluation. No scientific source, config, graph source, optimizer, loss, data pack, seed or environment was changed. The experimental placement existed only inside the probe processes.

Machine-readable record: `E04_D11_INVESTIGATION.json`. Analyses: `E04_D11_SHORT_ANALYSIS.json` and `E04_D11_TRACE_ANALYSIS.json`.

## 1. Repository state

- Before D11, the laptop, origin, GitHub (`git ls-remote`) and the GPU checkout were all at `de46f66`.
- The D11 protocol and tools were committed and pushed at `ba625b8` before execution. The GPU checkout was fast-forwarded to `ba625b8` and was clean throughout. The probe recorded tool hashes equal to the committed files.
- The original failed qualification record is unchanged: `E04_PRODUCTION_QUALIFICATION.json`, sha256 `d158a1ca…8843`, 41/42, failing `K.real_data_gradient_repeatability`.
- The D10 record and analysis are unchanged. The D10 raw outputs (`d10_probe_a6bb459`, read-only) re-verify against their SHA256SUMS.
- **D10 gate draft:** the uncommitted draft (`tools/e04_production_qualification_evidence.py` modified, `tests/test_e04_d10_gate.py` untracked) is still in the laptop worktree. It is neither committed nor activated. A non-destructive local snapshot exists at `refs/drafts/e04-d10-gate-unapproved`, which is not pushed.

## 2. Original graph and gradient placement

The nondeterministic path consists of **32 ScatterNd ops**. Their names are `gradients_1/warp_{B,C,P,T}/GatherNd{,_1,_2,_3}_grad/ScatterNd` (the L_G per-loss gradient) and the same 16 names under `gradients_7` (the step1 G-update gradient).

- **Forward side:** each ScatterNd is the `_GatherNdGrad` of a forward GatherNd in `warp()`, which runs on `/cpu:0`.
- **Inputs:**
  - indices: `warp_*/stack_k:0`, a Pack on CPU, int32 `[4,256,256,3]`;
  - updates: float32 `[4,256,256,1]` (`[4,256,256,3]` for warp_C and warp_T), from GPU;
  - shape: a Const int32 `[4]` on GPU.
- **Output:** float32, of the same shape as the params.
- **Device:** `tf.gradients` is called inside the Controller `/gpu:0` scope without `colocate_gradients_with_ops`. The gradient ScatterNd therefore inherits `/gpu:0`, and it executed on GPU:0 in every original-arm trace.
- **Downstream:** 435 ops depend on the targets, including:
  - 58 ApplyAdam;
  - 30 Conv2DBackpropFilter and 22 Conv2DBackpropInput;
  - 28 FusedBatchNormGradV3;
  - 8 ResizeBilinearGrad.
- **Probe sets that depend on the targets:** L_G and step1 only.
- **Colocation:** TF1 gradient aggregation creates `gradients_1/AddN_5` and `gradients_7/AddN_5` with `_class=[loc:@…/warp_T/GatherNd_3_grad/ScatterNd]`.
- **ResizeBilinearGrad:** all 40 ResizeBilinearGrad ops stay on GPU in both arms; they were not moved.

## 3. Signature-count difference (3935 vs 3932) — descriptive

This analysis is descriptive only and never a criterion. Second traces with partition graphs were captured in 7 clean processes per arm (S + T stages). The executed distinct-signature count varies between 3932 and 3936 in both arms, and in each process the two traces agree with each other. After canonical hashing of the partition graphs (node names excluded, values, types, devices and inputs kept, transfer edges resolved):

- within each arm, the **canonical node set is identical in all 7 processes**;
- the **canonical hash of every one of the 660 fetched outputs is identical in all 7 processes**;
- only the **multiplicity** of 11 canonical node classes (Const, Identity, RealDiv, Reshape and Tile, all on GPU) differs, and each class is present at least once in every process.

So the difference comes from how many structurally identical duplicate nodes survive the classic graph optimizer. It is a runtime graph-rewrite difference. It is not a difference in the graph as built, and not a difference in the mathematical computation. This is consistent with nondeterministic CSE or constant folding, but that mechanism was not verified. It also does not imply equal numerical results.

## 4. CPU placement implementation and scope

The `cpu_scatternd` arm makes two changes:
1. `Operation._set_device('/device:CPU:0')` on the 32 targets;
2. removal of the `loc:@<target>` reference from the two AddN ops. Without this, soft placement would keep 2 targets on GPU. The AddN ops keep their GPU device.

Both changes are applied between graph construction and session creation, inside the probe only.

The GraphDef differs in exactly 34 nodes:
- 32 nodes where only the device changed;
- 2 nodes where only `_class` changed.

No node was added or removed. The construction before the override was the qualified `61f1ed58…` (11315 ops).

**S1 passed in every process.** In both traces, all 32 targets executed on CPU, and no non-target op changed device type relative to the control. The CPU arm adds 80 runtime transfer nodes (1460→1540).

## 5. Short-probe results (3+3 clean processes, 25 iterations, interleaved O1 C1 O2 C2 O3 C3)

| Criterion | Result | Evidence |
|---|---|---|
| S1 placement scope | PASS | Scope exactly as pre-registered. CPU execution enforced for the backward ops. |
| S2 gradients present | PASS | The presence pattern is identical in all 6 processes. |
| S3 contract identity | PASS | Initial state, pairs, hard draws, inputs, counters, call logs and beta powers are identical. Firewall clean. |
| S4 forward identity | **FAIL** | Losses at t=0 and target-independent sets L_H, L_P, L_R, L_S, L_depth and step3 are equal across arms. **L_D and step2 are not.** |
| S5 CPU bitwise reproducibility | **FAIL** | Detailed below the table. |
| S6 finite | PASS | |
| S7 deviation from control | PASS | ‖g_cpu − g_orig‖ / minibatch ‖Δg‖ ranges from 4.6e-7 to 8.6e-7 (threshold 1e-3; 18 points). |
| S8 completion | PASS | All 14 processes have rc=0 and complete records. |

**S5 detail (CPU arm):**
- **Localization:** 0 origins in every CPU process, but 138, 146 and 138 tensors still vary.
- **In-process repeats:** L_G is not repeatable in all 3 processes (58 G variables, intermittent). L_D and step2 are not repeatable in P1 and P3 (17 variables, all in sub-discriminator `D/D3`). step1 is repeatable throughout.
- **Across processes:** state sha256 has P1 = P2 ≠ P3 for iterations 0–4, and all three differ from iteration 5.
- **Original arm, for comparison:** state differs across all three processes from iteration 0.

**What the CPU placement did.** It removed the 32 ScatterNd ops as localizable origins (32 in every original process, 0 in every CPU process) and made step1 repeatable. It did **not** make the computation bit-reproducible.

**Residual variation.** The remaining variation has no localizable origin: a tensor varies when fetched in one localization chunk but not in its producer's chunk.

**Forward ops vary too.** These ops vary at fixed state in all 14 D11 processes of **both** arms:
- `D/D3` input ConcatV2 and ResizeBilinear;
- FusedBatchNormV3, LeakyRelu and Sigmoid.

L_D also varied in original P1. None of this occurred in any of the 3 D10 processes, which used the same graph, seed, data, environment and base probe.

**Confound.** The D11-only additions are:
- the second FULL_TRACE with `output_partition_graphs`;
- the placement recording;
- the corrected bias groups, which do not change computation.

Whether any of these, or the environment, causes the forward variation is **not established**. No precise kernel cause is claimed.

## 6. Bounded results

Not run, as required by the pre-registered rule (S4 and S5 failed). C1–C6 were therefore not evaluated for D11. The corrected bias-sensitivity grouping (`tools/e04_d11_bias_detector.py`: 31 BN-cancelled biases, 15 G and 16 D, equal to the BN-sibling set) ran in every process. It affects variable groups only, and the D10 results were not rewritten.

## 7. Numerical differences against the original

- **Gradient deviation at t=0:** for L_G and step1, the CPU arm's deviation from the unchanged control has relative L2 between 1.8e-7 and 4.8e-7. This is the same order as the control's own GPU repeat differences.
- **State deviation:** after 1–2 iterations, CPU P1 against original P1 has normalized deviations of about 3e-6 to 9e-6 for G/D functional, and 1e-5 to 3e-4 for the Adam moments. These are comparable to original P1 against original P2.

The full table is in the JSON.

## 8. Runtime and memory

These are descriptive figures from 25-iteration probes that include diagnostics.

| Metric | Original | CPU ScatterNd | Ratio |
|---|---|---|---|
| Median seconds per production step | 1.225 | 1.275 | 1.04 |
| Peak GPU allocator | 3.32 GB | 5.70 GB | 1.72 |
| Peak RSS (ru_maxrss) | 10.60–10.67 M KiB | 10.72–10.75 M KiB | about 1.01 |

## 9. Fidelity

| Property | Assessment |
|---|---|
| Mathematical equivalence | Yes, by construction: the same op, inputs and attributes run on another device, and no op is added or removed. |
| Identical graph topology | GraphDef: yes (only 32 device fields and 2 `_class` attributes differ). Runtime partitions: no (CPU partition, plus 160 partition transfer nodes). |
| Identical device placement | No. |
| Numerical equivalence within measured precision | At t=0, within the C4 threshold. Trajectory level not measured. |
| Bitwise-identical trajectories | No (S5). Not achieved by the original arm either. |

## 10. Tests and evidence

- **Test isolation:** the atime pre/post snapshots (20627 files) are byte-identical. TEST 0/3027 and VAL 0/3121 were accessed, no manifest was modified, and there is no E04 run root.
- **Raw outputs:** `<rt>/qualification/e04_production/d11_probe_ba625b8`, read-only, 265 files. The SHA256SUMS copy is `E04_D11_RAW_SHA256SUMS.txt`. It includes the smoke runs (`smoke_not_interpreted/`) and the descriptive post-hoc scripts.
- **Hashes:** `E04_D11_INVESTIGATION_SHA256.txt`.
- **Pinning test:** `tests/test_e04_d11_investigation.py`.

## 11. Suitability for owner approval

**Not suitable.** D11 is `E04_D11_UNSUPPORTED` under the rule fixed before execution. CPU placement of the ScatterNd gradients is a correct and scoped intervention that removes the identified origins. It is not sufficient for bit-reproducible gradients or trajectories under the D11 probe. In addition, part of the failure (the forward variation in D/D3) also appears in the unchanged control under the same probe, so its origin is not settled.

## 12. Remaining blockers

1. `K.real_data_gradient_repeatability` remains failed. E04 stays `E04_PRODUCTION_QUALIFICATION_INCOMPLETE` (41/42).
2. D10 is not supported (de46f66), and D11 is unsupported (this record).
3. Residual non-bit-reproducibility after CPU ScatterNd placement: un-localized variation in L_G (G) and in L_D / step2 (D/D3), and cross-process state divergence from iteration 0.
4. The unexplained forward-op variation present in both D11 arms and absent from D10 (an instrumentation or environment confound).
5. Any further remedy needs a new owner decision and a new pre-registered protocol. Examples: ResizeBilinearGrad placement, isolating the D11 instrumentation effect, or a non-bitwise reproducibility criterion.
