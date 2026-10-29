# GPAT-TransferBench v1.0 — E04 D12 Procedural Reproducibility Addendum

**Status:** `OWNER_APPROVED`, 2026-10-30. Machine-readable companion:
`configs/amendments/e04_d12_procedural_reproducibility.yaml` (`acceptance_criteria_sha256`
`9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d`).

**Scope:** E04 production requalification only. The ruling authorizes activating `tools/e04_d12_procedural_gate.py`
in a separate implementation commit and one fresh real-data production requalification at qualification seed 60904.
**No E04 scientific seed (42, 1337, 2026) is authorized**, and this ruling creates no E04 scientific authority.

**Owner ruling:** "The owner formally APPROVES E04 D12 Procedural Reproducibility under the exact DRAFT criteria
committed at 57bd1f9. This is approval of the exact previously frozen R1–R10 criteria, not authorization to revise
those criteria. The owner accepts procedural scientific reproducibility without requiring bitwise-identical
floating-point GPU trajectories. The E04 baseline must preserve its architecture, loss functions, optimizer, data,
seeds, training budget, and checkpoint-selection protocol."

**Binding text:** sections 1–6 below are the draft committed at `57bd1f931010f8e0470fb0beac8209000f12e0f7`
(`docs/spec/amendments/GPAT_TransferBench_v1_0_E04_D12_Procedural_Reproducibility_Addendum_DRAFT.md`, sha256
`bdf93ad8…`; companion `configs/amendments/e04_d12_procedural_reproducibility.DRAFT.yaml`, sha256 `0ee2df02…`),
unchanged. The draft, its evidence (`outputs/audit/E04_D12_INVESTIGATION.*`) and the D10 / D11 records are kept
unchanged. Only section 7 restates the approval.

## 1. Why

The E04 production qualification is `E04_PRODUCTION_QUALIFICATION_INCOMPLETE` (41/42). Its only failed gate,
`K.real_data_gradient_repeatability`, requires bitwise-identical real-data gradients. Under the frozen environment
the E04 graph's GPU computation is not bitwise reproducible. D10 (numerical-reproducibility resolution) was not
supported. D11 (CPU ScatterNd placement) was unsupported and left a possible instrumentation confound. D12 resolved
that confound and checked the production path's scientific semantics under a protocol fixed before execution
(d1af9b6). The pre-registered result was `NO_SEMANTIC_DEFECT_FOUND`.

The owner wants a scientifically valid comparison baseline, not bitwise trajectories. D12 therefore proposes to
replace the bitwise gate with criteria on **procedural and scientific integrity**. The proposal keeps every other
gate and changes nothing in the method.

## 2. What does not change

- the frozen config and spec;
- the D1–D9 rulings;
- the M6D3e graph construction (61f1ed58…, 11315 ops) and its requested device placement;
- the optimizer, learning-rate schedule and every loss;
- the TRAIN population and the D4/D7 RNG algorithms;
- the precision environment, session configuration and environment locks;
- the checkpoint rule and the logging contract;
- the 41 other qualification gates.

## 3. Mandatory procedural invariants

1. Frozen source and config. The production identities and the construction equal their pinned values.
2. The approved TRAIN population (12668 rows) and the D4 pair ordering.
3. Deterministic, independent RNG streams (D4 pair order, D7 hard sample), reproduced exactly from the seed.
4. An identical initial state for every probe of the same seed.
5. A correct graph and losses: gradient presence equals differentiable reachability.
6. The correct optimizer sequence:
   - step 1: opt_G on loss_G_step1;
   - step 2: opt_D on loss_D;
   - step 3: opt_G on loss_G_step3;
   - each step is preceded by its BN-update calls.
7. Correct iteration and checkpoint counts. The phase goes 1, 2, 0, and the iteration count advances by 1 after step 3.
8. Finite gradients, losses and state.
9. A valid checkpoint and run lifecycle.
10. Strict TEST isolation.

## 4. Disclosed numerical behavior

- **Non-bitwise GPU computation.**
  - The 32 GPU ScatterNd gradients of the warp GatherNd ops are not bitwise reproducible.
  - In the forward path, the GPU AddN `T_A = B_low + C_low + T` (network.py:100) can vary with unvarying inputs in the same call.
    - Its maximum absolute difference is 4.77e-7, which is one float32 ulp at the tensor's maximum magnitude (6.99).
    - The difference reaches the D/D3 sub-discriminator through the reconstruction.
  - D12 localizes these ops. It does not claim a specific kernel mechanism.
- **Observed scope at fixed state.**
  - Forward tensors: relative L2 ≤ 5.7e-7.
  - Functional step gradients: relative L2 ≤ 1.9e-7 for step 1 and ≤ 1.6e-4 for step 2.
  - Step 3 was bitwise repeatable in every D12 replay.
- **Trajectories.**
  - Parameters, losses and checkpoints of two runs with the same seed may diverge from iteration 0.
  - Identical weights for the same seed are not guaranteed.
  - No bitwise reproducibility is claimed.
- **Not an instrumentation artifact.** The variation is intermittent at the process level. It occurs with no traced call, so it is not caused by tracing.

Numerical divergence never excuses any of the following:

- an invalid or missing operation;
- a missing gradient;
- non-finite state;
- a wrong update, slot or counter;
- a change outside the allowed variables;
- data leakage or VAL/TEST access;
- a corrupted checkpoint;
- a varying fed input;
- a data-movement op varying with identical inputs;
- any tensor or gradient beyond the integrity bound.

## 5. Replacement acceptance criteria (fixed before any requalification evidence)

The criteria are pinned by `acceptance_criteria_sha256` in the draft and implemented by `tools/e04_d12_procedural_gate.py`.

| Criterion | Requirement |
|---|---|
| R1 graph identity | The qualified construction holds, with no op added during a session. The canonical partition-graph set is 094566e1… and the fetched-output hash is fea3730e…. Executed device types are equal across traced processes. |
| R2 deterministic inputs and initial state | At least 3 processes × 3 iterations at seed 60904. The initial state is identical (951dad42…), and so are the per-iteration pair rows and input hashes. |
| R3 gradient connectivity | Gradient presence equals reachability for every loss and step. All 72 step gradients are present. |
| R4 optimizer application | One ApplyAdam per gradient, consuming that gradient. No random op on any step or BN path. |
| R5 state transitions | Only the allowed variables change. Adam stays within 4 ulp + 1e-5·\|Δ\| of the TF 1.15 reference. Beta powers are exact. Counters follow 1, 2, 0 and +1. Restores are bit-exact. Diagnostics never mutate state. |
| R6 finite | State, losses, gradients and forward tensors are finite. |
| R7 forward integrity | At fixed state: no fed input varies, no data-movement op varies with identical inputs, and every varying forward tensor has relative L2 ≤ 1e-3. |
| R8 gradient integrity | The functional-variable step-gradient relative L2 is ≤ 1e-3 for every step replay. |
| R9 retained gates | Exactly the 41 retained gates are present, and all pass. They cover checkpoint lifecycle (H), records (I), TEST isolation (J), data/RNG (D), graph (E/F), finiteness (G), authority (A/B/C) and runtime (K1, K-compat). |
| R10 firewall | No denied access and no runtime write in any probe process. |

The bound 1e-3 is the D10 C4 threshold value, fixed in the D12 protocol before any D12 evidence. It guards against
corruption and wrong computation. It is neither a reproducibility nor a performance threshold. **No criterion depends
on model performance.**

The following are recorded diagnostics only and never gate:

- bitwise gradient repeatability (the superseded K computation);
- cross-process state and loss distinctness;
- forward variation counts.

## 6. The superseded gate

`K.real_data_gradient_repeatability` stays **FAIL** in the original 41/42 record, which is not modified. On approval,
it is superseded for **future** qualification records only. It is still computed and recorded, never retroactively
passed and never counted as passed.

## 7. Approval and requalification

1. Approved: `configs/amendments/e04_d12_procedural_reproducibility.yaml`, status `OWNER_APPROVED`, `approved: true`,
   `acceptance_criteria_sha256` unchanged, committed on its own.
2. A separate commit wires the gate into the qualification evidence tool. The gate runs only when this approved
   amendment pins its criteria hash.
3. A new full requalification is executed:
   - all original cases;
   - 3 processes of `tools/e04_d12_production_replay.py` (3 iterations);
   - 2 processes of `tools/e04_d12_confound_probe.py --factor d10_trace2`;
   - unchanged graph and placement, seed 60904.
   It writes a new record. The original 41/42 record is not modified.
4. E04 is production-qualified under D12 only if every retained gate and every R criterion passes in the new record.
   Otherwise E04 stays `E04_PRODUCTION_QUALIFICATION_INCOMPLETE`.
5. Scientific seeds 42, 1337 and 2026 need a separate owner launch decision.
