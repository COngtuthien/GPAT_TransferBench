# E04 D12: procedural-reproducibility semantic investigation (result)

**D12 status: `E04_D12_READY_FOR_OWNER_APPROVAL`.** E04 remains `E04_PRODUCTION_QUALIFICATION_INCOMPLETE` (41/42).
D12 is not approved, no gate is active, there is no E04 scientific authority, and no scientific seed was launched.

The pre-registered decision was `NO_SEMANTIC_DEFECT_FOUND`. The protocol `E04_D12_INVESTIGATION_PROTOCOL.json` (64b1c858…)
was fixed at d1af9b6 before execution. Every criterion passed:

- A_authority;
- completion;
- B_restore_exact;
- B_production_semantics;
- C1, C2 and C3 integrity;
- D1_confound_reproduced;
- E_graph_and_placement.

## 1. Git and frozen authority

- **HEAD before D12:** d276aae6756426125e45afbe1202a55fd9b48832 (parent ba625b8). It is identical on the laptop, origin and the GPU checkout. D12 tools are at d1af9b6. The GPU checkout was clean at d1af9b6 for the whole run.
- **No frozen-input changes:** `git diff 1d02625..d1af9b6` is empty over methods, configs, environments, manifests, frozen_config_snapshot, docs/spec and tools/run_e04.py.
- **Contract and identities:** `production_io.verify_contract` passed on the laptop (`--no-tf`) and on the GPU. The identities sha256 is cc77c431…, which equals the D11 record and every D12 process.
- **Data:**
  - pre-pass 274a4b0a…: 12668 rows, all TRAIN;
  - feed pack db1908d5…: bound to the pre-pass.
- **Graph and state:**
  - production construction 61f1ed58… / 11315 ops, equal to the M6D3e reference;
  - initial state 951dad42… in all 19 processes.
- **Authority record:** `E04_D12_AUTHORITY.json`.

## 2. D10 / D11 and original evidence preserved

- **Records unchanged:**
  - `E04_PRODUCTION_QUALIFICATION.json` d158a1ca… (41/42; K.real_data_gradient_repeatability FAIL);
  - D10 record 2f0c0c00…;
  - D11 record 9305fb82….
- **Raw archives re-verified against the committed SHA256SUMS:**
  - evidence_raw_6aba90d;
  - d10_probe_a6bb459 (1346 files);
  - d11_probe_ba625b8.
- **Unapproved D10 gate draft:** the uncommitted laptop worktree change equals `refs/drafts/e04-d10-gate-unapproved` (09cc14b). D12 did not commit, activate or use it.

## 3. D3 instrumentation investigation

Stage C ran 4 factors × 4 clean processes, interleaved, on the unchanged graph and placement.

| factor | processes with forward variation |
|---|---|
| plain (no traced call, requested devices) | 4/4 |
| d10 (unchanged D10 probe) | 4/4 |
| d10_trace2 (D10 + D11's second FULL_TRACE) | 4/4 |
| d11_wrapper (all D11 hooks) | 3/4 |

On the production replay path (stage P), forward variation occurred in 6 of 9 iterations. **The D/D3 forward variation
occurs on the ordinary uninstrumented production path. It is not caused by tracing or the D11 instrumentation.**
D11's "absent in D10" came from the 3 original D10 processes. The unchanged D10 probe now shows the variation in 4/4.
The variation is therefore intermittent at the process level.

- **Origin.** The only forward op observed to vary while its inputs, fetched in the same call, did not vary is `T_A:0`. This is the GPU AddN `T_A = B_low + C_low + T` (network.py:100).
  - Its maximum absolute difference is 4.77e-7, one float32 ulp at the tensor's maximum magnitude (6.99). Larger ulp distances occur only where the sum nearly cancels.
  - It appears as a root in plain 4/4 and on the production path. In traced factors the variation appears downstream without a localizable root.
  - No kernel mechanism is claimed.
- **Scope.**
  - Reconstruction-side tensors: sub_3, mul_4, add, strided_slice_7.
  - The D3 input concat_2 → D/D3 ResizeBilinear → D/D3 conv1–conv4 (Conv2D, BiasAdd, FusedBatchNormV3, LeakyRelu) → D/D3/probability.
  - Loss-side Sub, Square, Sum and StridedSlice tensors.
  - D/D3 is the only sub-network affected.
- **Magnitude.** Maximum forward relative L2 is 5.7e-7 in every process. There are 0 varying fed inputs, 0 varying data-movement roots and 0 integrity-bound violations.

## 4. Graph and optimizer invariants (stage P: 3 processes × 3 iterations, 54 semantic replays)

- **Gradient connectivity:** presence equals differentiable reachability for every loss and step.
  - L_D: 72;
  - L_G, L_P, L_R, L_S: 58;
  - L_H, L_depth: 70;
  - step 1–3: 72 of 72.
- **Optimizer application:** 72 ApplyAdam per step, each targeting its variable and consuming its step gradient. No random op is on any step or BN path.
- **Per call:** only the allowed variables change.
  - Adam stays within tolerance: worst error/tolerance is var 0.30, m 0.125, v 0.125.
  - Beta powers are exact.
  - Counters go 1, 2, 0, and the iteration advances by 1.
  - All values are finite. Every restore is bit-exact. Diagnostics never mutated state. 11828 graph ops before and after the session.
- **Functional step-gradient relative L2 across replays from the same state:**
  - step 1 ≤ 1.9e-7;
  - step 2 ≤ 1.6e-4;
  - step 3 = 0 (bitwise repeatable in every replay).
  - Bound: 1e-3.
- **Step-replay variation:**
  - step 1 varies in 118–157 variables;
  - step 2 in 47–55 variables, all in D/D3;
  - step 3 in none.
  - Losses were identical across replays except one step 2 case.
- **Cross-process:**
  - pair rows, input hashes and the initial state are identical;
  - post-iteration states differ from iteration 0;
  - loss relative difference is ≤ 4.4e-6 by iteration 2.
- **Placement:** executed device types are consistent across all traced processes. The canonical partition set (094566e1…) and retval hash (fea3730e…) of all 8 traced processes equal the D11 original arm, so there is no runtime-rewrite difference.

**No scientific-semantic defect was found.**

## 5. Proposed D12 acceptance rules (draft, inactive)

- **Draft:** `configs/amendments/e04_d12_procedural_reproducibility.DRAFT.yaml`, status `DRAFT_FOR_OWNER_FINAL_APPROVAL`, approved false.
- **Addendum:** `docs/spec/amendments/GPAT_TransferBench_v1_0_E04_D12_Procedural_Reproducibility_Addendum_DRAFT.md`.
- **Gate:** `tools/e04_d12_procedural_gate.py`, inactive and imported by nothing. It refuses without an OWNER_APPROVED amendment pinning acceptance_criteria_sha256 9f1a2471….
- **Criteria R1–R10:**
  - R1 graph identity and executed placement;
  - R2 deterministic inputs and initial state;
  - R3 gradient connectivity;
  - R4 optimizer application;
  - R5 state transitions (scope, Adam, beta powers, counters, exact restores);
  - R6 finiteness;
  - R7 forward integrity;
  - R8 functional-gradient integrity (1e-3, pre-registered);
  - R9 the 41 retained gates;
  - R10 firewall.
- **No performance criterion.** Bitwise repeatability stays a diagnostic.
- **The K gate** stays FAIL in the original record. It is superseded only for future records after approval, never retroactively passed and never counted as passed.
- **Tool-validity dry run** (not qualification evidence): the draft gate passes on the D12 stage-P records, the d10_trace2 traces and the original record's gate list. It reports K as `recorded_pass: false, counted_as_passed: false`.

## 6. Tests and evidence hashes

- `E04_D12_ANALYSIS.json`: 749cd18f…
- `E04_D12_AUTHORITY.json`: b76a2e19…
- `E04_D12_RAW_SHA256SUMS.txt`: c820968a…
- **Raw outputs:** `<rt>/qualification/e04_production/d12_probe_d1af9b6`, read-only, 223 files, including `smoke_not_interpreted/` and the atime snapshots.
- **TEST isolation:** atime pre = post 9b25acbc… over 20627 files. TEST (3027) and VAL (3121) show 0 atime or mtime changes, the manifests are unchanged, and there is no E04 run root.
- Tests and file hashes: `E04_D12_INVESTIGATION.json` (`tests`) and `E04_D12_INVESTIGATION_SHA256.txt`.

## 7. Remaining blockers

1. Owner approval of the D12 draft. Nothing is activated.
2. After approval:
   - a separate commit wiring the gate;
   - a new full requalification execution and record.
   Until then E04 stays `E04_PRODUCTION_QUALIFICATION_INCOMPLETE` (41/42).
3. An owner decision on the unapproved D10 gate draft, which D12 does not use.
4. A separate owner launch decision for scientific seeds 42, 1337 and 2026, after a passing requalification.

## 8. Ready for owner approval

Yes. The investigation found no semantic defect. The draft, the inactive gate and the regression tests are committed. The acceptance criteria are fixed before any requalification evidence.
