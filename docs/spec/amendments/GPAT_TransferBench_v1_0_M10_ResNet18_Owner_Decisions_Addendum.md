# GPAT-TransferBench v1.0: M10 ResNet-18 downstream owner decisions O1–O7

Date: 2026-11-26. Machine-readable record: `configs/amendments/m10_resnet18_owner_decisions.yaml` (`OWNER_APPROVED`).
Precision pin: `configs/amendments/m10_resnet18_science_precision.yaml` (`OWNER_APPROVED`).

The owner approved the seven decisions that the M10 launch-authority DRAFT
(`configs/amendments/m10_resnet18_science_launch_authority.DRAFT.yaml`, commit `7d67a7f`) listed. The ruling is
additive. It changes none of these, which are kept byte-for-byte:

- the frozen spec §14–§16;
- `configs/frozen/downstream_resnet18.yaml` (`f117c573…`);
- the approved evaluator contract `configs/amendments/m8_downstream_evaluator_contract.yaml` (`67e4b62d…`).

| Decision | Approved option |
|---|---|
| O1 science precision | `A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC` |
| O2 threshold tie | `A_SMALLER_THRESHOLD` |
| O3 retention and run root | `A_SELECTED_ONLY_HDD_NEW_ROOT` |
| O4 environment | `A_APPROVE_UNCHANGED` |
| O5 order | `A_SEED_MAJOR` |
| O6 primary view | `A_RAW_MATCHED_PRIMARY` |
| O7 track labels | `A_CONFIRM` |

**O1.** CUDA autocast `float16` with `torch.amp.GradScaler('cuda')`: init scale 65536, growth 2, backoff 0.5, interval
2000. Each step runs `scaler.scale(loss).backward()`, then `scaler.unscale_(opt)`, then `clip_grad_norm_(…, 1.0)`, then
`scaler.step(opt)`, then `scaler.update()`. These settings are fixed:

- deterministic algorithms on;
- cuDNN deterministic on and benchmark off;
- TF32 off;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

BF16 or FP32 is never substituted, and determinism is never disabled. Skipped GradScaler steps are recorded.

The implementation rule below is disclosed as the agent's reading of "stop if it violates the frozen effective-step or
schedule requirements". It is not a further owner decision:

- The D6 index advances once per iteration, so the schedule, batches and iteration count never change.
- The run stops on a non-finite loss.
- The run stops on a non-finite gradient at grad scale ≤ 1.
- The run stops on an epoch with no applied update.

**O2 — scientific-protocol completion.** The hierarchy has three steps:

1. Minimize VAL video ACER over every unique midpoint plus 0 and 1.
2. Then take the threshold nearest 0.5.
3. Then take the **numerically smallest** threshold.

Every validly evaluated epoch stays eligible; `C_EPOCH_INELIGIBLE` is not implemented. The rule `spoof iff score >= t` is
unchanged, and so is D5 checkpoint selection: minimum ACER, then maximum AUC, then the earliest epoch.
`gpatbench/downstream/metrics.select_threshold` keeps documenting the incomplete spec-15 rule. The completed rule lives in
a separate, additive function.

**O3.** Only the selected checkpoint is retained. The output root is a new private directory
`/media/ext_data/student20261_gpat_m10_detectors`, laid out as `<group>/seed_<s>/`. It is on filesystem
`608647d2-9cb0-45b0-b999-73c080ba909b` and outside the bank root. A separate storage-authority record binds it. The
best-so-far state is persisted atomically and verified before an older best-so-far checkpoint is removed.

**O4.** `gpat-m8-downstream` is approved unchanged (lock `70202661…`).

**O5.** Seeds run in the order 42, 1337, 2026. Within each seed the order is E00, E01, E02, E03, E04, E05, E06b, E06c, for
24 runs, one detector process at a time. Each run waits until the previous run has been independently audited and its
evidence committed.

**O6.** RAW-MATCHED is primary and covers all 8838 valid outputs, with no filtering. E05 near-collapse and E04 clipped
outputs are included. QUALITY-MATCHED is secondary under the frozen specification. N_common across comparisons involving
GPAT cannot be finalized before the GPAT production population exists.

**O7.** The labels follow `outputs/audit/method_status.csv` (`fd97e7cb…`):

- E00 `binary`;
- E01–E05 and E06c `A_FAIR_COMMON_IDENTITY_FREE`;
- E06b `B_NATIVE_FULL_SECONDARY`.

This ruling does not launch scientific training. It does not authorize TEST access, DINOv3, GPAT or any generator or bank
change.
