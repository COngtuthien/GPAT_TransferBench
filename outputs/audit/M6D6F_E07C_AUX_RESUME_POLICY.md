# M6D6f — E07c auxiliary failure/restart policy, exact epoch-boundary resume, bitwise resume qualification

**Status:** PASS (candidate, uncommitted) · **Method:** E07c DiffFAS-BIN-IDFREE (controlled encoder reconstruction)
**Fidelity:** `CONTROLLED_ADAPTATION` · `DEV-021` (unchanged; no new fidelity class, no new deviation)
**Method status:** `IMPLEMENTED_NOT_EXECUTED` · **Authority:** `a00aba9f998fb0d91122617012af0dec7794c007` (M6D6e)
**Scope:** `QUALIFICATION_ONLY`, qualification seed **60606**, authorized real TRAIN access, no VAL, no TEST.
The scientific auxiliary seed-42 run was **not** launched. There are 0 scientific auxiliary runs and 0 main runs.

## 1. Why M6D6f exists

A3 §5.4/§5.4b require exactly one auxiliary encoder run (seed 42, 200 epochs). Pinned
`models/pretrain_classifier.py` (`3444691f…`) has **no** resume mechanism: `for epoch in range(0,num_epochs)`, with
no `torch.load`, no `load_state_dict`, no optimizer state and no start epoch. M6D6e therefore refused an existing
`seed_42` root (`AUX_RESUME_NOT_QUALIFIED`). Without a rule, an interruption at epoch 137 would leave open whether the
run restarts, whether it counts as the one run, and whether momentum, RNG and metrics survive. M6D6f answers this
before seed 42 is ever launched.

`run_logging_v1` already requires `resume_policy: APPEND_OR_EXPLICITLY_RECONCILE`, `resume_overwrite: FORBIDDEN` and
`resume_reconciliation_record_required: true`.

## 2. Owner decision — Amendment A8 (`DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`)

| File | SHA256 |
|---|---|
| `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A8_E07c_Aux_Resume_Policy.md` | `deb11b5cd80fa873bde1ef9e87160fed3a8065903cd8c81c9463b087fbf73b2f` |
| `configs/amendments/e07c_a8_aux_resume_policy.yaml` | `a098ce509156151ee6ce7ead1e9afa5711264b676a597e2514b1058ed46f471a` |

* **Logical run:** `ONE_LOGICAL_RUN`. Interruption and recovery may add OS processes, but every process continues the
  same run: same `run_id`, `run_uuid`, `start_utc`, `seed_42` root and seed. `process count ≠ scientific run count`,
  and the scientific count stays **1**. Each process has a separate process-session id, for audit only. An
  outcome-driven rerun is forbidden.
* **Boundary:** `EXACT_EPOCH_BOUNDARY_ONLY`. After committed epoch E, the run resumes at E+1. A partial epoch E+1 is
  logically rolled back and replayed from its beginning. There is no mid-batch or arbitrary-step continuation.
* **Epoch 0** is committed after seed, model, `.cuda()`, optimizer, criterion, DataLoader and A7 are in place, and
  before `iter(loader)`.
* **Failure policy:** only infrastructure interruptions may be resumed. NaN, Inf, corrupted state, identity drift, a
  torn metrics record or a non-bitwise resume is STOP_AND_REPORT. There is no automatic restart.
* The M6D6e contract's `AUX_RESUME_NOT_QUALIFIED` stays the historical M6D6e record. A8 is additive.

## 3. Implementation

* `methods/difffas/aux_resume.py` (new) holds the sidecar schema, the fail-closed transaction, verified loading,
  restore, reconciliation and the `LogicalRun` driver.
* `methods/difffas/aux_runner.py` gains a continuation-capable run context (`resume=True` requires the existing root;
  a fresh context still refuses it). `run_scientific` becomes the A8 logical-run driver. The step, the epoch loss
  (denominator 14467) and the whole-module save are unchanged, and the engine itself loads no state.
* `methods/difffas/aux_runner_io.py`: adds resume-qualification seed 60606 with labelled `qualification/m6d6f` roots.
  `--resume-state` is accepted, and every discovery-style flag stays refused.
* `tools/run_e07c_aux.py`: `--seed 42 --resume-state <EXACT_PATH>`. The path is resolved statically against
  `resume_state_index.json` before Torch is imported. There is no `--resume-latest` and no discovery.
* The pinned DiffFAS source is not patched.

**Sidecar** `checkpoints/resume/runner_state_epoch_<EEE>.pth`, kind `E07C_AUX_ENGINEERING_RESUME_STATE`, schema 1. It
holds:

* model `state_dict`, so it is self-contained;
* SGD `state_dict` with all 110 momentum buffers;
* torch CPU RNG, torch CUDA RNG (execution device and all devices), Python `random` and NumPy MT19937 state, as
  integer tensors plus primitives;
* the A7 state and 30 identity fields;
* the SHA256 of the current `encoder_final.pkl`;
* the metrics boundary: bytes, prefix SHA256, lines and last step;
* logical, physical and superseded counters.

It is serialized with `torch.save` and loaded **only** with `torch.load(BytesIO(bytes), weights_only=True,
map_location='cpu')`, after the bytes' SHA256 and size are verified against the index. `weights_only=True` was never
relaxed. A `TorchVersion` inside the environment identity was coerced to a plain string instead (development
iteration D2).

**Transaction:**

1. write `.partial`, then fsync;
2. compute SHA256;
3. `os.replace`, then fsync the directory;
4. write `resume_state_index.json` **last**;
5. explicitly prune the previous sidecar, with history kept (`bytes_pruned: true` plus `pruning_log`).

At each epoch boundary the whole-module `encoder_final.pkl` save and its `checkpoint_index.json` entry come first,
unchanged. Stale uncommitted files found at resume are recorded (SHA256 and size), then removed; they are never loaded.

**Restore order** (fresh process):

1. static gates;
2. seed and construct exactly as a fresh run;
3. verify the exact indexed path; the SHA256 must be well formed before open, and the bytes' SHA256 is checked before
   load;
4. load with `weights_only=True`;
5. check identity gates;
6. restore the model;
7. restore SGD;
8. check tensors and devices;
9. restore Python and NumPy RNG;
10. restore torch CPU RNG;
11. restore torch CUDA RNG;
12. check A7;
13. create the iterator.

The DataLoader is unchanged: `shuffle=True`, `generator=None`, 6 workers, `persistent_workers=False`. The committed
torch CPU RNG carries the next epoch's sampler seed.

## 4. Qualification (five fresh processes, seed 60606, real TRAIN, production engine + A8 driver)

| Process | Root (`q60606-b1eeb27a03b31c00-…`) | What it did | Physical steps |
|---|---|---|---|
| reference | `-reference` | epoch-0 sidecar → full epoch 1 (loss 1.3738) → `encoder_final.pkl` + epoch-1 sidecar → first epoch-2 step, probed | 57 |
| interrupted | `-interrupted` | same → ONE epoch-2 step logged → deliberate interruption (no epoch-2 sidecar) | 57 |
| restore | `-interrupted` (resumed) | explicit epoch-1 sidecar → reconcile → restore → replay the first epoch-2 step | 1 |
| e0interrupted | `-e0interrupted` | epoch-0 sidecar → ONE epoch-1 step logged → deliberate interruption | 1 |
| e0restore | `-e0interrupted` (resumed) | explicit epoch-0 sidecar → replay the first B256 step | 1 |

These are qualification processes, not scientific attempts. All five passed on the first attempt, and none was
superseded.

### Bitwise acceptance (tolerance: NONE; max |Δ| = 0)

| Comparison | Batch ids + order | Input | Labels | Loss | Gradients | Parameters | Momentum | CPU RNG | CUDA RNG |
|---|---|---|---|---|---|---|---|---|---|
| resumed epoch-2 step vs **reference** | = | = | = | `0x1.26b12p+0` = | 110/110 | 112/112 (+111 buffers) | 110/110 | = | = |
| resumed epoch-2 step vs **superseded** step | = | = | = | = | 110/110 | 112/112 | 110/110 | = | = |
| epoch-0 restore vs reference first B256 step | = | = | = | `0x1.11d5fep+1` = | 110/110 | 112/112 | 110/110 | = | = |
| epoch-0 restore vs superseded first step | = | = | = | = | 110/110 | 112/112 | 110/110 | = | = |

Resumed epoch-2 first step:

* batch `504ffe4b2eea2ddb…`;
* input `3f0abb400eda4cd6…`;
* gradients `b6621f1f5c9b8eb3…`;
* parameters after the step `a7c8419c7da5c2f9…`;
* momentum `5d1754fe53638256…`;
* torch CPU RNG `b0d838e4acde4a9b…`;
* torch CUDA RNG `15aaf26a47699822…`.

The RNG state before the epoch-2 iterator in the restore process equals the reference's epoch-1 boundary state
bitwise, and the same holds for epoch 0. The two fresh epoch-1 processes also agree bitwise:

* the same order `8561dd1d…`;
* the same 56 losses and epoch loss (1.3738414479363326, denominator 14467);
* the same `encoder_final.pkl` SHA256 `8cca4429…`;
* the same sidecar model state.

`BLOCKED_BY_AUX_RESUME_NONDETERMINISM` did **not** occur.

### Append-only reconciliation

The `-interrupted` root's `metrics.jsonl` (65 records) reads, in file order:

1. `run_start`, `resume_state_committed(0)`, steps 1…56, `epoch_complete`, `checkpoint_event`,
   `resume_state_committed(1)`;
2. **step 57 (physical, superseded)**;
3. `resume_reconciliation`, `e07c_aux_resume_reconciliation`;
4. **step 57 (replayed, authoritative)**.

The first 95,454 bytes are unchanged (prefix SHA256 verified). Nothing was truncated or rewritten. The
reconciliation record says:

* classification `SUPERSEDED_BY_RESUME_ROLLBACK`;
* step range [57, 57], epoch 2;
* boundary E=1 / step 56;
* restored sidecar `489c1804…`.

The logical view has 57 step records; the physical view has 58.

### Step accounting (qualification only; never scientific steps)

`interrupted` root after the restore process:

* logical authoritative 57: 56 committed + 1 in progress;
* physical 58;
* superseded 1;
* sessions 2.

`e0interrupted` root: logical 1, physical 2, superseded 1.

Physical optimizer steps per process: 57 / 57 / 1 / 1 / 1. A completed scientific run will report
**11200** logical steps.

### Access (per process, never combined)

| Process | TRAIN faces | VAL/TEST images | Raw images | Other manifests | 8838-row main relation | Firewall denials |
|---|---|---|---|---|---|---|
| reference | 17664 (14336 epoch 1 + 3328 epoch-2 prefetch) | 0 / 0 | 0 | 0 | not opened | 0 |
| interrupted | 17664 | 0 / 0 | 0 | 0 | not opened | 0 |
| restore | 3328 (13-batch prefetch) | 0 / 0 | 0 | 0 | not opened | 0 |
| e0interrupted | 3328 | 0 / 0 | 0 | 0 | not opened | 0 |
| e0restore | 3328 | 0 / 0 | 0 | 0 | not opened | 0 |

Every process exposed 14467 TRAIN rows and 0 VAL/TEST rows. No `runs/` path existed before or after any process.

### Retention

Qualification bytes are retained on the GPU runtime (not deleted):

* the qualification roots, 1.3 GB (`encoder_final.pkl` and committed sidecars);
* the bitwise probes, 3.3 GB.

All are `QUALIFICATION_ONLY` and never consumable. The only deletion was the explicit, index-recorded prune of each
superseded epoch-0 sidecar.

## 5. Statuses

New:

* `E07c_AUX_FAILURE_RECOVERY_POLICY_FROZEN`
* `E07c_AUX_EPOCH_BOUNDARY_RESUME_QUALIFIED`
* `E07c_AUX_APPEND_ONLY_RECONCILIATION_QUALIFIED`

`AUX_RESUME` leaves the not-qualified list. The method stays `IMPLEMENTED_NOT_EXECUTED` / `CONTROLLED_ADAPTATION`.

Still not qualified:

* `AUXILIARY_ENCODER_200_EPOCH_TRAINING`
* `AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT`
* `AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN`
* `MAIN_DIFFFAS_TRAINING_GRAPH`
* `MAIN_CHECKPOINT_RESUME`
* `MAIN_RUNNER_ENCODER_LOAD_INTEGRATION`
* `MAIN_PRODUCTION_RUNNER`
* `SCIENTIFIC_TRAINING`
* `M8_BANK`

The auxiliary encoder has not been trained scientifically.

## 6. Next single milestone (not executed)

**M6D6g**: launch the ONE scientific E07c auxiliary seed-42 run for 200 epochs with `tools/run_e07c_aux.py --seed 42`,
from a clean committed worktree. If an infrastructure interruption occurs, continue it under A8 with
`--resume-state <EXACT_PATH>`.
