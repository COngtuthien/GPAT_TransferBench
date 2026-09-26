# Amendment A8 — E07c auxiliary scientific-run failure/restart and exact epoch-boundary resume policy

Status: **OWNER-FROZEN · ADDITIVE · NON-DESTRUCTIVE**.
Class: **`DETERMINISTIC_IMPLEMENTATION_CLARIFICATION`** (A2 §1 taxonomy).
Milestone: M6D6f. Method: E07c — DiffFAS-BIN-IDFREE (controlled encoder reconstruction), auxiliary
conditioning encoder only.

The frozen specification, A1/A2/A3/A6/A7, the frozen E07c configs and the M6D6e production-runner
contract require **exactly one** auxiliary training run (`auxiliary_encoder_training_seed = 42`,
`auxiliary_encoder_training_runs = 1`, 200 epochs; A3 §5.4, §5.4b). Pinned
`models/pretrain_classifier.py` (SHA256 `3444691f…`) has **no** resume mechanism. No existing
authority therefore says what an interruption of that long run means. A8 is the owner's answer.
It defines recovery from infrastructure interruption only. It changes **no** uninterrupted
scientific trajectory, and bitwise trajectory equivalence is a hard acceptance criterion
(M6D6f qualification). `run_logging_v1` already requires `APPEND_OR_EXPLICITLY_RECONCILE`,
`resume_overwrite: FORBIDDEN` and a reconciliation record.

E07c fidelity stays **`CONTROLLED_ADAPTATION`** under **DEV-021**. A8 adds **no new fidelity class
and no new scientific deviation**. The source pin is unchanged (`murphytju/DiffFAS` @
`23f40519ec25a833ebc06842aa6fbab74fad4d15`), and the pinned source is not patched. A1, A2, A3, A6,
A7, their overlays, the frozen E07c configs and the M6D6e contract are not edited. The M6D6e
contract's `resume.status = AUX_RESUME_NOT_QUALIFIED` remains the historical M6D6e record; A8
supersedes it for the auxiliary run only after M6D6f qualification passes.

Machine-readable companion: `configs/amendments/e07c_a8_aux_resume_policy.yaml`.

## 1. Logical run identity

`AUXILIARY_SCIENTIFIC_RUN_IDENTITY = ONE_LOGICAL_RUN`.

* A scientific auxiliary execution may consist of several OS processes, but only because of
  interruption and recovery. Those processes continue **one** logical seed-42 run. They are not
  new scientific attempts, provided every A8 gate passes.
* `process count ≠ scientific run count`. The scientific count stays **1**.
* A resume preserves the original `run_id` (run_logging_v1 formula, label `E07c/aux_encoder`),
  the original `run_uuid`, the original `start_utc` and the same `seed_42` root. Each process
  receives a separate **process-session id** for audit only.
* A crash or restart must never be used to select a better trajectory. No rerun may be triggered
  by loss, accuracy, convergence or any other scientific outcome.

## 2. Exact epoch-boundary resume only

* Resume is permitted **only** from the latest **committed** epoch boundary: `completed_epoch = E`,
  and resume starts at epoch `E + 1`.
* There is no mid-batch continuation and no continuation from an arbitrary optimizer step.
* If the interruption happens during epoch `E + 1`, all optimizer work after committed epoch `E` is
  logically rolled back. Epoch `E + 1` is replayed from its beginning after exact state
  restoration. This is allowed **only** because the replayed trajectory is proven bitwise identical
  to uninterrupted execution (M6D6f).

## 3. Epoch 0 is a committed boundary

Before the first DataLoader iterator of training, the runner commits an engineering resume state
with `completed_epoch = 0`, `global_step = 0`. It is written **after** the seed is applied, the
model is constructed and moved to CUDA, the optimizer, criterion and DataLoader are constructed,
and the A7 precision state is established. It is written **before** `iter(loader)`, the first
batch, the first backward and the first `optimizer.step`. A crash before epoch 1 completes can
therefore resume without inventing a second scientific run.

## 4. The resume state is engineering infrastructure

The engineering resume sidecar is **not**:

* the scientific auxiliary checkpoint;
* an official DiffFAS checkpoint;
* a selection candidate;
* consumable by main DiffFAS;
* reportable as a trained model.

The scientific/source checkpoint remains exactly `checkpoints/encoder_final.pkl`, in the M6D6c/M6D6e
format `torch.save(whole nn.Module)`, overwritten after every completed epoch
(`pretrain_classifier.py:45`), with the final epoch-200 bytes selected (`FINAL_STATE_AFTER_EPOCH_200`).
There is no VAL/TEST selection. A8 does not change that format or its cadence. The sidecar stores
`state_dict` tensors internally only because it is an engineering recovery artifact; that is not a
substitution of the scientific checkpoint format.

## 5. Sidecar content and serialization

The sidecar holds enough state to reproduce the next training draw bitwise, and is self-contained:

* schema, version, kind `E07C_AUX_ENGINEERING_RESUME_STATE`, method `E07c`;
* logical run identity (`run_id`, `run_uuid`, `start_utc`), mode, and seed (42 in scientific mode);
* `completed_epoch`, `global_step = 56 · completed_epoch`;
* **model `state_dict`**, so the state never depends on the overwritten `encoder_final.pkl`;
* **SGD `state_dict`**, including every momentum buffer;
* torch CPU RNG, torch CUDA RNG of the execution device, and torch CUDA RNG of all devices;
* Python `random` state and NumPy global MT19937 state, both as integer tensors plus primitives;
* the A7 precision state;
* identities: git commit/branch/worktree status, config, A1, A3, A6, A7, A8, the M6D6e contract,
  run_logging_v1, the environment lock and interpreter, the source commit/tree, the split manifest,
  the K7 class map, the execution config and faces root, and the SHA256 of the runner code files;
* the SHA256 of the current scientific whole-module checkpoint, when one exists (`E ≥ 1`);
* the metrics-log reconciliation boundary: byte length, prefix SHA256, line count and last
  authoritative `global_step` of `metrics.jsonl` at commit time;
* logical, physical and superseded optimizer-step counters.

It is written with `torch.save` and is loadable with `torch.load(..., weights_only=True)`.
**No unsafe (`weights_only=False`) load is ever applied to a sidecar.** The M6D6c whole-module
loader is unchanged and is not used for sidecars.

## 6. Versioned, fail-closed commit transaction

```
checkpoints/resume/runner_state_epoch_000.pth
checkpoints/resume/runner_state_epoch_001.pth
...
resume_state_index.json
```

`resume_state_index.json` authorizes exactly **one** committed state. The write order is fixed:

1. write the new sidecar to `<name>.partial`, then flush, fsync and close;
2. compute its size and SHA256;
3. `os.replace` it to its versioned final name, then fsync the directory;
4. atomically replace `resume_state_index.json` **last**.

Only after the new index is committed may the previous sidecar's bytes be pruned. Pruning is
explicit: the old epoch, path, SHA256 and size remain in the index `history` with
`bytes_pruned = true`. Silent pruning is forbidden. A crash at any point before step 4 leaves the
previously committed state intact and authorized.

At epoch boundary `E ≥ 1` the order is:

1. the source-native whole-module save (`encoder_final.pkl`, through `.partial` then `os.replace`);
2. its `checkpoint_index.json` entry;
3. the sidecar commit;
4. pruning of the previous sidecar.

## 7. Restore order (fresh process)

1. Static CLI and identity gates (no Torch).
2. Seed exactly as a fresh run, then construct the exact production model, DataLoader, optimizer
   and criterion. This consumes the same RNG as a fresh run and is then overwritten.
3. Resolve the explicitly named sidecar. It must be inside `<run_root>/checkpoints/resume/`, must
   not be a symlink, and must be the index's committed entry. Its SHA256 must be well formed
   (checked before any file open), and the bytes must match that SHA256 and size **before**
   `torch.load`.
4. `torch.load(BytesIO(verified bytes), weights_only=True, map_location='cpu')`.
5. Validate every identity, `completed_epoch`, `global_step`, and the metrics prefix.
6. Restore the model `state_dict` (strict).
7. Restore the optimizer `state_dict`.
8. Validate the optimizer tensors and devices, including momentum on the parameter device.
9. Restore Python and NumPy RNG.
10. Restore torch CPU RNG.
11. Restore torch CUDA RNG.
12. Assert the A7 execution state.
13. Only then create the next training iterator.

Nothing that consumes RNG runs between step 9 and step 13.

## 8. DataLoader semantics are unchanged

E07c keeps `shuffle=True`, `generator=None`, `num_workers=6` and `persistent_workers=False`. Each
epoch's `RandomSampler` seed and worker base seed are drawn from the process torch CPU generator
when the iterator is created. The committed torch CPU RNG state at the boundary therefore carries
the next epoch's order. No explicit DataLoader generator is added, and no worker setting changes.

## 9. Metrics log is append-only

`metrics.jsonl` is **never** truncated or rewritten. On resume from boundary `E`:

* every physical record after the boundary is retained;
* the runner appends the run_logging_v1 `resume_reconciliation` record and an
  `e07c_aux_resume_reconciliation` record;
* that record classifies the step records with `global_step > 56·E` (and any epoch-complete or
  checkpoint events of epochs `> E`) as **`SUPERSEDED_BY_RESUME_ROLLBACK`**, with their step and
  epoch range;
* it states the committed boundary, the restored sidecar path and SHA256, and the process-session
  id;
* epoch `E + 1` is replayed from its beginning, and new authoritative records are appended.

The authoritative logical trajectory is recoverable from the file alone. Walking the records in
order, each reconciliation truncates the logical view to its boundary. The physically executed
records remain visible. A torn (non-newline-terminated) final line is **not** repaired
automatically: it is `STOP_AND_REPORT`.

## 10. Step accounting

Every summary reports three counters separately:

* `logical_authoritative_optimizer_steps`;
* `physical_optimizer_steps_executed`, counted from logged step records;
* `superseded_optimizer_steps`.

A completed run has **200 × 56 = 11200** logical steps whatever the physical count, and its final
model corresponds exactly to that logical trajectory.

## 11. Failure policy

Resume is allowed **only** after infrastructure or process interruption: SSH disconnect, process
crash, host reboot, scheduler termination, or a transient non-scientific infrastructure failure.

Resume is **never** a mechanism for:

* rerunning a bad loss or poor convergence;
* choosing another order;
* changing the seed, config, code, environment, batch size or precision.

A NaN or Inf, corrupted state, identity drift, a torn metrics record, or a non-bitwise resume is
**STOP_AND_REPORT**. There is no automatic restart.

## 12. Identity gates

A resume is refused if any of these differ from the original run:

* git commit, branch, or worktree cleanliness (no cross-commit and no dirty-worktree resume in
  scientific mode);
* source pin or tree;
* frozen config, A1, A3, A6, A7, A8, the M6D6e contract, or run_logging_v1;
* environment lock, recorded environment, or Python executable;
* TRAIN split manifest SHA256 or K7 class map;
* execution config or canonical faces root;
* seed, mode, or run id;
* A7 precision state.

## 13. CLI

* Initial run: `tools/run_e07c_aux.py --seed 42 --execution-config …`.
* Resume: `tools/run_e07c_aux.py --seed 42 --execution-config … --resume-state <EXACT_PATH>`.
* There is no `--resume-latest` and no automatic discovery. The given path must be exactly the file
  that `resume_state_index.json` currently authorizes, inside that run root.
* A normal invocation refuses an existing `seed_42` root.
* A resume invocation refuses a missing, empty or inconsistent root.

## 14. What A8 does not change

Architecture, K7 head, preprocessing, loss, SGD hyperparameters, batch 256, drop_last, shuffle,
6 workers, 200 epochs, seed 42, the single-run rule, the whole-module checkpoint format and cadence,
the final-state checkpoint rule, precision (A7) and the TRAIN population are unchanged. A8 applies
only to the auxiliary encoder. It does **not** qualify main-DiffFAS checkpoint resume
(`MAIN_CHECKPOINT_RESUME` stays not qualified).
