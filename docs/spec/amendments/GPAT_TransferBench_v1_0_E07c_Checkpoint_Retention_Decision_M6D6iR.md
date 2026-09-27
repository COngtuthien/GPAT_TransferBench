# GPAT-TransferBench v1.0 — E07c Main Checkpoint Retention Decision (M6D6iR)

**Record kind:** OWNER_DECISION / CHECKPOINT_RETENTION
**Classification:** DETERMINISTIC_IMPLEMENTATION_CLARIFICATION
**Status:** OWNER_APPROVED · ADDITIVE · NON-DESTRUCTIVE · PROSPECTIVE
**Authority commit:** `a433b28f861631b1e7828ff247c32f3e3d37bcca` (M6D6i)
**Machine-readable record:** `configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml`

This is not an amendment; no A9 is created. It is not a scientific adaptation, not a new deviation, not a
checkpoint-selection change and not a training-budget change. E07c remains DiffFAS-BIN-IDFREE (controlled encoder
reconstruction), CONTROLLED_ADAPTATION, DEV-021; `new_deviation = false`, `new_fidelity_class = false`.

## 1. Ambiguity resolved

The frozen E07c config (`checkpoint_retention`) retains checkpoint bytes "per official cadence", protects the
authoritative final, selected and officially required checkpoints, states `pruning_in_m6b: false` and anticipates
`if_pruned_later: metadata_and_sha256_records_must_remain`. `run_logging_v1` adds that metadata and SHA256 records
must remain in `checkpoint_index.json` and that silent pruning is FORBIDDEN. No authority stated who may prune
non-protected periodic checkpoint bytes, or when, during execution. This decision resolves only that logistics
question for E07c main scientific runs. The frozen specification requires each run to save the model checkpoint
hash; it contains no clause requiring every periodic checkpoint's bytes to be kept.

## 2. Scope

Applies ONLY to E07c main scientific runs for experiment seeds 42, 1337 and 2026 under
`<runtime_root>/runs/m6/E07c/seed_<seed>/`.

Does NOT apply to: the M6D6g auxiliary encoder run, `encoder_final.pkl`, auxiliary resume sidecars, qualification
artifacts, E01–E06c, M7, M8 or any other method.

## 3. Checkpoint creation, payload and selection are unchanged

Pinned `FAS_train.py` increments `iters` (line 42) before testing `iters % args.save_checkpoints_every_iters == 0`
(line 102) with the argparse default 10000; `iters` starts at 0 before the epoch loop and never resets, so there is
no save at step 0. With 8,838 A1 TRAIN rows, batch size 4 and `drop_last = False` (FAS_train.py:205 passes none):

| Quantity | Value |
|---|---|
| iterations per epoch | ceil(8838 / 4) = 2,210 (2,209 full batches + one batch of 2) |
| total optimizer iterations (400 epochs) | 2,210 × 400 = 884,000 |
| periodic checkpoints | global steps 10,000 … 880,000 = 88 |
| terminal checkpoint (A2-06 clause 3; 884,000 % 10,000 = 4,000) | 1 at global step 884,000 |
| checkpoint states physically created per seed | 89 |

Every periodic checkpoint MUST still be physically created with the exact source payload
`{model, ema, scheduler, optimizer, conf}` via `torch.save` (FAS_train.py:105-114). No skipped save, cadence change,
weights-only checkpoint, optimizer stripping, dtype change, pre-serialization compression, serialization-API change
or payload reduction is authorized. Selection stays `BASELINE_FINAL_STATE_V1`: the terminal state after the full
400-epoch budget; final = selected = terminal; VAL, TEST and loss never select; no early stopping.

## 4. Explicit pruning authorization and successor gate

Non-protected PERIODIC checkpoint bytes MAY be explicitly pruned. A periodic checkpoint C_k may be pruned ONLY AFTER
its direct successor C_{k+1} has completed ALL of:

1. checkpoint file creation completed;
2. file closed successfully;
3. path recorded;
4. epoch recorded;
5. global_step recorded;
6. file_size_bytes recorded;
7. SHA256 computed from the completed file;
8. `checkpoint_index.json` entry durably written;
9. SHA256/index verification passed.

If ANY condition fails, C_k MUST NOT be pruned. The last periodic checkpoint (880,000) MUST remain present until the
terminal checkpoint (884,000) has been created, closed, hashed, indexed and verified; only then may the 880,000 bytes
be pruned.

**Fail-closed rules.** Successor write, hash, index or verification failure: keep the predecessor. Pruning failure:
STOP_AND_REPORT; do not change scientific state; never rewrite index history to pretend pruning succeeded.
Insufficient disk space to create the successor while the predecessor is retained: STOP_AND_REPORT; the predecessor
is never deleted early to make room.

**Crash semantics.** At any instant a running seed keeps the previous verified periodic checkpoint (if one exists)
plus the checkpoint being created/verified until the successor transition completes. This is a storage/retention
rule; it does NOT qualify MAIN_CHECKPOINT_RESUME.

## 5. Protected checkpoint and role deduplication

The authoritative final, selected and officially required checkpoint roles all resolve, for a successfully completed
seed, to the SAME terminal state at global step 884,000. Its bytes are permanently retained and MUST NOT be pruned.
One physical terminal checkpoint satisfies all roles when its path and SHA256 are recorded for each role; duplicate
byte copies are not required. This is storage deduplication of identical logical roles, not deduplication of
different scientific states.

## 6. Metadata survives pruning; silent pruning remains forbidden

Pruning removes bytes, never history. Every `checkpoint_index.json` record keeps its original required fields
(`path`, `epoch`, `global_step`, `file_size_bytes`, `sha256`, `checkpoint_type`, `selected_for_final`,
`selection_reason`) permanently. Additive retention fields:

- retained: `bytes_present: true`, `bytes_pruned: false`;
- pruned: `bytes_present: false`, `bytes_pruned: true`, `pruned_utc`, `prune_reason:
  EXPLICIT_E07C_PERIODIC_RETENTION_POLICY`, `successor_checkpoint_global_step`, `successor_checkpoint_sha256`.

Every prune is explicit and logged, on the exact recorded path of an already-indexed, non-protected periodic
checkpoint. Forbidden: background deletion, unlogged cleanup, glob deletion, automatic keep-latest-N without
per-checkpoint provenance, removal of unknown paths.

## 7. No science change

Dataset, batch size, tail batch, epochs, optimizer, scheduler, EMA, precision, RNG, model, encoder, training
trajectory, cadence, checkpoint contents, selection, final state, evaluation and seeds are unchanged. Bytes are
removed only after the scientific state has been fully serialized and provenance-recorded.

## 8. Resume

MAIN_CHECKPOINT_RESUME remains unqualified. Pruning does not prove resume. A future resume milestone may rely only on
checkpoint bytes still present under this policy; any conflict is resolved prospectively (STOP), never by silently
retaining or deleting different files.

## 9. Future implementation (M6D6j)

M6D6j must: create every official periodic checkpoint; close it; hash it; write/index its full metadata; verify the
index against the bytes; identify the predecessor as periodic, non-selected, non-final and non-protected; explicitly
prune that predecessor; update only its retention metadata; preserve SHA256/path/size/epoch/global_step forever; and
record the prune in logs/evidence. Its tests must cover: no prune before successor verification; no prune on failed
save, hash failure or index failure; protected-final refusal; exact-path pruning only; no glob cleanup; no silent
cleanup; no modification of the scientific payload; checkpoint count and cadence unchanged.

## 10. Storage rationale (RESOURCE_PLANNING_ESTIMATE — not a scientific constant)

The read-only storage-feasibility audit returned GO_IF_OWNER_APPROVES_RETENTION_CHANGE: a mature checkpoint is about
2.476 GB, strict retain-all about 220 GB per seed and about 661 GB for three seeds, against about 443.5 GB available
on the only usable GPU data filesystem. Under this policy the expected peak is about 12.4 GB and the retained terminal
set about 7.4 GB. These figures are planning estimates; execution correctness never depends on them.
