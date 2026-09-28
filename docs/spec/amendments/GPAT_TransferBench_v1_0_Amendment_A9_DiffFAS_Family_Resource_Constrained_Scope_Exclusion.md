# Amendment A9 — DiffFAS family (E07c, E07b) resource-constrained scope exclusion

Status: **OWNER-APPROVED (DiffFAS family only) · ADDITIVE · NON-DESTRUCTIVE · PROSPECTIVE**.
Class: **`OWNER_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION`** (a scientific benchmark-scope change).
Scope: **the DiffFAS family only**: E07c DiffFAS-BIN-IDFREE (Track A) and E07b DiffFAS-NATIVE (Track B).
Milestone: M6A9, a standalone scope decision. It is **not** the M6 closure, and **M6 remains OPEN**
(`M6_CLOSED = false`); the identifier M6E stays reserved for the real closure. Audit evidence:
`outputs/audit/M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION.md` / `.json`. Authority commit: `632e274bde083fd175dc6a2645542d281a767063` (M6D6jF).
Machine-readable companion: `configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml`.

This IS an amendment, not a deterministic implementation clarification: E07c was an active required Track-A
baseline, and the owner is discontinuing the DiffFAS family. The frozen DOCX and Amendments A1–A8 are not edited. No
earlier milestone is relabelled.

**Out of A9's scope.** A9 does **not** cover:
- **E06b DSDG-NATIVE**. It is **not owner-excluded** and **not resource-blocked**, and it **remains ACTIVE M6 WORK**
  under A1 §3.
- **E06c DSDG-BIN-IDFREE** or any other non-DiffFAS baseline. These are untouched.
- Any GPAT configuration.

A9 is **not**:

- a source gap (`BLOCKED_BY_SOURCE_GAP` is **not** used);
- a model failure, an OOM or a technical implementation failure;
- a replacement of DiffFAS by another baseline;
- a silent deletion of DiffFAS from the benchmark (the E07c and E07b rows are retained as `N/A`).

## 1. Owner decision — E07c

E07c **DiffFAS-BIN-IDFREE** (controlled encoder reconstruction) will **not** continue scientific execution. The
reason is resource: the compute, wall-clock and storage budget for the frozen E07c contract (3 seeds × 400 epochs ×
2,210 iterations = 3 × 884,000 optimizer steps, 89 checkpoint states per seed) is disproportionate to the available
single-GPU resources.

Informational extrapolation only, not a measured result: the owner-interrupted seed-42 attempt took 40,987.14 s for
54,836 steps. Linear extrapolation gives about 660,700 s (≈ 7.6 days) per seed and ≈ 22.9 days for the three required
seeds, before M8 bank generation.

| Field | Value |
|---|---|
| exclusion class | `OWNER_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION` |
| M6 hard-gate status | `blocked` |
| blocker/exclusion reason | `OWNER_EXCLUDED_RESOURCE_CONSTRAINT` |
| fidelity class (unchanged, historical) | `CONTROLLED_ADAPTATION` |
| deviation (unchanged) | `DEV-021` |
| new fidelity class / new deviation | none / none |
| completed scientific seeds | 0 of 3 |
| eligible for a future bank | **false** |

Explicit consequences:

1. E07c seed 42 is not rerun. Seeds 1337 and 2026 are never started.
2. The partial seed-42 checkpoint is never used as a scientific checkpoint. It is not selected, not terminal, not
   final and not a `BASELINE_FINAL_STATE_V1` state.
3. No E07c M8 bank is generated. E07c is not used for downstream training or evaluation.
4. Later tables, plots and statistics **retain the E07c row** wherever their schema lists the Track-A methods. A
   self-run cell with no result is marked `N/A` with the provenance `EXCLUDED_OWNER_RESOURCE_CONSTRAINT (A9)`. E07c is
   never silently removed from a table schema. No placeholder number is entered.
5. Comparison and multiple-comparison families (spec §18: Holm–Bonferroni over GPAT-B0 vs the Binary-track baselines)
   contain only methods that actually completed and are eligible. The E07c exclusion is stated explicitly next to
   every such family.
6. All historical E07c implementation, qualification, auxiliary-encoder, failed-attempt and partial-run evidence is
   preserved. This follows spec §28: negative results and failed baselines stay in the evidence package.
7. The M6D6jR authorisation to launch E07c main scientific training is **superseded prospectively** by this
   exclusion. The M6D6jR record itself is not edited.

## 2. Preserved E07c seed-42 attempts

**Attempt 1** (authority `8358d8b6fe468478ad86a715616becb81bb9b339`). Failed at `RUN_CONTEXT_OPEN`:
`dict.update() got multiple values for keyword argument 'experiment_seed'`. 0 scientific optimizer steps and 0
scientific checkpoint writes. Documented by M6D6jF, which is unchanged.

**Attempt 2** (authority `632e274bde083fd175dc6a2645542d281a767063`). Runtime root
`/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/seed_42` on the GPU host. `run_id`
`394e17cccba0954d`, `runner_mode = SCIENTIFIC`, start `2026-09-28T06:33:06Z`, end `2026-09-28T17:56:18Z`. The owner
**manually interrupted** it after deciding not to continue DiffFAS. The traceback ends in a Python
`KeyboardInterrupt` during `train_iteration`; it is not an OOM or a model error.

- Last completed global step 54,836: epoch 25, iteration-in-epoch 1,795, wall clock 40,987.14273164095 s.
- `metrics.jsonl` has 54,900 lines. The checkpoint index has 5 records. There are 54 visualizations.
- `run_summary.json`: `completion_status = failed`, `failure_reason = "KeyboardInterrupt: "`,
  `final_or_selected_checkpoint_path = null`, `final_or_selected_checkpoint_sha256 = null`,
  `seed_level_evaluation_metrics = null`, `test_split_accessed = false`.
- Run-root footprint is about 2.5 GiB.

Interpretation: a **PARTIAL / OWNER-INTERRUPTED** scientific attempt. It is not a completed seed and it provides no
benchmark result.

Provenance: these runtime facts are owner-reported observations of the GPU host. This decision did not contact the GPU host
and did not re-read the run root.

**Historical logging inconsistency (recorded, not rewritten).** `run_summary.failure_reason` contains
`KeyboardInterrupt`, while `missing_field_reasons.failure_reason` says the caller did not supply it. This is
metadata only. It affects no step, checkpoint or result, and E07c is discontinued, so no hotfix milestone is
opened. The runtime files are not edited.

## 3. Partial checkpoint bytes — retained

Only one checkpoint's bytes are present: `checkpoints/model_050000.pt`. Its details are global step 50,000, epoch 22,
2,475,667,461 B, SHA256 `b4ae6325bd04fdf615db24ca0cfb883680401c494b9c9d39f4830dbc8d775cae`, type `periodic`,
`selected_for_final = false`, selection reason "Official cadence; not final selection", `bytes_present = true` and
`bytes_pruned = false`.

Decision: **RETAIN** (about 2.4 GiB). The checkpoint is classified
`PARTIAL_FAILED_RUN / NOT_ELIGIBLE_FOR_SCIENCE`. It is **not eligible** for M8, downstream evaluation or numeric
paper reporting. A9 grants **no** new pruning permission. The M6D6iR successor gate still applies, and the checkpoint
has no successor, so its bytes may not be pruned under M6D6iR either.

Any future removal needs a separate, explicit, prospective owner decision. That decision must record: path, original
size, SHA256, step, epoch, checkpoint type, original bytes-present state, prune reason and time, and evidence of
owner authorisation. No other E07c runtime evidence may be deleted.

## 4. E07b DiffFAS-NATIVE — excluded with the DiffFAS family

A1 §3 and `configs/frozen/fair_track_v1.yaml` (`track_b.status = DEFERRED_TO_M6_SECONDARY_TRACK`) deferred the
native Track-B rows *to* M6. Because the owner discontinues the whole DiffFAS family, **E07b DiffFAS-NATIVE** is
closed by this exclusion:

| Row | track | m6_gate_status | blocker_or_exclusion_reason | fidelity_class | future bank |
|---|---|---|---|---|---|
| E07b DiffFAS-NATIVE | B | `blocked` | `OWNER_EXCLUDED_RESOURCE_CONSTRAINT` (DiffFAS family discontinued) | `NOT_ASSESSED_NOT_IMPLEMENTED` | false |

- E07b was never implemented, and its fidelity was never assessed. `BLOCKED_BY_SOURCE_GAP` is not claimed.
- E07b has no bank, no downstream training or evaluation, and no numeric result.
- The A1 text and `fair_track_v1.yaml` are not altered. Nothing is deleted. Any Track-B table lists E07b as `N/A`
  (`EXCLUDED_OWNER_RESOURCE_CONSTRAINT (A9)`).
- No native reconstruction-pair manifest is created, and no identity is fabricated.

## 5. Consequences for M6 (M6 is NOT closed by A9)

- **E06b DSDG-NATIVE remains active M6 work.** A1 deferred it to M6, and no owner decision excludes it. It must be
  implemented and qualified as far as the frozen M6 gate requires, unless an independently evidenced blocker is
  found. No blocker may be invented to close M6.
- **`M6_CLOSED = false`.** M6 closes only when every remaining non-DiffFAS M6 row has a legitimate
  faithful/adapted/blocked tag. The authoritative `outputs/audit/method_status.csv` is created at that closure (M6E),
  not before, so no row receives a premature terminal status.
- The rows settled by A9 are E07c (`blocked`) and E07b (`blocked`). The fidelity and status evidence of the other
  Track-A rows (E01–E06c) is unchanged by A9.
- M6 acceptance is **not** the same as baseline scientific execution:
  `baseline_full_scientific_execution_complete = false`.
- The M6D6jR O3 decision is unchanged. The deterministic end-to-end smoke test is required. It is not waived and not
  completed. It is not an M6 closure requirement and does not block M7 generator training. It is a mandatory HARD
  GATE before the first FULL M8+ bank / evaluator / metrics / plots execution.
- **`outputs/audit/STAGE_STATE.json` is a stale historical fixture** (`STALE_HISTORICAL_FIXTURE`). It is NOT the
  current milestone authority after M5: it still reads M6 `NOT_STARTED`. The current authority is the execution
  ledger together with the milestone evidence and amendment records. It is left byte-unchanged (SHA256
  `a8abf828…`), because historical tests pin its content.

A9 changes no frozen config, no GPAT config, no non-DiffFAS baseline's contract, no data, split, pair manifest, probe, metric
or threshold. It performed no training, no optimizer step, no checkpoint write, no bank generation, no GPU contact and
no TRAIN, VAL or TEST access.
