# M6A9 — Amendment A9: DiffFAS family resource-constrained scope exclusion

| Field | Value |
|---|---|
| Milestone | M6A9, classification `M6A9_DIFFFAS_FAMILY_SCOPE_EXCLUSION` (a standalone scope decision, not the M6 closure) |
| Authority | `632e274bde083fd175dc6a2645542d281a767063` (M6D6jF), branch `m6-baselines` |
| Class | `OWNER_RESOURCE_CONSTRAINED_SCOPE_EXCLUSION`, a scientific scope change |
| A9 record | `configs/amendments/difffas_a9_resource_constrained_scope_exclusion.yaml` |
| A9 record SHA256 | `5ddd59f3560eea63471dd5bc1b4f65281e8f529a74a1e9d4d990ddbb427f3644` |
| Verifier | `tools/m6a9_difffas_scope_exclusion_preflight.py` (stdlib only) |

## Scope: the DiffFAS family only

| Row | Track | Decision | Fidelity | Future bank |
|---|---|---|---|---|
| E07c DiffFAS-BIN-IDFREE | A | `blocked`, `OWNER_EXCLUDED_RESOURCE_CONSTRAINT` | `CONTROLLED_ADAPTATION` / DEV-021 (unchanged) | false |
| E07b DiffFAS-NATIVE | B | `blocked`, `OWNER_EXCLUDED_RESOURCE_CONSTRAINT` (family discontinued) | `NOT_ASSESSED_NOT_IMPLEMENTED` | false |

**E07c:**
- Seed-42 attempt 1 (`8358d8b`) failed at `RUN_CONTEXT_OPEN` with 0 steps; it is preserved.
- Seed-42 attempt 2 (`632e274`, run_id `394e17cccba0954d`) was owner-interrupted at global step 54,836 of 884,000; it
  is preserved.
- 0 of 3 scientific seeds completed. Seeds 1337 and 2026 were never started.
- `model_050000.pt` (SHA256 `b4ae6325…5cd775cae`) is retained as `PARTIAL_FAILED_RUN / NOT_ELIGIBLE_FOR_SCIENCE`. It
  is never selected, terminal or final, and it is not eligible for an M8 bank, downstream evaluation or numeric
  reporting. No pruning permission is granted.

**E07b:** never implemented, and its fidelity was never assessed. It has no bank, no downstream training or
evaluation, and no numeric result. `BLOCKED_BY_SOURCE_GAP` is not claimed. The A1 text is unaltered.

## Out of scope

- **E06b DSDG-NATIVE** is not in A9. It is not owner-excluded and not resource-blocked, and it remains
  **ACTIVE M6 WORK** under A1 §3.
- E06c, E01–E05 and GPAT are unchanged.

## M6 state

**M6_CLOSED = false.** M6 remains OPEN. `outputs/audit/method_status.csv` is not created; it will be created at the
M6 closure (M6E), after the E06b M6 work is resolved. None of the withdrawn provisional M6E closure artifacts was
revived.

`outputs/audit/STAGE_STATE.json` is a `STALE_HISTORICAL_FIXTURE`. It is not the milestone authority after M5, and it
is left byte-unchanged.

The M6D6jR O3 smoke gate is unchanged: required, not completed, and a hard gate before the first full M8+ execution.

A9 performed no training, no optimizer step, no checkpoint write, no bank generation, no GPU contact, and no TRAIN,
VAL or TEST access. **M7 HAS NOT STARTED.**
