# M6D6iR — E07c main checkpoint retention owner decision

**Record kind:** OWNER_DECISION / CHECKPOINT_RETENTION · **Classification:** DETERMINISTIC_IMPLEMENTATION_CLARIFICATION ·
**Authority:** `a433b28f861631b1e7828ff247c32f3e3d37bcca` (M6D6i) · no amendment, no A9.
**Record:** `configs/amendments/e07c_m6d6ir_checkpoint_retention.yaml`
(SHA256 `94ccaad963dded8ebf903c3e07671463d51b44488074b8e7d121d074937845c6`).
E07c remains CONTROLLED_ADAPTATION, DEV-021 (`new_deviation = false`, `new_fidelity_class = false`); method status
IMPLEMENTED_NOT_EXECUTED.

## Static derivation (stdlib AST of pinned FAS_train.py + committed authority text; no torch, no data)

| Fact | Value | Source |
|---|---|---|
| TRAIN rows | 8838 | A1 `training_manifest.rows`; `M4_IDFREE_MANIFEST_AUDIT.json` |
| batch size / epochs / cadence | 4 / 400 / 10000 | argparse defaults; E07c config |
| drop_last | false | FAS_train.py:205 passes no `drop_last` |
| iterations per epoch | 2210 (last batch 2) | ceil(8838 / 4) |
| total iterations | 884000 | 2210 × 400 |
| save trigger | `iters = 0` (:30) · `iters = iters + 1` (:42) before `iters % 10000 == 0` (:102) | no save at step 0 |
| periodic checkpoints | 88 at 10000 … 880000 | |
| terminal checkpoint | 1 at 884000 (884000 % 10000 = 4000) | A2-06 clause 3 |
| created checkpoint states per seed | 89 | |
| payload | model, ema, scheduler, optimizer, conf | FAS_train.py:105 `torch.save` |
| index required fields | path, epoch, global_step, file_size_bytes, sha256, checkpoint_type, selected_for_final, selection_reason | run_logging_v1 |

## Decision (prospective; E07c main scientific runs, seeds 42/1337/2026 only)

- Creation, cadence, payload and selection (BASELINE_FINAL_STATE_V1; final = selected = terminal) unchanged.
- Non-protected periodic bytes may be pruned explicitly, only after the direct successor is created, closed,
  recorded, SHA256-hashed, durably indexed and verified. 880000 stays until the 884000 terminal is verified.
- Terminal checkpoint permanently retained; one physical file may carry the final/selected/officially-required roles.
- Index records are never removed; pruned records gain `bytes_present: false`, `bytes_pruned: true`, `pruned_utc`,
  `prune_reason: EXPLICIT_E07C_PERIODIC_RETENTION_POLICY`, successor global_step and SHA256.
- Silent pruning FORBIDDEN; fail-closed (keep predecessor; STOP_AND_REPORT on prune failure or insufficient space).
- Implementation belongs to M6D6j; MAIN_CHECKPOINT_RESUME is not qualified by this decision.

## Storage rationale — RESOURCE_PLANNING_ESTIMATE (non-binding)

Previous feasibility verdict GO_IF_OWNER_APPROVES_RETENTION_CHANGE: mature checkpoint ~2.476 GB; strict retain-all
~220 GB per seed, ~661 GB for three seeds; ~443.5 GB available on the only usable GPU data filesystem. Under this
policy: peak ~12.4 GB, retained terminals ~7.4 GB. The decision semantics, not these mutable figures, are authoritative.

## Operation

No training, no scientific or qualification run, 0 optimizer steps, 0 backward calls, 0 checkpoint creations /
deletions / moves / deserializations, no TRAIN/VAL/TEST access, no M8 output, no GPU contact.

## Statuses

Qualified: E07c_MAIN_CHECKPOINT_RETENTION_POLICY_FROZEN (alias MAIN_CHECKPOINT_RETENTION_POLICY_FROZEN).
Not qualified: MAIN_PRODUCTION_RUNNER, MAIN_CHECKPOINT_RESUME, MAIN_DIFFFAS_SCIENTIFIC_TRAINING, M8_BANK.
M6D6i statuses remain historical facts and are not re-qualified. Next: M6D6j.
