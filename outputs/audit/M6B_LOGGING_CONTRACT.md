# M6B — Run Logging Contract (`run_logging_v1`)

**Frozen 2026-09-21.** Machine-readable form: `configs/run_logging_v1.yaml`.
Binding for every M6 baseline run: **E01, E02, E03, E04, E05, E06c, E07c**.

The owner explicitly requires preservation of **all** experimental trajectories. Two rules govern
everything below:

1. **Never keep only the final epoch.** The complete history must survive training.
2. **Never fabricate a metric a method does not expose.** Missing or non-applicable fields are
   written as **explicit nulls with a stated reason** — never silently omitted.

---

## 1. Run identity — `run_manifest.json`

| Field | Notes |
|---|---|
| `method_id`, `experiment_seed` | |
| `run_id` | deterministic: `sha256(UTF-8(method_id + "\|" + str(seed) + "\|" + config_sha256 + "\|" + git_commit))`, first 16 hex chars |
| `run_uuid` | uuid4, for cross-reference |
| `git_commit`, `git_dirty` | |
| `config_path`, `config_sha256` | |
| `source_commits` | map `{source_name: pinned_commit}` |
| `host`, `gpu_model`, `gpu_count` | |
| `cuda_version`, `cudnn_version` | |
| `pytorch_version` | or `tensorflow_version` for E03 |
| `python_version` | |
| `dependency_fingerprint`, `environment_lock_path` | sha256 of the resolved environment lock |
| `start_utc`, `end_utc` | |
| `command_line` | |

The fully resolved config is written beside it as `resolved_config.yaml`.

## 2. Per-epoch / per-iteration history — `metrics.jsonl`

JSON Lines, **append-only**, one record per configured reporting interval.

`epoch` · `global_step` · `learning_rate` · `train_losses` (object: every loss the method exposes,
by name) · `train_metrics` · `val_losses` · `val_metrics` · `wall_clock_seconds` ·
`gpu_memory_bytes` (null + reason where impractical).

- `keep_only_final_epoch`: **FORBIDDEN**
- `full_history_must_survive_training`: **true**
- Resume: **append or explicitly reconcile**; overwriting earlier history is **FORBIDDEN**, and a
  reconciliation record is required.
- VAL is **diagnostic only** and may not select a checkpoint. TEST **must not appear** in the
  trajectory at all.

## 3. Non-learned methods (E01, E02) — `generation_log.jsonl`

E01 and E02 have no epochs. **Inventing epochs is FORBIDDEN.** They log per generated sample:

`method_id` · `experiment_seed` · `config_sha256` · `pair_or_sample_id` ·
`deterministic_pair_seed` · `operator_or_block_selection_metadata` · `generation_status` ·
`failure_reason` (null on success) · `output_path` · `output_sha256`

Run level: `wall_clock_seconds`, `output_manifest_path`, `run_summary`.

- **E01 additionally:** `selected_operator_name`, `magnitude_level`, `sampled_operator_parameters`,
  `selected_asset_filename`, `asset_enumeration_rule`.
- **E02 additionally:** `n_eligible`, `k_selected`, `selected_block_indices_sorted_ascending`,
  `numpy_version`, `pcg64_seed_integer`.

## 4. Checkpoint history — `checkpoint_index.json`

**Every checkpoint actually written** is recorded:

`path` · `epoch` · `global_step` · `file_size_bytes` · `sha256` · `checkpoint_type` ·
`selected_for_final` · `selection_reason`

`checkpoint_type` ∈ {`periodic`, `official`, `terminal`, `selected`}. The `selection_reason` must
cite the exact frozen rule that made a checkpoint selected — or state why it was not.
**TEST never selects.**

## 5. Final run summary — `run_summary.json`

`method_id` · `experiment_seed` · `run_id` · `final_or_selected_checkpoint_path` ·
`final_or_selected_checkpoint_sha256` · `checkpoint_selection_rule` ·
`seed_level_evaluation_metrics` · `training_duration_seconds` · `peak_vram_bytes` (null + reason
where unavailable) · `completion_status` ∈ {`completed`, `failed`, `interrupted`,
`resumed_completed`} · `failure_reason`.

## 6. Retention — two distinct concepts

| Concept | Policy |
|---|---|
| **A. Metric history** | `RETAIN_COMPLETE` — the whole trajectory, always |
| **B. Checkpoint bytes** | retained per each method's official cadence / frozen execution rule |

**Never deleted:** the authoritative final checkpoint, the selected checkpoint, and any
officially-required checkpoint. Intermediate periodic checkpoints follow the source or frozen
cadence. **M6B prunes nothing.** If a storage policy later prunes redundant intermediate checkpoint
bytes, their **metadata and SHA256 records must remain** in `checkpoint_index.json`. Silent pruning
is **FORBIDDEN**.

## 7. Output layout

```
<runtime_root>/runs/m6/<method_id>/seed_<seed>/
    resolved_config.yaml
    run_manifest.json
    metrics.jsonl
    checkpoints/
    checkpoint_index.json
    run_summary.json
    stdout.log
    stderr.log

<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/     # the single auxiliary encoder
```

Training outputs and checkpoint bytes in Git: **FORBIDDEN**. The repository keeps compact
provenance/audit summaries only.

## 8. Result aggregation

Seeds `[42, 1337, 2026]`. Each seed yields one authoritative final result under that method's frozen
checkpoint rule. **Primary result = mean and standard deviation over all three**, with per-seed
values reported.

**FORBIDDEN:** max over seeds as primary · best seed as primary · dropping a poor seed · rerunning
only a poor seed until it improves · selecting seeds using TEST.

A "best seed" may be used **only** for an explicitly-labelled qualitative visualisation, and never
replaces or influences the three-seed primary quantitative result.

**Checkpoint selection happens WITHIN each seed**; all three seed-level results then remain in the
aggregation. These two operations must never be conflated.

## 9. Data firewall

`TRAIN` — training/fitting where allowed. `VAL` — diagnostics and only frozen selection behaviour.
`TEST` — never training, never checkpoint selection, never hyperparameter selection, never
reconstruction choice, never seed selection. `test_code_path_in_m6: ABSENT`.
