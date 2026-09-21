# M6B — Baseline Execution Config Freeze

**Date:** 2026-09-21 · **Branch:** `m6-baselines` · **Starting commit:** `cadbfa01d7fdaeade614073b638fc9393f3b26ef`

Configs and observability **only**. Nothing was trained, generated or executed; no checkpoint was
created; no GPU job ran; TEST was never touched.

| Item | SHA256 |
|---|---|
| Frozen specification | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| Amendment A1 | `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472` |
| Amendment A2 | `b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d` |
| Amendment A3 | `b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a` |

Active M6 contract/source blockers: **0**. All seven Track-A rows are now **`CONFIG_FROZEN`**.

---

## 1. The seven configs

| Method | Config path | SHA256 | Fidelity class |
|---|---|---|---|
| E01 | `configs/methods/e01_fas_aug.yaml` | `87012ea73ab4195a…` | `FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION` |
| E02 | `configs/methods/e02_freqsub.yaml` | `fdf74776cc100d5b…` | `SPEC_DEFINED` |
| E03 | `configs/methods/e03_stdn.yaml` | `08dde851bc9e8ac6…` | `FAITHFUL_OFFICIAL` |
| E04 | `configs/methods/e04_physics_std.yaml` | `418617942ebd87c4…` | **`CONTROLLED_ADAPTATION`** |
| E05 | `configs/methods/e05_pcgan.yaml` | `478756e150c42783…` | **`CONTROLLED_ADAPTATION`** |
| E06c | `configs/methods/e06c_dsdg_bin_idfree.yaml` | `7176dd4cd4900732…` | **`CONTROLLED_ADAPTATION`** |
| E07c | `configs/methods/e07c_difffas_bin_idfree.yaml` | `dba34a9222b80ed6…` | **`CONTROLLED_ADAPTATION`** |

Full hashes are in `outputs/audit/M6B_CONFIG_FREEZE.json`. Each has a **byte-identical** copy under
`frozen_config_snapshot/configs/methods/`, as spec §0.1 rule 1 and the M0 guard require.

**Nothing was retuned.** Every scientific value is copied faithfully from the frozen spec, the
official sources at their pinned commits, or Amendments A1/A2/A3. Each config carries a
`field_provenance` block tagging where every field came from (`SPEC`, `PAPER`, `OFFICIAL_CODE`,
`AMENDMENT_A1/A2/A3`, `COMPUTED`, `M6B`).

**Nothing is left implicit.** No value depends on cwd, `os.listdir` ordering, hostname, shell
environment, `PYTHONHASHSEED`, automatic checkpoint choice, TEST performance or a "best seed".

## 2. Seeds — all three, always

Every method runs the frozen benchmark seed set **`[42, 1337, 2026]`**, three runs each. E01 and E02
are non-learned but *are* stochastic: their per-pair seed derives from `global_seed`, so their banks
differ per seed and all three are produced.

`best_seed`, `preferred_seed` and `presentation_seed` are **`FORBIDDEN`** in every config.

## 3. Primary result aggregation rule (benchmark-wide)

Each seed yields **one** authoritative final result under that method's already-frozen checkpoint
rule. The primary reported number is the **mean and standard deviation over all three seed results**,
with per-seed values also reported.

**Forbidden:** `max(seed_42, seed_1337, seed_2026)` as the primary result; choosing the best seed;
dropping a poor seed; rerunning only a poor seed until it improves; selecting seeds using TEST.

A "best seed" may later be used **only** for an explicitly-labelled qualitative visualisation, and
**never** replaces the three-seed primary quantitative result.

## 4. Checkpoint selection is WITHIN seed

Checkpoint selection and seed selection are different operations. A method may select a checkpoint
**inside each seed** only by its already-frozen rule; all three seed-level results then remain in the
aggregation.

| Method | Checkpoint rule | Terminal state | Overrides official? |
|---|---|---|---|
| E01 | `NOT_APPLICABLE_NO_TRAINED_GENERATOR` | — | — |
| E02 | `NOT_APPLICABLE_NO_TRAINED_GENERATOR` | — | — |
| E03 | `OFFICIAL_LATEST_FINAL_CKPT_50` | — | **No** — A2-06 clause 1 preserves it |
| E04 | `BASELINE_FINAL_STATE_V1` | iteration **150,000** | — |
| E05 | `BASELINE_FINAL_STATE_V1` | iteration **4,000** | — |
| E06c | `OFFICIAL_GENERATOR_EPOCH_200` | — | **No** — A2-06 clause 1 preserves it |
| E07c | `BASELINE_FINAL_STATE_V1` | end of the **400-epoch** budget | — |

No validation-based model selection was invented where the source defined none: every config sets
`selection_uses_val: false`. **TEST never checkpoint-selects** (`selection_uses_test: false`).

## 5. E07c — one auxiliary encoder, not three

```
auxiliary_encoder_training_seed = 42
auxiliary_encoder_training_runs = 1
```

The controlled conditioning encoder is trained **exactly once** (later, not here), checkpointed by
the frozen final-state rule, SHA256-recorded and frozen. **That same frozen checkpoint** is then
reused for E07c main seeds **42, 1337 and 2026**, which differ only by main experiment seed.

**Three separately pretrained encoders must NOT be trained.** This preserves the role of the
unavailable original `PADISI.pkl` — one fixed external pretrained asset shared across runs. Its run
directory is `<runtime_root>/runs/m6/E07c/aux_encoder/seed_42/`.

## 6. E04 — fixed auxiliary geometry

The 3DDFA_V2 commit, model assets, Q=140 vertex list and depth renderer are **fixed reconstruction
dependencies**, reused unchanged across all three E04 experiment seeds and **never re-derived or
retrained per seed**. Only the E04 *training* process consumes `[42, 1337, 2026]`.

## 7. Data firewall (every config)

`TRAIN` — method training/fitting where allowed. `VAL` — diagnostics only; `may_select_checkpoint:
false`. `TEST` — `allowed: false`, `used_for: []`, and explicitly never training, never checkpoint
selection, never hyperparameter selection, never reconstruction choice, never seed selection, with
`code_path_present: false`. **M6B contains no code path that uses TEST.**

## 8. Non-learned methods (E01, E02)

No epochs were invented (`epochs_must_not_be_invented: true`). They log per-sample records instead:
method, seed, config SHA256, pair id, deterministic pair seed, operator/block selection metadata,
generation status, failure reason, output path and output SHA256, plus run-level wall time and a run
summary.

E01 preserves the owner-frozen seed byte layout (`UTF-8(pair_id + decimal_string(global_seed))`, no
separator, `mod 2^31`) and lexicographic asset enumeration. E02 preserves round-half-up
`k = (n_eligible + 2) // 4`, canonical row-major eligible-block order, `PCG64`, sampling without
replacement, the dedicated `gpatbench.freqsub.block.v1` namespace, and the full 32-byte digest read
as a big-endian integer seed.

## 9. Run layout

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

Training outputs and checkpoint bytes are **never** committed to Git; the repository keeps compact
provenance/audit summaries only.

## 10. Trajectory retention

`keep_only_final_epoch` is **`FORBIDDEN`**. All epoch/iteration history survives training in an
append-only `metrics.jsonl`, one record per configured reporting interval, carrying epoch,
`global_step`, learning rate, all available train/VAL losses and metrics, wall-clock time and GPU
memory where practical. Missing or non-applicable fields are **explicit nulls with a reason** — a
method is never forced to fabricate a metric it does not expose. A resume **appends or explicitly
reconciles**; it never overwrites earlier history.

**Checkpoint bytes** are retained per each method's official cadence or frozen rule. The
authoritative final, selected and officially-required checkpoints are **never** deleted. M6B prunes
nothing; if pruning is ever needed, metadata and SHA256 records must remain.

## 11. Validation

`tools/m6b_validate_configs.py` parses YAML only — it loads no model, reads no benchmark data and
touches no TEST split. Result: **PASS, 0 failures**, covering parse/uniqueness, fidelity labels,
seeds, the E07c auxiliary-seed rule, the E04 fixed-geometry rule, all checkpoint rules against the
amendments, the TEST firewall, source pins, the no-weights-from-`source_cache` rule, and
snapshot byte-identity.

## 12. Status

All seven rows: **`CONFIG_FROZEN`**. Not `TRAINED`. Not `COMPLETE`.
