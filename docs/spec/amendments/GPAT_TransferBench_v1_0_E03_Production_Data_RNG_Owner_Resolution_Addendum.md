# GPAT-TransferBench v1.0: E03 STDN production data and RNG owner resolution (addendum)

| | |
|---|---|
| Status | **OWNER_RESOLVED**, 2026-10-25, before any E03 implementation or scientific training result |
| Method | E03 STDN, track A_FAIR_COMMON_IDENTITY_FREE, source `yaojieliu/ECCV20-STDN@c79f1f8c615d2b8471b3df29da881bb18dd54c90` |
| Machine-readable form | `configs/amendments/e03_production_data_rng_resolution.yaml` |
| Investigation evidence | `outputs/audit/E03_PRODUCTION_OWNER_RULINGS.{md,json}` |
| Unchanged | `configs/methods/e03_stdn.yaml` (`08dde851…`) and its snapshot; the M6D2b runtime compatibility addendum (`f5133f81…`); the source cache |

These rulings were made before implementation. They must not be reinterpreted because of any later result.

## Resolved execution fidelity

**`FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY`**

The frozen field `fidelity_class: FAITHFUL_OFFICIAL` stays as frozen history. E03 scientific execution must not be described only as `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`.

Reports keep three layers apart:

1. **Official, unchanged:**
   - the STDN graph, losses, optimizer graph, and BN and EMA behaviour;
   - the hard-sample mechanism and the G/D schedule;
   - the shared global_step semantics and learning-rate rule;
   - the checkpoint cadence and the ckpt-50 final rule;
   - 50 epochs × 2000 TRAIN steps, and 500 VAL steps per epoch.
2. **Controlled benchmark adaptation:**
   - the frozen Track-A frame-pair relations;
   - M2 canonical faces and `landmarks_px256`;
   - independent deterministic TRAIN and VAL data RNG;
   - a serialized deterministic feed.
3. **Runtime compatibility:** the qualified NVIDIA TF 1.15.5 runtime on the RTX 3090 (M6D2b).

## Ruling V: TRAIN/VAL randomness, `SEPARATE_DETERMINISTIC_SPLIT_RNG_STREAMS`

The upstream parser shares the process-global Python and NumPy RNG between `parse_fn` (TRAIN) and `parse_fn_val` (VAL) (`model/dataset.py:107/117/126/136` and `:169/180/189/200`). It also maps and prefetches with AUTOTUNE (`:67-71`).

Running VAL therefore changes later TRAIN augmentation, and with parallel prefetch the effect depends on timing. That conflicts with the benchmark determinism contract and the rule that VAL is diagnostics-only, so the coupling is **not** retained. This is a `CONTROLLED_BENCHMARK_DETERMINISM_ADAPTATION`, not official STDN behaviour.

- TRAIN and VAL data randomness are independent. VAL never advances any stream that controls future TRAIN pair order, flips, parser augmentation or any other controller data choice.
- **TRAIN parser seed** = `experiment_seed`.
- **VAL parser seed** = `1 + (u32 mod 2147483647)`, where `u32` is the first 4 bytes, read as an unsigned big-endian integer, of `SHA256(UTF8("GPAT_TRANSFERBENCH|E03|VAL_DATA|" + str(experiment_seed)))`. The result lies in the positive 31-bit range [1, 2³¹−1]. It never uses Python `hash()`, the pid, time, filesystem order or worker timing.

  | experiment seed | u32 | VAL parser seed |
  |---|---|---|
  | 42 | 1355875379 | **1355875380** |
  | 1337 | 3197900804 | **1050417158** |
  | 2026 | 1242808124 | **1242808125** |
  | 60903 (qualification only) | 2050698080 | **2050698081** |

- The streams are independent RNG objects (`numpy.random.RandomState(seed)` per split), never repeated reseeding or mutation of the process-global Python/NumPy RNG.
- No dummy `random.randint` draws imitate the upstream frame-choice RNG advance, because R1/R2 remove frame choice. The distributions of the retained stochastic augmentations are preserved; RNG call counts are not.

## Ruling R1: TRAIN relation is `manifests/pairs_train_v1.parquet`

- **Relation:** `pairs_train_v1` (sha256 `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243`, 8838 rows) is the authoritative Track-A relation, under Amendment A1 / `fair_track_v1` (E03 ∈ FAIR_COMMON_IDENTITY_FREE).
- **Mapping:** `target_live_id` → the STDN live input, and `source_spoof_id` → the STDN spoof input.
- **Not used:**
  - the upstream directory positional zip;
  - video-level or identity grouping (`video_id` is not an identity);
  - any newly generated relation.
- This is a controlled Track-A data adaptation.

## Ruling R2: VAL relation is `manifests/val_pairs_v1.parquet`

- **Relation:** sha256 `84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872`, 1905 rows: CASIA-FASD 576, MSU-MFSD 240, SiW-Mv2 1089.
- **Mapping:** `target_live_id` → live, and `source_spoof_id` → spoof.
- **Use:** 500 diagnostic steps per epoch. VAL is never used to select a checkpoint, tune a hyperparameter or choose a seed. TEST is never used.

## Ruling R3: diagnostic figures are retained

- **What is written:** the official figure writes, at TRAIN `step % 200 == 0` (`train.py:181-183`, `LOG_FR_TRAIN = 2000/10`) and VAL `step % 50 == 0` (`:201-203`, `LOG_FR_TEST = 500/10`), with zero-based steps.
- **Where:** under the run root, as diagnostics only.
- **Source of the image:** the `fig`/`fig_val` value already returned by the same official `sess.run` of that step. Writing a figure adds no `sess.run`, no iterator pull and no RNG draw, so it causes 0 optimizer applications and 0 global_step changes.

## Owner-specified execution semantics (summary)

**TRAIN schedule.** For zero-based step `s`:

| step | fetch (one `sess.run`) | applications | global_step |
|---|---|---|---|
| `s % 2 == 0` | `losses + [g_op, d_op, fig]` | 1 G + 1 D | +2 |
| otherwise | `losses + [g_op, fig]` | 1 G | +1 |

- The G+D fetch is never split into sequential runs, and no new G-before-D dependency is added.
- **Per epoch:** 2000 steps, 2000 G and 1000 D applications, global_step +3000.
- **Total:** 100 000 steps, 100 000 G, 50 000 D, final global_step 150 000.

**Learning rate.**

- `exponential_decay(6e-5, global_step, 20000, 0.9, staircase=True)`, applied on the shared global_step: boundaries at 20 000 k, about every 6.67 epochs, not every 10 epochs.

**Epoch order.**

- 2000 TRAIN steps → `saver.save(…, global_step=epoch+1)` → 500 VAL steps. So ckpt-e holds the pre-VAL state.
- ckpt-50 is written before the epoch-50 VAL block, which must still run. Only then is the run marked `completed`.
- Rule: `OFFICIAL_LATEST_FINAL_CKPT_50`. VAL never changes the selected checkpoint.

**Feed structure.** One pair row per dataset element. Each element supplies the canonical faces of `target_live_id` (live) and `source_spoof_id` (spoof) with their M2 `landmarks_px256 / 256`.

- **Pipeline:** `from_tensor_slices → shuffle(full population: 8838 TRAIN / 1905 VAL, explicit split-specific seed, reshuffle_each_iteration=True) → repeat(-1) → map(num_parallel_calls=1) → batch(2) → prefetch(1)`.
- **TF1.15 defaults:** `reshuffle_each_iteration=None` means True, and `batch` defaults to `drop_remainder=False`, which never applies under infinite repeat.
- **Iterators:** they persist across epochs and are never reset.
- **Determinism adaptation:** serialized map and fixed prefetch(1). These change neither membership, the augmentation distribution, the schedule nor the graph.

**Augmentation.**

- **TRAIN:** live and spoof each flip independently iff their split-stream NumPy draw is > 0.5. A flip maps x → 1 − x and then applies the canonical iBUG-68 `lm_reverse_list` via `methods/stdn/landmarks.py`. TRAIN also keeps the official `tf.image.random_brightness(…, 0.25)` on both images.
- **VAL:** independent flips from the VAL stream, and no brightness.

**Landmarks.** M2 `landmarks_px256` only, `[68, 2]` float32 and finite, divided by 256.

**VAL non-influence.** This must be re-proved on the real controller. Across VAL, every TRAIN-relevant state must be unchanged:

- all trainable variables, BN moving statistics, EMA variables and optimizer slots;
- the shared global_step and the checkpoint selection;
- the TRAIN iterator, the TRAIN parser RNG, and the future TRAIN pair and flip decisions.

VAL may advance only its own iterator, its own RNG and any VAL-only TF random state. Otherwise the result is `E03_VAL_BLOCK_HAS_SCIENTIFIC_STATE_EFFECT`.

**Fresh only.** E03 resume is not qualified, and scientific roots are fresh only. Upstream `Saver.restore` (`train.py:146-150`) does not count as benchmark resume.

## Implementation bindings within these rulings (implementer, recorded for audit)

- **TF graph-level seed:** `tf.set_random_seed(experiment_seed)`. It covers the unchanged graph's hard-sample `tf.random.uniform` ops and TRAIN `random_brightness`.
- **Explicit shuffle op seeds:** TRAIN `experiment_seed`, and VAL the VAL parser seed above.
- **Python `random`:** the feed no longer uses it. The process-global Python and NumPy RNG are still seeded with `experiment_seed` for completeness, and the feed uses neither.
