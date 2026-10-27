# E03 seed42: scientific run completed cleanly

**Status:** `E03_SEED42_COMPLETED_CLEANLY`, 88/88 gates passing. The machine-readable record is `E03_SEED42_COMPLETION.json`; it embeds both read-only host collections, so `tools/e03_seed_completion_audit.py build` can re-derive it.

This commit is reporting evidence only. The E03 scientific training authority stays **`d547fc9f8932a273d719ae589aade85a9d989b85`**.

| | |
|---|---|
| method | E03 **STDN**, track A_FAIR_COMMON_IDENTITY_FREE |
| execution fidelity | `FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY` (not a fully native reproduction) |
| run | seed = 42, run_id = **`a565b76bc7349532`** = sha256("E03\|42\|08dde851…\|d547fc9…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`: one run_start, command `tools/run_e03.py --seed 42 --execution-config configs/execution/m5_gpu_3090.yaml` |
| window | 2026-10-25T10:56:49Z → 2026-10-25T17:55:01Z, 25091 s of training, median 0.212 s per step |
| exit | `~/workdir/e03_s42.exit` = 0; tmux `e03_stdn_seed42` pane dead with status 0; no runner process; stdout/stderr/generation_log empty |

## Completion accounting

- **Epochs = 50**, **TRAIN steps = 100000**, **VAL steps = 25000** in 50 blocks of 500.
- **G applications = 100000**, **D applications = 50000**, **final shared global_step = 150000**. Every TRAIN record matches the official schedule: even steps run g_op + d_op (+2), odd steps g_op (+1).
- **Lifecycle order:** metrics.jsonl is exactly run_start, then for each epoch 2000 TRAIN, the ckpt-e event, 500 VAL. VAL never changes global_step.
- **Learning rate:** 6e-5 × 0.9^⌊global_step/20000⌋ on the shared global_step; the largest relative float32 read-back error is 2.5e-07.
- **Losses:** all g/d/a losses are finite in every TRAIN and VAL record.

Loss means (a_loss / d_loss / g_loss), diagnostics only:

| | epoch 1 | epoch 50 |
|---|---|---|
| TRAIN | 9.154 / 0.7769 / 86.69 | 5.594 / 0.4138 / 32.09 |
| VAL (diagnostics only) | 9.468 / 0.7386 / 91.16 | 6.957 / 0.5904 / 47.51 |

## Data feed (owner rulings V, R1, R2)

- **Seeds:** TRAIN parser/shuffle = 42; VAL parser/shuffle = 1355875380 (`1 + (u32_be(SHA256(UTF8("GPAT_TRANSFERBENCH|E03|VAL_DATA|" + str(seed))))[0:4] mod 2147483647)`); graph seed = 42.
- **Pair-row streams:** all 200000 TRAIN and 50000 VAL rows equal a separate CPU-only TF replay of the shuffle ops; every full pass is a fresh permutation (reshuffle_each_iteration=True).
- **Flips:** 200000/200000 TRAIN and 50000/50000 VAL pairs equal an independent numpy RandomState prediction, so VAL never perturbed the TRAIN stream.
- **Faces:** 12668 TRAIN and 2653 VAL samples consumed; split_v1, labels, datasets and M2 face sha256 match for all.
- **Feed pack:** manifest `c8c9850edf783a7fce392c934e721feb29a0b88a9223612a39bd33ee46813833`, payloads verified, 0 TEST rows.

## Checkpoints (OFFICIAL_LATEST_FINAL_CKPT_50)

- ckpt-1 … ckpt-50 (150 files) exist and equal the index and the checkpoint events; each event lies between its epoch's TRAIN and VAL blocks.
- Every checkpoint holds 747 finite variables with global_step = 3000·epoch; the 170 trainable variables match the M6D2b oracle and change between consecutive checkpoints.
- The TF Saver state names ckpt-50 as latest. The final checkpoint is **ckpt-50**:
  - `.data-00000-of-00001` **`5b337a63db63e405bf35bceefdacc101060cb7d1625b5a4d2d389b963e311753`**
  - `.index` `b6b9a0c4cfd6470959453cdb861341bcacff7047b66fb414f8dff504452dc899`
  - `.meta` `dbcf372027172396c616ba19f7d76475363d34e8bac0f438c0c7f5b97fae5f97`
- No partial, resume or runner-state file. Full list: `E03_SEED42_CHECKPOINT_MANIFEST.txt`.

## Firewall

Stat-only atime snapshots: pre 2026-10-25T10:56:13Z, post 2026-10-27T03:02:02Z, 20627 files. No image was opened by the audit.

| split | files | atime-sensitive | changed | changed, consumed | changed, not consumed | consumed |
|---|---|---|---|---|---|---|
| TEST | 3027 | 3027 | 0 | 0 | 0 | 0 |
| TRAIN | 14467 | 6598 | 4799 | 4799 | 0 | 12668 |
| VAL | 3121 | 2925 | 2457 | 2457 | 0 | 2653 |

**TEST = 0**: every TEST face was atime-sensitive and none changed; no consumed sample is TEST; `test_split_accessed` is false. VAL was read for diagnostics only and never selects.

## Environment (owner driver ruling)

- Driver: lock 595.84, observed **580.178.04**, `OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`; GPU model and UUID exact; no other lock difference (run_start and live re-check).
- TensorFlow 1.15.5, CUDA runtime 11080, cuDNN 8907, 59 packages equal to the lock; Python 3.8 pathlib backport disclosed.

## Run-root integrity

- 1159 files; LC_ALL=C `<path> <size> <sha256>` listing digest **`a7bed081022c09ab753ae359536e71a964966a484e45f435d51a5809b509b163`**, identical before and after the audit.
- metrics.jsonl `059c266e18cc118dd70a40c37fd65c39b2d37021af1c04b8d3f0c97573e8f000`

## Boundary

No downstream metric, bank, best-epoch or best-seed logic. ckpt-50 is the final checkpoint by the frozen rule, not by VAL.
