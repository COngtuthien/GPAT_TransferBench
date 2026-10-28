# E03 seed1337: scientific run completed cleanly

**Status:** `E03_SEED1337_COMPLETED_CLEANLY`, 88/88 gates passing. The machine-readable record is `E03_SEED1337_COMPLETION.json`; it embeds both read-only host collections, so `tools/e03_seed_completion_audit.py build` can re-derive it.

This commit is reporting evidence only. The E03 scientific training authority stays **`d547fc9f8932a273d719ae589aade85a9d989b85`**.

| | |
|---|---|
| method | E03 **STDN**, track A_FAIR_COMMON_IDENTITY_FREE |
| execution fidelity | `FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY` (not a fully native reproduction) |
| run | seed = 1337, run_id = **`e8c85cc378e778f5`** = sha256("E03\|1337\|08dde851…\|d547fc9…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`: one run_start, command `tools/run_e03.py --seed 1337 --execution-config configs/execution/m5_gpu_3090.yaml` |
| window | 2026-10-27T03:15:48Z → 2026-10-27T12:45:59Z, 34209 s of training, median 0.226 s per step |
| exit | `~/workdir/e03_s1337.exit` = 0; tmux `e03_stdn_seed1337` pane dead with status 0; no runner process; stdout/stderr/generation_log empty |

## Completion accounting

- **Epochs = 50**, **TRAIN steps = 100000**, **VAL steps = 25000** in 50 blocks of 500.
- **G applications = 100000**, **D applications = 50000**, **final shared global_step = 150000**. Every TRAIN record matches the official schedule: even steps run g_op + d_op (+2), odd steps g_op (+1).
- **Lifecycle order:** metrics.jsonl is exactly run_start, then for each epoch 2000 TRAIN, the ckpt-e event, 500 VAL. VAL never changes global_step.
- **Learning rate:** 6e-5 × 0.9^⌊global_step/20000⌋ on the shared global_step; the largest relative float32 read-back error is 2.5e-07.
- **Losses:** all g/d/a losses are finite in every TRAIN and VAL record.

Loss means (a_loss / d_loss / g_loss), diagnostics only:

| | epoch 1 | epoch 50 |
|---|---|---|
| TRAIN | 9.041 / 0.7836 / 87.48 | 5.513 / 0.4228 / 31.69 |
| VAL (diagnostics only) | 8.755 / 0.7644 / 82.02 | 6.935 / 0.5529 / 49.93 |

## Data feed (owner rulings V, R1, R2)

- **Seeds:** TRAIN parser/shuffle = 1337; VAL parser/shuffle = 1050417158 (`1 + (u32_be(SHA256(UTF8("GPAT_TRANSFERBENCH|E03|VAL_DATA|" + str(seed))))[0:4] mod 2147483647)`); graph seed = 1337.
- **Pair-row streams:** all 200000 TRAIN and 50000 VAL rows equal a separate CPU-only TF replay of the shuffle ops; every full pass is a fresh permutation (reshuffle_each_iteration=True).
- **Flips:** 200000/200000 TRAIN and 50000/50000 VAL pairs equal an independent numpy RandomState prediction, so VAL never perturbed the TRAIN stream.
- **Faces:** 12668 TRAIN and 2653 VAL samples consumed; split_v1, labels, datasets and M2 face sha256 match for all.
- **Feed pack:** manifest `c8c9850edf783a7fce392c934e721feb29a0b88a9223612a39bd33ee46813833`, payloads verified, 0 TEST rows.

## Checkpoints (OFFICIAL_LATEST_FINAL_CKPT_50)

- ckpt-1 … ckpt-50 (150 files) exist and equal the index and the checkpoint events; each event lies between its epoch's TRAIN and VAL blocks.
- Every checkpoint holds 747 finite variables with global_step = 3000·epoch; the 170 trainable variables match the M6D2b oracle and change between consecutive checkpoints.
- The TF Saver state names ckpt-50 as latest. The final checkpoint is **ckpt-50**:
  - `.data-00000-of-00001` **`316c94c066c32314c6819bc105d0043e1ddc7b27329afd48ddcfb8bc8d181551`**
  - `.index` `29b36079f28c557ab9a44daa8dedb85770ed9d32c5cd0b04d002f39ae386ab70`
  - `.meta` `a6119bdc1f830fdf0b968054ea929dce6e6e5838050ffbe06826d8e499a40e1c`
- No partial, resume or runner-state file. Full list: `E03_SEED1337_CHECKPOINT_MANIFEST.txt`.

## Firewall

Stat-only atime snapshots: pre 2026-10-27T03:15:09Z, post 2026-10-28T11:44:16Z, 20627 files. No image was opened by the audit.

| split | files | atime-sensitive | changed | changed, consumed | changed, not consumed | consumed |
|---|---|---|---|---|---|---|
| TEST | 3027 | 3027 | 0 | 0 | 0 | 0 |
| TRAIN | 14467 | 14467 | 12668 | 12668 | 0 | 12668 |
| VAL | 3121 | 3121 | 2653 | 2653 | 0 | 2653 |

**TEST = 0**: every TEST face was atime-sensitive and none changed; no consumed sample is TEST; `test_split_accessed` is false. VAL was read for diagnostics only and never selects.

## Environment (owner driver ruling)

- Driver: lock 595.84, observed **580.178.04**, `OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`; GPU model and UUID exact; no other lock difference (run_start and live re-check).
- TensorFlow 1.15.5, CUDA runtime 11080, cuDNN 8907, 59 packages equal to the lock; Python 3.8 pathlib backport disclosed.

## Run-root integrity

- 1159 files; LC_ALL=C `<path> <size> <sha256>` listing digest **`ad748419b53d4c07c54832282bbf8ca706ac5db53b9a6af2d59f767a5390095e`**, identical before and after the audit.
- metrics.jsonl `273454ca12af55debc8372b5b82a9e95dddcffdb62edbe09b51ad40a50e5e342`

## Disclosure: keyboard echo in the pane log

- The last line of `~/workdir/e03_s1337.pane.log` is the completion record preceded by 12 caret-notation cursor-key echoes (`^[[A`..`^[[D`). This is tty echo of keys pressed in an attached tmux client, not runner output; no E03 production or pinned-source code path reads stdin. `B.pane_log_clean` strips only that prefix and records the raw line.

## Boundary

No downstream metric, bank, best-epoch or best-seed logic. ckpt-50 is the final checkpoint by the frozen rule, not by VAL.
