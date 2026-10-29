# E03 seed2026: scientific run NEEDS REVIEW

**Status:** `E03_SEED2026_NEEDS_REVIEW`, 86/88 gates passing. The machine-readable record is `E03_SEED2026_COMPLETION.json`; it embeds both read-only host collections, so `tools/e03_seed_completion_audit.py build` can re-derive it.

This commit is reporting evidence only. The E03 scientific training authority stays **`d547fc9f8932a273d719ae589aade85a9d989b85`**.

| | |
|---|---|
| method | E03 **STDN**, track A_FAIR_COMMON_IDENTITY_FREE |
| execution fidelity | `FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY` (not a fully native reproduction) |
| run | seed = 2026, run_id = **`237887eebab18637`** = sha256("E03\|2026\|08dde851…\|d547fc9…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`: one run_start, command `tools/run_e03.py --seed 2026 --execution-config configs/execution/m5_gpu_3090.yaml` |
| window | 2026-10-28T11:50:05Z → 2026-10-28T18:52:21Z, 25335 s of training, median 0.215 s per step |
| exit | `~/workdir/e03_s2026.exit` = 0; tmux `e03_stdn_seed2026` pane dead with status 0; no runner process; stdout/stderr/generation_log empty |

## Completion accounting

- **Epochs = 50**, **TRAIN steps = 100000**, **VAL steps = 25000** in 50 blocks of 500.
- **G applications = 100000**, **D applications = 50000**, **final shared global_step = 150000**. Every TRAIN record matches the official schedule: even steps run g_op + d_op (+2), odd steps g_op (+1).
- **Lifecycle order:** metrics.jsonl is exactly run_start, then for each epoch 2000 TRAIN, the ckpt-e event, 500 VAL. VAL never changes global_step.
- **Learning rate:** 6e-5 × 0.9^⌊global_step/20000⌋ on the shared global_step; the largest relative float32 read-back error is 2.5e-07.
- **Losses:** all g/d/a losses are finite in every TRAIN and VAL record.

Loss means (a_loss / d_loss / g_loss), diagnostics only:

| | epoch 1 | epoch 50 |
|---|---|---|
| TRAIN | 9 / 0.7829 / 86.73 | 5.286 / 0.4149 / 29.65 |
| VAL (diagnostics only) | 8.757 / 0.7597 / 82.52 | 6.741 / 0.5958 / 45.55 |

## Data feed (owner rulings V, R1, R2)

- **Seeds:** TRAIN parser/shuffle = 2026; VAL parser/shuffle = 1242808125 (`1 + (u32_be(SHA256(UTF8("GPAT_TRANSFERBENCH|E03|VAL_DATA|" + str(seed))))[0:4] mod 2147483647)`); graph seed = 2026.
- **Pair-row streams:** all 200000 TRAIN and 50000 VAL rows equal a separate CPU-only TF replay of the shuffle ops; every full pass is a fresh permutation (reshuffle_each_iteration=True).
- **Flips:** 200000/200000 TRAIN and 50000/50000 VAL pairs equal an independent numpy RandomState prediction, so VAL never perturbed the TRAIN stream.
- **Faces:** 12668 TRAIN and 2653 VAL samples consumed; split_v1, labels, datasets and M2 face sha256 match for all.
- **Feed pack:** manifest `c8c9850edf783a7fce392c934e721feb29a0b88a9223612a39bd33ee46813833`, payloads verified, 0 TEST rows.

## Checkpoints (OFFICIAL_LATEST_FINAL_CKPT_50)

- ckpt-1 … ckpt-50 (150 files) exist and equal the index and the checkpoint events; each event lies between its epoch's TRAIN and VAL blocks.
- Every checkpoint holds 747 finite variables with global_step = 3000·epoch; the 170 trainable variables match the M6D2b oracle and change between consecutive checkpoints.
- The TF Saver state names ckpt-50 as latest. The final checkpoint is **ckpt-50**:
  - `.data-00000-of-00001` **`9dbd463cb3eb56c9ecc578dd275e0c2507a1e6d5f8b9b4876406d5011bd52cb3`**
  - `.index` `42107c3d56f1ec2892def0840eea520e1950ced453ffa09981899ded11ed96dc`
  - `.meta` `504845f926c1940a4aa0ba647dada611dba79044cb247dc11410614464f75543`
- No partial, resume or runner-state file. Full list: `E03_SEED2026_CHECKPOINT_MANIFEST.txt`.

## Firewall

Stat-only atime snapshots: pre 2026-10-28T11:49:31Z, post 2026-10-29T07:50:35Z, 20627 files. No image was opened by the audit.

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

- 1159 files; LC_ALL=C `<path> <size> <sha256>` listing digest **`f42708341b91e6b6d11b601b363e2dd5309dd7b5df33b65c6ed79ca032e25136`**, identical before and after the audit.
- metrics.jsonl `4d30934a91bc01133af5a34e068a44e543b1debecfe8294b9458eb7de5e42236`

## Boundary

No downstream metric, bank, best-epoch or best-seed logic. ckpt-50 is the final checkpoint by the frozen rule, not by VAL.
