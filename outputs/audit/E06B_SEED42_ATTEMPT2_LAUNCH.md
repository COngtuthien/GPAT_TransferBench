# E06b seed42 attempt 2: launch and epoch-1 qualification

**Status:** `E06B_SEED42_R2_RUNNING`.

This commit is evidence only. The training authority is **E06B_FIXED_AUTHORITY `d9ecbef7295fc6b41ced6a4966b49810d36a8c87`** ("E06b: fix scientific checkpoint context gate"). It applies to seeds 42, 1337 and 2026, and the GPU checkout stays pinned to it.

Machine-readable record: `E06B_SEED42_ATTEMPT2_LAUNCH.json`.

## Attempt 1 archived (not deleted)

| | |
|---|---|
| from | `<runtime_root>/runs/m6/E06b/seed_42` |
| to | `<runtime_root>/diagnostics/m6/E06B_SEED42_ATTEMPT1_d3c0ddd5dc82a4a1_CHECKPOINT_WRITER_FAILURE` |
| method | `mv -n -T`, a same-filesystem rename(2) to a destination that was absent beforehand |
| files | 14 before, 14 after |
| path/size/sha256 listing (`LC_ALL=C`) | `8ce8d5a8…f6ca` before = after |
| path/size/mtime listing (`LC_ALL=C`) | `0498f9c9…4dcc` before = after |

The canonical root was absent after the move. The attempt-1 traceback is kept in `~/workdir/e06b_s42_attempt1_tmux_pane.txt` and in the dead pane `m6_e06b_s42`.

## Attempt 2

- **run_id:** `aa96129289d080ba`, computed independently as `sha256("E06b|42|9d665dc2…417f|d9ecbef7…8c87")[:16]`. The manifest agrees.
- **Start:** 2026-10-21T17:18:10Z, fresh from epoch 1 / step 0. There is no resume, no copied weights and no optimizer state.
- **Process:** tmux `m6_e06b_s42_r2`, PID 2087837. The shell exit code goes to `~/workdir/e06b_s42_r2.exit`.
- **Preflight:** exit 0, `torch_imported` false, commit d9ecbef, fidelity `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`. The output is identical to attempt 1's apart from `git_commit`.

### Epoch 1: 21/21 gates pass

- **Steps and rows:** 16 steps, split as 15 × 240 + 120, with micro-batches of 12 × 20 and a 6 × 20 tail. Each of the 3720 spoof rows appeared exactly once.
- **Independent predictor:** 16/16 step pair hashes match, plus the epoch order. Through epoch 2 it matches 32/32 steps and 2/2 epoch orders.
- **Pairing:** same subject for 3720/3720, every partner is live, data is CASIA+MSU TRAIN only, and classes are print 0 / replay 1 (2080 / 1640).
- **Losses and gradients:** `loss_cls` was active (7.40–9.75) and `loss_pair` was active (0.0070–0.0130). Every loss was finite, all 55 owned gradients were finite, the netCls gradient was nonzero, there was 1 optimizer step per batch, mode was SCIENTIFIC, and VAL was null.
- **Determinism:** epoch 1 is bit-identical to attempt 1 in pair hashes, ε hashes, per-step losses, epoch order and visualization PNG sha256.

### Epoch-1 checkpoint writer: success

This was the first time the fixed gate ran in the real scientific lifecycle.

| file | bytes | sha256 |
|---|---|---|
| `netE_spoof_model_epoch_1_iter_0.pth` | 55 181 976 | `40ba3e4c61938f402aa07f718b71de60337ec668b1296003579054d012b1fd8e` |
| `netE_live_model_epoch_1_iter_0.pth` | 46 792 321 | `dec2f9cb83a50974295f1c8ea8db5d87ee845786e89c45aa78d4bfdbfde80440` |
| `netG_model_epoch_1_iter_0.pth` | 79 577 546 | `b5e68e8c2e3f366f765d97d855d231234d1e6f743665e4959d1e8b889ba464e7` |

All three files are indexed with matching hash and size. Each is `periodic`, has `selected_for_final` false, and has global_step 16. No `runner_state` sidecar exists, and the `e06b_checkpoint_event` was logged with `resume_sidecar: null`.

### Firewall (stat-only atime, pre 17:18:07Z, mid 17:20:25Z)

| split | atime-sensitive | changed |
|---|---|---|
| TEST (CASIA+MSU) | 992 / 992 | **0** |
| VAL (CASIA+MSU) | 1088 / 1088 | **0** |
| SiW (all) | 5580 / 13 575 | **0** |
| TRAIN (CASIA+MSU) | 8 / 4960 | 7 |

Attempt 1 read the TRAIN files under 30 minutes earlier, so relatime does not refresh most of their atimes.

**Not started:** E06b seed1337 and seed2026, any other E06b attempt, E05/E03/E04/E01/E02, banks, downstream evaluation and GPAT.
