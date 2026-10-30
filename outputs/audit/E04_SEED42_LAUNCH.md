# E04 seed42: scientific launch and early verification

**Status:** `E04_SEED42_LAUNCHED_EARLY_VERIFICATION_PASS`. The machine-readable record is `E04_SEED42_LAUNCH.json`.

This commit is reporting evidence only. The run continues on the GPU host. **This is not a completion record.**

The GPU checkout trains from launch commit **`18ba00e94813f005876c8d6235b0b848226b2c1d`**. It was clean-fast-forwarded from `b0f0ffd` before the preflight and is not moved during training. Between `b0f0ffd` and `18ba00e`, no file under `methods/`, `tools/run_e04.py`, `configs/methods`, `configs/execution` or `environments` changed.

| | |
|---|---|
| owner decision | B, 2026-10-30: one E04 Physics-STD scientific run, seed 42 only. Seeds 1337 and 2026 are not authorized. GPU sharing with an unrelated workload is permitted for this run only. |
| authority | qualification `b0f0ffd` (`E04_PRODUCTION_QUALIFIED_READY_FOR_SCIENTIFIC_EXECUTION`, 52/52); D12 ruling `c60481b`, activation `c676459`, criteria `9f1a2471…c199d0d` |
| host | `nd-System-Product-Name`, RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04 (D9: lock 595.84, driver-version-only; environment equals the lock otherwise) |
| run | seed 42, run_id **`40befa6004aee85c`**, recomputed as sha256(`E04\|42\|418617942e…\|18ba00e…`)[:16]. Root `runs/m6/E04/seed_42` was absent before launch. Fresh run, no resume. |
| launch | 2026-10-30T02:54:08Z in tmux `e04_physics_std_seed42` (remain-on-exit, pane log `~/workdir/e04_s42.pane.log`), runner PID 1511795. Training start 02:54:14Z, iteration 0 at 02:54:17Z. Command file `~/workdir/e04_s42.command`, exit marker `~/workdir/e04_s42.exit`. |
| data | pre-pass `274a4b0a…`, feed pack `db1908d5…` (read-only). 8838 pairs, 3830 live, 8838 spoof, 12668 faces; VAL 0, TEST 0. |
| RNG | pair 679869707 (stream `0dea19a0…`), hard 320493565 (stream `ed4f32ed…`), TF graph seed 42. The full streams were re-derived independently from the rule text and equal the approved vectors. |
| graph | construction `61f1ed5868d6774a…` with 11315 ops, equal to the qualified construction; 1,763,902 trainable parameters |

## Preflight

The exact documented command with `--preflight-only` and PYTHONHASHSEED=42 was run at 02:53:40Z.

- Exit 0, with empty stderr.
- TensorFlow was not imported, no image was decoded and the graph was not executed.
- `run_dir_exists` false. No E04 directory existed under `runs/m6`, no tmux server was running and no `run_e04` process existed.
- Identities, pinned manifests and every payload hash were verified.

## GPU resources and the nonexclusive exception

| time (UTC) | GPU memory used | utilization | compute processes |
|---|---|---|---|
| 02:47:20 (first inspection) | 5140 / 24576 MiB | 94 % | PID 773932, `student20261`, miniconda `python3.14` (unrelated YOLO), 4750 MiB |
| 02:53:53 (launch check) | 376 MiB, 23746 MiB free | 0 % | none |
| 02:56:52 (early) | 4703 MiB, 19419 MiB free | 91 % | PID 1511795 (E04), 4424 MiB |

- PID 773932 exited on its own between 02:47:20Z and 02:50:41Z. It was not touched.
- No other compute process was present at launch, so the nonexclusive exception was **not needed at launch**. It remains the authority if an unrelated workload shares the GPU later.
- **TensorFlow allocation:** the qualified session config uses `allow_growth = True`. Nothing was changed: no memory fraction, no new variable.
- **E04 TF MaxBytesInUse:** 2.89 GB in this run, against 3.27–3.43 GB in the qualification records.

## Early verification

The record is at iteration 123, 2026-10-30T02:56:50Z. The independent read-only verifier is `E04_SEED42_LAUNCH_VERIFY.py`.

- There is one `e04_run_start` event and there are 124 iteration records, contiguous from 0, with `global_step = iteration + 1`.
- Pair rows and hard-sample draws equal the independently re-derived seed-42 streams, record by record.
- The G and D learning rates equal 5e-5·10^-⌊t/45000⌋ and half of that.
- All logged loss components are finite, across steps 1, 2 and 3. No VAL or TEST field is present.
- The median step time is 1.20 s per iteration.
- **Enforced in-process by the runner:**
  - the LR schedule;
  - `global_iteration` +1 after Step 3 with phase 0, so three optimizer applications per iteration in the qualified order;
  - finite step losses.
- **Not observed in the scientific run:**
  - Per-iteration gradients. No extra fetch is permitted; gradient finiteness is a qualification gate.
  - Adam slot counters. These are checked at checkpoints and in the completion audit.
- **Logs:** `stdout.log`, `stderr.log` and `generation_log.jsonl` are empty. The pane log has TF startup messages only. The int-counter colocation note in it also appears in the D12 det_a and det_b stderr.
- No checkpoint has been written yet. The first is due at iteration 15000.

## Firewall (stat-only atime, classified by split_v1)

Snapshots: pre at 02:53:53Z, mid at 02:56:09Z, 20,627 files.

| split | files | atime-sensitive | changed |
|---|---|---|---|
| TEST | 3027 | 3027 | **0** |
| VAL | 3121 | 3121 | **0** |
| TRAIN | 14467 | 1799 | 0 |

There were no mtime changes, and all 12 manifests are unmodified. The run_manifest records `test_split_accessed` false, `val_split_accessed` false and `test_code_path_present` false.

## Not done

- Seeds 1337 and 2026 were not launched.
- No E01, E02 or GPAT run, no bank, and no VAL/TEST evaluation.
- No package, driver, source or config change.
- The YOLO workload was not touched.
- No automatic retry or resume.
- The unapproved D10 draft stays preserved and inactive.
