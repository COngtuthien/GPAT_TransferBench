# E04 seed1337: scientific launch and early verification

**Status:** `E04_SEED1337_LAUNCHED_EARLY_VERIFICATION_PASS`. The machine-readable record is `E04_SEED1337_LAUNCH.json`.

This commit is reporting evidence only. The run continues on the GPU host. **This is not a completion record.**

| | |
|---|---|
| owner decision | 2026-11-18: E04 scientific seed 1337, only after the seed-42 completion authority is committed and verified. Seed 2026 is not authorized. Concurrent GPU execution is permitted under the seed-42 resource-safety principles. |
| prerequisite | seed 42 `E04_SEED42_COMPLETED_CLEANLY` (90/90) at **`ea533db43dc3b9d0a4108364d68b2e8217c7c835`**, pushed, with remote blobs equal to the evidence SHA manifest |
| authority | launch commit **`18ba00e94813f005876c8d6235b0b848226b2c1d`**, the same as seed 42. The GPU checkout was not moved; `18ba00e..ea533db` changes only seed-42 evidence, its test and the audit tool. Qualification `b0f0ffd` (52/52); D12 ruling `c60481b`, criteria `9f1a2471…c199d0d` |
| host | `nd-System-Product-Name`, RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04 (D9: lock 595.84, driver-version-only). The kernel is now 7.0.0-43-generic; seed 42 ran on -38. Unattended upgrades on 11-03 and 11-13 changed it. The kernel is not part of the E04 lock, and run_start reports the environment equal to the lock except the driver. |
| run | seed 1337, run_id **`a42fdd2a2accaf0a`** = sha256(`E04\|1337\|418617942e…\|18ba00e…`)[:16], precomputed and equal to the manifest. Root `runs/m6/E04/seed_1337` was absent before launch. Fresh run, no resume. |
| launch | 2026-11-18T11:28:38Z in tmux `e04_physics_std_seed1337` (remain-on-exit, pane log `~/workdir/e04_s1337.pane.log`), runner PID 430330. Training start 11:28:43Z, iteration 0 at 11:28:47Z. Command file `~/workdir/e04_s1337.command` is the seed-42 wrapper with only the seed and file names changed. Exit marker `~/workdir/e04_s1337.exit`. |
| data | pre-pass `274a4b0a…`, feed pack `db1908d5…`. 8838 pairs, 12668 TRAIN faces; VAL 0, TEST 0. |
| RNG | pair 1349475396 (stream `5c18cc08…`), hard 35554372 (stream `2cdc01ca…`), TF graph seed 1337. The full streams were re-derived independently from the rule text and by the production function, and both equal the approved vectors. |
| graph | construction `61f1ed5868d6774a…`, 11315 ops, equal to the qualified construction; 1,763,902 trainable parameters |

## Preflight

The exact documented command with `--preflight-only` and PYTHONHASHSEED=1337 was run at 11:28:12Z.

- Exit 0, with empty stderr.
- TensorFlow was not imported, no image was decoded and the graph was not executed.
- `run_dir_exists` is false. `runs/m6/E04` held only `seed_42` and its lock. There was no tmux server, no `run_e04` process and no earlier `e04_s1337` host file.
- Identities and storage equal the seed-42 preflight. Pinned manifests and every payload hash were verified.

## GPU resources

| time (UTC) | GPU memory used | utilization | compute processes |
|---|---|---|---|
| 11:28:38 (launch check) | 61 MiB, 24061 MiB free | 0 % | none (P8, no slowdown reason active) |
| 11:29:37 (early) | 4493 MiB, 19629 MiB free | 82 % | PID 430330 (E04 seed 1337), 4424 MiB |

- No other compute process was present, so concurrent execution was **not used at launch**. The owner permission remains the authority if an unrelated workload shares the GPU later.
- **TensorFlow allocation:** the qualified session config, `allow_growth = True`, is unchanged.
- **E04 TF MaxBytesInUse:** 2.89 GB.

## Early verification

At iteration 28 (11:29:23Z), the independent read-only verifier `E04_SEED1337_LAUNCH_VERIFY.py` found:

- One `e04_run_start` event and 29 iteration records, contiguous from 0, with `global_step = iteration + 1`.
- Pair rows, hard-sample draws and hard input sha256 equal the independently re-derived seed-1337 streams, record by record.
- G and D learning rates equal 5e-5 and 2.5e-5.
- Every loss component of steps 1, 2 and 3 is finite. No VAL or TEST field is present.
- The median step time is 1.25 s per iteration, so the ETA is about 52 h if the GPU is not shared.
- run_start: seeding (1349475396, 35554372, 1337, 1337), the qualified structure, the D9 driver record, `environment_identity_equals_lock_except_driver` true, and feed pack `db1908d5…`.
- `stdout.log`, `stderr.log` and `generation_log.jsonl` are empty, and the pane log has no failure pattern.

## Firewall (stat-only atime, classified by split_v1)

Snapshots: pre at 11:28:38Z, mid at 11:29:37Z, 20,627 files, same paths.

| split | files | changed |
|---|---|---|
| TEST | 3027 | **0** |
| VAL | 3121 | **0** |
| TRAIN | 14467 | 315 (consumed faces, about 40 iterations × 8) |

There were no mtime changes, and all 12 manifests are unchanged.

## Not done

- Seed 42 was not rerun, and seed 2026 was not launched.
- No E01, E02 or GPAT run, no bank, and no VAL/TEST evaluation.
- No package, driver, source or config change. No unrelated GPU job was touched.
- No automatic retry or resume.
- The unapproved D10 draft stays preserved and inactive.
