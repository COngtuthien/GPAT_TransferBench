# E04 seed2026: scientific launch and early verification

**Status:** `E04_SEED2026_LAUNCHED_EARLY_VERIFICATION_PASS`. The machine-readable record is `E04_SEED2026_LAUNCH.json`.

This commit is reporting evidence only. The run continues on the GPU host. **This is not a completion record.** Seed 2026 is the final E04 scientific seed.

| | |
|---|---|
| owner decision | 2026-11-22 (local): final E04 scientific seed 2026, conditional on the seed-1337 completion audit passing and its authority being committed, pushed and verified. Concurrent GPU execution with unrelated workloads is permitted when capacity is sufficient. |
| prerequisites | seed 42 `E04_SEED42_COMPLETED_CLEANLY` (90/90) at `ea533db43dc3b9d0a4108364d68b2e8217c7c835`; seed 1337 `E04_SEED1337_COMPLETED_CLEANLY` (90/90) at **`6da67199b32710c728b9f16f89c32b0a26cad84f`**, pushed, remote head equal |
| authority | launch commit **`18ba00e94813f005876c8d6235b0b848226b2c1d`**, the same as seeds 42 and 1337. The GPU checkout was not moved; `18ba00e..6da6719` changes only seed evidence, its tests and the audit tool. Qualification `b0f0ffd` (52/52); D12 ruling `c60481b`, criteria `9f1a2471…c199d0d` |
| host | `nd-System-Product-Name`, RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04 (D9: lock 595.84, driver-version-only); kernel 7.0.0-43-generic, the same boot as seed 1337 (boot_id `5c3d5b94…`) |
| run | seed 2026, run_id **`e7100eb9a5ab0537`** = sha256(`E04\|2026\|418617942e…\|18ba00e…`)[:16], precomputed and equal to the manifest. Root `runs/m6/E04/seed_2026` was absent before launch. Fresh run, no resume. run_uuid `cef2352d-735c-4e05-804c-f314e8f9abf1` |
| launch | 2026-11-21T17:18:55Z in tmux `e04_physics_std_seed2026` (remain-on-exit, pane log `~/workdir/e04_s2026.pane.log`), pane PID 435471, runner PID 435565. Training start 17:19:00Z, iteration 0 at 17:19:07Z. Command file `~/workdir/e04_s2026.command` (sha256 `4c01a265…`) is the seed-1337 wrapper with only the seed, file names and comment changed. Exit marker `~/workdir/e04_s2026.exit`. |
| data | pre-pass `274a4b0a…`, feed pack `db1908d5…`. 8838 pairs, 12668 TRAIN faces; VAL 0, TEST 0. |
| RNG | pair 692890966 (stream `dc051324…`), hard 1053992267 (stream `563b431d…`), TF graph seed 2026. Both full 150000-iteration streams were re-derived by the rule-text reference (`tools/e04_rng_test_vectors.py`) and by `production_io.stream_hashes(2026)`, and both equal the approved vectors. |
| graph | construction `61f1ed5868d6774a…`, 11315 ops, equal to the qualified construction; 1,763,902 trainable parameters |

## Preflight

The exact documented command with `--preflight-only` and PYTHONHASHSEED=2026 was run at 17:17:53Z.

- Exit 0, with empty stderr.
- TensorFlow was not imported, no image was decoded and the graph was not executed.
- `run_dir_exists` is false. `runs/m6/E04` held only `seed_42`, `seed_1337` and their locks. There was no `e04_physics_std_seed2026` tmux session, no `run_e04` process and no earlier `e04_s2026` host file.
- Identities and storage equal the seed-1337 preflight (only the seed and derived seeds differ).

## GPU resources (concurrent execution used)

| time (UTC) | GPU memory used | free | utilization | compute processes |
|---|---|---|---|---|
| 17:18:55 (launch check) | 12558 MiB | 11564 MiB | 100 % | 3 unrelated (same account): BERT4Rec `board_ref.py` 7252 MiB; FFT-anti-spoofing `improved_ddpm` train 3512 MiB; `g_pilot.py` 1712 MiB |
| 17:19:40 (early) | 16987 MiB | 7135 MiB | 100 % | the same 3 unrelated + PID 435565 (E04 seed 2026) 4424 MiB |

- The owner permits sharing when capacity is sufficient. E04 seed 1337 needed 4424 MiB (TF MaxBytesInUse 2.89 GB), so 11.3 GiB free was sufficient. No unrelated job was killed, paused or changed.
- GPU health: P2, 73–74 °C, the only active clock-event reason is the SW power cap (0x4).
- **TensorFlow allocation:** the qualified session config, `allow_growth = True`, is unchanged. **E04 TF MaxBytesInUse:** 2.89 GB.
- The step time is slower than unshared (early median 2.11 s vs 1.25 s), so the ETA is roughly 3.5–4 days at the current sharing level. That is not a validity criterion.

## Early verification

At iteration 13 (17:19:35Z), the independent read-only verifier `E04_SEED2026_LAUNCH_VERIFY.py` (byte-identical to the seed-1337 verifier) found:

- One `e04_run_start` event and 14 iteration records, contiguous from 0, with `global_step = iteration + 1`.
- Pair rows, hard-sample draws and hard input sha256 equal the independently re-derived seed-2026 streams, record by record.
- G and D learning rates equal 5e-5 and 2.5e-5.
- Every loss component of steps 1, 2 and 3 is finite. No VAL or TEST field is present.
- run_start: seeding (692890966, 1053992267, 2026, 2026), the qualified structure, the D9 driver record, `environment_identity_equals_lock_except_driver` true, and feed pack `db1908d5…`.
- `stdout.log`, `stderr.log` and `generation_log.jsonl` are empty, and the pane log has no failure pattern.

## Firewall (stat-only atime, classified by split_v1)

Snapshots: pre at 17:18:55Z, mid at 17:19:40Z, 20,627 files, same paths.

| split | files | changed |
|---|---|---|
| TEST | 3027 | **0** |
| VAL | 3121 | **0** |
| TRAIN | 14467 | 136 (consumed faces, about 17 iterations × 8) |

There were no mtime changes, and all 12 manifests are unchanged.

## Not done

- Seeds 42 and 1337 were not rerun. No other seed or baseline was started.
- No E01, E02 or GPAT run, no bank, and no VAL/TEST evaluation.
- No package, driver, source or config change. No unrelated GPU job was touched.
- No automatic retry or resume. No automatic future completion audit is configured; closing seed 2026 needs a later session (`tools/e04_seed_completion_audit.py` needs `RUN_IDS[2026] = 'e7100eb9a5ab0537'` and the `e04_s2026_atime_post` snapshot first).
- The unapproved D10 draft stays preserved and inactive.
