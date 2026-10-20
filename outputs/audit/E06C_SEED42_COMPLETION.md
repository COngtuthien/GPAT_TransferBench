# E06c DSDG-BIN-IDFREE — seed 42 scientific run: completion audit

**Status: `E06C_SEED42_COMPLETED_CLEANLY`.** All 86 gates pass. Evidence: `E06C_SEED42_COMPLETION.json`, which embeds the read-only collected document and re-derives deterministically through `tools/e06c_seed42_completion_audit.py build`.

| Item | Value |
|---|---|
| Method | E06c DSDG-BIN-IDFREE |
| Fidelity / deviation | CONTROLLED_ADAPTATION / DEV-020 |
| Execution mode | GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 (nominal global 240, max microbatch 20, drop_last false) |
| Seed / run_id | 42 / `27dcaf23aac18d7a` (= sha256("E06c\|42\|7176dd4c…\|84809868…")[:16]) |
| Launch authority | `84809868acf472af5ba8cf6ab47165ae038e1012`. The GPU checkout is on branch m6-baselines with a clean worktree, and `git_dirty` is false. |
| Config SHA256 | `7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002` |
| Spec SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| TRAIN relation SHA256 | `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243` |
| Host | nd-System-Product-Name. It was reached as `student20261@10.31.131.201`. Only the network address changed: the hostname, paths and run identity are the same as at the launch over 100.121.84.44. |
| Run root | `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06c/seed_42` |
| Run window | 2026-10-18T06:44:08Z to 2026-10-19T00:39:46Z. Training took 64 538 s, with a median of 8.51 s/step. |
| Exit | `E06C_EXIT=0`. `completion_status` is completed, and the tmux session ended on its own after the exit. |

## Training completion
- Completed epoch 200, global step 7400, and 7400 per-step records. The step sequence 1..7400 is contiguous, with no duplicates and no missing steps. 7400 optimizer applications.
- All 200 epochs have 37 steps each, in the frozen plan of 36 × 240 + 1 × 198 = 8838 TRAIN pairs. Every per-step record checks out for epoch/iteration mapping, chunk plan ([20]×12 or [20]×9+[18]), LR 2e-4, SCIENTIFIC/seed 42, 55/55 finite owned gradients, and null VAL fields.
- **Data ordering:** an independent CPU prediction of the seeded DataLoader draws matches all 7400 logged batch pair-id hashes, and all 200 epoch-order hashes. Each epoch is a permutation of the 8838 TRAIN pairs, so every pair is used exactly once per epoch. All 200 `e06c_epoch_complete` events report rows = unique = 8838.
- Fresh run: one `e06c_run_start` event and no resume reconciliation. The command line has no `--resume-state`.

## Numerical health
- All 7400 records are finite: no NaN, no Inf, and no corrupted metric line. `loss_cls` and `loss_pair` are exactly 0 throughout, as the one-logit CE and lambda_pair = 0 require.
- The mean total_loss per epoch falls from 7703.4 at epoch 1 to 774.2 at epoch 200. Across the whole run, loss_rec ranges from 540.9 to 15801.8.
- All 21 engineering sidecars `runner_state_epoch_<E>.pth` were loaded with `torch.load(weights_only=True, map_location='cpu')`. In every one, netE_nir, netE_vis, netG and netCls are finite, as are the Adam `exp_avg` and `exp_avg_sq` (55 states). Adam's step equals global_step and LR is 2e-4.
- The whole-module DataParallel official `.pth` files were never unpickled. For epoch 200, their raw tensor storages (read as zip members) equal the epoch-200 sidecar tensors as sha256 multisets, for netE_spoof↔netE_nir (17), netE_live↔netE_vis (17) and netG (21).
- stderr.log is empty. stdout.log has only the 63 "Checkpoint saved to" lines, with no traceback, OOM, signal, filesystem error or checkpoint-write failure.

## Checkpoints
- There are 21 checkpoint events, at epochs 1, 10, 20, …, 200. That gives 63 official files, every one re-hashed and equal to `checkpoint_index.json` and its event record.
- **Final generator:** `checkpoints/netG_model_epoch_200_iter_0.pth`, 79 577 600 bytes, SHA256 `665684fddf11ec28b95e2a98122c85bfc5352bb8ea69ecb60caac9418c0871d9`. It is epoch 200 / step 7400, `checkpoint_type` selected, `selected_for_final` true, under the rule `OFFICIAL_GENERATOR_EPOCH_200`. It equals `run_summary.final_or_selected_checkpoint_sha256`.
- Every other official checkpoint has `selected_for_final` false: periodic before epoch 200, and terminal for netE at epoch 200.
- The 21 sidecars are ENGINEERING_RESUME_ONLY. They are not scientific checkpoints and are never selected. Each one's identity, epoch, step and SHA256 equal `resume_state_index.json`.
- The `checkpoints/` directory holds exactly the 84 indexed files. The deterministic manifest is `E06C_SEED42_CHECKPOINT_MANIFEST.txt`.

## Firewall
- **TEST benchmark access is 0.** All 3027 TEST faces were relatime-sensitive for the entire run window, and none changed atime between the launch snapshot (2026-10-18T06:43:17Z) and the post snapshot (2026-10-20T10:41:33Z, stat only).
- **VAL checkpoint-selection access is 0.** There is no VAL pass and no selection. The rule is the fixed epoch 200. None of the 3121 VAL faces changed atime, although only 468 of them were sensitive for the whole window. VAL coverage is structural: `CanonicalFaceReader` resolves only TRAIN-relation sample_ids, and both `val_split_accessed` and `test_split_accessed` are false.
- `val_pairs_v1.parquet` was untouched. All 12 668 TRAIN-relation faces changed atime inside the run window, as expected. None of the 1799 non-relation TRAIN faces changed.
- The `split_v1`/`inventory` atime at 2026-10-18T06:43:31Z was the auditor's pre-launch read. This audit re-read `split_v1` and `pairs_train_v1` only after the post snapshot.
- The upstream block named "test" writes TRAIN-sample visualization PNGs to `diagnostics/visualization`, at 21 epochs. It is not benchmark TEST.

## Run-root integrity
- 219 files. Listing digest `31609e65e6107c9a501c38113d2b9da7019790b09254e76ace7372a2680eb55c`, computed with the convention `find . -type f -printf "%P %s %T@\n" | sort | sha256sum`.
- metrics.jsonl `12fb184be3d5f655bef8549c2f5aea1b618e3a8a0ca818c221a194948b20588b`. run_manifest `9b3480c2…`, run_summary `9911ff2c…`, checkpoint_index `65eb5238…`, resume_state_index `b191865c…`, resolved_config `b18dead2…`, and final sidecar `6bc7a0d5…`. The full digests are in the JSON.
- The run root was byte-identical before and after the audit (tree plus small-file hashes). Nothing in the runtime root was written.

## Confirmations
- No VAL- or TEST-based checkpoint selection and no best-seed logic. No bank generation and no downstream evaluation.
- No checkpoints or large logs are committed.
- Scientific training authority for all three E06c seeds stays `84809868`. This closure commit is evidence and reporting only.
