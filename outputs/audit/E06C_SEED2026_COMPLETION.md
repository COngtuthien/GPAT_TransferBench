# E06c DSDG-BIN-IDFREE — seed 2026 scientific run: completion audit

**Status: `E06C_SEED2026_COMPLETED_CLEANLY`.** All 86 gates pass. Evidence: `E06C_SEED2026_COMPLETION.json`, which embeds the read-only collected document and re-derives deterministically through `tools/e06c_seed2026_completion_audit.py build`. The tool is the seed1337 closure tool with only the seed-specific constants, the parent-listing gate (now exactly the three frozen seeds) and the notes changed. The gate semantics are identical across all three seeds.

| Item | Value |
|---|---|
| Method | E06c DSDG-BIN-IDFREE |
| Fidelity / deviation | CONTROLLED_ADAPTATION / DEV-020 |
| Execution mode | GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2 (nominal global 240, max microbatch 20, drop_last false) |
| Seed / run_id | 2026 / `86b631403e3c4ee1` (= sha256("E06c\|2026\|7176dd4c…\|84809868…")[:16]) |
| Launch authority | `84809868acf472af5ba8cf6ab47165ae038e1012`. The GPU checkout is on branch m6-baselines with a clean worktree, and `git_dirty` is false. |
| Config SHA256 | `7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002` |
| Spec SHA256 | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| TRAIN relation SHA256 | `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243` |
| Host | nd-System-Product-Name, reached as `student20261@10.31.131.201` for both launch and audit |
| Run root | `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06c/seed_2026` |
| Run window | 2026-10-21T04:55:25Z to 2026-10-21T13:25:12Z. Training took 30 587 s, with a median of 3.82 s/step. Peak step GPU memory was 6.38 GB. |
| Exit | `E06C_EXIT=0` at 2026-10-21T13:25:15Z. `completion_status` is completed, and the tmux session `m6_e06c_s2026` ended on its own (no tmux server, no runner process at audit time). |

## Training completion
- Completed epoch 200, global step 7400, and 7400 per-step records. The step sequence 1..7400 is contiguous, with no duplicates and no missing steps. 7400 optimizer applications.
- All 200 epochs have 37 steps each, in the frozen plan of 36 × 240 + 1 × 198 = 8838 TRAIN pairs. Every per-step record checks out for epoch/iteration mapping, chunk plan ([20]×12 or [20]×9+[18]), LR 2e-4, SCIENTIFIC/seed 2026, 55/55 finite owned gradients, and null VAL fields.
- **Data ordering:** an independent CPU prediction of the seeded DataLoader draws (seed 2026, torch 2.12.1) matches all 7400 logged batch pair-id hashes, and all 200 epoch-order hashes. Each epoch is a permutation of the 8838 TRAIN pairs, so every pair is used exactly once per epoch. All 200 `e06c_epoch_complete` events report rows = unique = 8838.
- Fresh run: one `e06c_run_start` event and no resume reconciliation. The command line has no `--resume-state`.

## Numerical health
- All 7400 records are finite: no NaN, no Inf, and no corrupted metric line. `loss_cls` and `loss_pair` are exactly 0 throughout, as the one-logit CE and lambda_pair = 0 require.
- The mean total_loss per epoch falls from 7658.9 at epoch 1 to 748.3 at epoch 200. Across the whole run, loss_rec ranges from 529.4 to 14886.9. These are descriptive training values, not performance claims.
- All 21 engineering sidecars `runner_state_epoch_<E>.pth` were loaded with `torch.load(weights_only=True, map_location='cpu')`. In every one, netE_nir, netE_vis, netG and netCls are finite, as are the Adam `exp_avg` and `exp_avg_sq` (55 states). Adam's step equals global_step and LR is 2e-4.
- The whole-module DataParallel official `.pth` files were never unpickled. For epoch 200, their raw tensor storages (read as zip members) equal the epoch-200 sidecar tensors as sha256 multisets, for netE_spoof↔netE_nir (17), netE_live↔netE_vis (17) and netG (21).
- stderr.log is empty. stdout.log has only the 63 "Checkpoint saved to" lines, with no traceback, OOM, signal, filesystem error or checkpoint-write failure.

## Checkpoints
- There are 21 checkpoint events, at epochs 1, 10, 20, …, 200. That gives 63 official files, every one re-hashed and equal to `checkpoint_index.json` and its event record.
- **Final generator:** `checkpoints/netG_model_epoch_200_iter_0.pth`, 79 577 600 bytes, SHA256 `dc6c54b4357a51546c3abab3b957dd4b19044ae0188dde417df03446fc951878`. It is epoch 200 / step 7400, `checkpoint_type` selected, `selected_for_final` true, under the rule `OFFICIAL_GENERATOR_EPOCH_200`. It equals `run_summary.final_or_selected_checkpoint_sha256`.
- Every other official checkpoint has `selected_for_final` false: periodic before epoch 200, and terminal for netE at epoch 200.
- The 21 sidecars are ENGINEERING_RESUME_ONLY. They are not scientific checkpoints and are never selected. Each one's identity, epoch, step and SHA256 equal `resume_state_index.json`.
- The `checkpoints/` directory holds exactly the 84 indexed files. The deterministic manifest is `E06C_SEED2026_CHECKPOINT_MANIFEST.txt` (sha256 `b38ff1be…`).

## Firewall
- **TEST benchmark access is 0.** All 3027 TEST faces were relatime-sensitive for the entire run window, and none changed atime between the launch snapshot (2026-10-21T04:54:51Z) and the post snapshot (2026-10-21T16:32:16Z, stat only via `find -printf`, same command and file order as the pre snapshot).
- **VAL checkpoint-selection access is 0.** There is no VAL pass and no selection. The rule is the fixed epoch 200. All 3121 VAL faces were sensitive for the whole window and none changed atime. `CanonicalFaceReader` resolves only TRAIN-relation sample_ids, and both `val_split_accessed` and `test_split_accessed` are false.
- `val_pairs_v1.parquet` was untouched (atime 2026-10-17, sensitive). All 12 668 TRAIN-relation faces changed atime inside the run window, as expected. Those faces were not whole-window sensitive in the pre snapshot, because seed1337 had read them less than 24 h before. None of the 1799 non-relation TRAIN faces changed.
- The `split_v1`/`pairs_train_v1` atime of 2026-10-20T10:45Z was the seed42 closure audit's read. Later reads (the seed1337 closure audit, the seed2026 launch) fell inside relatime's 24 h window, so these two manifests are not atime-sensitive for this run. This audit re-read them only after the post snapshot.
- The upstream block named "test" writes TRAIN-sample visualization PNGs to `diagnostics/visualization`, at 21 epochs. It is not benchmark TEST.

## Run-root integrity
- 219 files. Listing digest `eda61bb71195efa37a1c624c27137b1181a15e25ec05e0f9b55e0073264e360b`, computed with the convention `find . -type f -printf "%P %s %T@\n" | LC_ALL=C sort | sha256sum`. It is byte-order (C) collation, the same as the seed42/seed1337 digests, which still reproduce (`31609e65…`, `87a0bed5…`).
- metrics.jsonl `0ee1d2a790621b5e96e836b811642653ea8ccfdd27e6b2234aaf38f54f5a52f5`. run_manifest `82b0370e…`, run_summary `79fcd658…`, checkpoint_index `3964a945…`, resume_state_index `bd5863a7…`, resolved_config `ceb7c172…`, and final sidecar `1dd004d1…`. The full digests are in the JSON.
- The run root was byte-identical before and after the audit (tree plus small-file hashes, and an independent shell listing digest taken before the collector ran). Nothing in the runtime root was written. The only new host file is the stat-only `~/workdir/e06c_s2026_atime_post.{tsv,time}` snapshot.

## Confirmations
- No VAL- or TEST-based checkpoint selection and no best-seed logic. No bank generation and no downstream evaluation.
- No checkpoints or large logs are committed.
- Scientific training authority for all three E06c seeds stays `84809868`. This closure commit is evidence and reporting only.
