# E06b seed2026: scientific training completed

**Status:** `E06B_SEED2026_COMPLETED_CLEANLY`, with 84/84 gates passing (`tools/e06b_seed2026_completion_audit.py`, a copy of the seed1337 tool with only the seed constants, parent gate and notes changed). The machine-readable record is `E06B_SEED2026_COMPLETION.json`, which embeds the read-only collected document so the result can be rebuilt. The checkpoint list is in `E06B_SEED2026_CHECKPOINT_MANIFEST.txt`.

| | |
|---|---|
| method | E06b DSDG-NATIVE, track B_NATIVE_FULL_SECONDARY, fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY |
| run | run_id `ec71778b28479de5`, seed 2026, a **single fresh scientific run** (no earlier seed2026 attempt, no resume) |
| training authority | **`d9ecbef7295fc6b41ced6a4966b49810d36a8c87`** (the checkpoint-context fix); clean GPU checkout; exit 0 |
| config sha256 | `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` |
| window | 2026-10-24T08:01:12Z to 11:40:54Z (13 183 s); median 3.80 s/step; peak 6.38 GB per step |
| seed42 attempt 1 | `d3c0ddd5dc82a4a1` @ 24f7049, `FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG`. **No relevance to seed2026**: seed2026 was trained at the d9ecbef fix. Attempt 1 is only re-checked as history: still archived under `diagnostics/m6/`, untouched (digest `8ce8d5a8…`), never selected |

This closure commit is evidence and reporting only. The E06b training authority stays d9ecbef.

## Training

- **Steps:** 200 epochs and 3200 step records, numbered 1 to 3200 with no gaps or duplicates. There are 3200 optimizer applications, one per step, and the run was fresh: no resume.
- **Per epoch:** 16 steps = 15 × 240 + 120, with micro-batches of 12 × 20 and a 6 × 20 tail. All 3720 CASIA+MSU TRAIN spoof rows are used exactly once (200/200 epoch-complete events).
- **Independent predictor:** a re-derivation from split_v1 without repo code, modelling torch RandomSampler and 8-worker same-subject `random.choice`. It matches **3200/3200 pair hashes** and **200/200 epoch spoof orders**.
- **Native pairing:** all 744 000 pairs are same-subject, the partner is live, the split is TRAIN and the data is CASIA/MSU only. Classes are print=0 (416 000) and replay=1 (328 000).
- **Loader:** batch 240, workers 8, persistent_workers false, drop_last false, shuffle true. Precision: FP32 with TF32 off and cudnn deterministic.

## Numerical health

- Every metric line parses (3443 lines: 3200 step records + 243 events). All losses are finite, with no NaN or Inf, and 55/55 owned gradients are finite at every step.
- `loss_cls` (λ_type·CE, K=2) was active throughout, ranging 0.835–9.199. `loss_pair` (λ_pair=5) was active, ranging 0.0075–0.0173.
- The netCls gradient was nonzero at every step (min 258). netCls is intentionally outside the optimizer.
- Epoch-mean `total_loss` fell from 8221.7 (epoch 1) to 2037.1 (10), 714.2 (100) and 545.2 (200).
- All 63 official `.pth` files were checked without importing their classes: every float storage was read raw, 952 773 822 elements in total, all finite.
- stderr is empty. stdout holds only the 63 "Checkpoint saved" lines. There are 0 hits for Traceback, LearnedRunContext/AttributeError, OOM, CUDA, signal, Errno, DataLoader or worker errors. **The seed42 attempt-1 checkpoint bug did not recur.**

## Checkpoints

- **Events and files:** 21 events (epoch 1, then 10, 20, …, 200), each with 3 official files: **63 files**, all on disk with sha256 and size matching the index.
- **Sidecars:** 0 runner_state / resume sidecars, and no resume index.
- **Final generator:** `checkpoints/netG_model_epoch_200_iter_0.pth`.
  - 79 577 600 bytes, sha256 **`e8d0d01cfdb59b92d0f24318ab4fec042188bf0a72f9c59054ea10204a326cad`**.
  - Epoch 200, step 3200, `selected`, `selected_for_final` true, rule OFFICIAL_GENERATOR_EPOCH_200.
  - It equals the run summary.
- **Other files:** the epoch-200 netE files are `terminal` and not selected. Every other file is `periodic` and not selected.
- There is no VAL selection, no TEST selection, no best epoch and no best-seed logic.

## Firewall

The firewall check uses stat-only atime snapshots: pre at 2026-10-24T08:01:09Z and post at 2026-10-25T03:36:24Z, both covering 20 627 files. The auditor only stats files; no image was opened for the audit.

| class | files | atime-sensitive | changed |
|---|---|---|---|
| CASIA+MSU TEST | 992 | 992 | **0** |
| CASIA+MSU VAL | 1088 | 1088 | **0** |
| SiW (TRAIN/VAL/TEST) | 13 575 | 13 575 | **0** |
| CASIA+MSU TRAIN | 4960 | 4960 | 4960 (all inside the run window) |

**TEST 0, VAL 0, SiW 0.** No manifest changed (val_pairs included), and no face file mtime changed. CASIA/MSU TRAIN was last read by seed1337 more than 24 h before the pre snapshot, so this time every TRAIN face was atime-sensitive, and all 4960 were read inside the run window: positive evidence that the run did read its TRAIN scope while no other class changed. Visualization is the upstream TRAIN last-batch diagnostic only.

## Run-root integrity

There are 197 files. The run root was unchanged across the whole audit: the tree, the small-file hashes and the sha listing all match before and after. An independent shell listing taken before the audit (`find | LC_ALL=C sort`, then `sha256sum`) gives the same sha listing digest.

| | sha256 |
|---|---|
| path/size/sha256 listing (LC_ALL=C) | `7f03af4324673e9a0ba2095e7db4ff145f68a8771f566f5a53e36b36d563d69f` |
| path/size/mtime listing (LC_ALL=C) | `efd862e49ac1f05e695ee4a78de55669f350e22dfdcd3402e127402e6e48b32f` |
| run_manifest.json | `2bf32dd74aea815b295bc674e1ab83e059d521740fb3c409c8e49457985043e7` |
| run_summary.json | `a62ab78acaf32d14a3700d7d2d1d3767f3446c6c5e1a5192ed3e993b68a40cdc` |
| metrics.jsonl | `0442a4ad100d930ba7bc84ac9df36573b5250c89d4a1b244f5a91519a9dcdea0` |
| resolved_config.yaml | `0272fdf09a150301503df5cfe3ca3f847d82869e8d1fb48366e7ee84077a474a` |
| checkpoint_index.json | `9a77e7eb74181864dad2096968256a0d1ebbd56973fe5a89dffbee667d7ed3aa` |
| final netG epoch 200 | `e8d0d01cfdb59b92d0f24318ab4fec042188bf0a72f9c59054ea10204a326cad` |
