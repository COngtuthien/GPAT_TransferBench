# E06b seed1337: scientific training completed

**Status:** `E06B_SEED1337_COMPLETED_CLEANLY`, with 84/84 gates passing (`tools/e06b_seed1337_completion_audit.py`, a copy of the seed42 tool with only the seed constants, parent gate and notes changed). The machine-readable record is `E06B_SEED1337_COMPLETION.json`, which embeds the read-only collected document so the result can be rebuilt. The checkpoint list is in `E06B_SEED1337_CHECKPOINT_MANIFEST.txt`.

| | |
|---|---|
| method | E06b DSDG-NATIVE, track B_NATIVE_FULL_SECONDARY, fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY |
| run | run_id `9ffda3ea3f6cc767`, seed 1337, a **single fresh scientific run** (no earlier seed1337 attempt, no resume) |
| training authority | **`d9ecbef7295fc6b41ced6a4966b49810d36a8c87`** (the checkpoint-context fix); clean GPU checkout; exit 0 |
| config sha256 | `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` |
| window | 2026-10-22T01:33:11Z to 05:12:28Z (13 157 s); median 3.79 s/step; peak 6.38 GB per step |
| seed42 attempt 1 | `d3c0ddd5dc82a4a1` @ 24f7049, `FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG`. **No relevance to seed1337**: seed1337 was trained at the d9ecbef fix. Attempt 1 is only re-checked as history: still archived under `diagnostics/m6/`, untouched (digest `8ce8d5a8…`), never selected |

This closure commit is evidence and reporting only. The E06b training authority stays d9ecbef, including for seed2026.

## Training

- **Steps:** 200 epochs and 3200 step records, numbered 1 to 3200 with no gaps or duplicates. There are 3200 optimizer applications, one per step, and the run was fresh: no resume.
- **Per epoch:** 16 steps = 15 × 240 + 120, with micro-batches of 12 × 20 and a 6 × 20 tail. All 3720 CASIA+MSU TRAIN spoof rows are used exactly once (200/200 epoch-complete events).
- **Independent predictor:** a re-derivation from split_v1 without repo code, modelling torch RandomSampler and 8-worker same-subject `random.choice`. It matches **3200/3200 pair hashes** and **200/200 epoch spoof orders**.
- **Native pairing:** all 744 000 pairs are same-subject, the partner is live, the split is TRAIN and the data is CASIA/MSU only. Classes are print=0 (416 000) and replay=1 (328 000).
- **Loader:** batch 240, workers 8, persistent_workers false, drop_last false, shuffle true. Precision: FP32 with TF32 off and cudnn deterministic.

## Numerical health

- Every metric line parses. All losses are finite, with no NaN or Inf, and 55/55 owned gradients are finite at every step.
- `loss_cls` (λ_type·CE, K=2) was active throughout, ranging 0.789–9.618. `loss_pair` (λ_pair=5) was active, ranging 0.0067–0.0171.
- The netCls gradient was nonzero at every step (min 258). netCls is intentionally outside the optimizer.
- Epoch-mean `total_loss` fell from 8078.6 (epoch 1) to 2106.0 (10), 717.2 (100) and 544.9 (200).
- All 63 official `.pth` files were checked without importing their classes: every float storage was read raw, 952 773 822 elements in total, all finite.
- stderr is empty. stdout holds only the 63 "Checkpoint saved" lines. There are 0 hits for Traceback, LearnedRunContext/AttributeError, OOM, CUDA, signal, Errno, DataLoader or worker errors. **The seed42 attempt-1 checkpoint bug did not recur.**

## Checkpoints

- **Events and files:** 21 events (epoch 1, then 10, 20, …, 200), each with 3 official files: **63 files**, all on disk with sha256 and size matching the index.
- **Sidecars:** 0 runner_state / resume sidecars, and no resume index.
- **Final generator:** `checkpoints/netG_model_epoch_200_iter_0.pth`.
  - 79 577 600 bytes, sha256 **`645538e866a6d835a1bc35240c2f15c27296742850df96d722fd4d38609b4f53`**.
  - Epoch 200, step 3200, `selected`, `selected_for_final` true, rule OFFICIAL_GENERATOR_EPOCH_200.
  - It equals the run summary.
- **Other files:** the epoch-200 netE files are `terminal` and not selected. Every other file is `periodic` and not selected.
- There is no VAL selection, no TEST selection, no best epoch and no best-seed logic.

## Firewall

The firewall check uses stat-only atime snapshots: pre at 2026-10-22T01:33:08Z and post at 2026-10-24T07:55:48Z, both covering 20 627 files. The auditor only stats files; no image was opened for the audit.

| class | files | atime-sensitive | changed |
|---|---|---|---|
| CASIA+MSU TEST | 992 | 992 | **0** |
| CASIA+MSU VAL | 1088 | 1088 | **0** |
| SiW (TRAIN/VAL/TEST) | 13 575 | 5 580 | **0** |
| CASIA+MSU TRAIN | 4960 | 0 | 0 |

**TEST 0, VAL 0, SiW 0.** No manifest changed (val_pairs included), and no face file mtime changed. CASIA/MSU TRAIN was read by seed42 attempt 2 less than 24 h before the pre snapshot, so under relatime its atimes were not sensitive; TEST and VAL stay fully sensitive. Visualization is the upstream TRAIN last-batch diagnostic only.

## Run-root integrity

There are 197 files. The run root was unchanged across the whole audit: the tree, the small-file hashes and the sha listing all match before and after. An independent shell listing taken before the audit (`find | LC_ALL=C sort`, then `sha256sum`) gives the same sha listing digest.

| | sha256 |
|---|---|
| path/size/sha256 listing (LC_ALL=C) | `f2cc290eb1a3b38b072958eb3cce17cdf0e290bb4715c300d073e8c249fe4f4e` |
| path/size/mtime listing (LC_ALL=C) | `c534146d36fa19c6e9cee89130341b40068a9e56a46d6d177cca677b7395ce8f` |
| run_manifest.json | `891a8a3d43e18fa7a650ac7e91361568e82cf00c6e57e71b05f2613dc3f1e786` |
| run_summary.json | `5f512a07dda198e74e987ff98f7b4127032721b5baff35ea1a4384b527fe3a02` |
| metrics.jsonl | `4ee2b6164f400c7cda6839a5cbfeeb6469ba0f469cb1859c828d6ba6f407be3d` |
| resolved_config.yaml | `9f45b231b2449cc8cf88685da068d6aa58e33198c40be41210f63b5b6841f433` |
| checkpoint_index.json | `c6416f30e690f2df15f64dbcacfdd35a867f15b14e8a393524ace8f83c5cf53f` |
| final netG epoch 200 | `645538e866a6d835a1bc35240c2f15c27296742850df96d722fd4d38609b4f53` |
