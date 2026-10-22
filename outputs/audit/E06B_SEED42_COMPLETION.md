# E06b seed42: scientific training completed (attempt 2)

**Status:** `E06B_SEED42_COMPLETED_CLEANLY`, with 84/84 gates passing (`tools/e06b_seed42_completion_audit.py`). The machine-readable record is `E06B_SEED42_COMPLETION.json`, which embeds the read-only collected document so the result can be rebuilt. The checkpoint list is in `E06B_SEED42_CHECKPOINT_MANIFEST.txt`.

| | |
|---|---|
| method | E06b DSDG-NATIVE, track B_NATIVE_FULL_SECONDARY, fidelity FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY |
| successful seed | **attempt 2**: run_id `aa96129289d080ba`, seed 42 |
| training authority | **`d9ecbef7295fc6b41ced6a4966b49810d36a8c87`** (the checkpoint-context fix); clean GPU checkout; exit 0 |
| config sha256 | `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` |
| window | 2026-10-21T17:18:10Z to 20:58:11Z (13 201 s); median 3.80 s/step; peak 6.38 GB per step |
| attempt 1 | `d3c0ddd5dc82a4a1` @ 24f7049, `FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG`. Failed technical history only; still archived under `diagnostics/m6/`, untouched (digest `8ce8d5a8…`); never selected, never used for a bank or downstream |

This closure commit is evidence and reporting only. The E06b training authority stays d9ecbef for seeds 1337 and 2026.

## Training

- **Steps:** 200 epochs and 3200 step records, numbered 1 to 3200 with no gaps or duplicates. There are 3200 optimizer applications, one per step, and the run was fresh: no resume.
- **Per epoch:** 16 steps = 15 × 240 + 120, with micro-batches of 12 × 20 and a 6 × 20 tail. All 3720 CASIA+MSU TRAIN spoof rows are used exactly once (200/200 epoch-complete events).
- **Independent predictor:** a re-derivation from split_v1 without repo code, modelling torch RandomSampler and 8-worker same-subject `random.choice`. It matches **3200/3200 pair hashes** and **200/200 epoch spoof orders**.
- **Native pairing:** all 744 000 pairs are same-subject, the partner is live, the split is TRAIN and the data is CASIA/MSU only. Classes are print=0 (416 000) and replay=1 (328 000).
- **Loader:** batch 240, workers 8, persistent_workers false, drop_last false, shuffle true. Precision: FP32 with TF32 off and cudnn deterministic.

## Numerical health

- Every metric line parses. All losses are finite, with no NaN or Inf, and 55/55 owned gradients are finite at every step.
- `loss_cls` (λ_type·CE, K=2) was active throughout, ranging 0.767–9.748. `loss_pair` (λ_pair=5) was active, ranging 0.0070–0.0163.
- The netCls gradient was nonzero at every step (min 258). netCls is intentionally outside the optimizer.
- Epoch-mean `total_loss` fell from 8303.9 (epoch 1) to 2074.1 (10), 720.8 (100) and 560.2 (200).
- All 63 official `.pth` files were checked without importing their classes: every float storage was read raw, 952 773 822 elements in total, all finite.
- stderr is empty. stdout holds only the 63 "Checkpoint saved" lines. There are 0 hits for Traceback, LearnedRunContext/AttributeError, OOM, CUDA, signal, Errno, DataLoader or worker errors. **The attempt-1 checkpoint bug did not recur.**

## Checkpoints

- **Events and files:** 21 events (epoch 1, then 10, 20, …, 200), each with 3 official files: **63 files**, all on disk with sha256 and size matching the index.
- **Sidecars:** 0 runner_state / resume sidecars, and no resume index.
- **Final generator:** `checkpoints/netG_model_epoch_200_iter_0.pth`.
  - 79 577 600 bytes, sha256 **`d56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa`**.
  - Epoch 200, step 3200, `selected`, `selected_for_final` true, rule OFFICIAL_GENERATOR_EPOCH_200.
  - It equals the run summary.
- **Other files:** the epoch-200 netE files are `terminal` and not selected. Every other file is `periodic` and not selected.
- There is no VAL selection, no TEST selection, no best epoch and no best-seed logic.

## Firewall

The firewall check uses stat-only atime snapshots: pre at 17:18:07Z and post at 2026-10-22T01:26:26Z, both covering 20 627 files.

| class | files | atime-sensitive | changed |
|---|---|---|---|
| CASIA+MSU TEST | 992 | 992 | **0** |
| CASIA+MSU VAL | 1088 | 1088 | **0** |
| SiW (TRAIN/VAL/TEST) | 13 575 | 5 580 | **0** |
| CASIA+MSU TRAIN | 4960 | 8 | 8 |

**TEST 0, VAL 0, SiW 0.** No manifest changed (val_pairs included), and no face file mtime changed. Visualization is the upstream TRAIN last-batch diagnostic only.

## Run-root integrity

There are 197 files. The run root was unchanged across the whole audit: the tree, the small-file hashes and the sha listing all match before and after.

| | sha256 |
|---|---|
| path/size/sha256 listing (LC_ALL=C) | `71f8f357a563cf9b2b691490073309756cb33029dd0db1a9199a101922575575` |
| path/size/mtime listing (LC_ALL=C) | `6f71e3d8ee4093dddc7907dd0db02bb209353b7d707f20f8aaa71f221b24be75` |
| run_manifest.json | `2cd9bd080aa3b212cb4ed4b70f372b245878e887110d3fbdccbe7662d140c6f7` |
| run_summary.json | `f9e1b56f8f25fb26fe98ab7185de8edeb611c729ee504f2543b1593d77e1a8dc` |
| metrics.jsonl | `bcdbefd87d352ead75a158e7324c189ca7499b7e946be619b8af23524f11099d` |
| resolved_config.yaml | `0eadd91f7eb4a9d13eb48e09b24c694f9ec0a30f76626e00305fe8d6ee0131b2` |
| checkpoint_index.json | `d08775373e125c883211efeb91e5000ecbe35746f524bdc2914029446f14bd9c` |
| final netG epoch 200 | `d56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa` |
