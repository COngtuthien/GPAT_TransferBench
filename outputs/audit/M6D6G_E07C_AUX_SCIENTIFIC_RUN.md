# M6D6g — E07c auxiliary-encoder scientific run (seed 42 × 200 epochs) and final checkpoint SHA256 recording

**Status:** PASS (candidate, uncommitted) · **Method:** E07c DiffFAS-BIN-IDFREE (controlled encoder reconstruction)
**Fidelity:** `CONTROLLED_ADAPTATION` · `DEV-021` (unchanged; no new fidelity class, no new deviation)
**Method status:** `IMPLEMENTED_NOT_EXECUTED` (the auxiliary encoder is trained; main DiffFAS is not)
**Authority:** `7642b23e60a1d89fcd481b03d4c9cb361f4c6e20` (M6D6f) · **Scope:** SCIENTIFIC, auxiliary seed **42**, TRAIN only.

This is the first and only scientific E07c auxiliary run. No scientific code, config, amendment, environment or
hyperparameter was changed before, during or after it. Main DiffFAS was not started. M7 and M8 were not started.

## 1. Pre-launch

* **Authority.** Laptop and `origin/m6-baselines` were at `7642b23` with a clean worktree. The GPU repo was clean at
  `a00aba9`, which is an ancestor of the remote. It was fetched and fast-forwarded with `--ff-only` to `7642b23` and
  stayed clean.
* **Resources.** GPU 0 (RTX 3090) had no compute process: 264 MiB used by the desktop, 0 % load. The runtime disk had
  414 GB free.
* **Run root.** `<runtime_root>/runs/m6/E07c/aux_encoder/seed_42` was absent, and so was `<runtime_root>/runs`. No
  M6D6g tmux session or process existed.
* **Qualification roots** (`m6d3a`, `m6d5e`, `m6d6e`, `m6d6f`) were retained untouched. They are not scientific
  assets.
* **Static preflight** (`--preflight-only`, locked interpreter, full launch environment) returned **PASS**, exit 0:
  * `torch_imported: false`;
  * commit `7642b23`, seed 42, 200 epochs;
  * `run_dir` = the `seed_42` auxiliary root;
  * `resume: null`;
  * the root was still absent afterwards.

  The complete output is in `M6D6G_E07C_AUX_SCIENTIFIC_RUNTIME_LOG.txt`.

## 2. Launch — exactly once

Launched at 2026-09-26T15:45:07Z in tmux session `m6d6g_e07c_aux`:

```
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/student20261/miniconda3/envs/gpat-m6-e07c/bin/python tools/run_e07c_aux.py \
  --seed 42 --execution-config configs/execution/m5_gpu_3090.yaml
```

No `--resume-state` was passed. The process exited with status 0.

## 3. Logical run identity

| Field | Value |
|---|---|
| run_id | `7014cdb366da52e4` |
| run_uuid | `7031fd2f-92d7-465a-94d1-ab99bf8a3960` |
| start / end UTC | 2026-09-26T15:45:10Z → 2026-09-26T20:02:40Z |
| training duration | 15 449.8 s (≈ 4 h 17 min) |
| process sessions | **1** (FRESH `fd7ad0ac-b769-445b-b33e-1426176ec5db`, pid 1666763) |
| interruptions / resumes | **0** / **0** |
| completion_status | `completed` |
| peak VRAM (per step) | 17 383 809 024 B (≈ 16.2 GiB) |

A laptop-side SSH wait loop lost its connection and exited with status 255. This affected monitoring only. The tmux
session and the training process were unaffected, and nothing was relaunched.

## 4. A8 boundaries and step accounting

* **Epoch 0.** `runner_state_epoch_000.pth` (`488572b5…`) was committed before the first iterator, with
  completed_epoch 0 and global_step 0.
* **Every epoch.** The whole-module `encoder_final.pkl` was saved and indexed first. The engineering sidecar was
  committed next. The previous sidecar was pruned only after the new index commit. This gives 201 sidecar commits and
  200 prunes; only `runner_state_epoch_200.pth` remains, as `COMMITTED_CURRENT`.
* **Step counts.** Logical optimizer steps = **11 200** = physical 11 200, with 0 superseded steps.
  `optimizer_applications` is also 11 200.
* **metrics.jsonl:**
  * 11 200 step records, with contiguous global steps 1..11200;
  * every one of the 200 epochs has exactly 56 steps at batch 256 (iterations 0..55);
  * lr 0.002 throughout;
  * all losses finite;
  * `val_losses` / `val_metrics` null in every record;
  * no reconciliation records, because the run was never resumed.
* **Per epoch.** 56 batches × 256 = 14 336 consumed, and 131 dropped by `drop_last`. The source-native loss is
  `running_loss / 14467`. All 200 epoch sample orders are distinct.

## 5. Training telemetry (not benchmark performance)

The table shows source-native epoch loss (`running_loss / 14467`) at selected epochs:

| Epoch | 1 | 2 | 5 | 10 | 25 | 50 | 100 | 150 | 199 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 1.3919 | 1.0790 | 0.3719 | 0.0218 | 0.0018 | 0.0012 | 0.0013 | 0.0013 | 0.0014 | 0.0015 |

Final epoch-200 loss: `0.001461103847973281` (`0x1.7f0504f8f0844p-10`). No decision used this curve. There was no
early stop, extension, rerun, earlier-checkpoint choice or LR change. Epoch 200 is the frozen final state.

## 6. Final scientific checkpoint

| Field | Value |
|---|---|
| path | `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E07c/aux_encoder/seed_42/checkpoints/encoder_final.pkl` |
| size | 185 136 819 bytes |
| **SHA256** | **`49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c`** |
| epoch / global_step | 200 / 11 200 |
| checkpoint_type / selected_for_final | `selected` / `true` |
| selection reason | `FINAL_STATE_AFTER_EPOCH_200 (A3 5.4 checkpoint_rule)` — no VAL, no TEST selection |
| format | `torch.save` of the WHOLE `nn.Module` (`custom_rn.ResNet`, BasicBlock [3,4,6,3], fc 512→7) |
| status | **`SHA256_RECORDED_PENDING_OWNER_FREEZE`** |
| authoritative_for_main_difffas | `false` |

The SHA256 was recomputed independently from the closed file with `sha256sum`. It equals every recorded copy:
* `run_summary.json` `final_or_selected_checkpoint_sha256`;
* the final `checkpoint_index.json` entry;
* the A8 committed-index `scientific_checkpoint_file`;
* the epoch-200 checkpoint event in `metrics.jsonl`.

The file was **not** deserialized to produce this evidence; M6D6c already qualified loading. The 199 earlier
epoch-boundary saves are `periodic` and not selected, and each was overwritten by the next epoch.

## 7. Data access

| Item | Value |
|---|---|
| TRAIN rows exposed | 14 467 (K7, frozen `split_v1.parquet` `fb9aeb36…`, class map `c7d23e3e…`) |
| TRAIN logical consumption | 11 200 logical steps × 256 = **2 867 200** (logical sample consumption, not a read count) |
| TRAIN physical face reads | `null`: not independently instrumented by the production scientific process |
| TRAIN superseded face reads | `null`: not independently instrumented (superseded **optimizer steps** = 0, recorded separately) |
| VAL rows / image reads | 0 / 0 (structural, see below) |
| TEST rows / image reads | 0 / 0 (structural, see below) |
| raw benchmark image reads | 0 (structural, see below) |
| main 8838-row relation | not opened |

**Face reads are not measured in this run.** In this repository, face-read counts are audit-hook measurements:
`train_face_opens` in the M6D6e/M6D6f qualification harnesses. The production scientific process has no such hook.
A face-read count also cannot be derived from optimizer steps, because DataLoader prefetch opens faces beyond the
consumed batches. M6D6e measured 3 329 face opens for a 1-step B=256 process. `run_logging_v1`
(`EXPLICIT_NULL_WITH_REASON`) therefore gives both face-read fields `null` with a reason. The 2 867 200 figure is
logical TRAIN sample consumption only: 11 200 A8-logged optimizer steps, each recording 256 unique TRAIN sample ids.

**The VAL/TEST/raw zeros are structural, not instrumented.**
* 0 VAL and 0 TEST rows were materialized from the split manifest.
* `CanonicalFaceReader` resolves only TRAIN-population sample ids and refuses all others.
* The main 8838-row relation is never opened by the auxiliary runner.

**The manifest read is a separate disclosure.** pyarrow decoded the allowlisted columns of the single row group of
`split_v1.parquet` (all splits) and filtered to TRAIN. That is a manifest read, not a face-image read.

## 8. Runtime artifacts (retained on the GPU runtime; not in Git)

| File | Bytes | SHA256 |
|---|---|---|
| resolved_config.yaml | 15 647 | `a389357eb8b979ee37bed82930be93b3f3333a7cb05050d3d1052119f0a24c78` |
| run_manifest.json | 5 284 | `c0bd3a89e47a52905d49eb307f4ba72b7ee89680f4e8439f7ea9023ed92d0891` |
| metrics.jsonl | 16 584 323 | `efb37d1fc683959056d6773527bfe35b6d4f6e96d8450d79c96036609b37ef56` |
| checkpoint_index.json | 89 666 | `d93a793681e9bfc93a64b541664ac2256f7e7add4d92ed32e07555396f850793` |
| resume_state_index.json | 514 082 | `652742dafa4900648b6d276e2b650b6d1a7ea8e7bdefc73e503f40b1808325b2` |
| run_summary.json | 2 489 | `3fc24a55547c6f78e1a61591fbe937e69989fb84bf9541450733974f1b2e30c9` |
| stdout.log | 7 892 | `04cc649915688bfa53b90d6006862adf5aadafd132edc143a828e37805bc7b58` |
| stderr.log | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| checkpoints/resume/runner_state_epoch_200.pth (engineering) | 370 095 195 | `374d4e52964e78d54f61c3112d66032827828aab5b7b173be664ecd05273e19e` |
| checkpoints/encoder_final.pkl (scientific) | 185 136 819 | `49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c` |

## 9. Statuses

**New:** `E07c_AUXILIARY_ENCODER_200_EPOCH_TRAINING_COMPLETED`, `E07c_AUXILIARY_ENCODER_SCIENTIFIC_CHECKPOINT_PRODUCED`,
`E07c_AUXILIARY_ENCODER_FINAL_SHA_RECORDED`, retaining `FINAL_SHA_PENDING_OWNER_FREEZE`.

**Not claimed:** `E07c_AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN` and `MAIN_DIFFFAS_SCIENTIFIC_TRAINING_COMPLETED`.
Method-level E07c is not complete.

**Still not qualified:**
* `AUXILIARY_ENCODER_SHA_FROZEN_FOR_MAIN`
* `MAIN_DIFFFAS_TRAINING_GRAPH`
* `MAIN_CHECKPOINT_RESUME`
* `MAIN_RUNNER_ENCODER_LOAD_INTEGRATION`
* `MAIN_PRODUCTION_RUNNER`
* `MAIN_DIFFFAS_SCIENTIFIC_TRAINING`
* `M8_BANK`

**Counters:** scientific auxiliary logical runs 1 (seed 42) · main DiffFAS scientific runs 0 · VAL/TEST training or
selection 0 · M8 bank false.

## 10. Next single milestone (not executed)

**M6D6h** — owner-freeze the recorded final auxiliary encoder SHA256 (`49a24a3a…107c`) for secure main-DiffFAS
consumption.
