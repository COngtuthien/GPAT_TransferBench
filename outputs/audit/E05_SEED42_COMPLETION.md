# E05 seed42: scientific run completed cleanly

**Status:** `E05_SEED42_COMPLETED_CLEANLY`, 65/65 gates passing. The machine-readable record is `E05_SEED42_COMPLETION.json`, which embeds the read-only host collection so the result can be rebuilt with `tools/e05_seed_completion_audit.py build`.

This commit is reporting evidence only. The E05 scientific training authority stays **`efdf753156045dd2bed52229d9eac869b18d72f8`**, and the GPU checkout remains clean at that commit.

| | |
|---|---|
| method | E05, **PCGAN (controlled architecture resolution)**, CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE |
| run | seed = 42, run_id = **`18463bf5cd471e55`** = sha256("E05\|42\|478756e1…\|efdf753…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`. There was one run_start, and the command was exactly `tools/run_e05.py --seed 42 --execution-config configs/execution/m5_gpu_3090.yaml`. |
| window | 2026-10-25T04:39:34Z → 05:01:29Z, 1314.6 s of training, median 0.308 s per iteration, peak 3.08 GB |
| exit | `~/workdir/e05_s42.exit` = 0. The tmux `m6_e05_s42` pane is dead with status 0, no runner process remains, and stdout, stderr and generation_log are all empty. |

## Completion accounting

- **iterations = 4000:** 4000 scientific records, with global_step exactly 1…4000, no gaps and no duplicates. The only events are 1 `e05_run_start` and 2 `e05_checkpoint_event`, so records_written = 4003.
- **Optimizer applications:** for every iteration t, D = G = hook_D = hook_G = t and the total is 2t. The final counts are **D = 4000, G = 4000, total 8000**. The learning rate is 1e-6 for both groups throughout.
- **Pair order:** **4000/4000** sampler positions equal the independent seed-42 prediction (`torch.Generator().manual_seed(42)`, one int64 draw, then `randperm(8838)`).
  - **Distinct TRAIN pairs = 4000**, with 0 repeats: 1149 casia_fasd, 530 msu_mfsd and 2321 siwmv2.
- **TRAIN scope:**
  - **Pair manifest:** every record equals its `manifests/pairs_train_v1.parquet` row (`a5e4fdae…`) and its `pair_sha256`.
  - **Splits and roles:** in every pair, x_src = the source_spoof_id face (spoof) and x_tgt = the target_live_id face (live). Both are TRAIN in `split_v1` and from the same dataset.
  - **Content check:** both consumed face PNG sha256 digests equal the M2 sample accounting for those TRAIN samples.
  - **Input tensors:** every input is [1,3,256,256] float32 in [-1, 1].
- **Numerical health:**
  - All 14 loss values are finite in all 4000 records, with no NaN or Inf.
  - The five G terms (L_rec, L_recblur, L_advrec, L_advmix, L_pat) are active in every record.
  - `L_G_total` equals their unit sum; the largest relative FP32 error is 1.9e-7.
  - D losses are finite and positive, and `L_D_total` = `L_D_image` + `L_D_patch` (largest relative error 6.0e-8).
- **Gradients:** 80/80 D and 59/59 G tensors are present, finite and nonzero in every iteration, with the inactive group's gradients always None.

Loss trajectory (first, last, mean of the last 100 iterations):

| | iteration 1 | iteration 4000 | last-100 mean |
|---|---|---|---|
| L_rec | 224.0 | 178.7 | 194.0 |
| L_recblur | 105.6 | 112.0 | 101.4 |
| L_advrec | 0.704 | 1.038 | 1.126 |
| L_advmix | 0.705 | 1.106 | 1.150 |
| L_pat | 0.712 | 1.686 | 1.913 |
| L_G_total | 331.7 | 294.5 | 299.6 |
| L_D_total | 2.831 | 1.120 | 0.989 |

## Checkpoints (BASELINE_FINAL_STATE_V1)

| iteration | file | type | selected_for_final | bytes | sha256 |
|---|---|---|---|---|---|
| 1 | `checkpoints/iteration_0001.pth` | periodic | false | 232,113,504 | `82d0606303d6a1ed38da5581758f12e444da603fb86817922ce7fb0dc87971df` |
| 4000 | `checkpoints/iteration_4000.pth` | selected (BASELINE_FINAL_STATE_V1) | **true** | 232,113,504 | **`4ad80b35e759154f741d249db8c6e43622cab916e13dc785ad9273bd65113cb1`** |

- **Index and events:** exactly two files, both indexed and matched by their events. Recomputed SHA256 and sizes equal the index.
- **Loading:** both checkpoints load with `torch.load(weights_only=True)`:
  - format `E05_BASELINE_STATE_V1`, containing the Encoder, Generator, ImageD and PatchD models;
  - 167 tensors, all floating and all finite.
- **Provenance:** iteration 4000 records D = 4000, G = 4000, total 8000, and iteration 1 records 1/1/2. Writing a checkpoint added zero optimizer steps.
- **Training changed the weights:** 139 tensors differ between iteration 1 and iteration 4000, which is all 80 + 59 trainable parameters.
- **Selection:** exactly one selected checkpoint, recorded in run_summary and checkpoint_index.
- **Sidecars:** no `.partial`, runner_state or resume file.
- **Manifest:** the deterministic checkpoint manifest is `E05_SEED42_CHECKPOINT_MANIFEST.txt`.

## Firewall

The stat-only atime snapshots are the seed42 pre-launch snapshot (04:39:17Z) and a post-run snapshot (07:03:55Z), 20 627 files each. No image was opened by the audit.

| class | files | atime-sensitive | changed |
|---|---|---|---|
| VAL (casia/msu/siwmv2) | 3121 | 3121 | **0** |
| TEST (casia/msu/siwmv2) | 3027 | 3027 | **0** |
| siwmv2 TRAIN | 9507 | 9501 | 4074, all faces of consumed pairs |
| casia/msu TRAIN | 4960 | 0 | 0 |

**VAL = 0 and TEST = 0.** No consumed sample is VAL or TEST, and `test_split_accessed` is false.

The CASIA and MSU TRAIN faces were read by E06b less than 24 hours before the pre snapshot, so relatime does not update them. Their TRAIN use is proven by the order, manifest and content evidence above, not by atime. No manifest was modified.

## Environment (owner driver ruling)

- **Driver:** `lock_driver_version = 595.84`, `observed_driver_version = 580.178.04`, `driver_policy = OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`.
  - The GPU model (RTX 3090) and UUID `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` match exactly.
  - Every other lock field also matches exactly, in the run_start event and in a live re-check after the run: no difference other than the driver.
  - This is not evidence that arbitrary driver changes are acceptable.
- **Python:** `/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python`, sha `d3255ce6…`.
- **Packages:** torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0, cuDNN 92000, 54 packages equal to the lock.
- **Extensions:** fused_bias_act `5afdb236…` and upfirdn2d `40ba9db7…`, equal on disk. No JIT build, no fallback and no package change.
- **Precision:** FP32, with TF32 and autocast off and cuDNN deterministic on. Seeds were set by `apply_framework_seed(42)`.

## Run-root integrity

- **Listing:** 10 files. The LC_ALL=C sorted `<path> <size> <sha256>` listing digest is **`526419f00fd272b2207b0b167441f4ec63b22d42687f4d0d0b8399613c6e1d1e`**.
  - It is identical before and after the audit, and equal to an independent shell listing taken before the audit.
  - The scientific root was not mutated.
- **metrics.jsonl:** `4b2fc19c90b2f0fd0c48212bb71a850ddafef398322d1533ac1a4649da5c04a9`
- **run_manifest.json:** `f1b0d1f6f84b718338786a5103ab949ffb8266d3e749d536a9ad96ccd4f2775a`
- **run_summary.json:** `023341eaf3edab37da2a591152c431633da2ba3aec79b4f48940218b21a8d0de`
- **resolved_config.yaml:** `33a3c86db4d666e77564d98fe13fccb1ef7cace45ea0004d16c08fad63046bf7`
- **checkpoint_index.json:** `fbb008ff35d2abcceaece0c9a4f82720a499d9e9670388edf180dee22c59517c`

## Boundary

The pair roles, seeded shuffle order, [-1, 1] input mapping and iteration-1/4000 checkpoints are owner-approved controlled implementation resolutions. They are not native, faithful or official PCGAN choices.

This closure reports no downstream metric, no bank and no best-seed or best-iteration logic.
