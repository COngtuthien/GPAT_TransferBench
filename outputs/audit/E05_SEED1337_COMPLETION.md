# E05 seed1337: scientific run completed cleanly

**Status:** `E05_SEED1337_COMPLETED_CLEANLY`, 65/65 gates passing. The machine-readable record is `E05_SEED1337_COMPLETION.json`, which embeds the read-only host collection so the result can be rebuilt with `tools/e05_seed_completion_audit.py build`. This is the same tool and the same gates used for seed42.

This commit is reporting evidence only. The E05 scientific training authority stays **`efdf753156045dd2bed52229d9eac869b18d72f8`**, and the GPU checkout remains clean at that commit.

| | |
|---|---|
| method | E05, **PCGAN (controlled architecture resolution)**, CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE |
| run | seed = 1337, run_id = **`036c806bb2f339b7`** = sha256("E05\|1337\|478756e1…\|efdf753…")[:16] (recomputed) |
| resume | `NOT_QUALIFIED_FRESH_ONLY`. There was one run_start, and the command was exactly `tools/run_e05.py --seed 1337 --execution-config configs/execution/m5_gpu_3090.yaml`. |
| window | 2026-10-25T07:12:02Z → 07:33:49Z, 1306.7 s of training, median 0.308 s per iteration, peak 3.08 GB |
| exit | `~/workdir/e05_s1337.exit` = 0. The tmux `m6_e05_s1337` pane is dead with status 0, no runner process remains, and stdout, stderr and generation_log are all empty. |

## Completion accounting

- **iterations = 4000:** 4000 scientific records, with global_step exactly 1…4000, no gaps and no duplicates. The only events are 1 `e05_run_start` and 2 `e05_checkpoint_event`, so records_written = 4003.
- **Optimizer applications:** for every iteration t, D = G = hook_D = hook_G = t and the total is 2t. The final counts are **D = 4000, G = 4000, total 8000**. The learning rate is 1e-6 for both groups throughout.
- **Pair order:** **4000/4000** sampler positions equal the independent seed-1337 prediction (`torch.Generator().manual_seed(1337)`, one int64 draw, then `randperm(8838)`).
  - **Distinct TRAIN pairs = 4000**, with 0 repeats: 1143 casia_fasd, 561 msu_mfsd and 2296 siwmv2.
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
| L_rec | 184.0 | 308.1 | 224.1 |
| L_recblur | 135.2 | 161.8 | 105.5 |
| L_advrec | 0.689 | 0.952 | 0.934 |
| L_advmix | 0.688 | 0.927 | 0.936 |
| L_pat | 0.902 | 1.855 | 2.194 |
| L_G_total | 321.4 | 473.6 | 333.7 |
| L_D_total | 2.643 | 0.900 | 0.906 |

The single-iteration values depend on which pair was drawn (batch 1). The last-100 means are given for context only; nothing is selected from them.

## Checkpoints (BASELINE_FINAL_STATE_V1)

| iteration | file | type | selected_for_final | bytes | sha256 |
|---|---|---|---|---|---|
| 1 | `checkpoints/iteration_0001.pth` | periodic | false | 232,113,504 | `cf6bc24e555eef1cddfe9aeaa2b78692496bac26e89fd4fa98ad554249725905` |
| 4000 | `checkpoints/iteration_4000.pth` | selected (BASELINE_FINAL_STATE_V1) | **true** | 232,113,504 | **`b6a4cb2c61b1b582f1acd1eb3514f6a15ea49e544e1dac48415004c6ce4f7b34`** |

- **Index and events:** exactly two files, both indexed and matched by their events. Recomputed SHA256 and sizes equal the index.
- **Loading:** both checkpoints load with `torch.load(weights_only=True)`:
  - format `E05_BASELINE_STATE_V1`, containing the Encoder, Generator, ImageD and PatchD models;
  - 167 tensors, all floating and all finite.
- **Provenance:** iteration 4000 records D = 4000, G = 4000, total 8000, and iteration 1 records 1/1/2. Writing a checkpoint added zero optimizer steps.
- **Training changed the weights:** 139 tensors differ between iteration 1 and iteration 4000, which is all 80 + 59 trainable parameters.
- **Selection:** exactly one selected checkpoint, recorded in run_summary and checkpoint_index.
- **Sidecars:** no `.partial`, runner_state or resume file.
- **Manifest:** the deterministic checkpoint manifest is `E05_SEED1337_CHECKPOINT_MANIFEST.txt`.

## Firewall

The stat-only atime snapshots are seed1337's own pre-launch snapshot (07:11:46Z) and a post-run snapshot (07:41:47Z), 20 627 files each. No image was opened by the audit.

| class | files | atime-sensitive | changed |
|---|---|---|---|
| VAL (casia/msu/siwmv2) | 3121 | 3121 | **0** |
| TEST (casia/msu/siwmv2) | 3027 | 3027 | **0** |
| siwmv2 TRAIN | 9507 | 5427 | 1846, all faces of consumed pairs |
| casia/msu TRAIN | 4960 | 0 | 0 |

**VAL = 0 and TEST = 0.** No consumed sample is VAL or TEST, and `test_split_accessed` is false.

TRAIN faces read by E06b or E05 seed42 less than 24 hours before the pre snapshot are not atime-sensitive under relatime. TRAIN use is proven by the order, manifest and content evidence above, not by atime. No manifest was modified.

## Environment (owner driver ruling)

- **Driver:** `lock_driver_version = 595.84`, `observed_driver_version = 580.178.04`, `driver_policy = OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY`.
  - The GPU model (RTX 3090) and UUID `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` match exactly.
  - Every other lock field also matches exactly, in the run_start event and in a live re-check after the run: no difference other than the driver.
  - This is not evidence that arbitrary driver changes are acceptable.
- **Python:** `/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python`, sha `d3255ce6…`.
- **Packages:** torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0, cuDNN 92000, 54 packages equal to the lock.
- **Extensions:** fused_bias_act `5afdb236…` and upfirdn2d `40ba9db7…`, equal on disk. No JIT build, no fallback and no package change.
- **Precision:** FP32, with TF32 and autocast off and cuDNN deterministic on. Seeds were set by `apply_framework_seed(1337)`.

## Run-root integrity

- **Listing:** 10 files. The LC_ALL=C sorted `<path> <size> <sha256>` listing digest is **`638c777fb676e7cb78b2ed92d1e6e3dae56e4217dc2c8c5021a973604b10a242`**.
  - It is identical before and after the audit, and equal to an independent shell listing taken before the audit.
  - The scientific root was not mutated.
- **metrics.jsonl:** `2c6a4746c1339dace16768b28213e66950255d4cc0f9b312f770fbeee14c2e98`
- **run_manifest.json:** `863af46ebb8a201692a8241332158dc2752a104b2b2f0d0f8368d64ec20bbaa2`
- **run_summary.json:** `ffe0b5299cad146e40ff11e86b3f1af5431d317830c7de0be96f2c75f48258b8`
- **resolved_config.yaml:** `366a641115e4c2b92f12da83ff97cd8a30c163884411580bcaeb15d13c200e2b`
- **checkpoint_index.json:** `d4cb8badc4ec016cdee20963d1cab287d3f8199e26020c63a6f03dfc14387db7`

## Boundary

The pair roles, seeded shuffle order, [-1, 1] input mapping and iteration-1/4000 checkpoints are owner-approved controlled implementation resolutions. They are not native, faithful or official PCGAN choices.

This closure reports no downstream metric, no bank and no best-seed or best-iteration logic.
