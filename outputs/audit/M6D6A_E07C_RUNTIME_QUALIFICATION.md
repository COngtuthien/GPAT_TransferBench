# M6D6a — E07c DiffFAS-BIN-IDFREE: execution environment + live architecture/runtime qualification

**M6D6a PASS (synthetic forward/runtime scope only).**

| Status | |
|---|---|
| **E07c_EXECUTION_ENVIRONMENT_QUALIFIED** | new isolated `gpat-m6-e07c`, lock `0c909de1…f450` |
| **E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED** | pinned `custom_rn.resnet18` (A3 head `Linear(512, 7)`), live A6 interface |
| **E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED** | pinned `BeatGANsAutoencModel` via `get_model_conf().make_model()` |
| **E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED** | encoder → AttentionBlock connectivity, fourth-output discard, forward-only loss path |
| **IMPLEMENTED_NOT_EXECUTED** | E07c scientific method status unchanged |
| **CONTROLLED_ADAPTATION** | preserved: A1 identity-free adaptation + A3 controlled encoder reconstruction, DEV-021; A6 is a source-traceability correction only |

Reporting label: **DiffFAS-BIN-IDFREE (controlled encoder reconstruction)**. It is not described as native, faithful or an
official reproduction. **Source/contract state: RESOLVED** (the unavailable PADISI encoder was resolved by the approved A3
controlled reconstruction). **Execution artifact: NOT YET PRODUCED** — the one auxiliary conditioning encoder is
`NOT_YET_TRAINED`; that is execution-stage incompleteness, not a source gap.

**Not qualified by M6D6a:** real TRAIN path · training graph / backward · training memory · auxiliary encoder training ·
checkpoint writer · resume · production runner · scientific training (0/3 seeds) · M8 bank.

## 1. Authority and synchronization

- **Laptop.** Branch `m6-baselines`, HEAD = `git ls-remote origin refs/heads/m6-baselines` =
  `596d8459c9f2e54a5f36e2463099b662b800eb7f` ("M6D5e: qualify E06c production runner"), worktree clean.
- **GPU.** Started clean at `cd125dbdd1b45f3963fc501c5f7dc6ff9091d536`, exactly one commit behind. Synchronized only by
  `git fetch origin` → `git merge-base --is-ancestor HEAD origin/m6-baselines` (PASS, distance 1) →
  `git merge --ff-only origin/m6-baselines`. Result `596d845`, clean. No reset, clean, stash, rebase or forced checkout.
- **Frozen spec** SHA256 `f7d23716…281489e` verified. Authority read: E07c method YAML (`dba34a92…`), A1 frozen adaptation
  (`aa9e9841…`) and both byte-identical snapshots, `run_logging_v1`, `source_pins.json`, M6C2b3 audit, `methods/difffas/`,
  M6C2b3 tests/preflight, ledger, index, CONFIG_STATUS. Amendments: A1 `03828716…`, A2 `b4fa7bfa…`, A3 `b12451537…`,
  **A6** located from committed evidence (M6A8 audit and `methods/difffas/contract.py`):
  `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md` (`759f72d8…`),
  overlay `configs/amendments/e07c_a6_feature_interface_source_correction.yaml` (`dd3f29aa…`).
- **Pinned source.** `murphytju/DiffFAS@23f40519ec25a833ebc06842aa6fbab74fad4d15`, tree `d190a5fb…`. It was absent on the
  GPU and was acquired in the laptop's recorded mode (full tree, depth 1, detached `FETCH_HEAD`); all 24 tracked files are
  SHA256-identical to the laptop cache. `custom_rn.py` `fa788c4d…`, `unet_autoenc.py` `127ecd59…` (= A6). No weights
  exist in that repository. The source cache is git-ignored and was never edited; its worktree stayed empty
  (`git status --porcelain --untracked-files=all` = ``) through every process.

## 2. Execution environment (`gpat-m6-e07c`)

D0b had left E07c `NEW_ENV_REQUIRED; BLOCKED_BY_VERSION_AMBIGUITY` (CUDA 11.3 vs 11.7 declarations; tensorfn missing).
It was resolved experimentally, without choosing either historical CUDA:

- The pinned import closure (FAS_train.py, config/diffconfig.py, diffusion.py, models/*) needs torch, torchvision, numpy,
  cv2, tqdm and **tensorfn** (+ pydantic, pyhocon). tensorfn 0.1.28 is written against the **pydantic v1 API**
  (`ArbitraryTypeError`, `__get_validators__`, an unannotated field), so pydantic 2 cannot import it.
- `conda create --offline -n gpat-m6-e07c --clone gpat-m5` (the E06c precedent), then a hash-pinned, no-index, no-deps
  install of exactly 14 packages, **each at its exact upstream `environment.yml` version**: tensorfn 0.1.28, pydantic 1.9.2,
  pyhocon 0.3.60, pyparsing 3.1.0, termcolor 1.1.0, tabulate 0.8.10, boto3 1.23.10, botocore 1.26.10, jmespath 0.10.0,
  s3transfer 0.5.2, urllib3 1.26.16, rich 12.6.0, commonmark 0.9.1, tqdm 4.62.3 (`environments/e07c.pip-requirements.txt`).
  pyhocon/termcolor are pure-Python sdists (inspected: no `.c/.pyx/.so`). `pip check`: no broken requirements.
- Framework packages come unchanged from gpat-m5: **Python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130,
  CUDA 13.0, cuDNN 92000, NumPy 2.4.6, OpenCV (headless) 5.0.0.93, Pillow 12.3.0**. **SciPy is not installed**: no pinned
  DiffFAS file imports it. GPU: **NVIDIA GeForce RTX 3090, compute capability 8.6, driver 595.84**.
- Historical Python 3.6.13 / PyTorch 1.10.2 (cu113 build) / CUDA 11.3 & 11.7 packages are **provenance, not execution
  pins** (E03 addendum principle; E05/E06c precedent). Only this combination was executed; no other combination is
  claimed equivalent. This runtime selection is disclosed for owner review.
- Protected environments (base, gpat-m5, gpat-m6-e03, gpat-m6-e04-geometry, gpat-m6-e05, gpat-m6-e06c, stdn): conda-explicit,
  pip-freeze and interpreter fingerprints **identical before and after**. No sudo, no system CUDA or driver change.
- `compatibility_patch = NONE`: no source edit, alias or shim. Engineering observations (no semantic effect):
  `reduction='elementwise_mean'` maps to `mean` with a UserWarning; NumPy 2 `__array__` copy-keyword DeprecationWarning in
  `GaussianDiffusion.__init__` (same float64 betas); botocore/lib2to3 import DeprecationWarnings; torch CUDA init runs the
  read-only `/sbin/ldconfig -p`.
- Lock `environments/e07c.lock.json` SHA256 `0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450`
  (binds runtime JSON, harness, conda-explicit, pip-freeze, install spec, executed identity, source closure, A6 overlay).
  The conda-explicit export equals gpat-m5's conda layer (the added packages are pip-layer only).

**Open items carried forward (not resolved here):** (1) torch 2.12.1 `torch.load` defaults to `weights_only=True`; the pinned
`BeatGANsAutoencModel.encoder(path)` loads a whole pickled module and `FAS_sample.py` loads a checkpoint dict containing a
tensorfn/pydantic conf, so an explicit decision is needed before any checkpoint is loaded. M6D6a loads nothing. (2) The E07c
production precision policy is not decided here; qualification ran FP32 with TF32 off (frozen cudnn benchmark=False,
deterministic=True applied); historical-stack TF32 behaviour was not measured.

## 3. A6 reconciliation

| | x32x32 | x16x16 | x8x8 | fourth output |
|---|---|---|---|---|
| Historical frozen YAML / A3 text (H×W×C, **unchanged**) | 32×32×128 | 16×16×256 | 8×8×512 | B×K |
| Effective A6 authority (H×W×C) | 32×32×256 | 16×16×512 | 8×8×512 | B×7 |
| **Live pinned-source tensors (NCHW, both processes)** | **[4,256,32,32]** | **[4,512,16,16]** | **[4,512,8,8]** | **[4,7]** |

The effective execution interpretation follows A6; the historical frozen files remain byte-identical. No projection,
channel adapter or architecture substitution exists. The live model makes the A6 widths necessary, not optional: each
consuming `AttentionBlock` applies `BatchNorm2d(channels)` to its conditioning map, and those blocks have 256 channels at
32×32 and 512 at 16×16 / 8×8 (`channel_mult (1,1,2,2,4,4)`, `model_channels 128`).

## 4. Conditioning encoder (construction + synthetic inference only)

Built through the committed A3 seam `methods/difffas/encoder.py::encoder_model` → pinned
`models/custom_rn.py::resnet18(pretrained=False)` (module `custom_rn`, SHA256 `fa788c4d…`), then `.cuda()` and `.eval()` as
`BeatGANsAutoencModel.encoder()` / `FAS_train.py:35` do. Topology **BasicBlock [3,4,6,3]** (a ResNet-34 layer layout
despite the name; not "corrected"), stage widths 64/256/512/512. The pinned `ResNet.__init__` hard-codes
`fc = Linear(512, 17)`; the A3 seam replaces only `fc` with **`Linear(512, 7)`**. A structural comparison with a freshly
built pinned `resnet18()` differs **only** in `fc` (17 → 7): no projection, no torchvision ResNet (`isinstance` and module
checks). **46,233,707 parameters (112 tensors), all trainable; 111 buffers**; FP32 on `cuda:0`; batch 4 at 256×256; all
outputs finite. The weights are the seeded random initialization: **UNTRAINED, in memory only, never saved, not the future
auxiliary checkpoint**. No CrossEntropy, SGD, backward, optimizer step or `encoder_final.pkl`.

## 5. Main DiffFAS model

`config/diffconfig.py::get_model_conf().make_model()` then `.to('cuda')` — the `FAS_train.py:216` `use_pair` false branch.
`BeatGANsAutoencModel` from pinned `models/unet_autoenc.py`: **159,363,974 parameters (694 tensors), 21 buffers**, FP32,
`cuda:0`; `in_channels 3`, first conv input channels 3, `image_size 256`, `out_channels 6` (learned sigma),
`model_channels 128`, `channel_mult (1,1,2,2,4,4)`, `attention_resolutions (32,16,8)`, `dropout 0.1`. The EMA copy uses the
identical constructor and was not built. Diffusion via the official tensorfn path: `load_config(DiffusionConfig,
config/diffusion.conf)` → `beta_schedule.make()` (linear, 1000 steps, float64 1e-4 → 2e-2) →
`create_gaussian_diffusion(betas, predict_xstart=False)` = **EPSILON / LEARNED_RANGE / MSE, 1000 timesteps**. The parsed
(not constructed) optimizer/scheduler config is AdamW lr 1e-5 and cycle (lr 1e-5, n_iter 2,400,000, warmup 5000,
decay [linear, flat]).

## 6. Synthetic forward qualification (qualification_seed 60601)

The qualification seed is `60601` (recorded as `qualification_seed`; `experiment_seed = null`); 42/1337/2026 were never
used. Inputs are deterministic analytic FP32 images in [−1, 1] (content / GT / style_spoof roles), distinct per row; no
file, decoder, manifest or RNG was involved in building them.

- **Encoder interface:** the four live shapes in §3.
- **Connectivity (eval mode, dropout off):** forward pre-hooks record every `AttentionBlock` conditioning tensor by
  storage identity. x32x32 → `input_blocks.11.1`, `output_blocks.7.1` (256 ch, 32×32); x16x16 → `input_blocks.14.1`,
  `output_blocks.4.1` (512 ch, 16×16); x8x8 → `input_blocks.17.1`, `middle_block.1`, `output_blocks.1.1` (512 ch, 8×8).
  7/7 AttentionBlock conditions are these three features; **the fourth output reaches 0 consumers**.
- **Discard proof:** replacing encoder output[3] by NaN leaves **all 45 traced stages** (input/middle/output blocks,
  AttentionBlocks, final projection) bitwise identical and finite (`unet_autoenc.py:178` `x32x32, x16x16, x8x8, _ = …`).
- **Consumption proof:** zeroing each feature changes the output of every AttentionBlock that consumes it (max |Δ| 6.2–11.0).
- **Zero-initialised projection disclosed:** pinned `get_model_conf` sets `resnet_use_zero_module=True`, so the final
  `self.out` convolution is zero-initialised and an untrained model's output is exactly 0. The proofs therefore use the
  45 internal stages, all finite; the zero final output is recorded as expected source behaviour, not as evidence.
- **FORWARD_LOSS_PATH_ONLY:** `diffusion.training_losses` in the `FAS_train.py:57-68` argument form (model train mode,
  encoder eval, prob 0.8, means 5, var 3, `use_pair=False`) under `torch.no_grad`: the model input is `[4,3,256,256]`,
  the encoder is called once, mse 0.99911, vb [4] ∈ [3.1e-6, 0.124], per-sample loss finite, mean 1.03511. No backward, no
  autograd graph. Train mode updates the native AttentionBlock BatchNorm running statistics (21 buffers); **no parameter
  changed**. The training graph is NOT qualified.

## 7. Repeatability

Two fresh processes (`process_1`, `process_2`) and the last exploratory attempt (`attempt_4`, identical harness bytes)
produced **byte-identical evidence JSON** (SHA256 `0ccc5a96…dab2`): identical initial parameters (encoder aggregate
`22bcb61b…`, main `325358fe…`), all forward and 45 internal-stage tensor hashes, losses and memory figures. Global
deterministic algorithms were **not** forced; equality is observed, not manufactured.

## 8. CUDA memory — SYNTHETIC_FORWARD_MEMORY_OBSERVATION

After construction 838,215,168 B allocated. Eval connectivity forwards: peak allocated 8,685,248,000 B, reserved
9,252,634,624 B. Forward-only loss path: peak allocated 8,697,844,736 B. Batch 4, `torch.no_grad`, no activation graph.
This does **not** qualify training memory, the scientific batch size or 400-epoch feasibility.

## 9. Access audit and counters

Python audit-hook firewall in every process: manifests, `.parquet`, images, `faces_256`, repository/runtime `data`/`cache`,
the runtime root outside the build root, `runs/`, and weight/array suffixes denied; writes only under
`builds/e07c_difffas`. Subprocesses restricted to git metadata (102), nvidia-smi (2) and one `/sbin/ldconfig -p`.

| Count (each process) | Value |
|---|---:|
| benchmark manifest opens | **0** |
| benchmark image opens | **0** |
| TRAIN / VAL / TEST image opens | **0 / 0 / 0** |
| runtime data / scientific run root accesses | **0 / 0** |
| firewall denials | **0** |
| `backward()` / `autograd.grad` | **0 / 0** |
| optimizer constructions / `optimizer.step()` | **0 / 0** |
| `torch.save` / `torch.load` | **0 / 0** |
| scientific checkpoints / auxiliary encoder trainings | **0 / 0** |
| scientific seed runs completed | **0 / 3** |

`<runtime_root>/runs` did not exist before or after any process; neither `runs/m6/E07c` nor
`aux_encoder/seed_42/checkpoints/encoder_final.pkl` was created.

## 10. Attempts (all retained in `M6D6A_E07C_RUNTIME_LOG.txt`)

1. Stopped at `import torch`: the harness's own subprocess firewall refused `ctypes.util.find_library`'s `/sbin/ldconfig -p`.
   No model was built. Fix: allow exactly that read-only argv.
2. Stopped in environment capture: `torch.load` was already replaced by the forbid-stub before its signature was read.
   Fix: capture the default before patching.
3. Stopped at a harness gate: "zeroing x32x32 changes the model output" failed because the untrained model's final output
   is identically zero (zero-initialised `self.out`), making final-output counterfactuals uninformative. Fix: prove
   consumption and discard on internal stages. No scientific setting changed.
4. PASS; its JSON is byte-identical to the final pair's.

All four fixes are harness-only engineering changes. No source file, config, hyperparameter or seed changed.

## 11. Tests and preflight

`tests/test_m6d6a_e07c_runtime.py`: 16 tests (pinned source identity, A6 interface and preserved history, custom_rn usage,
no torchvision, 512→7 head, live shapes, a CPU construction test where Torch exists, main model / use_pair=false /
3-channel, finiteness, fourth-output discard at the pinned consumer, zero optimizer/checkpoint, zero manifest/image
access, firewall rejections, environment lock identity, static contract, repeatability, memory label, wording). Existing
M6C2b3 tests are unchanged. Exact commands and counts are in the runtime log and audit JSON.
`python3 -I -S -B tools/m6d6a_e07c_runtime_preflight.py` is static: no Torch, no CUDA, no model, no manifest payload.

## 12. Frozen scientific settings (verified statically, not executed)

Seeds 42/1337/2026; main 400 epochs, batch 4, AdamW lr 1e-5, cycle scheduler (lr 1e-5, n_iter 2,400,000, warmup 5000,
decay [linear, flat]), EMA 0.9999 / before_warmup 0.0; DDIM, skip 10, sample_initial_noise 250, 25 effective steps,
cond_scale 2.0, generation loads `checkpoint["model"]` (not EMA). Auxiliary encoder contract (future, not executed):
exactly 1 run, seed 42, TRAIN only, K=7 (live, makeup, mask_2d, mask_3d, partial, print, replay; other_spoof excluded),
200 epochs, batch 256, drop_last, shuffle, workers 6, no augmentation, no class balancing, CrossEntropyLoss, SGD lr 0.002 /
momentum 0.9 / weight_decay 0.005, no scheduler, final state after epoch 200, whole-module save, reused for main seeds
42/1337/2026. `BASELINE_FINAL_STATE_V1` checkpoint rule.

**NO BENCHMARK DATA ACCESS. NO TRAIN/VAL/TEST IMAGE ACCESS. NO OPTIMIZER STEP. NO BACKWARD. NO CHECKPOINT. NO AUXILIARY
ENCODER TRAINING. NO SCIENTIFIC TRAINING. NO DIFFUSION SAMPLING. NO SYNTHETIC BANK. NO COMMIT. NO PUSH.**
