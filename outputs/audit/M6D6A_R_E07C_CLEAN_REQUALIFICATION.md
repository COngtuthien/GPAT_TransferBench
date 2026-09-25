# M6D6a-r — E07c clean corrective requalification

**M6D6a final status: CLEAN_REQUALIFICATION_PASS** (initial candidate status: **PROCEDURAL_FIREWALL_DEVIATION**, retained).
Classification: **CORRECTIVE_REQUALIFICATION** of the M6D6a synthetic forward/runtime qualification. Not a new scientific
milestone.

| Status | Basis |
|---|---|
| **E07c_EXECUTION_ENVIRONMENT_QUALIFIED** | existing `gpat-m6-e07c` re-verified against lock `0c909de1…f450`; not rebuilt or mutated |
| **E07c_CONDITIONING_ENCODER_RUNTIME_QUALIFIED** | pinned `custom_rn.resnet18` + A3 `Linear(512, 7)`; live A6 interface |
| **E07c_MAIN_ARCHITECTURE_RUNTIME_QUALIFIED** | pinned `BeatGANsAutoencModel` via `get_model_conf().make_model()` |
| **E07c_SYNTHETIC_FORWARD_RUNTIME_QUALIFIED** | internal-stage connectivity, fourth-output discard, forward-only loss path |
| **IMPLEMENTED_NOT_EXECUTED** | E07c scientific method status unchanged |
| **CONTROLLED_ADAPTATION** | A1 identity-free + A3 controlled encoder reconstruction, DEV-021; A6 source-traceability only |

Reporting label: DiffFAS-BIN-IDFREE (controlled encoder reconstruction); not native, faithful or an official
reproduction. Source/contract: RESOLVED (A3). Execution artifact: the one auxiliary encoder is NOT_YET_TRAINED.

## 1. Why a corrective rerun was required

The initial M6D6a candidate produced passing GPU evidence. Afterwards, while its runtime log was being assembled on the
laptop, an **unquoted shell here-document** evaluated the prose text `git ls-files | xargs sha256sum` in the repository
root. That streamed the raw bytes of all 666 Git-tracked files through `sha256sum`, including **11 benchmark
manifest/metadata files** (`manifests/artifact_probe_classes_v1.json`, `difffas_bin_idfree_train_v1.parquet`,
`inventory.parquet`, `inventory_videos.parquet`, `m2_sample_accounting.parquet`, `pair_train_stats_v1.json`,
`pairs_train_v1.parquet`, `raw_file_index.parquet`, `split_groups_v1.parquet`, `split_v1.parquet`, `val_pairs_v1.parquet`).

Retained facts: no parquet/json row was parsed; no manifest semantics were consumed; no image was opened (tracked `data/`
directories contain only `.gitkeep`); no GPU qualification or test process accessed benchmark data; no scientific
execution, optimizer step or checkpoint occurred; the hash command ran **after** the GPU evidence existed; the generated
Python failed (SyntaxError), so the hash output was never written as project evidence. The initial candidate therefore
stopped with **PROCEDURAL_FIREWALL_DEVIATION** before its runtime log, audit JSON, preflight run, ledger append and index
rebuild. A session-level zero-access claim could not be made for it, which is what this corrective run establishes.

## 2. History preserved, not overwritten

SHA256 recorded **before** the rerun and re-verified afterwards (`sha256sum -c`: all OK):

| Initial-candidate file | SHA256 |
|---|---|
| `outputs/audit/M6D6A_E07C_RUNTIME_QUALIFICATION.md` | `8ce6b02cad29f448c6a369003f8f95d7002367a8263837820d00e41c94c6f611` |
| `outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_1.json` | `0ccc5a963e4eeb277478ecdde725ce9c39151ed96dafb29de8825f21afa4dab2` |
| `outputs/audit/M6D6A_E07C_SYNTHETIC_PROCESS_2.json` | `0ccc5a963e4eeb277478ecdde725ce9c39151ed96dafb29de8825f21afa4dab2` |
| `methods/difffas/runtime_qualification.py` (harness) | `1eeb7c037dbb25876eb6257a4eb0dee40bc3a998d5d20c03a2b4b937abe12541` |
| `tests/test_m6d6a_e07c_runtime.py` | `15e3fb28455c77094f039b1851e136da35b8bceab13def6f830b7f10a76496dc` |
| `tools/m6d6a_e07c_runtime_preflight.py` (initial; never run) | `b7f02e2cafcdabf418aeadb63fc45c73025b0f6d1081b51e620840cc1008be7d` |
| `environments/e07c.lock.json` | `0c909de1e3e3e8c129e0d9f4aab6386cee8e4eac0ca45d79423f795cced5f450` |
| `environments/e07c.runtime.json` | `0855943bb1aa00d12bf61f8145bdfad4598b467f8226b2cc05842902796d2d61` |
| `environments/e07c.conda-explicit.txt` | `2c5a2bbcacf767c54eb2f21ce2648a8bd265dbb57f6d0e0cdc01839d8b42b864` |
| `environments/e07c.pip-freeze.txt` | `42cebd80a0df9986b33fb09486f6dc619dd3695b1cbe0978cc67062cba9ea285` |
| `environments/e07c.pip-requirements.txt` | `9ea38ed8b8eb4de00dff577cfb7a1360201fa66dd6c28bf174a8bfb01de26f8f` |

None of these files was edited. The initial report remains the initial candidate's historical record; its wording
predates this correction. The initial preflight `tools/m6d6a_e07c_runtime_preflight.py` was never run and references a
runtime log and audit JSON that the initial candidate never produced. It is superseded by
`tools/m6d6a_r_e07c_clean_requalification_preflight.py`, which validates both the preserved initial evidence and this
corrective evidence. The initial session's captured command record (environment creation, attempts 1–4, first pair,
tests) and the deviation disclosure are preserved in Part B of `M6D6A_R_E07C_RUNTIME_LOG.txt`.

## 3. Starting state

- Laptop: `m6-baselines`, committed HEAD `596d8459c9f2e54a5f36e2463099b662b800eb7f` = `git ls-remote` remote. Worktree
  held exactly the 11 expected untracked candidate files; no tracked modification; nothing staged.
- GPU: `m6-baselines`, HEAD `596d845`, clean. No sync needed.
- Ledger 115 rows; artifact index 640 lines (639 rows); CONFIG_STATUS and STAGE_STATE unchanged.

## 4. Same environment, source and implementation (verified, not rebuilt)

- `gpat-m6-e07c` was **not recreated or mutated**. Fresh exports equal the locked ones: conda-explicit `2c5a2bbc…4b864`,
  pip freeze `42cebd80…ea285`; interpreter `83c01e86…028afe` = lock identity; `pip check`: no broken requirements.
  Python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0, cuDNN 92000, NumPy 2.4.6, tensorfn 0.1.28,
  pydantic 1.9.2; RTX 3090, capability 8.6, driver 595.84. The executed identity in both new processes equals the lock
  identity.
- Protected environments (base, gpat-m5, gpat-m6-e03, gpat-m6-e04-geometry, gpat-m6-e05, gpat-m6-e06c, stdn): fingerprints
  identical to the recorded post-creation values.
- Pinned DiffFAS `23f40519…`, tree `d190a5fb…`, GPU worktree empty; 24 explicit source-file hashes identical to the laptop
  cache; `custom_rn.py` `fa788c4d…`, `unet_autoenc.py` `127ecd59…`.
- Same harness (`1eeb7c03…`), same launcher (`run_q.sh` `31952506…`), same qualification_seed **60601**, same batch 4.
  No code, model, environment or harness change was made for this rerun.

## 5. Clean process results (two new fresh GPU processes)

Both `M6D6A_R_E07C_SYNTHETIC_PROCESS_1.json` and `_2.json` report `status = PASS` and are **byte-identical to each other
and to the initial candidate evidence** (SHA256 `0ccc5a963e4eeb277478ecdde725ce9c39151ed96dafb29de8825f21afa4dab2`):

- **Encoder:** pinned `custom_rn` `ResNet`, BasicBlock `[3,4,6,3]`, widths 64/256/512/512; the only structural difference
  from the pinned `resnet18()` is `fc` 17 → 7; 46,233,707 parameters; untrained, in memory only, no weights loaded.
- **Live A6 interface:** `[4,256,32,32]`, `[4,512,16,16]`, `[4,512,8,8]`, `[4,7]` — measured, not hard-coded. The historical
  pre-A6 text (32×32×128, 16×16×256) in the frozen YAML and A3 is untouched.
- **Main model:** `BeatGANsAutoencModel`, 159,363,974 parameters, 3 input channels, `use_pair = False`, EPSILON /
  LEARNED_RANGE / MSE, 1000 steps, linear β 1e-4 → 2e-2.
- **Connectivity:** x32x32 → `input_blocks.11.1`, `output_blocks.7.1`; x16x16 → `input_blocks.14.1`, `output_blocks.4.1`;
  x8x8 → `input_blocks.17.1`, `middle_block.1`, `output_blocks.1.1`. The fourth output has 0 consumers; NaN-filling it leaves
  all 45 traced internal stages bitwise identical and finite; zeroing each feature changes every consuming block.
- **Zero-initialised final projection:** the untrained model's final output is exactly 0 (pinned
  `resnet_use_zero_module=True`); it is recorded as expected source behaviour and is not used as conditioning evidence.
- **FORWARD_LOSS_PATH_ONLY:** `diffusion.training_losses` in the `FAS_train.py:57-68` form under `torch.no_grad`, model input
  `[4,3,256,256]`, finite loss (mean 1.0351096); no backward.
- **Repeatability:** both processes byte-identical (initial encoder aggregate `22bcb61b…`, main `325358fe…`); deterministic
  algorithms not forced.
- **SYNTHETIC_FORWARD_MEMORY_OBSERVATION:** 838,215,168 B after construction; peak 8,685,248,000 B allocated /
  9,252,634,624 B reserved (eval forwards); 8,697,844,736 B (forward-only loss path). Does not qualify training memory,
  scientific batch size or 400-epoch feasibility.

## 6. Corrective-session access audit (complete session, laptop + GPU)

Every command in the corrective session used explicit allowlisted paths. There was no `git ls-files | xargs`, no recursive
repository or runtime scan, and no shell evaluation of report prose (the runtime log was written by a Python file writer).
Git worktree checks used Git's index/status metadata, the established repository method.

| Count (complete corrective session) | Value |
|---|---:|
| benchmark manifest reads | **0** |
| benchmark image reads | **0** |
| TRAIN / VAL / TEST reads | **0 / 0 / 0** |
| firewall denials (2 GPU processes + 4 audited test runs) | **0** |
| `backward()` calls | **0** |
| `optimizer.step()` calls | **0** |
| scientific checkpoints created | **0** |
| auxiliary encoder training runs | **0** |
| scientific seed runs completed | **0 / 3** |

`<runtime_root>/runs` was absent before and after both processes and after cleanup: no `runs/m6/E07c` root, no
`aux_encoder/seed_42/checkpoints/encoder_final.pkl`, no empty directory created by tooling.

## 7. Tests (actual counts, all audited)

| Suite | Laptop (no Torch) | GPU (`gpat-m6-e07c`, CUDA hidden) |
|---|---|---|
| `tests.test_m6d6a_e07c_runtime` | 16 run, 15 pass, 1 skip (CPU construction needs Torch) | 16 run, 16 pass |
| `tests.test_m6c2b3_{contract,encoder,runtime}` (unchanged) | 46 run, 45 pass, 1 skip | 46 run, 46 pass |

Audit counts: laptop 625 / 2230 events, GPU 3177 / 4306 events; manifest 0, image 0, data/runs 0, denied 0 in every run. The
GPU staged copies were byte-identical to the laptop files and were removed afterwards; the GPU worktree is clean at
`596d845`.

## 8. Preflight, ledger, index

`python3 -I -S -B tools/m6d6a_r_e07c_clean_requalification_preflight.py` is static (no Torch, CUDA, model, manifest or
image). It validates the committed authority, the exact expected worktree, the preserved initial evidence hashes, source
pin and A6 mapping, environment lock, both corrective process records and their byte-equality with the initial evidence,
the session access audit and zero counters, the unchanged frozen inputs, the single ledger append and the CRLF index.
Ledger 115 → 116 (exactly one M6D6a row carrying both the initial deviation and this corrective requalification).
Artifact index rebuilt last from committed metadata plus the explicit new artifacts. Exact results are in the handoff.

## 9. Open future issues (recorded, not resolved here)

- **torch.load / weights_only:** the qualified runtime's `torch.load` defaults to `weights_only=True`. The future whole-module
  auxiliary checkpoint (`BeatGANsAutoencModel.encoder(path)`) and the sampling loader (a checkpoint dict containing a
  tensorfn/pydantic conf) need an explicit loader decision before any checkpoint is consumed. Not exercised here.
- **Precision policy:** qualification ran FP32 with TF32 disabled. The authoritative E07c training precision policy still
  needs an explicit decision before scientific training. No AMP/TF32 behaviour is invented here.

These do not affect the forward-only environment/runtime result.

## 10. Still not qualified

Real TRAIN path · training graph / backward · training memory · auxiliary encoder training · checkpoint writer · resume ·
production runner · scientific training (0/3 seeds) · M8 bank.

**NO BENCHMARK DATA ACCESS IN THE CORRECTIVE SESSION. NO OPTIMIZER STEP. NO BACKWARD. NO CHECKPOINT. NO AUXILIARY ENCODER
TRAINING. NO SCIENTIFIC TRAINING. NO SYNTHETIC BANK. NO COMMIT. NO PUSH.**
