# M6D5e — E06c DSDG-BIN-IDFREE: production runner + real-TRAIN end-to-end qualification

**M6D5e PASS_AFTER_CLEAN_REQUALIFICATION.** Classification: **PRODUCTION_RUNNER_QUALIFICATION · REAL_TRAIN_QUALIFICATION_ONLY**.

> **Correction (M6D5e-r).** This report documents the **initial** M6D5e qualification (seed 60505, root `q60505-834bd365c0e4e88a`). That run
> is technically successful, but it is **not** an unconditional clean PASS. Its status is
> **M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION**. During development, `manifests/split_v1.parquet` was parsed across all splits,
> including VAL and TEST metadata rows (§3). The initial run also omitted the upstream visualization block and so skipped its two CPU RNG draws.
> For that initial process the valid claims are **NO VAL/TEST IMAGE ACCESS · NO VAL/TEST TRAINING USE · NO VAL/TEST SELECTION USE ·
> NO TEST METRIC USE**. It did read VAL and TEST metadata. The M6D5e PASS rests on the **clean requalification** (seed 60506, new root, audited TRAIN-only
> firewall, upstream visualization RNG semantics preserved). See
> [`M6D5E_R_E06C_CLEAN_REQUALIFICATION.md`](M6D5E_R_E06C_CLEAN_REQUALIFICATION.md). The initial evidence files (`M6D5E_E06C_*`) are kept
> byte-for-byte. The code that ran them has since been superseded by the M6D5e-r files, whose SHA256 the clean processes recorded.

| Status | |
|---|---|
| **E06c_PRODUCTION_RUNNER_QUALIFIED** | `methods/dsdg/runner.py` + `tools/run_e06c.py` (CLI not launched) |
| **E06c_REAL_TRAIN_DATA_PATH_QUALIFIED** | frozen TRAIN relation → canonical faces_256 → IDFreePairDataset |
| **E06c_DETERMINISTIC_TRAIN_LOADER_QUALIFIED** | same seed ⇒ same epoch order (ID-only A = B = real epoch) |
| **E06c_CHECKPOINT_WRITER_QUALIFIED** | epoch-1 official event (source exception) + engineering sidecar |
| **E06c_RESUME_PATH_QUALIFIED** | fresh-process resume **bitwise** equal to the uninterrupted reference |
| **E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED** | 36 real B=240 global batches via V2 (12 × 20) |
| **E06c_TAIL_BATCH_198_EXECUTION_QUALIFIED** | real B=198 tail (9 × 20 + 18) |
| **E06c_TRAIN_ONLY_FIREWALL_QUALIFIED** | clean requalification only: 0 VAL/TEST metadata or image accesses in every process |
| **E06c_UPSTREAM_VISUALIZATION_RNG_SEMANTICS_QUALIFIED** | clean requalification only: `train_generator.py:197-211` executed, CPU RNG replay-proven |
| **E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED** | M6D5b historical truth, unchanged |
| **E06c_SCIENTIFIC_FULL_TRAINING_NOT_YET_EXECUTED** | no seed 42/1337/2026 run exists |
| **E06c REMAINS IMPLEMENTED_NOT_EXECUTED** | for scientific-result status |
| **CONTROLLED_ADAPTATION PRESERVED** | Amendment A1, DEV-020; CONTROLLED_EXECUTION_ADAPTATION |

Reporting label: DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation). This is not native, faithful or official DSDG.

**Initial run: REAL TRAIN DATA ACCESSED FOR QUALIFICATION. QUALIFICATION OPTIMIZER STEPS EXECUTED** (37 in epoch 1, then 1 in-memory
reference step and 1 fresh-process step). **NO VAL/TEST IMAGE ACCESS. NO VAL/TEST TRAINING USE. NO VAL/TEST SELECTION USE.
NO TEST METRIC USE.** VAL and TEST **metadata** was read during development (§3). **NO SCIENTIFIC FULL TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC
BANK. NO DOWNSTREAM EVALUATION. NO COMMIT. NO PUSH.** Every qualification artifact is **QUALIFICATION_ONLY · NOT_SCIENTIFIC_CHECKPOINT ·
NOT_ELIGIBLE_FOR_SYNTHETIC_BANK · NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION · NOT_ELIGIBLE_FOR_REPORTING**. None of the losses below is a
benchmark result, and nothing was retuned on them.

## 1. Authority and GPU synchronization

- **Laptop.** HEAD = origin/m6-baselines = `cd125dbdd1b45f3963fc501c5f7dc6ff9091d536` ("M6D5d: qualify E06c tail batch execution"). Divergence
  was 0 0 and the worktree was clean. `AGENTS.md` was checked with `test -f`: absent. The ledger had 114 rows and the index 617 rows.
- **GPU.** It started at `666aba4` (clean, 0 0). We fetched only `refs/heads/m6-baselines` → `cd125db`, confirmed ancestry (distance 1) and ran
  `git merge --ff-only`. The result was `cd125db`, 0 0, clean. No reset, clean, stash, commit or push was used.
- **Staging.** The M6D5e files were copied to the GPU as temporary untracked copies, with SHA256 equal on both hosts, and removed at the end.

## 2. Identities

| | |
|---|---|
| source | JDAI-CV/FaceX-Zoo `16b793a7564a4b9308cf94e62bdb2ffacb3a725a`, tree `0d2216bd…`; closure equals the lock; cache unmodified (before = after) |
| environment | `gpat-m6-e06c`, lock `91416a20…4e95`; runtime identity equals the lock (python 3.11.16, torch 2.12.1+cu130, CUDA 13.0, cuDNN 92000, RTX 3090, driver 595.84) |
| LightCNN-29 v2 | `d0750746…9964`, 123,844,849 bytes, 60/60 matched, frozen, eval; unchanged before/after |
| frozen config | `configs/methods/e06c_dsdg_bin_idfree.yaml` `7176dd4c…`; A1 `a0f84fac…`; V1 overlay `aa9d7089…`; V2 overlay `f1ea0543…`; run_logging_v1 `d3362294…` |
| M6D5e contract | `configs/amendments/e06c_m6d5e_production_runner_contract.yaml` `e4420abb…` |
| executed code | every GPU process recorded the SHA256 of the M6D5e files it ran. They equal the retained files: runner_io `36c16e02…`, runner `60cb3cf4…`, runner_qualification `10b923f9…`, run_e06c `9246c3b9…` |

## 3. TRAIN relation and identity firewall

The relation is `manifests/pairs_train_v1.parquet`. Its SHA256 `a5e4fdae…c243` was checked **before** parsing. It has 8838 rows, all
**TRAIN**: CASIA 2520, MSU 1200, SiW 5118. It contains 8838 unique sources, 3830 unique targets and 12,668 unique sample_ids. Only the 8
allowlisted columns were materialized (`pyarrow.read_table(columns=…)`). The file contains `source_subject` and `target_subject`, and neither
was read (`subject_columns_read = []`). Every row passed through the unchanged `DSDGAdapter.project_pair`. A focused test replaces or nulls the
subject columns and shows the projected relation is unchanged. The runner fails closed on a SHA mismatch, a wrong row count, a non-TRAIN row,
an unknown dataset or a missing column. Each case is tested.

**Canonical lookup (established, not invented).** The lookup is `faces_256_root/<dataset>/<sample_id>.png`. The M2 writer defines it
(`tools/m2b_run.py::_face_path`), and the M4 pair executor and M5 probe read the same path. The PNGs were encoded with `cv2.imencode(RGB[:, :, ::-1])`.
The faces root is `/home/student20261/workdir/GPAT_TransferBench_runtime/data/processed/faces_256`. It comes from the storage block of
`configs/execution/m5_gpu_3090.yaml` (`c9d22abf…`) through `m2b.resolve_roots` containment, and it equals `<runtime_root>/data/processed/faces_256`.
The dataset is bound from the allowlisted pair column. The reader is `reader(sample_id)` only. It checks the 64-hex pattern and TRAIN-relation
membership, refuses symlinks and path escape, opens with `O_NOFOLLOW`, and decodes with `IMREAD_UNCHANGED` to uint8 [256,256,3] with the
channels reversed to RGB. It does no crop, resize, augmentation or jitter; the only normalization is the adapter's `/255`.

**Real TRAIN accesses.** The epoch consumed 8838 pair rows, and each row read 2 canonical faces (source + target), so 17,676 face reads covering
the 12,668 distinct TRAIN sample files. The resume probe steps read one more epoch-2 batch in each process. The DataLoader workers also
prefetched some epoch-2 batches that were never trained on. An audit-hook firewall was active in every process. It allowed only
`pairs_train_v1.parquet` (opened twice: SHA and contract) and faces whose `<dataset>/<id>.png` belongs to the TRAIN relation. It denied every other
manifest, faces_256 enumeration, `<runtime_root>/runs`, the repository data roots, and any write outside the qualification/build/TMPDIR roots.
**0 denials** were recorded in all three processes.

**DISCLOSED LAPTOP METADATA READ (M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION).** While resolving the lookup, `manifests/split_v1.parquet` was parsed once on the laptop. The columns were
sample_id, dataset, split, sha256 and sha256_kind, over all 20,640 metadata rows, which include VAL and TEST rows. Only `Counter(sha256_kind)` was
printed. This established that the pair `source_sha256`/`target_sha256` are frame hashes, not face-PNG hashes. No VAL or TEST sample, image or
face was opened or used, and neither the runner nor the harness has any code path to that file. This read nevertheless violated the stricter
M6D5e rule that the split manifest must not be opened, so the initial run is not labelled an unconditional clean PASS. M6D5e-r repeated the
qualification without opening that file at any point, under an audit that logs every benchmark-path access.

## 4. DataLoader and determinism

The loader is `DataLoader(batch_size=240, shuffle=True, num_workers=8, pin_memory=True, drop_last=False,
generator=torch.Generator().manual_seed(60505), worker_init_fn=seed_torch_worker)`. Verified in-process:
- `RandomSampler(replacement=False)` on that generator;
- `BatchSampler(240, drop_last=False)`;
- 37 batches, `persistent_workers=False`.

| Loader evidence (ID-only process, no image bytes) | construction A | construction B |
|---|---|---|
| epoch-1 pair-order SHA256 | `c1f45cff5e83b6aeac08c14e467b2b0e07468faabc99f49fd34523c2b0e63800` | identical |
| epoch-2 first-batch SHA256 | `bacdb1939312e3ca3eee0b372c4c6fbbc4696366e75a28747ea6af5bd8ee1e80` | identical |
| batch sizes | [240]×36 + [198] | identical |

The real training epoch observed the **same** epoch-1 order hash `c1f45cff…`. Both resume continuations drew the epoch-2 first batch
`bacdb193…`, as the ID-only loader predicted. Epochs 1 and 2 differ.

## 5. Real qualification epoch (seed 60505, qualification root only)

The qualification root is `<runtime_root>/qualification/m6d5e/E06c/q60505-834bd365c0e4e88a/` (run_id `834bd365c0e4e88a`). It did not exist before
the run. `<runtime_root>/runs/m6/E06c` was never created or opened.

| Gate | Observed |
|---|---|
| pair rows consumed / unique / dropped / duplicated | 8838 / 8838 / 0 / 0 |
| split | TRAIN only |
| global batch sizes | [240]×36 + [198] |
| optimizer applications = global steps | 37 = 37 |
| backward calls | 442 = 36 × 12 + 10 |
| B=198 tail chunks | [20, 20, 20, 20, 20, 20, 20, 20, 20, 18] |
| `zero_grad(set_to_none=False)` | once per global batch (38 in the process incl. the reference step; only `False` used) |
| owned gradients | 55/55 non-None and finite at **every** step (45,370,182 parameters) |
| netCls | excluded from Adam; gradient exactly 0 every step; bytes unchanged over the epoch |
| netIP | frozen, eval, no gradient; bytes unchanged |
| losses | all finite; `loss_cls = loss_pair = 0`; every total equals the pinned epoch-1 warmup assembly |
| CUDA OOM | none |
| peak VRAM | 6,383,762,432 bytes allocated (per-step peak; nvidia-smi ≈ 7.5 GiB during the run) |
| wall time | 151.13 s for the epoch (3.1–5.2 s per global step) |
| VAL / TEST (GPU processes) | not opened by the runner or harness (firewall: 0 denials); VAL/TEST **metadata** was read on the laptop during development (§3) |
| upstream visualization block | **not executed** (its two CPU `normal_` draws were omitted); corrected in M6D5e-r |
| warnings | none |

Global losses per optimizer step, from qualification epoch 1 (`epoch < 2` warmup assembly). These are engineering values,
**NOT_ELIGIBLE_FOR_REPORTING**:

| step | B | loss_rec | loss_kl | loss_mmd | loss_ip | loss_ort | loss_cls | loss_pair | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 240 | 15208.749349 | 18.266820 | 15.459603 | 7.425972 | 0.010480 | 0.0 | 0.0 | 15209.160978 |
| 2 | 240 | 13117.097738 | 63.020750 | 47.794598 | 7.500260 | 0.627251 | 0.0 | 0.0 | 13118.287166 |
| 3 | 240 | 11447.481201 | 115.292019 | 67.463799 | 7.470228 | 10.529545 | 0.0 | 0.0 | 11449.488757 |
| 4 | 240 | 10456.921794 | 159.390733 | 79.602425 | 7.436131 | 21.455074 | 0.0 | 0.0 | 10459.600637 |
| 5 | 240 | 10117.280924 | 195.025497 | 88.034790 | 7.443586 | 29.917263 | 0.0 | 0.0 | 10120.485136 |
| 6 | 240 | 9605.049398 | 213.008695 | 90.154854 | 7.433781 | 27.360308 | 0.0 | 0.0 | 9608.428974 |
| 7 | 240 | 9048.004598 | 223.881821 | 88.889999 | 7.325072 | 28.794476 | 0.0 | 0.0 | 9051.493512 |
| 8 | 240 | 8622.848145 | 229.117339 | 83.894676 | 7.335847 | 26.359194 | 0.0 | 0.0 | 8626.315215 |
| 9 | 240 | 8574.435954 | 235.805405 | 80.328796 | 7.363811 | 24.268780 | 0.0 | 0.0 | 8577.913622 |
| 10 | 240 | 8903.951213 | 245.254140 | 77.419380 | 7.331817 | 27.931526 | 0.0 | 0.0 | 8907.530581 |
| 11 | 240 | 8285.411540 | 255.824273 | 75.727654 | 7.209949 | 23.020645 | 0.0 | 0.0 | 8289.029365 |
| 12 | 240 | 7617.119019 | 272.168966 | 75.251595 | 7.297825 | 26.526936 | 0.0 | 0.0 | 7620.931472 |
| 13 | 240 | 7853.495646 | 284.481244 | 75.000313 | 7.249366 | 21.854038 | 0.0 | 0.0 | 7857.381496 |
| 14 | 240 | 7827.678711 | 300.471146 | 76.620903 | 7.194248 | 25.249418 | 0.0 | 0.0 | 7831.774068 |
| 15 | 240 | 7533.365438 | 306.837479 | 75.817337 | 7.209012 | 20.538424 | 0.0 | 0.0 | 7537.469460 |
| 16 | 240 | 7156.420898 | 322.303263 | 76.134666 | 7.238068 | 17.603561 | 0.0 | 0.0 | 7160.653694 |
| 17 | 240 | 7345.141235 | 331.594035 | 74.920074 | 7.151296 | 15.611443 | 0.0 | 0.0 | 7349.434004 |
| 18 | 240 | 7047.457113 | 343.543526 | 77.166977 | 7.144710 | 18.623043 | 0.0 | 0.0 | 7051.921895 |
| 19 | 240 | 6777.699666 | 352.265427 | 78.771576 | 7.119675 | 16.701731 | 0.0 | 0.0 | 6782.248250 |
| 20 | 240 | 6757.335327 | 363.103081 | 81.002846 | 7.119566 | 21.090908 | 0.0 | 0.0 | 6762.058491 |
| 21 | 240 | 6219.639323 | 374.359151 | 81.232849 | 6.984946 | 15.421666 | 0.0 | 0.0 | 6224.419309 |
| 22 | 240 | 6760.179118 | 377.682574 | 81.193604 | 6.981260 | 16.926039 | 0.0 | 0.0 | 6765.006953 |
| 23 | 240 | 6355.829346 | 385.752131 | 82.893074 | 7.046514 | 15.593414 | 0.0 | 0.0 | 6360.742197 |
| 24 | 240 | 6708.855225 | 391.740059 | 82.931808 | 6.993168 | 11.478176 | 0.0 | 0.0 | 6713.786657 |
| 25 | 240 | 6380.581014 | 399.682083 | 83.248390 | 6.968562 | 13.636652 | 0.0 | 0.0 | 6385.616371 |
| 26 | 240 | 6445.425334 | 401.115941 | 80.029007 | 6.932689 | 10.330569 | 0.0 | 0.0 | 6450.409416 |
| 27 | 240 | 6266.764648 | 406.771472 | 80.779594 | 6.977217 | 9.054579 | 0.0 | 0.0 | 6271.800477 |
| 28 | 240 | 6619.361491 | 403.581662 | 79.628525 | 6.939395 | 7.516115 | 0.0 | 0.0 | 6624.338148 |
| 29 | 240 | 6361.505941 | 416.048080 | 79.139503 | 6.912201 | 5.915431 | 0.0 | 0.0 | 6366.586093 |
| 30 | 240 | 6134.307536 | 421.962059 | 79.061737 | 6.860570 | 8.014508 | 0.0 | 0.0 | 6139.466525 |
| 31 | 240 | 5957.459391 | 422.913241 | 77.974167 | 6.845154 | 5.439201 | 0.0 | 0.0 | 5962.591109 |
| 32 | 240 | 6060.826701 | 425.750966 | 78.040970 | 6.794949 | 3.970677 | 0.0 | 0.0 | 6065.972276 |
| 33 | 240 | 6107.871785 | 425.359189 | 77.273407 | 6.869349 | 5.213633 | 0.0 | 0.0 | 6113.018941 |
| 34 | 240 | 6116.402303 | 424.593770 | 73.354218 | 6.825174 | 0.698228 | 0.0 | 0.0 | 6121.457017 |
| 35 | 240 | 5906.556600 | 421.187721 | 71.873344 | 6.851697 | 4.816491 | 0.0 | 0.0 | 5911.603892 |
| 36 | 240 | 5774.689250 | 427.601034 | 73.035027 | 6.813761 | 4.003451 | 0.0 | 0.0 | 5779.803782 |
| 37 | 198 | 5973.317861 | 422.288741 | 70.789536 | 6.790254 | 5.259215 | 0.0 | 0.0 | 5978.369139 |

## 6. Epoch-1 qualification checkpoint (source `epoch == 1` exception)

The pinned `misc/util.py::save_checkpoint` wrote `{'epoch': 1, 'model': <module>}`. Each file was reloaded and its state_dict equals the
in-memory module bitwise. They are indexed in `checkpoint_index.json` as `periodic`, `selected_for_final = false`, with reason "QUALIFICATION_ONLY …
NOT_SCIENTIFIC_CHECKPOINT; never a selection candidate".

| file (qualification root) | bytes | SHA256 |
|---|---:|---|
| `checkpoints/netE_spoof_model_epoch_1_iter_0.pth` | 55,181,976 | `87c54089b41f761caa3256f872dbc87f2a0e6ecd41bb967923fbbbc38665495b` |
| `checkpoints/netE_live_model_epoch_1_iter_0.pth` | 46,792,321 | `826df6cbe8e810b2796acd6e9ec72383445ced2faef6c8a1a84650d989ac072c` |
| `checkpoints/netG_model_epoch_1_iter_0.pth` | 79,577,546 | `6232ce824a7718a5fda498b786892d744b445754dddffa1dfb7d9f5fbfaf9f65` |

**Engineering resume sidecar.** The file is `checkpoints/runner_state_epoch_1.pth`: 544,543,075 bytes, SHA256
`bcc04958a4addfc69af6c5b0ba65566bd752480ef3c76d06f75eebef6511aa51`. It is indexed separately in `resume_state_index.json` with
`scientific_checkpoint = false` and `selection_candidate = false`.
- **Inventory:**
  - completed_epoch 1 and global_step 37;
  - netE_nir, netE_vis, netG and netCls state_dict;
  - Adam state_dict (55 entries);
  - Python, NumPy, torch-CPU and torch-CUDA RNG;
  - the DataLoader generator state;
  - resolved_config_sha256 `a1d5cf78…`, all identities, execution_mode and the trajectory-record count.
- **netIP** is NOT stored; it is reloaded from the SHA-verified asset.
- **Reload check.** The file was reloaded with `weights_only=True`, and its models, Adam state, RNG state and generator are all equal to the
  in-memory state.

The planned epoch-200 behaviour (netG `selected` under OFFICIAL_GENERATOR_EPOCH_200, netE_* `terminal`, cadence 1, 10, …, 200) is covered by
focused tests and by the unchanged `learned.checkpoint_metadata` validation. It was not executed.

## 7. Resume equivalence (epoch 2, step 1)

- **Reference.** In the train process, after the epoch-1 checkpoint event, ONE epoch-2 global step ran from the in-memory state. It was not
  logged to metrics.
- **Fresh process.** A new process loaded `checkpoints/runner_state_epoch_1.pth`, given by explicit path and matched by SHA against the index.
  It reconstructed LightCNN from the verified asset and restored every RNG and the loader generator. It reopened the RunContext in resume mode,
  which appended `resume_reconciliation`, and confirmed that metrics hold steps 1..37 with the last epoch 1. It then ran ONE epoch-2 global
  step and appended it as global_step 38.

| Compared | Reference | Fresh process |
|---|---|---|
| next-batch pair-id SHA256 | `bacdb193…1e80` | identical |
| global batch / chunks | 240 / 12 × 20 | identical |
| ε_cls / ε_nir / ε_vis SHA256 | `2c59f826…` / `ca715134…` / `d25c3d8d…` | identical |
| pass-1 delta / ort_mean / signs | — | identical SHA256 |
| losses (rec, kl, mmd, ip, ort) | 5555.263387044271, 431.3868509928385, 70.35545349121094, 6.8101963202158595, 5.313945293426514 | identical |
| total (post-warmup assembly, epoch 2) | 6069.129833141964 | identical |
| gradient inventory / 55 per-tensor gradient SHA256 | 55/55 finite | identical |
| Adam state (step 38.0; exp_avg / exp_avg_sq SHA256 × 55) | — | identical |
| post-step parameters (all five models) | — | identical SHA256 |
| RNG before / before step / after | — | identical |
| tensor diff (gradients, parameters) | — | 55/55 bitwise, max \|Δ\| = 0.0, cosine 1.0 |

**All 24 pre-registered fields are BITWISE identical.** The contract fixed this gate before the run, because M6D5d had observed
cross-process bitwise identity. A non-bitwise result would have been STOP_AND_REPORT, and no tolerance was used. The restored parameters,
Adam state, RNG and generator also equal the end-of-epoch-1 hashes from the train process. No second epoch ran beyond this single step in each
process.

## 8. Logging, checkpoint index, manifest, resolved config

- **`resolved_config.yaml`** SHA256 is `a1d5cf78f3c89fdf341cf04573c6f613008c2b74b6c2d3313036e39900283788`. It holds the frozen config verbatim
  plus a `_resolved` block with the A1 identity, the M6D5c/M6D5d/M6D5e overlay paths and hashes, the source pin, the lock, LightCNN, and the
  visible adaptations (OOM_RETAINED, 240/20/198, drop_last false). On resume it was re-derived byte-for-byte.
- **`run_manifest.json`** has every run_logging_v1 field, plus:
  - `runner_mode = QUALIFICATION`, `qualification_only = true`, `scientific_run = false`;
  - `experiment_seed = null` (with a reason) and `qualification_seed = 60505`;
  - `git_commit = cd125db`, `git_dirty = true` (uncommitted additive M6D5e files, whose SHA256 are listed);
  - `execution_mode = GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2`, `physical_batch_240 = OOM_RETAINED`, 240/20/198, `drop_last = false`;
  - the start and end UTC and the command line.
- **run_id** is `834bd365c0e4e88a` = sha256("E06c|60505|7176dd4c…|cd125db…")[:16]. The existing repository RunContext rule is preserved.
  The resolved-config SHA is recorded as an additional bound identity rather than as a second run-id convention.
- **`metrics.jsonl`** has 43 lines and is append-only. There are **38 trajectory records** (37 for epoch 1 plus the resumed epoch-2 step), one
  per global optimizer step, with contiguous steps 1..38. Each record carries the seven losses + total_loss, lr 2e-4, the wall clock, the per-step
  GPU memory, the batch size, the microbatch sizes, the batch pair-id SHA256, the ε SHA256 and the qualification labels. `train_metrics`,
  `val_losses` and `val_metrics` are explicit `null` with reasons. The 5 event records are `e06c_run_start`, `e06c_epoch_complete`,
  `e06c_checkpoint_event`, `resume_reconciliation` and `e06c_resume_reconciliation`. The resumed process only appended: the prior bytes are a
  prefix of the file. No trajectory record contains a TEST field. The only `test` token in the file is the pre-existing negative assertion
  `test_split_accessed: false`, inside the prior-manifest and prior-summary snapshots that the unchanged RunContext copies into its
  `resume_reconciliation` event.
- **`checkpoint_index.json`** lists the 3 official files with their actual size and SHA256, `selected_final: null`, and the qualification labels.
- **`run_summary.json`** has `completion_status = interrupted`, with the reason "QUALIFICATION_STOP_AFTER_RESUME_PROBE_STEP (planned; not a
  failure)". There is no final or selected checkpoint.
- **stdout/stderr.** `stdout.log` holds the 3 pinned "Checkpoint saved" lines; `stderr.log` is empty.

## 9. Production CLI (not launched)

`tools/run_e06c.py` hard-binds E06c, seeds {42, 1337, 2026}, 200 epochs, the frozen relation, V2 and `runs/m6/E06c/seed_<seed>/`. Before any
Torch import it refuses:
- any tuning or override flag: epochs, lr, batch, microbatch, drop_last, AMP/FP16/BF16, TF32, checkpoint selection, cadence, λ, qualification,
  "latest" resume;
- seed 60505 or any other non-experiment seed;
- a dirty worktree;
- a wrong launch environment;
- identity drift: contract, overlays, source, lock, pair manifest, LightCNN or execution config;
- an occupied seed directory without a consistent explicit `--resume-state`.

The focused tests cover these refusals. The CLI was **not** run in any mode.

## 10. Engineering incident (disclosed)

The first ID-only loader attempt deadlocked. DataLoader workers exchange tensor file descriptors through a multiprocessing AF_UNIX socket
under `TMPDIR`. The build-root path pushed that socket past the 108-byte limit (`OSError: AF_UNIX path too long`). The main process was
terminated after about 11 minutes. That mode reads no image bytes, did no GPU work and created no qualification root. The harness now requires
the short dedicated `TMPDIR=/tmp/gpat-m6d5e`, which the firewall allows. No scientific or data setting changed. During test development, two
test-fixture defects were fixed: toy hdim 8 versus the fixed ε width 128, and a toy netIP that was not deterministic. The runner was not changed
for either.

## 11. Tests and static preflight (initial run; superseded)

The counts, preflight and ledger statements below record the initial run. The current preflight validates the initial evidence as frozen
history and fully validates the clean requalification. The single M6D5e ledger row was corrected before commit (see the M6D5e-r report).

- **Tests.** `tests/test_m6d5e_e06c_runner.py` has 27 tests: 19 static, 4 engine (CPU toy networks with exactly 55 owned tensors) and 4 retained
  evidence. Focused M6D5a–e suite:
  - GPU (`gpat-m6-e06c`, `CUDA_VISIBLE_DEVICES=""`): **131 passed**;
  - laptop (no Torch): **131 run, OK, 31 skipped**.

  No existing test was modified, and the full repository suite was not run.
- **Preflight.** `python3 -I -S -B tools/m6d5e_e06c_runner_preflight.py --rebuild-index` passes. It is static: no torch, pyarrow or cv2 import, no
  parquet payload, no image, no model. It re-derives the 24 bitwise resume fields independently. It checks that the executed code bytes equal the
  retained files and that the M6D5a–d artifacts are byte-identical to `cd125db`. It verifies the single ledger append and the CRLF index.
- **Ledger.** 114 → **115** rows; the first 114 are byte-identical. Appended row: `M6D5E_E06C_PRODUCTION_RUNNER_QUALIFICATION`.
- **Index.** Rebuilt last, preserving CRLF.

Qualification weights (official files, sidecar, and the 362,997,231-byte `qualification_probes/reference_epoch2_step1.pt`) exist only under the
GPU qualification root. None of them is in Git or under `runs/m6/E06c/`.
