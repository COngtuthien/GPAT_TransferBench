# M6D5e-r — E06c clean firewall + upstream RNG-side-effect requalification

**M6D5e PASS_AFTER_CLEAN_REQUALIFICATION.** Classification: **PRODUCTION_RUNNER_QUALIFICATION · REAL_TRAIN_QUALIFICATION_ONLY ·
CORRECTIVE_REQUALIFICATION**.

| Status | Basis |
|---|---|
| **E06c_PRODUCTION_RUNNER_QUALIFIED** | clean run through `methods/dsdg/runner.py` (the production CLI `tools/run_e06c.py` is unchanged and not launched) |
| **E06c_REAL_TRAIN_DATA_PATH_QUALIFIED** | frozen TRAIN relation → canonical faces_256 → IDFreePairDataset; 12,668 distinct TRAIN faces read |
| **E06c_DETERMINISTIC_TRAIN_LOADER_QUALIFIED** | seed 60506: ID-only constructions A = B = real epoch order |
| **E06c_CHECKPOINT_WRITER_QUALIFIED** | epoch-1 official event (source exception) + engineering sidecar, reloaded equal |
| **E06c_RESUME_PATH_QUALIFIED** | fresh-process resume **bitwise** equal to the uninterrupted reference (24/24 fields, 55/55 tensors) |
| **E06c_TRAIN_ONLY_FIREWALL_QUALIFIED** | split manifest never opened; audited 0 VAL/TEST metadata and 0 VAL/TEST image accesses in every process |
| **E06c_UPSTREAM_VISUALIZATION_RNG_SEMANTICS_QUALIFIED** | `train_generator.py:197-211` executed; CPU RNG replay-proven; sidecar carries post-block state |
| **E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED** · **E06c_TAIL_BATCH_198_EXECUTION_QUALIFIED** | 36 × B=240 (12 × 20) + B=198 (9 × 20 + 18) |
| **E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED** | M6D5b historical truth, unchanged |
| **E06c_SCIENTIFIC_FULL_TRAINING_NOT_YET_EXECUTED** | no seed 42/1337/2026 run exists |
| **E06c REMAINS IMPLEMENTED_NOT_EXECUTED** | for scientific-result status |
| **CONTROLLED_ADAPTATION PRESERVED** | Amendment A1, DEV-020; CONTROLLED_EXECUTION_ADAPTATION |

Reporting label: DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation). This is not native, faithful or official DSDG.
Every qualification artifact is **QUALIFICATION_ONLY · NOT_SCIENTIFIC_CHECKPOINT · NOT_ELIGIBLE_FOR_SYNTHETIC_BANK ·
NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION · NOT_ELIGIBLE_FOR_REPORTING**.

This milestone has two runs. The table keeps them apart:

| | **Initial development run (M6D5e, retained)** | **Clean qualification rerun (M6D5e-r)** |
|---|---|---|
| status | **M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION** | **PASS** (all gates) |
| qualification seed / root | 60505 / `q60505-834bd365c0e4e88a` | 60506 / `q60506-cdaeea956b723317` (new; did not exist before) |
| evidence | `M6D5E_E06C_*` (byte-for-byte unchanged) | `M6D5E_R_E06C_*` |
| real TRAIN data used | true | true |
| VAL metadata read | **true** (during laptop development) | **false** (0 accesses, audited) |
| TEST metadata read | **true** (during laptop development) | **false** (0 accesses, audited) |
| VAL / TEST image bytes accessed | false / false | false / false (0 non-TRAIN face accesses, audited) |
| VAL / TEST used for optimization | false / false | false / false |
| VAL / TEST used for checkpoint selection | false / false | false / false |
| TEST metrics computed | false | false |
| upstream visualization block | omitted (two CPU RNG draws skipped) | executed, RNG semantics proven |
| valid access claim | **NO VAL/TEST IMAGE ACCESS · NO VAL/TEST TRAINING USE · NO VAL/TEST SELECTION USE · NO TEST METRIC USE** | **NO VAL ACCESS DURING CLEAN REQUALIFICATION · NO TEST ACCESS DURING CLEAN REQUALIFICATION** |

## 1. The initial procedural deviation (historical, not rewritten)

While the canonical face lookup was being developed on the laptop, `manifests/split_v1.parquet` was parsed once. The parse covered columns
`sample_id, dataset, split, sha256, sha256_kind` over all 20,640 rows, **including the VAL and TEST metadata rows**. Only
`Counter(sha256_kind)` was printed. No VAL or TEST sample, image or face was listed, opened or used. Nothing was optimized, selected or
evaluated on VAL or TEST. No TEST metric exists. The runner and the harness have no code path to that file. The GPU firewall recorded
0 denials. The stricter M6D5e prompt nevertheless forbade opening the split manifest, so the initial run keeps the status
**M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION**. Its report, evidence and runtime log are retained. The only changes to its report
are a correction banner and wording fixes: unqualified "no VAL/TEST access" claims were replaced with the four claims that are valid.
The disclosure itself is intact (`M6D5E_E06C_PRODUCTION_RUNNER_QUALIFICATION.md` §3 and `M6D5E_E06C_RUNTIME_LOG.txt`).

The historical runtime log is kept byte-for-byte. Its closing line, "no VAL/TEST access", is **superseded** by this report. It is accurate
only for image bytes, training, selection and metrics, not for metadata.

## 2. Corrective code changes (minimal; no scientific change)

Unchanged: architecture, losses, coefficients, optimizer, LR, batch policy, V2 microbatch formulas, checkpoint rule, scientific seeds
(42/1337/2026), the TRAIN manifest, the frozen config, A1, the M6D5c/M6D5d overlays and every M6D5a–d file. `tools/run_e06c.py` is
byte-identical (`9246c3b9…`).

| file | change | why |
|---|---|---|
| `methods/dsdg/runner_io.py` | qualification seed 60505 → **60506** (60505 recorded as the retained initial run and refused); visualization constants and cadence helpers | a new root (same HEAD and config would re-derive `q60505-834bd365…`); upstream visualization semantics |
| `methods/dsdg/runner.py` | `Trainer.visualize` (source block) and last-batch grid capture through the existing V2 `on_chunk` observer; the scientific loop calls it before `checkpoint_event`; resolved-config and manifest wording | upstream RNG side effect; accurate wording |
| `methods/dsdg/runner_qualification.py` | new build root `builds/e06c_dsdg_m6d5e_r`, TMPDIR `/tmp/gpat-m6d5er`, `M6D5E_R_*` outputs; the firewall logs every benchmark-path event (main process and forked workers) to an append-only TSV; `access_audit` gate; `visualization_point` RNG replay proof | strict audited TRAIN-only firewall; never overwrite initial evidence |
| contract, addendum, tests, preflight | M6D5e-r records and checks | documentation and verification |

The firewall needed no rule change: it already denied every manifest except the TRAIN relation. M6D5e-r adds **auditing**, meaning a count
of every access in every process, including DataLoader workers, whose reads the initial run could not count.

## 3. Firewall of the preparation and execution

- **`manifests/split_v1.parquet` was never opened or parsed** during M6D5e-r preparation or execution. That includes every test run and
  the preflight. The canonical lookup `faces_256_root/<dataset>/<sample_id>.png` is the fixed M2 convention. It was not rediscovered.
- The benchmark inputs were `manifests/pairs_train_v1.parquet` (SHA-checked before parsing) and the canonical TRAIN faces it references.
  Everything else read was immutable source, config, environment or model assets. The preparation read these files by exact path: the
  M6D5e sources, contract, addendum and reports; the pinned `train_generator.py` and `networks/generator.py`; and `microbatch_execution{,_v2}.py`.
- Every test run (laptop and GPU) ran under an audit wrapper. It denied and logged any open under `manifests/` other than the TRAIN
  relation, and any open under repository or runtime `data/`. The logs show TRAIN-relation opens only, with 0 denials. The test-only
  references to the `split_v1.parquet` path string pass it to a firewall callable, which refuses it; the file is never opened.

## 4. Clean GPU execution (RTX 3090, `gpat-m6-e06c`)

- **Sync.** The GPU worktree was at `cd125db`, clean. The six M6D5e-r files were staged as temporary untracked copies with SHA256 verified
  on both hosts, and removed afterwards. The worktree is clean again at `cd125db`.
- **Launch environment (every process).** `CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=60506 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TMPDIR=/tmp/gpat-m6d5er`.
- **Environment.** Runtime identity equals lock `91416a20…`: python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0,
  cuDNN 92000, driver 595.84. The GPU used 254 MiB with 0 compute processes before and after each process.
- **Executed code.** Every process recorded the SHA256 of the M6D5e files it ran, and they equal the retained files: contract `4123c93f…`,
  runner_io `262455ea…`, runner `b558ff56…`, runner_qualification `bddaad6c…`, run_e06c `9246c3b9…`.

| process (UTC 2026-09-25) | result |
|---|---|
| loader 12:52:10–12:52:12, pid 872388 | PASS: ID-only, no image bytes; A = B; epoch-1 order `f9ebe00c…b785`; epoch-2 first batch `d20a11e5…ebe3`; epochs differ |
| train 12:52:30–12:55:37, pid 872901 | PASS: 1 real-TRAIN epoch, visualization block, checkpoint event, 1 in-memory epoch-2 reference step |
| resume 12:56:08–12:56:24, pid 874948 | PASS: explicit sidecar, 1 epoch-2 step, 24/24 fields bitwise equal |

## 5. Real TRAIN qualification epoch (seed 60506)

The root is `<runtime_root>/qualification/m6d5e/E06c/q60506-cdaeea956b723317/` (run_id `cdaeea956b723317` = sha256("E06c|60506|7176dd4c…|cd125db…")[:16]).
`<runtime_root>/runs/m6/E06c` was never created or opened. The initial root `q60505-834bd365c0e4e88a` was not touched.

| Gate | Observed |
|---|---|
| TRAIN manifest | `a5e4fdae…c243` checked before parsing; 8838 rows, all TRAIN; 8 allowlisted columns; subject columns not read |
| pair rows consumed / unique / dropped / duplicated | 8838 / 8838 / 0 / 0 |
| global batch sizes | [240]×36 + [198]; epoch order `f9ebe00c…` = ID-only loader |
| optimizer applications = global steps | **37** |
| backward calls | **442** = 36 × 12 + 10 (process total 454 including the reference step) |
| B=198 tail chunks | [20 × 9, 18] |
| owned gradients | **55/55** non-None and finite at every step |
| netCls / netIP | excluded from Adam, gradient exactly 0, bytes unchanged / frozen, eval, no gradient, bytes unchanged |
| losses | all finite; `loss_cls = loss_pair = 0`; every total equals the pinned epoch-1 warmup assembly |
| CUDA OOM | none; per-step peak allocated 6,383,762,432 bytes |
| wall time | 150.83 s for the epoch |
| warnings | none |

Qualification epoch-1 global losses (engineering values, **NOT_ELIGIBLE_FOR_REPORTING**):

| step | B | loss_rec | loss_kl | loss_mmd | loss_ip | loss_ort | loss_cls | loss_pair | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 240 | 16570.725505 | 19.340703 | 15.425980 | 7.421959 | 1.878235 | 0.0 | 0.0 | 16571.166173 |
| 2 | 240 | 14359.323975 | 59.067189 | 44.177620 | 7.445607 | 0.086079 | 0.0 | 0.0 | 14360.431740 |
| 3 | 240 | 12151.607747 | 103.562771 | 58.778286 | 7.529543 | 0.323780 | 0.0 | 0.0 | 12153.309691 |
| 4 | 240 | 10941.372152 | 144.195286 | 70.275078 | 7.423991 | 4.159989 | 0.0 | 0.0 | 10943.632695 |
| 5 | 240 | 10381.779948 | 176.849183 | 75.268417 | 7.409537 | 1.617340 | 0.0 | 0.0 | 10384.391393 |
| 6 | 240 | 9582.883219 | 199.853723 | 79.482483 | 7.454955 | 4.633338 | 0.0 | 0.0 | 9585.797464 |
| 7 | 240 | 8788.942668 | 210.241182 | 78.964951 | 7.418560 | 0.494713 | 0.0 | 0.0 | 8791.913862 |
| 8 | 240 | 8637.146891 | 219.339152 | 75.669815 | 7.479831 | 0.074404 | 0.0 | 0.0 | 8640.172523 |
| 9 | 240 | 8579.776571 | 233.564929 | 74.240097 | 7.339727 | 4.410629 | 0.0 | 0.0 | 8582.972124 |
| 10 | 240 | 8110.331624 | 251.813661 | 73.789223 | 7.265656 | 7.549942 | 0.0 | 0.0 | 8113.735809 |
| 11 | 240 | 7770.470337 | 265.963549 | 77.142410 | 7.308719 | 7.192729 | 0.0 | 0.0 | 7774.046411 |
| 12 | 240 | 8104.947713 | 281.789584 | 78.460442 | 7.219572 | 0.792990 | 0.0 | 0.0 | 8108.630339 |
| 13 | 240 | 7836.790243 | 300.875407 | 78.447739 | 7.127091 | 11.354424 | 0.0 | 0.0 | 7840.768289 |
| 14 | 240 | 7475.859741 | 308.884791 | 76.563599 | 7.074555 | 13.759878 | 0.0 | 0.0 | 7479.922569 |
| 15 | 240 | 7280.514730 | 317.524239 | 78.542343 | 7.088505 | 7.641159 | 0.0 | 0.0 | 7284.622692 |
| 16 | 240 | 7270.273071 | 331.338432 | 78.397385 | 7.099919 | 17.787144 | 0.0 | 0.0 | 7274.619300 |
| 17 | 240 | 7378.581828 | 339.423444 | 77.220978 | 6.963345 | 7.397716 | 0.0 | 0.0 | 7382.891883 |
| 18 | 240 | 6932.794759 | 349.567464 | 80.128876 | 7.115050 | 6.386667 | 0.0 | 0.0 | 6937.226740 |
| 19 | 240 | 6887.800171 | 362.308856 | 83.608894 | 7.059433 | 12.519431 | 0.0 | 0.0 | 6892.455137 |
| 20 | 240 | 6859.252808 | 372.494537 | 82.281075 | 7.010077 | 3.793827 | 0.0 | 0.0 | 6863.908603 |
| 21 | 240 | 6793.901286 | 379.773369 | 85.558334 | 6.946070 | 10.062822 | 0.0 | 0.0 | 6798.724692 |
| 22 | 240 | 6632.451538 | 388.653509 | 87.459732 | 6.991294 | 5.863521 | 0.0 | 0.0 | 6637.341219 |
| 23 | 240 | 6601.776001 | 387.933533 | 83.644379 | 6.970019 | 1.855493 | 0.0 | 0.0 | 6606.580035 |
| 24 | 240 | 6673.367716 | 395.671084 | 85.875687 | 6.954124 | 1.710966 | 0.0 | 0.0 | 6678.269835 |
| 25 | 240 | 6574.000203 | 401.073120 | 86.035103 | 7.012634 | 2.773013 | 0.0 | 0.0 | 6578.969142 |
| 26 | 240 | 6178.233236 | 403.737612 | 88.494293 | 7.016829 | 2.684875 | 0.0 | 0.0 | 6183.252572 |
| 27 | 240 | 6356.909790 | 405.778285 | 83.716873 | 7.018971 | 4.431090 | 0.0 | 0.0 | 6361.919242 |
| 28 | 240 | 6174.594238 | 407.986689 | 85.093323 | 6.936851 | 7.983079 | 0.0 | 0.0 | 6179.674238 |
| 29 | 240 | 5992.662272 | 416.937319 | 82.954178 | 7.000048 | 6.721283 | 0.0 | 0.0 | 5997.798400 |
| 30 | 240 | 6032.961344 | 419.446978 | 83.517990 | 6.872429 | 4.244595 | 0.0 | 0.0 | 6038.102164 |
| 31 | 240 | 5967.255656 | 423.280101 | 84.963676 | 6.934072 | 6.061817 | 0.0 | 0.0 | 5972.468053 |
| 32 | 240 | 6016.178996 | 421.555511 | 82.052704 | 6.851818 | 0.469974 | 0.0 | 0.0 | 6021.288296 |
| 33 | 240 | 6008.828939 | 424.355868 | 79.920547 | 6.866830 | 3.393218 | 0.0 | 0.0 | 6013.974303 |
| 34 | 240 | 5642.967041 | 425.436180 | 77.932022 | 6.877667 | 0.914499 | 0.0 | 0.0 | 5648.078645 |
| 35 | 240 | 6118.562581 | 424.455327 | 76.182732 | 6.777201 | 1.650105 | 0.0 | 0.0 | 6123.653235 |
| 36 | 240 | 5546.363770 | 424.919202 | 76.894608 | 6.851558 | 6.236344 | 0.0 | 0.0 | 5551.512787 |
| 37 | 198 | 5616.284249 | 419.604358 | 72.871666 | 6.875449 | 6.202910 | 0.0 | 0.0 | 5621.339793 |

## 6. Upstream visualization block and CPU RNG semantics

The pinned trainer (`train_generator.py:197-211`) runs this block **after the epoch loop and before `save_checkpoint` (`:214`)** whenever
`epoch == 1 or epoch % 10 == 0`. It is a training-sample visualization. It is **not validation, reads no VAL/TEST data and selects nothing.**
Its two CPU draws `torch.zeros(240,128).normal_(0,1)` (noise, then noise_s) advance the torch CPU RNG, so omitting the block changes the
CPU RNG stream. M6D5e-r executes the **full block** (preferred option), in source order:

1. `noise`, then `noise_s`, each `[240,128]` standard-normal on the CPU default generator;
2. `netG(cat(noise_s, noise, noise))` on CUDA, in netG's training-time mode, under `no_grad`. The source never backpropagates `fake`, and
   netG has `InstanceNorm2d(track_running_stats=False)` and no dropout, so this is a memory-only difference with no parameter, buffer or RNG effect;
3. six grids via `torchvision.utils.save_image` into the qualification diagnostics directory, **separate from `checkpoints/`**.

| grid (`diagnostics/visualization/`) | images | bytes | SHA256 |
|---|---:|---:|---|
| `Epoch_001_img_spoof.png` (last TRAIN batch, 128 px) | 198 | 4,947,080 | `638885eb…599c` |
| `Epoch_001_img_live.png` | 198 | 5,100,795 | `9463d007…bb7f` |
| `Epoch_001_rec_spoof.png` (pre-step reconstructions, per V2 chunk) | 198 | 3,582,642 | `c01cf764…58b2` |
| `Epoch_001_rec_live.png` | 198 | 3,655,422 | `b3afe67e…d3c9` |
| `Epoch_001_fake_spoof.png` (noise fakes) | 240 | 16,164,244 | `6c94bc31…7d85` |
| `Epoch_001_fake_live.png` | 240 | 16,053,019 | `350638ff…16ac` |

**Proof (in-process, independent replay).** A separate `torch.Generator` was restored to the CPU state captured before the block and
drew two `zeros(240,128).normal_(0,1)` in source order. Results:
- it reproduced both noise tensors (`6a040807…` / `6cec91ab…`);
- it ended in **exactly** the global CPU RNG state observed after the block (`31b1de7f…` → `a6eb5e4e…`; the state advanced);
- the CUDA RNG, all parameters and the Adam state were unchanged by the block;
- the resume sidecar's torch-CPU RNG equals the post-block state.

Status: **UPSTREAM_VISUALIZATION_RNG_SEMANTICS_PRESERVED**. The production scientific loop runs the same `Trainer.visualize` at epochs
1, 10, …, 200 before each checkpoint event. Every checkpoint epoch is also a visualization epoch, so every sidecar holds the post-block state.

*Informational field, disclosed.* The train evidence contains `training_consumed_no_global_cpu_rng = false`. It is not a gate, and its
reference point was badly chosen: the CPU state was captured right after seeding, **before model construction**, and weight initialization
draws from the CPU RNG. So the field does not isolate the training loop. The torch DataLoader draws its base seed and permutation from the
seeded loader generator, and ε is drawn on CUDA. The field was not edited after the run.

## 7. Epoch-1 qualification checkpoint and resume sidecar

The pinned `misc/util.py::save_checkpoint` wrote the files below. Each reloads to a state_dict bitwise equal to the in-memory module. They
are indexed `periodic`, `selected_for_final = false`, "QUALIFICATION_ONLY … NOT_SCIENTIFIC_CHECKPOINT".

| file | bytes | SHA256 |
|---|---:|---|
| `checkpoints/netE_spoof_model_epoch_1_iter_0.pth` | 55,181,976 | `adab795c1108bd31e4d04bee3ff86689c854dd66330865d685df194470af986c` |
| `checkpoints/netE_live_model_epoch_1_iter_0.pth` | 46,792,321 | `5483b109b1f018dea04c75868754cb68ad98c23081bce88f4882f5267ef2fa1d` |
| `checkpoints/netG_model_epoch_1_iter_0.pth` | 79,577,546 | `c61932a9a33cff42d9523f28b5abf781a278158c5ef7b7192f3ecd7342e80a49` |
| `checkpoints/runner_state_epoch_1.pth` (engineering sidecar, `resume_state_index.json`) | 544,543,139 | `058df14fa7d8312cbfb79dba1e54efb1aef6543516b364b954fb61741f2fe5bb` |

The sidecar reloads under `weights_only=True`, and its models, Adam state (55 entries), RNG state (Python, NumPy, torch CPU, torch CUDA)
and loader generator equal the in-memory state. It has completed_epoch 1 and global_step 37, and is `scientific_checkpoint = false`,
`selection_candidate = false`. netIP is not stored. The `resolved_config.yaml` SHA256 is `b06df96f0798f0c8…bdea` and was re-derived
byte-for-byte on resume.

## 8. Resume equivalence (epoch 2, step 1)

The reference was ONE epoch-2 step from the in-memory post-checkpoint state in the train process, not logged. The fresh process loaded the
explicit `checkpoints/runner_state_epoch_1.pth`, SHA-matched against the index. It reloaded LightCNN from the verified asset, restored all
RNG and the loader generator, and reconciled metrics (37 records, last epoch 1). It then ran ONE epoch-2 step, appended as global_step 38.

| Compared | Reference = fresh process |
|---|---|
| next-batch pair-id SHA256 | `d20a11e5…ebe3` (= ID-only loader epoch-2 prediction) |
| global batch / chunks | 240 / 12 × 20 |
| ε_cls / ε_nir / ε_vis SHA256 | `f1fcfa57…` / `158c9558…` / `e6581158…` |
| losses rec, kl, mmd, ip, ort | 5477.795247395832, 415.1727091471354, 72.19483184814453, 6.780558188756305, 6.670567035675049 (cls = pair = 0) |
| total (post-warmup assembly) | 5978.613913615543 |
| 55 owned gradients | SHA256 identical; tensors 55/55 bitwise, max \|Δ\| = 0.0 |
| Adam state (step 38.0; 55 × exp_avg / exp_avg_sq) | identical |
| post-step parameters | all five models identical SHA256; 55/55 owned tensors bitwise, max \|Δ\| = 0.0 |
| pass-1 statistics, surrogate sums, RNG before / before step / after | identical |

**All 24 pre-registered fields are BITWISE identical** (no tolerance used). The restored parameters, Adam state, RNG and generator equal the
train process's end-of-epoch-1 hashes. `metrics.jsonl` is append-only. It holds 38 trajectory records, one per global optimizer step, plus
6 events, including `e06c_visualization_diagnostic` before `e06c_checkpoint_event`.

## 9. TRAIN-only benchmark access audit

Every benchmark-path event from every process was appended to `<build>/M6D5E_R_E06C_BENCHMARK_ACCESS_<mode>.tsv`. That covers the main
process and all forked DataLoader workers, and both allowed and denied events. The harness classified each event and gated on the result.
The raw logs were then re-checked independently with `awk`/`grep` on the GPU host.

| process | events | processes | TRAIN relation | TRAIN faces (distinct) | split manifest | VAL metadata | TEST metadata | non-TRAIN faces (≥ VAL/TEST images) | faces enumeration | runs/ | denied | log SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| loader | 2 | 1 | 2 | 0 (0) | **0** | **0** | **0** | **0** | 0 | 0 | 0 | `f1dde403…66c5` |
| train | 25,358 | 17 | 2 | 25,356 (12,668) | **0** | **0** | **0** | **0** | 0 | 0 | 0 | `9669967c…089f` |
| resume | 8,162 | 9 | 2 | 8,160 (6,617) | **0** | **0** | **0** | **0** | 0 | 0 | 0 | `ac696bfe…12bd` |

The TRAIN relation was opened twice per process, for the contract SHA and the relation SHA; pyarrow is then handed only that exact path.
The train process read all 12,668 distinct TRAIN sample files. The epoch reads 17,676 faces, and DataLoader prefetch of epoch-2 batches for
the reference step accounts for the rest. Every face path was classified against the frozen TRAIN relation alone, without reading any
split metadata. So **VAL image accesses = TEST image accesses = 0** follows from 0 non-TRAIN face accesses. No manifest other than
`pairs_train_v1.parquet` appears in any log. The logs (6.3 MB) remain in the GPU build root, and their SHA256 are bound in the evidence.

## 10. Tests, preflight, ledger, index

- **Tests.** `tests/test_m6d5e_e06c_runner.py` has 34 tests. New ones cover the access-audit firewall, the visualization cadence, the
  visualization RNG semantics (CPU engine) and separate initial/clean evidence classes. The focused M6D5a–e suites were run as follows:
  - GPU, `gpat-m6-e06c`, `CUDA_VISIBLE_DEVICES=""`, before the evidence existed: **138 run, OK, 8 skipped** (the evidence classes);
  - laptop, no Torch, after the evidence and reports: **138 run, OK, 32 skipped** (Torch/CUDA-only tests), with all 8 retained-evidence
    tests (initial and clean) passing.

  Every run was audited, and only `pairs_train_v1.parquet` was opened. No M6D5a–d test was modified.
- **Preflight.** `python3 -I -S -B tools/m6d5e_e06c_runner_preflight.py --rebuild-index` passes. It is static: no torch, pyarrow or cv2
  import, and no manifest, parquet, image or model. It checks that the initial evidence and runtime log are byte-frozen, fully validates the
  clean evidence, including the access audit, the visualization proof and the 24 bitwise resume fields (re-derived independently), and checks
  M6D5a–d immutability.
- **Ledger.** There are still **115 rows**, and the first 114 are byte-identical to `cd125db`. The single uncommitted M6D5e row was **corrected
  in place**; no second milestone row was added. It now records `decision = PASS_AFTER_CLEAN_REQUALIFICATION` and the explicit
  initial/clean access fields (`initial_development_{val,test}_metadata_access = true`, `clean_requalification_{val,test}_access = false`, …).
- **Index.** Rebuilt last, CRLF preserved.

Qualification weights (official files, sidecar, and the 362,997,231-byte probe `qualification_probes/reference_epoch2_step1.pt`) and the
diagnostic grids exist only under the GPU qualification root. None of them is in Git or under `runs/m6/E06c/`.

## 11. Final claims (clean rerun)

**REAL TRAIN DATA ACCESSED FOR QUALIFICATION. QUALIFICATION OPTIMIZER STEPS EXECUTED** (37 in epoch 1, then 1 in-memory reference step
and 1 fresh-process step; qualification counts only).
**NO VAL ACCESS DURING CLEAN REQUALIFICATION. NO TEST ACCESS DURING CLEAN REQUALIFICATION.
NO SCIENTIFIC FULL TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK. NO DOWNSTREAM EVALUATION. NO COMMIT. NO PUSH.**

For the initial development run the valid claims remain **NO VAL/TEST IMAGE ACCESS · NO VAL/TEST TRAINING USE · NO VAL/TEST SELECTION USE ·
NO TEST METRIC USE**, with VAL/TEST metadata read disclosed as **M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION**.

Awaiting owner review.
