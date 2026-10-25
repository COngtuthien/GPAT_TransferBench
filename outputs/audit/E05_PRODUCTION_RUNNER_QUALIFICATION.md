# E05 PCGAN (controlled architecture resolution): real-TRAIN production runner qualified

**Status:** `E05_PRODUCTION_RUNNER_QUALIFIED`, with 44/44 gates passing (`tools/e05_production_qualification_evidence.py`). The machine-readable record is `E05_PRODUCTION_RUNNER_QUALIFICATION.json`. It embeds the read-only host collection, so the gates can be rebuilt. The four roadmap P0 blockers (A–D) are closed.

| | |
|---|---|
| method | E05, reporting label **PCGAN (controlled architecture resolution)**, track A_FAIR_COMMON_IDENTITY_FREE, fidelity **CONTROLLED_ADAPTATION** (A2-07 architecture resolution, A5 blur operator, M6D4c runner resolution). Not "native PCGAN", not "faithful PCGAN", not an official reproduction: no official PCGAN code release exists. |
| executable architecture basis | `taesungp/swapping-autoencoder-pytorch@6baa180f1184ee79a6b967f9d80ee0e02a979ac7`, tree `25a434f1282a0177d6afdcc68ebfcf40bd805621` (disclosed) |
| frozen config | `configs/methods/e05_pcgan.yaml` `478756e1…` = snapshot; spec `f7d23716…`; A5 overlay `af28a48b…`; M6D4c overlay `8eee350c…` |
| environment | `gpat-m6-e05`, lock `849a1004…`: Python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0, cuDNN 92000, executable `d3255ce6…`; the existing fused_bias_act `5afdb236…` / upfirdn2d `40ba9db7…` binaries (no rebuild, no fallback) |
| M6D4e | `E05_CLEAN_REQUALIFICATION_PASS` (unchanged, re-verified as an input gate) |
| candidate commit qualified | `eede86819b49ec8b809ac5ce80612a86c6c28c78`; the production files in the authority commit are byte-identical (gate A) |
| host | `nd-System-Product-Name`, single RTX 3090 `GPU-b722cd9d-…`, reached via the authorized fallback route `100.121.84.44` because `10.31.131.201` returned "No route to host" |

## Disclosure: host driver version

The lock and M6D4e recorded NVIDIA driver **595.84**. The host now runs **580.178.04**: the 580 open modules were installed on 2026-10-10 and the host has run on them since its 2026-10-20 15:56 (+07) boot, which includes every E06c seed1337/2026 and E06b run. The GPU model and UUID are identical, and the driver still exposes CUDA 13.0.

The environment gate compares every lock field exactly (Python binary hash, all 54 packages, torch, torchvision, CUDA, cuDNN, compiler, nvcc, GPU capability, precision, launch environment). There are two exceptions. PYTHONHASHSEED must equal the run seed. The driver *version* string may differ, but only when the GPU model and UUID match, and both versions are recorded (`production_io.environment_differences`; tested). The driver is host-level and outside the conda environment. Nothing was installed or upgraded. This qualification ran on the current driver, which is the one seed42 will use. **This driver-version tolerance is an owner-review item before the seed42 launch.**

## Production design (closing blockers A–C)

**Changed files: only additions.** No existing file changed since `3b0d308`: the frozen config, A2/A5/M6D4c contract, source pin, architecture, losses, optimizer, budget and final-state rule are untouched (gate I).

| file | class |
|---|---|
| `methods/pcgan/production_io.py` | production integration (framework-free gates, pair reader, cadence, records) |
| `methods/pcgan/production_runner.py` | production integration (E05RunContext, dataset/loader, E05Trainer, loop, checkpoint writer, scientific entry) |
| `tools/run_e05.py` | scientific CLI |
| `methods/pcgan/production_qualification.py` | qualification harness |
| `tools/e05_production_qualification_evidence.py` | evidence |
| `tests/test_e05_production_runner.py` | test |

**Reused without modification:**
- `TrainingRunner`: the qualified D → G cycle, two Adam optimizers at lr 1e-6 with betas (0.9, 0.999). A failed partial iteration marks the runner failed and can never be retried.
- `training_qualification.existing_runtime`: the qualified generated source, the existing CUDA binaries and the pinned builders/crop.
- `runtime_qualification.environment`.
- The E06c canonical face reader and the M2 faces-root containment.

**Data path:**
- **Manifest gates:** `manifests/pairs_train_v1.parquet` `a5e4fdae…`. Its SHA256 is checked before parsing. Only the columns `pair_id, source_spoof_id, target_live_id, dataset, split` are read; the subject columns are never materialized. All 8838 rows must be TRAIN, with distinct source/target and each sample id bound to one dataset. Any other split is refused.
- **Pair roles:** spec 8.5 plus the M6D4c pair contract give `x_src = face(source_spoof_id)` (pattern/artifact) and `x_tgt = face(target_live_id)` (content), with m = G(z_pat_src, z_con_tgt).
- **Face reads:** faces come only from `<runtime_root>/data/processed/faces_256/<dataset>/<id>.png`. Symlinks and traversal are refused.

**Per-iteration pair order:**
- `DataLoader(batch_size=1, shuffle=True, num_workers=0, drop_last=False)` runs over the stored manifest order (pair_id ascending). Its generator is `torch.Generator().manual_seed(seed)` from `methods.common.learned.torch_loader_options`.
- Iteration *t* consumes position *t − 1* of the first pass: one int64 base-seed draw, then `randperm(8838)`. That gives 4000 distinct pairs with none repeated.
- Shuffling follows the pinned predecessor's train phase (`data/__init__.py:93`). An independent predictor equals the loader at all 4000 positions for seeds 42/1337/2026 (test 17) and for the qualification seed (gate D).

**Input mapping and randomness:**
- **Input values:** canonical uint8 RGB is mapped by (x/255 − 0.5)/0.5 into [-1, 1], following the pinned `data/base_dataset.py:137-141` (ToTensor + Normalize(0.5, 0.5)).
- **No augmentation:** no resize, crop or flip; the canonical faces are already 256 px.
- **Workers:** 0. The workers policy is IMPLEMENTATION_ADAPTABLE_NO_SCIENTIFIC_IMPACT, and the dataset itself has no randomness.
- **Seeding:** Python, NumPy, torch CPU and CUDA are seeded by `apply_framework_seed(seed)` with `PYTHONHASHSEED = seed`. Models are initialized after seeding. Crops and StyleGAN noise use the global RNG; the loader uses its own generator.

**Loop semantics:**
- Exactly 4000 benchmark iterations; a qualification run is limited to 3.
- Each iteration is one complete `TrainingRunner.cycle`. The trainer gates on:
  - the counter advancing once;
  - exactly one D and one G application, counted by the runner and by independent optimizer post-step hooks;
  - 80/80 D and 59/59 G gradients finite and present, with the inactive group's gradients None;
  - all 14 loss values finite and the five G terms positive;
  - inputs of [1,3,256,256] float32 in [-1, 1].
- Every iteration writes one `metrics.jsonl` record with:
  - global_step and benchmark_iteration, and lr for D and G;
  - all D and G terms and both totals;
  - wall-clock and step seconds, and GPU memory;
  - the seed and the method;
  - pair_id, source/target ids, the pair hash, the sampler index and the PNG byte hashes;
  - the application counts and gradient counts;
  - explicit null reasons for train/VAL metrics.

**Checkpoints (BASELINE_FINAL_STATE_V1):**
- **Cadence:** the pinned predecessor's own rule (`util/iter_counter.py` save_freq = 50000 images, plus the final save in `train.py:54`). At batch 1 within 4000 iterations that means iteration 1 (`periodic`, not selected) and iteration 4000.
- **Terminal checkpoint:** `checkpoints/iteration_4000.pth`, type `selected`, `selected_for_final = true`, reason `BASELINE_FINAL_STATE_V1`.
- **Content:** state_dicts of Encoder, Generator, ImageD and PatchD plus JSON provenance. There is no optimizer state, because resume is not qualified.
- **Write:** atomic, followed by a `weights_only=True` round trip that must equal every live tensor and an index record carrying path, epoch, global_step, size, sha256, type, selected_for_final and reason.
- **Guarantees:** no optimizer application happens while a checkpoint is written. A terminal record is refused unless the runner has really completed iteration 4000. No VAL/TEST/best selection is possible.

**Lifecycle and resume:**
- The run uses LearnedRunContext / run_logging_v1, writing resolved_config, run_manifest, metrics, checkpoint_index and run_summary. The summary carries the final path and SHA.
- **Fresh only:** an existing seed root is refused, and there is no resume flag. A failed scientific seed needs owner review and a fresh same-seed rerun after the technical correction.

**CLI `tools/run_e05.py`:**
- **Accepted arguments:** only `--seed`, `--execution-config` and `--preflight-only`.
- **Refused before any import:** about 80 override flags, covering iterations, batch, lr, betas, resolution, loss weights, alpha/beta, precision/TF32/AMP, checkpoint selection, best, datasets/splits/VAL/TEST, pair manifest, output/run root, qualification limits, resume and workers.
- **Preflight:** imports no torch, executes no model, reads no image and writes nothing. It verifies:
  - clean git on branch m6-baselines;
  - the seed and the exact launch environment;
  - the frozen config, snapshot and spec;
  - A2/A5/M6D4c/M6D4e;
  - the source pin and tree;
  - the lock and compatibility manifest;
  - the execution config;
  - the pair-manifest SHA;
  - that the faces root exists;
  - that the build root exists;
  - that the target root is absent.

## GPU qualification (closing blocker D)

**Preflight on the host (candidate `eede868`):**
- Seeds 42, 1337 and 2026 each exit 0 with `torch_imported = false`, `run_dir_exists = false`, 0 images read and no model executed.
- Refusal probes all exit 2: seed 60901, `--iterations 10`, `--resume`, and `PYTHONHASHSEED=1`.
- `runs/m6/E05` does not exist afterwards.

**Real-TRAIN qualification:**
- **Seed and roots:** seed **60901**, a qualification seed that is never an experiment seed. The roots are `<rt>/qualification/e05_production/E05/train-q60901-bb76724b550266c4/` and `terminal_probe-q60901-bb76724b550266c4/`, never under `runs/`.
- **Window:** 2026-10-25T04:30:47Z–04:30:55Z.
- **Firewall:** an audit hook was active before the first manifest byte was read.

**Data**
- The pair manifest gates hold: SHA, TRAIN only, allowlist, subject columns not read.
- The independent prediction gives sampler indices [6151, 4118, 3855], which are pairs PTR006152, PTR004119 and PTR003856 (siwmv2 TRAIN). The consumed records equal the frozen manifest rows exactly.
- The 6 faces opened, in order, are exactly the expected canonical paths. Their PNG byte SHA256s are recorded.
- Inputs are [1,3,256,256] torch.float32 in [-1, 1].

**Iterations**
- There were 3 iterations, global_step 1, 2 and 3.
- **Applications:** D = G = t after each iteration (hooks agree), total 6 at the end. Adam state step = t with 80 D and 59 G states.
- **Isolation at iteration 1:** the D step changed only ImageD and PatchD; the G step changed only Encoder and Generator. The groups do not overlap and no parameter is aliased (enforced by TrainingRunner).
- **Losses:**

  | iteration | L_rec | L_recblur | L_advrec | L_advmix | L_pat | L_G_total | L_D_total |
  |---|---|---|---|---|---|---|---|
  | 1 | 205.109 | 86.918 | 0.6846 | 0.6847 | 0.5390 | 293.936 | 2.8332 |
  | 2 | 197.438 | 128.425 | 0.6841 | 0.6826 | 0.4679 | 327.698 | 2.7370 |
  | 3 | 264.995 | 115.935 | 0.6834 | 0.6864 | 0.4100 | 382.709 | 2.6861 |

  All values are finite, every G term is active, and there was no NaN, Inf or OOM.
- **Gradients:** 80/80 D and 59/59 G gradient tensors are finite and nonzero; the inactive group's gradients are None.
- **Performance:** about 1.6 s for the first iteration (warm-up) and then about 0.31 s per iteration. Peak memory was 3.08 GB. Speed is not a gate.

**Checkpoint writer**
- **Periodic checkpoint:** `checkpoints/iteration_0001.pth`, 232,113,696 bytes, sha256 `8b5c3bc8…`, `periodic`, not selected. All 167 tensors round-trip equal, and the index entry matches the file.
- **Terminal-semantics probe:** a separate labelled root. It wrote `checkpoints/iteration_4000.pth` (`191edafd…`) as `selected`, `selected_for_final = true`, reason `BASELINE_FINAL_STATE_V1`, from the 3-iteration state (`actual_benchmark_iteration = 3`, `terminal_semantics_probe = true`, label `QUALIFICATION_TERMINAL_SEMANTICS_PROBE_NOT_A_SCIENTIFIC_COMPLETION`). No optimizer application happened during the write. It does not imitate a scientific completion record: the summary is `interrupted` with a null final checkpoint.
- **Writer refusals (5):**
  - a qualification run asking for 4000 iterations;
  - a terminal checkpoint without 4000 completed iterations;
  - the probe flag used in the train case;
  - re-writing the periodic checkpoint;
  - a periodic checkpoint in the probe case.

**Lifecycle**
- **Train root:** run_start event, then 3 step records, then a checkpoint event. The manifest and summary are `interrupted` with the planned reason `QUALIFICATION_STOP_AFTER_3_ITERATIONS`. experiment_seed is null and qualification_seed is 60901. stderr is empty.
- **Probe root:** a checkpoint event only.

**Firewall**
- No denials. The access log has 7 events from 1 process: PAIR_MANIFEST 1 and AUTHORIZED_TRAIN_FACE 6. Every other category is 0: split/VAL/TEST metadata, unauthorized faces, faces enumeration, `runs/` and other benchmark data.
- **Stat-only atime check** (pre 04:30:43Z, post 04:31:20Z, 20 627 files each):
  - the only changed faces are the 6 authorized siwmv2 TRAIN faces;
  - CASIA/MSU/SiW TEST (3027) and VAL (3121) are all atime-sensitive and unchanged;
  - no manifest and no face mtime changed.
- **VAL = 0, TEST = 0.**

## Tests

- **`tests/test_e05_production_runner.py`:**
  - **Results:** 22/22 in `gpat-m6-e05` on the GPU host; 22/22 in `gpat-m7-cpu`; 17 passed with 5 torch cases skipped in the repo `.venv`.
  - **Covers:** non-TRAIN rejection, the manifest SHA gate, the image contract, seeded order, the 4000 budget, one D + one G per iteration, terminal iteration 4000 selected, no VAL/TEST/best selection, fresh-only refusal, CLI refusals, preflight writing nothing, root separation, the real LearnedRunContext lifecycle, checkpoint index/summary semantics, the frozen inputs, the driver-version gate and the loop integration.
- **`tests/test_e05_production_qualification_evidence.py`:** rebuilds this evidence deterministically and checks that the authority's code equals the qualified candidate.
- **M6D4a/b/d, M6A7 and M6C2B2 regressions:** all pass in `gpat-m6-e05` (GPU), `gpat-m7-cpu` and `.venv`.
- **Broad `m6*` (`.venv`):** 975 run, with the 4 known errors from `/media/cong/Data` not being mounted.

## Boundary

Nothing beyond qualification ran. No scientific E05 seed was launched, and no `runs/m6/E05` root exists. E06b/E06c run roots were not touched. No other method, bank, downstream evaluation or GPAT work was started. The next milestone independently reviews this authority, including the driver disclosure, and launches E05 seed42.
