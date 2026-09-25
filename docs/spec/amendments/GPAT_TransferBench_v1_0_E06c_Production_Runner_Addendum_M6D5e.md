# GPAT-TransferBench v1.0: E06c production runner contract (M6D5e)

Status: **ADDITIVE · PRODUCTION_RUNNER_QUALIFICATION · REAL_TRAIN_QUALIFICATION_ONLY**.
E06c is still **DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation)**. Its fidelity class is
**CONTROLLED_ADAPTATION** (DEV-020). Its execution runs under **CONTROLLED_EXECUTION_ADAPTATION**
(`GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2`). For scientific results its status is still **IMPLEMENTED_NOT_EXECUTED**:
no seed has been trained. This addendum adds a runner. It changes no architecture, loss, coefficient, learning rate, optimizer,
batch policy, microbatch or epoch count.

The machine-readable companion is
[`e06c_m6d5e_production_runner_contract.yaml`](../../../configs/amendments/e06c_m6d5e_production_runner_contract.yaml)
(JSON subset of YAML). It binds these inputs by SHA256: the frozen E06c config, the A1 adaptation, the M6D5c V1 and M6D5d V2
overlays, `run_logging_v1`, the environment lock, the GPU execution storage config and the TRAIN pair manifest. None of them is
modified. Nothing from M6D5a–M6D5d is modified either.

**M6D5e-r correction.** The initial M6D5e qualification (seed 60505) is retained with the status
**M6D5E_INITIAL_RUN_PROCEDURAL_FIREWALL_DEVIATION**. While the canonical lookup was being developed, `manifests/split_v1.parquet` metadata
(TRAIN, VAL and TEST rows) was parsed once on the laptop. No VAL/TEST image bytes were accessed, VAL/TEST was not used for training or
selection, and no TEST metric was computed. That run also omitted the upstream visualization block and so skipped its two CPU RNG draws.
The M6D5e-r clean requalification (seed 60506, new root) re-ran the qualification under an audited TRAIN-only firewall and runs that
block (§4). Details: `outputs/audit/M6D5E_R_E06C_CLEAN_REQUALIFICATION.md`.

## 1. Two modes, one engine

| | SCIENTIFIC | QUALIFICATION |
|---|---|---|
| entry point | `tools/run_e06c.py` | `methods/dsdg/runner_qualification.py` |
| seed | 42, 1337 or 2026 (experiment seeds) | 60506 only (engineering; `experiment_seed = null`; 60505 = retained initial run, never reused) |
| epochs | 200 (hard-bound) | 1 real-TRAIN epoch + 1 resume probe step |
| run root | `<runtime_root>/runs/m6/E06c/seed_<seed>/` | `<runtime_root>/qualification/m6d5e/E06c/q60506-<run_id>/` |
| worktree | must be clean | uncommitted M6D5e files allowed and hashed into `run_manifest.json` |
| launched in M6D5e | **no** | yes |

The engine is `methods/dsdg/runner.py`. Both entry points use it, and nothing else is shared between them. The qualification harness
cannot write to `<runtime_root>/runs`. The audit-hook firewall denies that root, and the run-root function places every qualification
run under `qualification/`. If a qualification directory already exists, the run stops. It is never deleted or overwritten.

## 2. Data path (TRAIN only)

* **Relation.** The runner reads `manifests/pairs_train_v1.parquet`. It checks the SHA256
  (`a5e4fdae…c243`) before parsing and reads only the eight allowlisted columns (`pair_id, source_spoof_id, target_live_id, dataset, split,
  seed, source_sha256, target_sha256`). `source_subject` and `target_subject` are never materialized. Every row passes through the unchanged
  `DSDGAdapter.project_pair`. The runner fails closed on a SHA mismatch, a row count other than 8838, a non-TRAIN row, a dataset outside
  the frozen set, a missing column, a duplicate `pair_id`, or a sample bound to two datasets.
* **Canonical faces.** The lookup follows the repository's established convention: `faces_256_root/<dataset>/<sample_id>.png`. The M2 writer
  is `tools/m2b_run.py::_face_path`, and the M4 pair executor and M5 probe read the same path. The dataset comes from the allowlisted pair column
  (`pairs_v1.yaml: target_same_dataset_as_source: true`). The faces root comes from the storage block of `configs/execution/m5_gpu_3090.yaml`,
  which passes through `m2b.resolve_roots` containment, and it must equal `<runtime_root>/data/processed/faces_256`.
* **Reader.** The reader takes `reader(sample_id)` and nothing else. The sample_id must match `^[0-9a-f]{64}$` and belong to the TRAIN relation.
  Symlinks and path escape are refused, and the file is opened with `O_NOFOLLOW`. The reader decodes with `cv2.imdecode(IMREAD_UNCHANGED)`,
  requires uint8 `[256,256,3]`, and reverses the channels to RGB (M2 encoded `RGB[:, :, ::-1]`). It does no crop, resize, augmentation, jitter or
  normalization. The only normalization is the adapter's `/255`.
* **Dataset.** `IndexedPairDataset` is the unchanged `IDFreePairDataset` plus a row index used only for accounting. The official keys are
  kept: `'0'` = spoof, `'1'` = live, `'type'` = 0. The one-logit cross-entropy stays degenerate, as A1 discloses, and is not converted to two logits.
* **Firewall (qualification).** The canonical lookup above is fixed. It is never rediscovered from the split manifest, and
  `manifests/split_v1.parquet` is never opened or parsed. An audit hook denies every manifest except the TRAIN relation, every face outside the
  TRAIN relation, directory enumeration of faces_256, `<runtime_root>/runs` and repository data roots. It appends every benchmark-path event
  from the main process and every forked DataLoader worker to `<build>/M6D5E_R_E06C_BENCHMARK_ACCESS_<mode>.tsv`. Each process must end with
  0 VAL/TEST metadata accesses, 0 non-TRAIN face accesses (the upper bound on VAL/TEST image accesses) and 0 denials.

## 3. Loader and seeding

`DataLoader(batch_size=240, shuffle=True, num_workers=8, pin_memory=True, drop_last=False, generator=G,
worker_init_fn=seed_torch_worker)` is the pinned loader (`train_generator.py:94-97`) with the M6C2a seed contract added. `G` is
`torch.Generator().manual_seed(seed)`. Scientific runs seed through `learned.apply_framework_seed` (Python, NumPy, torch, CUDA,
deterministic cuDNN without benchmark), which requires `PYTHONHASHSEED=<seed>`. Qualification applies the same calls with 60506. Each epoch
consumes one int64 base-seed draw from `G` and one `randperm(8838)`, which gives `[240]×36 + [198]`. The evidence for each epoch is the SHA256 of
its newline-joined `pair_id` sequence. The loader uses no weighted sampling, balancing, padding, wraparound or sampler replacement.

## 4. Execution per global batch

Every loader batch goes to the unchanged `microbatch_execution_v2.run_global_batch_v2`, bound to the owner overlays through
`microbatch_execution.execution_guard` and `microbatch_execution_v2.execution_guard`. The historical `adapter.validate_batch()` still refuses
accumulation and is **not** an execution gate here. Each global batch runs these steps:
- draw ε_cls, ε_nir, ε_vis as [B,128] once, in the order cls → nir → vis;
- pass 1, the global MMD/ORT statistics;
- one `zero_grad(set_to_none=False)`;
- pass 2 over chunks: 12×20 for B=240, 9×20+18 for B=198;
- one `optimizer.step()`.

Adam(netE_nir + netE_vis + netG, lr 2e-4) runs with no scheduler, because the pinned trainer defines none. netCls and netIP are excluded, and
netIP is frozen and in eval mode. The warmup is epoch < 2 (epoch numbering starts at 1). Precision is FP32 only: no AMP/FP16/BF16, no TF32,
no activation checkpointing. `NVIDIA_TF32_OVERRIDE=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are required. A CUDA OOM is STOP_AND_REPORT
(no automatic microbatch change).

**Step gates, all fail-closed:**
- 55/55 optimizer-owned tensors have finite gradients;
- netIP receives no gradient;
- the netCls gradient is exactly 0;
- every global loss is finite, and `loss_cls = loss_pair = 0`;
- the labels are all 0;
- the inputs are FP32 [B,3,256,256] in [0,1];
- exactly one optimizer application per global batch.

**Epoch gate:** the batch sizes equal `[240]×36 + [198]`, and every TRAIN `pair_id` appears exactly once.

**Upstream visualization block (executed; M6D5e-r).** The pinned trainer runs a periodic training-sample visualization block
(`train_generator.py:197-211`) at `epoch == 1 or epoch % 10 == 0`. It runs after the epoch loop and before `save_checkpoint` (`:214`). It
is not validation and selects nothing. Its argparse cadence name is historical and does not refer to a data split. The block draws
`noise = torch.zeros(240,128).normal_(0,1)` and then `noise_s = torch.zeros(240,128).normal_(0,1)` on the CPU default generator. Those
draws advance the torch CPU RNG. The runner therefore executes the block at the source cadence and in source order (`Trainer.visualize`,
before `checkpoint_event`), so the resume sidecar holds the post-block CPU RNG state. The block writes six PNG grids through
`torchvision.utils.save_image` to `<run_dir>/diagnostics/visualization/Epoch_<EEE>_*.png`, separate from `checkpoints/`:
- the last TRAIN global batch at 128 px;
- that batch's pre-step reconstructions, collected per V2 chunk;
- the netG fakes of `cat(noise_s, noise, noise)`.

netG runs in its training-time mode under `no_grad`. The source never backpropagates `fake`. netG has only
`InstanceNorm2d(track_running_stats=False)` and no dropout, so the forward has no parameter, buffer, optimizer or CUDA-RNG effect.
`no_grad` is a memory-only difference. The initial M6D5e run omitted this block; M6D5e-r corrects that.

## 5. Logging (`run_logging_v1`, existing RunContext reused)

`E06cRunContext` subclasses `LearnedRunContext`. Every writer is inherited unchanged: atomic manifest/YAML, append-only fsync'd
`metrics.jsonl`, the checkpoint index, the resume reconciliation record and the lock. In scientific mode it is the unchanged LearnedRunContext.
Qualification mode binds the same writers to the qualification root with `experiment_seed = null`.

* **run_id.** The repository `RunContext` already defines the frozen rule `sha256(method|seed|config_sha256|git_commit)[:16]`, where
  `config_sha256` is the frozen E06c method-config SHA256. That rule is **preserved**. The task asked for the resolved-config SHA in run identity
  "unless an already-existing repository run-context implementation specifies another frozen rule". It does, so no second run-id convention is
  introduced. The exact SHA256 of `resolved_config.yaml` is recorded in `run_manifest.json`, `resume_state_index.json` and every resume state,
  and on resume it must match byte-for-byte.
* **resolved_config.yaml.** This is the frozen config verbatim, plus a `_resolved` block. That block lists the A1 identity, the M6D5c/M6D5d/M6D5e
  overlay paths and hashes, the source pin, the environment lock, LightCNN, and the visible execution adaptations (OOM_RETAINED, 240/20/198,
  `drop_last=false`). It does not claim that the frozen YAML contained the V2 resolution.
* **metrics.jsonl.** There is one trajectory record per **global optimizer step** (37 per epoch). Each record carries the seven losses +
  `total_loss`, the learning rate, the wall clock, the per-step peak CUDA memory, the actual global batch, the microbatch sizes, the batch pair-id
  SHA256 and the epsilon SHA256. `train_metrics`, `val_losses` and `val_metrics` are explicit `null` with reasons. Event records
  (`e06c_run_start`, `e06c_epoch_complete`, `e06c_checkpoint_event`, `resume_reconciliation`, `e06c_resume_reconciliation`) are separate. TEST
  keys are refused by `LearnedRunContext.log_epoch`.

## 6. Checkpoints

The rule is the pinned `if epoch % save_epoch == 0 or epoch == 1` with `save_epoch = 10`, so events fall at epochs 1, 10, 20, …, 200 (21 events). The
**epoch-1 exception is kept.** The writer is the pinned `misc/util.py::save_checkpoint`, which writes `torch.save({'epoch', 'model'})` to
`netE_spoof_/netE_live_/netG_model_epoch_<E>_iter_0.pth`. In `checkpoint_index.json`, epochs 1–190 are typed `periodic`. At epoch 200, `netG` is
`selected` (OFFICIAL_GENERATOR_EPOCH_200) and `netE_*` are `terminal`. The index records each file's actual size and SHA256. No selection uses
any data split.

## 7. Resume

At every checkpoint event the runner also writes the engineering sidecar `checkpoints/runner_state_epoch_<E>.pth`. It holds:
- the completed epoch and global step;
- the netE_nir, netE_vis, netG and netCls state;
- the Adam state;
- the Python, NumPy, torch-CPU and torch-CUDA RNG states;
- the DataLoader generator state;
- the resolved-config SHA256 and every identity;
- the execution mode.

netIP is not stored: it is re-read from the SHA-verified LightCNN asset. The sidecar is indexed in the separate
`resume_state_index.json`, because the frozen checkpoint-type enum cannot represent engineering state without ambiguity. The sidecar is
**not a scientific checkpoint and not a selection candidate.**

Resume is **explicit**: the caller must name the exact indexed file (no latest-file discovery). The runner loads it with
`torch.load(weights_only=True)` after checking size and SHA256. It then validates the method, mode, seed, run_id, resolved-config SHA256,
identities, epoch and step. `metrics.jsonl` must hold exactly steps 1..N, with the last epoch equal to the completed epoch. On any
disagreement the runner stops; it never truncates or overwrites. The inherited RunContext appends `resume_reconciliation`, and E06c appends
`e06c_resume_reconciliation`.

## 8. Production CLI safety

`tools/run_e06c.py` refuses the following before importing Torch:
- any override flag: epochs, lr, batch, microbatch, drop_last, AMP/FP16/BF16, TF32, checkpoint selection, save cadence, λ, qualification or
  "latest" resume;
- a seed outside {42, 1337, 2026}, including the qualification seeds 60505 and 60506;
- a dirty worktree;
- a wrong `PYTHONHASHSEED`, `NVIDIA_TF32_OVERRIDE` or `CUBLAS_WORKSPACE_CONFIG`;
- any identity drift: contract, overlays, source, lock, pair manifest, LightCNN or execution config;
- a non-empty seed directory without a consistent explicit `--resume-state`.

It was not launched in M6D5e.

## 9. Qualification discipline

The qualification run reads real TRAIN data and applies real optimizer steps: 37 in epoch 1, plus one in-memory reference step and one
fresh-process step. Its outputs are **QUALIFICATION_ONLY · NOT_SCIENTIFIC_CHECKPOINT · NOT_ELIGIBLE_FOR_SYNTHETIC_BANK ·
NOT_ELIGIBLE_FOR_DOWNSTREAM_EVALUATION · NOT_ELIGIBLE_FOR_REPORTING**. Its losses are engineering evidence, not benchmark results, and nothing
was retuned on them. The resume gates were pre-registered as bitwise, based on M6D5d's cross-process bitwise identity. A non-bitwise
outcome would have been STOP_AND_REPORT. The runner has no VAL or TEST code path. The M6D5e-r clean requalification proved 0 VAL/TEST
accesses per process through the access audit. The initial run's development-time metadata read is disclosed above and in the audit
reports.

Wording stays as follows: E06c is never described as native, faithful or an official reproduction. CONTROLLED_ADAPTATION is preserved.
