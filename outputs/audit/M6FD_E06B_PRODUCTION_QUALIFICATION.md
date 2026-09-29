# M6FD — E06b DSDG-NATIVE production runner: PRODUCTION_PATH_QUALIFICATION_ONLY on real TRAIN data (M6F-D)

**This is not scientific training.**
- scientific_training_completed = false; scientific seeds completed 0; scientific checkpoints 0; scientific bank false.
- No VAL, TEST or SiW data was accessed.
- The optimizer steps below exist solely as execution qualification, and their weights were discarded.

| Field | Value |
|---|---|
| Authority (laptop and GPU) | `4b54280c53f87c391151a5436b18e5fc51db88fe`; the GPU repo was fast-forwarded only, its tracked state was clean before and after, and the staged code was removed by name |
| GPU / environment | NVIDIA GeForce RTX 3090, 595.84, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7; `gpat-m6-e06c` (Python 3.11.16, torch 2.12.1+cu130, CUDA 13.0); runtime identity equals `environments/e06c.lock.json` (`91416a20fef6eb4b…`) |
| Frozen config | SHA256 `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` (unchanged; snapshot byte-identical) |
| Source / LightCNN | FaceX-Zoo `16b793a7564a` (13 files); LightCNN `d075074662227054…` |
| Qualification seed | 60801 (never an experiment seed) |
| Production runner | `methods/dsdg/native_runner.py` (engine), `tools/run_e06b.py` (fresh-only scientific CLI) |
| Qualification harness | `methods/dsdg/native_runner_qualification.py`, `tools/m6fd_e06b_cli_preflight_capture.py`; all code SHA256s pinned to the bytes that ran |

## 1. Real TRAIN population and data firewall

The native relation is read from `split_v1` with a TRAIN + CASIA/MSU pyarrow filter on allowlisted columns. The
split file is SHA256-checked before it is parsed, and no VAL/TEST row is materialized.

The population is 3,720 spoof and 1,240 live frames across 60 subjects (print 2,080, replay 1,640). No SiW row is
present.

Before execution, every row was asserted independently: 3720 rows in the full relation,
plus both qualification subsets. The assertions are:
- TRAIN only;
- CASIA or MSU only;
- live subject equals spoof subject for every candidate partner;
- a real `dataset::` subject id;
- attack_macro exactly print (0) or replay (1).

The firewall is the M6D5e audit hook, subclassed so that its single allowed manifest is split_v1. It logs every
benchmark-path event of the main process and all workers. In every case it recorded 0 denied, 0 VAL, 0 TEST, 0
other-manifest, 0 non-TRAIN-face and 0 scientific-root events. No `dsdg_identity_pairs_v1.parquet` was created.

## 2. Loader case (real relation, production DataLoader, 8 workers)

- One real pass: 15 × 240 + 120, every spoof row exactly once, workers
  observed [0, 1, 2, 3, 4, 5, 6, 7].
- Batch b is served by worker b % 8, and each worker seed equals the base seed + worker_id.
- **Every real live-partner draw equals `native.simulate_live_draws`**, and every draw stays within the same subject.
- Real image decode, 16 pairs (4 per dataset × class), through the production `CanonicalFaceReader` and
  `canonical_chw`: every tensor is [3, 256, 256] float32 in [0, 1] and non-constant. Example: spoof `000e9a74690619f1…`
  with live `ab4050b8a6f7957f…`, casia_fasd `casia_fasd::5`, print; spoof tensor
  min 0.0471, max 0.5843.
- The production `E06bDataset` item keys are ['0', '1', 'index', 'live_id', 'spoof_id', 'type'].

## 3. Production batch path on real TRAIN images: independent qualification cases

Each case ran in a fresh process from a fresh model state. The two cases are **not** consecutive scientific steps. Each
used a deterministic stratified subset in relation order (60 or 30 rows per dataset × class; the two subsets are
disjoint) through the production `E06bDataset`, `build_loader` (8 workers configured) and
`E06bTrainer.global_step_run`, which is the unchanged M6D5d `run_global_batch_v2`.

| case | B | chunks | epoch branch | print/replay | subjects | loss_cls | loss_pair | total | owned grads finite | netCls grad nonzero | Adam steps | peak alloc / reserved | step | TRAIN face opens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b240 | 240 | 12 × 20 | 1 | 120/120 | 58 | 7.8920 | 0.008724 | 14519.0572 | 55/55 | 258/258 | 1 | 6.01 GB / 7.23 GB | 4.00 s | 480 (454 distinct) |
| tail120 | 120 | 6 × 20 | 2 | 60/60 | 54 | 7.9250 | 0.008564 | 14761.4261 | 55/55 | 258/258 | 1 | 5.83 GB / 7.04 GB | 2.14 s | 240 (234 distinct) |

Losses for the logical-240 case (raw = weighted / lambda):

| term | raw | weighted |
|---|---|---|
| loss_cls | 0.789204 | 7.89204 |
| loss_ip | 0.00753675 | 7.53675 |
| loss_kl | 18.5798 | 18.5798 |
| loss_mmd | 0.319307 | 15.9654 |
| loss_ort | 1.96361 | 1.96361 |
| loss_pair | 0.00174485 | 0.00872427 |
| loss_rec | 14518.5 | 14518.5 |

In both cases:
- all losses are finite;
- `loss_cls` > 0 and active (netCls gradient nonzero);
- `loss_pair` > 0;
- all 55 owned gradients are finite;
- exactly one optimizer step was taken;
- the owned parameters changed, while netCls and netIP stayed unchanged;
- inputs lie in [0, 1];
- there was no OOM.

The weights were discarded; `torch.save` was forbidden and nothing but the access log was written.

## 4. Scientific CLI (`tools/run_e06b.py`) preconditions on the GPU host (no Torch, no run)

- The preflight passes: the identities, source pin, LightCNN, DSDG lock and faces root all verify.
- `/home/student20261/workdir/GPAT_TransferBench_runtime/runs/m6/E06b/seed_42` was not created, and Torch was not imported.
- Resume status is `NOT_QUALIFIED_FRESH_ONLY`.
- These 8 cases are all refused with exit 2: dirty_worktree, include_siw_flag, lambda_pair_override, qualification_limit_flag, qualification_seed_60801, resume_state_flag, tf32_not_disabled, wrong_pythonhashseed.
- `dirty=False` was injected only because the runner files were staged uncommitted for qualification. The real
  clean-worktree gate is proven by the `dirty_worktree` refusal.

## 5. Fidelity (owner decision at the M6F-D review)

- **new_scientific_deviation_found = false.**
- **final_method_fidelity = `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`.** This is the final M6 method-level
  fidelity for E06b, and it is **not** `CONTROLLED_ADAPTATION`.
- Runtime fidelity (M6F-C) is `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`.
- The frozen YAML's pending field is historical and is not rewritten. The final current fidelity is recorded here
  and, later, in `method_status.csv`.

The owner's rationale:
- the real CASIA/MSU TRAIN production data path passed, and native same-subject pairing passed on real data;
- K = 2 and lambda_pair = 5 are unchanged;
- all official losses are unchanged and active;
- the effective batch remains 240, and physical microbatch 20 is only runtime compatibility;
- the logical 240 and tail 120 paths passed on real images;
- worker = 8 native pairing determinism passed;
- no scientific configuration, loss, architecture or optimization semantics changed, and no new scientific deviation
  was introduced.

Implementation/lifecycle disclosures. None of these downgrades fidelity:
- No complete scientific run lifecycle was executed.
- The inherited E06c RunContext and checkpoint writers were not exercised at runtime for E06b, because qualification
  may not write a checkpoint.
- Resume is unqualified, so scientific runs are **FRESH ONLY**; a failed seed reruns the same seed (spec §27).
- Input comes from the frozen canonical `faces_256`, the dataset adapter shared by every method.
- `split_v1` is opened once and read with a TRAIN filter.
- Refusal messages carry an `E06c runner:` prefix, because the E06c `require` helper is reused. This is cosmetic, and
  the bytes are pinned to what ran.

**E06b status:**
- `CONFIG_FROZEN`
- `STATIC_ADAPTER_IMPLEMENTED`
- `GPU_GRAPH_QUALIFIED`
- `EXECUTION_MAPPING_QUALIFIED`
- `WORKER_DETERMINISM_QUALIFIED`
- `PRODUCTION_DATA_PATH_QUALIFIED`
- `PRODUCTION_BATCH_PATH_QUALIFIED`
- `SCIENTIFIC_CLI_PREFLIGHT_QUALIFIED`
- `SCIENTIFIC_RUN_LIFECYCLE_NOT_EXECUTED`
- `CHECKPOINT_WRITER_NOT_E06B_RUNTIME_EXERCISED`
- `RESUME_UNQUALIFIED_FRESH_ONLY`
- `SCIENTIFIC_TRAINING_NOT_EXECUTED`

scientific_training_completed = false; scientific_seeds_completed = 0; scientific_checkpoints_created = 0;
scientific_bank_created = false.

**M6_CLOSED = false. M7 HAS NOT STARTED.**
