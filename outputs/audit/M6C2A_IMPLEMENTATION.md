# M6C2a — learned runtime, E03 STDN and E06c DSDG-BIN-IDFREE

Starting commit: `425f0aebe28011cae11041d210166572269d0f6c`, branch `m6-baselines`.
Preflight working tree was clean. Date: 2026-09-22.

**IMPLEMENTED: static preparation, adapter seams and runtime support.**
**SYNTHETIC/STATIC VERIFIED. NOT BENCHMARK-EXECUTED.**
E03 and E06c are `IMPLEMENTED_NOT_EXECUTED`. This is not a claim that a
standalone end-to-end training launcher or a GPU execution environment has been validated.
Actual train execution was optional in this milestone and is not exposed by the preflight tool.
E01/E02 retain their accepted M6C1 state. E04/E05/E07c remain `CONFIG_FROZEN`.

## Common support

New modules: `methods/common/learned.py`, `learned_runlog.py`, `upstream.py`.
The M6C1 `config.py`, `seeding.py`, `runlog.py`, interfaces and tests are unchanged.

- Config loading reuses M6C1 snapshot/SHA authority. A caller-supplied altered config is refused.
- Source verification checks repository origin, exact commit/tree, every cited source file and the
  training/import closure against pinned Git blob bytes; missing/changed source fails closed.
  Known historical source-cache weight removals are neither restored nor read.
- External assets must remain outside the repository; frozen size/SHA are checked. No fallback.
- Seed plans consume `[42,1337,2026]`; explicit hooks seed Python, NumPy, TF1 or PyTorch.
  `PYTHONHASHSEED` is required at interpreter launch. PyTorch DataLoader generator and worker
  hooks are provided; CUDA seeding requires an explicit runtime call. cuDNN benchmark is disabled
  and deterministic mode enabled per the source-contract audit. Strict global deterministic
  algorithms are not forced. Kernel/device bitwise determinism is not asserted.
- The TF map hook uses `num_parallel_calls=1`, preserving random frame/flip callbacks while
  serializing their shared Python/NumPy RNG consumption. It was dispatch-tested, not TF-executed.
- Learned logging subclasses the existing atomic, append-only, locked/resume-safe runtime.
  Framework/hardware metadata and null reasons are explicit; it does not label learned runs as
  non-learned. Nested TEST metrics and checkpoint-rule overrides are refused. Synthetic log tests
  use a temporary directory and an empty checkpoint index.
- Checkpoint helpers support official/periodic/terminal/selected metadata. Plans contain no claimed
  checkpoint bytes/hash. Future actual writers must supply measured size/hash. Final selection is
  within seed and only at the method's official endpoint; neither VAL nor TEST can select.

## E03 — STDN

Files: `methods/stdn/{__init__,source,landmarks,adapter}.py`.
Source: `yaojieliu/ECCV20-STDN@c79f1f8c615d2b8471b3df29da881bb18dd54c90` — verified.
Fidelity: `FAITHFUL_OFFICIAL`.

The lazy binding exposes the unchanged official `train._step`, `model.model.Gen`,
`Disc_s`, `get_train_op`, and `model.warp` functions. It does not vendor or reconstruct STDN.
No source module was imported for model execution during this milestone.
`model/config.py` is the effective configuration; root `config.py` remains vestigial.
The mapping contains field names, with values taken from M6B, checked against the source class:

| GPAT field | Upstream setting | Frozen value |
|---|---|---|
| training.input_resolution / map_size | IMAGE_SIZE / MAP_SIZE | 256 / 32 |
| training.batch_size / g_d_ratio | BATCH_SIZE / G_D_RATIO | 2 / 2 |
| training.max_epoch / steps_per_epoch / val_steps | MAX_EPOCH / STEPS_PER_EPOCH / STEPS_PER_EPOCH_VAL | 50 / 2000 / 500 |
| optimizer.learning_rate / lr_decay.rate | LEARNING_RATE / LEARNING_RATE_DECAY_FACTOR | 6e-5 / 0.9 |
| optimizer.lr_decay.every_steps divided by steps_per_epoch | NUM_EPOCHS_PER_DECAY | 20000 / 2000 = 10 |
| optimizer.weight_ema_decay | MOVING_AVERAGE_DECAY | 0.9999 |

Adam defaults, loss EMA, decomposition, multi-scale losses, hard-sample draws and G:D update
path remain in the hash-verified official source; plans retain their frozen optimizer/loss contracts.
Both optimizers share the upstream global step; it is not equated to dataset iterations.

Landmarks accept only the frozen M2 `landmarks_px256` representation: finite `[68,2]` float32,
xy pixels divided by image width for both coordinates. FaceXFormer/iBUG68 ordering is retained.
The exact upstream one-based `lm_reverse_list` is extracted statically, converted to zero-based,
and checked as a permutation/involution. Flip uses `x -> 1-x` then that permutation, with no
width-minus-one substitution. Identity, exact 68-point semantic flip and double flip are tested.
Horizontal flip remains enabled. No real face, geometry cache or image was opened.

Checkpoint: **OFFICIAL_LATEST_FINAL_CKPT_50**, authoritative **ckpt-50**, every epoch retained.
VAL diagnostics never select; TEST has no adapter path. BASELINE_FINAL_STATE_V1 does not override it.

## E06c — DSDG-BIN-IDFREE

Files: `methods/dsdg/{__init__,source,adapter}.py`.
Source basis: `JDAI-CV/FaceX-Zoo@16b793a7564a4b9308cf94e62bdb2ffacb3a725a`,
`addition_module/DSDG` — verified, including generator, LightCNN, utilities, dataset import closure,
training shell/entrypoint and generation checkpoint reference.
Fidelity: `CONTROLLED_ADAPTATION`; provenance: `AMENDMENT_A1_IDFREE_ADAPTATION`.

The adapter maps frozen values to official argparse settings, retains upstream models/losses,
and provides a lazy component binding. The argument vector is explicitly NOT a launch command:
unadapted upstream `main()` must not be called because it uses identity pairing and unseeded workers.
A future runner wires the adapter/seed/logging hooks to that pinned training path.

`IDFreePairDataset` projects already-verified common TRAIN pair records with the A1 allowlist.
No duplicate manifest is created. Subject/attack columns are ignored and never reach a reader,
model or class target. The injected canonical reader receives sample IDs only; tests use synthetic
in-memory arrays. Canonical uint8 RGB256 is converted to CHW float32 /255, matching ToTensor;
no additional crop/resize is invented. Upstream keys remain `0`=spoof, `1`=live, `type`=class index.
Benchmark labels live=0/spoof=1 are distinguished from the **single-spoof CE index 0**.
`attack_type=1`, `lambda_pair=0`; identity supervision and config overrides fail closed.
The one-logit classification loss remains identically zero with zero gradient, as A1 discloses.
All other losses remain retained, including LightCNN own-input identity preservation.
The optimizer remains Adam lr 2e-4 over encoders + generator; the classifier stays outside it.

LightCNN is reverified at its frozen external runtime path:
`/media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar`.
Size **123844849 bytes**; SHA256
`d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964`.
No download, copy, torch.load, pickle execution or deserialization occurred.
The digest identifies the exact M6A4 safely inspected artifact: **60/60** parameters matched,
0 missing, 0 shape mismatches; extra `module.fc2.weight [80013,256]` is dropped by the official
non-strict filter because `is_train=False` omits the classification head. This compatibility is
reused hash-bound historical evidence, not a new framework load test.

The common learned runtime retains **arithmetic** effective-batch accounting:
`physical * accumulation * replica_factor = 240`. That arithmetic helper does not establish
objective equivalence or authorize E06c microbatch execution. It and its generic tests are unchanged.

**Pre-commit scientific correction — E06c enforces a stronger method-specific invariant:**
`E06C_REQUIRES_PHYSICAL_BATCH_240_FOR_OBJECTIVE_EQUIVALENCE`.
The authoritative normal path is physical batch **240**, accumulation **1**, replica factor **1**,
effective batch **240**. `DSDGAdapter.validate_batch()` guards `build_training_plan()` before
preparation/argument construction. Both `120 * 2 * 1` and `60 * 4 * 1` are rejected, even with an
OOM reason. Official DataParallel takes a global batch; its GPU count is not a batch multiplier.

Source evidence at pinned commit `16b793a7564a4b9308cf94e62bdb2ffacb3a725a`:
`third_party/source_cache/facexzoo/addition_module/DSDG/train_generator.py::main()`:

- Line 141: `loss_mmd = args.lambda_mmd * torch.abs(z_nir.mean(dim=0) - z_vis.mean(dim=0)).mean()`.
- Line 147: `loss_ort = args.lambda_ort * torch.abs((z_cls * z_nir).sum(dim=1).mean())`.
- Lines 175–178 include both terms in the total loss; lines 180–182 zero gradients, backpropagate
  that physical-batch loss, and step the optimizer.

Both retained losses apply absolute value **after** a batch mean. For equal microbatches with mean
statistics +1 and -1, the full-batch term is `abs((1-1)/2)=0`, while averaging the microbatch terms
is `(abs(1)+abs(-1))/2=1`. The synthetic latent test demonstrates this for MMD and angular
orthogonality; thus naive accumulation is not generally objective-equivalent. Linear sufficient
statistics could be combined in principle, but a correct loss/backward implementation preserving
both terms and upstream forward/RNG behavior has **not** been demonstrated here. No alternative
microbatch path is enabled on the strength of arithmetic accounting alone.

Future GPU preflight must attempt physical batch 240 first. If it OOMs: **STOP_AND_RESOLVE**.
Do not automatically reduce the batch or silently accumulate gradients. A semantics-preserving
implementation must be demonstrated and an execution-resolution decision made before changing
this guard. The frozen OOM allowance is necessary but not sufficient for a batch-dependent
objective; this is an implementation safety interpretation, with **no frozen YAML edits**.
The official non-drop-last tail policy is unchanged; this correction adds no padding, repetition
or drop-last override and does not treat a tail batch as an OOM fallback.

Checkpoint: **OFFICIAL_GENERATOR_EPOCH_200**, **netG_model_epoch_200_iter_0.pth**.
Official save at epoch 1 and every 10 remains the source path; no VAL-best override.
The periodic upstream `test` block writes TRAIN sample visualizations; it is not validation.
No such block was executed.

## Environment and validation

Observed development environment only, **not final execution pins**:
Python 3.12.3, NumPy 2.5.3, Pillow 12.3.0, opencv-python-headless 5.0.0.93.
Missing: TensorFlow, PyTorch, torchvision, matplotlib and SciPy. Nothing installed.
STDN needs legacy TF1 `tf.contrib`/graph APIs, unavailable in this Python environment. The upstream
README both excludes TF1.13 in its range and names TF1.13 as tested; no new pin is inferred.
The modern preparation package does not establish compatibility with the legacy upstream Python
runtime. A compatible isolated execution environment/runner must consume the serialized plan;
its integration must be tested before training. DSDG constructors/reparameterization use CUDA;
CPU construction was not attempted. No mocked dispatch test is reported as framework execution.

- M6C2a focused tests: **72/72 PASS**, 0 failures, 0 errors, 0 skipped.
- M6C1 focused regression: **83/83 PASS**, 0 skipped.
- M6B validator: **PASS, 0 failures**.
- `tools/m6c2a_preflight.py`: **PASS**, all six method/seed plans; frozen configs/logging,
  source trees, LightCNN, checkpoint, TEST firewall and E06c full-physical-batch policy verified.
- Current M0: **16/17 PASS**; only `test_manifests_are_stage_appropriate` fails on
  `manifests/artifact_probe_classes_v1.json`, unchanged and not repaired. A malformed draft ledger
  record was detected before commit: current record 87 (`M6C2A_LEARNED_BASELINE_IMPLEMENTATION`)
  was absent from committed HEAD and lacked `input_artifacts`. It was corrected in place before
  the milestone commit using the frozen configs, audit inputs, pinned source metadata and source
  trees consumed by M6C2a. No committed historical ledger record was rewritten; the committed
  86-record prefix remains byte-identical.

Preflight and focused M6C2a tests have separate Python file-open audits in the JSON artifact.
Each records **0 benchmark image opens, 0 manifest opens, 0 rejected data-open attempts**.
The hook blocks image/video, manifest, array/parquet/cache and production data paths. Git source
identity subprocesses are read-only; Python audit counts do not include Git's internal file opens.
The canonical index and M0 checks may hash tracked artifacts, including manifest bytes; they do
not load benchmark image pixels or execute a split. No data results inform scientific decisions.
The full repository suite was not run. Historical M6C1 data-access disclosures and all M6A1–M6C1
artifacts remain byte-unchanged, including its earlier full-regression UNVERIFIED state.

## Integrity and scope

Current source/test/tool/status hashes are in the JSON audit. The audit MD/JSON hashes are in the
new `M6C2A_PRECOMMIT_BATCH_SEMANTICS_CORRECTION` ledger record and the rebuilt canonical
artifact index. The prior `M6C2A_LEARNED_BASELINE_IMPLEMENTATION` record retains its historical hashes; self-referential hashes are not fabricated. The index excludes itself and the
append-only ledger by repository convention. Historical ledger bytes are preserved exactly.

NO TRAINING. NO BENCHMARK IMAGE EXECUTION. NO SYNTHETIC BANK GENERATION.
NO TEST DATA ACCESS (no TEST records or pixels loaded for method execution).
NO GPU JOB. NO SCIENTIFIC CHECKPOINT. NO FROZEN CONFIG CHANGE.
NO E04/E05/E07c IMPLEMENTATION. NO COMMIT. NO PUSH.
