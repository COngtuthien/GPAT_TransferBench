# M6D6i — E07c MAIN encoder-load integration + bounded training-graph qualification

**Method:** E07c DiffFAS-BIN-IDFREE (controlled encoder reconstruction) · CONTROLLED_ADAPTATION · DEV-021 ·
method status remains **IMPLEMENTED_NOT_EXECUTED**.
**Authority:** `84023aed3959d7ffda380b7612b860652579b443` (M6D6h owner freeze). **Qualification seed:** 60607.
`experiment_seed = null` (QUALIFICATION_ONLY_NOT_A_SCIENTIFIC_RUN).

This milestone is **QUALIFICATION TRAINING-GRAPH EXECUTION**, not **SCIENTIFIC TRAINING**. Two fresh qualification
processes each ran exactly one source-order iteration on synthetic structural input and then discarded all state.
Nothing trained here may be consumed by main scientific training, M8, downstream evaluation or paper results.

## Accounting (qualification vs scientific)

| Counter | Value |
|---|---|
| qualification_processes | 2 |
| qualification_checkpoint_deserializations | 2 |
| qualification_backward_calls | 2 |
| qualification_optimizer_steps = 2 | 2 |
| qualification_scheduler_steps | 2 |
| qualification_ema_updates | 2 |
| scientific_main_runs | 0 |
| scientific_optimizer_steps = 0 | 0 |
| scientific_checkpoint_deserializations | 0 |
| scientific_checkpoint_writes | 0 |
| experiment_seed_runs | 0 |

`torch.save` calls 0; main checkpoints 0; qualification checkpoints 0; TRAIN/VAL/TEST reads 0; manifest opens 0;
M8 outputs 0; firewall denials 0.

## Implementation

- `methods/difffas/main_graph.py` — reusable, I/O-free main DiffFAS core for M6D6j: `build_transform`
  (FAS_train.py:184-188), `build_dataloader` (:205), `build_training_objects` (:216-219, 222-223, 226 fresh-only,
  234-235), `epoch_progress` (:40), `train_iteration` (:31-33, 42-83) and `accumulate` (:21-26). Every replica is
  AST statement-equal to the pinned source (`verify_source_equivalence`); the pinned order
  `zero_grad -> backward -> scheduler.step -> optimizer.step -> accumulate` (:74-77, :83) is preserved, including
  scheduler.step() BEFORE optimizer.step(). Not replicated: print cadence, checkpoint save branch, visualization /
  sampling branch, resume body. FAS_train.py:34-35 is replaced by the A7 seam + caller `encoder.eval()`.
- `methods/difffas/main_graph_qualification.py` — `--mode process` (one qualification process) and
  `--mode launch` (GPU gates, frozen-checkpoint stat/SHA256 before/after, two fresh child processes, in-memory
  cross-process comparison over an anonymous pipe; nothing persisted).

## Frozen auxiliary encoder (first real deserialization)

SHA256 `49a24a3a7288782c144eaf4f58e83c76c37c1feeceea6de8a04b96a8fcfe107c`, 185136819 bytes, loaded ONLY through
`execution_policy.main_runner_encoder -> aux_checkpoint.load_frozen_aux_encoder`. Per process: 1 throwaway
`custom_rn.resnet18(pretrained=False)`, 1 SHA-verified audit event, 0 rejected, exactly 2 read-only opens of the
canonical path (verify_asset + read_verified), 1 `torch.load(BytesIO, weights_only=False)` after the SHA event,
unpickled globals == the M6D6c whole-module set. Before process 1 and after process 2 the file's size, SHA256,
inode, device, mode 664, nlink, uid/gid, mtime_ns and ctime_ns are identical. No chmod.

Encoder identity: `custom_rn.ResNet`, BasicBlock [3,4,6,3], fc Linear(512, 7) with bias, 46,233,707 parameters,
float32, cuda:0, `training=False` on every submodule after the caller's `encoder.eval()` (first statement inside
the seam). A6 shapes from the ONE loss-path encoder forward: x32x32 [4,256,32,32], x16x16 [4,512,16,16],
x8x8 [4,512,8,8], embg [4,7] (discarded). After backward every encoder grad is None and all encoder parameters and
buffers are bitwise unchanged.

## A7 RNG compatibility

The helper's CPU RNG state changes across the throwaway constructor; the CUDA, Python and NumPy states and CUDA
memory do not. The secure load and `encoder.eval()` change no RNG state. A post-hoc direct replay from the restored
pre-helper state reproduces the identical CPU state and object (17-way head, CPU). No 17-way object survives. The
DataLoader iterator sees the post-eval CPU state; replay from it reproduces first-batch order [0, 3, 1, 2]; the
no-constructor bypass control gives [7, 5, 2, 6]. The step itself consumes no CPU RNG.

## Training graph (exact B=4, synthetic structural input)

Synthetic N=8 in-memory analytic uint8 RGB 256x256 triplets `{content, style_spoof, GT}` (no label tensor) through
the pinned transform; DataLoader(batch_size=4, shuffle=True, num_workers=0); 1 iterator, 1 batch. This is a
STRUCTURAL QUALIFICATION SUBSTITUTE: the A1 TRAIN dataset, canonical face reader, guide mapping and production data
path are NOT qualified.

| Per process | Value |
|---|---|
| time_t | [177, 822, 605, 328] |
| cond_mask (prob 0.8) | [True, True, True, False] |
| loss | 1.0020060539245605 (`0x1.0083780000000p+0`) |
| mse | 0.9997536540031433 |
| vb | 0.002252309350296855 |
| applied first-step LR | 4.0000000000000003e-07 = anneal_linear(1e-5*0.04, 1e-5, 0/5000) |
| AdamW | lr 1e-5 (pre-scheduler), betas (0.9, 0.999), eps 1e-8, weight_decay 0, amsgrad False; state step 1 |
| EMA | accumulate(ema, model, 0): EMA named_parameters == main after the step; EMA buffers untouched |
| peak torch allocated / reserved | 13,690,067,456 / 15,497,953,280 bytes |
| nvidia-smi process peak | 15,152 MiB of 24,576 MiB |

Gradient reachability: 596 tensors hold autograd-connected gradients across time_embed, input_blocks, middle_block,
output_blocks, out and the 7 AttentionBlocks (126 tensors); 98 source-native None-grad tensors (unused
`cond_emb_layers.1` and self-attention `qkv`), identical in both processes. At the first step only `out.2.weight`
and `out.2.bias` are nonzero: the pinned `get_model_conf` zero-initialises the final projection, so every upstream
gradient is exactly 0. Exactly those two tensors change; AttentionBlock BatchNorm running statistics (21 buffers)
update in train mode. Source quirks recorded, not fixed.

## Repeatability (owner decision D1)

- **Pre-backward (gate):** batch identities/order, time_t, model input/cond_mask, encoder and model outputs,
  loss/mse/vb and every RNG observation are bitwise equal between the two fresh processes.
- **Post-backward — CHARACTERIZATION ONLY, not an acceptance gate, no tolerance:** gradients 0 of 149,953,798
  elements differ; main parameters after the step 0 of 159,363,974; EMA parameters 0 of 159,363,974 (max abs 0,
  max rel 0). The known nondeterministic op (bicubic interpolate backward, A7 leaves determinism NOT_SET) carried
  only zero gradients at this first step. This does not qualify BITWISE_DETERMINISTIC_MAIN_TRAINING.

B=4 fitted without OOM; `BLOCKED_BY_MAIN_B4_MEMORY` did not occur. No AMP, microbatch, TF32, activation
checkpointing or offload.

## GPU regression finding (disclosed, not caused by M6D6i)

On the GPU host 300/301 E07c regression tests pass; `test_m6d6e_e07c_aux_runner::test_24_cli_accepts_only_seed42`
errors because `tools/run_e07c_aux.py` (byte-identical) refuses to overwrite the existing M6D6g scientific run root on
that host. The laptop (no runtime root) passes all 301. The historical test was not modified.

## Statuses

Qualified: E07c_MAIN_RUNNER_ENCODER_LOAD_INTEGRATION_QUALIFIED, MAIN_RUNNER_ENCODER_LOAD_INTEGRATION,
E07c_MAIN_DIFFFAS_TRAINING_GRAPH_QUALIFIED, MAIN_DIFFFAS_TRAINING_GRAPH, E07c_MAIN_B4_TRAINING_MEMORY_QUALIFIED,
E07c_MAIN_A7_ORDER_RNG_COMPATIBILITY_QUALIFIED.

Not qualified: MAIN_PRODUCTION_RUNNER, MAIN_CHECKPOINT_RESUME, MAIN_DIFFFAS_SCIENTIFIC_TRAINING, M8_BANK.

## Carried to M6D6j (unchanged)

Real TRAIN path via the A1 manifest (first authorized main TRAIN access); run_logging_v1 and reporting interval;
checkpoint cadence + terminal checkpoint at 884,000 iterations; owner decision on the FAS_train.py:117-163
visualization branch (A7 3.1 cites its CPU randperm); checkpoint storage (~2.55 GB each, ~680 GB for three seeds vs
~414 GB free); spec smoke-test clause; tail batch of 2; wall clock; the M6D6e host-environment test finding.
