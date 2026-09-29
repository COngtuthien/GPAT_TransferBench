# M6FC — E06b DSDG-NATIVE GPU synthetic graph and execution qualification (M6F-C)

**QUALIFICATION_ONLY_SYNTHETIC · NON_SCIENTIFIC.**
- No face image, VAL or TEST is read.
- No checkpoint and no bank is created.
- **scientific_training_completed = false.**

| Field | Value |
|---|---|
| Milestone | M6FC, classification `M6FC_E06B_GPU_QUALIFICATION` |
| Authority (laptop and GPU) | `91d28a72f78d8f424da23cf07a015782e532ba0e`; the GPU repo was fast-forwarded only, and its tracked state was clean before and after |
| GPU | NVIDIA GeForce RTX 3090, 595.84, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7 |
| Environment | `gpat-m6-e06c` (owner decision): Python 3.11.16, torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0, cuDNN 92000, NumPy 2.4.6; FP32, no TF32, no autocast |
| Environment lock | `environments/e06c.lock.json`, SHA256 `91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95`; the runtime identity equals the lock |
| Source pin | FaceX-Zoo `16b793a7564a4b9308cf94e62bdb2ffacb3a725a` (tree `0d2216bdbdd9`), 13 files verified; 46 pinned statements verified |
| LightCNN | SHA256 `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964`, 123844849 B, 60/60 tensors |
| Frozen config | `configs/methods/e06b_dsdg_native.yaml`, SHA256 `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` (unchanged) |
| Qualification seed | 60701 (never an experiment seed) |
| Harness | `methods/dsdg/native_qualification.py`, SHA256 `2c16a247199a735232c845f0bbe93b6b551909948967c9cd34bd7f6d2b07c718` (the bytes that ran). It reuses the M6D5a/b/c/d modules unchanged. |

## 1. Live graph contract

- The classifier is built live as `Cls(128, 2)`: weight [2, 128], bias [2]. netCls has 258 parameters; the other
  modules match the M6D5a counts.
- Both labels execute: print = 0 and replay = 1 appear in every batch and every chunk.
- Live coefficients are lambda_pair = 5, lambda_mmd = 50, lambda_ip = 1000, lambda_type = 10, lambda_ort = 1,
  with hdim = 128.
- The optimizer is the official Adam over netE_nir + netE_vis + netG (55 tensors), lr 2e-4, excluding netCls and netIP.
- The networks contain 0 BatchNorm modules and 36 InstanceNorm
  modules without running statistics.
- No E06c guard (`lambda_pair = 0`, K = 1) is used.

Path A losses, reference case B = 4, epoch 2 (raw × lambda = weighted):

| term | raw | lambda | weighted |
|---|---|---|---|
| loss_cls | 0.891354 | 10 | 8.91354 |
| loss_ip | 0.00631832 | 1000 | 6.31832 |
| loss_kl | 18.0273 | 1 | 18.0273 |
| loss_mmd | 0.630513 | 50 | 31.5257 |
| loss_ort | 11.61 | 1 | 11.61 |
| loss_pair | 0.00235727 | 5 | 0.0117863 |
| loss_rec | 14676.8 | 1 | 14676.8 |

**Term isolation** (each loss backpropagated alone, B = 4, both classes):
- `loss_cls` = 6.6342; encoder gradient L2 293.9; netCls
  gradient nonzero on 258 elements.
- `loss_pair` = 0.0108445; encoder/generator gradient L2 3.371.
- Both terms are active and finite.

## 2. Direct vs microbatch equivalence (pre-registered gates)

Path A is the pinned full-batch transcription. Path B is the unchanged M6D5c/M6D5d global-statistic microbatch path
(`run_global_batch_v2`). Both use identical initial bytes, inputs, labels and epsilon.

The gates are the owner-approved M6D5c gates, verified equal at runtime and fixed before the first E06b run:
- total-loss relative error ≤ 1e-4;
- gradient cosine ≥ 0.999;
- gradient relative L2 ≤ 0.01;
- update cosine ≥ 0.99;
- plus a per-term relative error bound ≤ 1e-4.

| B | chunks | epoch | max term rel | total rel | grad cosine | grad rel L2 | grad max abs | update cosine | update rel L2 | update max abs |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | [2, 2] | 2 | 6.32e-07 | 2.25e-08 | 0.99999995 | 3.25e-04 | 0.403 | 0.999784 | 0.0208 | 4.00e-04 |
| 6 | [4, 2] | 2 | 1.21e-06 | 1.26e-07 | 0.99999975 | 7.01e-04 | 2.56 | 0.999620 | 0.0276 | 4.00e-04 |
| 4 | [2, 2] | 1 | 1.42e-06 | 7.92e-09 | 0.99999995 | 3.19e-04 | 0.31 | 0.999788 | 0.0206 | 4.00e-04 |

Every gate passes in all 3 cases.
- The full-batch gradient L2 is about 1.05e+05, so the max absolute
  gradient difference is negligible relative to it.
- The update max absolute difference (4.00e-04) equals 2 × lr. On the first Adam step,
  g/sqrt(v) is sign-like, so near-zero gradient elements can flip sign under FP32 noise. That is why the registered
  gate is a cosine (≥ 0.99; observed ≥ 0.999620), not an elementwise bound.

## 3. Logical 240 and tail 120 (microbatch 20; one Adam step each; each in a fresh process)

| case | logical B | chunks | print/replay | total loss | loss_cls | loss_pair | peak allocated | peak reserved | wall clock | optimizer steps |
|---|---|---|---|---|---|---|---|---|---|---|
| b240 | 240 | 12 × 20 | 120/120 | 15038.994315 | 8.285620 | 0.010599 | 5.97 GB | 7.21 GB | 4.059 s | 1 |
| tail120 | 120 | 6 × 20 | 60/60 | 15028.336039 | 8.744138 | 0.010436 | 5.78 GB | 7.02 GB | 2.178 s | 1 |

In both cases:
- all losses and all owned gradients are finite and present;
- netCls gradients are nonzero;
- the owned parameters changed, while netCls and netIP stayed unchanged;
- there were no OOM, no smaller-microbatch retry and no firewall denial.

## 4. Worker / randomness (real torch DataLoader, 8 workers, mock relation, CPU only)

The mock relation has the exact E06b cardinalities: 3,720 spoof, 1,240 live, 60 subjects. Over 2 epochs:
- workers = 8 is honoured;
- the batch plan is 15 × 240 + 120;
- batch b is served by worker b % 8;
- each worker's seed is the per-epoch base seed + worker_id;
- **every live-partner draw equals `methods.dsdg.native.simulate_live_draws`**;
- replay is deterministic, a different seed changes the draws, and the draws are redrawn each epoch;
- every draw stays within the same subject;
- the relation is invariant to input order.

No native pair manifest was created.

**Aborted attempt (retained):** `workers.aborted-attempt-1` on the GPU host ended in STOP_WORKER_GATE. My harness's
expected-sequence builder omitted torch `RandomSampler`'s second `randperm(n)[:n % n]`, which is drawn when the sampler
is exhausted. Epoch 1 matched exactly; epoch 2 was mispredicted. The fix is in the predictor only (verified against
the torch 2.12.1 source); the E06b model and the loader contract are unchanged. This run was a CPU mock with no
scientific impact.

## 5. Fidelity (owner decision at the M6F-C review)

- **Target source fidelity:** `FAITHFUL_OFFICIAL` (spec §8.6).
- **runtime_fidelity_assessment = `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`.** This is an owner decision, and it
  is **not** `CONTROLLED_ADAPTATION`. The execution classification is `EXECUTION_RUNTIME_COMPATIBILITY`. The owner's
  rationale:
  - the effective batch remains exactly 240;
  - the architecture is unchanged;
  - all seven official DSDG losses and their coefficients are unchanged, with lambda_pair = 5 and K = 2;
  - optimizer semantics are unchanged;
  - direct vs microbatch equivalence passes every pre-declared gate;
  - the logical batch of 240 and the tail of 120 both execute;
  - microbatching changes only floating-point accumulation order;
  - a single RTX 3090 replaces upstream multi-GPU execution without changing the scientific method.
- **final_method_fidelity = `PENDING_M6F_D_PRODUCTION_QUALIFICATION`.** The real production runner is not yet
  qualified. If M6F-D passes without a new scientific deviation, the final E06b row may use
  `FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY`.
- **Frozen config:** its field stays `PENDING_M6F_C_RUNTIME_QUALIFICATION` and is not rewritten (E03 precedent).
- **Observable deviations (disclosed):**
  - results are not bitwise identical, because FP32 accumulation order differs;
  - on the first Adam step, near-zero gradient elements can flip sign (bounded by 2 × lr);
  - a single GPU is used instead of upstream's 4-GPU `DataParallel`, which sees the same global batch of 240.

The aborted worker attempt (section 4) is classified `QUALIFICATION_HARNESS_ISSUE_ONLY`. It is not an E06b model or
loader failure.

**E06b status:**
- `CONFIG_FROZEN`
- `STATIC_ADAPTER_IMPLEMENTED`
- `GPU_GRAPH_QUALIFIED`
- `EXECUTION_MAPPING_QUALIFIED`
- `WORKER_DETERMINISM_QUALIFIED`
- `PRODUCTION_RUNNER_NOT_YET_QUALIFIED`
- `SCIENTIFIC_TRAINING_NOT_EXECUTED`

**M6_CLOSED = false. M7 HAS NOT STARTED.**
