# M6D5c — E06c DSDG-BIN-IDFREE: global-statistic-preserving microbatch execution resolution

**M6D5c PASS.** Execution mode: **GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1**
(classification CONTROLLED_EXECUTION_ADAPTATION, owner-approved).

| Status | |
|---|---|
| **E06c_GLOBAL_BATCH_240_EXECUTION_RESOLVED** | this milestone |
| **E06c_GLOBAL_STATISTIC_MICROBATCH_QUALIFIED** | this milestone |
| **E06c_TRAINING_GRAPH_EXECUTION_QUALIFIED** | this milestone (synthetic, microbatch execution) |
| **E06c_PHYSICAL_BATCH_240_STOPPED_OOM_RETAINED** | M6D5b historical truth, unchanged |
| **E06c_FULL_TRAINING_NOT_YET_EXECUTED** | |
| **E06c REMAINS IMPLEMENTED_NOT_EXECUTED** | method status unchanged |
| **CONTROLLED_ADAPTATION PRESERVED** | Amendment A1, DEV-020 |

GLOBAL BATCH = 240 · MICROBATCH = 20 · MICROBATCHES PER STEP = 12 · OPTIMIZER STEP PER GLOBAL BATCH = 1 ·
FP32 · NO AMP · NO TF32.

Reporting label: DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation). This path is not native,
faithful or official DSDG. It is not a bitwise reproduction of a physical-B=240 step either, because that step cannot run on the qualified
single RTX 3090 (M6D5b). Every step here is a synthetic qualification step; none of it is scientific training.

## Authority and GPU synchronization

Laptop HEAD = origin/m6-baselines = `0c62dbe0bc07b305b80eabadf050d8aedeb22859` ("M6D5b: record E06c physical batch
240 OOM"). Divergence is 0 0 and the worktree was clean. `AGENTS.md` was checked with `test -f`: absent. There was no recursive or generic discovery.
The ledger had 112 rows and the artifact index 592 rows. The GPU started at `6e3dd33`, clean and at 0 0. We fetched only `refs/heads/m6-baselines`, proved
ancestry (distance 1) and ran `git merge --ff-only`, which reached `0c62dbe` (0 0, clean). No reset, clean, stash, commit or push was used.
The new files were staged on the GPU as temporary untracked copies and removed afterwards (see runtime log).

## Owner execution resolution (additive)

- Overlay: `configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml` (JSON subset of YAML).
- Addendum: `docs/spec/amendments/GPAT_TransferBench_v1_0_E06c_Memory_Execution_Resolution_Addendum_M6D5c.md`,
  with the full MMD and orthogonality gradient derivations.
- Implementation: `methods/dsdg/microbatch_execution.py`, which has static imports and receives Torch from the caller. The GPU harness is
  `methods/dsdg/microbatch_qualification.py`. It imports the M6D5a `runtime.py` and M6D5b `training_graph.py` /
  `training_qualification.py` helpers **unchanged**.
- Unmodified: the frozen configs, `adapter.py` (its generic accumulation refusal is still in place and is re-verified in-process),
  M6D5a/M6D5b code, tests, preflights and audits, the source cache and `source_pins.json`.

Why ordinary accumulation is forbidden: `loss_mmd = 50·|mean₂₄₀(z_nir) − mean₂₄₀(z_vis)|.mean()` and
`loss_ort = |mean₂₄₀(Σ_d z_cls·z_nir)|` take the absolute value *after* the batch mean. In the B=240 run, the naive
average of 12 per-chunk values would have been **MMD 19.373 vs true 14.099**
and **orthogonality 3.167 vs true 0.266**. We recorded these numbers as a diagnostic and did not use them.

## Algorithm executed

1. **Epsilon** (`STOCHASTIC_DRAW_REPLAY_FOR_RECOMPUTATION`): `eps_cls`, `eps_nir` and `eps_vis` [240,128] FP32 standard-normal
   tensors are drawn once from the CUDA default generator, in the order cls → nir → vis. The draw is bitwise equal to three pinned
   `misc/util.py::reparameterize(0,0)` calls from the same RNG state, and it leaves the same RNG state (both processes).
   After the draw, the CUDA and CPU RNG states were identical at pass-1 end, before the step and after the step, so **no redraw
   happened anywhere**.
2. **Pass 1** (`torch.no_grad`, encoders only, 12 × 20): FP32 `delta_sum` and `ort_sum` give `delta` [128]
   (54 positive, 74 negative, 0 zero coordinates) and `ort_mean = 0.26619086` (sign +1). This pass made no backward call and no
   zero_grad call.
3. **Pass 2**: `optimizer.zero_grad(set_to_none=False)` was called exactly once, with 0 backward calls before it. Then the same 12 chunks
   ran with autograd. Mean terms were weighted by 20/240, the MMD and orthogonality terms used the pass-1-signed global surrogates, and the chunk objective followed the epoch-1
   warmup assembly. There were 12 `backward()` calls and no step between chunks.
4. **One** `optimizer.step()`.

The pass-2 latents were **bitwise equal** to the pass-1 latents (max |Δ| = 0 for z_cls, z_nir and z_vis), and 0 signs differed. So the
surrogates reproduce the global value and subgradient exactly for these z. Surrogate sums: MMD 14.0986694 against global
14.0986710; orthogonality 0.26619088 against 0.26619086. The only differences are FP32 summation order.

LightCNN: the target `nir_fc` and `vis_fc` ran under `torch.no_grad` (`requires_grad=False` in all 24 chunks). `rec_nir_fc` and `rec_vis_fc`
kept their graph (`requires_grad=True`, grad_fn `DivBackward0`) and were not detached. In every chunk, a nonzero finite gradient
reached `rec_nir`/`rec_vis` at 128 px through the frozen LightCNN. netIP stayed frozen and in eval mode, with no parameter gradient and 0 changed
elements.

## Small-batch reference check (process `reference`, seed 60503) — PASS

Both paths used the real E06c models and LightCNN on the GPU, with identical initial model bytes (restored and verified by SHA256),
identical inputs (`synthetic_pair`, B=4) and identical pre-drawn epsilon. Path A is the **unchanged M6D5b full-batch
graph** (`training_graph.training_forward`, with the pinned forward → zero_grad → backward → step order). Path B is the two-pass microbatch graph with
microbatch 2 (2 chunks).

| Gate (frozen in overlay before execution) | Threshold | Observed | |
|---|---|---|---|
| scalar losses finite | all | all 7 × 2 finite | PASS |
| epoch-1 total relative error | ≤ 1e-4 | 4.15e-9 (14713.30859 vs 14713.30865) | PASS |
| aggregate gradient cosine (45,370,182 owned) | ≥ 0.999 | 0.99999997 | PASS |
| aggregate gradient relative-L2 | ≤ 1e-2 | 2.40e-4 | PASS |
| post-step parameter-update cosine | ≥ 0.99 | 0.99984 | PASS |
| owned gradient coverage | 55/55 | 55/55 both paths | PASS |

Per-term relative differences: rec 0, mmd 0, cls 0, pair 0, kl 2.2e-7, ip 4.8e-7, ort 3.4e-6.
Gradient sign agreement was 0.99992. The Adam update relative-L2 is 1.79e-2. This was reported but **not a gate**: the first Adam step is about
`lr·sign(g)` per element, so the ~8e-5 of elements whose tiny gradients flip sign each contribute ±2·lr.
The residual differences come from FP32 kernel/batch-shape effects. Path A and Path B latents already differ by ≤ 7.2e-6
before any loss is computed (B=4 vs B=2 cuDNN convolutions). They are not formula errors. On CPU in float64, the new tests show
gradient equality to 1e-9 for microbatch 1, 2 and 3 at epochs 1 and 2.

Reference accounting (separate from the B=240 qualification): 2 optimizer constructions, 2 applications
(one per path), 3 backward calls (1 full batch + 2 chunks). The zero_grad calls were Path A's pinned `zero_grad()` and Path B's
`zero_grad(set_to_none=False)`.

## B=240 qualification: two fresh processes (seed 60503, PYTHONHASHSEED 60503) — PASS

Inputs: `training_graph.synthetic_pair(240)`. Their SHA256 is **identical to the M6D5b OOM inputs**
(x_spoof `e4ec45db…12c8`, x_live `b4fea619…42c0`). All 480 rows are distinct. `label_spoof` is all 0. No file, decoder or manifest was used.
Models, LightCNN (60/60 matched, 0 missing, only `module.fc2.weight` filtered) and the optimizer
(45,370,182 parameters / 55 tensors, 1 group, 0 overlap with netCls/netIP, lr 2e-4, weight_decay 0) are exactly as in M6D5b.

| Global value (epoch-1 semantics) | Process 1 | Process 2 |
|---|---:|---:|
| loss_rec | 14903.623942057 | identical |
| loss_kl | 17.419939677 | identical |
| loss_mmd (global, pass 1 FP32) | 14.098670959 | identical |
| loss_mmd (float64 recomputation from pass-1 z) | 14.098669812 | identical |
| loss_ip | 6.334388494 | identical |
| loss_pair (λ=0) | 0.0 | identical |
| loss_cls (one-logit CE) | 0.0 | identical |
| loss_ort (global) | 0.266190857 | identical |
| **full_loss_epoch1** (rec + 0.01·(kl+mmd+ip+pair+cls+ort)) | **14904.005133957** | identical |
| sum of the 12 executed chunk objectives | 14904.005 | identical |
| post-warmup total (**algebraic only; not executed**) | 14941.743132045 | identical |

Gradients before the step: netE_nir 17/17 tensors (13,790,048 nonzero), netE_vis 17/17 (11,692,640), netG 21/21
(19,887,494). All are finite, so **55/55 owned tensors had gradients**. netCls had 2 tensors with zero gradient, as expected: the one-logit CE is identically 0, and the
`pre_spoof` gradient is 0 in every chunk. netIP had no gradients.
After the step: Adam state had 55 entries, all with step 1.0, only for owned parameters, and finite moments. Every owned tensor changed (max |Δ| = 2.0000e-4 = lr,
the first Adam step). netCls and netIP had 0 changed elements.

### CUDA memory (process 1; process 2 byte-identical values)

| Phase | Allocated | Peak allocated (window) | Peak reserved |
|---|---:|---:|---:|
| after model construction + LightCNN | 229,401,088 | 353,616,896 | 367,001,600 |
| after optimizer + B=240 inputs | 606,890,496 | 606,890,496 | 744,488,960 |
| after epsilon allocation | 607,259,136 | 607,996,928 | 744,488,960 |
| pass-1 peak | 641,185,280 | 735,575,552 | 786,432,000 |
| pass-2 chunk 00 peak | 863,330,816 | 5,792,892,928 | 5,907,677,184 |
| pass-2 chunks 01–11 peak (each) | ≈863.4 M | 6,015,169,536 – 6,015,476,736 | 7,226,785,792 |
| before optimizer step | 863,668,736 | 863,668,736 | 7,226,785,792 |
| after optimizer step (+ Adam moments) | 1,234,233,856 | 1,416,894,976 | 7,228,882,944 |
| **overall peak** | | **6,015,476,736 (5.60 GiB)** | **7,228,882,944 (6.73 GiB)** |

Physical B=240 needed more than 22.8 GB for the forward alone (M6D5b). Microbatch 20 peaks at 6.0 GB allocated, on a
23.56 GiB device. Wall time: pass 1 was 0.21 s and pass 2 (12 chunks) 3.86 s.

### Repeatability (process 1 vs process 2)

Bitwise equality was not required, but it is what we observed. The epsilon SHA256 (3/3), the delta SHA256, the mmd_sign SHA256, all 7 global losses and the epoch-1 total,
ort_mean, the gradient SHA256 (55/55), the Adam `exp_avg` SHA256 (55/55), the post-step parameter SHA256 (all 117 tensors across
5 models), the gradient inventories and every memory figure are identical. The process JSONs differ only in `started_utc`/`ended_utc`, PIDs and
`phase_seconds`. The algorithm makes no claim of bitwise equality with a hypothetical physical-B=240 execution: the summation order
differs (per-chunk FP32 accumulation, different cuDNN batch shapes), and we disclose that difference.

## Modern zero_grad (M6D5b issue resolved for this path)

The pinned `optimizer.zero_grad()` is realized as `zero_grad(set_to_none=False)`, called exactly once before the first microbatch. That
matches the PyTorch 1.6 in-place semantics. On this first step no gradient existed beforehand (0 non-None), so both semantics are
identical here. The run observed 55/55 owned gradient coverage. The coverage concern M6D5b raised for later iterations is carried to
M6D5d, which must verify 55/55 coverage every step.

## Environment and identity

The environment was `gpat-m6-e06c`, and the lock SHA256 `91416a20…4e95` was verified in-process. All 14 non-launch identity fields matched the lock:
Python 3.11.16, torch 2.12.1+cu130, CUDA 13.0, cuDNN 92000, driver 595.84, and the package set. The environment was identical before and after each
process. FP32 default dtype, matmul and cuDNN TF32 off, `NVIDIA_TF32_OVERRIDE=0`, autocast off, matmul precision `highest`,
cuDNN deterministic, benchmark off. Pinned source `16b793a7…` (tree `0d2216bd…`): all 46 transcribed statements
matched verbatim, and the 13-file closure equalled the lock before and after. LightCNN was 123,844,849 B, SHA256 `d0750746…9964`, the same before and after.
`compatibility_patch = NONE`. The GPU was clean before every process: 254 MiB used, no compute processes (shell and in-process
`nvidia-smi`). A shell check after processes 1 and 2 exited also showed 254 MiB. The only warning was the known `torch.cuda.FloatTensor` deprecation raised by pinned `reparameterize`, which the epsilon proof calls.
The firewall recorded 0 denials in all three processes (3 LightCNN read opens each).

## Process accounting (entire milestone)

| Count | reference (B=4) | B=240 process 1 | B=240 process 2 | B=240 total |
|---|---:|---:|---:|---:|
| optimizer constructions | 2 | 1 | 1 | **2** |
| optimizer applications | 2 | 1 | 1 | **2** |
| backward calls (Tensor.backward = autograd.backward) | 3 | 12 | 12 | **24** |
| zero_grad calls | 2 | 1 (`set_to_none=False`) | 1 (`set_to_none=False`) | 2 |
| autograd.grad / torch.save / activation checkpoint | 0 | 0 | 0 | 0 |

That makes 3 GPU qualification processes. The figures the brief specifies for the B=240 qualification (2 constructions, 2 applications,
24 backward calls) are met exactly. The small-batch reference check the brief also requires adds 2 synthetic B=4 optimizer
applications and 3 backward calls. We disclose these separately and do not fold them into the B=240 counts.

## Tests, preflight, ledger, index

The new file is `tests/test_m6d5c_e06c_microbatch_execution.py`. It has 16 static contract tests, 11 Torch formula tests (float64 CPU toy nets against
the unchanged M6D5b transcription) and 4 retained-evidence tests. Exact runs are recorded in the runtime log and the audit JSON. The static
preflight `tools/m6d5c_e06c_memory_preflight.py` imports no Torch, uses no CUDA and builds no model, optimizer or data. It
checks the retained evidence. Ledger: **112 → 113**, with one `M6D5C_E06C_MEMORY_EXECUTION_RESOLUTION` row; the first 112 rows are
byte-identical. The artifact index was rebuilt last, with CRLF line endings.

physical_batch_240 = OOM_RETAINED · benchmark_training = false · benchmark_data_access = false · TEST_access = false ·
checkpoint_created = false · synthetic_bank = false.

**NO BENCHMARK DATA ACCESS. NO TEST ACCESS. NO BENCHMARK TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK.
NO RECURSIVE FILESYSTEM SEARCH. NO COMMIT. NO PUSH.**
