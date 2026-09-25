# GPAT-TransferBench v1.0: E06c memory execution resolution (M6D5c)

Status: **OWNER-APPROVED · ADDITIVE · CONTROLLED_EXECUTION_ADAPTATION**.
E06c is still **DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation)**. Its fidelity class is
**CONTROLLED_ADAPTATION** (DEV-020) and its status is **IMPLEMENTED_NOT_EXECUTED**. This addendum records the owner's
execution resolution for the physical-batch-240 CUDA OOM that M6D5b recorded. That OOM stays historical truth:
M6D5b remains byte-identical **STOP_AND_REPORT** evidence and is not rewritten.

The machine-readable companion is
[`e06c_m6d5c_memory_execution_resolution.yaml`](../../../configs/amendments/e06c_m6d5c_memory_execution_resolution.yaml).
It is written in the JSON subset of YAML, so the Python standard library can parse it. The frozen method YAML
(`configs/methods/e06c_dsdg_bin_idfree.yaml`) and the A1 config (`configs/frozen/dsdg_bin_idfree_v1.yaml`) are
**not modified**.

This path must not be called native, faithful or official DSDG, and it is not a bitwise reproduction of a
physical-B=240 step. That step cannot run on the qualified single RTX 3090.

## 1. Why a resolution was needed

M6D5b ran the exact pinned graph (`train_generator.py:87-182`) at physical batch 240 on a clean RTX 3090.
It hit CUDA OOM in the netG forward (`forward_classifier_generator`) with 0 optimizer applications, 0 backward passes and no
parameter change. Upstream `train_generator.sh` launches `gpu_ids='0,1,2,3'`, so upstream `DataParallel`
spread the global batch of 240 over four GPUs.

## 2. Frozen execution resolution: `GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1`

| Field | Value |
|---|---|
| global / effective batch | 240 |
| microbatch | 20 (fixed; not tuned on loss or quality; OOM at 20 → STOP_AND_REPORT) |
| microbatches per optimizer step | 12 |
| optimizer steps per global batch | 1 |
| gradient accumulation | yes, **with explicit global-statistic correction** |
| naive microbatch averaging of MMD / orthogonality | **forbidden** |
| precision | FP32; AMP off; TF32 off |
| activation checkpointing / CPU parameter offload / optimizer offload | off |
| physical batch 240 | `OOM_RETAINED` |

## 3. Why ordinary accumulation is wrong for E06c

Pinned `train_generator.py:141` and `:147`:

    loss_mmd = 50 * |mean_240(z_nir) - mean_240(z_vis)|.mean()
    loss_ort = 1 * |mean_240(sum_d z_cls*z_nir)|

The absolute value is applied *after* the batch mean. In general `average_c |mean_c(x)| ≠ |mean_240(x)|`. On the
qualified synthetic B=240 batch, the naive 12-chunk average would give MMD 19.37 against the true 14.10, and orthogonality 3.17 against
the true 0.266. So these two terms are computed from statistics over the whole global batch.

## 4. Why every other term is exact under chunking

All pinned layers act per sample. `Encoder`, `Encoder_s` and `Decoder_s` use only `InstanceNorm2d` with
`affine=False, track_running_stats=False`. There is no BatchNorm and there are no buffers that could be updated. LightCNN's `F.dropout` is gated
by `self.training`, and netIP runs in eval mode. `loss_rec`, `loss_kl`, `loss_ip`, `loss_cls` and `loss_pair` are all means over rows. Each row
has the same element count, so `Σ_c (m/240) · mean_c(ℓ) = mean_240(ℓ)` exactly. The qualification processes re-check this
per model: 0 BatchNorm modules and 36 InstanceNorm modules, all without running statistics or affine parameters.

## 5. Stochastic latents: `STOCHASTIC_DRAW_REPLAY_FOR_RECOMPUTATION`

`eps_cls`, `eps_nir` and `eps_vis` ([240,128], FP32, standard normal, CUDA default generator) are drawn **once per
global batch**, in the pinned order cls → nir → vis. Pass 1 and pass 2 then reuse the same slices, with
`z = mu + eps·exp(0.5·logvar)` (the same arithmetic as pinned `misc/util.py::reparameterize`). Pass 2 does not
redraw. Only the point where the draw happens changes; the epsilon distribution does not. The qualification also shows that
`torch.empty(...).normal_()` returns bit-for-bit the same tensors as three pinned `reparameterize(0, 0)` calls
from the same CUDA RNG state. Both leave the generator in the same state afterwards. This is **not** a claim of bitwise
equivalence with an impossible physical-B=240 step.

## 6. Two-pass algorithm

**Pass 1** runs under `torch.no_grad`, with no optimizer, no backward, no netCls, no netG and no netIP. For 12 chunks × 20 it runs
netE_nir and netE_vis, builds z_cls, z_nir and z_vis from the replayed epsilon, and accumulates in FP32:
`delta_sum += Σ_b(z_nir − z_vis)` and `ort_sum += Σ_b Σ_d z_cls·z_nir`. Then

    delta = delta_sum/240      ort_mean = ort_sum/240
    mmd_sign = sign(delta)     ort_sign = sign(ort_mean)      (sign(0) = 0, the torch.abs subgradient at 0)
    loss_mmd_global = 50·|delta|.mean()     loss_ort_global = 1·|ort_mean|

**Pass 2** calls `optimizer.zero_grad(set_to_none=False)` exactly once, before the first microbatch. This is a
modern-runtime compatibility decision that follows the PyTorch 1.6 in-place zero_grad semantics more closely. It then re-runs the same 12 chunks in the
same order, with autograd on and the same epsilon slices: encoders, netCls, netG and the LightCNN identity path.

    w = m/240                                             (= 1/12 for m = 20)
    mmd_surrogate_c = 50/(240·128) · Σ_{b∈c,d} mmd_sign_d · (z_nir − z_vis)_{bd}
    ort_surrogate_c = 1 · ort_sign/240 · Σ_{b∈c} Σ_d (z_cls·z_nir)_{bd}
    epoch-1 chunk_total = w·rec + 0.01·(w·kl + w·ip + w·cls + w·pair + mmd_surrogate_c + ort_surrogate_c)

`chunk_total.backward()` is called once per chunk, with no step between chunks. After all 12 chunks,
`optimizer.step()` is called exactly once.

### 6.1 Derivation: MMD

Let `δ_d = (1/B) Σ_b (n_bd − v_bd)` and `L = (λ/D) Σ_d |δ_d|` (λ = 50, B = 240, D = 128). Then
`∂L/∂n_bd = −∂L/∂v_bd = λ/(B·D) · sign(δ_d)`. With `s = sign(δ)` detached from pass 1,
`S = Σ_c λ/(B·D) Σ_{b∈c,d} s_d (n_bd − v_bd) = (λ/D) Σ_d s_d δ_d = (λ/D) Σ_d |δ_d| = L`, and
`∂S/∂n_bd = λ/(B·D) · s_d`, which is exactly `∂L/∂n_bd`. When z is fixed and s is the pass-1 sign, S reproduces the
global value and its subgradient over the full global batch. The only differences are ordinary floating-point summation order.

### 6.2 Derivation: orthogonality

Let `μ = (1/B) Σ_b Σ_d c_bd n_bd` and `L = λ|μ|` (λ = 1). With `s = sign(μ)` detached,
`S = Σ_c λ s/B Σ_{b∈c} Σ_d c_bd n_bd = λ s μ = λ|μ|`, and `∂S/∂c_bd = λ s n_bd / B = ∂L/∂c_bd` (similarly for n).

### 6.3 Condition for exactness

The derivation needs the pass-2 latents to equal the pass-1 latents that fixed the signs. Epsilon replay makes this hold, and
so does recomputing on the same chunk shapes with deterministic cuDNN. In both B=240 processes, the pass-2 z_cls, z_nir and z_vis were
**bitwise equal** to pass 1, and the sign mismatches between pass 2 and pass 1 were 0.

## 7. LightCNN identity path

netIP stays frozen and in eval mode. It is not in the optimizer. The target features `nir_fc` and `vis_fc` are evaluated under `torch.no_grad`.
This is an **approved dead-gradient elimination**: the input images need no gradient, netIP is frozen, and `loss_ip` detaches the
targets. `rec_nir_fc` and `rec_vis_fc` are **not** under no_grad and are **not** detached, so the gradient flows from
`loss_ip` through the frozen LightCNN operations into the reconstruction and then into netG and the encoders. Every one of the 24 pass-2
chunks confirmed a nonzero finite gradient at the 128-px reconstructions.

## 8. Optimizer

`torch.optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()), lr=2e-4)`.
There is one optimizer and one param group, with 45,370,182 parameters in 55 tensors. netCls and netIP are excluded. There is one application per
global batch and no state update between microbatches.

## 9. Scope

M6D5c qualifies the execution mechanism on synthetic inputs only. It does **not** authorize benchmark data,
TRAIN/VAL/TEST execution, full training, a scientific checkpoint, a synthetic bank or the production 200-epoch
runner. Production execution, logging, resume and checkpoints belong to M6D5d. A microbatch other than 20 needs a
separate owner approval.
