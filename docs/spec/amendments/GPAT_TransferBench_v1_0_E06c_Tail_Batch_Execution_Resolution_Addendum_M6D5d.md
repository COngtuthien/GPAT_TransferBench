# GPAT-TransferBench v1.0: E06c tail-batch execution resolution (M6D5d)

Status: **OWNER-APPROVED · ADDITIVE · CONTROLLED_EXECUTION_ADAPTATION**.
E06c is still **DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation)**. Its fidelity class is
**CONTROLLED_ADAPTATION** (DEV-020) and its status is **IMPLEMENTED_NOT_EXECUTED**. This is an execution policy, not a
retune of any model hyperparameter.

The machine-readable companion is
[`e06c_m6d5d_tail_batch_execution_resolution.yaml`](../../../configs/amendments/e06c_m6d5d_tail_batch_execution_resolution.yaml).
It is written in the JSON subset of YAML. The following stay unchanged: the frozen E06c configs, the M6D5c overlay, M6D5c V1
(`methods/dsdg/microbatch_execution.py`) and all M6D5a/M6D5b/M6D5c evidence. The M6D5b physical-B=240 OOM stays historical truth.

## 1. Why a second resolution was needed

`GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V1` (M6D5c) runs exactly B = 240 = 12 × 20, because its
`chunk_slices()` requires B to be divisible by 20. The pinned trainer
(`JDAI-CV/FaceX-Zoo@16b793a7`, `addition_module/DSDG/train_generator.py:94-97`) builds its loader like this:

    train_loader = torch.utils.data.DataLoader(
        GenDataset_s(img_root=args.img_root, list_file=args.train_list, attack_type=args.attack_type),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

The call passes no `drop_last`, and `drop_last` appears nowhere in the file. The PyTorch default is `drop_last=False`, and the qualified
runtime (torch 2.12.1) confirms this in-process. The frozen Track-A TRAIN relation has **8838** rows, so

    8838 = 36 × 240 + 198  →  36 full global batches + 1 final global batch of 198  →  37 optimizer steps per epoch.

Keeping that final partial batch is the source-consistent behaviour. V1 cannot run it.

## 2. `GLOBAL_STATISTIC_PRESERVING_MICROBATCH_V2`

For any actual global batch size `1 ≤ B ≤ 240`, V2 splits the batch into ordered, contiguous, exhaustive and disjoint chunks of at most 20
(`chunk_slices_v2`). There is no empty chunk. No sample is dropped, duplicated, padded or replaced, and nothing wraps around into the next
epoch. Examples: 240 → 12 × 20; 198 → 9 × 20 + 18; 21 → 20 + 1; 20 → 20; 19 → 19.

Every per-chunk computation is the **unchanged V1 function**, called with the actual B:

| Term | Per-chunk contribution |
|---|---|
| loss_rec, loss_kl, loss_ip, loss_cls, loss_pair | `(m/B) · chunk mean` |
| MMD | `λ_mmd / (B·128) · Σ sign(δ_B).detach() · (z_nir − z_vis)`, with `δ_B = mean_B(z_nir − z_vis)` from pass 1 over all B samples |
| orthogonality | `λ_ort · sign(μ_B).detach() / B · Σ_b Σ_d z_cls·z_nir`, with `μ_B = mean_B(Σ_d z_cls·z_nir)` from pass 1 |

The final smaller chunk gets no rescaling beyond these formulas. The M6D5c derivation (addendum M6D5c §6.1–6.3)
carries over word for word with 240 replaced by B. Summed over the chunks, the surrogates equal `λ_mmd·|δ_B|.mean()` and
`λ_ort·|μ_B|`, and they reproduce those terms' subgradients over the full global batch.

Epsilon: `eps_cls`, `eps_nir` and `eps_vis` are drawn **once for each actual global batch** as `[B,128]`, FP32 standard normal from the CUDA
default generator, in the order cls → nir → vis, and replayed in both passes. For B = 198 the tensors are `[198,128]`; they are never a
`[240,128]` draw truncated to 198. RNG consumption therefore follows the actual batch size.

Optimizer: `zero_grad(set_to_none=False)` once per global batch, then one backward per chunk (10 for B = 198), then
**one** `optimizer.step()` after all chunks, with no step between chunks. Adam still covers `netE_nir + netE_vis + netG` at lr 2e-4, with netCls and
netIP excluded. Precision is FP32 with AMP, TF32, activation checkpointing and CPU/optimizer offload all off. The maximum microbatch is fixed at 20. An OOM means
STOP_AND_REPORT, and the microbatch is never reduced automatically.

For B = 240, the V2 chunk plan equals V1's and V2 runs the same calls in the same order. This was checked
directly on the GPU.

## 3. Scope

M6D5d qualifies the tail-batch execution mechanism, the epoch batching arithmetic and the source provenance on synthetic
inputs only. It does **not** authorize benchmark training, VAL or TEST access, reading image bytes, scientific checkpoints, a synthetic
bank, or the production 200-epoch runner. The planned count of 37 optimizer steps per epoch (7400 for 200 epochs) is algebra, not
an executed count.
