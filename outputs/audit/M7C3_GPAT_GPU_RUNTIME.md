# M7C3 — GPAT GPU runtime qualification + R-04 Level-2 teacher parity

Authority: `33955b05ff70289c42386c3ac1623bae4a5fae5e` (M7C2b). GPU host: `student20261@100.121.84.44`, one RTX 3090,
driver 580.178.04. The M6 capture recorded 595.84; this is recorded and is not a blocker, because the
torch/CUDA/cuDNN stack is pinned and deterministic repeatability is bitwise on the current stack. This is
candidate evidence; owner review comes before commit. No production runner, scientific training, checkpoint, bank,
VAL image or TEST image was involved.

## GPU runtime (`M7C3_GPAT_GPU_QUALIFICATION.json`, `M7C3_GPAT_GPU_REPEATABILITY.json`)

| Gate | Result |
|---|---|
| `gpat-m7-gpu` environment | offline clone of gpat-m5 (byte-identical to the committed capture) + ptwt 1.0.1 + PyWavelets 1.9.0, hash-pinned. `pip check` is clean. All 8 protected envs are unchanged. Lock SHA-256 `24c983eb…`. |
| NAFNet pin on torch 2.12.1 / CUDA | `saved_variables` works (DeprecationWarning only). LayerNorm forward parity is exact; gradient relative error is 1.8e-7 (fp32). NAFBlock is finite in fp32 and fp16 autocast. |
| Deterministic CUDA | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cudnn deterministic, TF32 off, `use_deterministic_algorithms(True)`. No GPAT op was rejected and no replacement was adopted. The bilinear fixed-matrix forward parity is 1.8e-7 (evaluated only). |
| Repeatability | Two fresh processes give byte-identical digests: 12 ops plus the B0 core (DWT, G_res, x_hat, A, D logits, loss, 4 gradients; fp32 and AMP). |
| D11 / D12 on GPU | 2.98e-7; zero residual 0.0; source independence 0.0. Same under AMP. |
| AMP / fp32 boundaries | Every conv/linear in E_art, G_res, D and the heads outputs fp16. Every required fp32 op outputs fp32 inside autocast. Teacher adapters must be called with autocast disabled. |
| B0 / B3 batch-4 forward | Finite. The shapes are identical; B3 adds [4,6] and [4,60] logits. |
| One synthetic accumulation group (B3) | Finite. D and G gradients are not contaminated. Clipping applies after unscale, and the binding-clip demo gives norm 1.000. The scalers are independent: with an injected inf, D skips its step and halves its scale, while G steps. |
| DEV-022 | Group CE / labelled count is exact and differs from sample-weighted means. Zero labelled rows give an exact differentiable 0. |
| VRAM, batch 4 (reserved) | B0/B3 forward 0.52 GiB; B3 one group 1.81 GiB; FaceXFormer adapter alone 2.87 GiB; B3 + all teachers 4.18 GiB. No OOM. |

## R-04 Level 2 (`M7C3_R04_LEVEL2_ATTEMPT1.json`, `M7C3_R04_LEVEL2_TEACHER_PARITY.json`)

The subset is 192 TRAIN live targets: 64 CASIA, 64 MSU, 64 SiW. The selection-list SHA-256 is `d02a8a96…`. Only these
192 PNGs were opened, each verified against the M2 face SHA-256. No VAL or TEST image was opened. Teachers:
AdaFace `52cca7c6…`, FaceXFormer `327a7558…`, F_art ResNet-18 `f37072fd…`, ArtifactProbe `b5ace6c2…`.

**Attempt 1 — clip-emulating adapter — FAIL (preserved byte-identical, SHA-256 `146de245…`).**
- The landmark max-abs check failed: 0.01423092931509018 > floor 0.013857558369636536.
- Every other gate passed.
- Cause: the float adapter was 1.05–1.12 LSB from the PIL input, while the replicas dither only ±0.5 LSB.

**Owner decision.** Exact frozen forward + approved differentiable surrogate backward. This is a RUNTIME_COMPATIBILITY
refinement of A10 D07; there is no new DEV and no gate was relaxed.

**Attempt 2 — `EXACT_FORWARD_SURROGATE_BACKWARD_COMPATIBILITY` — PASS** against the attempt-1 floors, used verbatim.

| Gate | Floor (attempt 1) | Attempt 2 |
|---|---|---|
| AdaFace min cosine | 0.9993640184402466 | 0.9997437000274658 (adapter unchanged) |
| Landmark max mean error | 0.004059360362589359 | 0.0 |
| Landmark max abs error | 0.013857558369636536 | 0.0 |
| Parsing min argmax agreement | 0.9574697017669678 | 1.0 |
| Parsing min D09 Dice | 0.8415264443192334 | 1.0 |

- **Input parity.**
  - The exact-forward uint8 output is bitwise equal to PIL on the 44-image corpus (CPU and GPU), on 2000/200 random images (CPU), and on 192/192 TRAIN faces.
  - The normalized fp32 tensors are bitwise equal: max_abs 0.0.
  - FaceXFormer landmark and parsing-logit outputs are bitwise identical to the frozen reference.
- **Surrogate backward.** It equals the M7C2a clip-emulating VJP bitwise (max_abs 0.0) and is finite and non-zero. Two fresh GPU processes give identical forward and gradient digests.
- **Teacher gradient structure.** Input gradients are finite and non-zero. Teacher parameter gradients are None, and the x_t targets do not require grad.

## Exact-forward adapter contract (`methods/gpat/teacher_preprocess.facexformer_input_exact`)

1. `u8 = clip(round(255·(x+1)/2), 0, 255)` with `torch.round` (half to even). This is the inverse of the canonical decode
   `u8/127.5 − 1` and is the M7C2a R-04 quantization operator.
2. The resample is Pillow 12.3.0 `libImaging/Resample.c` read from source (`Resample.c` SHA-256 `808c212d…`):
   - coefficients: float64 `precompute_coeffs` (bicubic a = −0.5, sequential normalization), then `normalize_coeffs_8bpc` `(int)(±0.5 + k·2²²)`;
   - accumulator: starts at 2²¹;
   - saturation: `clip8 = lookup[ss >> 22]`, i.e. saturate to 0..255;
   - pass order: horizontal pass, uint8 temp, then vertical pass.

   It runs as exact float64 integer arithmetic on the device, with no PIL and no CPU transfer.
3. ToTensor divides by 255 in fp32 using a 0-dim device tensor. CUDA scalar division multiplies by the reciprocal and
   was up to 1 ulp off; that was found and fixed before the recorded run. Normalize follows with the ImageNet mean/std.
4. Backward is a custom `autograd.Function` that re-evaluates the M7C2a `facexformer_input` surrogate under
   `enable_grad` and returns its `torch.autograd.grad` VJP. It is not recursive and is deterministic.

The M7C2a `facexformer_input` stays unchanged as the surrogate. Its 1.13-LSB approximation is no longer used in the
FaceXFormer teacher forward.

## Authority record and timing

Additive record: `configs/amendments/gpat_m7c3_gpu_runtime_resolution.yaml` (ADDITIVE_RUNTIME_QUALIFICATION_RECORD; not an
A10 or frozen-spec amendment; no DEV). It freezes the following; changing the quantization rule requires owner review:

- the owner-approved FaceXFormer teacher-input quantization rule, `u8 = clip(round_half_to_even(255·(x+1)/2), 0, 255)`
  (OWNER_RUNTIME_COMPATIBILITY_CLARIFICATION);
- the exact-forward / surrogate-backward contract;
- the attempt-1 floors;
- the attempt-2 results.

Informational batch-4 timing on the RTX 3090 (median over 50 synchronized iterations after 10 warm-up iterations) is
not a gate:

| Case | Median |
|---|---|
| exact adapter | 2.40 ms |
| FaceXFormer forward with the exact adapter | 37.68 ms |
| old clip-emulating adapter | 0.82 ms |

## Laptop data partition

`/dev/nvme0n1p5` was mounted read-only only to transfer the four teacher assets. The background Nautilus service was
closed with `nautilus -q`, then the partition was unmounted with `udisksctl unmount` at 2026-10-03 19:53 +07. There have
been no further reads.
