# M7C2b — GPAT static core implementation + synthetic CPU qualification

Authority: `c9a12e10eff959a31aaa361cbff98469aae1e7c2` (M7C2a). Environment: `~/.venvs/gpat-m7-cpu` (CPU only, not
mutated). Synthetic tensors only: no GPU, no TRAIN/VAL/TEST sample, no teacher weights, no training loop, optimizer,
DataLoader, checkpoint or bank. R-04 Level 2 remains deferred (it blocks GPU/scientific training only).
Machine-readable evidence: `outputs/audit/M7C2B_GPAT_STATIC_CORE.json` (deterministic; regenerate with
`tools/m7c2b_gpat_static_core_evidence.py`).

## Modules (`methods/gpat/`)

| Module | Content | Authority |
|---|---|---|
| `config.py` | hash-verified read-only loader of GPAT-B0..B3 (+ snapshot equality, A10/M7B/M7C2a record hashes), immutable values, frozen architecture check | spec 23.2, M7B |
| `wavelet.py` | ptwt Haar level 1, reflect, axes (-2, -1), fp32; cA→LL, cH→LH, cV→HL, cD→HH | A10 D11, M7B R-01 |
| `highpass.py` | fp32 wrapper of the M7C2a reference `teacher_preprocess.highpass` ([-1, 1], 9×9, σ 1.5, reflect-101) | A10 D10.17, N-01 |
| `artifact_encoder.py` | E_art: ResNet-18 body, conv1 3→12 (extra = mean(RGB)/3), layer3 spatial code, z_a = L2(GAP(layer4)); IMAGENET1K_V1 from the SHA-256 verified local file only | spec 9.2, A10 D10.16/D10.24 |
| `heads.py`, `grl.py` | attack head Linear(512, 6) fixed order; identity head GRL(1.0) + Linear(512, 60) | A10 D02/D05, M7B |
| `generator.py` | NAFResidualUNet with pinned NAFBlock (R-05 loader), FiLM Linear(512, 2C) zero-init, skip after FiLM, AvgPool2d(2, 2) bottleneck, no bottleneck FiLM | spec 9.3, A10 D10.8-12, SHAPE_TRACE |
| `composition.py` | tanh/sigmoid activations, LL/HF composition, IDWT, artifact map A, D12 `artifact_scale` hook ∈ {0, 1} | spec 9.1/9.4, A10 D10.4/D12 |
| `discriminator.py` | PatchGAN on concat(RGB, HP), pix2pix layout, pad 1, 30×30 logits | spec 10.4, A10 D10.13 |
| `spectral.py` | S_radial / S_orient / L_spec (M7C2a reference forms, fp32) | A10 D01/D10.1-2, N-07 |
| `losses.py` | pure L_id, L_lm, L_parse, L_low (re-DWT of x_hat), N-08 selection LFErr, L_artcon, L_spec, L_type, DEV-022 L_idadv (sum/count + group), L_budget, L_TV, L_bg, face mask, L_D, L_Gadv, generator assembler (+λ_idadv) | spec 10, A10, M7C2a |
| `ema.py` | EMA of E_art / G_res only (floating EMA, non-floating copy), no disk I/O | A10 D10.21, N-06 |
| `schedule.py`, `batching.py`, `identity_labels.py` | closed-form LR/curriculum, 8838/4/2 accounting, caller-fed identity map | N-07, A10 D15, IDENTITY_CLASS_ORDER |
| `model.py` | GPATCore facade: E_art + G_res + enabled heads (+ optional, never-called D) | — |

## Parameter counts (all trainable)

| Module | Count | M7C1 expectation |
|---|---|---|
| G_res | 31,677,421 | ~31.68M |
| E_art (no FC) | 11,204,736 | ~11.2M |
| PatchGAN D | 2,767,809 | ~2.77M |
| attack head | 3,078 | — |
| identity head | 30,780 | — |

## Gates (measured)

| Gate | Value | Threshold |
|---|---|---|
| D11 DWT→IDWT, 100×3×256×256 uniform[-1, 1], seed 42 | 2.98e-7 | < 1e-5 |
| HP vs cv2 GaussianBlur reflect-101 | 1.19e-7 | ≤ 1e-6 |
| D12 zero residual: x_hat − IDWT(DWT(x_t)) | 0.0 | < 1e-5 |
| D12 zero residual: x_hat − x_t | 2.38e-7 | < 1e-5 |
| D12 source independence (same x_t, two x_s) | 0.0 | < 1e-5 |
| live source dependence (non-vacuity) | 2.15e-3 | > 1e-5 |
| γ = 0 LL_syn == LL_t | exact | exact |
| PatchGAN output | [N, 1, 30, 30] | 30×30 |
| training L_low, zero residual | 6.62e-8 | — (fp32 noise) |
| training L_low, randomly initialized residual | 6.66e-8 | no threshold |
| training L_low, perturbed x_hat (+0.1 sin pattern) | 0.127 | > 1e-3 (non-vacuous) |
| L_low gradient into x_hat (abs sum) | 1.79, finite | > 0 |
| N-08 selection LFErr at γ = 0 | 0.0 exact | exact |

## Owner review clarifications (M7C2b; no new deviation, no amendment)

- **Training L_low** (frozen spec §10.1, literal): `L_low = mean |DWT(x_hat).LL − LL_t|`. It takes a fresh, differentiable
  fp32 ptwt Haar level-1 reflect DWT of `x_hat`; `LL_t` comes from the original `DWT(x_t)`. It never reads the
  internal `LL_syn`, so γ = 0 does not bypass the DWT.
- **Selection LFErr** (M7C2a N-08, unchanged): `||LL_syn_internal − LL_t||_1 / (||LL_t||_1 + 1e-8)`, used only for
  checkpoint selection. It is exactly 0 for every γ = 0 candidate. The two contracts are separate functions
  (`losses.l_low`, `losses.lferr_selection`), and a static test fails if one implementation serves both.
- **PatchGAN** (IMPLEMENTATION_CLARIFICATION): every Conv4×4 has padding 1 and bias=True.
  - Layer 1: 6→64, s2, no norm, LeakyReLU(0.2).
  - Layers 2–4: 64→128 s2, 128→256 s2, 256→512 s1, each `InstanceNorm2d(affine=False, track_running_stats=False)` + LeakyReLU(0.2).
  - Layer 5: 512→1 s1, no norm, no activation, no sigmoid.
  - Output [N, 1, 30, 30].
- **DEV-022 group normalization** (RUNNER_CONTRACT_FROZEN_NOT_YET_IMPLEMENTED): `L_idadv_group = Σ_j ce_sum_j /
  Σ_j labelled_count_j`; it is an exact differentiable zero when no row is labelled. A microbatch's share is never
  multiplied again by the sample weight n_j / n_group. No training loop exists in M7C2b.
- Other choices made in the owner instruction:
  - FiLM layers are zero-initialized (identity modulation).
  - The 16→8 bottleneck pool is `AvgPool2d(2, 2)`. It is bitwise equal to `adaptive_avg_pool2d(8)` on CPU; the CUDA
    check belongs to the N-09 GPU qualification.
- The official IMAGENET1K_V1 state dict has no BatchNorm `num_batches_tracked` entries. Loading leaves those counters
  at 0. The focused test uses the locally cached official file (SHA-256 verified, never downloaded) when it is present.
- `methods/gpat/__init__.py`: the docstring now describes the current state. There is no code change, and the M7C2a
  version is still verifiable from git history.
