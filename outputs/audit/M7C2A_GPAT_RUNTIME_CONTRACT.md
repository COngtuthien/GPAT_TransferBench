# M7C2a — GPAT implementation/runtime contract freeze, CPU environment and pinned NAFNet source

| | |
|---|---|
| Milestone | M7C2a (contract freeze + CPU qualification; no GPAT architecture, training, checkpoint or bank) |
| Authority commit | `5f04912b652ae93f72ef6fc2dcdaa41dea55760d` (M7B) |
| Owner-resolution record | `configs/amendments/gpat_m7c2a_implementation_resolution.yaml` (layered on A10 + M7B; not an amendment; no DEV) |
| Machine evidence | `outputs/audit/M7C2A_GPAT_CPU_QUALIFICATION.json` (written by `tools/m7c2a_gpat_cpu_qualification.py`) |
| Preflight / tests | `tools/m7c2a_gpat_runtime_contract_preflight.py`, `tests/test_m7c2a_gpat_runtime_contract.py` |
| Environment | `gpat-m7-cpu` — `environments/gpat_m7_cpu.{conda-explicit.txt,pip-freeze.txt,runtime.json,lock.json}` |
| Date | 2026-09-30 |

## 1. Owner resolutions encoded

| Id | Decision | Class |
|---|---|---|
| R-04 | FaceXFormer CLIP_EMULATING_DIFFERENTIABLE_COMPATIBILITY: `v = clamp((x+1)/2, 0, 1)` → horizontal PIL-bicubic matrix pass → `clamp(0,1)` → vertical pass → `clamp(0,1)` → ImageNet normalization; no rounding; x_hat never modified. AdaFace: `clamp(x, -1, 1)` → exact INTER_AREA 256→112 matrix in [-1, 1] → BGR flip. Level-1 gates frozen; Level-2 contract frozen, executed on the GPU host before training (TRAIN only) | RUNTIME_COMPATIBILITY (refines A10 D07) |
| R-05 | Option D: AST-select `SimpleGate`, `NAFBlock`, `LayerNormFunction`, `LayerNorm2d` from the hash-verified pinned checkout and compile them unchanged in a `{torch, nn, F}` namespace; `ctx.saved_variables` verbatim; no BasicSR import | RUNTIME_COMPATIBILITY |
| N-01 | HP(x) = x − GaussianBlur(x) (9, σ 1.5, reflect-101, per channel) on the GPAT [-1, 1] tensor for E_art, F_art and D; no ImageNet normalization; ArtifactProbe unchanged | OWNER_SCIENTIFIC_CLARIFICATION_OF_UNSPECIFIED_INPUT_DOMAIN |
| N-03 | 224 categorical parsing mask → 256 by torch `nearest-exact`; foreground 1..10; 15×15 square dilation | IMPLEMENTATION_CLARIFICATION |
| N-04 | Pre-update joint gradient: x_hat once per microbatch; G loss with D gradient-disabled; D loss on `x_hat.detach()`; at the group boundary D unscale/clip/step/update, then G unscale/clip/step/update; no G re-forward; EMA after the G step | IMPLEMENTATION_CLARIFICATION |
| N-05 | D trains from generator update 1 (λ_adv = 0 in the G objective during epochs 1–5) | IMPLEMENTATION_CLARIFICATION |
| N-06 | E_art BatchNorm in train mode (warmup and all generator epochs), no SyncBN; full E_art state carries from warmup (optimizer/scaler do not); EMA over every floating state tensor, non-floating copied; EMA evaluated in eval mode | IMPLEMENTATION_CLARIFICATION |
| N-07 | Closed-form LR, attack-warmup LR and curriculum (below); soft Dice (classes 1..10, eps 1e-6) + 0.1·KL(p_t‖p_h) (floor 1e-8); S_radial (log1p power, floor radius, 128 bins r = 0..127, DC included); S_orient (raw power, DC excluded, 0 < r ≤ 0.5 cycles/pixel, 8 bins over [0, π)) | IMPLEMENTATION_CLARIFICATION |
| BCE | `L_D_real = BCEWithLogits(D(real), 1, mean)`, `L_D_fake = BCEWithLogits(D(fake.detach()), 0, mean)`, **`L_D = 0.5·(L_D_real + L_D_fake)`** (M7C2A-OBS-01 resolved; equal to one mean BCE over the concatenated balanced real/fake batch, measured difference 1.1e-16 in fp64); `L_Gadv = BCEWithLogits(D(x_hat), 1, mean)` with x_hat not detached and D gradient-disabled; logits/reduction never below fp32 | OWNER_IMPLEMENTATION_CLARIFICATION |
| Identity order | UTF-8 lexicographic sort of unique (dataset, source_subject) from TRAIN identity metadata (CASIA, MSU); K must equal 60; `identity_class_map.json` with source-manifest and mapping SHA-256; shared by B2/B3; not executed in M7C2a | IMPLEMENTATION_CLARIFICATION |
| N-08 | LFErr on the internal LL coefficients used to reconstruct x_hat (never a re-DWT of x_hat); γ = 0 gives exactly 0 for every candidate, so all 51 tie | NUMERICALLY_STABLE_EQUIVALENT_IMPLEMENTATION |
| N-09 | Seeds for Python/NumPy/torch CPU/CUDA; GPU: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cudnn benchmark off / deterministic on, TF32 off, `use_deterministic_algorithms(True)`; replacement only by a proven-equivalent fixed operator, else STOP | RUNTIME_DETERMINISM_COMPATIBILITY |
| Shape trace | no bottleneck FiLM; encoder skip after stage FiLM; decoder skip added before decoder NAFBlocks; decoder FiLM after them; bilinear ×2 then Conv3×3 with bias | IMPLEMENTATION_CLARIFICATION |

Schedules (tested endpoints): `lr(u) = 2e-4·(u−1)/5524` for u = 1..5525 (lr(1) = 0, lr(5525) = 2e-4); for
u = 5526..66300, `t = (u−5526)/60774`, `lr = 2e-6 + 0.5·(2e-4 − 2e-6)·(1 + cos πt)` (lr(5526) = 2e-4, lr(66300) = 2e-6);
same schedule for D. Attack warmup `lr(s) = 0.5·1e-4·(1 + cos(π(s−1)/1389))`, s = 1..1390 (1e-4 → 0). Curriculum:
u 1..5525 `s_hf = 0.02 + 0.03·(u−1)/5524`, λ_adv 0, λ_con 0.5, λ_spec 0.25; u 5526..16575 0.10 / 0.05 / 1.0 / 0.5;
u 16576..66300 0.15 / 0.10 / 1.0 / 0.5.

Resolved: **M7C2A-OBS-01** (L_D above). Recorded: **M7C2A-OBS-02** PACKAGING_METADATA_ANOMALY — the official
PyWavelets 1.9.0 wheel (sha256 `d76b7fa8…4521a`, equal to the PyPI digest) reports `pywt.__version__ == '1.8.0'` from its own
`pywt/version.py`; `importlib.metadata.version('PyWavelets') == '1.9.0'`, `pywt.__file__` lies inside `gpat-m7-cpu`, and the
installed `version.py` matches the dist-info RECORD. The value is recorded, never patched. Remaining deferred: **R-04
Level 2** only (blocks GPU runtime qualification and scientific training, not static implementation).
State: STATIC_CORE_IMPLEMENTATION_ALLOWED · GPU_RUNTIME_NOT_QUALIFIED · SCIENTIFIC_TRAINING_NOT_ALLOWED.

## 2. CPU environment `gpat-m7-cpu`

Exact replication of the committed gpat-m5 / m6_core_gpu capture: the conda layer from
`environments/m6_core_gpu.conda-explicit.txt` (exported package set identical, 30 packages) and the pip layer from
`environments/m6_core_gpu.pip-freeze.txt` installed offline with `--no-deps` from a hash-recorded wheelhouse (the same
`torch 2.12.1+cu130` / `torchvision 0.27.1+cu130` wheels; CUDA is unused on the laptop). `pip freeze --all` differs from
the base capture by exactly `+ptwt==1.0.1` and `+PyWavelets==1.9.0`; `pip check` is clean. No existing environment was
modified.

| Python | torch | torchvision | numpy | Pillow | OpenCV | ptwt | PyWavelets |
|---|---|---|---|---|---|---|---|
| 3.11.16 | 2.12.1+cu130 | 0.27.1+cu130 | 2.4.6 | 12.3.0 | 5.0.0 (headless 5.0.0.93) | 1.0.1 | 1.9.0 |

ptwt checks: D11 DWT→IDWT max error 2.98e-7 (< 1e-5); ptwt `(cA, (cH, cV, cD))` equals `pywt.dwt2` (1.8e-7); R-01
directional: horizontal stripes populate only cH, vertical stripes only cV.

## 3. Pinned NAFNet source (R-05)

`megvii-research/NAFNet@2b4af71ebe098a92a75910c233a3965a3e93ede4`, tree `8a8508a34a23261ea97af0e369ff6dcd6bc11c73`
(fetched 2026-09-30, depth 1, into git-ignored `third_party/source_cache/nafnet`; no weight file present).

| File | sha256 |
|---|---|
| `basicsr/models/archs/NAFNet_arch.py` | `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967` |
| `basicsr/models/archs/arch_util.py` | `5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005` |
| `basicsr/models/archs/local_arch.py` | `c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26` (provenance only, never executed) |
| `LICENSE` | `a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad` (MIT, megvii-model, followed by Apache-2.0 for BasicSR-derived portions) |

Selected definitions (AST line ranges): LayerNormFunction 264–289, LayerNorm2d 291–300 (`arch_util.py`); SimpleGate
22–25, NAFBlock 27–80 (`NAFNet_arch.py`). No `basicsr`/`lmdb` module is imported. CPU results: fp64 gradcheck PASS
(LayerNormFunction, NAFBlock); `ctx.saved_variables` works with the expected DeprecationWarning; native-autograd parity —
forward bitwise identical in fp64 and fp32; fp64 gradient differences ≤ 2.8e-14; fp32 gradient differences ≤ 1.3e-5
(LayerNorm weight, magnitude 69) and ≤ 1.2e-4 (NAFBlock parameters, magnitude 351); NAFBlock(32) has 8,224 parameters.

## 4. R-04 Level 1 (44-image deterministic synthetic corpus; no dataset image)

| Gate | Threshold | Measured |
|---|---|---|
| FaceXFormer matrix vs PIL float operator | ≤ 1e-6 | 2.1e-8 |
| FaceXFormer clip-emulating adapter vs PIL uint8 | ≤ 1.13 LSB | 1.098 LSB |
| … after ImageNet normalization | ≤ 0.0198 | 0.01922 |
| AdaFace matrix vs cv2 float operator | ≤ 1e-6 | 1.1e-14 |
| AdaFace adapter vs cv2 uint8 | ≤ 0.5/127.5 + 1e-6 | 0.0039216 |
| High-pass float path vs cv2 | ≤ 1e-6 | 1.8e-7 |
| ArtifactProbe float preprocessing vs frozen | ≤ 1e-6 | 3.0e-7 |

Diagnostics (not thresholds): FaceXFormer mean 0.150 LSB, RMSE 0.261, p99 0.758; AdaFace mean 0.00102, p99 0.00392.
Repeat run bitwise identical. Gradient structure (4 non-saturated synthetic cases, real frozen F_art, pinned FaceXFormer
architecture with random frozen weights, frozen AdaFace stand-in because the AdaFace code is not on the laptop): input
gradients finite and non-zero, teacher `.grad` None, x_t targets detached. N-09 CPU pre-evidence (not adopted):
`adaptive_avg_pool2d(16→8)` vs `AvgPool2d(2, 2)` bitwise equal; bilinear ×2 vs fixed matrices ≤ 4.8e-7 forward, ≤ 9.5e-7
gradient.

Level 2 was not executed. Its contract (TRAIN-only subset of up to 64 target faces per dataset by smallest
SHA-256(sample_id), 8 seeded re-quantization replicas per image, envelopes from the frozen path's own re-quantization
noise) is frozen in the record; its numeric thresholds are frozen in the GPU qualification record before training.
Owner-confirmed replica semantics: in the teacher's pre-quantization 0..255 domain add deterministic seeded noise in
[-0.5, +0.5] LSB, apply the exact frozen round/clip quantization, run the frozen reference teacher, and use the output
dispersion as the semantic noise floor (never ±0.5 added to already-quantized integer pixels).

## 5. What M7C2a does not change

The frozen DOCX; A1–A10 and the A10 record; the M7B record; `configs/methods/gpat_b{0,1,2,3}.yaml` and snapshots;
`configs/frozen/*`; `outputs/audit/method_status.csv`; the M6E closure; `deviation_report.md` (no new DEV);
STAGE_STATE; `models/registry.yaml`; `third_party/registry.yaml`; every existing environment and M6 artifact. Only
`third_party/source_pins.json` (additive `nafnet` entry) and `configs/CONFIG_STATUS.md` (appended note) change among
existing files. No GPU, no TRAIN/VAL/TEST sample, no training, checkpoint or bank. M6_CLOSED = true.
