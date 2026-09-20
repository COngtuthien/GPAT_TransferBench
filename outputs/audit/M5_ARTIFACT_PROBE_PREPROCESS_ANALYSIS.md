# M5 — ArtifactProbeNet Preprocessing Analysis (pre-flight)

> **RESOLVED 2026-09-20.** Every question raised below was answered by the owner and frozen in `configs/frozen/artifact_probe.yaml` (sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f`). The decision record is `M5_ARTIFACT_PROBE_OWNER_DECISIONS.md` and the contract summary is `M5_ARTIFACT_PROBE_FROZEN_CONTRACT.md`. **The text below is preserved unchanged as the analysis that motivated those decisions.** M5 is still NOT_STARTED: nothing has been trained.

**Date:** 2026-09-20 · Read-only measurement on 36 frozen M2 canonical faces (12 per dataset,
deterministically hash-selected from TRAIN+VAL only — **TEST was never opened**).
Tool: `tools/m5_probe_preprocess_probe.py` · Data: `M5_HP_VARIANT_MEASUREMENTS.json`
Candidate implementation: `gpatbench/probe/preprocess.py` (**not frozen**)

## 1. What is fixed

```
HP(x) = x - GaussianBlur(x, kernel=9, sigma=1.5)        # spec §4 table, §12.1
probe input: 224x224 high-pass RGB                       # spec §12.1
```

Nothing else about the transform is stated. The table below turns each remaining choice into a
measured number, so it is clear which are real and which are noise. The reference scale is the
residual itself: the typical per-face peak `|HP|` is **≈ 0.245** in the unit interval (range over
the 36 faces: 0.096 – 0.589).

## 2. Measured impact of every open choice

| choice | variants compared | max abs difference | mean abs difference | verdict |
|---|---|---|---|---|
| Gaussian implementation | OpenCV vs torchvision | **6.6e-07** | 5.7e-08 | **not execution-affecting** (float32 rounding) |
| Gaussian implementation | OpenCV vs explicit separable conv | 2.3e-07 | 2.7e-08 | **not execution-affecting** |
| Gaussian implementation | torchvision vs explicit conv | 5.8e-07 | 5.2e-08 | **not execution-affecting** |
| value domain | uint8→/255 after HP vs [0,1] before HP | 3.3e-07 | 2.8e-08 | **not execution-affecting** (HP is linear) |
| **border mode** | reflect101 vs replicate | **0.2604** | 1.6e-04 | **execution-affecting** (border pixels) |
| **border mode** | reflect101 vs constant-0 | **0.5990** | 4.0e-03 | **execution-affecting** |
| **resize order** | resize→HP vs HP→resize | **0.0836** | 2.3e-03 | **execution-affecting** (≈34% of the typical peak) |
| **interpolation** | AREA vs BILINEAR | **0.0449** | 9.9e-04 | **execution-affecting** |
| **interpolation** | AREA vs BICUBIC | **0.1317** | 2.0e-03 | **execution-affecting** |

So four things must be decided (Q-07 / D-M5-03) and two are settled by measurement.

## 3. Item-by-item resolution status

**A. Source image — RESOLVED by the project.** The frozen M2 canonical face, 256×256 RGB uint8
(`preprocess_v1.yaml`). No other image exists at this stage and §12.1 offers no alternative.

**B. Resize order — OPEN (part of Q-07).** `resize_then_highpass` and `highpass_then_resize` differ
by up to 0.0836. They are not equivalent: resizing a residual is not the residual of a resized
image, because the Gaussian and the resampling kernel do not commute.

**C. Interpolation — OPEN (part of Q-07).** AREA / BILINEAR / BICUBIC differ by up to 0.132. Note
§4 already fixes a project convention for the *canonical face*: INTER_AREA when downscaling,
INTER_CUBIC when upscaling. 256→224 is a downscale, so INTER_AREA is the consistent candidate —
but §4's rule is scoped to building the canonical face, not to the probe input, so adopting it is
an owner decision rather than an inheritance.

**D. Gaussian implementation — EFFECTIVELY SETTLED, still pin one.** OpenCV, torchvision and an
explicit separable convolution agree to 6.6e-07. The scientific result does not depend on the
choice. One should still be pinned so the checkpoint is byte-reproducible. Project precedent:
`tools/audit_casia_resize.py` already uses `cv2.GaussianBlur(y, (9,9), 1.5)` for this operator —
diagnostic precedent, not authority.

**E. Border / padding mode — OPEN (part of Q-07).** reflect101 (OpenCV default) vs replicate differ
by 0.26; vs constant-0 by 0.60. This only affects the 4-pixel margin, but the probe looks at
high-frequency residuals, where a border artefact is exactly the kind of cue it might latch onto.
Note the project already fixes `boundary_mode=reflect` for the Haar DWT (§4), which is a nearby
convention but a different operator.

**F. Pixel domain before HP — SETTLED.** uint8/255 and [0,1] agree to 3.3e-07 because HP is linear.
Either is fine; `[0,1] float32` is the natural form.

**G. Signed residual — MUST be retained.** Measured range across the 36 faces: min ≈ **−0.469**,
max ≈ **+0.589**. Clipping to [0,1] or round-tripping through uint8 would destroy roughly half the
signal. This is not a judgement call: any implementation that discards the sign is wrong.

**H. Post-HP normalization — OPEN (D-M5-03).** The spec is silent. Applying ImageNet mean/std to a
zero-centred residual maps the input to roughly **[−4.17, +0.81]** (a residual near 0 minus a mean
of ~0.45 divided by a std of ~0.22 lands near −2), which is far from the distribution the pretrained
weights were trained on. Omitting it leaves a near-zero-mean, low-variance input, which also does
not match. Both are defensible, they are not equivalent, and the spec does not choose — so neither
may be assumed. **This must not be inferred from "it is a pretrained ResNet".**

**I. dtype — SETTLED.** float32 for the network input.

## 4. Candidate implementation

`gpatbench/probe/preprocess.py` implements the operator with `resize_order`, `interpolation`,
`border_mode` and `post_normalization` as **required keyword arguments with no defaults**. A caller
must state every open decision explicitly, and unknown values raise `ProbeContractError`. The module
therefore cannot be used to freeze a choice by accident while the contract is open.

Tested properties that hold for **every** variant (`tests/test_m5_probe_preflight.py`): output shape
`3×224×224`, float32, constant image → exactly zero in the interior, signed residual preserved,
channel order preserved as RGB, repeated calls bit-identical, and a known impulse response matching
an explicitly constructed Gaussian kernel.

## Resolution (2026-09-20)

**Q-07 and D-M5-03 resolved.** Whole 256×256 canonical face, **no crop**; `uint8 → float32/255`; **OpenCV `INTER_AREA`** resize 256→224; then per-channel `GaussianBlur(9×9, σ=1.5, BORDER_REFLECT_101)`; signed residual; **no post-high-pass normalisation of any kind**. Output float32 `3×224×224`, RGB. The frozen entry point is `gpatbench.probe.preprocess.frozen_probe_input`, and the candidate API above keeps its no-default arguments so the two cannot be confused.
