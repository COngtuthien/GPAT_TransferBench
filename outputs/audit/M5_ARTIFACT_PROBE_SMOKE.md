# M5 — ArtifactProbeNet CPU-Safe Smoke

**Date:** 2026-09-20 · **Result: PASS — 22/22 checks** · Machine-readable:
`M5_ARTIFACT_PROBE_SMOKE.json` · Tool: `tools/m5_probe_smoke.py`

**This is not a training run.** No `optimizer.step()` was called on real data, no checkpoint was
written, and no performance number is claimed or implied. The forward passes below exist only to
prove the wiring.

## 1. Contract

| check | result |
|---|---|
| frozen config loads and hashes to `e263b370…` | PASS |
| `frozen_config_snapshot` byte-identical | PASS |
| class order = `live, makeup, mask_2d, mask_3d, partial, print, replay`, K=7 | PASS |
| TRAIN/VAL class counts recomputed from `split_v1.parquet` and matching the frozen contract | PASS |
| class weights recomputed in float64 and equal to the frozen values | PASS |

## 2. Frozen transform on real canonical faces

8 deterministically hash-selected faces per split (TRAIN and VAL only):

| | TRAIN | VAL |
|---|---|---|
| tensor shape | `3×224×224` | `3×224×224` |
| dtype | float32 | float32 |
| min | −0.2923 | measured, negative |
| max | +0.2926 | measured, positive |
| mean | 1.07e-06 | ≈ 0 |
| signed residual | **yes** | **yes** |

The near-zero mean is the expected signature of a high-pass residual, and the mixed sign confirms it
was not clipped, offset or absolute-valued. Repeating the transform on the same face gives a
bit-identical tensor, and the frozen entry point is byte-equal to the explicit-option path under
the frozen options — so `frozen_probe_input` cannot drift from the recorded decisions.

## 3. Model and a single forward pass

| check | result |
|---|---|
| parameters | 11,180,103 — **all trainable**, 0 frozen |
| `fc` | `Linear(512, 7)` |
| forward on a real 4-sample VAL batch | logits `[4, 7]`, all finite |
| embedding | `[4, 512]`, L2 norms = 1.0 (max deviation 6e-08) |

## 4. Loss

`CrossEntropyLoss(weight=…, reduction="mean")` with the weight tensor in the frozen class order.
The weights are **computed in float64** per the contract and the tensor is float32 to match the
model dtype, which PyTorch requires; the cast is element-wise with a maximum error of 6.0e-08 and
does not reorder anything. Loss value finite.

## 5. Macro-F1 and checkpoint rule

| property | result |
|---|---|
| perfect prediction → macro-F1 = 1.0 | PASS |
| a class with no samples and no predictions → F1 = 0, not dropped | PASS |
| the vector always has 7 entries, even with one observed class | PASS |
| `is_better(0.5, 0.5)` is **False** → an exact tie keeps the earlier epoch | PASS |
| `is_better(0.5 + 1e-12, 0.5)` is **True** → strict improvement replaces | PASS |

## 6. Cosine schedule

`CosineAnnealingLR(T_max=30, eta_min=0.0)`, stepped once per epoch:

- epoch 1 LR = `1e-4` exactly (**no warmup**)
- epoch 30 LR = `2.7390523158633003e-07`
- LR after the 30th step = `0.0` = `eta_min`
- closed-form cross-check agrees to a maximum **relative** difference of 4.0e-16 (13/30
  bit-identical); the PyTorch scheduler is authoritative, the closed form is audit evidence only

Full sequence: `M5_ARTIFACT_PROBE_LR_SCHEDULE.csv`.

## 7. Refusals (all verified)

| attempt | result |
|---|---|
| `load_rows("TEST")` | **refused** |
| `ProbeDataset("TEST")` | **refused** |
| authoritative `run(dry_run=False)` on this CPU-only machine | **refused** (E-M5-01) |
| `load_config(configs/proposed/artifact_probe.proposed.yaml)` | **refused** (not the frozen config) |
| synthetic images in the loader | **structurally impossible** — every sample is resolved from the frozen M2 face store via the split manifest; there is no code path that accepts a generated image |

## 8. What was deliberately not done

No optimizer step on real data, no candidate checkpoint, no accuracy or macro-F1 value from real
predictions, and no claim about how the probe will perform. The model was constructed with
pretrained weights and run forward only to verify shapes, finiteness and the embedding contract.
