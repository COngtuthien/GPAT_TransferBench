# M5 — ArtifactProbeNet Owner Decisions (2026-09-20)

**These are OWNER RESOLUTIONS of details the frozen specification left under-specified.** The
original §12.1 did **not** uniquely dictate them, and no report may present them as if it had. The
pre-decision analysis is preserved in `configs/proposed/artifact_probe.proposed.yaml` and the five
`M5_ARTIFACT_PROBE_*` pre-flight documents.

Frozen contract: `configs/frozen/artifact_probe.yaml`
sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f`
**M5 remains NOT_STARTED** — no authoritative training has run and no checkpoint exists.

---

## D-M5-01 — Classification population · `RESOLVED_BY_OWNER_LIVE_PLUS_SPOOF_ATTACK_MACRO`

- **Previous ambiguity:** "Labels: attack_macro over real TRAIN only" could mean the spoof attack
  families only (K=6) or every real row (K=7), because `attack_macro` is a §3.2 enum that includes
  `live` and is populated on every row.
- **Owner decision:** all real M2-COMPLETE samples — TRAIN for optimization, VAL for checkpoint
  selection. **K = 7.** No synthetic, no TEST.
- **Rationale:** `live` is an authoritative `attack_macro` token in the frozen population, and a
  probe that can emit it can report that a synthetic image lost enough attack evidence to look
  live. This is an independent measurement probe and does not violate Amendment A1.
- **Consequence:** TRAIN 14,467 / VAL 3,121 rather than 8,838 / 1,905; `live` becomes the largest
  class and enters the macro-F1 average; the class weights and the selection metric change
  accordingly.

## Class order · frozen lexical, K = 7

`live, makeup, mask_2d, mask_3d, partial, print, replay` → indices 0…6.
`other_spoof` is **excluded** — it has zero TRAIN and zero VAL observations, and no unused output
logit is created. The mapping is stored in the frozen config and must be stored in the checkpoint.

## D-M5-02 — Class weights · `RESOLVED_BY_OWNER_BALANCED_INVERSE_FREQUENCY`

- **Previous ambiguity:** "inverse-frequency weights clipped to [0.5, 3.0]" named no normalisation,
  and the pre-flight showed the literal `w_c = 1/n_c` clips **every** class to 0.5, i.e. no
  weighting at all.
- **Owner decision:** `w_raw(c) = N / (K · n_c)`, then `w(c) = min(3.0, max(0.5, w_raw(c)))`,
  computed in **float64** over the **TRAIN** population only. No renormalisation after clipping, no
  per-dataset weights, no `WeightedRandomSampler`, nothing derived from VAL or TEST.
- **Recomputed from `split_v1.parquet` (`fb9aeb36…`), asserted against the expected counts:**

| class | n (TRAIN) | `w_raw = N/(K·n)` | `w` after clip |
|---|---|---|---|
| live | 5,629 | 0.367154785168642 | **0.5** *(clipped low)* |
| makeup | 759 | 2.722943722943723 | 2.722943722943723 |
| mask_2d | 96 | 21.52827380952381 | **3.0** *(clipped high)* |
| mask_3d | 1,056 | 1.957115800865801 | 1.957115800865801 |
| partial | 1,911 | 1.0814831427076326 | 1.0814831427076326 |
| print | 2,838 | 0.7282291352058794 | 0.7282291352058794 |
| replay | 2,178 | 0.9489046307228125 | 0.9489046307228125 |

N = 14,467, K = 7. Full record: `M5_ARTIFACT_PROBE_CLASS_WEIGHTS.csv`.
- **Consequence:** the clip binds exactly two classes; the remaining five carry genuine
  inverse-frequency weighting. The runtime tensor is built in the frozen class order and is
  float32 to match the model dtype (element-wise cast of the float64 values; max cast error
  6.0e-08). The trainer recomputes the weights from the manifest and refuses on any mismatch.

## Q-07 — Input geometry · `RESOLVED_BY_OWNER_FULL_FACE_RESIZE_THEN_HP`

- **Previous ambiguity:** the recorded Q-07 was resize vs crop; the pre-flight showed the resize
  order, the interpolation and the Gaussian border mode are equally execution-affecting (measured
  up to 0.084, 0.132 and 0.599 against a typical residual peak of 0.245).
- **Owner decision:** the **whole** frozen 256×256 canonical face, **no crop of any kind**;
  uint8 → float32 [0,1]; resize the complete image 256→224 with **OpenCV `INTER_AREA`**; then
  per-RGB-channel `GaussianBlur(9×9, σx=σy=1.5, BORDER_REFLECT_101)`; then
  `HP = resized − blurred`, signed. No stochastic geometry.
- **Rationale:** deterministic downsampling of the complete canonical face preserves the full M2
  face support and matches §4's own downscale convention.
- **Consequence:** the probe sees the entire face at a fixed scale; the residual is computed after
  resizing, so it is the high-frequency content of the 224 image rather than a resampled 256
  residual.

## Q-07 (numeric domain) and D-M5-03 — `RESOLVED_BY_OWNER_NO_POST_HP_IMAGENET_NORMALIZATION`

- **Previous ambiguity:** the spec never said whether ImageNet mean/std is applied after the
  high-pass; the pre-flight measured that doing so maps the input to ≈[−4.17, +0.81].
- **Owner decision:** **no** post-high-pass normalisation of any kind — no ImageNet mean/std, no
  dataset statistics, no min-max. The only scaling anywhere is `uint8 → float32 / 255.0` before
  resize and high-pass. The residual stays **signed**: never clipped to [0,1], never offset by 0.5,
  never mapped to uint8, never absolute-valued, never squared, never converted to grayscale. Output
  is float32 `3×224×224`, channel order RGB.
- **Explicitly recorded:** *pretrained initialization ≠ a requirement to apply natural-RGB ImageNet
  normalization to a signed residual domain.*
- **Consequence:** the network input distribution is deliberately unlike ImageNet's; the pretrained
  weights are an initialization, not a distributional assumption.

## D-M5-04 — Fine-tune scope · `RESOLVED_BY_OWNER_FULL_FINE_TUNE`

- **Previous ambiguity:** §12.1 named the initialization, not the optimisation scope.
- **Owner decision:** `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` with `model.fc` replaced by
  `Linear(512, 7)`. **Every** parameter trainable — `conv1`, `bn1`, `layer1…layer4`, `fc` — under a
  single AdamW. No frozen backbone, no differential LR, no gradual unfreeze. The global seed is set
  **before** the classifier is constructed so its initialization is reproducible.
- **Verified:** 11,180,103 parameters, all trainable; `fc.out_features = 7`.
- **Consequence:** the backbone adapts to the signed-residual domain, which is consistent with
  D-M5-03 — a frozen ImageNet backbone would be reading an input distribution it never saw.

## D-M5-05 — VAL macro-F1 · `RESOLVED_BY_OWNER_FIXED_CLASS_SET_MACRO_F1`

- **Previous ambiguity:** class set, `zero_division` and implementation were unstated.
- **Owner decision:** a 7×7 int64 confusion matrix over **all** real VAL samples in the frozen class
  order. Per class, `precision = TP/(TP+FP)` (0 if the denominator is 0), `recall = TP/(TP+FN)`
  (0 if 0), `F1 = 2PR/(P+R)` (0 if 0); `macro_F1` = the arithmetic mean over **all seven** frozen
  classes. A class that receives no prediction is **not** dropped. No sklearn dependency, no
  weighted F1, no micro F1, no dataset-macro averaging.
- **Consequence:** the metric is comparable across epochs even when a rare class (`mask_2d`, 24 VAL
  samples) collapses, because its F1 contributes 0 rather than disappearing from the average.

## D-M5-06 — Checkpoint selection · `RESOLVED_BY_OWNER_MAX_VAL_MACRO_F1_EARLIEST_TIE`

- **Previous ambiguity:** no rule for two epochs with identical macro-F1.
- **Owner decision:** evaluate at the end of every epoch 1…30; replace the best **only** when
  `new_macro_f1 > best_macro_f1` (strict `>`, never `>=`), so an exact tie keeps the **earlier**
  epoch. No epsilon tie window, training loss is not a tie-break, TEST is never involved.
- **Consequence:** selection is a deterministic function of the VAL macro-F1 sequence alone.

## D-M5-07 — Cosine schedule · `RESOLVED_BY_OWNER_LITERAL_COSINE_NO_WARMUP`

- **Previous ambiguity:** "cosine decay" named no implementation, stepping, `T_max`, `eta_min` or
  warmup.
- **Owner decision:** AdamW `lr = 1e-4`, `weight_decay = 1e-4`;
  `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=30, eta_min=0.0)`, stepped **once per epoch**
  after that epoch's optimizer updates **and** after its validation/checkpoint measurement. Epoch 1
  therefore trains at exactly `1e-4`. No warmup, no per-batch cosine, no restarts, no `min_lr 1e-6`.
  The §14 downstream warmup is **not** imported.
- **Recorded:** the full epoch→LR sequence is in `M5_ARTIFACT_PROBE_LR_SCHEDULE.csv`. Epoch 1 =
  `1e-4`, epoch 30 = `2.7390523158633003e-07`, and the LR reaches `0.0` after the 30th step. The
  closed-form column agrees with the authoritative PyTorch scheduler to a maximum **relative**
  difference of 4.0e-16 (13 of 30 values are bit-identical); the scheduler, not the closed form, is
  authoritative.

## D-M5-08 — Augmentation · `RESOLVED_BY_OWNER_NO_DATA_AUGMENTATION`

- **Previous ambiguity:** `NO_AUGMENTATION_SPECIFIED` in §12.1, with a frozen §14.1 augmentation
  block nearby.
- **Owner decision:** canonical face → deterministic resize → deterministic high-pass, and nothing
  else. No `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomErasing`, rotation,
  affine, blur, noise, MixUp or CutMix.
- **Rationale:** §14.1 is scoped to the downstream evaluator; importing it would be an undocumented
  protocol change and would also silently resolve the Q-07 geometry through `RandomResizedCrop`.
- **Consequence:** each sample has exactly one deterministic representation, so the only stochastic
  element in training is the batch order.

## D-M5-08 — DataLoader

TRAIN `shuffle = true` under an explicit `torch.Generator` seeded with **42**; VAL `shuffle = false`
in canonical `sample_id` order. Both `batch_size = 64`, `drop_last = false`. GPU defaults
`num_workers = 4`, `pin_memory = true`, `persistent_workers = true`, `prefetch_factor = 2`. Each
worker derives deterministic Python / NumPy / torch seeds from the frozen seed. Sample membership
never depends on worker scheduling.

## Embedding

ResNet-18 penultimate vector after global average pooling, before `fc`; dimension **512**;
`F.normalize(embedding, p=2, dim=1, eps=1e-12)`. Never classifier logits, no projection head.

## E-M5-01 — Execution machine · `RESOLVED_BY_OWNER_GPU_REQUIRED`

- **Previous ambiguity:** the pre-flight found this machine is CPU-only while §12.1 requires AMP.
- **Owner decision:** authoritative M5 training runs on the GPU host **`sparc5090`**, expected root
  `/home/sparc/workdir/longnm/GPAT_TransferBench`. **CPU fallback is forbidden and AMP may not be
  disabled.** AMP is `torch.autocast(device_type="cuda", dtype=torch.float16)` with
  `torch.amp.GradScaler("cuda")` (or the exact non-deprecated equivalent for the pinned GPU build);
  if that is not correctly supported, **STOP** rather than silently switching to CPU, bfloat16 or
  float32. A separate GPU preflight must measure every remote fact before training
  (`M5_GPU_EXECUTION_PLAN.md`) — nothing about the remote environment is assumed here.
- **Consequence:** the trainer hard-refuses an authoritative run without CUDA, which was verified.

## Determinism

Seed 42 across Python `random`, NumPy, torch CPU, torch CUDA (all devices), the DataLoader
generator and the worker RNGs. `cudnn.benchmark = false`, `cudnn.deterministic = true`,
`torch.use_deterministic_algorithms(true)`, TF32 disabled for matmul and cuDNN,
`float32_matmul_precision = highest`. If an operation lacks a deterministic kernel the run **stops**
— the guard is never disabled silently. Scientific reproducibility is required;
**byte-identical checkpoints across different GPU architectures are not promised.**

## Preserved limitation (not "fixed")

`makeup`, `mask_2d`, `mask_3d` and `partial` occur **only** in SiW-Mv2 in the current population.
This is **not** corrected by resampling, dataset balancing, label merging or dropping classes.
ArtifactProbeNet is a frozen pooled measurement tool and some attack classes are dataset-specific in
the available corpus; later generator metrics must be interpreted accordingly.
