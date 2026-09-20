# M5 — ArtifactProbeNet Frozen Contract (summary)

**Frozen:** 2026-09-20 · `configs/frozen/artifact_probe.yaml`
sha256 **`3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe`**
(pre-correction `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f` is SUPERSEDED_BY_VALIDATION_EPOCH_CONTRACT_CORRECTION; see
`M5_VALIDATION_EPOCH_CONTRACT_CORRECTION.md`)
Snapshot `frozen_config_snapshot/configs/frozen/artifact_probe.yaml` — byte-identical.
**Unresolved execution-affecting fields: 0.**

**M5 is still NOT_STARTED.** Pre-flight status: `READY_FOR_GPU_EXECUTION_PREFLIGHT`. No authoritative
training has run, no checkpoint exists, and `models/artifact_probe/` does not exist.

Decisions and their rationale: `M5_ARTIFACT_PROBE_OWNER_DECISIONS.md`. These are **owner
resolutions of under-specified details**, not things the frozen specification uniquely dictated.

## The contract in one table

| item | frozen value | source |
|---|---|---|
| purpose | independent measurement probe; **never** GPAT `E_art` | spec §12.1 |
| population | all real M2-COMPLETE rows; TRAIN 14,467 optimises, VAL 3,121 selects | D-M5-01 |
| TEST / synthetic | **never loaded** / **never present** | D-M5-01 |
| classes | `live, makeup, mask_2d, mask_3d, partial, print, replay`, **K = 7**, lexical | D-M5-01 |
| excluded class | `other_spoof` (0 TRAIN, 0 VAL; no unused logit) | D-M5-01 |
| class weights | `w = clip(N/(K·n_c), 0.5, 3.0)`, float64, TRAIN only, no renorm | D-M5-02 |
| input source | frozen M2 canonical face 256×256 RGB uint8, **no crop** | Q-07 |
| pipeline | `/255 → INTER_AREA 256→224 → GaussianBlur(9, σ1.5, REFLECT_101) → signed residual` | Q-07 |
| post-HP normalization | **none** | D-M5-03 |
| output tensor | float32 `3×224×224`, RGB | Q-07 |
| backbone | `resnet18(IMAGENET1K_V1)`, `fc → Linear(512, 7)`, **fully fine-tuned** | D-M5-04 |
| weight sha256 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | §10 |
| optimizer | AdamW lr 1e-4, wd 1e-4, batch 64, 30 epochs | spec §12.1 |
| scheduler | `CosineAnnealingLR(T_max=30, eta_min=0.0)`, once per epoch, **no warmup** | D-M5-07 |
| AMP | `autocast(cuda, float16)` + `GradScaler("cuda")`; CPU/bf16/fp32 fallback **forbidden** | §19 |
| augmentation | **none** | D-M5-08 |
| loaders | TRAIN shuffle (seeded generator, 42), VAL fixed order, `drop_last=false` | D-M5-08 |
| metric | fixed 7-class macro-F1, zero-division → 0, unpredicted classes **not** dropped | D-M5-05 |
| validation schedule | a VAL pass at the end of **every** epoch 1…30 (**30 passes**); `contract.validation_epochs()` is the only source | D-M5-06 |
| checkpoint | max VAL macro-F1, strict `>`, exact tie keeps the **earlier** epoch; any epoch in 1…30 may win | D-M5-06 |
| embedding | penultimate 512-D after GAP, `F.normalize(p=2, dim=1, eps=1e-12)` | §17 |
| execution | GPU host `sparc5090`; CPU authoritative training **forbidden** | E-M5-01 |
| determinism | seed 42 everywhere; cuDNN deterministic; TF32 off; stop on a missing kernel | §20 |

## Frozen class weights (recomputed from `split_v1.parquet` `fb9aeb36…`)

| idx | class | TRAIN n | VAL n | `w_raw` | `w` |
|---|---|---|---|---|---|
| 0 | live | 5,629 | 1,216 | 0.367154785168642 | **0.5** |
| 1 | makeup | 759 | 159 | 2.722943722943723 | 2.722943722943723 |
| 2 | mask_2d | 96 | 24 | 21.52827380952381 | **3.0** |
| 3 | mask_3d | 1,056 | 224 | 1.957115800865801 | 1.957115800865801 |
| 4 | partial | 1,911 | 406 | 1.0814831427076326 | 1.0814831427076326 |
| 5 | print | 2,838 | 623 | 0.7282291352058794 | 0.7282291352058794 |
| 6 | replay | 2,178 | 469 | 0.9489046307228125 | 0.9489046307228125 |

## The single code path and its refusals

`python -m gpatbench.cli train-probe --config configs/frozen/artifact_probe.yaml` →
`gpatbench.probe.train.run`. There is no second trainer. It refuses, rather than adapting, on:

a non-frozen config · a changed config sha256 · a snapshot that is not byte-identical · a changed
split-manifest sha256 · a class-count mismatch · a class-weight mismatch · a wrong ResNet-18 weight
hash · an authoritative run without CUDA · a TEST split · any augmentation flag · post-HP ImageNet
normalization · a model that is not fully trainable or does not emit 7 logits.

`--dry-run` performs the same gates plus model/loss/metric/scheduler construction and a single
forward pass, and never calls `optimizer.step()` or writes a checkpoint.

## Preserved limitation

`makeup`, `mask_2d`, `mask_3d` and `partial` occur **only** in SiW-Mv2. This is deliberately **not**
corrected by resampling, balancing, merging or dropping classes. Later generator metrics must be
interpreted with that in mind.
