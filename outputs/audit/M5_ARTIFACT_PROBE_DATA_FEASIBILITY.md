# M5 — ArtifactProbeNet Data Feasibility (pre-flight)

> **Config SHA updated 2026-09-21.** The ArtifactProbeNet config was corrected to state the validation schedule explicitly (`CONFIG_SEMANTIC_AMBIGUITY_FOUND_BEFORE_EXECUTION`; see `M5_VALIDATION_EPOCH_CONTRACT_CORRECTION.md`). Current sha256 `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe`; the sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f` quoted below is the pre-correction value, retained as history. No other scientific choice changed and nothing had been trained.

> **RESOLVED 2026-09-20.** Every question raised below was answered by the owner and frozen in `configs/frozen/artifact_probe.yaml` (sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f`). The decision record is `M5_ARTIFACT_PROBE_OWNER_DECISIONS.md` and the contract summary is `M5_ARTIFACT_PROBE_FROZEN_CONTRACT.md`. **The text below is preserved unchanged as the analysis that motivated those decisions.** M5 is still NOT_STARTED: nothing has been trained.

**Date:** 2026-09-20 · Read-only. Machine-readable: `M5_ARTIFACT_PROBE_PREFLIGHT.json`
Tool: `tools/m5_probe_preflight.py` · Source: `manifests/split_v1.parquet` (`fb9aeb36…`, unchanged)

**TEST rows are counted below for integrity only. TEST is never loaded, never trained on and never
used for checkpoint selection.**

## 1. Population

| split | total | live | spoof | CASIA | MSU | SiW |
|---|---|---|---|---|---|---|
| TRAIN | **14,467** | **5,629** | **8,838** | 3,360 | 1,600 | 9,507 |
| VAL | **3,121** | **1,216** | **1,905** | 672 | 320 | 2,129 |
| TEST | 3,027 | — | — | — | — | — |

TRAIN live by dataset: CASIA 840 · MSU 400 · SiW 4,389. TRAIN spoof: CASIA 2,520 · MSU 1,200 ·
SiW 5,118. There are **0 null or empty `attack_macro` values** anywhere.

## 2. `attack_macro` enum coverage

Spec §3.2 defines `attack_macro` as an enum — `live | print | replay | mask_2d | mask_3d | makeup |
partial | other_spoof` — and marks it present for **every** row (unlike `attack_raw`, which is
spoof-only). Measured on the split:

- present: `live, makeup, mask_2d, mask_3d, partial, print, replay` (7 tokens)
- **absent from the data:** `other_spoof` — no row carries it
- `live` appears on **exactly** the rows with `label_binary = 0`, and on no other row (verified)

That last fact is what makes the population question a real choice rather than a formality: the
`live` token is a genuine `attack_macro` value, not a placeholder.

## 3. Candidate population A — SPOOF classes only (K = 6)

TRAIN 8,838 · VAL 1,905 · 139 batches/epoch at batch 64 (138 if `drop_last`)

| class | TRAIN | VAL | TRAIN datasets |
|---|---|---|---|
| makeup | 759 | 159 | SiW only |
| mask_2d | **96** | 24 | SiW only |
| mask_3d | 1,056 | 224 | SiW only |
| partial | 1,911 | 406 | SiW only |
| print | 2,838 | 623 | CASIA, MSU, SiW |
| replay | 2,178 | 469 | CASIA, MSU, SiW |

Imbalance (max/min TRAIN) = **29.6**. TRAIN-only classes: none. VAL-only classes: none.

## 4. Candidate population B — LIVE + spoof classes (K = 7)

TRAIN 14,467 · VAL 3,121 · 227 batches/epoch at batch 64 (226 if `drop_last`)

Same six rows as above plus:

| class | TRAIN | VAL | TRAIN datasets |
|---|---|---|---|
| live | **5,629** | 1,216 | CASIA, MSU, SiW |

Imbalance (max/min TRAIN) = **58.6**. TRAIN-only classes: none. VAL-only classes: none.

## 5. Class ordering

**Lexical class token ascending**, which is deterministic and needs no external mapping. No existing
frozen mapping defines another order — `configs/frozen/attack_map_v1.yaml` maps dataset-native raw
tokens to `attack_macro` values but does not fix a class index order. The order is not frozen here,
because the class **set** depends on the unresolved D-M5-01.

## 6. Structural confound (must be disclosed)

Four of the six spoof classes (`makeup`, `mask_2d`, `mask_3d`, `partial`) occur **only** in SiW-Mv2.
Only `print` and `replay` span all three datasets. A probe can therefore score well partly by
identifying the dataset rather than the attack family. The §13 metrics that consume this probe are
computed within a dataset, so they remain usable; any cross-dataset reading of probe classes is not
supported and must not be claimed.

`mask_2d` is also genuinely rare: 96 TRAIN and 24 VAL samples. With K = 6, one VAL class holding 24
samples has a coarse F1 (each sample moves macro-F1 by roughly 0.7 percentage points), which makes
the checkpoint-selection metric noisy for that class — a further reason D-M5-05 must be frozen
rather than left to library defaults.

## 7. Feasibility (no training was run)

| quantity | value |
|---|---|
| ResNet-18 parameters (default 1000-class head) | 11,689,512 |
| parameter bytes, fp32 | 46,758,048 (≈ 44.6 MiB) |
| checkpoint estimate, weights only | ≈ 44.6 MiB |
| checkpoint estimate, weights + AdamW states | ≈ 133.8 MiB |
| penultimate feature dimension | **512** (matches the spec's 512-D embedding) |
| input tensor, fp32 3×224×224 | 602,112 B per sample |
| input tensor per batch of 64 | 38,535,168 B (≈ 36.8 MiB) |
| batches/epoch, population A | 139 (`drop_last=False`) / 138 (`True`) |
| batches/epoch, population B | 227 (`drop_last=False`) / 226 (`True`) |
| images to read per epoch | 8,838 (A) or 14,467 (B) frozen 256×256 PNGs |

**No wall-clock time, throughput or GPU-memory figure is reported**, because no training was run and
this machine has no GPU. Fabricating one would be worse than omitting it.

Storage: the probe reads the existing frozen M2 canonical faces in place. **No new image storage is
required** — the high-pass input is computed on the fly. The only new persistent artifacts would be
the checkpoint (≈ 45 MiB, git-ignored) and small text/JSON records.

## Resolution (2026-09-20)

**D-M5-01 resolved to candidate B: LIVE + spoof, K = 7.** The frozen population is TRAIN 14,467 / VAL 3,121 with the seven lexical classes; `other_spoof` is excluded because it has zero observations in both splits and no unused logit is created. 227 batches/epoch at batch 64 with `drop_last=false`. The SiW-only confound described above is **preserved, not corrected** — no resampling, balancing, merging or class dropping.
