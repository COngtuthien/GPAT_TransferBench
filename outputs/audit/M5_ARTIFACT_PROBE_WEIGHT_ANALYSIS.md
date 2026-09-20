# M5 — ArtifactProbeNet Class-Weight Analysis (pre-flight)

> **Config SHA updated 2026-09-21.** The ArtifactProbeNet config was corrected to state the validation schedule explicitly (`CONFIG_SEMANTIC_AMBIGUITY_FOUND_BEFORE_EXECUTION`; see `M5_VALIDATION_EPOCH_CONTRACT_CORRECTION.md`). Current sha256 `3f6c4fbbc1e9f380ad0b550110dbc2e09be8b3c932c0b232652d6c378d1a3ffe`; the sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f` quoted below is the pre-correction value, retained as history. No other scientific choice changed and nothing had been trained.

> **RESOLVED 2026-09-20.** Every question raised below was answered by the owner and frozen in `configs/frozen/artifact_probe.yaml` (sha256 `e263b370797545c5c15ffdf0a1f28077c94fa815d643285be258f13fb4f4226f`). The decision record is `M5_ARTIFACT_PROBE_OWNER_DECISIONS.md` and the contract summary is `M5_ARTIFACT_PROBE_FROZEN_CONTRACT.md`. **The text below is preserved unchanged as the analysis that motivated those decisions.** M5 is still NOT_STARTED: nothing has been trained.

**Date:** 2026-09-20 · Read-only. Machine-readable: `M5_ARTIFACT_PROBE_PREFLIGHT.json`
**Decision id: D-M5-02 — OWNER_DECISION_REQUIRED**

## 1. The frozen wording

> spec §12.1: *"Loss: class-weighted cross-entropy with inverse-frequency weights clipped to
> [0.5, 3.0]."*

"Inverse-frequency" names no normalisation, and with a clip range of `[0.5, 3.0]` the normalisation
is not cosmetic — it decides whether the loss is weighted at all. A repository-wide search found no
existing definition in the spec, in any config, or in any code.

## 2. The literal reading is degenerate — shown explicitly

With `w_c = 1 / n_c`, the largest weight is `1/96 ≈ 0.0104` and the smallest is `1/5629 ≈ 0.00018`.
Every value is far below the lower clip bound, so **after clipping every class has weight exactly
0.5**:

| population | classes | all weights after clip | effect |
|---|---|---|---|
| A (spoof only, K=6) | makeup, mask_2d, mask_3d, partial, print, replay | all **0.5** | weighted CE ≡ 0.5 × unweighted CE |
| B (live+spoof, K=7) | + live | all **0.5** | weighted CE ≡ 0.5 × unweighted CE |

The clip range carries no information and the word "class-weighted" becomes vacuous. That cannot be
the intent, which is why this needs an owner decision rather than a literal implementation.

## 3. All candidates, population A (K = 6)

| class | n | `1/n` | `N/(K·n)` | mean-one | max-one | median-freq |
|---|---|---|---|---|---|---|
| makeup | 759 | **0.5** | 1.9407 | 0.5640 | **0.5** | 1.9545 |
| mask_2d | **96** | **0.5** | **3.0000** | **3.0000** | 1.0000 | **3.0000** |
| mask_3d | 1,056 | **0.5** | 1.3949 | **0.5** | **0.5** | 1.4048 |
| partial | 1,911 | **0.5** | 0.7708 | **0.5** | **0.5** | 0.7763 |
| print | 2,838 | **0.5** | 0.5190 | **0.5** | **0.5** | 0.5227 |
| replay | 2,178 | **0.5** | 0.6763 | **0.5** | **0.5** | 0.6811 |
| | | *degenerate* | 5 of 6 unclipped | 1 of 6 unclipped | 1 of 6 unclipped | 5 of 6 unclipped |
| max/min after clip | | **1.00** | **5.78** | 6.00 | 2.00 | 5.74 |

## 4. All candidates, population B (K = 7)

| class | n | `1/n` | `N/(K·n)` | mean-one | max-one | median-freq |
|---|---|---|---|---|---|---|
| live | 5,629 | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** |
| makeup | 759 | **0.5** | 2.7229 | 0.6498 | **0.5** | 2.5178 |
| mask_2d | **96** | **0.5** | **3.0000** | **3.0000** | 1.0000 | **3.0000** |
| mask_3d | 1,056 | **0.5** | 1.9571 | **0.5** | **0.5** | 1.8097 |
| partial | 1,911 | **0.5** | 1.0815 | **0.5** | **0.5** | 1.0000 |
| print | 2,838 | **0.5** | 0.7282 | **0.5** | **0.5** | 0.6734 |
| replay | 2,178 | **0.5** | 0.9489 | **0.5** | **0.5** | 0.8774 |
| | | *degenerate* | 5 of 7 unclipped | 1 of 7 unclipped | 1 of 7 unclipped | 5 of 7 unclipped |
| max/min after clip | | **1.00** | **6.00** | 6.00 | 2.00 | 6.00 |

Bold values are pinned by the clip.

## 5. Recommendation (for owner approval, not frozen here)

**`w_c = N / (K · n_c)`** — the "balanced" normalisation, identical to scikit-learn's
`class_weight='balanced'`:

1. it is the standard meaning of "inverse-frequency class weights";
2. it is the only candidate that is both non-degenerate **and** scale-free (independent of N and K);
3. it is the reading under which the frozen clip `[0.5, 3.0]` actually does what a clip is for —
   binding only the extreme minority (`mask_2d`) and the extreme majority, leaving the middle
   classes untouched. Under `mean_one` or `max_one` the clip swallows 5 or 6 of the classes, which
   makes the specified range look arbitrary.

`median_frequency` gives nearly identical numbers to `balanced` here, so the choice between those
two is low-impact; the choice against `raw_inverse` is decisive.

## 6. Constraints on whatever is chosen

- the weight tensor order **must** match the frozen class index order (lexical, once D-M5-01 fixes
  the class set);
- every weight must be finite, positive and inside `[0.5, 3.0]` after clipping — asserted at
  construction time;
- **no `WeightedRandomSampler`** in addition to weighted cross-entropy: that would compensate for
  imbalance twice;
- **no per-dataset weighting** — not specified, and it would interact with the dataset/class
  confound described in the data feasibility report.

## Resolution (2026-09-20)

**D-M5-02 resolved to `N/(K·n_c)` with the frozen clip.** Recomputed from the manifest over the 7-class population: `live` 0.5 (clipped low), `makeup` 2.722943722943723, `mask_2d` 3.0 (clipped high), `mask_3d` 1.957115800865801, `partial` 1.0814831427076326, `print` 0.7282291352058794, `replay` 0.9489046307228125. Computed in float64 over TRAIN only, no renormalisation after clipping. Exactly two classes are pinned by the clip. Full record: `M5_ARTIFACT_PROBE_CLASS_WEIGHTS.csv`.
