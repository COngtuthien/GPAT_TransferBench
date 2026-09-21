# M6A5b — Owner Reconstruction Decisions (E04, E07c)

**Status:** `OWNER_APPROVED` · **Date:** 2026-09-21 · **Branch:** `m6-baselines`
**Starting commit:** `6e81f797e767d44f0d02c8910919fbc03133e0cc`

The owner changed direction: **both blocked baseline rows must become executable**, reconstructed as
closely as possible to the published papers and official code — **not** invented as arbitrary weak
substitutes, **not** deliberately handicapped, and **never** described as faithful reproductions.

| Blocker | Owner-approved direction | Technical contract | Status after M6A5b |
|---|---|---|---|
| `E04-2` | **OPTION B — `CONTROLLED_ADAPTATION`** | benchmark-defined under owner authorization | `CONTRACT_READY_CONTROLLED_ADAPTATION` |
| `E07c-4` | **OPTION B — `CONTROLLED_ADAPTATION`** | benchmark-defined under owner authorization | `CONTRACT_READY_CONTROLLED_ADAPTATION` |

### Authority vs. derivation

**Contract class:** `BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION_UNDER_OWNER_AUTHORIZATION`.

The owner approved the **direction** — controlled reconstruction for E04 and E07c, reconstructed as
close to the papers and official code as reasonably possible. The owner did **not** independently
specify each technical value. The following were **derived by the benchmark audit** after that
authorization: for **E04**, the selection of 3DDFA_V2, its exact commit and assets, the Q = 140 FPS
rule, the exact 140 indices and the depth-rendering details; for **E07c**, the K = 7 substitute
taxonomy, the use of GPAT TRAIN as encoder-training data, the auxiliary encoder seed, and the training
details recovered from `pretrain_classifier.py`.

**None of these details was specified by the original E04 or E07c authors.**

The M6A5a Option-A (`retain as N/A`) recommendation was **advisory only** and is **superseded**.

Frozen additively in **Amendment A3**
(`b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a`). The frozen DOCX, A1 and A2 are
unchanged.

---

## 1. E04 — what is preserved, what is substituted

**Preserved from the paper (not touched):** `L_depth` is **retained**; `α₁ = 100` and every published
coefficient stand; `K = 32`; the L1 form of Eq. 21; the additive/inpainting decomposition; the
150,000-iteration budget; `E04-3` under `BASELINE_FINAL_STATE_V1`.

**Decisive paper finding that narrowed the substitution scope** — Eq. 21, verbatim:

> "the depth ground truth **M₀ for a live face contains face-like shape and the depth for spoof should
> be zero**"

So the **spoof half of `M₀` is exactly paper-specified** (all-zeros 32×32) and required no
reconstruction whatsoever. The substitute governs **live faces only**.

### Substituted components

| Component | Original | Unavailable | Substitute |
|---|---|---|---|
| 3DMM fitting | `[61]` Dense Face Alignment over BFM | BFM registration-gated, version unidentifiable; `[61]` MATLAB test-only | **3DDFA_V2** @ `1b6c676…` with `bfm_noneck_v3.pkl` (BFM2009 derivative, academic-use grant) |
| Q = 140 vertices | paper's unpublished set | only the count is published | deterministic 68 anchors + 72 FPS picks over the canonical neutral mesh |
| Depth render | authors' renderer | never described | **official 3DDFA_V2 depth semantics** (z-buffer, per-sample min–max → [0,1], background 0) |

**Why 3DDFA_V2 and not something easier:** it performs the paper's exact function (fits a 3DMM with
shape + expression bases and emits a dense 38,365-vertex mesh), its model is a **BFM2009 derivative** —
the same family `[61]` requires — it ships its **own official depth renderer**, it carries its **own**
68 iBUG landmark correspondence, and it is deterministic.

**Licence — factual record, not a legal interpretation:** the 3DDFA_V2 **code** carries an MIT licence
file; its `bfm/readme.md` **states academic-use terms** for the model asset. The external model bytes
are **not committed to, and not redistributed through, this repository**; use of these assets **must
comply with their applicable upstream licence terms**; **no licence gate was bypassed during this
task**. This benchmark offers **no independent legal interpretation** of those terms.

**Fixed auxiliary assets:** the pinned commit, BFM-derived asset, 3DDFA checkpoint, Q = 140 list and
depth renderer are **fixed** reconstruction dependencies, reused unchanged for **every** E04
experiment seed (`42`, `1337`, `2026`). They are never retrained or re-derived per seed.

**Rejected on closeness grounds, not convenience:** MediaPipe Face Mesh (Apache-2.0 and fully public,
but a landmark regressor with **no shape/expression basis**, so it cannot perform the "fit a 3DMM"
role) and FLAME pipelines (registration-gated like BFM, different model family).

---

## 2. E07c — what is preserved, what is substituted

**The encoder architecture is NOT substituted.** The repository's own
`custom_rn.resnet18` — a `[3,4,6,3]` **ResNet-34 topology despite its name** — is used, *not*
`torchvision.resnet18`. The whole-module `torch.save` checkpoint format is preserved.

**Statically verified feature interface at 256×256:** `x32x32` = 32×32×128, `x16x16` = 16×16×256,
`x8x8` = 8×8×512, `embg` = (B, num_classes).

**The key containment fact:** `models/unet_autoenc.py` does
`x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)` — the fourth output is **discarded**.
Therefore changing 17 → 7 classes alters **only the discarded head**; the consumed feature interface is
**shape-identical**.

| Component | Original | Unavailable | Substitute |
|---|---|---|---|
| Pretraining objective | 17-way attack classification on PADISI-USC | released `PADISI.pkl`; the PADISI-USC dataset; the 17-class mapping | **K = 7** attack-macro classification over the frozen taxonomy, **TRAIN split only** |

Classes and verified TRAIN coverage (all non-empty, total **14,467**): live 5629, print 2838,
replay 2178, partial 1911, mask_3d 1056, makeup 759, mask_2d 96. This is the **nearest attack-category
objective** the frozen ontology supports and preserves the original **multiclass** conditioning intent
that binary live/spoof would destroy. No label is invented.

**Encoder training contract — official values used verbatim:** `Resize((256,256))` → `ToTensor` →
`Normalize([0.5]*3, [0.5]*3)`; no augmentation; no class balancing; `CrossEntropyLoss`; SGD
`lr = 0.002`, `momentum = 0.9`, `weight_decay = 5e-3`; no scheduler; batch 256, `drop_last`, `shuffle`;
**200 epochs**; final-state checkpoint (official `torch.save` overwrites in-loop, so the surviving file
*is* the final epoch's).

**Benchmark-defined additions are exactly four:** the K = 7 label space, the TRAIN data source, the
**auxiliary encoder seed** (official code sets no seed at all), and the **single-run** rule.

**Auxiliary encoder seed vs. E07c experiment seed — frozen.** The missing `PADISI.pkl` is an
**external pretrained conditioning asset**, not something retrained once per experiment seed:

```
auxiliary_encoder_training_seed = 42
auxiliary_encoder_training_runs = 1
```

The encoder is trained **exactly once**, checkpointed by the frozen final-state rule, SHA256-recorded,
frozen, and then **the same frozen checkpoint** is reused for **all** E07c main seeds — 42, 1337 and
2026. **Three different conditioning encoders must not be trained.** The DiffFAS main run seed still
follows the benchmark experiment seed set; only the auxiliary encoder seed is fixed at 42. Rationale:
this most closely reproduces the role of the unavailable original `PADISI.pkl`, which would have been
one fixed external pretrained asset shared across runs.

---

## 3. Fairness guardrails actually applied

- No substitute was chosen because it is easier, faster, expected to perform worse, or favourable to
  GPAT. MediaPipe would have been *easier* to install than 3DDFA_V2 and was rejected anyway, on
  closeness-to-paper grounds.
- Class balancing was **not** added to the encoder, because official code has none — adding it would
  have been a benchmark invention, and removing it is not a handicap but fidelity.
- **Exactly one** reconstruction contract per method, chosen ex ante. **Zero** candidate encoders
  trained. No multi-option downstream search. VAL never selects; **TEST is never used for anything.**

**Disclosed interaction (advantages the baseline, not GPAT):** the E07c substitute label space is also
the ArtifactProbeNet taxonomy behind the ArtSim metric. Separate models, no shared weights, no TEST —
so not leakage — but aligning E07c's conditioning to the ArtSim taxonomy could in principle favour
**E07c** on that metric. Recorded now so it cannot surface later as an undisclosed advantage.

---

## 4. Fidelity labels

| Method | Fidelity class | Fidelity provenance | Reporting label |
|---|---|---|---|
| E01 | `FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION` | A2-01 deterministic asset enumeration | FAS-Aug |
| E02 | `SPEC_DEFINED` | frozen spec §8.2 (+ A2-02/03/04 execution details) | Frequency Substitution |
| E03 | `FAITHFUL_OFFICIAL` | — | STDN |
| **E04** | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A3_CONTROLLED_GEOMETRY_DEPTH_RECONSTRUCTION` | Physics-STD (controlled geometry/depth reconstruction) |
| **E05** | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A2_A2_07_ARCHITECTURE_RESOLUTION` | PCGAN (controlled architecture resolution) |
| **E06c** | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A1_IDFREE_ADAPTATION` | DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation) |
| **E07c** | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A1_IDFREE_ADAPTATION` + `AMENDMENT_A3_CONTROLLED_ENCODER_RECONSTRUCTION` | DiffFAS-BIN-IDFREE (controlled encoder reconstruction) |

**E01 is not a controlled scientific adaptation** — the official transformation/operator behaviour is
preserved and only host-dependent `os.listdir` ordering was replaced, solely for reproducibility.
**E02 is not an adaptation of an external method** — it is defined directly by frozen spec §8.2, and
M6A3 merely completed spec-silent execution details. **E06c and E07c are not native or faithful** DSDG
/ DiffFAS: Amendment A1 explicitly adapts both to BIN-IDFREE semantics. Provenance strings such as
"A1 adaptation semantics" are **provenance, never a fidelity class**.

**E04 and E07c results are comparable experimental baselines, but they are controlled reconstructions,
not exact reproductions of the original authors' unreleased pipelines.** Forbidden wording: *native*,
*official reproduction*, *faithful reproduction*. These must never be collapsed into "native baseline".

---

## 5. Resulting state

| | Count |
|---|---|
| Active blockers before M6A5b | **2** |
| Resolved in M6A5b | **2** |
| **Active after M6A5b** | **0** |

All seven Track-A baseline rows (E01, E02, E03, E04, E05, E06c, E07c) are **contract-ready for M6B**.
**Training is not complete** — no baseline and no encoder was trained, and no `configs/methods/*.yaml`
was created.

## 6. Constraints observed

No baseline training. No encoder training. No synthetic-bank generation. No TEST access. No downstream
model selection. No method config created. No benchmark data processed. Frozen DOCX, A1 and A2
unchanged. Not committed, not pushed.
