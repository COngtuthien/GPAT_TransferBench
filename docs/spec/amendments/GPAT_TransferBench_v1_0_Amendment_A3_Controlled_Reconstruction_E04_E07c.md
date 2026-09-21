# GPAT-TransferBench v1.0 — Amendment A3

## Controlled Reconstruction of E04 (Physics-STD) and E07c (DiffFAS-BIN-IDFREE)

**Status:** `OWNER-APPROVED` (normative amendment) · **Type:** ADDITIVE, NON-DESTRUCTIVE
**Technical contract class:** `BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION_UNDER_OWNER_AUTHORIZATION`
**Date:** 2026-09-21 · **Milestone:** M6A5b · **Branch:** `m6-baselines`

### Authority vs. derivation — read this first

Two different things are recorded in this amendment and must not be conflated.

| | |
|---|---|
| **OWNER-APPROVED DIRECTION** | Controlled reconstruction for **E04** and **E07c** (both `E04-2` and `E07c-4` = OPTION B), plus authorization to reconstruct the missing pieces **as close to the papers and official code as reasonably possible**. |
| **BENCHMARK-DEFINED TECHNICAL CONTRACT** | Every concrete technical value below, derived by the benchmark audit **after** that authorization under the §3 hierarchy. |

The owner authorized the **direction**; the owner did **not** independently specify each technical
value. The following are **benchmark-defined choices**, not owner specifications:

- **E04** — the selection of 3DDFA_V2; its exact commit and assets; the Q = 140 farthest-point-sampling
  rule; the exact 140 vertex indices; the depth-rendering details.
- **E07c** — the K = 7 substitute taxonomy; the use of GPAT TRAIN as encoder-training data; the
  auxiliary encoder seed; and the training details recovered from `pretrain_classifier.py`.

Equally, **nothing below was specified by the original E04 or E07c authors.** Where a value comes from
the papers or the official repositories it is tagged `PAPER` / `OFFICIAL`; everything tagged
`BENCHMARK` is this benchmark's own derivation, and the original authors neither supplied nor endorsed
it.

This amendment **adds** execution contracts. It does **not** edit, weaken or supersede the frozen
specification, Amendment A1 or Amendment A2. Where A3 is silent, those documents govern.

| Item | SHA256 |
|---|---|
| Frozen specification | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| Amendment A1 | `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472` |
| Amendment A2 | `b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d` |
| M6A5b starting commit | `6e81f797e767d44f0d02c8910919fbc03133e0cc` |

---

## 1. Why faithful reproduction was impossible

**E04 — Physics-Guided STD.** M6A1–M6A4 established, from the **published** TPAMI 2022 text, the
official MSU CVLab project page, the author's GitHub account and the pinned upstream trees, that four
execution-critical ingredients were never released: the licence-gated Basel Face Model (version not
even identifiable from the paper, which never names it), the exact **Q = 140** vertex indices (only
the count and a qualitative rationale are published), the **depth-target rendering procedure** (the
published Algorithm 1 takes `M₀` as a pre-computed *input*), and any executable geometry route (no
PhySTD code release exists; the ECCV20-STDN repository contains no geometry code at all; the paper's
`[61]` is MATLAB/MatConvNet test-only *and* BFM-gated).

**E07c — DiffFAS-BIN-IDFREE.** The official conditioning encoder (`PADISI.pkl`) was publicly promised
by the repository **owner** on 2024-09-24 and never delivered; the repository has 0 releases and 0
tags, its last push *predates* the promise, and four follow-up requests through 2025-03-10 went
unanswered. The PADISI-USC dataset is itself licence-gated and is not benchmark data, and the
17-class `ImageFolder` label mapping is unpublished.

Under spec §8 both rows would otherwise be `BLOCKED_BY_SOURCE_GAP`.

## 2. Why the owner directed controlled reconstruction rather than N/A

The owner directed that **both rows must be executable**, so that the benchmark reports measured
comparisons rather than two empty rows, while stating plainly what was reconstructed. The instruction
was explicitly to reconstruct **as closely as possible** to the published papers and official code —
**not** to invent arbitrary weak substitutes, **not** to handicap baselines, and **not** to claim
faithful reproduction.

| Blocker | Owner-approved direction | Technical contract |
|---|---|---|
| `E04-2` | **OPTION B — `CONTROLLED_ADAPTATION`** | benchmark-defined under owner authorization (§4) |
| `E07c-4` | **OPTION B — `CONTROLLED_ADAPTATION`** | benchmark-defined under owner authorization (§5) |

The M6A5a packet's Option-A recommendation was **advisory only** and is superseded.

## 3. Reconstruction hierarchy (binding)

1. Preserve exact paper equations and losses whenever specified.
2. Preserve exact official architecture/code whenever available.
3. Preserve official hyperparameters whenever specified.
4. Preserve published training budget and optimizer behaviour.
5. For missing scientific components, choose the nearest reproducible substitute performing the
   **same function** in the published method.
6. Freeze every substitute and every newly introduced parameter **before** training.
7. Do **not** tune substitutes using TEST.
8. Do **not** choose substitutes based on which one makes GPAT look better.

Every non-original component is tagged `CONTROLLED_ADAPTATION` and records its original requirement,
the unavailable artifact, the substitute, the rationale, and the expected scientific difference.

---

## 4. E04 reconstruction contract

### 4.1 What is preserved from the paper (unchanged)

`L_depth` is **retained**; no published loss coefficient is altered. `α₁ = 100` and all of
`{α₀…α₆, β}` stand exactly as published. `K = 32`. The L1 form of Eq. 21 stands. The additive/
inpainting decomposition, the multi-scale discriminators and the training budget (150,000 iterations,
batch 8, lr 5e-5 with /10 at 45,000) are unchanged. `E04-3` remains governed by
`BASELINE_FINAL_STATE_V1` (A2-06): terminal state at iteration 150,000.

**Paper-derived and therefore NOT substituted — Eq. 21 verbatim:**

> "We follow the auxiliary FAS [18] to estimate an auxiliary depth map M, where the depth ground
> truth **M₀ for a live face contains face-like shape and the depth for spoof should be zero**."

This is decisive for scope: **the spoof half of `M₀` is exactly specified by the paper and requires no
reconstruction at all** (`M₀ = 0₍₃₂ₓ₃₂₎` for every spoof sample). The substitute governs **live faces
only**.

### 4.2 Geometry substitute — `CONTROLLED_ADAPTATION`

| | |
|---|---|
| **Original requirement** | 3DMM fitting via the paper's `[61]` Dense Face Alignment, over the Basel Face Model |
| **Unavailable artifact** | BFM (registration-gated, version unidentifiable); `[61]` is MATLAB/MatConvNet test-only |
| **Substitute** | **3DDFA_V2**, `cleardusk/3DDFA_V2` @ `1b6c67601abffc1e9f248b291708aef0e43b55ae` |
| **Model asset** | `configs/bfm_noneck_v3.pkl`, SHA256 `89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca` (24,393,598 B) |
| **Rationale** | It performs the paper's exact function — fits a 3DMM (40-d shape basis + 10-d expression basis + weak-perspective pose) and emits a dense 38,365-vertex mesh. Its model is a **BFM2009 derivative**, i.e. the same model family `[61]` requires, so the substitution stays inside the paper's 3DMM lineage. It ships its own official depth renderer, carries its own 68 iBUG landmark correspondence, is deterministic (single feed-forward pass, no sampling), and has a fixed vertex topology identical for every image. |
| **Licence (factual record, not a legal interpretation)** | The 3DDFA_V2 **code** carries an MIT licence file. Its `bfm/readme.md` **states academic-use terms** for `bfm_noneck_v3.pkl`, verbatim: *"The modified BFM2009 face model in ../configs/bfm_noneck_v3.pkl is only for academic use. For commercial use, you need to apply for the commercial license."* The external model bytes are **not committed to, and not redistributed through, this repository**. Use of these assets **must comply with their applicable upstream licence terms**. **No licence gate was bypassed during this task.** The readme also requests the BFM citation: Paysan et al., AVSS 2009. This benchmark offers **no independent legal interpretation** of those terms. |
| **Expected scientific difference** | The fitted shape will differ from the authors' `[61]` fit in detail: a different regressor, a BFM2009 *derivative* rather than the original BFM, and 40/10 basis dimensions. The **function** — a pose-aligned dense facial surface — is preserved. |

Rejected on closeness grounds, not convenience: **MediaPipe Face Mesh** (Apache-2.0, fully public, but
a landmark regressor with **no shape/expression basis**, so it cannot perform the paper's "fit a 3DMM"
role) and **FLAME-based pipelines** (registration-gated like BFM, and a different model family).

Model assets live **outside** the repository at
`/media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/tddfa_v2/`, are **not** git-tracked,
and are **not** stored in `source_cache`. Full provenance:
`outputs/audit/M6A5_TDDFA_ASSET_PROVENANCE.json`.

### 4.2b E04 assets are FIXED auxiliary reconstruction assets (not per-seed)

The selected 3DDFA_V2 artifacts are **fixed auxiliary reconstruction assets**, analogous to a frozen
preprocessing/geometry dependency. They are **never retrained, refitted or re-derived per benchmark
seed**. The **identical** pinned code commit, BFM-derived model asset, 3DDFA regressor checkpoint,
Q = 140 index list and depth renderer are reused unchanged for **every** E04 experiment seed
(`42`, `1337`, `2026`). Only the E04 **training** run consumes the benchmark experiment seed. No new
scientific choice is introduced by this clause; it states the intended semantics explicitly.

### 4.3 Q = 140 rule — `CONTROLLED_ADAPTATION`

**ORIGINAL:** the unpublished paper-specific Q = 140 vertex set.
**SUBSTITUTE:** a benchmark-defined **deterministic 140-vertex coverage set** over the canonical
neutral mesh. No index was ever chosen by visual inspection.

Frozen rule (derivable by `tools/m6a5_e04_q140_derive.py`):

| Element | Frozen value |
|---|---|
| Canonical mesh | `bfm_noneck_v3.pkl` mean shape, `α_shp = α_exp = 0`, vertices `= u.reshape(3, -1, order='F').T` → (38365, 3) |
| Semantic anchors | the model's own 68 iBUG landmark vertices, `vertex_id = keypoints.reshape(68,3)[:,0] // 3` |
| Candidate mask | mesh vertices whose canonical `(x, y)` lies inside or on the 2-D convex hull of the 68 anchors (monotone-chain hull, pure numpy) → **27,233** of 38,365 candidates |
| Seed | **all 68 anchors**, in canonical iBUG order 0…67 |
| Extension | farthest-point sampling to 140, maximizing the minimum **Euclidean distance in canonical 3-D** to the already-selected set |
| Tie-break | smallest vertex index |
| Ordering | 68 anchors in iBUG order, then 72 FPS picks in selection order |
| Image dependence | **none** — computed once from the canonical neutral mesh, identical for every image |
| Frozen list | `outputs/audit/M6A5_E04_Q140_VERTEX_SET.json` |
| `vertex_indices` SHA256 | `1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f` |

Seeding with the 68 semantic landmarks is deliberate: the paper's sparse→dense warping (Eq. 14–16,
Fig. 6) computes offsets between **corresponding** vertices and Delaunay-triangulates them, so
semantic correspondence is the property that must be preserved. The 72 FPS additions supply the
"cover the face region" coverage the paper states.

**Disclosed upstream quirk:** `keypoints` row 44 is `[37149, 37150, 36263]` — its z component indexes a
different vertex from its x/y. The contract therefore uses the consistent x-component rule above.

### 4.4 Depth target `M₀` — mixed paper-derived / benchmark-defined

| Element | Value | Origin |
|---|---|---|
| Spoof `M₀` | all-zeros 32×32 | **PAPER** (Eq. 21) |
| Live `M₀` | face-like shape, 32×32, values in [0,1] | **PAPER** (Eq. 21 + §3.2 output range) |
| `K` | 32 | **PAPER** |
| Working frame | canonical 256×256 face (frozen M2 geometry frame) | **BENCHMARK** (existing frozen contract) |
| Mesh | dense, `recon_vers(dense_flag=True)`: `pts3d = R @ (u + w_shp·α_shp + w_exp·α_exp).reshape(3,-1,order='F') + offset` | **OFFICIAL 3DDFA_V2** |
| Projection / coords | official `similar_transform(pts3d, roi_box, size)`: x,y in image pixels, y-axis flipped to image convention, z scaled by `s = (scale_x+scale_y)/2` then shifted so `min(z) = 0` | **OFFICIAL 3DDFA_V2** |
| z convention | **relative** facial-surface depth, origin at the nearest vertex — **not** absolute camera distance | **OFFICIAL**, and matches the amendment's stated principle |
| Visible-surface | z-buffer, buffer initialized to `-1e8`, larger z wins | **OFFICIAL** (`Sim3DR.rasterize`) |
| Rasterization | official `Sim3DR.rasterize` at the pinned commit | **OFFICIAL** |
| Normalization | per-sample min–max over all mesh vertices: `z = (z - z.min()) / (z.max() - z.min())` → [0,1] | **OFFICIAL** (`utils/depth.py`) |
| Background | `0` (`with_bg_flag=False` → `np.zeros_like`) | **OFFICIAL** (`utils/depth.py`) |
| Face mask | the z-buffer's "written" mask, stored alongside `M₀` | **BENCHMARK** |
| 256 → 32 resize | `cv2.INTER_AREA` | **BENCHMARK** — reuses the already-frozen `preprocess_v1` downscale rule for side ≥ 256 |
| Clipping | final clip to `[0,1]` | **BENCHMARK** (guards INTER_AREA rounding) |

The depth **semantics are taken from the official 3DDFA_V2 routine**, not invented here: per-sample
min–max normalization to [0,1] with background 0 is exactly `utils/depth.py`. Only the working frame,
the stored mask, the resize rule and the final clip are benchmark-defined, and each reuses an existing
frozen benchmark convention.

### 4.5 E04 status and label

`CONTRACT_READY_CONTROLLED_ADAPTATION` · fidelity **`CONTROLLED_ADAPTATION`**.
Reporting label: **Physics-STD (controlled geometry/depth reconstruction)**.

---

## 5. E07c reconstruction contract

### 5.1 What is preserved (unchanged)

The pinned official DiffFAS source governs the architecture and every executable setting. The
conditioning encoder is **not** removed, **not** disabled, and **not** replaced by a torchvision or
ImageNet ResNet18. The A1/DEV-021 unpaired code path, the frozen deterministic style-guide rule, the
DDIM sampler contract (`DDIM_skip = 10`, `sample_initial_noise = 250` ⇒ 25 steps, `cond_scale = 2.0`)
and `E07c-5` under `BASELINE_FINAL_STATE_V1` all stand.

### 5.2 Encoder architecture — preserved exactly, NOT substituted

The substitute encoder uses the repository's **own** `models/custom_rn.py::resnet18`, which is
`_resnet('resnet18', BasicBlock, [3,4,6,3])` — a **ResNet-34 layer topology despite the name**. It is
**not** replaced by `torchvision.models.resnet18`. Statically verified feature interface at 256×256:

| Output | Stage | Shape |
|---|---|---|
| `x32x32` | after `layer2` | 32×32×**128** |
| `x16x16` | after `layer3` | 16×16×**256** |
| `x8x8` | after `layer4` | 8×8×**512** |
| `embg` | after `avgpool` + `fc` | (B, num_classes) |

`models/unet_autoenc.py` consumes `x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)` — the
fourth output **is discarded**. Therefore **changing the class count alters only the discarded head;
the consumed feature interface is shape-identical.** This is what makes the substitution well
contained.

Checkpoint format is preserved: `torch.save(model)` of the **whole `nn.Module`**, matching
`unet_autoenc.encoder()`'s `torch.load(path)` followed by `.cuda()`.

### 5.3 Pretraining objective substitute — `CONTROLLED_ADAPTATION`

| | |
|---|---|
| **Original requirement** | 17-way attack classification on PADISI-USC (`ImageFolder` over `pasidi_attack/`) |
| **Unavailable artifact** | the released `PADISI.pkl` weights; the PADISI-USC dataset (licence-gated, not benchmark data); the 17-class directory→label mapping (unpublished) |
| **Substitute** | **K = 7** attack-macro classification over the **frozen benchmark taxonomy**, using the **frozen TRAIN split only** |
| **Classes** | `live, makeup, mask_2d, mask_3d, partial, print, replay` — `manifests/artifact_probe_classes_v1.json`, index order as frozen there |
| **Rationale** | This is the **nearest attack-category classification supported by the frozen ontology** (hierarchy tier B, and arguably tier A: the official objective *was* attack-category classification). It preserves the original **multiclass** conditioning intent, which binary live/spoof would destroy. The mapping is deterministic from frozen metadata — no label is invented. |
| **TRAIN coverage** | all 7 classes non-empty: live 5629, print 2838, replay 2178, partial 1911, mask_3d 1056, makeup 759, mask_2d 96 — total **14,467** |
| **Expected scientific difference** | A different image domain (CASIA/MSU/SiW-Mv2 rather than PADISI-USC) and 7 rather than 17 categories, so the learned conditioning space differs from the authors'. Class imbalance is more pronounced than a balanced PADISI folder split would be. The **function** — an attack-category-discriminative multi-scale feature extractor — is preserved. |

`other_spoof` is excluded exactly as the frozen taxonomy excludes it (zero TRAIN and zero VAL
observations).

**Disclosed interaction:** this taxonomy is also the ArtifactProbeNet label space used for the ArtSim
metric. The encoder and the probe are separate models sharing no weights, and no TEST data is
involved, so this is not leakage — but aligning E07c's conditioning to the ArtSim taxonomy could in
principle favour **E07c** on that metric. It is disclosed here because it advantages the baseline, not
GPAT, and must not be discovered later.

### 5.4 Encoder training contract

Recovered from official `models/pretrain_classifier.py` — used verbatim wherever executable:

| Setting | Value | Origin |
|---|---|---|
| Architecture | `custom_rn.resnet18()` | **OFFICIAL** |
| Head | `fc = nn.Linear(512, K)`, `K = 7` | OFFICIAL form, **benchmark K** |
| Input resolution | `Resize((256, 256))` | **OFFICIAL** (and identical to the canonical face) |
| Preprocessing | `ToTensor` → `Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])` | **OFFICIAL** |
| Augmentation | **none** | **OFFICIAL** (official transform list has none) |
| Class balancing | **none** | **OFFICIAL** (plain `ImageFolder` + `shuffle`; adding balancing would be a benchmark invention) |
| Loss | `CrossEntropyLoss` on the 4th forward output | **OFFICIAL** |
| Optimizer | SGD, `lr = 0.002`, `momentum = 0.9`, `weight_decay = 5e-3` | **OFFICIAL** |
| Scheduler | **none** | **OFFICIAL** (no scheduler in official code) |
| Batch size | 256, `drop_last = True`, `shuffle = True` | **OFFICIAL** |
| Dataloader workers | 6 | OFFICIAL (implementation-adaptable per Appendix B) |
| Epochs | **200** | **OFFICIAL** |
| Checkpoint rule | **final training state after epoch 200** | **OFFICIAL** — `torch.save` runs inside the epoch loop and overwrites the same path, so the surviving file *is* the final epoch's. Consistent with `BASELINE_FINAL_STATE_V1`. |
| Data source | frozen canonical 256 TRAIN faces, TRAIN split only | **BENCHMARK** |
| `auxiliary_encoder_training_seed` | **42** | **BENCHMARK** — official code sets no seed at all |
| `auxiliary_encoder_training_runs` | **1** | **BENCHMARK** |

**Benchmark-defined additions are exactly four:** the K = 7 label space, the TRAIN data source, the
auxiliary encoder seed, and the single-run rule. Everything else is official.

#### 5.4b Auxiliary encoder seed vs. E07c experiment seed — frozen

The missing `PADISI.pkl` is an **external pretrained conditioning asset**, not something the official
pipeline retrains once per experiment. The reconstruction preserves that role:

```
auxiliary_encoder_training_seed = 42
auxiliary_encoder_training_runs = 1
```

The controlled encoder is trained **exactly once** on the frozen GPAT TRAIN split. After training:
select the checkpoint by the already-frozen **final-state** rule; compute and record its **SHA256**;
freeze it; then reuse **that same frozen checkpoint** for **all** E07c main-training seeds.

| E07c experiment seed | Conditioning encoder | DiffFAS training seed |
|---|---|---|
| 42 | frozen encoder **X** | 42 |
| 1337 | **the same** frozen encoder **X** | 1337 |
| 2026 | **the same** frozen encoder **X** | 2026 |

**Three different conditioning encoders MUST NOT be trained.** The main E07c/DiffFAS run seed still
follows the benchmark experiment seed `{42, 1337, 2026}`; only the **auxiliary** encoder seed is fixed
at 42.

**These two seeds are different things and must never be conflated:**

| Term | Value | Meaning |
|---|---|---|
| `auxiliary_encoder_training_seed` | **42**, fixed | seeds the one-off pretraining of the substitute conditioning encoder |
| E07c experiment seed | **42 / 1337 / 2026** | seeds each DiffFAS main training run, per the frozen benchmark seed set |

**Scientific rationale:** this most closely reproduces the role of the unavailable original
`PADISI.pkl`, which would have been **one fixed external pretrained asset shared across DiffFAS
runs**. Its SHA256 is recorded before DiffFAS consumes it.

**No hidden model selection.** Exactly one encoder is trained under this ex-ante contract. Training
several candidates and keeping whichever yields the best DiffFAS result is **forbidden** — that would
be an undisclosed search. VAL may be used only for diagnostics, never to select the encoder. **TEST is
never used**, for anything.

### 5.5 E07c status and label

`CONTRACT_READY_CONTROLLED_ADAPTATION` · fidelity **`CONTROLLED_ADAPTATION`**.
Reporting label: **DiffFAS-BIN-IDFREE (controlled encoder reconstruction)**.

---

## 6. Fidelity labels and required disclosure

The normalized Track-A fidelity taxonomy is:

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

**E04 and E07c results are comparable experimental baselines, but they are controlled
reconstructions, not exact reproductions of the original authors' unreleased pipelines.**

Forbidden wording for these rows: *native*, *official reproduction*, *faithful reproduction*.
Allowed wording: *controlled reconstruction*, *controlled adaptation*, *paper-guided reconstruction*.

Result tables must distinguish **source-faithful executable baseline** from **controlled
reconstruction**; these must never be collapsed into "native baseline".

## 7. Standing prohibitions

TEST is never used for training, tuning, checkpoint selection, threshold selection, substitute
selection or any other decision. No substitute was chosen because it is easier, faster, expected to
perform worse, or favourable to GPAT. No multi-option downstream benchmark search was performed: one
reconstruction contract was chosen ex ante for each method. Every substitute and every newly
introduced parameter is frozen in this amendment **before** any training.

## 8. Scope

The frozen DOCX, Amendment A1 and Amendment A2 are **not** edited. No `configs/methods/*.yaml` is
created by this amendment. No training, synthetic-bank generation, TEST access or model inference is
authorized or performed by it.
