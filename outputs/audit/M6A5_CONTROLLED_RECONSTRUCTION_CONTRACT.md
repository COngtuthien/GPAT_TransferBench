# M6A5b — Controlled Reconstruction Contract (E04, E07c)

**Frozen 2026-09-21, BEFORE any training.** Authority: Amendment A3.
This document is the executable specification; A3 is the normative instrument.

**Contract class:** `BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION_UNDER_OWNER_AUTHORIZATION`.
The owner approved the **direction** (controlled reconstruction for E04 and E07c, reconstructed as
close to the papers and official code as reasonably possible). Every concrete technical value in this
document was **derived by the benchmark audit** after that authorization — the owner did not
independently specify each value, and **none of it was specified by the original E04 or E07c
authors**. Values tagged `PAPER` / `OFFICIAL` come from the publications or the official
repositories; everything tagged `BENCHMARK` is this benchmark's own derivation.

Every value below is frozen ex ante. Nothing here was selected by looking at downstream results, and
**TEST was never used for anything.**

---

# PART A — E04 Physics-STD

## A.1 Unchanged from the paper

| Item | Value | Origin |
|---|---|---|
| `L_depth` | **retained** (never removed) | PAPER |
| `α₁` | **100** (unchanged) | PAPER |
| `α₀…α₆`, `β` | unchanged | PAPER |
| Depth-loss form | `L_depth = (1/K²)·E_{i∼L∪S}‖M_i − M_i0‖_F` (L1) | PAPER Eq. 21 |
| `K` | **32** | PAPER |
| Spoof `M₀` | **all-zeros 32×32** | PAPER Eq. 21 |
| Live `M₀` | face-like shape, values in [0,1] | PAPER Eq. 21 + §3.2 |
| Training budget | 150,000 iterations, batch 8, lr 5e-5, /10 at 45,000 | PAPER |
| Checkpoint | terminal state @ iteration 150,000 | A2-06 `BASELINE_FINAL_STATE_V1` |

Because the spoof branch is fully paper-specified, the reconstruction affects **live samples only**.

## A.2 Geometry engine (pinned)

```
repository : https://github.com/cleardusk/3DDFA_V2
commit     : 1b6c67601abffc1e9f248b291708aef0e43b55ae   (2022-01-23, tree 28492fd3…)
code       : third_party/source_cache/tddfa_v2      (SOURCE ONLY - no model assets)
assets     : /media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/tddfa_v2/
             (outside the repo, untracked, never in source_cache)
```

| Asset used by the contract | Bytes | SHA256 |
|---|---|---|
| `configs/bfm_noneck_v3.pkl` | 24,393,598 | `89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca` |
| `configs/tri.pkl` | 913,040 | `0562a594d8369f7d1c86306522ff76fd3a2c81cf90ba9443893c1191ad0abbb8` |
| `configs/param_mean_std_62d_120x120.pkl` | 713 | `090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795` |
| `weights/mb1_120x120.pth` | 13,755,269 | `a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79` |

Inference settings: `arch = mobilenet_v1`, `widen_factor = 1.0`, input `120×120`,
`shape_dim = 40`, `exp_dim = 10`, `param_dim = 62`, de-normalized by the pinned
`param_mean_std`. Single deterministic forward pass — no sampling, no iterative fitting.
Model geometry: **38,365 vertices**, **76,073 triangles**, fixed topology for every image.
FaceBoxes is **not** used: the benchmark supplies pre-cropped canonical 256 faces.

**Licence — factual record, not a legal interpretation:** the 3DDFA_V2 **code** carries an MIT
licence file. Its `bfm/readme.md` **states academic-use terms** for the model asset, verbatim:
*"The modified BFM2009 face model in ../configs/bfm_noneck_v3.pkl is only for academic use. For
commercial use, you need to apply for the commercial license."* The external model bytes are **not
committed to, and not redistributed through, this repository**. Use of these assets **must comply with
their applicable upstream licence terms**. **No licence gate was bypassed during this task.** The
readme also requests the BFM citation: Paysan et al., AVSS 2009. This benchmark offers **no
independent legal interpretation** of those terms.

**Fixed auxiliary assets (not per-seed).** These artifacts are fixed auxiliary reconstruction assets,
analogous to a frozen preprocessing/geometry dependency. They are **never retrained, refitted or
re-derived per benchmark seed**: the identical pinned code commit, BFM-derived asset, 3DDFA regressor
checkpoint, Q = 140 list and depth renderer are reused unchanged for **every** E04 experiment seed
(`42`, `1337`, `2026`). Only the E04 **training** run consumes the benchmark experiment seed.

## A.3 Q = 140 vertex set (frozen)

```
canonical mesh : bfm_noneck_v3 mean shape, alpha_shp = alpha_exp = 0
                 V = u.reshape(3, -1, order='F').T        -> (38365, 3)
anchors        : vertex_id = keypoints.reshape(68,3)[:,0] // 3     (68 iBUG landmarks)
candidate mask : canonical (x,y) inside-or-on the 2D convex hull of the 68 anchors
                 (monotone-chain hull, pure numpy)        -> 27,233 candidates
seed           : all 68 anchors, canonical iBUG order 0..67
extension      : farthest-point sampling to 140, maximizing min EUCLIDEAN distance in
                 canonical 3D to the already-selected set
tie-break      : smallest vertex index
ordering       : [68 anchors in iBUG order] ++ [72 FPS picks in selection order]
```

| Property | Value |
|---|---|
| Frozen list | `outputs/audit/M6A5_E04_Q140_VERTEX_SET.json` |
| `vertex_indices` SHA256 | `1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f` |
| Reproducer | `tools/m6a5_e04_q140_derive.py` (re-run verified bit-identical) |
| Image-dependent? | **No** — computed once from the canonical neutral mesh |
| Min pairwise distance | 1825.1 canonical units |
| Face-region coverage | max distance to nearest selected 13,640.3; mean 6,261.3 |

Seeding with the 68 semantic landmarks is deliberate: the paper's sparse→dense warping (Eq. 14–16,
Fig. 6) computes offsets between **corresponding** vertices and Delaunay-triangulates them, so
semantic correspondence is the property that must survive. The 72 FPS picks supply the stated
face-region coverage. **No index was chosen by visual inspection.**

**Disclosed upstream quirk:** `keypoints` row 44 is `[37149, 37150, 36263]` — its z component indexes a
different vertex (12087) from its x/y (12383). The contract uses the consistent x-component rule.

## A.4 Depth target `M₀` (frozen)

For a **spoof** sample: `M₀ = zeros((32,32), float32)` — **PAPER**, no rendering.

For a **live** sample:

| Step | Rule | Origin |
|---|---|---|
| 1. Frame | canonical 256×256 face (frozen M2 geometry frame) | BENCHMARK |
| 2. Mesh | `pts3d = R @ (u + w_shp·α_shp + w_exp·α_exp).reshape(3,-1,order='F') + offset`, `dense_flag=True` | OFFICIAL |
| 3. Projection | official `similar_transform(pts3d, roi_box, size)` — x,y → image px; y flipped to image convention; z scaled by `s=(scale_x+scale_y)/2`; then `z -= z.min()` | OFFICIAL |
| 4. Coordinate system | right-handed image frame, origin top-left, +x right, +y **down** after the flip | OFFICIAL |
| 5. z convention | **relative** facial-surface depth, origin at the nearest vertex; larger = closer to camera. **Not** absolute camera distance | OFFICIAL |
| 6. Normalization | `z = (z − z.min()) / (z.max() − z.min())` over **all** mesh vertices, per sample → [0,1] | OFFICIAL (`utils/depth.py`) |
| 7. Visible surface | z-buffer, buffer init `−1e8`, **larger z wins** | OFFICIAL (`Sim3DR.rasterize`) |
| 8. Rasterization | official `Sim3DR.rasterize` at the pinned commit | OFFICIAL |
| 9. Background | **0** (`with_bg_flag=False` → `np.zeros_like`) | OFFICIAL (`utils/depth.py`) |
| 10. Face mask | the z-buffer "written" mask, stored alongside `M₀` | BENCHMARK |
| 11. Resize | 256 → 32 via `cv2.INTER_AREA` | BENCHMARK (reuses frozen `preprocess_v1` downscale rule for side ≥ 256) |
| 12. Clip | final clip to `[0,1]` | BENCHMARK (guards INTER_AREA rounding) |
| 13. Output | `float32`, shape `(32,32)`, range `[0,1]` | PAPER (`K=32`, [0,1]) |

**Only steps 1, 10, 11 and 12 are benchmark-defined**, and each reuses an existing frozen benchmark
convention. Steps 2–9 are official 3DDFA_V2 behaviour, so the depth *semantics* are authoritative
rather than invented here.

## A.5 Capability verification (performed, no benchmark data)

Verified on **synthetic** 3DMM parameters (`α_shp = α_exp = 0`, synthetic frontal pose):
dense mesh `(38365,3)` ✓ · projected vertices via official `similar_transform` ✓ ·
32×32 depth in `[0,1]` with background 0 and a clear face-like central ridge ✓.

Numpy reproduced the official Sim3DR z-buffer semantics because torch and the Sim3DR Cython extension
are absent from this audit environment. **The authoritative runtime renderer remains the official
Sim3DR at the pinned commit**, which M6B must build (`sh ./build.sh`).

---

# PART B — E07c DiffFAS-BIN-IDFREE

## B.1 Unchanged (pinned official source governs)

DiffFAS source pinned as recorded in `third_party/source_pins.json`. The conditioning encoder is
**not removed, not disabled, not replaced by an ImageNet/torchvision ResNet18**. A1/DEV-021 unpaired
code path, the frozen deterministic style-guide rule, and the DDIM sampler contract
(`DDIM_skip = 10`, `sample_initial_noise = 250` ⇒ 25 steps, `cond_scale = 2.0`) all stand.
`E07c-5` remains under `BASELINE_FINAL_STATE_V1`.

## B.2 Encoder architecture — preserved exactly

`models/custom_rn.py::resnet18` = `_resnet('resnet18', BasicBlock, [3,4,6,3])` — a **ResNet-34 layer
topology despite the name**. Statically verified at 256×256:

| Output | Stage | Shape | Consumed by `unet_autoenc`? |
|---|---|---|---|
| `x32x32` | after `layer2` | 32×32×**128** | **yes** |
| `x16x16` | after `layer3` | 16×16×**256** | **yes** |
| `x8x8` | after `layer4` | 8×8×**512** | **yes** |
| `embg` | `avgpool`+`fc` | (B, K) | **DISCARDED** |

`unet_autoenc.forward` does `x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)` and builds the
18-entry `cond` list. **Changing the class count therefore alters only the discarded head; the
consumed interface is shape-identical.** Checkpoint format preserved: `torch.save` of the **whole
`nn.Module`**, matching `torch.load(path).cuda()`.

## B.3 Substitute pretraining objective

| | |
|---|---|
| Original | 17-way attack classification on PADISI-USC (`ImageFolder`) |
| Unavailable | released `PADISI.pkl`; PADISI-USC dataset; the 17-class directory→label mapping |
| **Substitute** | **K = 7** attack-macro classification, frozen taxonomy, **TRAIN split only** |
| Class source | `manifests/artifact_probe_classes_v1.json` (index order as frozen) |
| Classes | `live, makeup, mask_2d, mask_3d, partial, print, replay` |
| Excluded | `other_spoof` — exactly as the frozen taxonomy excludes it (zero TRAIN, zero VAL) |

Verified TRAIN coverage — all 7 non-empty, total **14,467**:

| Class | live | print | replay | partial | mask_3d | makeup | mask_2d |
|---|---|---|---|---|---|---|---|
| TRAIN n | 5629 | 2838 | 2178 | 1911 | 1056 | 759 | 96 |

## B.4 Encoder training contract (frozen)

| Setting | Value | Origin |
|---|---|---|
| Architecture | `custom_rn.resnet18()` | OFFICIAL |
| Head | `fc = nn.Linear(512, 7)` | OFFICIAL form, benchmark `K` |
| Input resolution | `Resize((256,256))` | OFFICIAL |
| Preprocessing | `ToTensor` → `Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])` | OFFICIAL |
| Augmentation | **none** | OFFICIAL |
| Class balancing | **none** | OFFICIAL (adding it would be a benchmark invention) |
| Loss | `CrossEntropyLoss` on the 4th forward output | OFFICIAL |
| Optimizer | SGD `lr=0.002`, `momentum=0.9`, `weight_decay=5e-3` | OFFICIAL |
| Scheduler | **none** | OFFICIAL |
| Batch size | 256, `drop_last=True`, `shuffle=True` | OFFICIAL |
| Workers | 6 | OFFICIAL (implementation-adaptable, App. B) |
| Epochs | **200** | OFFICIAL |
| Checkpoint | **final state after epoch 200** | OFFICIAL (`torch.save` overwrites in-loop) + consistent with `BASELINE_FINAL_STATE_V1` |
| Save format | `torch.save(model)` — whole `nn.Module` | OFFICIAL |
| Data source | frozen canonical 256 TRAIN faces, **TRAIN only** | BENCHMARK |
| `auxiliary_encoder_training_seed` | **42** | BENCHMARK (official sets none) |
| `auxiliary_encoder_training_runs` | **1** | BENCHMARK |

**Benchmark-defined additions: exactly four** — label space, data source, auxiliary encoder seed,
single-run rule.

## B.5 Auxiliary encoder seed vs. E07c experiment seed, and the checkpoint rule

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

**Three different conditioning encoders MUST NOT be trained.**

| Term | Value | Meaning |
|---|---|---|
| `auxiliary_encoder_training_seed` | **42**, fixed | seeds the one-off pretraining of the substitute encoder |
| E07c experiment seed | **42 / 1337 / 2026** | seeds each DiffFAS main training run |

**Scientific rationale:** this most closely reproduces the role of the unavailable original
`PADISI.pkl`, which would have been **one fixed external pretrained asset shared across DiffFAS
runs**.

**No hidden model selection:** exactly one encoder under this ex-ante contract. Training several
candidates and keeping whichever yields the best DiffFAS result is **forbidden**. VAL may be used for
diagnostics only, never for selection. **TEST never selects anything.**

---

# PART C — Status

| Method | State | Fidelity class | Fidelity provenance |
|---|---|---|---|
| E01 | `READY_FOR_M6B` | `FAITHFUL_OFFICIAL_WITH_DETERMINISM_CLARIFICATION` | A2-01 deterministic asset enumeration |
| E02 | `READY_FOR_M6B` | `SPEC_DEFINED` | frozen spec §8.2 (+ A2-02/03/04) |
| E03 | `READY_FOR_M6B` | `FAITHFUL_OFFICIAL` | — |
| **E04** | `READY_FOR_M6B` | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A3_CONTROLLED_GEOMETRY_DEPTH_RECONSTRUCTION` |
| **E05** | `READY_FOR_M6B` | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A2_A2_07_ARCHITECTURE_RESOLUTION` |
| **E06c** | `READY_FOR_M6B` | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A1_IDFREE_ADAPTATION` |
| **E07c** | `READY_FOR_M6B` | **`CONTROLLED_ADAPTATION`** | `AMENDMENT_A1_IDFREE_ADAPTATION` + `AMENDMENT_A3_CONTROLLED_ENCODER_RECONSTRUCTION` |

**E01 is not a controlled scientific adaptation** (official operator behaviour preserved; only
host-dependent enumeration order replaced, for reproducibility). **E02 is not an external-method
adaptation** (defined by frozen spec §8.2). **E06c/E07c are not native or faithful** DSDG/DiffFAS.

Active M6 contract/source blockers: **0**. **Training is not complete** — nothing was trained, and no
`configs/methods/*.yaml` exists.

**E04 and E07c are comparable experimental baselines, but they are controlled reconstructions, not
exact reproductions of the original authors' unreleased pipelines.**
