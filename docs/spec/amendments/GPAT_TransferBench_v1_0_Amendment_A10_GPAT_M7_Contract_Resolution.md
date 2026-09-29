# GPAT-TransferBench v1.0 — Amendment A10
## GPAT M7 Contract Resolution

| | |
|---|---|
| **Amendment** | A10 |
| **Title** | GPAT M7 Contract Resolution |
| **Milestone** | M7A (contract resolution only; no GPAT code, no config for B1/B2/B3, no training) |
| **Authority commit** | `8d4ddb3c55e8809398346dbdc10d15318e14b60d` (M6E: close M6 baseline implementation milestone) |
| **Amends** | `docs/spec/GPAT_TransferBench_v1_0_Frozen_Specification_2026.docx` (sha256 `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`), additively |
| **Machine-readable record** | [`configs/amendments/gpat_a10_m7_contract_resolution.yaml`](../../../configs/amendments/gpat_a10_m7_contract_resolution.yaml) |
| **Decision date** | 2026-09-29 |
| **Decision timing** | Before any GPAT code, GPAT environment, GPAT training, GPAT checkpoint, GPAT bank or GPAT result exists. No TEST row has ever been used for any GPAT decision. |
| **Status** | Owner decisions D01–D17 as issued in the M7A instruction; additive, non-destructive, prospective. The candidate becomes authority only when the owner commits it. |

## 1. Why this amendment exists

The M7 entry audit (read-only, 2026-09-29) found that GPAT E08–E11 could not start because the frozen specification is
internally inconsistent or silent on items that change the trained method: the `lambda_dir` key (DEV-003), the identity
adversary sign (Q-05), the parser class semantics (Q-21), the dataset scope of B1/B2/B3 under Amendment A1, the
attack-type loss and warmup, the differentiable path through the frozen teachers, and about two dozen loss and
architecture operators. The owner resolved all of them. This amendment records those decisions, classifies each one, and
binds them to the frozen authority. It changes nothing that already exists.

## 2. Authority chain

| Level | Authority | Role |
|---|---|---|
| 1 | Frozen specification DOCX; adopted amendments A1–A9; M6E closure (`outputs/audit/M6E_FINAL_M6_CLOSURE.*`, `outputs/audit/method_status.csv`); the owner decisions in the M7A instruction | highest; never overridden |
| 2 | This amendment (A10) and its resolution YAML | encodes the Level-1 owner decisions; subordinate to the frozen spec and A1–A9 |
| 3 | Historical project `/home/cong/PRISM_FAS_C_LLM_Project` (branch `gpu-work/e7-gpat-bank-prep`, HEAD `6f0642a1d05c35e4c1778d329fb547f1b22a4115`) | **HISTORICAL_IMPLEMENTATION_REFERENCE_ONLY — not authority**. Not part of the authority chain and not a source of scientific hyperparameters, dataset scope or teacher choices. It can supply an implementation detail only where Level 1–2 is silent, it adds no science, and its provenance is recorded. No historical file is copied in M7A |

Where the historical project conflicts with Level 1–2, Level 1–2 wins and the conflict is recorded (§9).

## 3. Decision classes

| Class | Meaning |
|---|---|
| FROZEN_SPEC_FACT | stated verbatim in the frozen spec; restated for binding only |
| OWNER_SCIENTIFIC_CLARIFICATION | the spec is silent or inconsistent and the owner fixed a value that affects the trained method, without leaving the spec's intent |
| IMPLEMENTATION_CLARIFICATION | operator wiring, bookkeeping or arithmetic with no alternative scientific meaning |
| RUNTIME_COMPATIBILITY | how the frozen method is executed in this runtime; the method is unchanged |
| CONTROLLED_ADAPTATION | a departure from the spec's literal data/supervision contract; carries a DEV id |
| EXECUTION_STORAGE_POLICY | what is saved and when; selection semantics unchanged |
| HISTORICAL_IMPLEMENTATION_REFERENCE | a Level-3 source cited for an implementation detail only |

## 4. Owner decisions D01–D17

### D01 — `lambda_dir` (closes DEV-003) · OWNER_SCIENTIFIC_CLARIFICATION
`lambda_dir = 0.5` (spec §23.2) **is the coefficient of the S_orient term inside L_spec**:
`L_spec = ||S_radial(x_hat) − S_radial(x_s)||_1 + lambda_dir · ||S_orient(x_hat) − S_orient(x_s)||_1`.
It is not a standalone directional loss and not an inert key. With the frozen value 0.5 this is numerically identical to
the §10.2 formula. DEV-003 is resolved.

### D02 — identity adversary sign (closes Q-05) · OWNER_SCIENTIFIC_CLARIFICATION
`GRL(alpha = 1.0)` is kept. The objective contains **`+ lambda_idadv · CE(identity_head(GRL(z_a)), source_subject)`**.
The identity head minimizes CE; the GRL performs the single intended reversal on the encoder path, so E_art maximizes
the identity CE. `GRL + (− lambda_idadv · CE)` is forbidden: it reverses twice and makes the encoder *minimize* the
identity CE. The "−" in the §10.3 `L_G` line is read as naming the adversarial role, not as an arithmetic sign on top of
the GRL. `lambda_idadv = 0.1` for B2/B3 and 0.0 for B0/B1 (frozen §5.3).

### D03 — B2/B3 dataset scope and identity label · CONTROLLED_ADAPTATION (DEV-022)
- Datasets: **CASIA-FASD + MSU-MFSD + SiW-Mv2** (the full common TRAIN pair population, 8,838 pairs).
- Identity label: **`source_subject`** of the pair (E_art only sees x_s, spec §9.1; H4 names source-identity leakage).
- CASIA/MSU rows: identity CE active (35 + 25 = 60 source subjects).
- SiW-Mv2 rows (no subject id): identity CE **masked**.
- Normalization: the identity CE of an optimizer step is the sum of per-row CE over the **labelled rows contributing to
  that optimizer step**, divided by that labelled-row count. If an optimizer step contains zero labelled rows,
  `L_idadv = 0` exactly.
- Never fabricated: pseudo ids, clustering ids, `video_id` or `content_group_id` as identity, AdaFace pseudo-identity,
  filename-derived identity, DEV-013 as a same-person guarantee (A1 §4, fair_track_v1 `identity_policy`).
- This is the only controlled dataset/supervision adaptation introduced by A10. It is registered as **DEV-022**.

### D04 — B1 dataset scope · OWNER_SCIENTIFIC_CLARIFICATION
B1 uses CASIA-FASD + MSU-MFSD + SiW-Mv2. SiW is instantiable for B1 because B1 needs `attack_macro`, which every SiW
TRAIN row has, and does not need identity.

### D05 — attack-type supervision (B1/B3) · OWNER_SCIENTIFIC_CLARIFICATION
- Input: `z_a` = global E_art(x_s) feature (L2-normalized GAP(layer4), 512-D, spec §9.2).
- Head: `Linear(512, 6)`, `bias=True`.
- Class order, exactly: `0 makeup, 1 mask_2d, 2 mask_3d, 3 partial, 4 print, 5 replay`. No live class.
- Samples: spoof sources only.
- Loss: unweighted CrossEntropy, mean reduction; `lambda_type = 0.2` (frozen §5.3).
- L_type is **not** applied to x_hat.
- The warmup head (D06) is the head used during generator training.

### D06 — B1/B3 E_art warmup · OWNER_SCIENTIFIC_CLARIFICATION
10 epochs before the 60 generator epochs (not counted among them). Data: TRAIN spoof-source frames only, the same six
classes, the same 12-channel E_art input. AdamW, lr `1e-4`, weight_decay `1e-4`, cosine per optimizer step to 0, batch 64,
`drop_last = false` (tail kept: 8,838 = 138 × 64 + 6 → 139 steps per epoch, 1,390 steps in total). E_art weights and the
attack-head weights carry into the generator stage; optimizer state does not (the generator stage starts a fresh Adam).
Same experiment seed. No VAL selection in the warmup: the state after warmup epoch 10 is used.

### D07 — differentiable frozen-teacher path · RUNTIME_COMPATIBILITY
A differentiable **exact-linear compatibility adapter** replaces only the non-differentiable resampling of the frozen
preprocessing (FaceXFormer: PIL BICUBIC 256→224; AdaFace: cv2 INTER_AREA 256→112). Both frozen resamplers are linear before
uint8 rounding, so they are reproduced as fixed resampling matrices. Preserved unchanged: channel order (AdaFace BGR),
normalization constants, teacher weights, teacher code, eval mode, `requires_grad = False` on teacher parameters. The
x_hat path stays differentiable; x_t goes through the same adapter and is detached. Teachers compute in fp32. A parity test
against the frozen PIL/cv2 path is required before training (M7C); its tolerance is frozen at that qualification from the
measured rounding bound, not here.

### D08 — landmark loss · OWNER_SCIENTIFIC_CLARIFICATION
The pinned FaceXFormer regresses 68 × 2 coordinates (`landmarks_prediction_head = MLP(…, 136, 3)`); it exposes no heatmap.
`H_k` in spec §10.1 is read as landmark k's native output. `L_lm = mean L1(landmarks(x_hat), landmarks(x_t))` in
FaceXFormer's native normalized coordinate space. No fabricated heatmaps; no pixel conversion in the training loss.

### D09 — parser semantics (closes Q-21 for M7) · OWNER_SCIENTIFIC_CLARIFICATION
Class 0 = background; classes 1..10 = non-background. No semantic names are assigned to 1..10. No void/ignore class.
- VAL Dice: argmax masks; classes 1..10; a class is omitted for a sample only when absent from both masks; mean over the
  valid classes; then mean over VAL samples.
- L_parse: soft Dice over classes 1..10 plus `0.1 · KL` (frozen §10.1) over all 11 classes.

### D10 — underspecified GPAT operators

| # | Item | Owner-frozen definition | Class |
|---|---|---|---|
| 1 | S_radial | per RGB channel; power `|FFT|²`; `log(1 + power)`; integer-radius bins through Nyquist (128 bins at 256×256); each radial spectrum normalized to sum 1; loss = mean L1 over channels/bins | OWNER_SCIENTIFIC_CLARIFICATION |
| 2 | S_orient | FFT power orientation histogram; 8 angular bins over [0, π); DC excluded; normalized-frequency radius r ∈ (0, 0.5]; histogram normalized to sum 1; loss = mean L1 | OWNER_SCIENTIFIC_CLARIFICATION |
| 3 | TV | anisotropic L1 total variation, mean reduction | OWNER_SCIENTIFIC_CLARIFICATION |
| 4 | Normalize(·) in §9.4 | no per-image min–max; `A = Normalize(u)` with `u = M ⊙ (0.5·A_rgb + 0.5·A_freq)` and `Normalize(u) = clip(u / 2.0, 0.0, 1.0)`; the denominator applies to the whole map `u`, never to `A_rgb` alone (§5) | OWNER_SCIENTIFIC_CLARIFICATION |
| 5 | A_freq resolution | channel mean, then bilinear ×2 to 256×256, `align_corners = false` | IMPLEMENTATION_CLARIFICATION |
| 6 | L_budget | per-image hinge, then batch mean | OWNER_SCIENTIFIC_CLARIFICATION |
| 7 | M_face_dilated | foreground = parser classes 1..10 of x_t; nearest resize 224 → 256; binary dilation with a 15×15 square kernel | OWNER_SCIENTIFIC_CLARIFICATION |
| 8 | FiLM | after each stage's NAFBlocks; `Linear(512, 2C)`; `(1 + gamma) · h + beta` | OWNER_SCIENTIFIC_CLARIFICATION |
| 9 | Downsample | NAFNet `Conv2d(C, 2C, kernel_size=2, stride=2)` | IMPLEMENTATION_CLARIFICATION |
| 10 | Skip | element-wise addition | IMPLEMENTATION_CLARIFICATION |
| 11 | Bottleneck | `1×1 conv 256 → 512`, then adaptive average pool to 8×8 | IMPLEMENTATION_CLARIFICATION |
| 12 | Band order | band-major `[LL_rgb, LH_rgb, HL_rgb, HH_rgb]`; output channels 0..2 = ΔLL, 3..11 = ΔHF, 12 = mask | IMPLEMENTATION_CLARIFICATION |
| 13 | PatchGAN | pix2pix style; LeakyReLU(0.2) after every layer except the final; InstanceNorm on layers 2..4; padding 1; canonical output 30×30 | OWNER_SCIENTIFIC_CLARIFICATION |
| 14 | D real sample | x_s of the current TRAIN batch | OWNER_SCIENTIFIC_CLARIFICATION |
| 15 | D:G updates | 1:1 per optimizer update, same accumulation boundary | IMPLEMENTATION_CLARIFICATION |
| 16 | Source-band upsampling | bilinear ×2, `align_corners = false` | IMPLEMENTATION_CLARIFICATION |
| 17 | High-pass | frozen Q-07 operator: Gaussian 9×9, σ 1.5, reflect-101 (torch `reflect` padding) | OWNER_SCIENTIFIC_CLARIFICATION |
| 18 | F_art | frozen ImageNet ResNet-18 on HP(x) at 256×256; feature = GAP(layer4), 512-D | OWNER_SCIENTIFIC_CLARIFICATION |
| 19 | Generic L1 reductions | mean over elements unless a frozen formula states otherwise | OWNER_SCIENTIFIC_CLARIFICATION |
| 20 | x_hat range | no clamp during training; clamp only at the output/export conversion boundary | OWNER_SCIENTIFIC_CLARIFICATION |
| 21 | EMA scope | E_art + G_res; no EMA for the frozen teachers or the discriminator | IMPLEMENTATION_CLARIFICATION |
| 22 | Generator LR schedule | per optimizer step; linear warmup from 0; cosine decay to 2e-6 at the final optimizer step | IMPLEMENTATION_CLARIFICATION |
| 23 | Curriculum | interpolated per optimizer step on the frozen stage boundaries; `delta_scale_hf` is the runtime residual scale | IMPLEMENTATION_CLARIFICATION |
| 24 | E_art initialization | exactly as frozen in §9.2 (conv1 channels 0:3 = pretrained; each extra channel = mean(pretrained conv1 over RGB)/3) | FROZEN_SPEC_FACT |

### D11 — DWT/IDWT sanity · IMPLEMENTATION_CLARIFICATION
Production path: ptwt, Haar, level 1, `boundary_mode = reflect`, per RGB channel, fp32. Mandatory input: torch
uniform[-1, 1], shape `[100, 3, 256, 256]`, CPU `torch.Generator` seed 42. Metric: maximum absolute reconstruction error
over all values. PASS iff error `< 1e-5`. No dataset access.

### D12 — `artifact_scale = 0` · IMPLEMENTATION_CLARIFICATION
`artifact_scale = 0` disables **all** residual contributions (HF delta and LL delta), so `x_hat = IDWT(DWT(x_t))`.
Required: `max_abs(x_hat − x_t) < 1e-5`, and for the same x_t with two different x_s: `max_abs(x_hat_1 − x_hat_2) < 1e-5`.

### D13 — VAL ArtSim · OWNER_SCIENTIFIC_CLARIFICATION
VAL ArtSim = cosine similarity of the **M5 ArtifactProbe v1** embeddings (spec §13 "Artifact embedding cosine"), not GPAT's
F_art. Bound checkpoint: `models/artifact_probe/artifact_probe_v1.pt`, sha256
`b5ace6c263d8473215ff2ab825b98541bdcf332251512216cbd93546f32e54ff` (authority record
`outputs/audit/artifact_probe_v1.sha256`; not tracked in git; GPU host).

### D14 — checkpoint selection · IMPLEMENTATION_CLARIFICATION (+ NME normalizer: OWNER_SCIENTIFIC_CLARIFICATION)
Candidates: EMA at epochs 10..60 inclusive, N = 51. Higher is better: ID, Dice, ArtSim; lower is better: NME, LFErr
(ranked as −NME, −LFErr). Average rank for ties; `p = (rank − 1)/(N − 1)`. Frozen weights:
`SelectionScore = 0.25·p(ID) + 0.15·p(Dice) + 0.25·p(ArtSim) + 0.20·p(−NME) + 0.15·p(−LFErr)`. Tie-break: higher raw
ArtSim, then earlier epoch. NME normalizer: bounding box of the target's 68 landmarks. VAL only; TEST never participates.

### D15 — epoch and tail · IMPLEMENTATION_CLARIFICATION
One epoch = one deterministic shuffled pass over all 8,838 TRAIN pairs; physical batch 4; gradient accumulation 2;
`drop_last = false`; 2,210 physical microbatches; **1,105 optimizer updates**; the final accumulation group holds 6 actual
samples. Losses are sample-weighted by the actual samples in the optimizer step. The tail is never discarded. Totals:
66,300 generator updates per run; 5-epoch LR warmup = 5,525 updates; EMA starts at the end of epoch 5 (update 5,525).

### D16 — EMA candidates and VAL · EXECUTION_STORAGE_POLICY
The training process never reads VAL. All 51 EMA candidates of a run are saved. After training, a separate deterministic
VAL-selection process evaluates **all 51** before selecting, and records for every candidate: epoch, checkpoint SHA-256,
the five raw metrics, the ranks, the percentile values and SelectionScore. The selected checkpoint is protected.
Non-selected candidates may be pruned only later, only after all 51 scores exist, selection is final and hashes/index are
verified, and only under an explicit logged retention policy. Selection metadata stays permanently auditable.
(Rationale: aggregated percentile ranks are not independent of irrelevant alternatives, so no candidate can be discarded
before all 51 are scored.)

### D17 — environment · RUNTIME_COMPATIBILITY (deferred to M7C; nothing installed in M7A)
Clone `gpat-m5` / `m6_core_gpu` offline, add only GPAT dependencies (ptwt, PyWavelets, the hash-pinned NAFNet architecture
source of §6), and freeze a new immutable GPAT lock. DSDG/DiffFAS environments are not reused unless future evidence
requires it.

## 5. D10.4 — `Normalize` (owner-confirmed at M7A review)

The frozen §9.4 artifact map is unchanged. With the intermediate variable `u` named explicitly:

```
A_rgb  = mean_channel(|x_hat − x_t|)
A_freq = upsample(|ΔLH| + |ΔHL| + |ΔHH| + γ·|ΔLL|)
u      = M ⊙ (0.5·A_rgb + 0.5·A_freq)
A      = Normalize(u)
Normalize(u) = clip(u / 2.0, 0.0, 1.0)
```

The denominator 2.0 applies to the **whole** pre-normalized artifact map `u` (2.0 is the maximum of `A_rgb` on the
[-1, 1] range). It does **not** mean `clip(A_rgb / 2, 0, 1)`, and it does not remove `A_freq` or `M` from `A`. The owner
confirmed this reading at the M7A review (2026-09-29).

## 6. NAFNet source pin (architecture source only)

| Field | Value |
|---|---|
| Repository | `https://github.com/megvii-research/NAFNet` (spec §30 R16) |
| Commit | `2b4af71ebe098a92a75910c233a3965a3e93ede4` (immutable; `refs/heads/main` at read-only discovery on 2026-09-29; committed 2024-03-29) |
| `basicsr/models/archs/NAFNet_arch.py` | sha256 `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`; symbols `SimpleGate` (L22–25), `NAFBlock` (L27–81; defaults DW_Expand=2, FFN_Expand=2, drop_out_rate=0) |
| `basicsr/models/archs/arch_util.py` | sha256 `5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005`; symbols `LayerNormFunction` (L264–289), `LayerNorm2d` (L291–301) |
| `basicsr/models/archs/local_arch.py` | sha256 `c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26`; import-time dependency of `NAFNet_arch.py` line 20 only (`Local_Base`); not architecturally used |
| `LICENSE` | sha256 `a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad` (MIT) |
| Weights | none; G_res is trained from scratch |

Known import caveats for M7C (not resolved here): `arch_util.py` imports `basicsr.utils.get_root_logger`, and
`LayerNormFunction.backward` reads `ctx.saved_variables`. Both must be qualified on the pinned torch before use. Nothing
was installed or vendored in M7A; the historical project contains no NAF implementation or pin.

## 7. Track-B instantiability clarification (A1 and fair_track_v1 unchanged)

Amendment A1 and `configs/frozen/fair_track_v1.yaml` are not modified. A10 adds: **Track-B dataset instantiability is
evaluated per method according to the metadata that variant actually requires.** The blanket
`siwmv2: NOT_INSTANTIABLE_MISSING_SUBJECT_ID` of `fair_track_v1.track_b.dataset_coverage` applies to methods that need
subject identity to instantiate their training relation (E06b, E07b). Therefore:

| Variant | Track | SiW-Mv2 | Basis |
|---|---|---|---|
| E08 / B0 | A | included | A1 Track A (unchanged) |
| E09 / B1 | B | **included** | needs attack_macro only (D04) |
| E10 / B2 | B | **included, identity CE masked** | D03 / DEV-022 |
| E11 / B3 | B | **included, identity CE masked** | D03 / DEV-022 |

No pseudo identity is permitted anywhere.

## 8. Variant deltas (from B0)

| Variant | Delta from B0 |
|---|---|
| B0 (E08) | frozen §23.2 `configs/methods/gpat_b0.yaml` (unchanged) + A10 D01, D07–D17 |
| B1 (E09) | + attack head `Linear(512, 6)`, `lambda_type = 0.2`, D05 loss, D06 warmup; datasets per D04 |
| B2 (E10) | + GRL(1.0) identity head `Linear(512, 60)`, `lambda_idadv = 0.1`, D02 sign, D03 masking (DEV-022) |
| B3 (E11) | B1 ∪ B2 |

The B1/B2/B3 config files are **not** created by M7A.

## 9. Historical GPAT reconciliation (Level 3, reference only)

The historical GPAT (`src/prism_fas/synthesis/` at `6f0642a1`) is a different, recipe-conditioned PRISM-FAS M8 generator:
224×224 [0,1] images, a 4-conv GroupNorm artifact encoder plus a 41-D LLM recipe encoder, 4 FiLM residual blocks at band
resolution, HF-only residual composited by a recipe support mask and clamped, no ΔLL, no discriminator, no identity
adversary, no attack head, no EMA, CASIA+MSU only, a validation-loss checkpoint rule, and a hand-written orthonormal Haar
instead of ptwt. The frozen spec states "Standalone GPAT study — no PRISM-FAS-B components". Consequently **no historical
scientific choice is imported**. The full per-component table (with file SHA-256 and classification) is in the resolution
YAML and `outputs/audit/M7A_GPAT_CONTRACT_RESOLUTION.md`.

Historical details recorded as future implementation references (conceptual only; nothing copied):

| Detail | Source (historical HEAD `6f0642a1`) | sha256 | Class |
|---|---|---|---|
| frozen-teacher lock (`requires_grad_(False)`, `train()` override pinned to eval) | `src/prism_fas/synthesis/quality_models.py` `DifferentiableAdaFace` (L70–106) | `07ab07b885a707b1587f069c462d7c26cde84e2dfb53f73d897cfe00095dc2f0` | COMPATIBLE_IMPLEMENTATION_REFERENCE |
| FiLM modulation form `h·(1+γ)+β` | `src/prism_fas/synthesis/gpat_model.py` `FiLMResidualBlock` (L60–73) | `5d149bdfab0bf6195809072f8998861e3cf4ea4bc166ede4f9890d7b77d459c1` | USEFUL_BUT_REQUIRES_ADAPTATION |
| anisotropic L1 TV (mean of horizontal + vertical) | `src/prism_fas/synthesis/gpat_losses.py` `total_variation_loss` (L74–79) | `994c16b919c15786e7d0c8af6ea2082bf52154b0c220f47fddf0e25ca9535c0e` | COMPATIBLE_IMPLEMENTATION_REFERENCE |
| deterministic per-(seed, epoch) SHA-256 → PCG64 permutation, tail kept | `src/prism_fas/synthesis/gpat_trainer.py` `batch_slices` (L56–67) | `92a9f058c3ab2fdc34c20e4bdd46efc3dea047a935efe6ca5a6690eadb0dedb7` | USEFUL_BUT_REQUIRES_ADAPTATION |
| per-step LambdaLR warmup + cosine mechanism | `src/prism_fas/synthesis/gpat_trainer.py` `cosine_schedule` (L40–53) | `92a9f058c3ab2fdc34c20e4bdd46efc3dea047a935efe6ca5a6690eadb0dedb7` | USEFUL_BUT_REQUIRES_ADAPTATION |
| checkpoint save/load with RNG state and identity binding | `src/prism_fas/synthesis/gpat_checkpoint.py` `save_checkpoint`/`load_checkpoint` (L51–99) | `42c4eb0b70c603de701f694f38e2c7f4ea45ae66b2651aceb01075d79a1d591b` | USEFUL_BUT_REQUIRES_ADAPTATION |
| pure-torch Haar (only as an independent cross-check oracle, never the production transform) | `src/prism_fas/synthesis/dwt.py` `haar_dwt2`/`haar_idwt2` (L20–60) | `756876c358bc8420d16587a0bdb690568a9a50ec744428db5fdcfb336940a086` | SUPERSEDED_BY_CURRENT_AUTHORITY |

Explicitly **not** carried forward: the historical AdaFace checkpoint (CVLFace export
`43bd2d57…`, refused by the frozen registry), its bicubic 112 resize and [0,1] clamp, the LaPa parser names (conflicts with
D09), the recipe/LLM conditioning, support-mask compositing and clamping in the forward, the style/map/strength/residual
losses and weights, AdamW β(0.9, 0.999) and its learning rates, the 5 % warmup fraction and min LR 1e-6, validation-loss
selection with early stopping, the CASIA+MSU-only scope, and the 224×224 [0,1] contract.

## 10. Deviation register

- DEV-003 → resolved by A10/D01 (status update appended to `outputs/audit/deviation_report.md`).
- Q-05 → resolved by A10/D02.
- Q-21 → resolved for M7 by A10/D09 (class names remain unassigned; M9 region crops still need names if they require them).
- **DEV-022** (new; verified as the next unused number after DEV-021): D03 masked identity supervision on SiW-Mv2 for
  B2/B3 — CONTROLLED_ADAPTATION.
No other deviation id is created; in-house clarifications are not deviations.

## 11. Residual items handed to M7B/M7C (non-scientific; not decided here)

These were found while encoding A10. None changes a decision above; each needs closing before the named sub-milestone.

| Id | Item | Needed by |
|---|---|---|
| R-01 | Mapping of ptwt's level-1 detail tuple to the names LH/HL/HH (a labelling convention only; must be used consistently for E_art source bands, G_res and A_freq) | M7B |
| R-02 | Resampling of the 128×128 mask M to 256×256 inside §9.4 A (TV(M) stays at 128×128) | M7B |
| R-03 | Whether VAL selection metrics are computed on the export-boundary uint8 image through the frozen preprocessing or on the float x_hat through the D07 adapter | M7B |
| R-04 | Teacher-parity tolerance (D07) | M7C |
| R-05 | NAFNet import caveats (§6) on the pinned torch | M7C |
| R-06 | AMP autocast scope for G/E_art/D (spec: fp16 AMP; teachers fp32 per D07) | M7B |
| R-07 | Optimizer membership of the identity head (B2/B3) and attack head (B1/B3) in the G/E_art Adam | M7B |

## 12. What A10 does not change

The frozen DOCX; A1–A9; `configs/frozen/fair_track_v1.yaml`; `configs/methods/gpat_b0.yaml` and its snapshot;
`outputs/audit/method_status.csv`; `outputs/audit/M6E_FINAL_M6_CLOSURE.*`; every M6 method config and frozen snapshot;
E06b fidelity; the A9 DiffFAS exclusion; E07c historical evidence; `outputs/audit/STAGE_STATE.json`. `M6_CLOSED` stays true.
M7 scientific training has not started.
