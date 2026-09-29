# M7B — GPAT B1/B2/B3 config freeze

| | |
|---|---|
| Milestone | M7B (config freeze only; no GPAT code, environment, training, checkpoint or bank) |
| Authority commit | `d59b9a302c61ed440e711abf6ffab9560458f779` (M7A: resolve GPAT scientific and execution contract) |
| Amendment | A10, document sha256 `7baf7888394c4cdd57db09e604bd910153a612e1765b8dc57de3c1d5d833d337`, record sha256 `57b37f09ca4279ad3999f4f9efa21fbf6fa39836574a0336bd13e34b2bd53203` (both unchanged) |
| Owner clarifications | `configs/amendments/gpat_m7b_owner_clarifications.yaml` (R-01, R-02, R-03, R-06, R-07; additive; not an amendment; not frozen-spec facts) |
| Machine evidence | `outputs/audit/M7B_GPAT_CONFIG_FREEZE.json` (includes the full field-level comparison matrix) |
| Preflight / tests | `tools/m7b_gpat_config_freeze_preflight.py`, `tests/test_m7b_gpat_config_freeze.py` |
| Date | 2026-09-29 |

## 1. Outcome

| Config | Experiment | Track | sha256 (live = snapshot) |
|---|---|---|---|
| `configs/methods/gpat_b0.yaml` | E08 / GPAT-B0 | A | `0a268d2c9d1914a9e687d08cdfb96d5bf793eaa22559476b8e68d3cf58035d6a` (unchanged) |
| `configs/methods/gpat_b1.yaml` | E09 / GPAT-B1 | B | `60d8e6581c3026f969ae92a79903a172f1ade2c95a8ea155b9cb99435a1b90cc` |
| `configs/methods/gpat_b2.yaml` | E10 / GPAT-B2 | B | `4875138aff301145ccd763039386f92e561a9ea83fbaa4e6682b49b251fe3af9` |
| `configs/methods/gpat_b3.yaml` | E11 / GPAT-B3 | B | `620303695d97ba4bab2e6081b29242080db7ead4976599709d88183edac462ca` |

Snapshots: `frozen_config_snapshot/configs/methods/gpat_b{1,2,3}.yaml`, byte-identical to the live configs. Each config
repeats every B0 field first, then the variant, status, provenance and data blocks, the supervision blocks and a
`shared_contract` block that is byte-identical in all three files.

M7 status: **CONFIGS_FROZEN · IMPLEMENTATION_NOT_STARTED · ENVIRONMENT_NOT_CREATED · SCIENTIFIC_TRAINING_NOT_STARTED**.
M6_CLOSED = true.

## 2. A10 §11 residual items

| R-id | Topic | Required by | Status after M7B | Blocks M7B | Blocks M7C | Authority |
|---|---|---|---|---|---|---|
| R-01 | ptwt level-1 detail tuple → LH/HL/HH | M7B | RESOLVED | no | no | M7B owner clarification (IMPLEMENTATION_CLARIFICATION) |
| R-02 | 128×128 mask M inside the §9.4 map | M7B | RESOLVED | no | no | M7B owner clarification (IMPLEMENTATION_CLARIFICATION) |
| R-03 | VAL image representation | M7B | RESOLVED | no | no | M7B owner clarification (SCIENTIFIC_EVALUATION_CLARIFICATION) |
| R-04 | teacher-parity tolerance (D07) | M7C | DEFERRED_TO_M7C | no | yes | A10 D07 (frozen at the M7C parity qualification) |
| R-05 | NAFNet import caveats | M7C | DEFERRED_TO_M7C | no | yes | A10 §6 |
| R-06 | AMP autocast scope | M7B | RESOLVED | no | no | M7B owner clarification (RUNTIME_COMPATIBILITY) |
| R-07 | optimizer membership of the heads | M7B | RESOLVED | no | no | M7B owner clarification (IMPLEMENTATION_CLARIFICATION, consistent with A10 D02/D06) |

A10 §11 contains exactly R-01…R-07; no other R item exists.

Observation found while freezing, closed at owner review (not a variant delta, shared by B0–B3):

| Id | Item | Status | Class | Authority | Blocks M7B | Blocks M7C |
|---|---|---|---|---|---|---|
| M7B-OBS-01 | Discriminator LR schedule, AMP, GradScaler and gradient clipping | RESOLVED_FROM_FROZEN_SPEC | FROZEN_SPEC_DERIVED_EXECUTION_POLICY | frozen spec §10.5 (not an A10 amendment, not a new owner decision) | no | no |

Confirmed at owner review (no new DEV id):

| Item | Decision | Class |
|---|---|---|
| Identity head bias (B2/B3) | `Linear(512, n_subjects = 60, bias = True)` | OWNER_IMPLEMENTATION_CLARIFICATION |
| VAL clamp policy | VAL_PRE_EXPORT_FLOAT_NO_CLAMP: selection metrics consume `x_hat_float` before export; no clamp for VAL scoring; no uint8, encode/decode or export-clamp round trip (D10.20 clamps only at the uint8/export boundary) | OWNER_EVALUATION_CLARIFICATION |
| GradScaler topology | independent `G_SCALER` (G_OPT), `D_SCALER` (D_OPT), `WARMUP_SCALER` (B1/B3 WARMUP_OPT; never reused by G_SCALER) | IMPLEMENTATION_CLARIFICATION |
| Band/channel mapping | channels 0..2 ΔLL, 3..5 ΔLH, 6..8 ΔHL, 9..11 ΔHH, 12 mask_logit (D10.12); `A_freq_128` channel mean (D10.5) | IMPLEMENTATION_CLARIFICATION |

Identity class-index order stays deferred to M7C (deterministic, TRAIN-only, recorded in the run).

## 3. Supervision / loss matrix

| | B0 | B1 | B2 | B3 |
|---|---|---|---|---|
| lambda_type | 0.0 | 0.2 | 0.0 | 0.2 |
| lambda_idadv | 0.0 | 0.0 | 0.1 | 0.1 |
| attack head | — | `Linear(512, 6)`, bias | — | same as B1 |
| identity head | — | — | GRL(1.0) → `Linear(512, 60)` | same as B2 |
| attack warmup | — | 10 epochs | — | same as B1 |
| DEV-022 | — | — | SiW identity CE masked | same as B2 |
| labels used | binary | + attack_macro | + source_subject | + both |

Every other scientific loss coefficient is identical to B0 (see §4).

## 4. B0 → B1/B2/B3 comparison (every B0 field)

| Field | B0 | B1 | B2 | B3 | Class |
|---|---|---|---|---|---|
| `method` | GPAT-B0 | GPAT-B1 | GPAT-B2 | GPAT-B3 | PROVENANCE_ONLY |
| `supervision` | binary_only | binary_plus_attack_macro | binary_plus_identity_adversary | binary_plus_attack_macro_plus_identity_adversary | PROVENANCE_ONLY |
| `input_resolution` | 256 | 256 | 256 | 256 | SAME |
| `wavelet.{library,family,level,boundary_mode}` | ptwt, haar, 1, reflect | same | same | same | SAME |
| `gamma` | 0.0 | 0.0 | 0.0 | 0.0 | SAME |
| `artifact_encoder.{backbone,input_channels,highpass}` | resnet18 IN1K, 12, 9/1.5 | same | same | same | SAME |
| `artifact_encoder.attack_type_head` | false | true | false | true | B1_DELTA |
| `artifact_encoder.identity_adversary` | false | false | true | true | B2_DELTA |
| `residual_generator.*` (8 fields) | NAFResidualUNet 32, [2,2,4,8], 12, [2,2,2,2], 0.15, 0.05, 13 | same | same | same | SAME |
| `training.*` (13 fields) | 60, 4, 2, 8, Adam, 2e-4, (0.5,0.999), 0, 5, 2e-6, amp, clip 1.0, EMA 0.999, seeds 42/1337/2026 | same | same | same | SAME |
| `loss.lambda_id / lm / parse / kl / low / con` | 1.0 / 1.0 / 0.5 / 0.1 / 2.0 / 1.0 | same | same | same | SAME |
| `loss.triplet_margin / lambda_spec / lambda_dir / lambda_adv` | 0.2 / 0.5 / 0.5 / 0.1 | same | same | same | SAME |
| `loss.lambda_type` | 0.0 | 0.2 | 0.0 | 0.2 | B1_DELTA |
| `loss.lambda_idadv` | 0.0 | 0.0 | 0.1 | 0.1 | B2_DELTA |
| `loss.lambda_budget / artifact_min / artifact_max` | 0.5 / 0.01 / 0.25 | same | same | same | SAME |
| `loss.lambda_tv / tv_delta_scale / lambda_bg` | 0.05 / 0.25 / 1.0 | same | same | same | SAME |

Fields added by M7B (not in the B0 file), classified leaf by leaf in the JSON evidence:

| Block | Class (leaf count) |
|---|---|
| `variant`, `status`, `provenance`, `data.dataset_scope_authority` | PROVENANCE_ONLY (29) |
| `data` (manifests, datasets, splits, TEST firewall) | SAME (18) |
| `data.splits.TRAIN.used_for` (adds `attack_head_warmup`) | B1_DELTA (1) |
| `data.label_fields_used` | B3_UNION (1) |
| `attack_type_supervision`, `attack_warmup` | B1_DELTA (64) + SAME (1: `class_weights` null) |
| `identity_adversary` | B2_DELTA (39) |
| `shared_contract` (identical in B1/B2/B3; applies to B0) | IMPLEMENTATION_CLARIFICATION (216) |

Totals over all 423 compared leaves: SAME 67, B1_DELTA 67, B2_DELTA 41, B3_UNION 1, IMPLEMENTATION_CLARIFICATION 216,
PROVENANCE_ONLY 31, unexplained 0.

## 5. Frozen shared contract (all variants)

- **R-01 bands:** `ptwt.wavedec2` order `(cA, (cH, cV, cD))`; `LL = cA, LH = cH, HL = cV, HH = cD`; no display-convention
  swap; axes `(-2, -1)`. M7C must test directional synthetic patterns, the exact coefficient order and DWT→IDWT.
- **R-02 mask:** `M = sigmoid(mask_logit)`, soft, no threshold, 128×128, used directly for the wavelet-domain product (no
  resize); for the artifact map only `M_256 = bilinear_interpolate(M, (256, 256), align_corners = False)`.
  `A_rgb = mean_channel(|x_hat − x_t|)`; `A_freq_128 = mean_channel(|ΔLH| + |ΔHL| + |ΔHH| + γ|ΔLL|)`;
  `A_freq_256 = bilinear_x2(A_freq_128)`; `u = M_256 · (0.5·A_rgb + 0.5·A_freq_256)`; `A = clip(u / 2.0, 0, 1)` (D10.4).
- **R-03 VAL:** selection metrics (ID, Dice, ArtSim, NME, LFErr) use the native float EMA output `x_hat_float` before
  any uint8 conversion, encoding or export; each frozen evaluator gets it through its own frozen/differentiable (D07)
  adapter; ArtifactProbe gets evaluator-compatible float preprocessing; no uint8 or encoded round trip. TEST never.
- **R-06 precision:** CUDA fp16 autocast for E_art, G_res, PatchGAN D and the enabled heads; fp32 for DWT/IDWT, LL/HF
  composition, x_hat, A, FFT, S_radial, S_orient, TV/budget scalars, sensitive loss reductions, CE, Dice, KL, teacher
  cosines and all frozen teacher/evaluator forwards (AdaFace, FaceXFormer, F_art, ArtifactProbe). No bf16. GradScaler;
  `unscale_` then clip 1.0. x_hat stays differentiable into the teachers; only the x_t targets are detached.
- **R-07 optimizers:** `G_OPT = Adam(G_res + E_art + enabled heads)` — B0 `[G_res, E_art]`, B1 `+ attack_type_head`,
  B2 `+ identity_adversary_head`, B3 `+ both`. D_OPT is a separate Adam over D only. Frozen modules never optimized.
  B1/B3 `WARMUP_OPT = AdamW(E_art + attack_type_head)`, discarded after warmup epoch 10, then a fresh G_OPT.
- **Discriminator (M7B-OBS-01, spec §10.5):** `D_OPT = Adam(D)`, lr 2e-4, β (0.5, 0.999), wd 0; the same per-update
  schedule as G (linear warmup from 0 over epochs 1–5 = 5,525 updates to 2e-4, cosine to 2e-6 at the final update); fp16
  autocast for the D forward; `D_SCALER`; clip 1.0 over D trainable parameters only, after `D_SCALER.unscale_(D_OPT)`.
- **GradScalers:** independent `G_SCALER` / `D_SCALER` / `WARMUP_SCALER`; per optimizer: `scale(loss).backward()` →
  `unscale_` → `clip_grad_norm_(its parameters, 1.0)` → `step` → `update`, at the A10 D15 accumulation boundary.
- **Budget (D15):** 8,838 TRAIN pairs, physical 4, accumulation 2, drop_last false → 2,210 microbatches, 1,105 updates per
  epoch, tail group of 6 samples, 66,300 updates per run; LR warmup 5,525 updates from 0, cosine to 2e-6; EMA 0.999 from
  update 5,525 over E_art + G_res; seeds 42/1337/2026. B1/B3 warmup: 139 steps/epoch (last batch 6), 1,390 steps, outside
  the 60 generator epochs.
- **Selection (D13/D14/D16):** 51 EMA candidates (epochs 10..60), separate post-training VAL process, average-rank ties,
  `p = (rank − 1)/50`, weights 0.25/0.15/0.25/0.20/0.15, tie-break higher raw ArtSim then earlier epoch, NME normalized
  by the target landmark bounding box, ArtSim = M5 ArtifactProbe v1.
- **Sanity references (not executed):** D11 DWT→IDWT `[100,3,256,256]` uniform[-1,1], CPU seed 42, fp32, max_abs < 1e-5;
  D12 zero residual (HF and LL) → x_hat ≈ x_t < 1e-5 and source-independent < 1e-5.
- **Environment:** NOT_CREATED_M7B. Future base gpat-m5 / m6_core_gpu + ptwt, PyWavelets, NAFNet architecture source
  `megvii-research/NAFNet@2b4af71ebe098a92a75910c233a3965a3e93ede4` (no weights).

## 6. Loaders and validators

`methods/common/config.py` (M6 baseline loader, fixed `METHOD_FILES`), `tools/m6b_validate_configs.py` (fixed M6B
`EXPECTED`) and the other M6 preflights enumerate explicit baseline configs and never glob `configs/methods/`; B0 is not
registered in any of them either. GPAT config loading belongs to the M7C runtime, so **no loader or validator was
changed**. The M7B validator is `tools/m7b_gpat_config_freeze_preflight.py`. The M6E and M7A preflights assert B1–B3
absence only at their own candidate time; their tests read the commit that added them and stay valid.

## 7. What M7B does not change

The frozen DOCX; A1–A10 and the A10 record; `configs/methods/gpat_b0.yaml` and its snapshot; `configs/frozen/*`;
`outputs/audit/method_status.csv`; the M6E closure; `outputs/audit/deviation_report.md` (no new DEV id); STAGE_STATE;
every M6 method config. No GPU, TRAIN/VAL/TEST sample access, environment, install, runtime code, training, checkpoint or
bank. `methods/gpat/` still holds only `.gitkeep`.
