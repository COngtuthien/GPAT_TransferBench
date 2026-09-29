# M7A — GPAT contract resolution and historical GPAT reconciliation

| | |
|---|---|
| Milestone | M7A (contract resolution; M7 scientific training NOT started) |
| Authority commit | `8d4ddb3c55e8809398346dbdc10d15318e14b60d` (M6E) |
| Amendment | `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A10_GPAT_M7_Contract_Resolution.md` |
| Resolution record | `configs/amendments/gpat_a10_m7_contract_resolution.yaml` |
| Machine evidence | `outputs/audit/M7A_GPAT_CONTRACT_RESOLUTION.json` |
| Preflight / tests | `tools/m7a_gpat_contract_resolution_preflight.py`, `tests/test_m7a_gpat_contract_resolution.py` |
| Date | 2026-09-29 |

## 1. Outcome

- Owner decisions D01–D17 are encoded in A10 (Level 2). DEV-003 and Q-05 are resolved; Q-21 is resolved for M7.
- DEV-022 (verified as the next unused id) registers the only controlled adaptation: B2/B3 masked identity CE on SiW-Mv2.
- B1 keeps SiW-Mv2. Track-B instantiability is per method (A1 and fair_track_v1 unchanged).
- NAFNet architecture source pinned: `megvii-research/NAFNet@2b4af71ebe098a92a75910c233a3965a3e93ede4` (not vendored).
- The historical project was reconciled as a Level-3 reference; no historical scientific choice was imported.
- No GPAT code, no B1/B2/B3 config, no environment, no package install, no GPU, no training, no checkpoint, no bank, no
  TEST/TRAIN/VAL data read. `method_status.csv`, the M6E closure, A1–A9, `gpat_b0.yaml` and STAGE_STATE are unchanged.

## 2. Historical project audited (read-only)

| Field | Value |
|---|---|
| Path | `/home/cong/PRISM_FAS_C_LLM_Project` |
| Git | yes; branch `gpu-work/e7-gpat-bank-prep`; HEAD `6f0642a1d05c35e4c1778d329fb547f1b22a4115` |
| Worktree | dirty **before** M7A (2 modified tracked files, 363 untracked files); `git status --porcelain=v1 --untracked-files=all` sha256 `3b91017a2ecb0d3734b82cc2edc03772da0de9d863697d92f6ad62350a51eb77`, `git diff` sha256 `541030621c84ed0154fc0a5714a188a284649814134f84d87982d0bc27cea5c1`; both re-verified unchanged by the preflight |
| GPAT design | recipe-conditioned PRISM-FAS M8 generator (224×224, [0,1], 41-D LLM recipe conditioning, HF-only residual, no discriminator) |

GPAT implementation core (all clean at HEAD; sha256 in the resolution record):
`src/prism_fas/synthesis/{dwt,gpat_model,gpat_losses,gpat_trainer,gpat_checkpoint,gpat_contracts,quality_models,masks,m8_pipeline}.py`,
`configs/synthesis/gpat_m8.yaml`, `docs/M8_GPAT_CONTRACT.md`, `tests/test_m8_gpat_synthetic_bank.py`, `modal_m8.py`,
`scripts/m8_local_smoke.py`. GPAT consumers (banks, E7 three-fold evaluation, detector, recipes, pipeline adapters and
`reports/c_ext_q1q2_v1/e7_three_fold/{e7c_gpat_prep,gpat_bank}`) use generated banks and are NOT_RELEVANT to M7 training.
The historical project contains **no** NAFNet/NAFBlock, ptwt, GRL, identity adversary, attack-type head, warmup, EMA,
discriminator, spectral (S_radial/S_orient) loss or rank-based VAL selection.

## 3. Old-GPAT → current-GPAT reconciliation

| Component | Current authority | Historical file | Historical behaviour | Classification | Action |
|---|---|---|---|---|---|
| E_art | ResNet-18 ImageNet, 12-ch, 512-D global + layer3 spatial (§9.2) | `gpat_model.py::ArtifactEncoder` | 4 conv + GroupNorm, masked GAP, 128-D, RGB | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| G_res | NAFResidualUNet 32/[2,2,4,8]/12/[2,2,2,2] (§9.3) | `gpat_model.py::GPATResidualModel` | 4 FiLM residual blocks at band resolution | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| DWT/IDWT | ptwt Haar L1 reflect (§4; D11) | `dwt.py::haar_dwt2/haar_idwt2` | hand-written orthonormal Haar, tol 1e-6 | SUPERSEDED_BY_CURRENT_AUTHORITY | optional test oracle only |
| Residual injection form | `scale·tanh(·)`, `M ⊙ ΔHF` (§9.1, §9.3) | `GPATResidualModel.forward` | `tanh·map·0.15` on HF | COMPATIBLE_IMPLEMENTATION_REFERENCE | reference only |
| Compositing / clamp | none; no clamp in training (D10.20) | `GPATResidualModel.forward` | support-mask composite + clamp(0,1) | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not carry |
| HF scale | 0.15 (§9.3) | `gpat_m8.yaml` | 0.15 | EXACT_MATCH_CURRENT_AUTHORITY | none |
| LL delta | ΔLL head, `0.05·γ` (§9.1) | `GPATResidualModel` | no ΔLL head | SUPERSEDED_BY_CURRENT_AUTHORITY | spec head required |
| Mask M | free sigmoid, shared over RGB (§9.3) | `GPATResidualModel.forward` | sigmoid × recipe support mask | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| FiLM | `Linear(512,2C)`, `(1+γ)h+β` after NAFBlocks (D10.8) | `gpat_model.py::FiLMResidualBlock` | MLP, `(1+γ)h+β` inside a residual block | USEFUL_BUT_REQUIRES_ADAPTATION | form only |
| NAF blocks | pinned NAFNet (A10 §6) | — | absent | NOT_RELEVANT_TO_CURRENT_GPAT | use pin |
| PatchGAN | pix2pix 70×70 (§10.4; D10.13–15) | `gpat_losses.py::loss_manifest` | no discriminator | SUPERSEDED_BY_CURRENT_AUTHORITY | implement per spec |
| Loss set | §10 losses | `gpat_losses.py::compute_losses` | style/identity/map/strength/TV/residual | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| TV operator | anisotropic L1, mean (D10.3) | `gpat_losses.py::total_variation_loss` | anisotropic L1 mean on masked deltas | COMPATIBLE_IMPLEMENTATION_REFERENCE | operator only |
| Spectral S_radial/S_orient | D10.1–2 | — | absent | NOT_RELEVANT_TO_CURRENT_GPAT | per A10 |
| Teacher freezing | frozen, eval, no grad (D07) | `quality_models.py::DifferentiableAdaFace` | `requires_grad_(False)`, `train()` pinned to eval | COMPATIBLE_IMPLEMENTATION_REFERENCE | mechanics only |
| Teacher weights / preprocessing | registry AdaFace `52cca7c6…`, exact-linear INTER_AREA, BGR, no clamp (D07) | `DifferentiableAdaFace`; `gpat_m8.yaml::identity_model` | CVLFace `43bd2d57…` (refused by registry), bicubic, clamp | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| Landmark / parsing | D08 coordinates; D09 unnamed 1..10 | `masks.py::PARSING_LABELS` | LaPa names "verified empirically" | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | no names |
| Identity adversary / GRL | D02, D03 | — | absent | NOT_RELEVANT_TO_CURRENT_GPAT | per A10 |
| Attack-type head | D05 | — | absent | NOT_RELEVANT_TO_CURRENT_GPAT | per A10 |
| Warmup | D06 | — | absent | NOT_RELEVANT_TO_CURRENT_GPAT | per A10 |
| Optimizer / schedule values | Adam 2e-4 β(0.5,0.999); warmup 5 epochs from 0; min 2e-6 | `gpat_m8.yaml` | AdamW β(0.9,0.999), 5 % warmup, min 1e-6, early stop | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not carry |
| Schedule mechanism | per step (D10.22) | `gpat_trainer.py::cosine_schedule` | per-step LambdaLR, warmup `(step+1)/W` | USEFUL_BUT_REQUIRES_ADAPTATION | warmup from 0 |
| Deterministic batching | shuffled pass, tail kept (D15) | `gpat_trainer.py::batch_slices` | SHA-256 → PCG64 permutation, tail kept | USEFUL_BUT_REQUIRES_ADAPTATION | re-key; groups of 2 |
| Precision / clip | fp16 AMP + GradScaler; clip 1.0 (§10.5) | `GPATTrainer` | same | EXACT_MATCH_CURRENT_AUTHORITY | none |
| Seeding | repository conventions | `seed_everything` | torch only | SUPERSEDED_BY_CURRENT_AUTHORITY | repo conventions |
| EMA | 0.999 from end of epoch 5; E_art + G_res | — | absent | NOT_RELEVANT_TO_CURRENT_GPAT | per spec |
| Checkpointing | all 51 EMA candidates, hashed (D16) | `gpat_checkpoint.py` | model/optimizer/RNG + identity binding | USEFUL_BUT_REQUIRES_ADAPTATION | add EMA/D state |
| VAL selection | rank SelectionScore, separate process (§10.6; D14, D16) | `gpat_m8.yaml::checkpoint_selection` | min val loss, early stop, val inside training | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| Dataset scope | CASIA+MSU+SiW for B0–B3 | `gpat_m8.yaml::data` | CASIA+MSU only | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| Pairing / subjects | `pairs_train_v1`; `source_subject` for B2/B3 CE | `gpat_m8.yaml::pair_plan` | recipe pair plan (896/224) | SUPERSEDED_BY_CURRENT_AUTHORITY | frozen manifests |
| Input contract | 256×256, [-1,1] | `gpat_contracts.py` | 224×224, [0,1] | CONFLICTS_WITH_CURRENT_AUTHORITY_DO_NOT_USE | do not use |
| Recipe / LLM conditioning | none (spec: no PRISM-FAS-B components) | `gpat_model.py::RecipeEncoder` | 41-D recipe encoder | NOT_RELEVANT_TO_CURRENT_GPAT | do not use |

Counts: EXACT_MATCH 2 · COMPATIBLE_REFERENCE 3 · REQUIRES_ADAPTATION 4 · SUPERSEDED 5 · CONFLICTS 11 · NOT_RELEVANT 7.

## 4. Per-decision comparison with the historical implementation

| Decision | Historical behaviour | Match? | Carry forward |
|---|---|---|---|
| D01 lambda_dir | no L_spec, no lambda_dir | n/a | nothing |
| D02 GRL sign | no identity adversary | n/a | nothing |
| D03 B2/B3 scope + masked identity | CASIA+MSU only; no identity labels | no | nothing |
| D04 B1 scope | CASIA+MSU only; no attack head | no | nothing |
| D05 attack loss | absent | n/a | nothing |
| D06 warmup | absent | n/a | nothing |
| D07 teacher path | differentiable AdaFace but wrong checkpoint, bicubic, clamp | partial | freezing mechanics only |
| D08 landmarks | no landmark loss | n/a | nothing |
| D09 parser | LaPa names assigned | **conflict** | nothing |
| D10 operators | TV and FiLM forms match; all else absent or different | partial | TV/FiLM forms as references |
| D11 DWT sanity | own Haar, tol 1e-6 max-abs | different library | optional oracle |
| D12 artifact_scale=0 | LL hard-locked, HF bounded; no ΔLL | partial | nothing |
| D13 ArtSim | absent | n/a | nothing |
| D14 selection | min validation loss | **conflict** | nothing |
| D15 epoch/tail | tail kept, deterministic shuffle | compatible | batching mechanism (re-keyed) |
| D16 EMA/VAL | no EMA; validation inside training | **conflict** | nothing |
| D17 environment | Modal/L4 cloud runtime | different | nothing |

## 5. HISTORICAL_ALTERNATIVE_FOR_FUTURE_ABLATION (not adopted)

1. Hard support-mask compositing (exact identity outside a region) instead of the soft `L_bg` term.
2. Masked first/second-moment wavelet style loss on HF bands as an alternative to `L_spec`.
3. Empirically named FaceXFormer parser classes (would allow excluding hair from `M_face_dilated`).

None materially overrides the owner decisions; each would be a separate, explicitly approved ablation.

## 6. NAFNet pin

`https://github.com/megvii-research/NAFNet` @ `2b4af71ebe098a92a75910c233a3965a3e93ede4` (read-only `git ls-remote`
and a shallow clone into a scratch directory outside the repository; nothing vendored or installed):

| File | sha256 | Needed for |
|---|---|---|
| `basicsr/models/archs/NAFNet_arch.py` | `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967` | `SimpleGate` L22–25, `NAFBlock` L27–81 |
| `basicsr/models/archs/arch_util.py` | `5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005` | `LayerNormFunction` L264–289, `LayerNorm2d` L291–301 |
| `basicsr/models/archs/local_arch.py` | `c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26` | import-time only (`NAFNet_arch.py` line 20) |
| `LICENSE` | `a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad` | MIT |

Architecture source only; no weights. Import caveats for M7C: `basicsr.utils.get_root_logger`, `ctx.saved_variables`.

## 7. Deviation register changes (append-only)

DEV-003 → APPROVED (resolved by A10/D01); Q-05 → resolved (D02); Q-21 → resolved for M7 (D09); **DEV-022** created
(D03, CONTROLLED_ADAPTATION, E10/E11); Track-B instantiability clarification recorded. The top summary table is
byte-unchanged; the M7A section supersedes it for these ids.
