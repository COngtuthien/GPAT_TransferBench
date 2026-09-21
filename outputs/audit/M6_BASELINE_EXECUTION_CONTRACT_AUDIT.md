# M6A1 — Baseline Execution-Contract Audit (E01–E07c)

**Date:** 2026-09-21 · **Branch:** `m6-baselines` · **Starting HEAD:** `89ea43e3d32d87d0641ec57bb62e6c72694dbe8a`
**Frozen spec sha256:** `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`
**Amendment A1 sha256:** `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472`

Companion: `M6_BASELINE_SOURCE_PROVENANCE.md` (sections A, D, E), `M6_BASELINE_SOURCE_GAPS.json`.

**Evidence classes.** Every execution-affecting value below carries exactly one:
`SPEC` (frozen specification) · `OFFICIAL_CODE` (pinned repository, file:line) · `PAPER`
(published text, quoted) · `M4_FROZEN_ADAPTATION` (frozen M4 / Amendment A1 artifact) ·
`OWNER_DECISION` (recorded owner resolution) · `INFERENCE` (derived here — **never** a scientific
parameter) · `UNKNOWN` (no source; blocks the method).

**No value in this document was invented.** Where a source is silent the row says `UNKNOWN` and the
method is marked BLOCKED in §G rather than given a default.

---

## B / C. Per-method execution contract with source evidence

### E01 — FAS-Aug · pin `RizhaoCai/FAS-Aug@0da1dd79…`

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| Input | pair-manifest target live 256×256; source spoof **not used** | `SPEC` | §8.1 |
| Operator set | 8 operators: `Color_Diversity, Color_Distortion, Reflection, BN_Halftone, Moire_Pattern, SFC_Halftone, Hand_Trembling, Low_Resolution` | `OFFICIAL_CODE` | `data/FAS_Augmentations.py:19–28` (`self.al`); `Original` is commented out and `config/default.py:77 NUM_OPS = 8` confirms the count |
| Operator ranges (min,max) | `(0.1,10) (0.1,10) (0.01,0.2) (0.01,0.4) (0.01,0.3) (0.01,0.2) (1,16) (0.01,0.9)` | `OFFICIAL_CODE` | `data/FAS_Augmentations.py:19–28` |
| Magnitude map | `mag = level·(high−low) + low` | `OFFICIAL_CODE` | `data/FAS_Augmentations.py:205` |
| Magnitude quantisation | `level = k/(NUM_MAG−1)`, `k ∈ {0..9}`, `NUM_MAG = 10` | `OFFICIAL_CODE` | `config/default.py:78`; `data/transform.py:70–72` |
| Operator selection | deterministic round-robin by `pair_id` | `SPEC` | §8.1 — **overrides** the official random policy (`data/transform.py:68` `random.choices(al, k=2)`) |
| Parameter RNG | `seed = SHA256(pair_id + global_seed) mod 2^31` | `SPEC` | §8.1 |
| RNG channel | Python stdlib `random`, seeded globally | `OFFICIAL_CODE` | operators draw from module-level `random` (`FAS_Augmentations.py:86,87,124,125,151,186,187`; `fas_aug_helper.py:32,36,50`); official driver seeds it at `train.py:53` |
| Canonical size | 256×256 | `SPEC` §8.1 + `OFFICIAL_CODE` `config/default.py:40 IN_SIZE = 256` (agree) |
| Output | one spoof-labeled 256×256 image per pair; operator name + all sampled parameters in provenance | `SPEC` | §8.1 |
| Track A ≡ Track B | yes | `SPEC` | §8.1 |
| Assets | 90 backgrounds, 48 noise textures, 190 moiré textures, 11 RGB + 7 CMYK ICC profiles | `OFFICIAL_CODE` | `data/fas_aug_helper.py:16–26`; `FAS_Augmentations.py:75–85,113–122` |

**E01-1 — CONFLICT (blocking): spec requires every output spoof-labeled; official code says 3 of the
8 operators do not produce a spoof.** The official operator table carries a fourth field
`changeLabel`, and it is `False` for `Color_Diversity`, `Hand_Trembling` and `Low_Resolution`
(`FAS_Augmentations.py:22,26,27`). The official consumer acts on it: `data/zip_dataset.py:50`
`tar = self.face_label if not changeLabel else 0` — i.e. an augmented image only becomes spoof
(label 0) when `changeLabel` is `True`; otherwise it keeps its original live label. Spec §8.1
nevertheless requires “**one spoof-labeled** canonical 256×256 image per intended pair” over
round-robin across “the official FAS-Aug operator set”. Under a uniform round-robin, **3/8 = 37.5 %
of the E01 synthetic bank would be labeled spoof although the method's own authors label it live.**
The spec cannot be modified and the official semantics cannot be ignored, so this is escalated, not
resolved. Candidate resolutions (**none adopted**): restrict the round-robin to the 5
`changeLabel=True` operators; keep all 8 and record the conflict as a declared deviation; or have
the owner issue a controlled override. Each changes E01's synthetic label distribution.

**E01-2 — DEFECT (blocking): `level = 0` yields a byte-identical copy of the live target, labeled
spoof.** `FAS_Augmentations.py:203–204`: `if level == 0: return img.copy(), False`. With the
official `NUM_MAG = 10`, `k = 0` occurs for 1 in 10 draws, so roughly **10 % of E01 “spoof” outputs
would be unmodified live images**. The spec fixes the seed but not the admissible level set. No
default was chosen.

**E01-3 — DETERMINISM HAZARD (blocking): texture choice depends on filesystem enumeration order.**
`fas_aug_helper.py:17–26` builds `R_NAME_LIST/BN_NAME_LIST/MP_NAME_LIST` with `os.listdir` at import
time — **unsorted** — and `getTexture` picks with `random.choice(...)` by index
(`fas_aug_helper.py:50`). Identical seeds therefore select different textures on hosts whose
directory order differs, which breaks the benchmark's byte-reproducibility requirement. Sorting the
lists would fix it but is a change to official behaviour; **not applied**.

**E01-4 — noted, non-blocking.** `Color_Diversity` and `Color_Distortion` ignore their magnitude
argument entirely (`FAS_Augmentations.py:71,95` — the parameter is literally named `nil`); their only
sampled parameters are ICC profile names drawn from insertion-ordered dicts, which is deterministic
under a seeded `random`. Asset paths are CWD-relative (`'data/profile/RGB Profiles/'`,
`'data/background/'`), so the adapter must fix the working directory; this is engineering, not
science.

---

### E02 — Frequency Substitution · spec-defined, no code source

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| Status | `CONTROLLED_ADAPTATION` | `SPEC` | §8.2 |
| Transform | per-channel FFT2 at 256×256, `fftshift` | `SPEC` | §8.2 |
| Eligible region | radial normalised `r ∈ [0.20, 0.50]` (`r=0` centre, `r=0.5` Nyquist) | `SPEC` | §8.2 |
| Block grid | non-overlapping 16×16 | `SPEC` | §8.2 |
| Block eligibility | ≥ 75 % of pixels in the eligible region | `SPEC` | §8.2 |
| Selection fraction | exactly 25 % of eligible blocks | `SPEC` | §8.2 |
| Substitution | target magnitude ← source magnitude in selected blocks **and conjugate-symmetric counterparts**; target phase preserved | `SPEC` | §8.2 |
| Reconstruction | `A_mix·exp(jφ_t)`, `ifftshift`, `ifft2`, real part, clip `[0,255]`, no spatial post-filter | `SPEC` | §8.2 |
| Track A ≡ Track B | yes; block mask saved as edit map | `SPEC` | §8.2 |
| Rounding of “exactly 25 %” | **`UNKNOWN`** | — | §8.2 silent (floor / round / ceil differ whenever `n_eligible % 4 ≠ 0`) |
| Selection algorithm | **`UNKNOWN`** | — | §8.2 says “using pair-specific RNG seed” but fixes neither the RNG nor the draw (permutation vs. sampling without replacement); different choices select different blocks from the same seed |
| Pair seed formula | **`UNKNOWN` for E02** | — | the `SHA256(pair_id + global_seed) mod 2^31` formula is stated in §8.1 **for FAS-Aug only**; extending it to E02 would be `INFERENCE` |
| Conjugate-counterpart index convention | **`UNKNOWN`** | — | for even `N=256` the shifted index map is `p ↔ 256−p`, so a 16×16 block `[16b, 16b+16)` maps to `(240−16b, 256−16b]`, which is **offset by one index** from the block grid. §8.2 does not say whether to mirror the block geometry or the exact index set, and the two give different spectra |

E02's **algorithm source is the frozen SPEC**, and it needs no third-party code — but four
execution-affecting, spec-silent choices remain unresolved, so **its execution contract is not
complete**. Source provenance stays `SPEC_DEFINED`; execution readiness stays **BLOCKED**. Each of the
four changes the produced pixels, so per M6A1 (“Never silently promote an inference into a scientific
parameter”) none was chosen.

---

### E03 — STDN · pin `yaojieliu/ECCV20-STDN@c79f1f8c…`

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| Status | `FAITHFUL_OFFICIAL`; TensorFlow 1.x (`>1.8.0, <1.13.0`, tested TF 1.13.0 / Py 3.6) | `SPEC` §8.3 + `OFFICIAL_CODE` `README.md` |
| Image size | 256 | `OFFICIAL_CODE` | `model/config.py:35` |
| Map size | 32 | `OFFICIAL_CODE` | `model/config.py:36` |
| Batch size | **2** | `OFFICIAL_CODE` | `model/config.py:39` — see **E03-1** |
| G:D ratio | 2 | `OFFICIAL_CODE` | `model/config.py:40` |
| Learning rate | 6e-5, exponential decay ×0.9 every `10.0 × 2000` steps, `staircase=True` | `OFFICIAL_CODE` | `model/config.py:41,42,46,47`; `model/model.py:101–108` |
| Optimizer | `tf.train.AdamOptimizer(lr)` (TF defaults β₁=0.9, β₂=0.999, ε=1e-8) | `OFFICIAL_CODE` | `model/model.py:115` |
| Weight EMA | `MOVING_AVERAGE_DECAY = 0.9999`; loss EMA 0.9 | `OFFICIAL_CODE` | `model/config.py:43`; `model/model.py:111,122` |
| Epochs / steps | `MAX_EPOCH = 50`, `STEPS_PER_EPOCH = 2000`, val 500 | `OFFICIAL_CODE` | `model/config.py:44,46,47` |
| Trace decomposition | `recon1 = (1−s)·img − b − resize(C,256) − T`; `trace = img − recon1` | `OFFICIAL_CODE` | `train.py:48–49` |
| Multi-scale discriminators | 256 / 160 / 40 | `OFFICIAL_CODE` | `train.py:32–33,53,60` |
| Loss weights | `g_loss = 50·esr + gan + (10·reg_live + 1e-4·reg_spoof)`; `d_loss = Σ/4`; `a_loss = 5·esr_a + {0.0 or 0.1}·pixel` | `OFFICIAL_CODE` | `train.py:94–96,99–103,106–109` |
| Hard-sample scaling | `U(0.1, 0.8)` per component | `OFFICIAL_CODE` | `train.py:63–66` |
| Geometry input | `XXX.npy` = **68-point 2-D landmarks**, normalised by image width | `OFFICIAL_CODE` | `README.md` (“The landmark (68) should be provided in the `XXX.npy` file”); `model/dataset.py:111,116,135`; verified on the shipped sample: shape `(68, 2)`, `float32` |
| Warp construction | 68 landmarks + 16 fixed border anchors → Delaunay triangulation → linear tri-interpolation of the offset field | `OFFICIAL_CODE` | `model/warp.py:88–107` |
| Supervision | binary Live/Spoof only; **no** attack-type labels | `SPEC` | §8.3 |

**E03-1 — CONFLICT, resolved by execution trace (non-blocking).** The repository ships **two**
config files that differ in exactly one value: root `config.py:38 BATCH_SIZE = 5` vs
`model/config.py:39 BATCH_SIZE = 2`. Both files are 2815 bytes with different sha256
(`a2658fca…` vs `31dbee51…`), confirming a one-character divergence. **Resolution:** both entry
points import the latter — `train.py:23` and `test.py:21` are `from model.config import Config`; the
root `config.py` is imported by nothing (verified by grep across the repository). The effective
official default is therefore **`BATCH_SIZE = 2`**, and root `config.py` is vestigial. Recorded
because the conflict is visible to any later reader.

**E03-2 — geometry source is satisfied by the frozen cache (non-blocking).** Spec §8.3 permits
producing STDN's geometry “deterministically from the frozen geometry cache”. M2 stores
`landmarks_px256` as `[68, 2] float32` on the canonical 256 grid for all 20,615 samples
(`outputs/audit/M2A_PREPROCESS_CONTRACT.md:60`; `configs/frozen/preprocess_v1.yaml:86`), with
deterministic re-run equality proven (`outputs/audit/M2_DETERMINISTIC_VALIDATION.json`). The point
count matches exactly. `generate_offset_map` is **ordering-invariant** — Delaunay triangulation plus
per-index correspondence — so the exact 68-point semantic layout does not affect the warp, provided
both faces come from the same detector, which holds. The one place ordering matters is the
horizontal-flip permutation `lm_reverse_list` (`model/dataset.py:99–100`), which assumes the iBUG-68
layout; whether FaceXFormer's 68-point output is iBUG-ordered is recorded as `UNKNOWN`
(`M2A_PREPROCESS_CONTRACT.md:58` says only “68-point layout”). Flip is a **training augmentation**,
so the cleanest resolution is to disable it — but that is an owner call, listed in the gaps file.

**E03-3 — seed: SPEC precedence, adapter required (non-blocking).** STDN sets **no random seed
anywhere**: frame choice uses `random.randint` and flip uses `np.random.rand` inside a `tf.py_func`
(`model/dataset.py:106,116,127,135`), and hard-sample mixing uses unseeded `tf.random.uniform`
(`train.py:63–69`), with no `tf.set_random_seed`. Resolved by SPEC precedence — see §K, which also
records that a global seed alone is **not** sufficient here. **Not implemented in M6A1.**

---

### E04 — Physics-Guided STD · paper only (arXiv:2012.05185)

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| Status | `FAITHFUL_PAPER` (Q-09: no official release located as of 2026-09-21) | `SPEC` §8.4 + §E of provenance audit |
| Framework | TensorFlow | `PAPER` | §4.1 “PhySTD is implemented in Tensorflow” |
| Initial LR | 5e-5 | `PAPER` | §4.1 |
| Iterations | 150,000 total | `PAPER` | §4.1 |
| LR schedule | ÷10 every 45,000 iterations | `PAPER` | §4.1 |
| Batch size | 8 | `PAPER` | §4.1 |
| Weight init | normal `[0, 0.02]` | `PAPER` | §4.1 |
| Training loss weights | `{α₁…α₆} = {100, 5, 1, 1e-4, 10, 1}` | `PAPER` | §4.1 |
| Inpainting threshold | `β = 0.1` | `PAPER` | §4.1 |
| Loss composition | `L = α₁L_depth + α₂L_G + α₃L_P + α₄L_R` (Eq. 23); `L = α₅L_S + α₆L_H` (Eq. 24) | `PAPER` | Eqs. 23–24 |
| Trace regularisation | `λ = 1` in Eq. 6 | `PAPER` | §3 “Based on Eqn. 6 with λ = 1” |
| Scoring weight `α₀` | **`UNKNOWN`** — “α₀ is empirically determined from the training or validation set” | `PAPER` | §4.1 |
| Required geometry | **140 landmarks + 3DMM fitting**, used to render the depth map | `PAPER` | §4.1 “We use the open-source face alignment [70] and 3DMM fitting [60] to crop the face and provide 140 landmarks”; §3 “alignment [60] to estimate the 3D shape and render the depth” |
| Supervision | binary/weak Live/Spoof; **no** attack-type labels | `SPEC` | §8.4 |

**E04-1 — `α₀` is NON_BLOCKING · OWNER-CONFIRMED 2026-09-21.** `α₀` appears only in Eq. 10, the
**test-time real-vs-spoof scoring** formula `score = ½K⁻²‖M‖₁ + (α₀/2N²)(‖B‖₁+‖C‖₁+‖T‖₁+‖P‖₁)`,
described as “the weight for the spoof trace”. GPAT-TransferBench uses E04 **only as a synthetic
generator** and scores with the benchmark's own frozen downstream evaluators, so Eq. 10 is never
evaluated.

`decision_taken: OWNER_CONFIRMED_OUT_OF_SCOPE_FOR_GENERATOR` ·
`requires_owner_confirmation: false` · `owner_confirmation_date: 2026-09-21`.

**This does not resolve E04-2.**

**E04-2 — BLOCKER: the depth supervision target cannot be produced from the frozen geometry cache.**
`L_depth` carries `α₁ = 100`, the **largest** weight in Eq. 23, and its target is a depth map rendered
from a 3DMM fit driven by **140** landmarks. The frozen M2 cache provides **68** landmarks, no 3DMM
fit and no depth map, and the two cited tools (`[60]` Dense Face Alignment, `[70]` Bulat &
Tzimiropoulos) are external dependencies named in neither the spec nor the registry. Per §8.4 —
“If a tensor dimension, architecture block, loss coefficient or optimization setting that changes the
scientific method is missing, mark `BLOCKED_BY_SOURCE_GAP` instead of choosing one” — **E04 is
BLOCKED**. Dropping `L_depth`, substituting a 68-point fit, or pinning a third-party 3DMM are all
method-changing and all require an owner decision.

---

### E05 — PCGAN · paper only (arXiv:2604.09018)

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| Status | `FAITHFUL_PAPER` (Q-08: no official release located as of 2026-09-21) | `SPEC` §8.5 + §D of provenance audit |
| Optimizer | Adam, betas `(0.9, 0.999)` | `PAPER` | §“Detailed Experimental Configuration” |
| Learning rate | 1e-6 initial | `PAPER` | ibid. |
| Batch size | 1 | `PAPER` | ibid. |
| Iterations | 4,000 | `PAPER` | ibid. |
| Face crop | dataset-provided box or MTCNN, padding 0.6 | `PAPER` | ibid. |
| Hyper-parameters | `α = 0.2`, `β = 1e-6` | `PAPER` | ibid. |
| Total loss | `L_PCGAN = L_rec + L_recblur + L_advrec + L_advmix + L_pat` — **all unit-weighted** | `PAPER` | Eq. 5 |
| **Input resolution** | **`x ∈ R^{3×1024×1024}`** | `PAPER` | §3.1.1 “Encoder” |
| Latents | `z_pat ∈ R^{8×512×512}`, `z_con ∈ R^8` | `PAPER` | §3.1.1 |
| Encoder | swapping auto-encoder [34] with `netE_num_downsampling_sp` reduced **4 → 1** | `PAPER` | §3.1.1 |
| Generator conditioning | `z_pat` injected into every CNN block by AdaIN [35]; `z_con` fed forward | `PAPER` | §3.1.1 |
| Discriminators | discriminator of [35] + an additional patch discriminator | `PAPER` | §3.1.1 |
| Blur loss | downsample **1024 → 512** | `PAPER` | §3.1.2 Eq. 3 |
| Required output | full **256×256** RGB synthetic image | `SPEC` | §8.5 |
| Supervision | pooled TRAIN live/attack labels; no attack-type supervision; no GPAT identity/landmark losses in the main row | `SPEC` | §8.5 |
| `[34]`, `[35]` upstream repos | **`UNKNOWN`** | — | named only as citations; neither the spec nor the registry pins them |

**E05-1 — BLOCKER: irreconcilable resolution conflict.** PCGAN is defined end-to-end at 1024×1024:
the encoder signature, the `8×512×512` artifact latent, the single-downsampling design and the
1024→512 blur loss are all tied to it. The benchmark's canonical face is **256×256** and §8.5 requires
a 256×256 output. Running the method at 256 changes the latent geometry to `8×128×128` and
redefines Eq. 3 — an architecture and loss change. Upsampling 256 → 1024 is worse: the method's
entire premise is preserving *fine-grained high-frequency artifact patterns*, and interpolation from
a 256 crop fabricates exactly that band. There is no faithful path at the benchmark's resolution
without an owner override. **No resolution chosen.**

**E05-2 — BLOCKER: two unpinned upstream architectures.** The encoder/generator come from “[34]”
(swapping auto-encoder) and the discriminator + AdaIN from “[35]”. A FAITHFUL_PAPER build requires
pinning both at exact commits; neither is named in spec §30 nor in `third_party/registry.yaml`.

---

### E06c — DSDG-BIN-IDFREE (Track A) · pin FaceX-Zoo `16b793a7…`

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| Pair relation | frozen common relation from `manifests/pairs_train_v1.parquet`, replacing the official same-subject online pick | `M4_FROZEN_ADAPTATION` | Amendment A1 / DEV-020; `configs/frozen/dsdg_bin_idfree_v1.yaml` |
| `lambda_pair` | **0** | `M4_FROZEN_ADAPTATION` | DEV-020 (official pair loss forces the two reconstructions' identity features to match — invalid across different people) |
| Spoof-type target | all collapsed to one class | `SPEC` §5.1 + `M4_FROZEN_ADAPTATION` |
| `subject_id_global` | never consumed | `M4_FROZEN_ADAPTATION` | Amendment A1 |
| Datasets | CASIA + MSU + SiW | `M4_FROZEN_ADAPTATION` | Amendment A1 |
| Optimizer | `Adam(netE_nir + netE_vis + netG, lr)` — `netCls` deliberately **not** in the optimizer | `OFFICIAL_CODE` | `train_generator.py` |
| Loss assembly | per-term λ folded in, then unweighted sum: `loss = rec + kl + mmd + ip + pair + cls + ort` | `OFFICIAL_CODE` | `train_generator.py:178` |
| Warm-up | first 2 epochs: `rec + 0.01·(kl+mmd+ip+pair+cls+ort)` | `OFFICIAL_CODE` | `train_generator.py:174–176` |
| Identity-preserving net | LightCNN-29 v2, frozen, on 128×128 grayscale | `OFFICIAL_CODE` | `train_generator.py:72–83,108,154–162` |
| **Effective execution values** | see **E06c-1** below — uniquely resolved by README → shell → argparse | `OFFICIAL_CODE` | `README.md`, `train_generator.sh`, `train_generator.py:20–43` |
| `ip_model` path | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` | `OFFICIAL_CODE` | `train_generator.sh`; README gives the download link and this exact directory |

**E06c-1 — RESOLVED_BY_OFFICIAL_EXECUTION_PATH.** The v1 audit treated the argparse defaults and
`train_generator.sh` as two equally authoritative sources and stopped. Re-reading the pinned README
shows they are **not** peers. `addition_module/DSDG/README.md`, Training → DSDG, bullet 3, states
verbatim:

> Run [./train_generator.sh](…/train_generator.sh) to train the generator.

`train_generator.sh` is therefore the **documented official training entrypoint**, and the precedence
chain **README → shell script → argparse default** is explicit. The effective values are unique: a
value the shell passes comes from the shell, and a value it does not pass falls through to the parser.
Of the 21 parser arguments the script passes 19; the two it does not are `--lr` and `--print_freq`.
**`lr` is therefore resolved at the parser default `2e-4`, not unresolved** — the script simply never
overrides it.

| Field | Effective value | Source |
|-------|-----------------|--------|
| `lr` | **2e-4** | argparse fall-through (`train_generator.py:22`) — not passed by the script |
| `print_freq` | 20 | argparse fall-through (`train_generator.py:30`) |
| `batch_size` | 240 | `train_generator.sh` |
| `all_epochs` | 200 | `train_generator.sh` |
| `hdim` | 128 | `train_generator.sh` |
| `pre_epoch` | 0 | `train_generator.sh` |
| `workers` | 8 | `train_generator.sh` |
| `test_epoch` / `save_epoch` | 10 / 10 | `train_generator.sh` |
| `attack_type` | 4 (OULU-NPU) | `train_generator.sh` |
| `λ_mmd` | 50 | `train_generator.sh` |
| `λ_ip` | 1000 | `train_generator.sh` |
| `λ_pair` | 5 | `train_generator.sh` |
| `λ_type` | 10 | `train_generator.sh` |
| `λ_ort` | 1 | `train_generator.sh` |
| `ip_model` | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` | `train_generator.sh` |

Applying the **already-frozen DEV-020** overrides — which are not reopened — gives the Track-A
contract:

| Field | Track-A value | Source |
|-------|---------------|--------|
| `λ_pair` | **0.0** (official 5) | `configs/frozen/dsdg_bin_idfree_v1.yaml` `losses.lambda_pair` |
| `attack_type` | **1** (official 4) | `configs/frozen/dsdg_bin_idfree_v1.yaml` `binary_supervision.attack_type_arg` |
| `lr`, `batch_size`, `all_epochs`, `hdim`, `λ_mmd`, `λ_ip`, `λ_type`, `λ_ort`, `workers`, `test_epoch`, `save_epoch`, `pre_epoch` | as the official table above | official execution path |

The frozen DEV-020 config states that M6 training execution settings “are deliberately NOT frozen
here”, so the official execution path is exactly what supplies them; the two files are complementary,
not in conflict.

**Batch size — effective vs. physical.** Appendix B distinguishes three separate rules, and
`batch_size` falls under the *first*, not the second:

| Appendix B row | Allowed? | Rule |
|---|---|---|
| Third-party native hyperparameters | **No arbitrary changes** | Official source-of-truth; deviations logged |
| GPU physical batch due OOM | Yes, **implementation-only** | Keep effective batch by grad accumulation; record |
| DataLoader workers | Yes | May adapt to OS; no scientific impact |

Therefore, for E06c:

- **Official effective batch size: 240** — source: official `train_generator.sh`. This is a
  third-party native hyperparameter and is **NOT** implementation-only.
- **Official GPU list:** `0,1,2,3`.
- **DataLoader workers:** `8` is the official value, but it is implementation-adaptable under
  Appendix B.
- **Physical batch on the benchmark GPU** MAY be reduced **only if required by memory constraints**.
- **If the physical batch is reduced**, gradient accumulation **MUST** preserve
  **effective batch = 240**, and the implementation-only deviation **MUST** be logged.

> **Correction.** This supersedes the earlier wording “batch 240 (4-GPU) and workers 8 are
> implementation-only under §29”. (That citation was also wrong: the change-control table is
> **Appendix B**, not §29.) Only a physical-batch reduction forced by OOM is
> implementation-only; the *effective* batch 240 is a native hyperparameter. **No GPU execution
> decision is made in M6A1.**

Nothing here was resolved by preference.

**E06c-2 — BLOCKER: `REQUIRED_OFFICIAL_EXTERNAL_WEIGHT_NOT_FETCHED`.** The v1 audit called this
weight “unavailable”; that was wrong. The pinned README supplies both an official download link and
the expected local path, so the weight **is** externally available and has simply not been fetched,
because M6A1 forbids downloading model weights.

| Property | Value |
|----------|-------|
| Official link provenance | `addition_module/DSDG/README.md`, Training → DSDG bullet 1: “Download the LightCNN-29 model from this [link](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view) and put it to `./ip_checkpoint`.” |
| Expected filename / path | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` (as passed by `train_generator.sh`) |
| Required role | `netIP` is built and loaded unconditionally (`train_generator.py:72–83`), frozen (line 83), set to `eval` (line 108), and computes `loss_ip` with **λ_ip = 1000** on the official path; `loss_ip` is **retained** by DEV-020 |
| Local download status | **NOT_FETCHED** |
| Local sha256 | **UNKNOWN** — M6A1 forbids weight download |

It **remains an execution blocker** until a later owner-authorized weight acquisition and
verification step. Approving a substitute identity network instead would change the
identity-preservation objective and is a scientific decision.

---

### E07c — DIFFFAS-BIN-IDFREE (Track A) · pin `murphytju/DiffFAS@23f40519…`

| Field | Value | Evidence class | Exact source |
|-------|-------|----------------|--------------|
| `use_pair` | **false** — official unpaired code path | `M4_FROZEN_ADAPTATION` | DEV-021; `configs/frozen/difffas_bin_idfree_v1.yaml` |
| Conditioning | `style_spoof` only; target is the ε of GT | `M4_FROZEN_ADAPTATION` | DEV-021 |
| Style guide draw | deterministic raw-byte SHA-256 over the binary TRAIN spoof pool of the same dataset | `M4_FROZEN_ADAPTATION` | DEV-021 |
| `style_id` | `SPOOF_BINARY`; `attack_raw` never a class target | `M4_FROZEN_ADAPTATION` | DEV-021 |
| `subject_id_global` | never consumed | `M4_FROZEN_ADAPTATION` | Amendment A1 |
| Datasets | CASIA + MSU + SiW | `M4_FROZEN_ADAPTATION` | Amendment A1 |
| Image size | 256×256, `Normalize([0.5]*3,[0.5]*3)` | `OFFICIAL_CODE` | `FAS_train.py:184–188` |
| Diffusion schedule | linear β, `n_timestep 1000`, `linear_start 1e-4`, `linear_end 2e-2` | `OFFICIAL_CODE` | `config/diffusion.conf` |
| Optimizer | AdamW, lr 1e-5 | `OFFICIAL_CODE` | `config/diffusion.conf` |
| Scheduler | cycle, lr 1e-5, `n_iter 2,400,000`, warmup 5000, decay `[linear, flat]` | `OFFICIAL_CODE` | `config/diffusion.conf` |
| EMA | decay 0.9999 after warmup, 0 before | `OFFICIAL_CODE` | `FAS_train.py:21–26,83` |
| Argparse defaults | `cond_scale 2, guidance_prob 0.2, DDIM_skip 10, max_epochs 400, means_size 5, var_size 3, batch_size 4, sample_algorithm ddpm, sample_initial_noise 250` | `OFFICIAL_CODE` | `FAS_train.py:255–266` |
| Effective seed | **1** | `OFFICIAL_CODE` | `FAS_train.py:180` `seed_torch(seed=1)` |
| cuDNN | `benchmark=False, deterministic=True` | `OFFICIAL_CODE` | `FAS_train.py:174–175` |
| Conditioning encoder | pretrained classifier via `--pretrain_classifier` (default `./PADISI.pkl`) | `OFFICIAL_CODE` | `FAS_train.py:34,254`; `models/unet_autoenc.py:73–82` |
| Encoder weights | **`UNKNOWN`** — not in the repository, must not be downloaded | — | see §C of the provenance audit |

**E07c-1 — official defect, superseded by the SPEC seed contract (non-blocking).** `--random_seed`
(default 2024) is **never read**; `main()` unconditionally calls `seed_torch(seed=1)`
(`FAS_train.py:180`), so the effective *official* seed is **1** and passing `--random_seed` has no
effect.

> **v1 statement WITHDRAWN.** The v1 audit said “Any M6 config must record 1, not 2024.” That is
> wrong: the frozen spec mandates seeds `[42, 1337, 2026]` and marks training seeds **not**
> changeable, so neither 1 nor 2024 may be recorded as an M6 seed. `1` is recorded only as the
> official code's hard-coded behaviour. See §K.

**E07c-2 — CONFLICT (non-blocking, resolved by execution trace).** `config/diffusion.conf` declares
`dataloader.batch_size 8`, `num_workers 2`, `drop_last true`, but `FAS_train.py:205` builds the
loader as `DataLoader(dataset, batch_size=args.batch_size, shuffle=True)` — using the **argparse**
`batch_size` (default **4**) and ignoring the conf's `num_workers` and `drop_last` entirely. The
effective official values are `batch_size = 4`, default workers, `drop_last = False`. The conf's
dataloader block is dead configuration. Recorded so a later reader does not "restore" it.

**E07c-3 — CONFLICT (non-blocking).** `FAS_sample.py` defaults disagree with `FAS_train.py`:
`use_pair` **False** vs True, `sample_algorithm` **ddim** vs ddpm, `random_seed` **1029** vs 2024.
Inference-time defaults differ from training-time defaults in the same repository. For E07c the
training value is fixed by DEV-021 (`use_pair=false`); the **sampling algorithm for generation is
`UNKNOWN`** and must be an owner decision, since ddpm and ddim produce different images from the
same weights.

**E07c-4 — BLOCKER: pretrained conditioning encoder required.** See §C of the provenance audit. The
dependency is unconditional and independent of `use_pair`.

---

## G. READY / BLOCKED per Track-A baseline

| ID | Method | Status | Why |
|----|--------|--------|-----|
| E01 | FAS-Aug | **BLOCKED** | E01-1 spec-vs-official label semantics; E01-2 `level=0` identity outputs; E01-3 unsorted-`os.listdir` nondeterminism |
| E02 | Frequency Substitution | **BLOCKED** | E02-1…E02-4. Algorithm source is the frozen SPEC, but four execution-affecting spec-silent choices remain unresolved — the contract is **not** complete |
| E03 | STDN | **BLOCKED** | **E03-2r** only (iBUG-68 flip-permutation assumption). E03-1, E03-2 resolved; E03-3 resolved by SPEC precedence (adapter required); **checkpoint selection `SOURCE_RESOLVED`** |
| E04 | Physics-Guided STD | **BLOCKED** | E04-2 depth/3DMM supervision (α₁=100) not producible from the frozen 68-point cache; **E04-3 `CHECKPOINT_SELECTION_UNSPECIFIED`**. E04-1 (α₀) is NON_BLOCKING, owner-confirmed 2026-09-21 |
| E05 | PCGAN | **BLOCKED** | E05-1 1024×1024 method vs 256×256 benchmark; E05-2 upstream `[34]`/`[35]` unpinned; **E05-3 `CHECKPOINT_SELECTION_UNSPECIFIED`** |
| E06c | DSDG-BIN-IDFREE | **BLOCKED** | **E06c-2** only — LightCNN-29 v2 `NOT_FETCHED` (officially linked). E06c-1 resolved by the official execution path; seed resolved by SPEC precedence (adapter required); **checkpoint selection `SOURCE_RESOLVED`** |
| E07c | DIFFFAS-BIN-IDFREE | **BLOCKED** | E07c-3 generation sampler undecided; E07c-4 PADISI conditioning encoder not fetched (no official link); **E07c-5 `CHECKPOINT_SELECTION_UNSPECIFIED`**. E07c-1, E07c-2 resolved |

**Zero methods are READY.** No `configs/methods/*.yaml` was created, per M6A1: “If even one scientific
field remains unresolved, keep that method config absent and document the blocker.”

E03 and E06c now carry a **single** residual blocker each — `E03-2r` (a geometry-layout question that
disabling the flip augmentation would settle) and `E06c-2` (an officially linked weight that has not
been fetched). Neither is a missing scientific value, and both methods' checkpoint selection is
`SOURCE_RESOLVED`. **E02 is not in that group:** its algorithm source is the frozen SPEC, but four
execution-affecting spec-silent choices remain unresolved.

The checkpoint-selection audit (§M) **added three blockers** — `E04-3`, `E05-3`, `E07c-5` — so the
blocking-gap count rose from 14 to **17**. E04, E05 and E07c each lack any source that names the
generator state used for bank generation, and M6A1 did **not** choose final, best or latest for any
of them.

## H. Track-B secondary notes

Amendment A1 moved E06b (DSDG-NATIVE) and E07b (DiffFAS-NATIVE) to the secondary native/full track,
restricted to **CASIA + MSU**; SiW-Mv2 remains `NOT_INSTANTIABLE_MISSING_SUBJECT_ID`. Nothing in
M6A1 changes that: **no identity was fabricated for SiW**, no native pair manifest was created, and
`manifests/dsdg_identity_pairs_v1.parquet` and `manifests/difffas_recon_pairs_v1.parquet` remain
absent. Track B stays deferred; E06b/E07b inherit every blocker of E06c/E07c above plus the native
attack-type supervision requirements of §8.6/§8.7. E01's Track A and Track B are identical by §8.1,
so E01's blockers apply to both.

## I. Scope statement

During M6A1 the following **did not occur**:

- **No training.** No training loop, no `optimizer.step()`, no gradient computation.
- **No synthetic generation.** No diffusion sampling, no augmentation applied to any dataset image,
  no synthetic bank written.
- **No TEST access.** The TEST split was neither read nor referenced by any code or artifact created
  here.
- **No checkpoint created.**
- **No external/separate model-weight download was performed.** Four STDN repository-embedded
  checkpoint files were fetched as part of the source checkout, inventoried, **never used**, and
  removed from the source-only working tree; their exact sha256 and git blob IDs are preserved in
  `third_party/source_pins.json`. Two further weight *dependencies* (E06c LightCNN-29 v2, E07c PADISI
  encoder) were identified and deliberately left **NOT_FETCHED**.
- **No commit, no push.**
- **No frozen artifact modified.** The spec, Amendment A1, and the frozen M2/M3/M4/M5 artifacts are
  byte-unchanged; no DSDG or DiffFAS source file was altered.

---

## M. Checkpoint-selection audit (all learned baselines)

Evidence priority: `FROZEN SPEC` → `M4 FROZEN ADAPTATION` → `OFFICIAL CODE` → `PAPER` → `UNKNOWN`.

### M.0 What the frozen specification does and does not fix

- **TEST may never be used for checkpoint selection**, and **TEST never replaces VAL**.
- Required training seeds are `[42, 1337, 2026]`.
- The global pseudocode passes VAL to every method:
  `train_method(method, track, split.TRAIN, validate=split.VAL, seed=seed)` (spec line 900).
- **§10.6 “Generator checkpoint selection”** fixes an explicit VAL `SelectionScore` — EMA epochs 10–60
  on `val_pairs_v1.parquet`, `0.25·rank(ID) + 0.15·rank(Dice) + 0.25·rank(ArtSim) + 0.20·rank(−NME) +
  0.15·rank(−LFErr)` — **for GPAT only**. It is **not** applied to baselines.
- **§14's** “minimum VAL video-level ACER, tie-break maximum VAL video-level AUC” governs the
  **downstream evaluators**, not generators.
- **No baseline generator selection rule exists in the spec.** A baseline is therefore resolved only
  if its *own* source fixes one; otherwise the field is a blocking gap.

### M.1 Summary

| ID | Status | Basis |
|----|--------|-------|
| E01 FAS-Aug | `NOT_APPLICABLE` | no trained generator checkpoint |
| E02 Freq. Substitution | `NOT_APPLICABLE` | no trained generator checkpoint |
| E03 STDN | **`SOURCE_RESOLVED`** | `OFFICIAL_CODE` — final/latest, `ckpt-50` |
| E04 Physics-Guided STD | **`CHECKPOINT_SELECTION_UNSPECIFIED`** (BLOCKING, **E04-3**) | `UNKNOWN` |
| E05 PCGAN | **`CHECKPOINT_SELECTION_UNSPECIFIED`** (BLOCKING, **E05-3**) | `UNKNOWN` |
| E06c DSDG-BIN-IDFREE | **`SOURCE_RESOLVED`** | `OFFICIAL_CODE` — `netG` epoch 200 (final) |
| E07c DIFFFAS-BIN-IDFREE | **`CHECKPOINT_SELECTION_UNSPECIFIED`** (BLOCKING, **E07c-5**) | `UNKNOWN` |

**No checkpoint was chosen in M6A1** for any unresolved method.

### M.2 E03 — STDN · `SOURCE_RESOLVED`

| Question | Answer |
|---|---|
| **a.** checkpoints saved | full graph via `tf.train.Saver(max_to_keep=50)` → `ckpt-1` … `ckpt-50`, all retained |
| **b.** when | **every epoch, unconditionally** — `train.py:186`, `global_step=epoch+1`; `MAX_EPOCH=50` |
| **c.** official selection | **YES.** `test.py:61–63` restores `tf.train.get_checkpoint_state(LOG_DIR).model_checkpoint_path`, which is the **latest** checkpoint by TensorFlow's checkpoint-state semantics |
| **d.** final explicit or inferred | **EXPLICIT** — it is a code path, not an inference; corroborated by the repository's own released weights being `ckpt-50` = `MAX_EPOCH` (inventoried, never used, removed in M6A1) |
| **e.** official validation exists | A VAL loop exists (`STEPS_PER_EPOCH_VAL=500`, `train.py:188–195`) but it **only logs** `g_loss`/`d_loss`/`a_loss`. It runs *after* the save, does not gate saving, and tracks no best checkpoint |
| **f.** GPAT VAL role | VAL is supplied per spec line 900 and may be logged, but does **not** select the checkpoint, because the official source already fixes selection. **TEST is never involved** |
| **g.** unique source-grounded checkpoint | **YES** — final/latest, `ckpt-50` (epoch 50 = `MAX_EPOCH`) |
| **h.** could two choices differ | **NO** |

### M.3 E04 — Physics-Guided STD · **BLOCKING (E04-3)**

| Question | Answer |
|---|---|
| **a. / b.** saved, when | **UNKNOWN** — no official code located (Q-09); the paper states no save schedule |
| **c.** official selection | **NO** |
| **d.** final explicit or inferred | **MERELY INFERRED.** The paper gives only a *budget*: “We train in total 150,000 iterations with a batch size of 8.” The nearest phrase, “We first train the PhySTD till convergence”, concerns a downstream trace-classification experiment, not checkpointing |
| **e.** official validation | Not for selection — the paper mentions a validation set only for `α₀`, which is scoring-only and owner-confirmed out of scope |
| **f.** GPAT VAL role | Available per spec line 900, but the spec fixes no baseline rule, so VAL alone does not determine it. **TEST may never be used** |
| **g.** unique checkpoint | **NO** |
| **h.** could two choices differ | **YES** — iteration 150,000 vs a best-by-metric state give different banks |

**E03 checkpoint semantics were not borrowed**: the E04 paper does not adopt them.

### M.4 E05 — PCGAN · **BLOCKING (E05-3)**

| Question | Answer |
|---|---|
| **a. / b.** saved, when | **UNKNOWN** — no official code located (Q-08); the paper states no save schedule |
| **c.** official selection | **NO** |
| **d.** final explicit or inferred | **MERELY INFERRED** — “Adam optimizer for 4000 iterations” is a budget only; the paper never says which generator state synthesises the images |
| **e.** official validation | The protocol reports “the average performance over the last 10 epochs” and a “best epoch” gap, but those describe **detector** evaluation (Tabs. 2–3), not generator selection |
| **f.** GPAT VAL role | Available per spec line 900; the spec fixes no baseline rule. **TEST may never be used** |
| **g.** unique checkpoint | **NO** |
| **h.** could two choices differ | **YES** |

**Additional incompatibility:** an “average over the last 10 epochs” protocol has *no single
checkpoint at all*, which is structurally incompatible with the benchmark's one-bank-per-seed
contract.

### M.5 E06c — DSDG-BIN-IDFREE · `SOURCE_RESOLVED`

| Question | Answer |
|---|---|
| **a.** checkpoints saved | three files per save point — `netE_spoof_`, `netE_live_`, `netG_` (`train_generator.py:214–217`) |
| **b.** when | epoch 1 and every `save_epoch`; under the official entrypoint `save_epoch=10`, `all_epochs=200` → epochs 1, 10, 20, …, **200** |
| **c.** official selection | **YES.** `generated.py:23` defaults `--pre_model` to `./result/model_oulu_p1/netG_model_epoch_200_iter_0.pth`. `epoch_200` equals `all_epochs=200` from `train_generator.sh`, and `model_oulu_p1` is that script's `checkpoint_path` — so the official *generation* entrypoint names the **final-epoch** generator |
| **d.** final explicit or inferred | **EXPLICIT** — the filename literally encodes epoch 200 = `all_epochs` |
| **e.** official validation exists | **NO.** The `test` block (`train_generator.py:198–211`) only writes sample images with `vutils.save_image`. It computes **no held-out validation metric**, so it is **not** validation and is not treated as such |
| **f.** GPAT VAL role | VAL is supplied per spec line 900 but does not select the checkpoint, since the source fixes it. **TEST is never involved** |
| **g.** unique source-grounded checkpoint | **YES** — `netG` at epoch 200 (final); the generator only |
| **h.** could two choices differ | **NO** |

### M.6 E07c — DIFFFAS-BIN-IDFREE · **BLOCKING (E07c-5)**

| Question | Answer |
|---|---|
| **a.** checkpoints saved | one dict per save point holding `model`, `ema`, `scheduler`, `optimizer`, `conf` (`FAS_train.py:102–114`) |
| **b.** when | every `save_checkpoints_every_iters` = **10,000 iterations**, as `model_{iters:06d}.pt`; `max_epochs=400`, so the count depends on dataset size |
| **c.** official selection | **NO.** `FAS_sample.py:112` declares `--model_path` as `required=True` with **no default**, and the repository README is three lines with no checkpoint guidance |
| **d.** final explicit or inferred | **NEITHER** — the source declines to name one and requires the user to supply a path |
| **e.** official validation exists | **NO.** The `save_images_every_iters` block (`FAS_train.py:117–163`) only writes a sample grid; no held-out metric, no best-checkpoint tracking |
| **f.** GPAT VAL role | Available per spec line 900; the spec fixes no baseline rule. **TEST may never be used** |
| **g.** unique checkpoint | **NO** |
| **h.** could two choices differ | **YES** — different iteration checkpoints give different banks |

**Resolved sub-detail:** *which tensor set* to load **is** fixed by the source — `FAS_sample.py:46`
loads `torch.load(model_path)["model"]`, not `["ema"]`, although both are saved. Only **which
iteration** is unspecified.

---

## K. Seed-contract re-audit — SPEC precedence over official execution defaults

### K.0 What the frozen specification requires

> “All self-run results are reported over seeds **[42, 1337, 2026]** as mean ± standard deviation;
> test thresholds come from VAL only.” — spec line 16

Appendix B (“What Claude is allowed to tune vs. not allowed to tune”) settles precedence explicitly: **`Training seeds | No | 42/1337/2026`** —
i.e. training seeds are **not** changeable. The same table says *“Third-party native hyperparameters |
No arbitrary changes | Official source-of-truth”*, but seeds are mandated separately, so **the SPEC
takes precedence over any hard-coded or missing seed in official code.**

This is classified as **SPEC precedence over official execution defaults**, not as a silently chosen
hyperparameter. **No adapter was implemented in M6A1.**

Two adjacent Appendix B rows — *“GPU physical batch due OOM | Yes implementation-only”* and
*“DataLoader workers | Yes … no scientific impact”* — are applied to E06c in **E06c-1** above, where
the effective batch 240 is classified as a native hyperparameter rather than implementation-only.

### K.1 E03 — STDN

| | |
|---|---|
| **Official behaviour** | No seed surface at all: unseeded `random.randint` / `np.random.rand` inside `tf.py_func` (`model/dataset.py:106,116,127,135`), unseeded `tf.random.uniform` (`train.py:63–69`), no `tf.set_random_seed`. |
| **Aggravating factor** | `model/dataset.py:65,67` use `dataset.shuffle(...).map(num_parallel_calls=AUTOTUNE)`, so parallel `py_func` calls consume the **shared** Python/NumPy global RNG in a nondeterministic order. **A global seed alone is not sufficient.** |
| **SPEC requirement** | each required M6 run uses its benchmark seed from `[42, 1337, 2026]` |
| **Required future adapter** | propagate the benchmark seed into Python `random`, NumPy and the TensorFlow graph-level seed, **and** make RNG consumption order deterministic (derive the per-element seed from the sample id, or serialise the `map` stage) |
| **Status** | `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED` |

The model math is unchanged and the spec's three-seed mean ± std reporting is what absorbs stream
choice, so this is a design obligation rather than an unresolved scientific value. It does not affect
**E03-2r**, which remains E03's only blocker — and note that disabling the flip augmentation would
settle E03-2r *and* remove one of the two Python-level RNG consumers at once.

### K.2 E06c — DSDG-BIN-IDFREE

| | |
|---|---|
| **Official behaviour** | **No seeding of any kind** in `train_generator.py`, `generated.py` or `data/generation_dataset.py` — no `torch.manual_seed`, no `random.seed`, no `np.random.seed`. `cudnn.benchmark = True` (`train_generator.py:57`, `generated.py:36`) selects algorithms nondeterministically. `DataLoader(shuffle=True, num_workers=8)` has no `worker_init_fn` and no explicit generator. Latent draws use `torch.zeros(...).normal_(0,1)` (`train_generator.py:199–200`, `generated.py:47–48`). |
| **Note** | Pinning the source does **not** make this reproducible. |
| **SPEC requirement** | each required M6 run uses its benchmark seed from `[42, 1337, 2026]` |
| **Required future adapter** | seed Python `random`, NumPy, torch CPU and CUDA; set `cudnn.benchmark = False` and `cudnn.deterministic = True`; add a `worker_init_fn` and an explicit `generator`. The repository already implements exactly this pattern in `gpatbench/probe/train.py` (`seed_everything`, `apply_determinism`, `worker_init_fn`). |
| **Status** | `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED` |

Does not affect **E06c-2**, which remains E06c's only blocker.

### K.3 E07c — DIFFFAS-BIN-IDFREE

| | |
|---|---|
| **Official behaviour** | `FAS_train.py:180` calls `seed_torch(seed=1)` unconditionally; `--random_seed` (default 2024) is never read. `seed_torch` (`FAS_train.py:167–175`) **does** seed Python `random`, `PYTHONHASHSEED`, NumPy, torch and CUDA and sets `cudnn.benchmark=False`, `cudnn.deterministic=True`, so the RNG surface is **complete** — it is simply hard-coded to 1. `DataLoader(shuffle=True)` uses default workers (0). |
| **SPEC requirement** | each required M6 run uses its benchmark seed from `[42, 1337, 2026]` |
| **Required future adapter** | parameterise `seed_torch` so the benchmark seed is propagated instead of the constant 1 |
| **Status** | `RESOLVED_BY_SPEC_PRECEDENCE_BUT_IMPLEMENTATION_ADAPTER_REQUIRED` |

E07c has the healthiest RNG surface of the three — the only defect is the hard-coded constant. Does
not affect **E07c-3** or **E07c-4**.

---

## J. Checks run

All CPU-only and audit-safe. No training loop, optimizer step, diffusion sampling, bank generation,
TEST evaluation or weight download was invoked.

| Check | Result |
|-------|--------|
| `git status --short` | only intentional audit/provenance changes (9 paths, listed in the final report) |
| Frozen artifacts diff (`configs/frozen`, `manifests`, `docs/spec`) | **empty** — byte-unchanged |
| DSDG / DiffFAS source files | unmodified; caches show only the pre-existing weight deletions |
| Weight-extension scan of all four caches | 0 active weight files remaining |
| Source-pin verification (commit, sha256, size, blob, line count) | all cited files **PASS** — see §L |
| `python -m unittest discover -s tests` at HEAD (pristine worktree) | 501 tests, 8 failures, 48 errors, 14 skipped |
| `python -m unittest discover -s tests` after M6A1 (incl. closeout) | 501 tests, **9** failures, 48 errors, 13 skipped |
| Net effect of M6A1 on the suite | **no new failure** |
| `python -m unittest discover -s tests -p 'test_m0_bootstrap.py'` | 1 failure — `test_manifests_are_stage_appropriate`, **pre-existing at HEAD** (see §J.2) |
| `git diff --check -- . ':(exclude)outputs/audit/ARTIFACT_INDEX.csv'` | clean, exit 0 |
| `outputs/audit/third_party_source_pins.sha256` vs HEAD | **byte-identical** (restored) |

### J.1 The 48 errors and 1 apparent new failure are pre-existing, not M6A1's

The **48 errors** are all `ModuleNotFoundError` in this laptop's `.venv`: `scipy` (23),
`zstandard` (23), `torch` (2). They are identical before and after M6A1 and are an environment gap,
not a repository defect. They should be re-checked on the GPU box, where those modules exist.

Pinning E01 and E03 initially broke three guards — `test_artifact_index_hashes`,
`test_no_invented_urls` and `test_third_party_registry`. All three are fixed (see §F.1 of the
provenance audit and the rebuilt `ARTIFACT_INDEX.csv`).

`test_m2b_preflight … test_smoke_shards_pass_the_same_audit` *appears* new only because the baseline
was taken in a fresh `git worktree`, where the **git-ignored** smoke shards
(`outputs/exploratory/m2a_smoke/runC/logit_shards/`) do not exist, so the test skips itself — which
is also why `skipped` went 14 → 13. In the real working tree the shards exist, the test runs, and it
fails because `verify_shard_index` needs `zstandard` (`gpatbench/preprocess/logit_store.py:51,60`).
Proven independent of this milestone: with `third_party/registry.yaml`, `third_party/source_pins.json`
and `tests/test_m0_bootstrap.py` stashed back to HEAD, the test fails identically. M6A1 modified no
file under `gpatbench/` or `outputs/exploratory/`.

### J.2 `test_manifests_are_stage_appropriate` is pre-existing and out of M6A1's scope

The M0 stage guard rejects `manifests/artifact_probe_classes_v1.json`, an **M5** artifact that
`tests/stage_guard.py::allowed_manifests()` does not list. It fails identically in the pristine HEAD
worktree baseline, M6A1 created nothing under `manifests/`, and fixing an M5 stage-guard gap is
outside this audit's scope. Reported rather than silently repaired.

---

## L. Source-pin verification

Every cited file in `third_party/source_pins.json` was re-verified against its git-ignored local cache
on **2026-09-21**, checking **pinned commit, commit tree, sha256, `size_bytes`, git blob ID and line
count**.

| Source | Commit | Tree | Cited files |
|--------|--------|------|-------------|
| `difffas` | OK | OK | 5 / 5 |
| `dsdg` | OK | OK | 6 / 6 |
| `fas_aug` | OK | OK | 6 / 6 |
| `stdn` | OK | OK | 8 / 8 |

**25 cited files across 4 sources — RESULT: PASS.** Active model-weight files remaining in all four
caches: **0**.

**Line-count convention corrected.** The repository counts lines as `len(data.split(b"\n"))`, so a
terminal newline contributes one extra element. The v1 `fas_aug` and `stdn` entries used logical line
counts instead, which made 9 of the 14 new entries mismatch; they now follow the repository
convention and the convention is recorded in each new source entry. The pre-existing `dsdg` and
`difffas` entries were already correct and were not touched.
