# M6A2 — Source-Derivable Blocker Resolution

**Date:** 2026-09-21 · **Branch:** `m6-baselines` · **Starting commit:** `9dc2492750fff98fd2fa8676715eda58df3a9d46`
**Frozen spec sha256:** `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`
**Amendment A1 sha256:** `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472`

Additive to M6A1. **No M6A1 history was rewritten.** Machine-readable companion:
`M6A2_BLOCKER_STATE.json`.

**Scope:** determine which M6A1 blockers are resolvable strictly from frozen SPEC, M4/A1 frozen
adaptation semantics, official pinned source, or official paper / cited upstreams. Anything still
requiring a judgment call is left explicitly unresolved with a minimal decision surface.
**No training, no generation, no TEST access, no checkpoints, no weight downloads, no method configs,
no commit.**

**Resolution precedence used:** FROZEN SPEC → M4/A1 frozen adaptation → explicit official execution
entrypoint → official source defaults → paper → inference. **Inference was never used to close a
blocker.**

---

## 0. Result

| | Before (M6A1) | Resolved in M6A2 | After |
|---|---|---|---|
| Blocking gaps | **17** | **6** | **11** |

| Blocker | New status |
|---|---|
| E01-1 | **RESOLVED_BY_SPEC** |
| E01-2 | **RESOLVED_BY_SPEC** |
| E01-3 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E02-1 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E02-2 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E02-3 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E02-4 | **RESOLVED_BY_SPEC** |
| E03-2r | **RESOLVED_BY_PINNED_UPSTREAM** |
| E04-2 | STILL_BLOCKED_SOURCE_GAP |
| E04-3 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E05-1 | **RESOLVED_BY_PINNED_UPSTREAM** |
| E05-2 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E05-3 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |
| E06c-2 | STILL_BLOCKED_REQUIRED_WEIGHT_NOT_FETCHED |
| E07c-3 | **RESOLVED_BY_OFFICIAL_EXECUTION_PATH** |
| E07c-4 | STILL_BLOCKED_SOURCE_GAP |
| E07c-5 | STILL_BLOCKED_OWNER_DECISION_REQUIRED |

Per-method remaining: **E01** 1 · **E02** 3 · **E03** **0** · **E04** 2 · **E05** 2 · **E06c** 1 · **E07c** 2.

> **E03 (STDN) now has zero remaining source-derivable blockers.** Its config precedence, 68-point
> geometry source, landmark ordering, checkpoint selection and seed contract are all settled.

---

## 1. E01 — FAS-Aug

### E01-1 `changeLabel` — **RESOLVED_BY_SPEC**

The question the task posed was whether `changeLabel` is **(A)** part of the operator transformation
or **(B)** only the authors' training-label policy. **The answer is B, and it is decidable from
source.**

`apply_augment` (`data/FAS_Augmentations.py:204-218`) unpacks `changeLabel` from `augment_dict` and
returns it *alongside* the image. The transformation itself is `res = augment_fn(img.copy(), mag)` —
`augment_fn` **never receives `changeLabel`**, so the produced pixels are bit-identical regardless of
its value. Its only consumers in the entire repository are `data/transform.py:31,35,38,47-49` (an OR
accumulation) and `data/zip_dataset.py:46-48`:

```python
tar = self.face_label if not changeLabel else 0
```

That is a **training-label assignment**, i.e. the original authors' policy for their own classifier.

Frozen §8.1 defines the benchmark's bank label independently: *"Output: one spoof-labeled canonical
256×256 image per intended pair."* Under FROZEN SPEC > official source defaults, **E01-1 is
resolved**. The operator set is unchanged — all 8 operators retained, none removed or modified.

**Mandatory disclosure:** three operators (`Color_Diversity`, `Hand_Trembling`, `Low_Resolution`) are
ones the FAS-Aug authors would label *live* in their own pipeline. Under §8.1 every E01 bank image is
spoof-labeled. This divergence from the authors' label policy must be disclosed with the E01 row.

### E01-2 `level == 0` — **RESOLVED_BY_SPEC**

Complete official magnitude path traced: `config/default.py:78` `NUM_MAG=10` → `data/transform.py:70-72`
`random_parse_policies` draws `mag = random.choice(range(num_mags))` and emits `level = mag/(num_mags-1)`,
so the level grid is `{0, 1/9, …, 1}` → `data/FAS_Augmentations.py:203-204` `if level == 0: return
img.copy(), False`.

`level = 0` is therefore a **genuinely reachable official parameter** (probability 1/10 per draw) on
the intended generation path — not an undefined implementation choice. §8.1 says operator parameters
*"follow the official repository defaults"*, and neither spec nor source bans a no-op draw. The
correct classification is **degenerate but source-correct**, not a scientific contract gap.

**`level = 0` is retained**, because no source or spec text explicitly permits removing it.

**Mandatory disclosure:** ≈10 % of E01 bank images will be byte-identical copies of the live target,
spoof-labeled. This must be measured and disclosed with the E01 row, never silently removed.

### E01-3 asset order — **STILL_BLOCKED_OWNER_DECISION_REQUIRED**

All filesystem-order dependencies traced to `data/fas_aug_helper.py:17-26`, which builds the three
texture name lists with `os.listdir` **at import time** (unsorted, filesystem-dependent); line 50
`getTexture` then picks by index with `random.choice`.

**No canonical official order exists:**

| Candidate source | Result |
|---|---|
| Committed asset list | **None.** `data/data_list/*.csv` are dataset *frame* lists (CASIA/MSU/REPLAY/OULU/SiW), not texture lists |
| Config | **None.** No config references `data/background`, `data/noiseTexture` or `data/MPTexture` |
| Archive metadata | **None.** `.gitattributes` contains only LF normalisation |
| Repository tree | **Exists** — git sorts tree entries bytewise; verified `git ls-tree data/background/` is exactly lexicographic (90 entries; noiseTexture 48, MPTexture 190). **But the official code does not use it.** |
| Frozen spec | **None.** No global ordering rule; the only ordering texts are scoped elsewhere (§6 candidate hash, §6 lexical `sample_id` tie-break, §7.x subsample hash) |

The authors' original filesystem order is unrecoverable, so the official index→asset mapping cannot
be reproduced by any means. Sorting **would** change that mapping and is therefore a real
deterministic adaptation. **Not chosen.** See the decision surface in §7.

---

## 2. E02 — Frequency Substitution

### E02-1 rounding — **STILL_BLOCKED** (proved *not* moot)

Audit-only exact computation on the frozen geometry (fftshifted 256×256; 16×16 non-overlapping
blocks; `r = √((k_x/256)² + (k_y/256)²)` with `k = p − 128`; eligible `r ∈ [0.20, 0.50]`; block
eligible if ≥ 75 % of its 256 pixels are eligible):

```
eligible pixels   = 43,186
ELIGIBLE BLOCKS   = 159   (of 256)
159 × 0.25        = 39.75          159 mod 4 = 3
```

Robust to the boundary convention — closed `[0.20,0.50]`, half-open `[0.20,0.50)` and open
`(0.20,0.50)` all yield **159**. The count is **not** divisible by 4, so E02-1 is **not** moot:
floor → 39 blocks, round/ceil → 40 blocks.

### E02-2 / E02-3 RNG — **STILL_BLOCKED**

An exhaustive grep of the frozen spec for *pair-specific / global_seed / pair seed / rng /
deterministic seed* returns exactly three lines: 148 (`split_seed`), **232** (§8.1) and **240** (§8.2).

§8.1 line 232 defines `seed = SHA256(pair_id + global_seed) mod 2^31` **explicitly and only for
FAS-Aug operator parameters**. §8.2 line 240 says only *"using pair-specific RNG seed"* and fixes
neither the RNG family nor the draw procedure.

**No generic frozen pair-RNG rule exists for §8.2 to inherit.** The pair manifest's own `seed` column
is the **constant** `20260814` for all 8,838 rows (verified in `manifests/pairs_train_v1.parquet`),
not a per-pair seed. M4's frozen adaptation does not cover it either: `configs/frozen/pairs_v1.yaml`
defines digests under namespaces `gpatbench.pair.source_seed.v1` (preimage
`source_sample_id | split_seed`) and `gpatbench.pair.candidate.v1`, both **scoped to pair
construction** and with a different preimage from §8.1's. That explicit namespacing is evidence
*against* silent reuse. Extending §8.1's rule to E02 would be analogy, which this task forbids.

### E02-4 conjugate counterparts — **RESOLVED_BY_SPEC**

**Exact derivation.** For even `N = 256` after `fftshift`, shifted index `p` carries frequency
`k = p − 128`, `k ∈ [−128, 127]`. For a real image `X[k] = conj(X[−k mod N])`, and the shifted index
of `−k` is `(256 − p) mod 256`. Hence:

- **pixel map `p ↔ 256 − p`**
- exactly two **self-conjugate** bins: `p = 0` (`k = −128`, the **Nyquist** bin, since `128 ≡ −128 mod 256`)
  and `p = 128` (`k = 0`, **DC**)

**Block-alignment proof.** Block `b` covers `p ∈ [16b, 16b+16)`; its counterpart set is
`{241 − 16b, …, 256 − 16b}`. Requiring that to be a grid block needs `16m = 241 − 16b`, which is
impossible because **241 is not divisible by 16**. Verified for all 16 blocks: `aligned = False` for
every `b`; the exact counterpart is offset by **exactly one index per axis**. So selecting a 16×16
block **does** require selecting a non-grid-aligned counterpart.

**Why the spec settles it.** The frozen phrase is *"selected blocks **and the conjugate-symmetric
counterparts**"*. "Conjugate-symmetric counterpart" names an exact mathematical object — the image of
the selected pixel set under `k → −k`. It does not assert the counterpart is itself a block, and
nothing in §8.2 requires the replaced region to be a union of grid blocks. The convention is
therefore **uniquely determined as pixel-wise mirroring**.

**Numerical corroboration** (audit-only toy arrays, no dataset):

| Convention | `max|imag|` after `ifft2` | relative to `max|real|` |
|---|---|---|
| pixel-wise counterpart | 8.89 × 10⁻¹⁴ | 3.3 × 10⁻¹⁶ — exactly Hermitian |
| block-level mirror | 5.60 | **2.1 × 10⁻²** |

Only pixel-wise mirroring yields a valid real-image spectrum; under a block mirror the spec's
"real part" instruction would silently discard 2.1 % of signal. **Resolved: pixel-wise `p ↔ 256 − p`,
with `p = 0` and `p = 128` self-conjugate** (a selected self-conjugate bin needs no partner).

---

## 3. E03 — STDN landmark order · **RESOLVED_BY_PINNED_UPSTREAM**

M2 pinned FaceXFormer at `Kartik-3004/facexformer@10fe8291f8a64e2ca1daf938e3e0007bd860303b`
(`M2A_MODEL_PROVENANCE.csv`, `models/registry.yaml`). Re-cloned in M6A2: **HEAD matches exactly**,
tree `67d7da58079000149fabf6c7975c62bbcb4c8527`, 38 files, **zero weight files**.

The pinned repository itself does **not** document the ordering — it is inference-only code,
`visualize_landmarks` draws every point with no index semantics, and there is no flip permutation,
connectivity list or dataset code. So the ordering was traced **upstream to the paper**:

> FaceXFormer, arXiv:2403.12960v3, Appendix **F.2 "Landmarks Detection"**, verbatim:
> *"We utilize the **300W dataset [82]** for the training and evaluation of FaceXFormer. … All images
> are annotated with 68 landmark points."*

Reference **[82]** = *"Christos Sagonas, Georgios Tzimiropoulos, Stefanos Zafeiriou, and Maja Pantic.
300 faces in-the-wild challenge: the first facial landmark localization challenge. ICCVW 2013."* —
i.e. the **iBUG 300-W** benchmark, whose 68-point annotation *is* the iBUG-68 scheme. A regression
head trained on 300-W emits points in the 300-W index order by construction. This is documented
upstream, **not** an inference from the point count.

**The STDN side was verified independently.** `model/dataset.py:96-100` `lm_reverse_list` was parsed
and compared against the canonical iBUG-68 horizontal-flip permutation (jaw 17→1; brows 27→22, 21→18;
nose bridge 28-31; nostrils 36→32; eyes 46,45,44,43,48,47 / 40,39,38,37,42,41; outer mouth 55→49,
60→56; inner mouth 65→61, 68→66):

```
IDENTICAL = True      valid permutation of 1..68 = True      involution = True
```

Both sides are provably the 300-W / iBUG-68 ordering. **Horizontal flip may remain enabled — it was
not disabled to avoid the issue.**

---

## 4. E04 — Physics-Guided STD

### E04-2 geometry — **STILL_BLOCKED_SOURCE_GAP** (narrowed, upstreams identified)

**References resolved exactly:**

- **[60]** = *Y. Liu, A. Jourabloo, W. Ren, X. Liu, "Dense Face Alignment", ICCV Workshops 2017* —
  the same MSU lab as E03/E04. Official repo **`yaojieliu/ICCVW2017-DenseFaceAlignment@01cfb1aa9845c9ce6c0ac5fbbb16454ee771f848`**.
- **[70]** = *A. Bulat and G. Tzimiropoulos, "How far are we from solving the 2D & 3D face alignment
  problem? (and a dataset of 230,000 3D facial landmarks)", ICCV 2017* — `1adrianb/face-alignment`.

**Roles (paper, verbatim):** §4.1 *"We use the open-source face alignment [70] and 3DMM fitting [60]
to crop the face and provide 140 landmarks"*; §3 *"we use [60] to fit a 3DMM model and extract the 2D
locations of Q facial vertices"* (Eq. 13); Eq. 16 *"We apply the dense face alignment [60] to estimate
the 3D shape and render the depth ground truth M⁰"*, `K = 32`.

**The scientific route is still not uniquely determined** — four independent sub-gaps:

1. **License-gated external asset.** [60]'s README requires *"Download the Basel Face Model (BFM)"*
   from `faces.cs.unibas.ch`. BFM is distributed only under an individually signed academic licence;
   it is neither bundled nor freely fetchable, and M6A2 forbids acquisition.
2. **Q = 140 vertex set unspecified.** The paper says only *"We select Q = 140 vertices to cover the
   face region so that they can represent non-rigid deformation"* — no index list, no rule, no
   released file. Different 140-vertex choices give different warps.
3. **Depth rendering unspecified.** How the fitted 3D shape becomes the 32×32 `M⁰` is not described,
   and [60]'s repository performs alignment, not depth rendering.
4. **Reproducibility today.** [60] is MATLAB/MatConvNet *testing* code with compiled MEX binaries and
   no training code; the paper pins no [70] commit.

`L_depth` carries `α₁ = 100`, the dominant term in Eq. 23, and its target cannot be produced at all
without (1) + (3). **FaceXFormer's 68 points were not substituted for the 140** — the paper does not
permit it.

### E04-3 checkpoint — **STILL_BLOCKED_OWNER_DECISION_REQUIRED**

Final paper-level sweep for *checkpoint / best model / we select / converge / final model / early
stop / validation / released model / model zoo / pre-trained model / snapshot / save the model /
last-final iteration / selected model / model selection* returned only three hits, none a selection
rule: the **budget** *"We train in total 150,000 iterations with a batch size of 8"*; *"We first train
the PhySTD till convergence"* (prose about a downstream trace-classification experiment); and *"Source
code and pre-trained models will be publicly available upon publication"* (no URL — Q-09).
**"Final" was not inferred from a training budget.**

---

## 5. E05 — PCGAN

### E05-2 upstreams — **STILL_BLOCKED_OWNER_DECISION_REQUIRED** (substantially narrowed)

**[34] is conclusively identified and pinned by three exact matches — not name similarity:**

| PCGAN statement | `taesungp/swapping-autoencoder-pytorch` |
|---|---|
| *"reduce the parameter `netE_num_downsampling_sp` **from 4** to 1"* | `models/networks/encoder.py:35` — `--netE_num_downsampling_sp, default=4` |
| `z_pat ∈ R^{8×512×512}` (8 channels) | `models/swapping_autoencoder_model.py:12` — `--spatial_code_ch, default=8` |
| [34] = Park, Zhu, Wang, Lu, Shechtman, Efros, Zhang, NeurIPS 2020 | repo: *"Official Implementation of Swapping Autoencoder for Deep Image Manipulation (NeurIPS 2020)"* |

**Pin:** `6baa180f1184ee79a6b967f9d80ee0e02a979ac7`, tree `25a434f1282a0177d6afdcc68ebfcf40bd805621`,
73 files, **zero weights**. It also supplies the discriminators PCGAN names —
`models/networks/discriminator.py:2` imports `Discriminator as OriginalStyleGAN2Discriminator`, and
`models/networks/patch_discriminator.py:97` defines `StyleGAN2PatchDiscriminator`.

**[35]** = *T. Karras, S. Laine, T. Aila, "A style-based generator architecture for GANs", CVPR 2019*
(StyleGAN v1); official repo `NVlabs/stylegan@1e0d5c781384ef12b50ef20a62fee5d78b38e88f`. **Not
pinned**, because its role is ambiguous:

> **Residual conflict — modulation mechanism.** PCGAN states the latents are integrated *"through
> adaptive instance normalization [35]"*. **AdaIN is the StyleGAN-v1 mechanism.** But its declared
> base [34] uses **StyleGAN2 weight demodulation** — `models/networks/stylegan2_layers.py:210,217,257,272`
> (`class ModulatedConv2d`, `demodulate=True`), `generator.py:80,114,150` (`SpatialCodeModulation`),
> and `generator.py:89` states the layers *"borrow heavily from StyleGAN2 code"*. StyleGAN2 explicitly
> **replaced** AdaIN with demodulation, so the paper's prose and its stated base disagree on an
> execution-affecting architectural detail.

Narrowed from *"two unpinned upstream architectures"* to *"one pinned upstream plus one
modulation-mechanism fork"*.

### E05-1 resolution — **RESOLVED_BY_PINNED_UPSTREAM** (M6A1's framing was too strong)

M6A1 called 1024 vs 256 an *"irreconcilable resolution conflict"*. **That was wrong.** The task asked
for an exact split between architectural requirement and experiment setting:

| Aspect | Verdict | Evidence |
|---|---|---|
| Is 1024 mathematically required? | **No** | [34]'s own launchers run the same architecture at **256** (afhq, bedroom, church: `crop_size=256`), **512** (ffhq512) and **1024** (ffhq1024), varying `netE_num_downsampling_sp`. `crop_size` is a CLI option |
| Fully convolutional, admits 256? | **Yes** | `encoder.py` loops `for i in range(self.opt.netE_num_downsampling_sp)` with no absolute spatial constants; the only `512` literal in encoder/generator is `generator.py:143` `min(512, ch)`, which caps **channels** |
| Latent geometry scales deterministically? | **Yes** | `z_pat` extent = input / 2^`sp`. At 1024, `sp=1` → 512×512 (matches the paper). At 256, `sp=1` → **8×128×128** |
| Do losses hard-code 1024/512? | **No** | Eq. 3 is defined as *"Since only a **1/2 downsampling** is applied"* — a **relative** factor. "1024 × 1024 to 512 × 512" is that rule instantiated at the authors' resolution |

**Conclusion: 1024 is an experiment setting, not an architectural requirement.** At 256 the route is
fully determined (`sp=1`, `spatial_code_ch=8` → `z_pat = 8×128×128`, blur = ÷2, 256 → 128) and no
equation or architecture changes. What remains is a **status-tag** decision (FAITHFUL_PAPER vs
CONTROLLED_ADAPTATION), not a missing scientific value. Upsampling 256 → 1024 remains strongly
inadvisable: it fabricates the high-frequency artifact band the method measures.

### E05-3 checkpoint — **STILL_BLOCKED_OWNER_DECISION_REQUIRED**

Final sweep for *released model / model zoo / pre-trained model / we use the final|trained|resulting /
snapshot / save the model / last-final iteration / selected model / model selection* returned **zero
hits**. The only training-state statement is the budget *"Adam optimizer for 4000 iterations"*. The
paper's *"average performance over the last 10 epochs"* and *"best epoch"* (Tabs. 2-3) describe
**detector** evaluation following [9] — they are **not** PCGAN generator checkpoint selection and were
not confused with it; an average over epochs is not a single checkpoint at all. No official code
exists (Q-08), so no code-level convention can settle it.

---

## 6. E06c / E07c

### E06c-2 LightCNN — **STILL_BLOCKED_REQUIRED_WEIGHT_NOT_FETCHED** (scientific identity fully resolved)

| Property | Value |
|---|---|
| Architecture | `network_29layers_v2(resblock, [1,2,3,4])` in `DataParallel` — `networks/__init__.py:22-25`, defined **in the pinned repo** at `networks/light_cnn.py` (`mfm` / `resblock` / `network_29layers_v2`). **No external code dependency.** |
| Upstream paper | Wu, He, Sun, Tan, *"A Light CNN for Deep Face Representation with Noisy Labels"*, IEEE TIFS 2018, 13(11):2884-2896 |
| Upstream repo | `AlfredXiangWu/LightCNN` (HEAD `7b38a6f2d20865b8c008c6d24cf977309af88114`) |
| **Released-file identity** | DSDG README links Drive id `1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS`; the **upstream** LightCNN README's "LightCNN-29 v2" link is `https://drive.google.com/open?id=1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS` — **the same file id.** DSDG points at the **original authors' released checkpoint**, not a re-hosted copy |
| Expected filename / path | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` (as passed by `train_generator.sh`) |
| Load format / compatibility | `train_generator.py:75-81` — `torch.load`, reads `checkpoint['state_dict']`, filters to keys present in the `DataParallel`-wrapped `network_29layers_v2` state dict, loads **non-strictly**; then frozen (line 83) and set to `eval` (line 108) |
| Role | `loss_ip` with `λ_ip = 1000` on the official path; retained by DEV-020 |
| **Official checksum** | **None published** by either repository |
| Local status | **NOT_FETCHED** — M6A2 forbids acquisition; local sha256 **UNKNOWN** |

**Source contract: RESOLVED. Execution: still blocked** pending owner-authorized acquisition and hash
verification. Because no upstream digest exists, the sha256 observed at acquisition would itself
become the authoritative pin and must be recorded then. **The whole scientific contract is not
unknown merely because the bytes are absent.**

### E07c-3 sampler — **RESOLVED_BY_OFFICIAL_EXECUTION_PATH**

`FAS_sample.py` **is** the repository's dedicated standalone **generation** entrypoint: it loads a
trained model from `--model_path`, builds the conditioning encoder, takes `--live_face_path` and
`--spoof_face_path`, runs the sampler and writes the synthesized image — precisely the
bank-generation operation. Its default is **DDIM** (`FAS_sample.py:116`,
`default='ddim', choices=['ddpm','ddim']`).

`FAS_train.py`'s `ddpm` default governs **only** the periodic training-time visualization: that
sampler sits inside `if (iters)%args.save_images_every_iters==0` (`FAS_train.py:117-163`), which
writes a sample grid to `{iters}_output.png` and never produces a bank.

Under *explicit official execution entrypoint > official source defaults*, the bank-inference sampler
is **DDIM**. **This is not chosen because DDIM is faster.**

Related parameters from the same entrypoint: `DDIM_skip=10`, `sample_initial_noise=250` (so
`seq = range(0,250,10)`, **25 DDIM steps**), `cond_scale=2.0`, `means_size=5`, `var_size=3`,
`use_pair=False`. **Consistency check:** that `use_pair=False` default coincides with frozen DEV-021
`use_pair=false`. **Seed:** unlike `FAS_train.py`'s hard-coded `seed_torch(seed=1)`, the generation
entrypoint **parameterises** it — `FAS_sample.py:33` `seed_torch(args.random_seed)` — so the benchmark
seed propagates natively at generation with no adapter; the repository default `1029` is superseded by
the frozen spec seed contract (M6A1 §K).

### E07c-4 encoder — **STILL_BLOCKED_SOURCE_GAP**

Architecture and role resolved (`models/unet_autoenc.py:73-77` `torch.load`s a **whole pickled
`nn.Module`**, nominally a `custom_rn.resnet18` returning `(x32x32, x16x16, x8x8, embg)`; dataset
implied by `./PADISI.pkl` and `--protocol PADISI`). But the **release search of authoritative sources
found nothing**: the README is three lines ending *"The detailed readme.md is coming soon!"*; the repo
has **no** shell scripts, examples, notebooks or docs; a grep for `padisi / download / release /
huggingface / drive.google / *.pkl` returns only the argparse default, the `PADISIDataset` class and
torchvision's standard ImageNet ResNet URLs; and `murphytju/DiffFAS` has **zero releases and zero
tags**. No official bytes are locatable, so no checksum exists.

**Contrast with E06c:** LightCNN has an official link and a resolved released-file identity; this
encoder has **neither** — it is unlinked *and* not fetched. **No substitute was proposed**, per the
task.

### E07c-5 checkpoint — **STILL_BLOCKED_OWNER_DECISION_REQUIRED**

`FAS_sample.py:112` declares `--model_path` `required=True` with **no default**; no README guidance, no
command examples, no shell scripts, no releases, no tags. Training saves `model_{iters:06d}.pt` every
**10,000 iterations** (`FAS_train.py:102-114`) with no best-checkpoint tracking and no held-out
validation metric. The source names neither a final, a latest nor a best checkpoint, and **"final" was
not inferred.** Sub-detail already resolved in M6A1: the **tensor set** is fixed —
`FAS_sample.py:46` loads `["model"]`, not `["ema"]`; only *which iteration* is unspecified.

---

## 7. Decision surfaces for the 11 residual gaps

Full structured surfaces — `known`, `unknown`, `admissible_owner_choices`, `scientific_consequence`,
`recommended_for_reproducibility`, `recommendation_evidence_class` — are in
`M6A2_BLOCKER_STATE.json` under each blocker's `residual_decision`.

> **Every `recommended_for_reproducibility` field carries evidence class `INFERENCE, NOT
> AUTHORITATIVE`. No recommendation was recorded as a frozen decision. M6A2 froze no owner choice.**

---

## 8. Newly pinned sources

All checkouts are under `third_party/source_cache/` (git-ignored), **source-only**, at exact commits,
with **no actively retained model weights**.

| Cache | Repository | Commit | Role | Embedded weights |
|---|---|---|---|---|
| `facexformer` | `Kartik-3004/facexformer` | `10fe8291f8a64e2ca1daf938e3e0007bd860303b` | E03-2r evidence (matches the M2 pin exactly) | none |
| `defa_iccvw2017` | `yaojieliu/ICCVW2017-DenseFaceAlignment` | `01cfb1aa9845c9ce6c0ac5fbbb16454ee771f848` | E04 reference [60] | **5 model assets + 1 dataset archive + 88 compiled binaries — inventoried and removed** |
| `swapping_autoencoder` | `taesungp/swapping-autoencoder-pytorch` | `6baa180f1184ee79a6b967f9d80ee0e02a979ac7` | E05 reference [34] | none |

### `defa_iccvw2017` — removed binaries

| Category | Count | Size | Examples (sha256 prefix) |
|---|---|---|---|
| Model assets | 5 | 73.8 MB | `DeFA.mat` `5ac1d36e…`, `DeFA_Frontal.mat` `788e26e3…`, `Model_Expression.mat` `4ab0ac3a…`, `Model_info.mat` `68b3c747…`, `Model_Para.mat` `3e9b5f97…` |
| Dataset archive | 1 | 7.4 MB | `AFLW2000-68-Annotations.zip` |
| Compiled binaries | 88 | 12.5 MB | `*.mexa64`, `*.mexw64`, `*.o`, `*.obj` |

94 deletions, **zero source files modified**. Full hashes and git blob IDs are in
`third_party/source_pins.json`. **None was used.**

`NVlabs/stylegan` was **identified but not pinned**, because E05-2's modulation-mechanism fork leaves
its role undetermined.
