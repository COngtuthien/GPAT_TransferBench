# M6A3a — Owner Decision Packet

**Date:** 2026-09-21 · **Branch:** `m6-baselines` · **Starting commit:** `0d94d5f92f531d6ce22f59bebcd5d2444fe7e9fd`

The 11 blockers left open by M6A2, rendered decision-ready. Machine-readable companion:
`M6A3_OWNER_DECISION_PACKET.json`.

> **Nothing is resolved here.** Every **Recommended** line is evidence class
> **`INFERENCE, NOT AUTHORITATIVE`** and is *not* an owner decision. No config was frozen, no weight
> downloaded, no method YAML created, no repository state changed beyond these two files.

Two constraints hold across every checkpoint question below: **TEST may never be used for checkpoint
selection**, and **GPAT's §10.6 SelectionScore is never auto-applied to a baseline**.

---

## 0. Summary

| Blocker | Method | Category | Recommended *(inference)* |
|---|---|---|---|
| E01-3 | E01 | `OWNER_CONTRACT_CHOICE` | **A** |
| E02-1 | E02 | `OWNER_CONTRACT_CHOICE` | **B** |
| E02-2 | E02 | `OWNER_CONTRACT_CHOICE` | **A** |
| E02-3 | E02 | `OWNER_CONTRACT_CHOICE` | **B** |
| E04-2 | E04 | `EXTERNAL_SOURCE_GAP_NOT_SOLVABLE_BY_SIMPLE_CHOICE` | **A or C** |
| E04-3 | E04 | `OWNER_CONTRACT_CHOICE` | **C** (defer) |
| E05-2 | E05 | `OWNER_CONTROLLED_ADAPTATION_REQUIRED` | **A** |
| E05-3 | E05 | `OWNER_CONTRACT_CHOICE` | **C** (defer) |
| E06c-2 | E06c | `OWNER_ASSET_ACQUISITION_APPROVAL` | **A** |
| E07c-4 | E07c | `EXTERNAL_SOURCE_GAP_NOT_SOLVABLE_BY_SIMPLE_CHOICE` | **A** |
| E07c-5 | E07c | `OWNER_CONTRACT_CHOICE` | **C** (defer) |

Category counts: 7 contract choices · 2 external source gaps · 1 controlled adaptation · 1 asset
acquisition.

---

## 1. E01-3 — FAS-Aug texture enumeration order · `OWNER_CONTRACT_CHOICE`

**Question.** In what order must the FAS-Aug texture asset directories be enumerated before the
per-pair RNG selects a texture?

**Known.** Official code builds the three name lists with `os.listdir` *at import time*
(`fas_aug_helper.py:17-26`) and picks by index (line 50) — filesystem-dependent. **No canonical
official order exists**: no committed asset list (`data/data_list/*.csv` are dataset *frame* lists),
no config reference, no archive metadata. A canonical **repository** order does exist — git sorts
tree entries bytewise; verified lexicographic for all three dirs (background 90, noiseTexture 48,
MPTexture 190). The frozen spec has **no** global asset-ordering rule. The authors' original
filesystem order is **unrecoverable**.

**Unknown.** Which enumeration order to freeze.

| | Choice | Deterministic | Literal to official code |
|---|---|---|---|
| **A** | Lexicographic (bytewise) — coincides exactly with git tree order | ✅ | ❌ |
| **B** | Keep raw `os.listdir` | ❌ | ✅ |
| **C** | Owner-supplied explicit frozen asset-order manifest (with per-file sha256) | ✅ | ❌ |

**Tradeoff, stated plainly:** **B is closest to the literal official code but is host-dependent**, so
the E01 bank is not reproducible and the benchmark's deterministic-rerun requirement cannot be met.
**A and C are deterministic adaptations.**

- **Scientific consequence:** the choice changes which texture each pair receives, hence the pixels of
  every `Reflection` / `BN_Halftone` / `Moire_Pattern` output. Operator set, ranges and seed unchanged.
- **Fidelity:** E01's tag is **FAITHFUL_OFFICIAL**. A and C require a *declared deviation* on that tag;
  B preserves the letter of the code but forfeits reproducibility.
- **Amendment A1 affected:** no. **Classification change:** A/C → FAITHFUL_OFFICIAL + 1 declared
  deviation; B → tag unchanged but rerun-hash requirement unmet.
- **Unblocks:** E01 fully — this is E01's only remaining blocker.

**Recommended: A** — the only option that is both deterministic and *already canonical inside the
pinned repository*, so it needs no new artifact. C is equally defensible if the owner wants the order
pinned explicitly. · `INFERENCE, NOT AUTHORITATIVE`

---

## 2. E02-1 — rounding of "exactly 25 %" · `OWNER_CONTRACT_CHOICE`

**Question.** How is *"exactly 25 % of eligible blocks"* rounded when the product is not an integer?

**Known (exact computation on the frozen geometry).** 43,186 eligible pixels → **159 eligible blocks**
of 256. **159 × 0.25 = 39.75**, so the choice is real, not moot. The count 159 is invariant to the
boundary convention (closed, half-open and open intervals all give 159).

| | Rule | Count for n = 159 | General rule |
|---|---|---|---|
| **A** | floor | **39** | `floor(n/4)` |
| **B** | round-half-up | **40** | `floor(n/4 + 0.5)` |
| **C** | ceil | **40** | `ceil(n/4)` |

> **B and C give the same count here but encode different general rules.** They diverge elsewhere —
> `n=157` → 39.25 gives floor **39**, round-half-up **39**, ceil **40** (B ≠ C); `n=158` → 39.50 gives
> 39 / 40 / 40; `n=156` → 39.00 gives 39 / 39 / 39. **Freeze the rule, not the number 40**, because the
> eligible count would change if any geometry parameter were ever revised.

- **Scientific consequence:** one extra or one fewer substituted block per pair (39 vs 40 of 159),
  changing the pixels of every E02 output.
- **Reproducibility:** all three equally reproducible — purely definitional.
- **Fidelity / A1 / classification:** none. E02 is already `CONTROLLED_ADAPTATION` by §8.2, which
  exists precisely because the paper does not fix these details.

**Recommended: B** — least biased across arbitrary counts; floor systematically under-selects, ceil
systematically over-selects. Purely a convention argument. · `INFERENCE, NOT AUTHORITATIVE`

---

## 3. E02-2 — block-selection procedure · `OWNER_CONTRACT_CHOICE`

**Question.** By what deterministic procedure are the *k* blocks chosen from the 159 eligible blocks,
**given a seed**? *(Seed construction is a separate question — E02-3. Do not conflate.)*

**Known.** §8.2 fixes neither RNG family nor draw procedure. No other frozen section defines a generic
element-selection procedure. Repository precedent exists but is **not authoritative here**:
`configs/frozen/pairs_v1.yaml` selects by deterministic SHA-256 **ranking**, not by an RNG object.

| | Choice | RNG object | Durability |
|---|---|---|---|
| **A** | Hash ranking: rank eligible blocks by `SHA256(seed_bytes ‖ UTF8(block_index))`, take first *k* | ❌ | Depends on nothing but SHA-256; identical across platforms and library versions |
| **B** | `numpy.random.Generator(PCG64(seed)).choice(n, k, replace=False)` | ✅ | Stable for a fixed NumPy version; `.choice` is an implementation detail that has changed historically |
| **C** | `random.Random(seed).sample(range(n), k)` | ✅ | Stable for a fixed CPython version; `sample`'s algorithm carries no cross-release stability guarantee |

- **Scientific consequence:** different procedures select different block subsets from the **same**
  seed, changing the pixels of every E02 output.
- **Reproducibility:** A is most durable; **B and C bind the frozen contract to a library version.**
- **Fidelity / A1 / classification:** none.

**Recommended: A** — removes an entire class of future reproducibility risk and is already precedented
in this repository's frozen M4 contract. · `INFERENCE, NOT AUTHORITATIVE`

---

## 4. E02-3 — seed construction · `OWNER_CONTRACT_CHOICE`

**Question.** What exactly is the *"pair-specific RNG seed"* for E02 — i.e. what is its preimage?
*(Selection procedure is E02-2.)*

**Known.** §8.1 defines `seed = SHA256(pair_id + global_seed) mod 2^31` **explicitly and only for
FAS-Aug**. §8.2 says only *"using pair-specific RNG seed"*. The pair manifest's `seed` column is the
**constant** `20260814` for all 8,838 rows — not a per-pair seed. M4's frozen digests are namespaced
`gpatbench.pair.source_seed.v1` / `gpatbench.pair.candidate.v1`, both **scoped to pair construction**;
that explicit namespacing is evidence *against* silent reuse. `pair_id` format **is** frozen
(`PTR%06d` / `PVA%06d`).

| | Choice | Exact proposed preimage |
|---|---|---|
| **A** | Reuse the §8.1 formula verbatim | `SHA256(UTF8(pair_id + str(global_seed)))` → big-endian integer → `mod 2^31` |
| **B** | New namespaced E02 digest, M4 style | `SHA256(UTF8('gpatbench.freqsub.block.v1\|' + pair_id + '\|' + str(global_seed)))` → **raw 32-byte digest carried forward as bytes** |
| **C** | Owner specifies another preimage | *(owner to state)* |

- **Scientific consequence:** **A** makes E01 and E02 derive their seeds from the *same* preimage for
  the same pair, correlating the two methods' randomness — which may or may not be intended. **B**
  guarantees independent streams.
- **⚠ A depends on OBS-1** (§9): it cannot be implemented until `global_seed` is *defined* and the
  preimage byte layout fixed. **B as written fixes the byte layout but still needs `global_seed`
  defined.**
- **Fidelity / A1 / classification:** none.

**Recommended: B** — a distinct namespace avoids the E01/E02 correlation, and the M4 precedent already
demonstrates the required byte-level discipline. · `INFERENCE, NOT AUTHORITATIVE`

---

## 5. E04-2 — Physics-Guided STD geometry route · `EXTERNAL_SOURCE_GAP_NOT_SOLVABLE_BY_SIMPLE_CHOICE`

**Question.** How should E04's geometry supervision route (3DMM fit → 140 landmarks → 32×32 depth
ground truth for `L_depth`, **α₁ = 100**) be obtained, given that no single source determines it?

**Known.** [60] = *Dense Face Alignment*, ICCVW 2017 → `yaojieliu/ICCVW2017-DenseFaceAlignment@01cfb1aa…`
(pinned source-only in M6A2). [70] = Bulat & Tzimiropoulos, ICCV 2017 → `1adrianb/face-alignment`.
Paper §4.1: *"We use the open-source face alignment [70] and 3DMM fitting [60] to crop the face and
provide 140 landmarks"*; Eq. 16: *"We apply the dense face alignment [60] to estimate the 3D shape and
render the depth ground truth M⁰"*, `K = 32`. The frozen M2 cache has 68 landmarks, no 3DMM, no depth.

### The four sub-gaps, and what can actually solve each

| # | Sub-gap | Solvable by |
|---|---|---|
| 1 | **Basel Face Model** is license-gated ([60]'s README requires it from `faces.cs.unibas.ch` under an individually signed academic licence) | **Asset-acquisition approval** (subject to the licence being granted) |
| 2 | **Which Q = 140 vertices.** Paper says only *"We select Q = 140 vertices to cover the face region"* — no index list, no rule, no released file | **NOT** acquisition. Controlled adaptation or the authors |
| 3 | **Depth rendering** of the fitted shape into the 32×32 `M⁰` — undescribed; [60]'s repo does alignment, not depth | **NOT** acquisition. Controlled adaptation or the authors |
| 4 | [60] is legacy MATLAB/MatConvNet with compiled MEX, no training code; no [70] commit pinned | Partly **engineering** (container + MATLAB licence). Only *inference* is needed from [60], so test-only code is acceptable in principle |

> **Key finding: even with full asset-acquisition approval, sub-gaps 2 and 3 remain.** E04-2 therefore
> **cannot** be closed by an acquisition approval alone — which is why it is *not* categorised as
> `OWNER_ASSET_ACQUISITION_APPROVAL`.

| | Choice | Consequence |
|---|---|---|
| **A** | Preserve **FAITHFUL_PAPER** and keep E04 blocked | No executable main row; fidelity preserved honestly; reported as a disclosed source gap |
| **B** | Approve a **fully documented controlled adaptation** (owner-defined 140-vertex set + depth rendering, with an equation-to-code traceability table) | E04 becomes executable, **but its tag must change FAITHFUL_PAPER → CONTROLLED_ADAPTATION** and be disclosed wherever the E04 row appears. Requires BFM acquisition first |
| **C** | **Exclude E04** from the main executable generator table, retain the source-gap disclosure | Benchmark loses one row but makes no unsupported fidelity claim |

- **Fidelity:** **B requires a downgrade that must not happen silently.** A and C preserve the tag.
- **Amendment A1 affected:** no. **Blocks:** E04 entirely; **E04-3 is moot until this is answered.**

**Recommended: A or C** — both keep the fidelity claim honest. B is legitimate only if the owner
accepts the downgrade and documents the adaptation in `methods/physics_std/source_traceability.md` as
§8.4 requires. · `INFERENCE, NOT AUTHORITATIVE`

---

## 6. E04-3 — E04 checkpoint selection · `OWNER_CONTRACT_CHOICE`

**Question.** Which E04 model state generates the synthetic bank? *(Checkpoint only.)*

**Known.** Final paper sweep found **no** selection rule — only the budget *"We train in total 150,000
iterations with a batch size of 8"*. *"Train till convergence"* concerns a downstream
trace-classification experiment, not checkpointing. §10.6 is GPAT-only; §14 is downstream-only; **no
baseline rule exists.**

| | Choice | Consequence |
|---|---|---|
| **A** | Final iteration (150,000) | Reproducible, but "final" is **not** a paper fact — it would be a declared benchmark convention |
| **B** | VAL-based selection under a **newly frozen** benchmark baseline rule | Uses the VAL split the global pseudocode already passes to `train_method`. **GPAT's §10.6 SelectionScore must not be borrowed automatically** |
| **C** | Remain blocked | Consistent with E04-2 option A or C |

**Depends on E04-2.** **Recommended: C (defer)** — E04 cannot be trained while E04-2 is open, so
answering now would freeze a convention for a method that may never execute. · `INFERENCE, NOT AUTHORITATIVE`

---

## 7. E05-2 — PCGAN modulation/discriminator · `OWNER_CONTROLLED_ADAPTATION_REQUIRED`

**Question.** Which concrete modulation and discriminator does PCGAN use — StyleGAN-v1 **AdaIN** as its
prose cites [35], or the **StyleGAN2 weight demodulation** that its declared base [34] actually
implements?

### What [34] is now resolved to

`taesungp/swapping-autoencoder-pytorch@6baa180f1184ee79a6b967f9d80ee0e02a979ac7`, tree
`25a434f1282a0177d6afdcc68ebfcf40bd805621`, 73 files, **zero weights**. Identity proven by **three
exact matches**: PCGAN names `netE_num_downsampling_sp` and its default **4** (`encoder.py:35`);
PCGAN's `z_pat` has **8** channels and the repo default `spatial_code_ch` is **8**
(`swapping_autoencoder_model.py:12`); the repo description matches the [34] citation verbatim.

**[34]'s pinned repo contains the complete required module set** — encoder, generator, StyleGAN2
discriminator (`discriminator.py:2`), and patch discriminator (`patch_discriminator.py:97`).

### What remains unresolved about [35] / AdaIN

PCGAN's prose says the latents are integrated *"through adaptive instance normalization [35]"* —
**AdaIN is the StyleGAN-v1 mechanism**. But [34] uses **StyleGAN2 weight demodulation**
(`stylegan2_layers.py:210,217,257,272` — `ModulatedConv2d`, `demodulate=True`;
`generator.py:80,114,150` `SpatialCodeModulation`; `generator.py:89` *"borrow heavily from StyleGAN2
code"*). **StyleGAN2 explicitly replaced AdaIN with demodulation.** [35] = Karras et al. CVPR 2019,
`NVlabs/stylegan@1e0d5c78…` (TensorFlow), identified but deliberately **not** pinned.

> **Direct answer to the question posed:** a controlled implementation **can** rely on the pinned
> swapping-autoencoder for the **complete** required module set. **Doing so does change the PCGAN
> paper contract**, because [34] implements StyleGAN2 demodulation whereas PCGAN's text specifies
> AdaIN.

| | Choice | Complete module from one pin? | Consequence |
|---|---|---|---|
| **A** | Rely entirely on pinned [34] (StyleGAN2 demodulation + StyleGAN2 discriminator + patch discriminator) | ✅ | Executable immediately, zero weights. Departs from PCGAN's written AdaIN description → row must be tagged **CONTROLLED_ADAPTATION** |
| **B** | Implement PCGAN's prose literally (AdaIN + StyleGAN-v1 discriminator) | ❌ | Honours the text but departs from [34], which PCGAN says it follows. Requires writing an AdaIN generator present in neither pinned repo (NVlabs/stylegan is TF) — a from-scratch reimplementation with its own fidelity risk |
| **C** | Seek clarification from the PCGAN authors | — | Only route that could preserve a genuine FAITHFUL_PAPER claim. Unbounded latency |

- **Fidelity:** E05's tag is already *"FAITHFUL_PAPER or CONTROLLED_ADAPTATION (depends on code
  availability)"*. Q-08 found no official code, and this fork means **a genuine FAITHFUL_PAPER claim is
  not currently supportable under A or B**. Settle the tag as part of this answer.
- **Amendment A1 affected:** no. **Blocks:** E05; **E05-3 is moot until this is answered.**

**Recommended: A** — PCGAN's own parameter-level instruction ties it concretely to [34]'s code, and [34]
is the only artifact that can actually be pinned and run. Reading the AdaIN citation as loose
attribution is *inference*, which is exactly why this is the owner's call. · `INFERENCE, NOT AUTHORITATIVE`

---

## 8. E05-3 — PCGAN checkpoint selection · `OWNER_CONTRACT_CHOICE`

**Question.** Which PCGAN generator state synthesises the bank? *(Checkpoint only — separated from the
architecture/source decision in E05-2.)*

**Known.** Final sweep returned **zero** hits for released model / model zoo / snapshot / final
iteration / model selection. Only the budget exists: *"Adam optimizer for 4000 iterations"*, batch 1,
lr 1e-6. The paper's *"average over the last 10 epochs"* and *"best epoch"* describe **detector**
evaluation (Tabs. 2-3), **not** generator selection — and an average over epochs is not a single
checkpoint at all. No official code exists (Q-08).

| | Choice | Consequence |
|---|---|---|
| **A** | Final iteration (4,000) | Reproducible; a declared benchmark convention, not a paper fact |
| **B** | VAL-based newly frozen benchmark baseline rule | Owner must define it; §10.6 SelectionScore **not** auto-borrowed |
| **C** | Remain blocked | Defer until E05-2 is settled |

**Depends on E05-2.** **Recommended: C (defer)** — a checkpoint rule for an unsettled architecture is
premature. · `INFERENCE, NOT AUTHORITATIVE`

---

## 9. E06c-2 — LightCNN-29 v2 acquisition · `OWNER_ASSET_ACQUISITION_APPROVAL`

**Question.** May the LightCNN-29 v2 identity-preserving checkpoint be acquired **in a later
asset-acquisition step**, hashed and verified, so E06c can execute?

**Scientific identity is fully resolved — only the bytes are absent.**

| Property | Value |
|---|---|
| Architecture | `network_29layers_v2(resblock,[1,2,3,4])` in `DataParallel` (`networks/__init__.py:22-25`), **vendored** at `networks/light_cnn.py` — no external code dependency |
| Upstream | Wu, He, Sun, Tan, *IEEE TIFS* 2018 13(11):2884-2896; `AlfredXiangWu/LightCNN` (HEAD `7b38a6f2…`) |
| **Provenance** | DSDG README links Drive id `1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS`; the **upstream** LightCNN README's own "LightCNN-29 v2" link is the **same id** — DSDG points at the original authors' release, not a re-host |
| Expected path | `./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar` |
| Load format | `torch.load` → `checkpoint['state_dict']` → filtered to keys present in the wrapped net → **non-strict** load, then frozen and `eval` |
| Role | `loss_ip`, **λ_ip = 1000**, retained by DEV-020 |
| Official checksum | **None published** |
| Local status | **NOT_FETCHED**; sha256 **UNKNOWN** |

| | Choice | Consequence |
|---|---|---|
| **A** | Authorize the download in a later asset-acquisition step, record the observed sha256 **as the authoritative pin**, verify key match against `define_IP()`, then continue | E06c executable with its official identity loss intact. Since no upstream digest exists, the sha256 observed at acquisition *becomes* the pin and must be recorded then. **Note:** the official loader is **non-strict** — the verification step should also assert the intersecting key set is complete, or a silently partial load would go unnoticed |
| **B** | Keep E06c blocked | Track-A primary comparison loses its DSDG row |
| **C** | Substitute another identity encoder, **only as a declared controlled adaptation** | Changes the feature space in which `loss_ip` (λ=1000) is measured, so it changes the trained generator. Forfeits the official-loss claim |

- **Amendment A1 affected:** **A → no. C → yes, indirectly** — DEV-020 records that all losses other
  than `loss_pair` are **retained**; swapping the identity network means `loss_ip` is no longer the
  official term, so DEV-020 would need an explicit addendum. **A1's identity firewall is not breached
  either way**: `loss_ip` compares each reconstruction to its *own* input
  (`identity_requirement: NONE` in the frozen config), so no cross-identity supervision is reintroduced.
- **Classification:** A and B → unchanged (`CONTROLLED_ADAPTATION`). C → same tag plus an additional
  declared deviation.
- **Unblocks:** E06c fully — its only remaining blocker.

**Recommended: A** — the identity is fully established and the link is the original authors' own
release, so acquisition is a provenance-clean step rather than a scientific choice. · `INFERENCE, NOT AUTHORITATIVE`
**No download was performed in M6A3a.**

---

## 10. E07c-4 — DiffFAS conditioning encoder · `EXTERNAL_SOURCE_GAP_NOT_SOLVABLE_BY_SIMPLE_CHOICE`

**Question.** How should the DiffFAS conditioning encoder be obtained, given that **no official release
could be located**?

**Known.** Architecture and role resolved: `models/unet_autoenc.py:73-77` `torch.load`s a **whole
pickled `nn.Module`**, nominally a `custom_rn.resnet18` returning `(x32x32, x16x16, x8x8, embg)`. The
dependency is **unconditional** — `training_losses` calls `model(...)` without `cond`, so `forward`
takes the `cond is None` branch and calls `encode(x_cond, encoder)` on **every** training step;
`use_pair=false` does **not** remove it. Exhaustive search found nothing: README is three lines ending
*"coming soon!"*; no shell scripts, examples, notebooks or docs; grep returns only the argparse
default, `PADISIDataset`, and torchvision's ImageNet URLs; **zero releases, zero tags**.

> **This is why it is a source gap, not an acquisition approval.** Contrast E06c: LightCNN has an
> official link and a resolved released-file identity. This encoder has **neither** — unlinked *and*
> not fetched.

| | Choice | Consequence |
|---|---|---|
| **A** | Keep the faithful E07c row **blocked** until the authors supply the encoder | Track-A loses its DiffFAS row. No unsupported claim |
| **B** | Owner-approved **benchmark-trained substitute**, declared as a controlled adaptation | E07c executable, but conditioning **features differ** from the authors', so the generator differs. **Not equivalent** to the official encoder and must never be presented as such. Needs an explicit DEV record and an A1 addendum |
| **C** | Remove/disable the conditioning path entirely | **Not mathematically justified by anything in the source.** The encoder *is* the style-conditioning path; removing it makes the diffusion **unconditional** and eliminates the style-transfer mechanism that defines DiffFAS. It also **contradicts frozen DEV-021** (*"conditioning is style_spoof only and the target is the epsilon of GT"*) |

**None of A, B, C is equivalent to any other.**

- **Amendment A1 affected:** A → no. **B → yes** (addendum recording the substitute). **C → yes, it
  contradicts DEV-021 outright and would require re-opening Amendment A1.**
- **Classification:** A → unchanged. B → `CONTROLLED_ADAPTATION` + added declared deviation. C → the
  `USING_OFFICIAL_UNPAIRED_CODE_PATH` qualifier would no longer be accurate.
- **Blocks:** E07c; **E07c-5 is moot until this is answered.**

**Recommended: A** — B and C are real scientific departures. If the owner needs the row, B is the
lesser departure, but it must be **declared, never implied**. · `INFERENCE, NOT AUTHORITATIVE`
**No substitute was proposed or trained in M6A3a.**

---

## 11. E07c-5 — DiffFAS checkpoint selection · `OWNER_CONTRACT_CHOICE`

**Question.** Which DiffFAS training checkpoint iteration generates the bank? *(Checkpoint only.)*

**Known.** `FAS_sample.py:112` declares `--model_path` `required=True` with **no default**; no README
guidance, no command examples, no shell scripts; **zero releases, zero tags**. Training saves
`model_{iters:06d}.pt` every **10,000 iterations** with **no** best-checkpoint tracking and **no**
held-out validation metric. **Already resolved (M6A1):** *which tensor set* is fixed by the source —
`FAS_sample.py:46` loads `["model"]`, not `["ema"]`. Only *which iteration* is unspecified.

| | Choice | Consequence |
|---|---|---|
| **A** | Final (last saved) training checkpoint | Reproducible; a declared benchmark convention, **not** a source fact |
| **B** | VAL-based benchmark selection rule, newly frozen for baselines | Owner must define it; §10.6 SelectionScore **not** auto-borrowed |
| **C** | Remain blocked | Consistent with E07c-4 option A |

**Depends on E07c-4.** **Recommended: C (defer)** — E07c cannot be trained while E07c-4 is open.
· `INFERENCE, NOT AUTHORITATIVE`

---

## 12. OBS-1 — newly observed, **not** a re-opened blocker

**Spec §8.1's FAS-Aug seed is incompletely specified in two ways, and this affects E01 as well as
E02-3 option A.**

1. **`global_seed` is never defined.** It appears **exactly once** in the entire frozen spec (line 232)
   and is defined nowhere — not in the spec, not in any frozen config (the two mentions in
   `configs/frozen/pairs_v1.yaml:45,140` merely quote §8.1). The candidates differ *scientifically*: if
   `global_seed = split_seed 20260814` the E01 bank is **identical** across the three benchmark seeds;
   if it is the per-run seed from `[42,1337,2026]` the E01 bank **differs per seed**. The global
   pseudocode places bank generation *inside* the `for seed in [42,1337,2026]` loop, which bears on
   this — but reading intent from loop placement would be **inference**, so nothing was concluded.
2. **The preimage byte layout is not fixed anywhere** — no separator, no encoding, no
   integer-conversion convention for `mod 2^31`. By contrast the M4 `pairs_v1` digests specify their
   byte layout exhaustively. (`pair_id` *format* **is** frozen, so only the seed's definition and the
   concatenation are open.)

**Why it matters:** E02-3 option A proposes reusing this formula and cannot be implemented until both
facets are settled. It also affects **E01** directly, whose only listed blocker is E01-3.

**Action requested:** fold into the E02-3 answer, or issue as a separate decision. **Not resolved here
and not counted among the 11 blockers.** Evidence class: `OBSERVATION`.

---

## 13. Recommended answer order

Derived from the current blocker state — prerequisite chains and how many methods each answer unblocks.

| Tier | Answer | Why | Unblocks |
|---|---|---|---|
| **1** | **E02-1, E02-2, E02-3** | Three pure convention choices, zero external dependencies. Fully unblocks E02 — fastest route to an implementable contract | **E02** |
| **2** | **E01-3** | One convention choice, no external dependency. Fully unblocks E01. After E02 only because E01-3 interacts with OBS-1 | **E01** |
| **3** | **E06c-2** | One acquisition approval that fully unblocks a **Track-A primary** method whose scientific identity is already resolved — highest value per answer of any remaining blocker | **E06c** |
| **4** | **E07c-4** | Source gap on the other **Track-A primary** method; must precede E07c-5 | — |
| **5** | **E05-2** | Controlled-adaptation decision; must precede E05-3 | — |
| **6** | **E04-2** | Deepest source gap; must precede E04-3 | — |
| **7** | **E04-3, E05-3, E07c-5** | All three are the **same question** — *"which checkpoint generates the bank?"* — and each is moot until its prerequisite above is answered | — |

> **Collapse opportunity:** tier 7 is one question asked three times. **A single frozen benchmark-wide
> baseline checkpoint rule would answer E04-3, E05-3 and E07c-5 at once**, and would be more
> defensible than three independently chosen conventions.

---

## 14. Answer template

```
E01-3:
E02-1:
E02-2:
E02-3:
E04-2:
E04-3:
E05-2:
E05-3:
E06c-2:
E07c-4:
E07c-5:

OBS-1:   (optional — global_seed definition + §8.1 preimage byte layout)
```
