# M6A4 — External Asset and Source Resolution

**Date:** 2026-09-21 · **Branch:** `m6-baselines`
**Starting commit:** `121107168a72785ff9a3f91bcb99c71bc648b2e0`

Follows M6A1 (source/execution audit), M6A2 (source-derivable resolution) and M6A3
(owner-approved execution-contract freeze). **No M6A3 owner decision was revisited.**

Scope: authoritative-source research plus the one owner-authorized external weight acquisition
(E06c). No training, no bank generation, no TEST access, no model inference, no method configs.

## Provenance

| Item | SHA256 |
|---|---|
| Frozen specification | `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e` |
| Amendment A1 | `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472` |
| Amendment A2 | `b4fa7bfa75e5a977348c468d1bfe3e0004c3920293ecd9178700f9f4869fca8d` |

All three verified unchanged at the start and end of M6A4.

---

## 0. Result

| | Count |
|---|---|
| Active blockers before M6A4 | **3** |
| Resolved in M6A4 | **1** (`E06c-2`) |
| **Active after M6A4** | **2** |

| Blocker | Method | Status after M6A4 | Active |
|---|---|---|---|
| `E06c-2` | E06c | `RESOLVED_EXTERNAL_WEIGHT_ACQUIRED_AND_PINNED` | no |
| `E04-2` | E04 | `STILL_BLOCKED_SOURCE_GAP` | **YES** |
| `E07c-4` | E07c | `STILL_BLOCKED_SOURCE_GAP` | **YES** |

**E04 and E07c remain BLOCKED.** No claim of successful reproduction is made for either.

---

## 1. E06c-2 — LightCNN-29 v2 · RESOLVED

### 1.1 Identity verification (before download)

| Source | Evidence |
|---|---|
| Upstream author README, `AlfredXiangWu/LightCNN`, line 82 | "The model of LightCNN-29 v2 is released on [Google Drive](https://drive.google.com/open?id=**1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS**)." |
| DSDG README (pinned source), line 21 | "Download the LightCNN-29 model from this [link](https://drive.google.com/file/d/**1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS**/view) and put it to `./ip_checkpoint`." |

The two file ids are **identical**, so DSDG points at the **original author's release**, not a
re-hosted copy. **No mirror was substituted.**

### 1.2 Acquisition

| Field | Value |
|---|---|
| Author-facing URL | `https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view` |
| Resolved download URL | `https://drive.usercontent.google.com/download?id=1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS&export=download&confirm=t` |
| Resolved file id | `1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS` |
| Acquisition (UTC) | start `2026-09-21T14:09:40Z`, end `2026-09-21T14:09:51Z` |
| HTTP status / type | `200` / `application/octet-stream` |
| `content-disposition` | `attachment; filename="LightCNN_29Layers_V2_checkpoint.pth.tar"` |
| HTML / error content | **none** — verified not HTML |
| **Byte size** | **123,844,849** |
| **SHA256** | **`d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964`** |

The served filename matches, byte-for-byte, the path DSDG's `train_generator.sh` passes as
`--ip_model`. Upstream publishes **no** checksum, so this SHA256 is an **observed pin** — exactly as
M6A2 anticipated and A2-08 authorized.

**Storage:** `/media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar`
— outside the repository, outside `third_party/source_cache/`, **not** git-tracked.

### 1.3 Safe checkpoint inspection — no execution

`torch` is **not installed** in this environment, so `torch.load(weights_only=True)` was
unavailable. **This limitation is documented here before the method actually used**, which is not a
weaker fallback but a stronger one: a **restricted, non-executing unpickler** whose `find_class`
never imports or resolves a real object and instead returns an inert stub. `weights_only=True` still
executes real rebuild functions; this inspection **executed nothing at all** from the file. No
fallback to real unpickling was performed, and no model inference was run.

| Field | Value |
|---|---|
| Format | torch **legacy** (non-zip) serialization |
| Pickle protocol | 2 |
| Torch magic number | `0x1950a86a20f9469cfc6c` (authentic torch checkpoint) |
| Torch protocol version | 1001 |
| Top-level keys | `state_dict`, `epoch`, `arch`, `prec1` |
| `state_dict` present | **yes** |
| Tensor count | **61**, all `float32`, all `module.`-prefixed |
| Non-tensor entries | `epoch = 50`, `arch = 'LightCNN'`, `prec1 = 99.46` |
| Globals requested | `collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor` |
| Unexpected executable / pickled objects | **NONE** — every global is torch/OrderedDict |

Key-name examples: `module.conv1.filter.weight` `[96,1,5,5]`, `module.group1.conv.filter.weight`
`[192,48,3,3]`, `module.fc.weight` `[256,8192]`, `module.fc2.weight` `[80013,256]`.

**Corroborating identity evidence from the metadata:** `arch = 'LightCNN'`; `module.fc2.weight` has
**80013** output classes, matching the upstream README's documented `num_classes = 80013` for
LightCNN-29v2; and `prec1 = 99.46` is consistent with the upstream README's reported LightCNN-29v2
LFW accuracy (99.43 % / 99.53 %).

### 1.4 Compatibility with `define_IP()`

Expected keys were derived **statically** from the vendored source
(`networks/light_cnn.py`, `networks/__init__.py`) by mirroring its constructors — no torch, no
instantiation, **no inference**.

Target: `define_IP() = torch.nn.DataParallel(network_29layers_v2(resblock, [1,2,3,4], is_train=False))`.

| Metric | Value |
|---|---|
| Expected parameters | **60** |
| Checkpoint tensors | **61** |
| **Matched (name + exact shape)** | **60** |
| **Missing** | **0** |
| **Shape mismatches** | **0** |
| **Unexpected** | **1** — `module.fc2.weight` `[80013, 256]` |
| DataParallel `module.` prefix on all checkpoint keys | **yes** (as expected) |

`module.fc2.weight` is the upstream 80013-class recognition head. `define_IP()` passes
`is_train=False`, so `fc2_` is never constructed, and the DSDG loader's filter
(`train_generator.py:77`, `if k in model_dict`) drops it. This is the **documented official
behaviour**, not a defect.

Although the official loader is non-strict, the explicitly reported matched subset is **complete**:
every one of the 60 `define_IP()` parameters is supplied with an exactly matching shape.

### 1.5 Classification

All five conditions are met — exact official identity verified, bytes acquired, SHA256 recorded,
structure compatible with `define_IP()`, no unresolved ambiguity:

**`E06c-2` → `RESOLVED_EXTERNAL_WEIGHT_ACQUIRED_AND_PINNED`**

E06c now has **no remaining blocker**. Its method YAML is still **not** created, because M6B has not
started.

---

## 2. E04-2 — Physics-Guided STD geometry/depth route · STILL BLOCKED

`FAITHFUL_PAPER` preserved. **No controlled adaptation was created in M6A4.**

### 2.1 Sources inspected (2026-09-21)

arXiv:2012.05185 abstract + ar5iv full text; the **published journal version** (Liu & Liu, *Spoof
Trace Disentanglement for Generic Face Anti-Spoofing*, **IEEE TPAMI, May 2022**, 17 pages, full text
extracted and searched); the official MSU CVLab project page with all outbound links enumerated; the
author's GitHub account (all 7 public repositories); the author's academic homepage; the pinned
ECCV20-STDN and Dense Face Alignment trees; and the Basel Face Model official pages.

### 2.2 New authoritative findings

- The **published TPAMI 2022** version renumbers references: 3DMM fitting is **[61]** (arXiv's
  [60]) and face alignment remains **[70]**. `[61]` = Liu, Jourabloo, Ren, Liu, *Dense face
  alignment*, ICCVW 2017; `[70]` = Bulat & Tzimiropoulos, ICCV 2017. This **confirms M6A2's
  identification** under the published numbering. **No new source repository is introduced**, and
  `third_party/source_pins.json` therefore needed no change.
- The published version **still** says *"Source code and pre-trained models will be publicly
  available upon publication."* No such release exists: the author's GitHub account holds 7 public
  repositories and **none** is PhySTD/STDN+/PAMI-2022, and the official MSU CVLab project page —
  which lists **both** the ECCV 2020 and PAMI 2022 papers — offers exactly one code link for this
  lineage, `https://github.com/yaojieliu/ECCV20-STDN`.
- The pinned **ECCV20-STDN** tree (12 source files) contains **zero** occurrences of `140`,
  `vertex`, `vertices`, `3DMM`, `BFM`, `Basel`, or any depth-rendering code. The ECCV20 release does
  **not** supply the PhySTD geometry route.
- The TPAMI 2022 paper has **no supplementary-material section** (full-text search for
  `supplementar` → no match).
- The TPAMI 2022 paper **never names the Basel Face Model** (`BFM`, `Basel Face Model` → no match).
  The BFM requirement reaches E04 only **transitively**, via the `[61]` Dense Face Alignment README.

### 2.3 Sub-gap status — all four remain unresolved

**(1) Basel Face Model — `LICENSE_GATED_NOT_ACQUIRED`.** Required by the `[61]` DeFA README
("Setup: 1. Download the Basel Face Model (BFM)"), not by the PhySTD paper. Official acquisition
page from that README: `http://faces.cs.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model`;
current official pages `https://faces.dmi.unibas.ch/bfm/` and `.../bfm2017.html`. **The exact version
is NOT identifiable** — neither the paper nor the DeFA README states which BFM version; DeFA
(ICCVW 2017) predates BFM-2017 and links the original page, but nothing fixes it. **Expected files are
not documented** — the README says only "unzip the file into the root folder"; the pinned DeFA tree's
removed binaries (`Model_Para.mat`, `Model_info.mat`, `Model_Expression.mat`, `DeFA.mat`) are DeFA
model files, not the BFM distribution. **Why automated acquisition is not permitted:** BFM is
distributed only under a signed academic end-user licence requiring a request form, an official
academic e-mail address and a verifiable university affiliation. It cannot be fetched
programmatically and must not be redistributed. **It was NOT downloaded and the licence was NOT
bypassed.**

**(2) Q = 140 vertex set — `COUNT_STATED_INDICES_NEVER_PUBLISHED`.** The published text states only:
*"We select Q=140 vertices to cover the face region so that they can represent non-rigid
deformation, due to pose and expression."* and §4.1 *"We use the open-source face alignment [70] and
3DMM fitting [61] to crop the face and provide 140 landmarks."* **No index list, table, appendix,
supplementary file or author-released config/data file enumerating which 140 vertices are selected
was located** in any searched source. Reverse-engineering a set, reusing another paper's 140-point
set, or substituting the FaceXFormer 68 landmarks are all prohibited and were **not** done.

**(3) Depth rendering — `GROUND_TRUTH_RENDERING_NEVER_DESCRIBED`.** The paper says only: *"We apply
the dense face alignment [61] to estimate the 3D shape and render the depth ground truth M₀."* What
**is** specified is only the **network output**: *"It outputs a face depth map M ∈ R^{32×32}, where
the depth values are normalized within [0,1]"* (K = 32). The decisive distinction is that **M₀ — the
depth ground truth — is an INPUT to the published training algorithm** (*"Algorithm 1: STDN+ Training
Iteration. Input: … ground truth depth map M₀, preliminary mask P₀"*), so the paper treats it as
pre-computed and never describes how it is produced. Unspecified: camera/projection model,
visible-surface handling, ground-truth normalization, rasterization/interpolation, foreground/
background convention, output scale and range, and the resize rule to 32×32. **No step was inferred.**

**(4) Executable route — `NO_EXECUTABLE_OFFICIAL_ROUTE`.** No PhySTD/TPAMI code was ever released
(`RESOLVED_BY_DOCUMENTED_SEARCH_NO_OFFICIAL_RELEASE_LOCATED_AS_OF_2026_09_21`). The `[61]` DeFA
release is MATLAB/MatConvNet **test-only** code that additionally requires the licence-gated BFM, so
even with BFM it is not a reproducible fitting-to-depth pipeline.

### 2.4 Classification

Per A2-09 and task §3.4, E04-2 resolves only if the **complete** route becomes uniquely
reproducible. All four sub-gaps remain unknown.

**`E04-2` → `STILL_BLOCKED_SOURCE_GAP`.** The existence of upstream repositories does **not** make
E04 ready.

---

## 3. E07c-4 — DiffFAS PADISI conditioning encoder · STILL BLOCKED

Faithful official DiffFAS route preserved: **no substitute trained, encoder not disabled, not
replaced with ImageNet ResNet18, no PADISI training details invented.**

### 3.1 Where the search went (2026-09-21)

The pinned `murphytju/DiffFAS` tree (full grep); GitHub API for repository metadata (**size 214 KB**,
last push **2024-09-23T11:57:47Z**), **releases → 0**, **tags → 0**, **branches → 1 (`main`)**,
**full commit history → 5 commits, all enumerated**, **issues (all states) → 1 issue, all 6 comments
read**; the **ECCV 2024 official supplementary PDF**
(`https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07068-supp.pdf`, 5 pages, text extracted
and searched — **no** occurrence of `PADISI.pkl`, `pretrain`, `encoder`, `classifier`, or any
download link); arXiv:2409.08572 / ECVA listings; and a web search for an author-controlled mirror.

Search terms: `PADISI.pkl`, `PADISI`, `classifier`, `encoder`, `pretrained classifier`, `pretrained
encoder`, `custom_rn`, `resnet18`, `download`, `checkpoint`, `model_path`, `conditioning`,
`autoencoder`.

### 3.2 Decisive maintainer-level finding

In `https://github.com/murphytju/DiffFAS/issues/1` ("Pretrained Classifier model ?", opened
2024-09-24, which asks precisely "Could you share 'PADISI.pkl' file?"), the repository **OWNER**
(`murphytju`, `author_association = OWNER`) replied on **2024-09-24T02:53:15Z**:

> "I will update the required pretrained models, readme.md and further check the code. Please wait
> for a moments."

**It was never delivered.** The repository's last push (2024-09-23) **predates** that promise, there
are **0 releases and 0 tags**, and four further requests (2024-10-22, 2024-11-08, 2025-03-05,
2025-03-10) received no maintainer reply. This is authoritative **maintainer-level confirmation**
that the weights were never released — not an inference from absence.

### 3.3 New finding — the official training recipe DOES exist

M6A4 newly establishes that the official repository contains the encoder's **training script** at
`models/pretrain_classifier.py`. The training protocol therefore **no longer has to be invented**.
This **refines** the gap; it does **not** close it.

| Item | Official value |
|---|---|
| Architecture | `custom_rn.resnet18()` with `fc = nn.Linear(512, 17)` → **17 classes** |
| Architecture note | `custom_rn.resnet18` is `_resnet('resnet18', BasicBlock, [3,4,6,3])` — a **ResNet-34 layer configuration despite the name**. Recorded for disclosure. |
| Transform | `Resize((256,256))` → `ToTensor` → `Normalize([0.5]*3, [0.5]*3)` |
| Dataset root | `/media/26d532/gxx/PADISI_USC/pasidi_attack/` (torchvision `ImageFolder`; the authors' local absolute path) |
| Batch / workers | 256 / 6, `drop_last=True`, `shuffle=True` |
| Optimizer | SGD `lr=0.002`, `momentum=0.9`, `weight_decay=5e-3` |
| Criterion | `CrossEntropyLoss` on the 4th forward output (`embg`) |
| Epochs | 200 |
| Save | `torch.save(resnet18, './PADISI.pkl')` — saves the **whole** `nn.Module`, consistent with `unet_autoenc.encoder()`'s `torch.load(path)` |

**Still missing despite this finding:** the released `PADISI.pkl` **weights** (maintainer-acknowledged
above); the **PADISI-USC dataset** itself (licence-gated, and **not** one of this benchmark's datasets
— CASIA-FASD, MSU-MFSD, SiW-Mv2); the **17-class label mapping** (torchvision `ImageFolder` assigns
labels by sorted subdirectory name, and the 17 subdirectory names are not published, so **even a
faithful re-run of the official script is not uniquely reproducible**); and confirmation that the
paper's encoder came from exactly this script.

The dependency is **unconditional**: `diffusion.training_losses` calls `model(...)` without `cond`,
so `encode(x_cond, encoder)` runs on **every** training step. `use_pair=false` (DEV-021) does not
remove it.

### 3.4 Classification

**No official file or link was located**, so task §4.2 applies (not §4.1) and **nothing was
downloaded for E07c**. The E06c acquisition authorization was **not** applied to E07c.

**`E07c-4` → `STILL_BLOCKED_SOURCE_GAP`**, dated conclusion
**`NO_OFFICIAL_ENCODER_RELEASE_LOCATED_AS_OF_2026_09_21`**. Absolute nonexistence is **not** claimed.

Any future decision to train a replica encoder would require a **new, explicit owner authorization**
and would change E07c's fidelity class. M6A4 makes no such decision.

---

## 4. Constraints observed

No training. No optimizer steps. No synthetic-bank generation. No TEST access. No model inference.
No checkpoint created by training. No baseline implementation. No method config. No undocumented
asset substituted. No missing scientific detail invented. No BFM download. No PADISI download. Only
the E06c weight — explicitly owner-authorized — was acquired. The frozen spec, Amendment A1 and
Amendment A2 are unchanged. Not committed, not pushed.
