# M6A1 — Baseline Source & Provenance Audit (E01–E07c)

**Date:** 2026-09-21 · **Branch:** `m6-baselines` · **Starting HEAD:** `89ea43e3d32d87d0641ec57bb62e6c72694dbe8a`
**Frozen spec sha256:** `f7d2371682ad9504f59f0c6984e4b1a741f855f1b1f60838f9d60b160281489e`
**Amendment A1 sha256:** `03828716def5e535d82445974972bf71a5c8ecc60392fac4b884bcbe060e3472`
**Scope:** source identity, pinning and provenance only. **No training, no generation, no TEST access,
no checkpoint, no model weights downloaded.** See §I of `M6_BASELINE_EXECUTION_CONTRACT_AUDIT.md`.

Companion artifacts: `M6_BASELINE_EXECUTION_CONTRACT_AUDIT.md` (sections B, C, G, H, I, plus §K
seed-contract re-audit, §L source-pin verification and §M checkpoint-selection audit),
`M6_BASELINE_SOURCE_GAPS.json` (machine-readable), and
`M6A1_THIRD_PARTY_SOURCE_PINS_PROVENANCE.json` (current source-pin provenance; the A1 record
`third_party_source_pins.sha256` is preserved byte-unchanged).

---

## A. Per-method source / provenance table

| ID | Method | Spec code link (§30) | Source actually used | Pinned commit | Identity evidence | Source status |
|----|--------|---------------------|----------------------|---------------|-------------------|---------------|
| E01 | FAS-Aug | R4 → `github.com/RizhaoCai/FAS_Aug` (**redirect stub**) | `github.com/RizhaoCai/FAS-Aug` | `0da1dd79bad00e225b8cb6977c7f3f06ee8f7517` | stub README names it; README cites arXiv:2409.03501 = spec R4; account = paper's first author | `OFFICIAL_REPO_PINNED` |
| E02 | Frequency Substitution | — (none; spec-defined) | spec §8.2 only | N/A | spec declares `CONTROLLED_ADAPTATION` | `SPEC_DEFINED` |
| E03 | STDN | R3 → `github.com/yaojieliu/ECCV20-STDN` | same | `c79f1f8c615d2b8471b3df29da881bb18dd54c90` | repo description names the ECCV 2020 paper; arXiv:2007.09273 abstract links this exact URL | `OFFICIAL_REPO_PINNED` |
| E04 | Physics-Guided STD | R2 → **Code column empty** | paper only (arXiv:2012.05185) | N/A | see §E | `NO_OFFICIAL_CODE_RELEASE_LOCATED_AS_OF_2026_09_21` |
| E05 | PCGAN | R7 → **Code column empty** | paper only (arXiv:2604.09018) | N/A | see §D | `NO_OFFICIAL_CODE_RELEASE_LOCATED_AS_OF_2026_09_21` |
| E06a/b/c | DSDG variants | R5 → FaceX-Zoo `addition_module/DSDG` | same | `16b793a7564a4b9308cf94e62bdb2ffacb3a725a` | pinned under Amendment A1; unchanged | `OFFICIAL_REPO_PINNED` |
| E07a/b/c | DiffFAS variants | R6 → `github.com/murphytju/DiffFAS` | same | `23f40519ec25a833ebc06842aa6fbab74fad4d15` | pinned under Amendment A1; unchanged | `OFFICIAL_REPO_PINNED` |

Newly pinned in M6A1: **E01 FAS-Aug** and **E03 STDN**. DSDG and DiffFAS were already pinned under
Amendment A1 and were **not re-pinned or modified**.

---

## A.1 E01 — resolution of the FAS-Aug source ambiguity

The spec's R4 Code column is `https://github.com/RizhaoCai/FAS_Aug` (underscore). That repository
exists but is a **one-file redirect stub**:

- commit `2ae330a6fe5d63abb2bf65fdcc8252e1353f4d18`, single file `README.md`:
  ```
  # FAS_Augmentation
  Please check https://github.com/CeceliaSoh/FAS-Aug or https://github.com/RizhaoCai/FAS-Aug
  ```

Per M6A1 (“Do NOT select a random implementation merely because the method name matches … if source
identity is ambiguous: mark `BLOCKED_BY_SOURCE_GAP`”), both named candidates were cloned and compared
rather than one being chosen. **The ambiguity is resolved by proof of identity, not by preference:**

| Candidate | HEAD | Tree | Files | Size |
|-----------|------|------|-------|------|
| `CeceliaSoh/FAS-Aug` | `0da1dd79bad00e225b8cb6977c7f3f06ee8f7517` | `462fb9582eca0370145db41745367a20cd7448e0` | 470 | 441 M |
| `RizhaoCai/FAS-Aug` | `0da1dd79bad00e225b8cb6977c7f3f06ee8f7517` | `462fb9582eca0370145db41745367a20cd7448e0` | 470 | 441 M |

`diff -rq --exclude=.git` between the two working trees produced **no output**. Identical commit,
identical tree hash, identical content — one is a fork/mirror of the other, so the choice cannot
change any scientific value. The commit is `release updating`, dated 2024-09-18.

Authorship evidence (repo `README.md`, verbatim):

> The code repo for the paper ["Towards Data-Centric Face Anti-Spoofing: Improving Cross-domain
> Generalization via Physics-based Data Synthesis"](https://arxiv.org/pdf/2409.03501v1).

> ```
> @article{cai2024towards,
>   title={Towards Data-Centric Face Anti-Spoofing: ...},
>   author={Cai, Rizhao and Soh, Cecelia and Yu, Zitong and Li, Haoliang and Yang, Wenhan and Kot, Alex},
>   journal={arXiv preprint arXiv:2409.03501}, year={2024}}
> ```

arXiv:2409.03501 is exactly the paper in spec §30 R4, and both GitHub accounts correspond to the
paper's first two authors. **Canonical pin: `https://github.com/RizhaoCai/FAS-Aug` @ `0da1dd79…`** —
chosen because the spec's own stub lives under that account and Rizhao Cai is the first author. The
byte-identical mirror is recorded so the choice is auditable and reversible.

`third_party/source_cache/fas_aug_spec_stub/` retains the stub repository as evidence.

---

## B. Binary weight inventory (required by M6A1 §12)

Rule applied: record what was present, remove it from the source-only cache, record the removal,
never alter a scientific source file.

| Cache | Weight files found at checkout | Action | Source files altered |
|-------|-------------------------------|--------|----------------------|
| `fas_aug` | **none** | — | no |
| `stdn` | 4 (see below) | **removed** | no |
| `facexzoo` (DSDG) | 1, `addition_module/DSDG/DUM/checkpoint/CDCN_U_P1.pkl` (removed under A1) | already removed | no |
| `difffas` | **none** | — | no |

### stdn — repository-embedded checkpoint, fetched with the source, never used, removed

**No external or separate model-weight download was performed.** These four files were embedded in
the upstream Git repository and therefore arrived as part of the ordinary source checkout. They were
inventoried, **never used**, and removed from the source-only working tree.

| Path (repo-relative) | sha256 |
|----------------------|--------|
| `data/pretrain/checkpoint` | `bcd1252e…` |
| `data/pretrain/ckpt-50.data-00000-of-00001` | `1504f86e…` |
| `data/pretrain/ckpt-50.index` | `0cc93458…` |
| `data/pretrain/ckpt-50.meta` | `2eda16e3…` |

Full-length digests are in `third_party/source_pins.json`. After removal, `git status` inside the
`stdn` cache shows exactly four deletions and no modification to any `.py`, `.md` or data file.

### fas_aug — large files are method assets, not weights

The 441 M checkout contains **zero** files matching any weight extension
(`.pt .pth .pkl .ckpt .tar .h5 .onnx .npy .pb .bin .safetensors .caffemodel`, `ckpt*`, `checkpoint*`).
The 32 files over 1 MB are the augmentation assets the operators require and are part of the method:
90 background plates (`data/background/`), 190 moiré textures (`data/MPTexture/`), 48 blue-noise
textures (`data/noiseTexture/`), plus 19 RGB and 7 CMYK ICC colour profiles under `data/profile/`.
All 11 RGB and all 7 CMYK profile names referenced by the code are present.

---

## C. Pretrained-weight *dependencies* discovered (not downloaded)

These are third-party weights that the official code **requires at run time**. None was downloaded.
Each is an execution blocker tracked in §F of `M6_BASELINE_SOURCE_GAPS.json`.

**“NOT_FETCHED” is not “unavailable.”** For E06c the pinned README gives an explicit download link and
an expected local path, so the weight is externally available and simply has not been fetched, because
M6A1 forbids downloading model weights. Its local sha256 is therefore `UNKNOWN`. Status:
`REQUIRED_OFFICIAL_EXTERNAL_WEIGHT_NOT_FETCHED`. Verbatim, from
`addition_module/DSDG/README.md` (Training → DSDG, bullet 1):

> Download the LightCNN-29 model from this
> [link](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view) and put it to
> `./ip_checkpoint`.

For E07c no official link is given anywhere in the DiffFAS repository, so that dependency is weaker
still: unlinked **and** not fetched.

| Method | Required weight | Referenced by | Local status | Consequence |
|--------|-----------------|---------------|------------------|-------------|
| E06a/b/**c** | LightCNN-29 v2 (`ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar`) | `train_generator.py:72–83`; `netIP` computes `loss_ip` (λ_ip=**1000** under the official entrypoint) | **NOT_FETCHED** (officially linked in the README) | E06c cannot train until an owner-authorized acquisition step |
| E07a/b/**c** | PADISI-trained conditioning encoder (`--pretrain_classifier`, default `./PADISI.pkl`) | `FAS_train.py:34,254`; `models/unet_autoenc.py:73–82` `encoder()` does `torch.load(path)` | **NOT_FETCHED** (no official link given) | E07c cannot train without it or an owner-approved substitute |
| E03 | none for training (the removed `ckpt-50` is inference-only) | — | removed | not blocking |

**E07c note.** The dependency is unconditional. `diffusion.training_losses` calls `model(...)`
without `cond`, so `BeatGANsAutoencModel.forward` takes the `cond is None` branch and calls
`self.encode(x_cond, encoder)` on **every** training step. `use_pair=false` (DEV-021) changes only
whether the content image is concatenated into `x`; it does **not** remove the encoder. Any
substitute encoder trained on GPAT TRAIN would change the method's conditioning features and is a
scientific decision for the owner, not an implementation detail.

---

## D. Q-08 — PCGAN official code availability

**Status:** `RESOLVED_BY_DOCUMENTED_SEARCH_NO_OFFICIAL_RELEASE_LOCATED_AS_OF_2026_09_21`

The required documented search was completed and located **no official release as of 2026-09-21**. A
repository and paper search cannot prove non-existence; this is an operational finding, not a
universal claim that no official code can exist.

| Check | Result |
|-------|--------|
| Paper identity | **Verified real.** arXiv:2604.09018, “Domain-generalizable Face Anti-Spoofing with Patch-based Multi-tasking and Artifact Pattern Conversion”, Jung, Jeong, Kim, Min, Yoo, Choi; *Pattern Recognition* 179 Part B, 113640 (2026). Matches spec §30 R7 exactly. |
| Code URL in paper | **None.** The only link in the full text is the publisher DOI `https://doi.org/10.1016/j.patcog.2026.113640`. No “code is available”, no GitHub URL, no supplementary link. |
| First author's GitHub (`SeungjinJung`) | 4 public repos — `GD-FAS`, `SCTL`, `MAYA`, `SeungjinJung.github.io`. **No PCGAN repo.** |
| GitHub search | `PCGAN face anti-spoofing`, `artifact pattern conversion anti-spoofing`, `patch-based multi-tasking face anti-spoofing` → **0 results each.** |

**Resolution:** the spec's empty R7 Code column is consistent with this search. E05 target status is
`FAITHFUL_PAPER`. Whether a faithful reconstruction is *feasible* is a separate question answered in
§F of the gaps file — it is **not**, at the benchmark's 256×256 canonical resolution, without an
owner override.

---

## E. Q-09 — Physics-Guided STD official code availability

**Status:** `RESOLVED_BY_DOCUMENTED_SEARCH_NO_OFFICIAL_RELEASE_LOCATED_AS_OF_2026_09_21`

The required documented search was completed and located **no official release as of 2026-09-21**. A
repository and paper search cannot prove non-existence; this is an operational finding, not a
universal claim that no official code can exist.

| Check | Result |
|-------|--------|
| Paper identity | **Verified.** arXiv:2012.05185, “Physics-Guided Spoof Trace Disentanglement for Generic Face Anti-Spoofing”, Yaojie Liu, Xiaoming Liu. Matches spec §30 R2. |
| Code statement in paper | “Source code and pre-trained models will be publicly available upon publication.” **No URL given.** |
| First author's GitHub (`yaojieliu`) | 5 public repos — `CVPR2019-DeepTreeLearningForZeroShotFaceAntispoofing`, `ECCV2018-FaceDeSpoofing`, `ECCV20-STDN`, `ICCVW2017-DenseFaceAlignment`, `dataset-watchdog`. **None is the PAMI work.** |
| GitHub search | `physics guided spoof trace` → 0 results. `spoof trace disentanglement` → 2 results: `yaojieliu/ECCV20-STDN` (= E03) and `gazeai/STDN-PyTorch` (**self-declared unofficial**). |

**Resolution:** the spec's empty R2 Code column is consistent with this search. E04 target status
stays `FAITHFUL_PAPER`.

### E.1 `E03_E04_LINEAGE_OVERLAP` — disclosure required; the rows stay distinct

arXiv:2012.05185 states verbatim:

> “A preliminary version of this work was published in the Proceedings European Conference on
> Computer Vision (ECCV) 2020 [31]. We extend the work from three aspects.”

where [31] is `Y. Liu, J. Stehouwer, and X. Liu, “On disentangling spoof traces …”` — i.e. **spec R3 /
E03**. The finding is therefore:

- E03 and E04 have **shared publication/method lineage**.
- E04 is an **extended journal-stage method whose preliminary version was E03**.
- `yaojieliu/ECCV20-STDN` is the released code of E04's *preliminary* version, not of E04.
- The **frozen GPAT-TransferBench specification nevertheless defines E03 and E04 as distinct
  experimental conditions.** They **remain distinct method IDs and distinct result rows.** Lineage
  establishes a disclosure obligation, not permission to merge benchmark rows.

This is consistent with the spec, which at §8.4 already instructs E04 to “reuse the STDN-compatible
geometric alignment/correction machinery”. Two disclosure obligations follow:

1. The E03 and E04 rows must be presented with their lineage stated, so that no reader treats them as
   independently derived corroboration of one another.
2. A FAITHFUL_PAPER E04 will legitimately share substantial machinery with the FAITHFUL_OFFICIAL E03
   code. That overlap must be declared in the E04 traceability table rather than presented as an
   independent reimplementation.

**The frozen specification was not altered and no row was merged.**

---

## F. Provenance registry updates

Extended in place — no parallel registry created:

- `tests/test_m0_bootstrap.py` — two registry guards made authorization-aware; see §F.1.
- `third_party/registry.yaml` — E01 and E03 rows updated with the resolved repository, pinned commit,
  `url_verification`, `source_pins` pointer, and the E04/E05 rows updated with the Q-09/Q-08
  resolutions.
- `third_party/source_pins.json` — new `fas_aug` and `stdn` entries: repository, pinned commit, tree,
  commit date/subject, checkout mode, cited files with git blob IDs / sha256 / size / line counts,
  weight inventory and removals. The existing `dsdg` and `difffas` entries keep every scientific
  field byte-unchanged — commit, tree, commit date/subject, cited files, hashes, weight records —
  with one factual completion: their `method_ids` lists omitted `E06c` and `E07c`, which Amendment A1
  itself created against these same pins. The two ids were added and the omission recorded in a
  `method_ids_note`. The gap was found by the tightened registry guard, not by inspection.

All checkouts remain source-only under `third_party/source_cache/` (git-ignored, `.gitignore:60`).

### F.1 Two M0 guards were stale and are now stricter, not looser

Pinning E01 and E03 made two guards in `tests/test_m0_bootstrap.py` fail. Both encoded the
pre-M6A1 world rather than a scientific invariant, so both were rewritten to enforce the invariant
directly. Neither was relaxed to make this milestone pass.

**`test_third_party_registry`** asserted `pinned_commit is None` for every method outside A1's
E06/E07 set. That contradicts spec §8.1 (“pin the repository commit on first setup”) and the M6A1
instruction to pin baseline sources. It now applies to **every** method with a pin: the commit must
be 40 hex characters, `source_pins` must point at `third_party/source_pins.json`, and the commit
must actually be recorded there under a source that lists this `method_id`. The old rule could not
detect an unrecorded or mismatched pin on E06/E07; the new one can, and it immediately found the
`method_ids` omission described in §F.

**`test_no_invented_urls`** required every registry URL to appear verbatim in the spec. E01 cannot
satisfy that, because the spec's own R4 link is a redirect stub and the repository it names is not
in the spec text. Deleting the URL would have hidden which code E01 runs. A URL is now accepted
when it is verbatim in the spec **or** when `source_pins.json` records it as a pinned repository (or
a recorded byte-identical mirror) whose `spec_code_link` *is* verbatim in the spec. Every registry
URL therefore still traces back to the frozen document, through one recorded, auditable hop.

If the owner prefers the original guards, reverting them reverts only the E01/E03 pins — no
scientific value in this audit depends on either test.
