# M4 — DSDG Official Source Analysis for the Track-A Identity-Free Adaptation

**Date:** 2026-09-20 · **Amendment:** A1 · **Deviation:** DEV-020
**Repository:** `https://github.com/JDAI-CV/FaceX-Zoo`
**Pinned commit:** `16b793a7564a4b9308cf94e62bdb2ffacb3a725a` (2022-08-12, "Update README.md")
**Relevant path:** `addition_module/DSDG`
**Provenance:** `third_party/source_pins.json` (per-file sha256 + git blob id)
**Checkout:** sparse, blob-filtered, depth 1, source only. The one binary file that arrived
(`DUM/checkpoint/CDCN_U_P1.pkl`) was removed; **no model weights were downloaded**.

Everything below is read from the pinned code, not inferred from names. Line references are to the
pinned files; no external source file is reproduced in full here.

## 1. Official live/spoof relationship

`data/generation_dataset.py` · `GenDataset_s`:

- `file_reader()` splits each list line into `img_name, label, domain_flag, spoof_label` and builds
  `make_pair_dict[label][domain_flag]`. Only lines with `domain_flag == '0'` enter
  `img_spoof_list`, so **the dataset is indexed by SPOOF samples** — one item per spoof frame.
- `__getitem__` takes the spoof line, then `img_name_live = self.get_pair(label, '1')`.
- `get_pair` is exactly `random.choice(self.make_pair_dict[label][domain_flag])`.

What is `label`? `data/make_train_list.py` writes `label = video_name[4:6]`. OULU-NPU video names are
`Phase_Session_UserID_Type` (e.g. `1_1_01_1`), so `video_name[4:6]` is the **user id**. The shipped
`train_list/train_list_oulu_p1.txt` confirms it: `1_1_01_1/… 01 1 1`.

**Conclusion (matches the prompt's stated understanding):** the official relation is
*same-subject, online, random*: for each spoof frame, one live frame of the **same person** is drawn
uniformly at random, re-drawn every epoch. There is no materialized pair list.

The item returned is `{'0': spoof_crop, '1': live_crop, 'type': spoof_type}` where `spoof_type` maps
the OULU spoof label `2,3,4,5 → 0,1,2,3`.

## 2. Exact loss inventory

`train_generator.py`, per batch, with `img_nir` = spoof, `img_vis` = live,
`img = cat(img_nir, img_vis)`, `rec = netG(cat(z_cls, z_nir, z_vis))`,
`rec_nir = rec[:, 0:3]`, `rec_vis = rec[:, 3:6]`, and `netIP` a frozen LightCNN-29v2 identity
extractor applied to grayscale 128×128 crops with L2-normalised outputs:

| term | exact official expression | what it relates | requires same person? |
|---|---|---|---|
| `loss_rec` | `reconstruction_loss(rec, img, True) / 2` | each reconstruction vs **its own** input (channel-wise halves of the same tensor) | **No** — self-preservation |
| `loss_kl` | mean of `kl_loss` over `(mu_nir, logvar_nir)`, `(mu_vis, logvar_vis)`, `(mu_a, logvar_a)`, / 3 | each posterior vs its prior | **No** |
| `loss_mmd` | `lambda_mmd * abs(z_nir.mean(dim=0) - z_vis.mean(dim=0)).mean()` | **batch-level** mean of the two latent populations | **No** — a distribution-level alignment, not a per-sample identity match |
| `loss_ip` | `lambda_ip * (MSE(rec_nir_fc, nir_fc.detach()) + MSE(rec_vis_fc, vis_fc.detach())) / 2` | `rec_spoof` vs **spoof input**, and `rec_live` vs **live input**, separately | **No** — self-preservation (this confirms case **A** of the prompt's §12) |
| `loss_cls` | `lambda_type * CrossEntropy(netCls(z_cls), label_spoof)` | attack latent vs spoof-type label | **No** (needs attack type, not identity) |
| `loss_ort` | `lambda_ort * abs((z_cls * z_nir).sum(dim=1).mean())` | two latents of the **same** sample | **No** |
| `loss_pair` | `lambda_pair * MSE(rec_nir_fc, rec_vis_fc)` | identity feature of the **reconstructed spoof** vs identity feature of the **reconstructed live** | **YES — this is the only term whose correctness requires the two images to be the same person** |

Warm-up: for `epoch < 2` every term except `loss_rec` is scaled by `0.01`; afterwards all terms are
summed unweighted (their own `lambda_*` already applied).

`lambda_pair` is an independent CLI argument, `--lambda_pair`, default `0.5`. Setting it to `0`
removes `loss_pair` exactly and touches nothing else. **The prompt's §9 understanding is confirmed
with no correction needed.**

## 3. One official detail worth recording

The optimizer is
`optim.Adam(list(netE_nir.parameters()) + list(netE_vis.parameters()) + list(netG.parameters()))` —
**`netCls` parameters are never updated**. `loss_cls` still backpropagates into `z_cls` and therefore
shapes `netE_nir`, but the classifier head itself stays at its initialisation. This is the official
behaviour at the pinned commit and is not changed by the adaptation.

## 4. The binary collapse is mathematically degenerate — stated honestly

`networks/__init__.py::define_G(hdim, attack_type)` builds `Cls(hdim, attack_type)`, whose head is
`nn.Linear(hdim, attack_type)`. DSDG-BIN collapses every spoof type into one class, i.e.
`attack_type = 1`. `CrossEntropyLoss` over a **single** logit is identically
`-log(softmax(x)[0]) = -log(1) = 0` with zero gradient for every input.

**Therefore `loss_cls` contributes nothing under the binary collapse**, and `z_cls` is then shaped
only by `loss_rec`, `loss_kl` and `loss_ort`. This is recorded rather than worked around:
`attack_macro` and `attack_raw` are **not** secretly substituted, and no replacement supervision is
invented. Track A forbids attack-type supervision, so a vanishing attack-type loss is the correct
consequence, not a bug to patch.

## 5. What DEV-020 changes, and what it does not

**Changed — exactly two things:**

1. **The live/spoof relation.** The official same-subject online `random.choice` is replaced by the
   frozen common fair pair from `manifests/pairs_train_v1.parquet`
   (`source_spoof_id → target_live_id`). This gives every Track-A method the same dataset, the same
   frozen TRAIN split, the same source population and the same target-selection protocol across
   CASIA, MSU and SiW.
2. **`lambda_pair = 0`.** `loss_pair` forces the identity features of the two reconstructions to
   match. Under the common fair pair the two images are deliberately **different** people (CASIA/MSU)
   or different videos and content groups (SiW), so optimising that term would push the generator
   toward an assertion that is false by construction.

**Unchanged:** `loss_rec`, `loss_kl`, `loss_mmd`, `loss_ip`, `loss_cls`, `loss_ort`, the encoder /
decoder / classifier architecture, `hdim`, the reparameterisation, the warm-up schedule, the
identity-preserving network and the 256→128 interpolation before `netIP`. Nothing was removed for
convenience; only the term whose correctness mathematically requires shared identity was disabled.

**Subject-metadata firewall.** `subject_id_global` is not used for sampling, loss, model input or
class target. The Track-A training relation is fully determined by
`(source_spoof_id, target_live_id)` from the common pair manifest, which carries no identity
requirement for SiW and only an *inequality* constraint for CASIA/MSU.

## 6. Classification

`CONTROLLED_ADAPTATION` — **not** `FAITHFUL_OFFICIAL`. DSDG-BIN-IDFREE may never be described as
reproducing official DSDG training.
