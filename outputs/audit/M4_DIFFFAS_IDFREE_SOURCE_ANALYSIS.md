# M4 — DiffFAS Official Source Analysis for the Track-A Identity-Free Adaptation

**Date:** 2026-09-20 · **Amendment:** A1 · **Deviation:** DEV-021
**Repository:** `https://github.com/murphytju/DiffFAS`
**Pinned commit:** `23f40519ec25a833ebc06842aa6fbab74fad4d15` (2024-09-23, "Update README.md")
**Provenance:** `third_party/source_pins.json` (per-file sha256 + git blob id)
**Checkout:** depth 1, source only, 2.3 MB, **0 binary weight files**; no model weights downloaded.

Everything below is traced through the pinned code. The flag's existence was not taken as evidence
of its effect.

## 1. What the dataset returns

`FAS_dataset.py` — all three dataset classes (`OCIMDataset`, `WMCADataset`, `PADISIDataset`) share
the same shape:

- the JSON gives a list of `(content_path, gt_path)` pairs;
- `style_folder = os.path.dirname(gt_path)`, and
  `style_file = random.choice(os.listdir(<style_folder>))`;
- the item is `{'content': content_img, 'style_spoof': style_img, 'GT': gt_img}`.

Two consequences matter:

- the style guide is drawn **online and at random from the GT's own style folder** — no seed, no
  fixed choice;
- `random.choice(os.listdir(...))` enumerates the whole folder, so **the GT file itself is inside the
  official support**. A self-guide is officially possible (relevant to §24 of the amendment task).

## 2. What training does with each quantity

`FAS_train.py::train` builds `x_start = batch['GT']`, `cond_input = [batch['content'],
batch['style_spoof']]` and calls `diffusion.training_losses(..., use_pair=args.use_pair)`.

`diffusion.py::training_losses` (pinned):

```
img, target_pose = cond_input                     # img = content, target_pose = style_spoof
cond_mask = prob_mask_like((B,), prob=prob, ...)  # classifier-free-guidance dropout mask
img[~cond_mask] = zeros                           # written to, see §3
x_t = self.q_sample(x_start, t, noise=noise)      # noise = randn_like(x_start)
model_input = torch.cat([x_t, img], 1) if use_pair_flag else torch.cat([x_t], 1)
model_output = model(x=model_input, encoder=encoder, t=..., cond_mask=cond_mask,
                     x_cond=target_pose, prob=prob, means_size=..., var_size=...)
target = noise                                    # ModelMeanType.EPSILON
terms["mse"] = F.mse_loss(target, model_output, reduction='elementwise_mean')
```

`create_gaussian_diffusion(betas, predict_xstart=False)` with default `learn_sigma=True` fixes
`model_mean_type = EPSILON` and `model_var_type = LEARNED_RANGE`, so the `vb` branch is active. That
branch calls `_vb_terms_bpd(model=lambda *a, r=frozen_out: r, ...)`, which forwards to
`p_mean_variance` **without** `x_cond`; that takes the `x_cond is None` path and simply returns the
frozen tensor. **Content does not enter the variational term either.**

`models/unet_autoenc.py::forward` is reached during training with `cond=None`, so it takes the
branch `x_cond = cond_mask.view(-1,1,1,1) * x_cond` and encodes **`x_cond` = `style_spoof`** into
the conditioning stack. The content image is never passed as `cond` during training.

## 3. `use_pair` dependency table

| quantity | `use_pair=True` | `use_pair=False` | requires same identity? | role |
|---|---|---|---|---|
| `GT` (`x_start`) | diffused to `x_t`; defines `noise`'s shape | identical | No | the image being modelled |
| `noise` | `randn_like(x_start)`; the **training target** | identical | No | epsilon target |
| `style_spoof` (`target_pose`) | passed as `x_cond`, masked by `cond_mask`, encoded into the conditioning stack | identical | No | spoof-style conditioning |
| `content` (`img`) | **concatenated into the model input** (`cat([x_t, img], 1)`, `in_channels = 3+3`) | **absent from the model input**; not `x_cond`; not `cond`; not the target; does not affect `noise`; absent from the `vb` term | Yes (when paired) | **`use_pair=False` → INERT_API_PLACEHOLDER** |
| `cond_mask` | CFG dropout mask | identical | No | drops the conditioning with probability `1 - prob` |

`FAS_train.py::main` sets `conf.in_channels = 3+3` only in the `use_pair == True` branch; otherwise
the model is built from `get_model_conf()` with its default 3 input channels, so a content tensor
could not be concatenated even if one tried.

**Answer to the task's §17:** yes — `use_pair=True` concatenates the content image with the noisy
`x_t`; `use_pair=False` removes the content image from the diffusion model input. And, proven by
the trace above, with `use_pair=False` the content image also reaches **no** loss, **no** encoder
condition, **no** target and **no** noise initialisation during training.

**One honest caveat.** The line `img[~cond_mask] = zeros` still *writes into* the content tensor
before the branch. That is a write, not a read: when `use_pair=False` the tensor is never read
again, and `cond_mask` itself is a Bernoulli draw that does not depend on the content's values. So
the content's **values** are mathematically inert. The tensor is still materialised and moved to the
GPU, which is a cost, not a dependency.

**Verdict: `content_training_role = INERT_API_PLACEHOLDER` under `use_pair=false`.** The
identity-free design proceeds. (Had any dependency survived, DEV-021 would have been stopped and
reported instead.)

## 4. Inference (for later milestones, not executed here)

`p_mean_variance` and `FAS_sample.py` take `x_cond = [cond, target_pose]` and call
`model.forward_with_cond_scale(x=cat([x, target_pose], 1) if use_pair else cat([x]), ..., cond=cond,
...)`. Generation therefore has its own content/`cond` path, which is a separate question from
training and is **not** settled by this document. Track-A generation fairness is fixed by the
amendment instead: the target live image and the source spoof image must come from the same common
pair universe as every other source-conditioned method.

## 5. What DEV-021 changes, and what it does not

**Changed — exactly two things:**

1. **`use_pair = false`.** This is an **official code path**, not a modification: no line of the
   official model, diffusion or loss code is altered. It removes the same-identity
   content-conditioning requirement, which is what made native DiffFAS impossible for SiW-Mv2.
2. **Deterministic guide selection.** The official `random.choice` is replaced by a frozen raw-byte
   SHA-256 ranking so the benchmark is reproducible:

```
source_digest    = SHA256(UTF8("gpatbench.trackA.difffas.guide.source.v1|"
                               + gt_spoof_sample_id + "|" + str(split_seed))).digest()   # RAW 32 bytes
candidate_digest = SHA256(source_digest
                          || UTF8("|gpatbench.trackA.difffas.guide.candidate.v1|"
                                  + guide_spoof_sample_id)).digest()
rank key         = (int.from_bytes(candidate_digest, "big", signed=False), guide_spoof_sample_id)
```

first by ascending key, over the eligible pool in canonical lexical `guide_sample_id` order. No
Python `random`, no NumPy random, no filesystem order.

**Guide eligibility (Track A):** M2 COMPLETE · TRAIN · SPOOF · **same dataset**. Subject identity is
**not** part of eligibility, and `attack_raw` is **not** part of eligibility — all attacks collapse
into one binary spoof style pool, which is precisely what lets SiW participate without identity
metadata. `style_id = SPOOF_BINARY` is benchmark provenance only; the dataset id is never fed as a
semantic class label, and the guide image itself remains a real TRAIN spoof image.

**Self-guide policy:** the official support includes the GT file, so the Track-A deterministic
support includes it too. The count is reported diagnostically and **no selection is modified after
the fact to reduce it**.

**Unchanged:** the diffusion process, the beta schedule, the epsilon parameterisation, the learned
variance range, the classifier-free-guidance dropout, the UNet, the style encoder and the
`GT`/`style_spoof` roles.

## 6. Classification

`CONTROLLED_ADAPTATION_USING_OFFICIAL_UNPAIRED_CODE_PATH` — **not** `FAITHFUL_NATIVE`.
DIFFFAS-BIN-IDFREE may never be described as native DiffFAS.
