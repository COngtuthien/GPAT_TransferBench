# GPAT-TransferBench v1.0 — E05 training-runner owner resolution (M6D4c)

Status: **OWNER-APPROVED · ADDITIVE · CONTRACT ONLY**.
E05 remains **PCGAN (controlled architecture resolution)**, **CONTROLLED_ADAPTATION**,
**IMPLEMENTED_NOT_EXECUTED**. This addendum records the owner's explicit controlled
reconstructions for the three gaps in M6D4b. It authorizes no execution in M6D4c.
M6D4b remains byte-identical historical **STOP_AND_REPORT** evidence; its unresolved
findings were correct at that milestone. M6D4a is neither rewritten nor requalified.

The machine-readable companion is
[`e05_training_runner_resolution.yaml`](../../../configs/amendments/e05_training_runner_resolution.yaml).
It uses the JSON subset of YAML, enabling static parsing with only Python's standard
library. It is an additional immutable contract input for a future runner; it does
not modify the frozen method YAML, A2, A5, or the existing PCGAN implementation.

## Authority and attribution

Starting branch: `m6-baselines`. HEAD and local `origin/m6-baselines` both equal
`5adb76285b9b9c4dc25bb4a33f3be7d8575a8abc`; starting worktree clean, divergence `0 0`.
Authority is the frozen E05 contract and preserved M6D4b resolutions, augmented
only by the explicit owner decisions in this addendum.

New estimator, discriminator and iteration choices are **owner-approved controlled
reconstructions**, not paper facts or recovered private author code. They MUST NOT
be attributed as `PAPER`, `OFFICIAL_PCGAN`, `AUTHOR_SPECIFIED`, native PCGAN,
faithful PCGAN, or official reproduction.

The nearest executable predecessor is `taesungp/swapping-autoencoder-pytorch`,
commit `6baa180f1184ee79a6b967f9d80ee0e02a979ac7`. Behavior derived from it is labeled
**CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED**. Owner-specific decisions are
**CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**. The source/line/hash evidence
already reviewed in M6D4b is retained; M6D4c does not reexecute or import that source.
Existing distance semantics retain their PAPER / frozen-evidence provenance.

## Explicit pair and preserved generator objective

One explicit pair contains distinct FP32 source and target tensors, each
`[1,3,256,256]`. Frozen batch size remains one. No even-minibatch swap is allowed.

```text
(z_pat_src, z_con_src) = E(x_src)
(z_pat_tgt, z_con_tgt) = E(x_tgt)
r_src = G(z_pat_src, z_con_src)
m     = G(z_pat_src, z_con_tgt)
norm2(a-b)[n] = sqrt(sum over C,H,W of (a[n]-b[n])**2)
```

Spatial codes remain `[1,8,128,128]`, global codes `[1,2048]`.

**Owner resolution #1:**

```text
L_rec = mean_n norm2(x_src-r_src)[n]
```

This is **source reconstruction only**: at batch one, one unsquared Euclidean norm
over the source image. There is no target reconstruction term, mean over source
and target reconstructions, MSE, RMS, L1, pixel-count normalization, or whole-batch
norm. The norm itself retains PAPER / existing frozen evidence provenance. The
explicit source-only estimator is **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**.

PCGAN Eq.2 supplies generic reconstruction semantics but leaves the explicit-pair
estimator open. Eq.4 and the already resolved `L_advrec` identify the source
reconstruction path. The owner selects the minimal source-only estimator to avoid
inventing another target-reconstruction contribution.

The remaining M6D4b generator semantics are preserved exactly:

```text
L_recblur = mean_n norm2(B(x_tgt)-B(m))[n]
L_advrec  = mean softplus(-D(r_src))
L_advmix  = mean softplus(-D(m))
L_pat     = mean softplus(-PatchD(reference_features, mixed_features))
L_G_total = L_rec + L_recblur + L_advrec + L_advmix + L_pat
```

All five weights are exactly **1**. `alpha=0.2` and `beta=1e-6` retain frozen PMN
scope and do not weight these losses. `D` and `PatchD` output raw logits; the means
cover batch and scalar scores (`[1,1]` for image-D, `[8,1]` for PatchD).

`B` is exactly A5 `avg_pool2d(kernel_size=2, stride=2, padding=0,
ceil_mode=False, count_include_pad=False)`, independently applied to target and
mixed images. It maps 256 to 128 with no fallback, learned parameter or detach.
The blur distance sums C,H,W inside the square root, then takes the outer mean.

For `L_pat`, independent pinned random crop draws come from source (reference) and
mixed images (candidate). Use `extract_features(crops_ref, aggregate=True)` and
`extract_features(crops_mixed, aggregate=False)`, followed by
`discriminate_features(reference_features, mixed_features)`. Reference features
are averaged over eight crops then expanded to eight; candidates stay separate.
There is no generator-branch detach or obsolete PatchD forward substitution.

## Owner resolution #2 — discriminator objectives

For the D step, both generated fake tensors are detached:

```text
d_src = D(x_src)
d_tgt = D(x_tgt)
d_rec = D(r_src.detach())
d_mix = D(m.detach())

L_D_real  = 0.5 * (mean softplus(-d_src) + mean softplus(-d_tgt))
L_D_rec   = mean softplus(d_rec)
L_D_mix   = mean softplus(d_mix)
L_D_image = 1.0 * L_D_real + 0.5 * L_D_rec + 0.5 * L_D_mix
```

The logistic softplus family is compatible with the resolved logit semantics and
pinned predecessor. Image-D weights **1 / 0.5 / 0.5** are explicitly
**CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED**. Using **both source and target
as equally weighted real samples** is a symmetric explicit-pair owner choice,
**CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**. It is not from the PCGAN paper.

**Owner resolution #2B:** all three D-step PatchD crop draws are independent:

```text
crops_ref  = independent random crops from x_src
crops_pos  = another independent random crop draw from x_src
crops_fake = independent random crop draw from m.detach()

ref_feat  = extract_features(crops_ref,  aggregate=True)
pos_feat  = extract_features(crops_pos,  aggregate=False)
fake_feat = extract_features(crops_fake, aggregate=False)
p_real = discriminate_features(ref_feat, pos_feat)
p_fake = discriminate_features(ref_feat, fake_feat)

L_D_patch_real = mean softplus(-p_real)
L_D_patch_fake = mean softplus(p_fake)
L_D_patch = 1.0 * L_D_patch_real + 1.0 * L_D_patch_fake
L_D_total = L_D_image + L_D_patch
```

Crop/feature interface provenance is **A2 pinned executable architecture / M6D4a
qualified**. The positive-versus-fake construction and patch weights **1 / 1** are
**CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED**. The unit image-plus-patch
combination is **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**; no extra coefficient.

No R1, patch R1, lazy-R1 scaling, gradient penalty, additional patch regularizer,
VGG/perceptual loss, GPAT identity loss or landmark loss is introduced.

## Parameter ownership and owner resolution #3 — complete iteration

Preserve disjoint groups: **G optimizer = Encoder + Generator**;
**D optimizer = Image D + Patch D**. Both use frozen **Adam, lr=1e-6,
betas=(0.9,0.999)**. Do not introduce weight decay. Do not adopt upstream lr=.002,
beta1=0, beta2=.99, or lazy-R1 scaling. This is a contract, not a live parameter
membership or optimizer-state measurement.

One benchmark iteration is exactly one complete **D → G** cycle:

1. At current iteration `t`, generate `r_src_D` and `m_D` with current E/G parameters
   from the explicit pair. Detach both before the image-D/PatchD objectives.
   Compute `L_D_total`; apply exactly one D Adam update. E/G parameters must not update.
2. After D completes, **recompute the encoder and generator paths** for `r_src_G`
   and `m_G` with the **same explicit source/target pair**. Do not reuse stale
   pre-D fake tensors. Compute all five G losses using the now-updated
   discriminators. D/PatchD parameters are frozen for parameter updates, while
   gradients propagate through their computation to generated images and E/G.
   Apply exactly one G Adam update. D/PatchD parameters must not update.

D-first ordering is **CONTROLLED_RECONSTRUCTION_PREDECESSOR_DERIVED**.
Grouping both applications into one benchmark iteration is
**CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION** for the frozen budget. It does not
inherit the predecessor's separate alternating outer calls or extra R1 applications.
The PCGAN paper is not claimed to specify this exact optimizer-state convention.

`benchmark_iteration` starts at **0**. Both steps share current `t`. Increment
exactly once, **after the G step successfully completes**, `t → t+1`. Individual
Adam applications do not increment this counter. Terminal scientific state is
`benchmark_iteration=4000`: **4000 complete D→G cycles**, implying **4000 D and
4000 G optimizer applications** in an uninterrupted complete scientific run.
This application-count consequence is **CONTROLLED_RECONSTRUCTION_OWNER_RESOLUTION**.
These are future normative counts; M6D4c optimizer applications are **0**.

## Stochasticity, checkpoint and execution boundaries

Generator StyleGAN noise remains enabled with fresh standard-normal `[B,1,H,W]`
draws according to the pinned architecture; initial trainable strength is zero.
No `fix_noise`. Pinned crops remain eight random 128×128 crops per image, random
horizontal sign, independent x/y scales uniform in `[1/8,1/4)`, offsets
`(2*U-1)*(1-scale)`, bilinear grid sampling, zeros padding, `align_corners=False`.
Keep independent draws where specified; no center-crop substitution.
M6D4c executes no noise or crop draws.

Frozen FP32-only policy, TF32/AMP/autocast off, cuDNN benchmark off, and experiment
seeds `[42,1337,2026]` remain unchanged. No TEST use. Checkpoint policy remains
**BASELINE_FINAL_STATE_V1**, terminal iteration **4000**, no VAL or TEST selection,
no best seed. A terminal save, if required later, occurs after completion of 4000
cycles with **zero additional optimizer applications**. M6D4c creates no checkpoint.

## Complete M6D4b A–J disposition

| Item | Component | M6D4c disposition | Basis |
| --- | --- | --- | --- |
| A | L_rec | CONTRACT_RESOLVED | Source-only owner estimator; preserved unsquared L2 |
| B | L_recblur | CONTRACT_RESOLVED | Preserved exact A5 pooling and no detach |
| C | L_advrec | CONTRACT_RESOLVED | Preserved raw-logit softplus source reconstruction |
| D | L_advmix | CONTRACT_RESOLVED | Preserved raw-logit softplus mixed image |
| E | L_pat | CONTRACT_RESOLVED | Preserved independent crops and qualified feature interface |
| F | discriminator | CONTRACT_RESOLVED | Explicit image/patch sampling, losses and weights above |
| G | parameter ownership | CONTRACT_RESOLVED | Disjoint E+G / image-D+PatchD; frozen Adam |
| H | update schedule | CONTRACT_RESOLVED | Complete D→G, fresh G paths, counter after successful G |
| I | StyleGAN noise | CONTRACT_RESOLVED | Preserved fresh pinned random noise, no fix_noise |
| J | batch-one explicit pair | CONTRACT_RESOLVED | Preserved explicit pair; A/F sampling gaps resolved |

These are **normative contract decisions**, not numerical/runtime qualification.

## Immutable bindings and static validation

The overlay binds the full SHA256 of all eight required readings, including base
E05 YAML, A5 overlay, both M6D4b STOP audits, A2, A5, M6D4a MD, and source
traceability. Full hashes are also recorded in the M6D4c audit JSON/MD.
The preflight compares each with fixed expected SHA256 and committed bytes.
It additionally verifies both historical M6D4b process files, test and preflight.
Tracked changes outside ledger/index are rejected, protecting M6D4a and implementation.

`python3 -I -S -B tools/m6d4c_e05_training_resolution_preflight.py` validates the
contract, in-memory invalid-contract rejections, hashes, ledger and CRLF index.
`--contract-only` performs the contract checks before finalization.
`--rebuild-index` performs the sole preflight write, last, from 550 historical
metadata rows plus five explicit new artifact paths. No historical index target
is opened. No benchmark/runtime-data traversal or framework import is needed.

Final status: **E05_ARCHITECTURE_RUNTIME_QUALIFIED** (retained prior evidence),
**E05_TRAINING_RUNNER_CONTRACT_RESOLVED**,
**E05_TRAINING_RUNNER_NOT_YET_QUALIFIED**, **E05 IMPLEMENTED_NOT_EXECUTED**,
**CONTROLLED_ADAPTATION**. No runner implementation, runtime/numerical qualification,
optimizer construction/application, Torch/TensorFlow import, CUDA/model execution,
environment mutation, benchmark data/image access or enumeration, scientific
checkpoint, synthetic bank, commit, or push occurs in M6D4c.
