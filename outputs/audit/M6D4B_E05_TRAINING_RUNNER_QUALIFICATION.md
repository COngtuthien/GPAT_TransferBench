# M6D4b — E05 training-runner qualification: STOP

**STOP_AND_REPORT — unresolved scientific training choices.** Phase A review is complete;
required numerical/update semantics are not all resolved. No runner or numerical loss
implementation was added, no Adam constructed, and no diagnostic optimizer application run.

E05 remains **E05_ARCHITECTURE_RUNTIME_QUALIFIED**,
**E05_TRAINING_RUNNER_NOT_YET_QUALIFIED**, **IMPLEMENTED_NOT_EXECUTED**,
**CONTROLLED_ADAPTATION**. Reporting label: PCGAN (controlled architecture resolution).

## Authority and reconciliation

Laptop HEAD and origin/m6-baselines were `370d36bcbd670861aa5ebf120fa1ee0bd13d9d37`,
clean, divergence 0 0. GPU began at `05e90efa6c36e11405822c79fed8511c2179f043`
with exactly two tracked modifications and fourteen untracked M6D4a files.
All sixteen mapped to the authoritative commit and matched its content SHA256.
Every file was archived preserving relative paths under:

`/home/student20261/workdir/GPAT_TransferBench_runtime/recovery/M6D4a_pre_M6D4b_sync_20260924T104404_313345Z`

All archive/current/authoritative hashes were verified before explicit-path tracked
restore and exact untracked-file unlink. Clean-worktree gate preceded the exact
requested fetch and fast-forward merge. GPU HEAD and origin now equal `370d36b`,
clean, divergence 0 0; every resulting committed/worktree file matches its archive.
The JSON retains all four hashes per path and synchronization evidence. Archive
remains outside Git. No cleanup ledger row, reset, broad clean/remove or force checkout.

Authority order: frozen E05 benchmark contract; [PCGAN Eq.1–5 and training text](https://arxiv.org/html/2604.09018v1#S3.SS1.SSS2);
A2/A5; pinned architecture or compatible implementation details only. Full required
repository inputs were reviewed, plus frozen specification section 8.5 and the exact
pinned predecessor code. Source SHA256 and symbol line ranges are in the JSON.
The source remains an architecture/predecessor basis, not official PCGAN code.

## Phase A evidence table

In this table, `d(a,b)[n]=sqrt(sum_{c,h,w}(a[n,c,h,w]-b[n,c,h,w])^2)`;
`r_src=G(z_pat_src,z_con_src)` and `m=G(z_pat_src,z_con_tgt)`.
Resolved formulas are evidence mappings only; none were numerically executed.

| Item | Topic / result | Evidence | Resolution | Remaining gap |
| --- | --- | --- | --- | --- |
| A | L_rec — **PARTIALLY_RESOLVED** | PCGAN Eq.2; A5 distance-scope clause; adapter.loss_mapping | Operands x and G(E(x)). Unsquared Euclidean norm: d(a,b)[n]=sqrt(sum over C,H,W of (a[n]-b[n])**2), then arithmetic mean over sampled images. No pixel-count normalization; not MSE, RMS, L1, or whole-batch norm. | The empirical reconstruction sample set for one explicit source/target pair is not fixed: source only, target only, or a mean of both. Eq.4 identifies source reconstruction for L_advrec; Eq.2 retains generic x. Adding a second reconstruction contribution or choosing its weight cannot be silently inferred from the upstream half-batch optimization. |
| B | L_recblur — **RESOLVED** | PCGAN Eq.3; A5 document and overlay; blur.blur_inputs | mean_n d(B(x_tgt),B(G(z_pat_src,z_con_tgt)))[n], d as in A. B is avg_pool2d(kernel_size=2,stride=2,padding=0,ceil_mode=False,count_include_pad=False). Sum C,H,W at 3x128x128 inside sqrt, mean over pairs outside. Apply independently; no detach; batch one leaves a single norm. | None for this component. |
| C | L_advrec — **RESOLVED** | PCGAN Eq.4; adapter.loss_mapping; pinned discriminator.py and stylegan2_layers.Discriminator; pinned loss.gan_loss | r_src=G(z_pat_src,z_con_src). Let ell_D be the pinned scalar raw logit, shape [1,1]. D_probability=sigmoid(ell_D). -log(sigmoid(ell_D(r_src)))=softplus(-ell_D(r_src)); mean over samples and the singleton score dimension. This is a stable evaluation of the specified generator objective, not adoption of upstream weights. | None for this component. |
| D | L_advmix — **RESOLVED** | Frozen explicit source/target mapping; PCGAN Eq.3-4; A5 application branches; pinned discriminator output | m=G(z_pat_src,z_con_tgt) with spatial source first and global target second. mean_n softplus(-ell_D(m)[n]); scalar logit per image; same probability interpretation as C. No upstream batch swap. | None for this component. |
| E | L_pat — **RESOLVED_ARCHITECTURE_MAPPING** | PCGAN Eq.1; A2-07; M6D4a active patch interface; pinned util.apply_random_crop and StyleGAN2PatchDiscriminator.extract_features/discriminate_features | Reference features come from random source crops; candidate features from independent random mixed crops. Use the qualified active interface discriminate_features(reference_aggregated,candidate), not obsolete BasePatchDiscriminator.forward. Reference features averaged over eight crops then expanded to eight; candidate features not aggregated. Pinned ordered concatenation supplies eight scalar raw logits. L_pat=mean over batch and eight scores of softplus(-ell_P). Probability=sigmoid(logit). Crop and reference aggregation details below are architectural inheritance, not new paper facts. No generator-branch detach. | None for this component. |
| F | Discriminator-side objective — **UNRESOLVED** | Frozen E05 losses only specify five generator terms; PCGAN Eq.1-5 and section 4.1; A2-07 architecture-only scope; A5 blur-only scope; pinned compute_image_discriminator_losses/compute_patch_discriminator_losses (not binding PCGAN losses) | Image-D real objective, reconstructed fake objective, mixed fake objective, PatchD positive/reference objective, PatchD mixed fake objective, all their relative weights and final reductions have no complete authoritative PCGAN definition. A conventional logistic realization would use softplus(-real_logit), softplus(fake_logit); that family alone does not fix the weights or sample measures. No numerical D objective selected. | Owner/source resolution required for real sample set (source/target/both), image real/rec/mix weights and means, patch positive construction (including independent same-source reference/candidate crop sets), patch fake pairing, patch real/fake weights, and image-vs-patch combination. Upstream uses image weights 1,0.5,0.5 and patch weights 1,1; these are not automatically authorized. No R1 authorized. |
| G | Optimizer parameter ownership — **SUPPORTED_NOT_INSTANTIATED** | Required M6D4b E+G versus image-D+PatchD separation; pinned get_parameters_for_mode and optimizer constructor | The supported grouping is E+G and image D+PatchD. Upstream exposes the same disjoint component groups, compatible with optimizing the specified E/G objective. This supports implementation grouping, not the upstream loss or schedule. Counts from M6D4a imply E+G=4629193 and D+PatchD=53381666. No optimizer or live parameter-group membership inventory was constructed; no duplicate-membership qualification claimed. | None for this component. |
| H | Optimizer schedule and iteration ownership — **UNRESOLVED** | PCGAN Eq.5 surrounding text and section 4.1; Frozen 4000 iteration budget; A2-06 terminal checkpoint; pinned toggle_training_mode/train_one_step inspected but not inherited | Reconstruction and mixture are both required during training. This does not define optimizer ordering, number of applications per complete iteration, whether G and D both execute each iteration, fake recomputation after an update, or counter ownership. Terminal remains iteration 4000; no diagnostic iteration was executed. | Owner/source resolution required for G/D order, frequency, applications per iteration, fake reuse/recomputation and one complete-iteration counter boundary. Upstream train_one_step alternates D first then G on successive calls, with a separate D counter and occasional extra R1 step. Its outer-step count is not a PCGAN benchmark iteration definition. |
| I | StyleGAN stochastic noise — **RESOLVED** | A2-07; M6D4a noise policy; pinned NoiseInjection.forward and generator defaults | Preserve netG_use_noise=True. NoiseInjection draws fresh standard-normal [B,1,H,W] tensors when no explicit/fixed noise exists; trainable scalar noise strength initializes to zero. No fix_noise call. Future repeatability must reset diagnostic RNG in clean processes without altering this behavior; no M6D4b seed or noise draws executed. | None for this component. |
| J | Batch-one explicit source and target — **RESOLVED_WITH_A_SAMPLING_GAP** | Frozen batch_size=1; PCGAN distinct-source/target condition; A5 mixed branch; M6D4a explicit encoding interface | Two distinct FP32 [1,3,256,256] generated tensors are explicit inputs. E(x_src)=(z_pat_src,z_con_src), E(x_tgt)=(z_pat_tgt,z_con_tgt). m=G(z_pat_src,z_con_tgt). Spatial [1,8,128,128], global [1,2048]. No even-minibatch swap; no batch-size change. This interface itself needs no new scientific assumption. | The L_rec sample set remains the A gap; interface qualification does not resolve training sampling or D objective sampling. |

## Numerical boundary and required resolution

The unsquared norm is supported; MSE is not an alternative adopted here. The norm
reduces all image coordinates per sample, then the expectation is an outer sample
mean. For batch one this is one norm. The unresolved reconstruction issue is which
members of the explicit pair enter that empirical expectation, not squaring the norm.
A source-only estimate and an average of both reconstructions can estimate the same
population expectation but produce different per-iteration gradients and RNG use.
The exact estimator must be resolved for this runner's qualification.

The discriminators return raw logits. The compatible generator probability mapping
is `sigmoid(logit)` and the stable negative logarithm is `softplus(-logit)`.
Image scores are singleton per image. Patch scores are averaged over eight crops
and then samples, with feature averaging confined to the reference branch before
classification. A negative log of the raw discriminator output is invalid.
No epsilon, clipping, extra regularizer, or upstream coefficient is introduced.

The five-term total is exactly `L_rec + L_recblur + L_advrec + L_advmix + L_pat`.
All weights remain one; alpha 0.2 and beta 1e-6 retain their existing PMN scope.
No initial loss values or external FP32 sum-equality claim exists for this STOP.

A complete discriminator objective cannot be reported as resolved. To show why
importing upstream would be a scientific choice, its image-D code combines real,
reconstruction-fake and mixed-fake logistic losses with weights **1, 0.5, 0.5**;
its PatchD uses independent crops of the same real input for reference/positive,
and mixed crops for negative, weighted **1, 1**. Its optimizer takes means and sums.
Neither A2 architecture inheritance nor A5 blur resolution authorizes these PCGAN
training weights. The generator objective alone does not determine them. Changing
real/fake balance or real sample selection changes discriminator gradients.

The upstream optimizer alternates D and G on separate outer calls (first D),
maintains its own D counter, and can make an additional lazy-R1 application.
Importing that convention would choose the interpretation of the fixed 4000-step
budget. The required PCGAN order, applications per iteration, every-iteration
cadence, fake refresh policy and complete-iteration counter boundary remain unset.
No diagnostic `0 -> 1` transition was performed.

The owner/source resolution must specify:

1. Reconstruction sample set and averaging for one explicit source/target pair.
2. Image-D real sample set, fake paths, exact real/rec/mix coefficients and reductions;
   PatchD positive/reference pairing and sampling, fake pairing, coefficients and
   reductions; the final image/patch discriminator objective. No R1 is authorized.
3. G/D order, cadence, applications per complete iteration, reuse or recomputation
   of generated inputs, and exactly when the benchmark iteration increments.

No suggested alternative in this audit is adopted as an owner decision.

## Architecture, crop, noise and optimizer boundaries

M6D4a measurements retained without remeasurement:

| Component | Parameters | Interface |
| --- | ---: | --- |
| StyleGAN2ResnetEncoder | 929,960 | spatial [1,8,128,128], global [1,2048] |
| StyleGAN2ResnetGenerator | 3,699,233 | [1,3,256,256] |
| StyleGAN2Discriminator | 28,859,521 | [1,1] |
| StyleGAN2PatchDiscriminator | 24,522,145 | crops [1,8,3,128,128], scores [8,1] |

Component ownership supports E+G (4,629,193 parameters) versus D+PatchD (53,381,666).
The prior process files contain parameter names/shapes. M6D4b did not instantiate
optimizer membership, check live duplicate identities, or audit Adam moments.
Frozen Adam remains `lr=1e-6, betas=(0.9,0.999)`; **effective instantiated kwargs are
not applicable**. No lazy-R1 scaling or weight decay is authorized. No optimizer
state exists from this milestone: exp_avg, exp_avg_sq and step are unmeasured.

The exact qualified crop operation expands each image to eight crops, samples a
horizontal sign, independent x/y scales uniformly in [1/8,1/4), and offsets from
`(2*U-1)*(1-scale)`. It samples a normalized grid through `grid_sample` with
`align_corners=False` (default bilinear interpolation and zeros padding) to 128x128.
Source/reference and mixed crop calls are independent. Reference features are
averaged over crops then expanded; candidate features stay separate. The
`return_rect` argument exists but is unused: crop metadata is not returned.
No center-crop substitution, new crop sampling, or diagnostic seed was used here.

Pinned generator noise stays stochastic, with fresh normal draws and trainable
strength; initial zero strength does not authorize fixing noise. No `fix_noise`
call or new qualification RNG experiment occurred. FP32-only, TF32/AMP/autocast
off and cuDNN benchmark off remain required for any future training qualification.
No M6D4b model process exists in which to remeasure precision behavior.

Expected connectivity for a future implementation (analytical only): reconstruction
connects E's spatial/global reconstruction branches and G; blurred reconstruction,
mixed adversarial and patch losses connect source spatial and target global
branches and G. Target-spatial and source-global output branches are not used by
the mixed path, although encoder trunk parameters are shared. Reconstruction
adversarial loss uses source spatial/global branches and G. Adversarial losses
must propagate through frozen D/PatchD to generated inputs. D steps must isolate
E/G with the resolved fake-input detachment semantics. These are requirements,
not per-parameter measured gradient inventories; no finite/nonzero-gradient,
parameter-change or isolation PASS is asserted.

## Environment and non-execution evidence

Read-only GPU verification checked Python executable/version/hash, all installed
package versions, all 67 original code files, all 67 compatibility-copy code files,
source cleanliness, both extension hashes and their Ninja recipes. No Torch/model
import, CUDA execution, build or environment mutation occurred in this check.

Environment: **gpat-m6-e05**. Lock SHA256:
`849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50`.
The prior qualified runtime is Python 3.11.16, Torch 2.12.1+cu130, CUDA 13.0,
on RTX 3090. Environment files and M6D4a evidence remain byte-identical.

| Existing real custom operator | Verified binary SHA256 |
| --- | --- |
| fused_bias_act | 5afdb2369817ffea68530b1e88d75596a77ed08a397f83122b618a18225718ab |
| upfirdn2d | 40ba9db73e1790cd3d904a1fed09c5441180c6995a9a396030cfaa52efd1d081 |

Both requested M6D4b process JSON files are explicit **NOT_RUN_PHASE_A_STOP**
records, with `launched=false` and `measurements=null`. They are not synthetic
qualification evidence. Initial loss values, D losses, external total, gradient
inventories, initial/post-step parameter hashes, Adam state, iteration transition
and two-process repeatability are **NOT MEASURED**. Optimizer applications: **0**.
G-step and D-step parameter-change diagnostics: **NOT RUN**.

Checkpoint static policy remains BASELINE_FINAL_STATE_V1, terminal iteration 4000,
no VAL/TEST selection, and zero extra optimizer steps. No checkpoint bytes,
`torch.save`, pretrained loading, generated image files or synthetic bank.

## Validation and finalization

**Laptop: 74 passed, 1 skipped (75 tests), zero failures/errors.** Includes all
65 prior E05/A5/runtime evidence tests plus ten STOP-integrity/rejection tests.
The existing real Torch CPU pooling test is skipped because laptop Torch is absent.
Python I/O firewall recorded zero denied actual access attempts; rejection tests
use simulated paths/events without opening prohibited data. Full suite and GPU
training qualification were not run. Preflight PASS verifies STOP artifact integrity
only; it does not qualify the training runner.

Ledger **106 -> 107**, exactly one
`M6D4B_E05_TRAINING_RUNNER_QUALIFICATION` STOP row. The first 106 rows remain
byte-identical; prefix SHA256
`5402e9d9207222be01104675ba95003ed9bbbdb7f85a8336778cd91ed4e1448d`.
Artifact index rebuilt **last**, **550 rows**, CRLF preserved, six explicit new
artifact paths. Historical rows carried from committed metadata; no manifest or
benchmark-data target opened. No scientific implementation/config/history changed.
Laptop changes: six new audit/preflight/test files plus appended ledger and index;
HEAD/origin unchanged. GPU remains synchronized and clean; STOP artifacts are local.

**NO BENCHMARK DATA ACCESS. NO BENCHMARK IMAGE DECODE OR FILENAME ENUMERATION.
NO TRAIN/VAL/TEST SAMPLE ACCESS. NO PAIR MANIFEST SAMPLE ACCESS.
NO E05 BENCHMARK TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK.
NO PRETRAINED WEIGHTS. NO COMMIT. NO PUSH.**
