# E04 Training Graph Owner Resolution Addendum — M6D3d

Status: **E04_TRAINING_GRAPH_CONTRACT_RESOLVED**. Owner approved; additive;
resolution only. Fidelity **CONTROLLED_ADAPTATION**. No graph implementation,
TensorFlow execution, optimizer execution, environment creation or data access
is authorized or performed in this milestone.

## Authority and historical preservation

Starting authority is committed `m6-baselines` HEAD
`80dd758475f985377170308351ed2cf99a809d7d`, after the M6D3c STOP audit.
This document is the current owner-resolution layer over base E04, A3 and A4.
The [M6D3c evidence addendum](GPAT_TransferBench_v1_0_E04_Training_Graph_Resolution_Addendum.md)
and all M6D3b/M6D3c audits, scripts and ledger rows remain byte-identical.
Their historical assessments are preserved; the following owner resolutions
supersede their decision dispositions for future E04 graph work.

The machine-readable contract is
[e04_training_graph_resolution.yaml](../../../configs/amendments/e04_training_graph_resolution.yaml).
It uses JSON syntax, a YAML subset, so the static preflight requires only the
Python standard library. It is an additive overlay, never a rewrite of
`configs/methods/e04_physics_std.yaml` or `frozen_config_snapshot/`.

## Complete owner decision table

| Decision | Resolved contract | Provenance |
|---|---|---|
| Encoder widths | F1=128x128x64; F2=64x64x96; F3=32x32x128; ConvBlocks k3c64s2, k3c96s2, k3c128s2 | PAPER, Fig.4 and encoder description; not a controlled width choice |
| Binary-only P0 | P0=zeros_like(P), for every spoof sample | CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0 |
| Negative prior | Unavailable type-conditioned negative prior contributes exactly zero; no normalized quotient is evaluated | Consequence of the owner-approved P0 adaptation |
| Active mask learning | Keep the primary squared error between P and (TA>beta), beta=.1; keep the inpainting branch | PAPER term retained under the controlled P0 adaptation |
| Hard depth target | I_hard is spoof; its M0 is exact float32 zeros(32,32) | PAPER spoof-depth rule and hard-sample depth supervision |
| L_S target | Matching warped/synthesized ground-truth trace with paper stop-gradient; synthetic input also detached as specified | PAPER; not a benchmark invention |
| Three steps | Step 1 G Eq.23 → Step 2 D L_D → Step 3 G Eq.24, every minibatch | PAPER |
| Discriminator LR | 0.5 times scheduled generator LR | PAPER |
| Adam semantics | beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0 for G and D | Frozen A4 |
| Adam state ownership | One G Adam state shared by separate Step-1/Step-3 applies; one separate D Adam state | Owner-approved CONTROLLED_RECONSTRUCTION |
| Iteration ownership | Increment global_iteration once after Step 3; schedule by complete minibatch iteration, not applies | Owner-approved CONTROLLED_RECONSTRUCTION |
| STDN evidence | Reference only when PhySTD is silent; never import G_D_RATIO=2 or skip a PhySTD step | Owner instruction; paper schedule controls |

## Encoder correction

The published feature widths and ConvBlock labels are **PAPER-derived and
resolved**, not new benchmark architecture choices. M6D3c's encoder-width blocker
is closed by this owner clarification. No alternate predecessor widths are
selected. This milestone records the stated paper graph contract; it does not
construct layers, invent additional internal widths, or produce measured graph
shapes/parameter counts.

## Binary-only P0 and finite negative-prior semantics

The original P0 prior distinguishes print/replay, 3D mask/makeup, partial-eye
and partial-mouth attacks. E04 freezes `binary_weak_live_spoof` supervision and
forbids attack-type labels. The owner therefore sets `P0 := zeros_like(P)` for
every spoof sample. This is **CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0**, not
original-author behavior. No dataset-specific spatial prior is invented.

The published negative-prior expression contains a denominator equal to the
squared norm of P0. Substituting zero mechanically would produce 0/0. The
owner-approved meaning is a **constant-zero negative-prior contribution**:
the normalized quotient must not be evaluated for this zero-prior contract.
No arbitrary epsilon or alternate nonzero mask is introduced. This is the
implementation consequence of the approved zero contribution, not an additional
mask-design choice.

The primary mask-learning term `||P - (T_A > beta)||^2` remains active with
`beta=0.1`. P0 is the unavailable supervision prior; it is not the predicted P.
Predicted P, IP and the inpainting path remain active, as does the frozen
regularization. Only the type-conditioned negative prior is neutralized.

## Step-3 paper-derived targets

I_hard is a synthesized spoof. Its depth ground truth is therefore exactly
zero, with per-sample shape `(32,32)` and dtype `float32`. This follows the
paper's spoof-depth definition and hard-sample depth loss; it is not a
controlled target invention. Original live examples included for batch balance
retain A3/A4 live-depth semantics and are not relabeled as zero-depth spoof.

For L_S, supervise against the warped/synthesized ground-truth trace belonging
to the synthesized input. Preserve the paper's stop-gradient on that target
and on the synthesized input used for the supervised pass. Do not make the
target trainable or substitute an unrelated reconstruction.

Equation numbering is version-specific in the already committed evidence:

| Semantic rule | arXiv:2012.05185v1 | Author PAMI PDF |
|---|---|---|
| Spoof-depth ground truth | Eq.16 | Eq.21 |
| Synthesized trace supervision L_S | Eq.21 | Eq.22 |
| Hard-sample depth L_H | Eq.22 | Eq.23 |
| Frozen weighted objectives | Eq.23 / Eq.24 | Differently numbered/ordered; not substituted |

This crosswalk preserves the owner's semantic resolution without treating
different equation numbers as different targets. The source URLs and byte
hashes remain in the unchanged M6D3c evidence; no new literature retrieval was
needed for this owner-resolution milestone.

## Sequential optimization and iteration ownership

For the current outer iteration t=`global_iteration`, use
`lr_G(t)=5e-5 * 10**(-floor(t/45000))` and `lr_D(t)=0.5*lr_G(t)`.
All three applies use that same t; individual apply counts do not drive LR.

| Order | Objective | Trainable group | Optimizer state | LR | Iteration increment |
|---|---|---|---|---|---|
| Step 1 | 100*L_depth + 5*L_G + L_P + 1e-4*L_R | Shared generator/encoder/trace/depth | Generator Adam | lr_G(t) | None |
| Step 2 | L_D | Discriminators | Separate discriminator Adam | lr_G(t)/2 | None |
| Step 3 | 10*L_S + L_H | Same generator group | Same Generator Adam as Step 1 | lr_G(t) | Once, after Step 3 completes |

One G Adam state means shared moment slots and beta-power accumulators, carried
across both G applications and subsequent minibatches. It is not two independent
G optimizers, and the two objectives are not collapsed into one apply. The D
optimizer has independent state with identical A4 Adam semantics except its
learning rate. The global iteration is owned by completion of the three-step
minibatch; passing it independently to each apply would violate this contract.

The three-step schedule and half D LR are PAPER. Adam is frozen A4. Shared G
state and once-per-minibatch iteration ownership are the explicit controlled
execution resolution for unpublished optimizer-state detail. STDN's G_D_RATIO
is not imported: there is no ratio-based alternation, skipped D step or skipped
hard-sample step.

## Complete component disposition table

Every component in the M6D3c inventory is mapped below. `CONTRACT_RESOLVED`
describes normative decisions, not successful construction or runtime gates.
The table does not claim new numeric implementation defaults absent from the
owner instruction. Code-level realization, variable/state inventories and
numerical verification belong to a later authorized qualification milestone.

| Component | Resolved decision | Provenance | Status |
|---|---|---|---|
| frequency-decomposed inputs | Retain paper frequency decomposition IB=low32(I), IC=low128(I)-low32(I), IT=I-low128(I), with input gains 1,15,25; follow the PhySTD graph, not predecessor RGB/YUV input. | PAPER | CONTRACT_RESOLVED |
| generator encoder | RESOLVED PAPER widths: F1=128x128x64, F2=64x64x96, F3=32x32x128; Fig.4 ConvBlocks k3c64s2,k3c96s2,k3c128s2. No controlled width choice and no alternative STDN encoder. | PAPER | CONTRACT_RESOLVED |
| trace decoder | Retain paper Fig.4 decoder: convT k3c64s1, U1/U2/U3 convT k3c32s2, encoder shortcuts, final conv k3c13s1; shared decoder predicts B,C,T,P,IP. | PAPER | CONTRACT_RESOLVED |
| three additive trace components | Keep B,C,T and the paper additive trace composition; do not introduce predecessor multiplicative s or replace B with predecessor global bias. | PAPER | CONTRACT_RESOLVED |
| one inpainting component | Keep P and IP active. P0 is a separate supervision prior, not P itself: setting P0 to zero does not set the predicted P to zero. | PAPER, CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0 | CONTRACT_RESOLVED |
| pseudo-depth / auxiliary FAS head | Retain paper depth head from F1,F2,F3,U3 with spatial attention, 32x32 output and [0,1] depth semantics. Retain live A3/A4 supervision and exact-zero spoof supervision. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| spoof reconstruction / generation model | Preserve additive plus inpainting physical model (1-P)*(Ilive+TA)+P*TP; do not add an unrequested cycle loss. | PAPER | CONTRACT_RESOLVED |
| live reconstruction | Preserve the three paper hierarchical live reconstructions using B; B,C; B,C,T respectively, together with the inpainting term. Do not replace them by resizing one reconstruction. | PAPER | CONTRACT_RESOLVED |
| spoof synthesis / trace transfer | Preserve paper geometric trace transfer and source inpainting appearance, with A3 fixed geometry/Q140. No geometry or image pipeline is executed or changed here. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| harder sample synthesis | Retain paper trace perturbations and harder-spoof synthesis. I_hard is spoof; M0 is exactly zero and L_S uses the matching warped/synthesized ground-truth trace with paper stop-gradient. Original live samples remain available for the balanced hard-pass batch. | PAPER | CONTRACT_RESOLVED |
| discriminators | Keep four paper PatchGAN discriminators with independent weights and one real/synthetic map each; no predecessor three-scale dual-head substitution. | PAPER | CONTRACT_RESOLVED |
| multi-scale discriminator structure | Keep D1/D2/D3 live pairs at 32/96/256 and D4 spoof pairs at 256; retain Fig.4 convolution labels k3c32s2,k3c64s2,k3c96s2,k3c128s1. | PAPER | CONTRACT_RESOLVED |
| L_depth | Keep frozen L1 depth loss and K=32. Real live targets follow A3/A4; every spoof target is exact zero. Step-3 synthetic spoof uses that same paper spoof rule. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| L_G | Retain the paper four-discriminator least-squares generator objective and frozen alpha2=5 in Step 1. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| L_P | P0=zeros_like(P) for every spoof. The unavailable type-conditioned negative prior is constant zero, with no normalized 0/0 evaluation. Keep \|\|P-(TA>beta)\|\|^2 active, beta=.1 and alpha3=1; keep P/IP trainable under their intended losses. | PAPER, CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| L_R | Preserve requested arXiv trace regularizer, lambda=1 and frozen alpha4=1e-4. M6D3d does not authorize substituting a different paper-version regularizer or STDN live/spoof coefficients. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| L_S | Target is the warped/synthesized ground-truth trace corresponding to the synthesized input. Stop gradients through the supervision target and synthetic input exactly as in the paper. Provenance PAPER, not a benchmark-created target. | PAPER | CONTRACT_RESOLVED |
| L_H | Apply the paper hard-sample depth loss to synthesized spoof with an exact zero float32 32x32 target; preserve K=32 and alpha6=1. Do not zero targets of original live examples included for batch balance. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| L_D | Apply paper discriminator objective in Step 2 of every minibatch; D uses separate Adam state and half the scheduled G LR. | PAPER, FROZEN_BENCHMARK_CONTRACT_A4, CONTROLLED_RECONSTRUCTION | CONTRACT_RESOLVED |
| Eq.23 / Eq.24 weighted objectives | Step 1 is 100*L_depth+5*L_G+L_P+1e-4*L_R; Step 3 is 10*L_S+L_H. They are separate sequential G applications sharing one Adam state, not a single combined application. | PAPER, FROZEN_BENCHMARK_CONTRACT, CONTROLLED_RECONSTRUCTION | CONTRACT_RESOLVED |
| variable sharing | One generator parameter set is reused for original and hard samples; D1-D4 remain independent. G steps share Adam slots and beta-power accumulators; D optimizer state is separate. | PAPER, CONTROLLED_RECONSTRUCTION | CONTRACT_RESOLVED |
| variable groups | Step 1 and Step 3 update the generator/depth/trace group; Step 2 updates the discriminator group. Actual variable names, gradient paths and nontrainable transitions are later qualification evidence, not results from this milestone. | PAPER, CONTROLLED_RECONSTRUCTION | CONTRACT_RESOLVED |
| update ordering | Every minibatch executes Step 1 G Eq.23, then Step 2 D L_D, then Step 3 G Eq.24. G Adam state persists across both G applications. Increment global_iteration once after Step 3; index all three step LRs by the same outer iteration. | PAPER, FROZEN_BENCHMARK_CONTRACT_A4, CONTROLLED_RECONSTRUCTION | CONTRACT_RESOLVED |
| normalization / activation | Retain paper Fig.4 batch normalization and leaky ReLU behavior except output layers; retain balanced original-live/hard batches. STDN is reference only where PhySTD is silent, not authority to replace paper activation behavior. | PAPER | CONTRACT_RESOLVED |
| initialization | Preserve Normal(0,.02) weight initialization. The eventual trainable/nontrainable initializer inventory remains a qualification deliverable; no state is allocated here. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| stop-gradient rules | Use paper stop_gradient for both synthesized Step-3 input and its warped/synthesized ground-truth trace target; the target is supervision, not another trainable prediction. | PAPER | CONTRACT_RESOLVED |
| spatial resolutions / channel widths | Published F1/F2/F3 widths and ConvBlock labels are resolved PAPER facts. Retain input256, decoder13 channels, depth32 and paper D scales. Actual graph output/variable shapes will be measured during qualification. | PAPER, FROZEN_BENCHMARK_CONTRACT | CONTRACT_RESOLVED |
| optimizer / learning rate / iteration / seeds | Adam A4 (.9,.999,epsilon1e-8,no weight decay); scheduled G LR5e-5 /10 each45000 complete iterations; D LR=G/2; G and D states separate, G state shared across its two steps. Stop at global_iteration150000; seeds42,1337,2026 unchanged. No STDN G_D_RATIO scheduling. | PAPER, FROZEN_BENCHMARK_CONTRACT_A4, CONTROLLED_RECONSTRUCTION | CONTRACT_RESOLVED |
| qualification precision / runtime | Contract status is resolved; graph/runtime remain not yet qualified. Future authorized qualification must honor FP32, disabled TF32/AMP/XLA auto-jit and precision rewrites. This milestone does not create/probe an environment or import TensorFlow. | FROZEN_BENCHMARK_CONTRACT, ENGINEERING_COMPATIBILITY | CONTRACT_RESOLVED |

## Preserved values and final state

Input256; batch8; budget150000 complete iterations; Normal(0,.02); initial G
LR5e-5 divided by10 every45000; Adam(.9,.999,epsilon1e-8), zero weight decay;
alphas=(100,5,1,1e-4,10,1); beta=.1; lambda=1; K32; terminal iteration150000;
experiment seeds42/1337/2026; TEST forbidden. No scientific value in the base
configuration or frozen snapshot is edited.

Final status:

- E04_GEOMETRY_RUNTIME_QUALIFIED (retained from M6D3a).
- E04_TRAINING_GRAPH_CONTRACT_RESOLVED.
- E04_TRAINING_GRAPH_NOT_YET_QUALIFIED.
- E04_TRAINING_RUNTIME_NOT_YET_QUALIFIED.
- E04 IMPLEMENTED_NOT_EXECUTED.
- CONTROLLED_ADAPTATION.

No graph implementation or execution, optimizer execution, TensorFlow import,
environment creation, benchmark data access, checkpoint or synthetic bank.
No commit and no push. Static preflight success is contract/audit validation
only and must never be reported as graph or runtime qualification.
