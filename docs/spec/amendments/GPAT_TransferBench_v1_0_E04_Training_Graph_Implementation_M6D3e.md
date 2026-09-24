# E04 M6D3e graph realization and compatibility record

Normative authority remains committed M6D3d at
`90263169f0731b397dfa459d7ac94eaafd2c69ec`. This additive implementation record
is written before synthetic optimizer execution. It does not reopen M6D3d
decisions or amend frozen configs. Fidelity remains CONTROLLED_ADAPTATION.

## Tensor and operator realization

| Component | Realization | Provenance |
|---|---|---|
| Frequency input | Native TF1 bilinear downsample to 32/128, bilinear upsample to 256; align_corners=False, half_pixel_centers=False; no antialias filter. Concatenate IB,15*IC,25*IT. | PAPER decomposition; ENGINEERING_COMPATIBILITY TF1 coordinate/API spelling |
| Encoder | Initial 3x3 c32; each Fig.4 three-convolution ConvBlock uses its published width (64,96,128), strides 1,1,2; SAME padding. | PAPER Fig.4 inset/labels, M6D3d feature widths; ENGINEERING_COMPATIBILITY SAME padding realization |
| Decoder | convT 3x3 c64 stride1; concatenate F3 then upsample U1 c32; concatenate F2 then U2; concatenate F1 then U3; 3x3 c13 output. | PAPER Fig.4; predecessor concatenation only for the unspecified shortcut API |
| 13 output channels | B=0:3, C=3:6, T=6:9, P logit=9:10, IP=10:13. B/C/T/IP are linear outputs; sigmoid P realizes a bounded mixing mask. | PAPER components; ENGINEERING_COMPATIBILITY storage order and bounded mask realization |
| Auxiliary depth | Channel max/mean spatial attention for F1,F2,F3,U3 with 7,5,3,7 kernels; sigmoid attention gates; resize attended features to32 and concatenate; 3x3 c32 then3x3 c1 sigmoid depth. | PAPER Fig.4 attention/head; ENGINEERING_COMPATIBILITY sigmoid range realization |
| Discriminators | Four independent RGB networks, scales32/96/256/256, hidden3x3 widths32/64/96/128 strides2/2/2/1; final3x3 c1 sigmoid patch map. | PAPER structure/range; ENGINEERING_COMPATIBILITY final one-channel projection and SAME padding |
| Normalization | BN after non-output conv/convT, momentum.99, epsilon1e-5, trainable scale and offset; leaky ReLU slope.2. No BN/leaky activation on output or attention projection. | PAPER BN/leaky behavior; OFFICIAL_PREDECESSOR BN settings where silent; ENGINEERING_COMPATIBILITY TF native leaky default and gate projection |
| Initialization | Every convolution/transpose-convolution/attention kernel Normal(0,.02); bias0; BN gamma1/beta0; running mean0/variance1. | PAPER kernels; standard BN/bias ENGINEERING_COMPATIBILITY exceptions |
| Reductions | Expectation over batch; squared Frobenius terms sum spatial/channel squares within each sample. L1 terms sum absolute pixel/channel errors; depth additionally divides by32 squared. No implicit image-size rescaling of coefficients. | PAPER equations as retained by M6D3d |
| Positive P loss | Broadcast single-channel P against RGB Boolean TA>beta, sum the squared differences. No unrequested absolute-value threshold or RGB collapse. | Literal M6D3d primary expression with TF broadcasting ENGINEERING_COMPATIBILITY |
| Negative P prior | zeros_like(P); separate constant0 contribution; normalized quotient absent from graph. | CONTROLLED_RECONSTRUCTION_BINARY_ONLY_P0 |

The trace regularizer is the retained arXiv component sum over B,C,T,P, not a
different-version combined-trace norm. The sigmoid mask does not force P to zero.
IP has no intensity penalty and remains connected through physical reconstruction
and effective trace supervision. The final raw decoder tensor remains 13 channels.

## Geometry and hard samples

Q140 is an ordered correspondence interface. No Q140 identities are rederived,
no geometry weights are loaded, and no 3DDFA process is invoked. Given supplied
source/target pixel coordinates, SciPy LinearNDInterpolator uses Delaunay linear
interpolation to form inverse sampling offsets: target pixel p reads source
p+offset[p]. Outside the correspondence hull, displacement is zero. Native TF
bilinear sampling clips coordinates at the image boundary (border replication).
This is coordinate/API compatibility for geometric transfer, not a substitute
learned geometry model. Qualification uses generated 140-point correspondences
covering the image rectangle with a small nonrigid deformation.

Warp the three additive bands, predicted P and source spoof RGB. New spoof is
`(1-P_warp)*(live+TA_warp)+P_warp*source_warp`. Hard synthesis draws three
independent uniform[0,1] factors per spoof for B/C/T and one Bernoulli(.5)
inpainting-keep decision per spoof. The worker supplies those draws as tensors,
fixed within a minibatch so all three steps consume the same fixture. No factor
or seed is selected based on measured performance. These are the paper's hard
perturbations with an explicit per-example RNG/API realization.

The effective hard target is the matching synthesized-spoof-minus-live trace,
including the inpainting contribution, with stop_gradient. The hard-pass input
also has stop_gradient. It contains the original live half plus the synthesized
spoof half; L_H acts on the spoof half with exact-zero depth. Original depth loss
uses deterministic generated live face-like targets and exact-zero spoof targets.

## Optimizer and native state execution

There are exactly two Adam instances. G Step1 and Step3 share moment slots and
beta powers; D Step2 owns separate slots. Phase guards enforce G→D→G. Only Step3
completion increments global_iteration. No G_D_RATIO is consumed. All three
learning rates are indexed by the same pre-increment iteration.

TF layers BN supplies native UPDATE_OPS. Pure diagnostic forwards/gradient
checks do not fetch those assignment operations. The controller explicitly
executes the update collections for the forward passes belonging to each step,
in this order, before the optimizer application:

| Step | BN update collections, in order |
|---|---|
| 1 | Original G, then D |
| 2 | Original G producing current fake inputs, then D |
| 3 | Original G producing current hard inputs, then hard-pass shared G |

The two G collections update the same running statistics sequentially, avoiding
competing assignments in Step3. D statistics can change during Step1 and G
statistics during Step2 because those networks participate in the forward path;
their unrelated trainable parameters are excluded from that optimizer. The
qualification records every changed nontrainable variable, not just trainables.
BN update ordering is controller/API compatibility; it does not change the
paper's batch-normalization equations. No state is reset between steps.

Native bilinear Resize/Gather sampling executes on CPU so TF1 provides a
deterministic supported backward path; convolutions execute on GPU with native
determinism controls. Device placement does not replace either mathematical
operator. FP32 and all disabled precision-rewrite settings are recorded by the
worker. No optimizer tensors or model state are saved as a checkpoint.

## Qualification scope

Batch8 is four live followed by four spoof tensors of shape256x256x3. The
qualification-only seed is314159. Each clean qualification process is permitted
exactly one complete three-apply minibatch. Graph construction, raw loss and
gradient diagnostics are additional forward/backward evaluations, not additional
optimizer applications. Tests inspect that evidence rather than launching extra
training. Per-tensor names/shapes, hashes and numerical summaries are audit
evidence; no tensor bank, scientific checkpoint or benchmark run directory is
created. A failed scientific gate invokes STOP_AND_REPORT, not a method change.
