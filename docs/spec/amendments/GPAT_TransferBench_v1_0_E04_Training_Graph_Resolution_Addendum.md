# E04 Training Graph Resolution Addendum — M6D3c evidence stop

Status: **STOP_AND_REPORT — EVIDENCE ONLY; NOT AN EXECUTABLE GRAPH FREEZE**.

This additive document records Phase A evidence. It does not select a scientific
architecture, amend the frozen E04 config, or authorize optimizer execution.
The complete requested component inventory is present below; scientific resolution
is incomplete wherever explicitly marked. No graph-resolution YAML is created.
M6D3b remains historical, unchanged STOP evidence.

## Authority and version reconciliation

The owner's exact primary source is [arXiv:2012.05185v1](https://arxiv.org/pdf/2012.05185).
The [author-hosted PAMI PDF](http://cvlab.cse.msu.edu/pdfs/liu_liu_PAMI2022.pdf)
was already identified in M6D3b. It names the architecture STDN+ and changes
equation numbering and some scientific definitions. It is comparison evidence,
not permission to replace the owner's arXiv source or frozen objectives.

The source URLs and byte hashes are recorded in the paired JSON. The pinned
[STDN source](https://github.com/yaojieliu/ECCV20-STDN/tree/c79f1f8c615d2b8471b3df29da881bb18dd54c90)
provides predecessor evidence, not official PhySTD code. Downloaded model.py and
train.py hashes match the committed source-pins record. Source files were never
imported or run. PDF page images were rendered only from the downloaded public
paper; no benchmark image was accessed.

The frozen compositions remain `100*L_depth + 5*L_G + L_P + 1e-4*L_R`
and `10*L_S + L_H`. They match arXiv v1 Eq.23/24. The final-paper PDF's
Eq.24 places coefficients on different losses and Eq.25 is the second objective.
Its regularizer and printed frequency sizes also differ. These differences are
disclosed; none overrides the requested frozen contract.

## Unresolved scientific choices — stop gate

### GRAPH-01 — encoder internal widths

The schematic supports three downsampling blocks and shows a three-slab inset but does not assign each internal convolution a channel width. The predecessor endpoint widths differ.

- Three convolutions per block with constant widths (64,64,64), (96,96,96), (128,128,128), stride2 at the end.
- Three convolutions per block retaining predecessor internal 64,96 widths, changing only endpoints: (64,96,64), (64,96,96), (64,96,128), stride2 at the end.

Both attain published endpoints; learned capacity, parameter count and feature transformations differ. No source establishes a unique E04 mapping; neither alternative is selected.

Required: Exact ordered layer/channel/stride/padding table from the owner or author, or explicit owner selection of a controlled reconstruction.

### GRAPH-02 — binary-only inpainting supervision

Paper P0 uses attack-specific prior regions; base/A3/A4 provide no binary-only rule. RGB TA magnitude-to-one-channel threshold also lacks an executable reduction definition.

- A fixed attack-independent nonzero P0 prior.
- A geometry-derived attack-independent P0 prior.

Different gradients to P and different supported inpainting regions. All-zero P0 makes the displayed denominator zero; omitting the term changes the loss. No choice selected.

Required: Freeze an attack-label-free P0 rule, its exact values/geometry dependency and denominator behavior, plus TA magnitude/channel reduction.

### GRAPH-03 — optimizer path and hard-sample target semantics

Paper prose has three stages while Algorithm1 ends with one collective update; predecessor sums G objectives into one Adam application. Hard-sample effective inpainting target is not an executable recipe.

- One Adam G application to the combined objectives and one D application.
- Separate Adam applications for Eq.23 and Eq.24 with a D application and explicit ordering.

Different slot evolution and minimum path coverage; variable and BN state transitions differ. Not resolved by the shared learning-rate coefficient alone.

Required: Freeze optimizer instances/slot sharing, path ordering, one outer-iteration increment, detach boundaries and effective hard-sample trace target.

The encoder alternatives are counterexamples demonstrating underdetermination,
not recommendations or a planned architecture search. Even interpreting the inset
as three convolutions leaves the ordered widths unresolved. No missing layer
count is inferred from intuition. Under the owner's explicit stop rule, Phase B/C
cannot begin while materially different architectures remain plausible.

Other incomplete implementation details are visible in the table: padding,
output activations, attention details, norm reduction, BN/slope settings and
hard-sample randomness. They must be resolved together in a future executable
freeze; passing a tensor smoke test would not establish their scientific validity.

## Graph evidence table

| Component | Paper equation / figure / section | Exact published behavior | STDN predecessor evidence | Current frozen E04 contract | Implementation resolution | Provenance class |
|---|---|---|---|---|---|---|
| frequency-decomposed inputs | arXiv v1 Eq.9, section 3.2, Fig.4; author PDF Eq.10-11 | v1: IB=low32(I), IC=low128(I)-low32(I), IT=I-low128(I); concatenate IB,15*IC,25*IT. Low-pass downsamples then upsamples. Author PDF explicitly says bilinear, but prints n1=128,n2=64. | model/model.py:23 concatenates RGB and YUV (6 channels); not the PhySTD frequency input. | RGB input 256; batch 8. Frequency operator and interpolation coordinates not frozen. | Record arXiv v1 as requested primary evidence; do not silently import the differing author-PDF scales. No operator instantiated. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| generator encoder | arXiv v1 section 3.2 and Fig.4 | Initial k3c32s1; three ConvBlocks labelled k3c64s2,k3c96s2,k3c128s2; endpoints F1=128x128x64,F2=64x64x96,F3=32x32x128. Inset depicts three conv-like slabs with final /2, without per-layer channel labels. | model/model.py:24-36 uses initial c64 and three explicit (c64,c96,c64) blocks, last conv stride 2. All predecessor endpoints have c64. | No ordered encoder layer/channel table in base/A3/A4. | UNRESOLVED_ARCHITECTURE: block endpoint labels do not determine internal widths. Do not invent a layer list or claim the predecessor is the published E04 encoder. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| trace decoder | arXiv v1 Fig.4 and section 3.2 | F3 feeds convT k3c64s1, then three convT k3c32s2 stages U1/U2/U3 with encoder shortcuts, then conv k3c13s1 at 256. | model/model.py:38-43 uses three c64 transpose convolutions with concatenation shortcuts and separate multi-resolution trace heads. | Additive/inpainting decomposition retained; no exact shortcut operator or output activation frozen. | Partial topology evidence only. Shared final 13 channels are documented; shortcut placement/channel ordering and activation policy must be explicit in a future executable contract. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| three additive trace components | arXiv v1 Eq.5 and Fig.4; author PDF Eq.11 | B,C,T are RGB components; TA=low32(B)+low128(C)+T in v1. Decoder emits them at input resolution before band limitation. | model/model.py:41-48 instead emits s,b,C,T with global s/b, pooled C, full-resolution T and tanh trace heads. | Three additive components retained; no new numerical range frozen. | Retain component identities; predecessor multiplicative s and its global b must not be copied into E04. No graph built. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| one inpainting component | arXiv v1 Eq.3-8, Fig.4 | One inpainting process comprises P in R^(256x256x1) and RGB live content IP. P gates replacement; together they use 4 of the 13 output channels. | No P/IP in predecessor Gen. | Inpainting retained; beta=0.1; binary weak supervision only. | P/IP are required. Activation/range enforcement and preliminary mask P0 remain unselected; no predecessor equivalent fills them. | PAPER, FROZEN_BENCHMARK_CONTRACT |
| pseudo-depth / auxiliary FAS head | arXiv v1 section 3.2 and Fig.4 | Uses F1,F2,F3 and U3, spatial attention with displayed kernels 7,5,3,7 respectively, resizing/concatenation to 32; k3c32s1 then k3c1s1; M is 32x32 in [0,1]. | model/model.py:50-57 concatenates resized F1/F2/F3 without attention or U3; c64,c64,c1 head and dropout in two layers; unbounded output. | K=32; L_depth required; live target A3/A4, spoof target exact float32 zeros. | Do not transplant predecessor ESR head. Attention activation and final range mapping require an explicit future resolution. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| spoof reconstruction / generation model | arXiv v1 Eq.3 | Ispoof=(1-P)*(Ilive+TA)+P*TP; TP is spoof appearance in replaced regions. | Predecessor uses multiplicative s and additive b,C,T; no replacement process. | Additive/inpainting model preserved. | Equation recorded; no additional cycle-reconstruction loss invented. | PAPER, FROZEN_BENCHMARK_CONTRACT |
| live reconstruction | arXiv v1 Eq.5 and Eq.11 | Ihi=(1-P)*(I-low32(B)-low128(C)-T)+P*IP; Imid=(1-P)*(low128(I)-low32(B)-low128(C))+P*IP; Ilow=(1-P)*(low32(I)-low32(B))+P*IP. | train.py:47 uses (1-s)*I-b-resize(C)-T; later resizes one reconstruction for all scales. | Paper hierarchical live reconstruction retained. | Separate hierarchical reconstructions are required; resizing a single high-resolution reconstruction is not equivalent. Not executed. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| spoof synthesis / trace transfer | arXiv v1 Eq.8, Eq.13-15, section 3.3.1 | Transfer source TA,P and source image to live target: (1-P)*(Ilive+TA)+P*Isource after geometric correction. Q=140 landmark offsets are interpolated using Delaunay; sampling uses bilinear interpolation. | train.py:49-50 warps one effective residual then adds it to a live image; no separate P/source-image transfer. | A3 fixes geometry engine and Q140. This task permits generated tensors only and requires no image/geometry pipeline. | Future synthetic qualification can supply explicit generated offsets, but cannot silently replace the scientific transfer path with raw TA addition. No warp executed. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| harder sample synthesis | arXiv v1 section 3.3, Algorithm 1 | Three random weights in [0,1] multiply B,C,T; P is removed with probability 0.5. Original live samples accompany hard samples to balance batch normalization. | train.py:66-84 samples predecessor factors in [0.1,0.8], chooses component changes and a mixture of alternate sample paths. | No perturbation granularity/PRNG mapping frozen; qualification uses a separate fixed seed. | Do not reuse predecessor random recipe. Per-sample versus per-batch draws and trace-target handling are unresolved details. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| discriminators | arXiv v1 section 3.4, Fig.4 | Four separate PatchGAN discriminators, same architecture without shared weights; each returns a one-channel real/synthetic map described as in [0,1]. | model/model.py:61-79 Disc_s has two independent linear output heads for live/spoof per scale. | Multi-scale discriminators required; no exact head activation frozen. | Four one-head models required; predecessor three two-head models are not an equivalent replacement. Output range mapping remains unresolved. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| multi-scale discriminator structure | arXiv v1 section 3.4, Fig.4 | D1,D2,D3 consume real live versus Ilow,Imid,Ihi at 32,96,256; D4 consumes real/synthesized spoof at 256. Figure lists conv x4: k3c32s2,k3c64s2,k3c96s2,k3c128s1 before output. | train.py:33-34,51-63 uses 256,160,40; Disc_s hidden widths end at 96. | No padding/head kernel or patch output dimensions fixed. | Known hidden widths/scales recorded. Exact patch map dimensions cannot be certified before padding and final projection are resolved. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| L_depth | arXiv v1 Eq.16; author PDF Eq.21 | Text specifies L1 depth error, divided by K squared, averaged over live/spoof; displayed norm uses F notation despite L1 prose. | Predecessor esr_loss uses scalar -1/+1 class targets, not face-shaped live depth. | A3 explicitly preserves L1; K=32; exact zero spoof M0. Synthetic live supervision only in this milestone. | L1 and target semantics already resolved by frozen contract; do not replace with Frobenius L2 or predecessor ESR. No value/gradient evaluated. | PAPER, FROZEN_BENCHMARK_CONTRACT |
| L_G | arXiv v1 Eq.17; author PDF Eq.19 | Sum of four least-squares fake-to-real errors: D1(Ilow),D2(Imid),D3(Ihi),D4(synthesized spoof) versus 1, expectation over live/spoof. | train.py gan_loss sums six mean-square terms from three dual-head discriminators. | alpha2=5 for L_G in the frozen Eq.23 composition. | Four terms required. Spatial sum versus mean must be explicitly resolved; predecessor reduce_mean is not a paper proof. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| L_P | arXiv v1 Eq.19 and Fig.7; author PDF Eq.18 | Spoof-only expectation of squared P-versus-(TA>beta) error plus squared P*P0 norm divided by squared P0 norm. Text calls for trace magnitude; P0 marks regions not to inpaint and depends on attack knowledge. | Absent from predecessor. | beta=0.1; alpha3=1; attack-type labels FORBIDDEN; no binary-only P0 or RGB-to-scalar threshold rule frozen. | UNRESOLVED_SCIENTIFIC_SUPERVISION: do not invent P0, remove this term, add denominator epsilon, or choose a magnitude/channel reduction without a recorded resolution. | PAPER, FROZEN_BENCHMARK_CONTRACT |
| L_R | arXiv v1 Eq.20; author PDF Eq.17 | v1 sums squared norms of B,C,T,P over live/spoof. Author PDF instead regularizes the combined TA and P; cross terms make these non-equivalent. | Predecessor sums means of squared s,b,C,T; separate live/spoof weights 10 and 1e-4. | lambda=1; alpha4=1e-4; no predecessor live/spoof weight split authorized. | Record v1 component regularizer as requested evidence, distinguish author-PDF difference. No silent substitution or loss execution. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| L_S | arXiv v1 Eq.21; author PDF Eq.22 | Pixelwise L1 synthetic supervision stops gradients through synthesized input and warped supervision target. G is also used as reconstruction notation; prose describes recovery of overall trace effect. | train.py:80-84 detaches synthetic inputs; pixel_loss detaches trace_warp. The effective predecessor trace includes its multiplicative term. | alpha5=10; exact effective target for inpainting and hard perturbations not frozen. | Preserve both explicit stops. UNRESOLVED_TARGET: do not equate the target to TA alone or confuse reconstructed RGB with trace residual. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| L_H | arXiv v1 Eq.22; author PDF Eq.23 | Depth supervision on harder synthesized spoof samples, with 1/K squared scaling; hard pass contains original live samples for BN balance. | Predecessor esr_loss_a uses -1/+1 labels on both halves. | alpha6=1, K=32; spoof M0=0; no removal of depth loss. | Synthetic spoof zero targets retained; inclusion of balancing live examples in the loss versus BN-only must be explicit. Not evaluated. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| L_D | arXiv v1 Eq.18; author PDF Eq.20 | Four real-to-1 and four synthetic-to-0 least-squares terms over independent discriminators. | Predecessor has twelve terms divided by 4. | No separately frozen discriminator coefficient; all optimizer paths must be qualified. | Required optimizer objective even though six named generator/depth losses are the requested finite gates. Do not inherit the predecessor /4 factor. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| Eq.23 / Eq.24 weighted objectives | arXiv v1 Eq.23-24; author PDF Eq.24-25 | v1: a1*L_depth+a2*L_G+a3*L_P+a4*L_R and a5*L_S+a6*L_H. Author PDF Eq.24 orders a1*L_R+a2*L_P+a3*L_G+a4*L_depth instead. | Predecessor train.py combines g_loss+a_loss in one generator optimizer. | Explicitly fixed: 100*L_depth+5*L_G+L_P+1e-4*L_R; 10*L_S+L_H. | Frozen/user compositions prevail. Author-PDF coefficient placement cannot override them. External numerical verification remains NOT_RUN. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| variable sharing | arXiv v1 section 3.2,3.4, Algorithm 1 | One generator is reused for original/hard inputs; encoder supports depth and trace tasks. D1-D4 do not share weights. | utils.py uses tf.AUTO_REUSE; train.py reuses scope STDN for both passes. | No frozen variable names or BN update policy. | Share generator trainables across calls; separate discriminator parameter sets. Exact reuse and native nontrainable updates require a later graph freeze. | PAPER, OFFICIAL_PREDECESSOR |
| variable groups | arXiv v1 section 3.5 and Algorithm 1 | Generator/depth and discriminator objectives apply to their corresponding network parts. | model/model.py:116 selects trainables by scope; train.py uses STDN and Disc. | Gates require intended changes and unrelated-group invariance. | Expected groups: shared generator (encoder/trace/depth) and D1-D4. No actual names, shapes, gradient matrix or parameter counts exist yet. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| update ordering | arXiv v1 section 3.5 and Algorithm 1 | Three training stages every minibatch; discriminator learning rate is half. Algorithm computes all losses before its final back-propagation/update line; prose describes three stages. | train.py constructs one G optimizer on g_loss+a_loss and one D optimizer; get_train_op increments the supplied global step for each. | 150000 outer iterations and terminal iteration 150000; minimum synthetic steps only. | UNRESOLVED_OPTIMIZER_SEMANTICS: combined G update versus separate Eq.23/Eq.24 Adam applications changes slots and update count. Do not let each path increment the scientific iteration independently. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| normalization / activation | arXiv v1 Fig.4 caption and section 3.5 | Batch normalization and leaky ReLU after conv/convT except final layers; balanced original-live/hard batch. Numerical BN settings, leaky slope and bounded-output implementation are not specified. | utils.py:107 onward uses BN decay .99, epsilon 1e-5, scale=True, immediate updates; learned PReLU, not fixed leaky ReLU. | FP32 and native state recording required; no BN/slope defaults frozen. | Predecessor BN values are candidate provenance only; its PReLU cannot be presented as the published activation. No settings selected. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| initialization | arXiv v1 section 4.1 | Weights initialized from Normal(0,0.02). | utils.py convolution weights random_normal stddev=.02, biases zero; BN/PReLU have their own initializers. | Normal(0,0.02) preserved; all state initialization must be recorded. | Weight distribution retained. Bias, BN affine/running-state and output-head initialization must be explicitly enumerated before execution; no empirical initializer gate run. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| stop-gradient rules | arXiv v1 Eq.21; Algorithm 1 | Synthetic input and warped supervision target in L_S are explicitly detached to avoid collapse. | train.py uses stop_gradient on synthetic input alternatives and trace_warp target. | Every intended loss-to-variable gradient and unrelated-group invariance must be verified. | Record required L_S detach edges; D optimizer variable exclusion is distinct from stopping all gradients through D during G loss. No gradient graph exists. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| spatial resolutions / channel widths | arXiv v1 Fig.4, sections 3.2-3.4 | Input frequency concat 256x256x9; initial conv c32; F1/F2/F3 endpoints 128x128x64,64x64x96,32x32x128; decoder output 256x256x13; depth 32x32x1; D inputs 32,96,256,256 RGB. | Predecessor differs in RGB+YUV input, encoder widths, decoder widths, trace output scales and discriminator scales. | Input [8,256,256,3], K32; output inventory, trainable/state inventory and op count are gates. | Endpoint facts are not a complete layer table. Exact internal widths, padding and patch outputs remain uncertified; do not fabricate counts. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| optimizer / learning rate / iteration / seeds | arXiv v1 section 4.1; A4-2; A2 final-state convention | TensorFlow; batch8; 150000 iterations; lr5e-5 divided by10 every45000; discriminator lr half per section3.5. | model/model.py:100-128 uses staircase exponential decay and tf.train.AdamOptimizer(lr), scoped gradients and moving averages. | Adam beta1=.9,beta2=.999,epsilon=1e-8,weight_decay=0; seeds42,1337,2026; terminal150000. Required base LR checkpoints fixed by user. | All frozen values preserved. LR expected values recorded but no TensorFlow boundary test or optimizer path executed. | PAPER, OFFICIAL_PREDECESSOR, FROZEN_BENCHMARK_CONTRACT |
| qualification precision / runtime | M6D3c owner instruction; prior E03 lock | Historical package versions do not determine scientific layer choices. | Previously qualified TF1 NVIDIA environment is recorded in the E03 lock. | FP32; disable TF32,AMP,XLA auto-jit/precision rewrites; two clean processes; generated tensors only. | Exact requested Python path returned ENOENT on this host. No environment mutation or creation; no remote probe because Phase A failed. Precision and repeatability gates NOT_RUN. | ENGINEERING_COMPATIBILITY, FROZEN_BENCHMARK_CONTRACT |

## Preserved qualification contract

Fidelity CONTROLLED_ADAPTATION; method IMPLEMENTED_NOT_EXECUTED. Input256,
batch8, budget150000, Normal(0,.02), Adam(.9,.999,epsilon1e-8), no weight decay,
base LR5e-5 divided by10 each45000, alpha=(100,5,1,1e-4,10,1), beta=.1,
lambda=1, K32, terminal150000, experiment seeds42/1337/2026, TEST forbidden.

Expected base LR gates (not runtime measurements):

| Step | Base LR |
|---|---|
| 0 | 5e-5 |
| 44999 | 5e-5 |
| 45000 | 5e-6 |
| 89999 | 5e-6 |
| 90000 | 5e-7 |
| 135000 | 5e-8 |

No trainable/nontrainable tensor inventory, parameter count, TF operation count,
initializer measurement, loss value, gradient, update, LR-runtime check or
repeatability result exists for this attempt. All these gates remain NOT_RUN.
The requested two clean processes were not started. Optimizer steps: zero.
E04_GEOMETRY_RUNTIME_QUALIFIED is retained from M6D3a, not re-tested here.
