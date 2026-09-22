# E05 implementation traceability

Reporting: **PCGAN (controlled architecture resolution)**, `CONTROLLED_ADAPTATION`.
Authority: immutable M6B E05 YAML + A2-07 architecture resolution + A5 blur resolution.
The base/snapshot and A2/A5 document/overlay bytes are verified at preparation.

Source basis: `taesungp/swapping-autoencoder-pytorch` commit
`6baa180f1184ee79a6b967f9d80ee0e02a979ac7`. This is not an official PCGAN release.
The adapter checks repository, commit, tree, required regular files, Git blob
identity, and cited provenance hashes. It never restores or loads pretrained weights.

| Component | Pinned file and symbol |
| --- | --- |
| Encoder and spatial code | `models/networks/encoder.py:StyleGAN2ResnetEncoder.__init__/forward` |
| Generator | `models/networks/generator.py:StyleGAN2ResnetGenerator.__init__/forward` |
| Styled convolutions | `models/networks/stylegan2_layers.py:StyledConv.__init__`, `ModulatedConv2d.__init__/forward` |
| RGB output | `models/networks/stylegan2_layers.py:ToRGB.__init__` (retains `demodulate=False`) |
| Image discriminator | `models/networks/discriminator.py:StyleGAN2Discriminator.__init__`, `stylegan2_layers.py:Discriminator` |
| Patch discriminator | `models/networks/patch_discriminator.py:StyleGAN2PatchDiscriminator` |
| Adam construction evidence | `optimizers/swapping_autoencoder_optimizer.py:SwappingAutoencoderOptimizer.__init__` |

`ModulatedConv2d` constructs an `EqualLinear` style modulation and uses reciprocal
square-root normalization of style/weight statistics when demodulation is enabled.
The pinned `new_demodulation=True` path is retained as source authority. A2
requires this executable architecture, including its discriminator; StyleGAN-v1
AdaIN is not substituted. Layer details remain in verified upstream source,
rather than a vendored or paper-rewritten network.

At 256, frozen `netE_num_downsampling_sp=1` and `spatial_code_ch=8` imply an
8×128×128 spatial code. The generator upsamples once to full 3×256×256 RGB.
Other inherited option defaults are extracted from the pinned parser AST with
file provenance, including global code channels (upstream default 2048), network
capacities, noise, antialiasing, and patch settings. This follows A2's executable
basis; it is not a claim to reproduce every PCGAN prose detail.

The upstream training loop is not launched: its even-image-batch swap, L1 loss,
and lazy R1 optimizer scaling do not implement the frozen E05 five-term objective
and batch-one configuration. Static plans retain Adam lr=1e-6, betas=(0.9,0.999),
batch=1, 4000 iterations, and the exact unit-weighted five terms.

The [paper §3.1.2](https://arxiv.org/html/2604.09018v1#S3.SS1.SSS2) provides
PCGAN Eq.1–5 loss identities. Alpha/beta remain in the frozen loss metadata;
the paper uses them in PMN Eq.10, and they do not reweight PCGAN's five unit terms.
The adapter exposes symbolic loss mappings, not a new numerical loss reduction.

A5 alone supplies the added blur: `torch.nn.functional.avg_pool2d`, kernel 2,
stride 2, padding 0, ceil_mode=False, count_include_pad=False. Both target and
generated branches pass through the same operator without detach. There is no
fallback. Its provenance is `BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION`.
The hook imports Torch only on explicit invocation; preflight never invokes it.

Plans use common frozen-config/source/seeding/logging support. The common learned
checkpoint metadata helper now accepts E05's iteration-based final state so
`LearnedRunContext` can retain its existing logging interface. Final selection is
within seed at iteration 4000, with zero extra steps if a terminal save is needed.
Neither VAL nor TEST selects a checkpoint, and no best seed is chosen.

This milestone implements static adapters/plans and a lazy pooling hook. It does
not construct a model or implement/launch a training runner. Future execution
requires runner integration, compatible PyTorch/torchvision dependencies and
verification of the pinned custom-op environment. Source imports that can compile
CUDA extensions are not performed during static preparation.
