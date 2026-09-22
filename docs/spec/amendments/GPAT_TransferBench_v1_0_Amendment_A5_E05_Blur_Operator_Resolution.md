# Amendment A5 — E05 blur operator resolution

Status: **OWNER-APPROVED · ADDITIVE · NON-DESTRUCTIVE**

This amendment resolves exactly one execution gap found during M6C2b2:
the `blur(.)` operator in E05 PCGAN's `L_recblur`. E05 remains
**PCGAN (controlled architecture resolution)**, fidelity `CONTROLLED_ADAPTATION`.
This amendment does not implement E05 or change any other scientific parameter.
The original M6B YAML, its snapshot, and Amendments A1–A4 remain unchanged.

## Evidence and scope

The [PCGAN paper, §3.1.2, Eq. 3](https://arxiv.org/html/2604.09018v1#S3.SS1.SSS2)
compares blurred target and mixed generated images, with blurring implemented
through downsampling from 1024×1024 to 512×512. It does not identify the
interpolation family, convolution kernel, antialias flag, padding, coordinate
convention, or boundary mode. The existing benchmark mapping is 256×256 to
128×128. The stated purpose is to suppress high-frequency spoof artifacts while
retaining semantic facial structure.

Eq. 3 is typeset with an L2 norm (subscript 2). A5 specifies only its blur
operator; it does not introduce a squared-distance rule, change the existing
reconstruction distance/reduction, or reinterpret the other losses.

A2-07 resolves the inherited executable architecture only. The pinned basis is
`taesungp/swapping-autoencoder-pytorch` at
`6baa180f1184ee79a6b967f9d80ee0e02a979ac7`. It does not implement PCGAN
`L_recblur`. Inspected paths relative to that pinned source are:

| Source | Observed behavior | Why it is not authoritative for this loss |
| --- | --- | --- |
| `util/util.py:resize2d_tensor` | Bilinear interpolation, `align_corners=False` | Generic resize used for visualization, without a PCGAN blur-loss binding |
| `models/networks/loss.py:VGG16Loss.__init__/vgg_forward` | `Downsample([1,2,1], factor=2)` | Replacement of VGG pooling, unrelated to PCGAN `L_recblur` |
| `models/networks/stylegan2_layers.py:Downsample.__init__/forward` | Caller-supplied filter, `upfirdn2d` resampling | Architecture resampling does not define the added PCGAN loss |
| `models/swapping_autoencoder_model.py:compute_generator_losses` | Upstream reconstruction/adversarial losses | No PCGAN `L_recblur` implementation |

## Owner-approved operator

Provenance: **`BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION`**.
This choice is not `PAPER`, `OFFICIAL_PCGAN`, `AUTHOR_SPECIFIED`, or
`OFFICIAL_SWAPPING_AUTOENCODER`, and does not claim to recover private author code.

The exact framework reference is:

```python
torch.nn.functional.avg_pool2d(
    x,
    kernel_size=2,
    stride=2,
    padding=0,
    ceil_mode=False,
    count_include_pad=False,
)
```

For input `N × C × 256 × 256`, output is `N × C × 128 × 128`:

```text
y[n,c,i,j] = (x[n,c,2i,2j] + x[n,c,2i+1,2j]
             + x[n,c,2i,2j+1] + x[n,c,2i+1,2j+1]) / 4
```

Each disjoint 2×2 block maps to its arithmetic mean. There is no padding,
overlap, explicit rounding/quantization, resize after pooling, or boundary
extrapolation. No Gaussian, bilinear, bicubic, nearest-neighbor, StyleGAN2
`[1,3,3,1]`, or VGG `[1,2,1]` fallback is authorized. Normal floating-point
arithmetic applies; this does not impose a new execution-environment precision pin.

Average pooling is the owner-selected minimal parameter-free half-resolution
low-pass operator for the paper's stated role. It introduces no tunable sigma,
learned parameter, coordinate mapping mode, `align_corners` choice, antialias
toggle, or boundary extension.

## Application and unchanged contract

The SAME operator MUST be applied independently to `x_tgt` and
`G(z_pat_src, z_con_tgt)` before evaluating the existing reconstruction distance.
Neither branch may be detached merely because of blur. The operator has zero
learnable parameters. No fallback operator is authorized.

`L_rec`, `L_recblur`, `L_advrec`, `L_advmix`, and `L_pat`, their frozen weights,
`alpha=0.2`, and `beta=1e-6` remain unchanged. Architecture, supervision, optimizer,
seeds, budget, output, and checkpoint rules are unaffected.

## Immutable additive input

Overlay: `configs/amendments/e05_a5_blur_operator_resolution.yaml`.

Base: `configs/methods/e05_pcgan.yaml`, unchanged SHA256:
`478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de`.

Future E05 execution must consume this overlay as an additional immutable input,
verify its base-config binding, and retain its identity/hash in execution
provenance. The overlay contains only this blur resolution and identity metadata.
