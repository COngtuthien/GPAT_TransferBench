# Amendment A6 — E07c feature-interface source correction

Status: **OWNER-APPROVED · ADDITIVE · NON-DESTRUCTIVE**.
Provenance: **`SOURCE_GROUNDED_CONTRACT_CORRECTION`**.

This amendment resolves only the consumed encoder-channel contradiction discovered
during M6C2b3 preflight. It corrects a benchmark documentation/static-analysis
error; it does not introduce a new scientific architecture decision or model
adaptation. E07c implementation has not resumed.

## Contradiction and source evidence

A3 §5.2 and the frozen M6B E07c metadata describe the first two consumed encoder
outputs as 32×32×128 and 16×16×256. These descriptions contradict the exact
already-pinned source:

- Repository: `https://github.com/murphytju/DiffFAS`.
- Commit: `23f40519ec25a833ebc06842aa6fbab74fad4d15`.
- File: `models/custom_rn.py`.
- SHA256: `fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3`.

| Source location at the pinned commit | Evidence |
| --- | --- |
| `custom_rn.py:123–124`, `BasicBlock` | `expansion = 1` |
| `custom_rn.py:267`, `ResNet.__init__` | `layer2 = self._make_layer(block, 256, layers[1], stride=2, ...)` |
| `custom_rn.py:268`, `ResNet.__init__` | `layer3 = self._make_layer(block, 512, layers[2], stride=2, ...)` |
| `custom_rn.py:269`, `ResNet.__init__` | `layer4 = self._make_layer(block, 512, layers[3], stride=2, ...)` |
| `custom_rn.py:329–346`, `ResNet._forward_impl` | Returns layer2/layer3/layer4 tensors directly as `o32x32,o16x16,o8x8`, followed by the classifier output |
| `custom_rn.py:367–369`, `resnet18` | Calls `_resnet("resnet18", BasicBlock, [3,4,6,3], ...)` |

With the pinned stem and stage strides at 256×256, the corrected interface is
listed below in **H×W×C** notation (actual tensors are NCHW):

| Output | Superseded A3/M6B description | Corrected authoritative dimensions |
| --- | --- | --- |
| `x32x32` | 32×32×128 | **32×32×256** |
| `x16x16` | 16×16×256 | **16×16×512** |
| `x8x8` | 8×8×512 | **8×8×512**, unchanged |
| `embg` | B×K | **B×7**, already-authorized K=7 |

## Authority and prohibited changes

A3's normative intent remains **preserve the exact pinned DiffFAS conditioning
encoder architecture**. The erroneous channel descriptions were not an instruction
to modify that architecture. A6 supersedes only those descriptions; the immutable
base YAML and A3 remain intact as historical contract inputs.

No 256→128 or 512→256 projection is authorized. No additional learned or fixed
channel adapter is authorized. Do not alter `BasicBlock`, `layer2`, `layer3`, or
`layer4`; do not replace `custom_rn.resnet18` or use torchvision ResNet18. The
`[3,4,6,3]` block topology remains unchanged, with the channel widths actually
defined by the pinned source.

This is a benchmark-side source-traceability correction, not retuning,
architecture redesign, improved DiffFAS, or an official author correction.
E07c fidelity remains **`CONTROLLED_ADAPTATION`** for the existing A1 ID-free
adaptation and A3 conditioning-objective reconstruction reasons. No new fidelity
downgrade is introduced.

## Fourth output and preserved K7 objective

`ResNet.__init__` (`custom_rn.py:271`) uses a classifier input of
`512 * BasicBlock.expansion = 512`. A3's head remains `fc = nn.Linear(512, 7)`.
The first three feature tensors are computed before the head, so changing K
affects only the fourth output's dimension. It neither explains nor changes
the consumed channel dimensions.

In pinned `models/unet_autoenc.py`, `BeatGANsAutoencModel.encode` (line 78) returns
the encoder's four outputs. Its `forward` (line 178) assigns
`x32x32, x16x16, x8x8, _ = self.encode(x_cond, encoder)` and places the first three
in the conditioning list; the fourth output is discarded by this conditioning
path. Consumer SHA256:
`127ecd59fbbbd0d191a713aae62b11ea3de7058d42c2889100ff6a44199d2963`.

K7 classes/order, objective, auxiliary seed/run count, whole-module checkpoint
format, main training/sampling settings, identity firewall, and all other A1/A3
semantics remain unchanged.

## Additive immutable execution input

Overlay: `configs/amendments/e07c_a6_feature_interface_source_correction.yaml`.
Base: `configs/methods/e07c_difffas_bin_idfree.yaml`, unchanged SHA256
`dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c`.
A3 SHA256 remains
`b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a`.

Future E07c preparation must consume base + A1/A2/A3 + A6, verify the overlay's
base/A3/source identities, and record its hash. This amendment edits neither the
frozen configs/snapshots nor A1–A5, adds no implementation, and executes no model.
