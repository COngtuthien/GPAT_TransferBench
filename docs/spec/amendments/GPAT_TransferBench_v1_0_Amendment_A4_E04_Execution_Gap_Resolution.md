# GPAT-TransferBench v1.0 Amendment A4

## E04 execution-gap resolution

**Status:** OWNER-APPROVED · ADDITIVE · NON-DESTRUCTIVE
**Applies to:** E04 Physics-STD (controlled geometry/depth reconstruction)
**Base:** M6B frozen E04 configuration; Amendment A3

This amendment resolves exactly two execution-affecting gaps identified during
the M6C2b1 preflight. It does not change any other scientific parameter and
does not edit the M6B YAML or its frozen snapshot.

## A4-1 — live-depth conversion order

For a live E04 depth target `M0`, the execution order is frozen as follows:

1. reconstruct and project the dense 3DDFA mesh;
2. normalize vertex `z` exactly as pinned 3DDFA_V2 `utils/depth.py`;
3. rasterize through the pinned official Sim3DR path;
4. preserve the official renderer's `uint8` output at 256×256;
5. resize 256×256 to 32×32 with `cv2.INTER_AREA` while still `uint8`;
6. cast the resized result to `float32`;
7. divide by `255.0`;
8. clip to `[0, 1]`;
9. emit exactly shape `(32, 32)`.

For spoof samples, `M0` remains `zeros((32,32), dtype=float32)` and no
reconstruction is invoked.

This is a **BENCHMARK_DEFINED** clarification of Amendment A3, not a PhySTD
paper fact. The pinned renderer writes normalized color values into an image-
typed `uint8` buffer. Resizing that buffer before conversion is therefore the
nearest semantics-preserving interpretation. Converting to float before
resizing is explicitly not the selected path; the two orders are numerically
non-equivalent because area interpolation and 8-bit quantization do not
commute.

## A4-2 — E04 optimizer

Published PhySTD/TPAMI execution details identify TensorFlow, initial learning
rate `5e-5`, 150,000 iterations, batch 8, division of the learning rate by 10
every 45,000 iterations, and normal initialization `[0, 0.02]`. They do not
identify the optimizer algorithm, and no official PhySTD implementation was
located.

For this controlled reconstruction, E04 uses:

* algorithm: `Adam`;
* semantic reference: TensorFlow `tf.train.AdamOptimizer(lr)`;
* `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`;
* weight decay: none.

The existing learning rate, schedule, iteration budget, batch size and weight
initialization remain unchanged. The Adam choice is
`BENCHMARK_DEFINED_NEAREST_OFFICIAL_PREDECESSOR_BEHAVIOR`, not PAPER,
OFFICIAL_PHYSTD, or AUTHOR_SPECIFIED. The provenance is the pinned official
predecessor `yaojieliu/ECCV20-STDN` at commit
`c79f1f8c615d2b8471b3df29da881bb18dd54c90`, whose
`model/model.py` constructs `tf.train.AdamOptimizer(lr)`.

## Scope and immutability

The machine-readable overlay at
`configs/amendments/e04_a4_execution_resolution.yaml` is an additional
immutable execution input. It contains only these two resolutions, the base
config identity, and provenance. No training, model execution, benchmark-data
access, checkpoint creation, or configuration rewrite is performed by A4.
