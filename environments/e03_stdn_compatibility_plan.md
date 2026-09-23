# E03 STDN compatibility plan — M6D2a

**BLOCKED_BY_MULTIPLE_REASONS. Candidate matrix only; no final environment pin.**
E03 remains `FAITHFUL_OFFICIAL`, `IMPLEMENTED_NOT_EXECUTED`, and `ENVIRONMENT_NOT_QUALIFIED`.
No M6D2b environment creation is authorized or performed by this document.

## Evidence and candidate matrix

**SOURCE FACT:** pinned STDN README line 10 says “Tested on Python 3.6 & Tensorflow 1.13.0.”
and also requires “Tensorflow >1.8.0 and <1.13.0.” The strict upper bound excludes the
claimed tested version. Neither statement is silently preferred or corrected.

| ID | Python / TensorFlow | Package evidence | CUDA / cuDNN | Unchanged STDN / contrib | RTX 3090 and fidelity decision |
|---|---|---|---|---|---|
| A | 3.6 / literal 1.13.0 | No exact release files in queried indexes | Exact build unknown | Author claim; binary unavailable | Blocked by contradictory version evidence, exact-package absence, and unproven GPU route |
| B older | 3.6 / 1.9, 1.10, 1.11 | cp36 Linux wheels exist | 9 / 7 upstream | No: missing required symbols; contrib alone insufficient | Patching forbidden; legacy GPU route unproven |
| B reference | 3.6 / 1.12.0 | cp36 wheels and Conda GPU records | 9 / 7 upstream; Conda 9.0 / >=7.1.0,<=8.0a0 or 9.2 / >=7.2.0,<=8.0a0 | Static API candidate; contrib present; execution untested | Best README-range reference; no supported full Ampere route established |
| C adjacent | 3.6 / 1.13.1 | cp36 wheels and Conda records | 10.0 / 7.4 upstream; Conda variants differ | TF1 API candidate; contrib present | Outside both literal version claims; review required; GPU route unproven |
| C upstream | 3.6–3.7 / 1.15.x | Linux wheels; 1.15.0 Conda records | 10.0 / 7.4 for upstream 1.15.0; Conda >=7.6.4,<8 | TF1 API candidate; contrib retained | Later TF1 is not source-authorized fidelity; old-runtime GPU route unproven |
| C NVIDIA | 3.8 / NVIDIA 1.15.5, image 23.03-tf1-py3 | Vendor documents prebuilt image; no pull/digest verified | 12.1.0 / 8.7.0 | Potentially unchanged STDN; inspected vendor contrib retained | Credible Ampere hardware candidate; needs version/fidelity and numerical-default review first |
| D | 3.10 / TF 2.21 + compat.v1 | Installed live; public wheel metadata | Build 12.5.1 / major 9 | No; contrib absent; source edits needed | SOURCE CODE / FRAMEWORK ADAPTATION; not authorized |

**PACKAGE-METADATA FACT:** [TensorFlow Linux build table](https://www.tensorflow.org/install/source)
distinguishes the upstream build runtimes above. PyPI JSON and Anaconda linux-64 records,
including filenames, tags, dependencies and query hashes, are retained in the companion
[audit JSON](../outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.json). Python 3.6.13
metadata exists. No full transitive solver was run. Modern NumPy/protobuf/Keras/Matplotlib
must not be assumed to resolve with legacy TF1; a later review must resolve and record
those dependencies, Pillow/OpenCV, and their Python ABI constraints separately.
No queried public index supplied exact TF 1.13.0; this does not establish absence from
all historical or private build channels. RC and 1.13.1 are not equivalent substitutes.

**SOURCE FACT:** TF 1.9/1.10 root exports lack `ensure_shape`. TF 1.11's math export list
lacks `reduce_mean`/`reduce_sum`. TF 1.12 has the highlighted symbols, including
`tf.random.uniform`, `tf.ensure_shape`, `tf.initializers.global_variables`, and TF1
session/optimizer APIs. This narrows the practical range candidate to 1.12 among the
checked releases. All 62 direct TF symbol references are recorded with local file/line
text; iterator methods and optimizer/session methods are recorded separately.
Static symbols are necessary but do not prove native kernels, negative autotune values,
TensorShape behavior, contrib normalization, or complete execution compatibility.

**HOST FACT:** RTX 3090, capability 8.6, driver 595.84; Ubuntu 24.04.4 LTS,
kernel 7.0.0-29-generic, glibc 2.39, Conda 26.7.2, system nvcc 12.0.140.
System nvcc is independent of a future isolated CUDA runtime or container user space.

**COMPATIBILITY INFERENCE:** old CUDA application binaries may forward-JIT on Ampere
when they contain suitable PTX, but that condition does not establish every TensorFlow
kernel or linked cuDNN library. Do not declare all CUDA 9/10 binaries impossible, and do
not label them supported for full STDN without evidence. See the
[NVIDIA Ampere compatibility guide](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html)
and [historical framework matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/).
Replacing cuDNN 7 with cuDNN 8 beside a TF1 wheel is not an established ABI-compatible solution.
A CPU-only TF 1.12 diagnostic route could avoid GPU-library constraints but would not
resolve this requested RTX 3090 execution route or qualify the benchmark.

## Conditional route and M6D2b gates

The most credible hardware research lead is NVIDIA's documented
[23.03 TF1 container](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-23-03.html).
The host meets its published driver/capability gates. That establishes a potential
hardware route, not STDN fidelity or readiness. A container also separates old Python
user space from Ubuntu 24.04; no container runtime availability, registry pull, immutable
digest or native-library load was checked.

There is **no recommended executable M6D2b candidate yet**. Before any candidate is used:

1. Resolve the contradictory version evidence explicitly, or approve a documented
   framework-version fidelity review. Record why TF 1.12, TF 1.13.1, or NVIDIA TF 1.15.5 is
   acceptable; never edit the frozen scientific contract implicitly.
2. Resolve the benchmark wrapper boundary. Committed helpers use future annotations,
   `importlib.metadata.packages_distributions`, `str.removesuffix`, and `Path.is_relative_to`.
   Python 3.6 cannot run those helpers unchanged, and the Python 3.8 NVIDIA image also
   lacks required modern APIs. A modern controller/legacy worker boundary is a proposal
   requiring engineering and fidelity review, not an implemented workaround. The prior
   implementation audit explicitly does not claim a standalone qualified launcher.
3. For an approved NVIDIA candidate, inspect actual TF1/contrib imports and preserve
   Adam defaults, moving-average ordering, batch norm `updates_collections=None`,
   variable reuse, resize, RNG and checkpoint semantics. Review TF32, AMP and graph
   rewrite defaults; NVIDIA documents accelerated math in its
   [TF1 Ampere discussion](https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/).
   Disabling a precision feature alone is not proof of scientific equivalence.
4. Only under a separately authorized M6D2b, resolve isolated packages or an immutable
   container digest, record complete dependencies and loaded CUDA/cuDNN, then run
   import/API checks before any separately authorized synthetic graph check. Keep
   benchmark inputs, TEST, weights and training blocked until their own gates pass.

TF2 `compat.v1` cannot recover `tf.contrib`; the
[official migration guidance](https://www.tensorflow.org/guide/migrate/upgrade)
requires manual changes. Replacing layers, rewriting imports, PyTorch conversion,
loss/topology/optimizer/checkpoint changes require explicit fidelity review/amendment.
None is approved here.

## Frozen values

E03; `FAITHFUL_OFFICIAL`; learned=true; source `yaojieliu/ECCV20-STDN` at
`c79f1f8c615d2b8471b3df29da881bb18dd54c90`; checkpoint rule
`OFFICIAL_LATEST_FINAL_CKPT_50`; resolution 256, map 32, batch 2, G:D 2,
50 epochs, 2000 steps/epoch, 500 VAL steps; seeds 42/1337/2026; TEST forbidden.
All remain unchanged. This plan creates no environment or package lock.
