# M6 GPU environment isolation plan

Status: **PROPOSED — NOT AN EXECUTION LOCK**
Milestone: M6D0
Starting commit: `f2a625b1b2ee832e3517561e8f64a8cde3ae9d0c`

This plan separates source requirements, historical host evidence, and proposed
compatibility choices. It installs nothing and does not turn a proposal into a
package pin. A final environment lock may be recorded only after an environment
is created, the required imports and method-specific smoke checks pass, and the
real resolved package set is captured.

## Current readiness

The recorded execution endpoint is `student20261@100.121.84.44`, repository
`/home/student20261/workdir/GPAT_TransferBench`, runtime root
`/home/student20261/workdir/GPAT_TransferBench_runtime`. The M6D0 read-only SSH
attempt reached authentication but failed with `Permission denied
(publickey,password)`. Consequently, current GPU repository state, platform,
compilers, conda environments, disk, assets, and runtime-root existence are
**UNVERIFIED**. Historical M5 evidence is retained below as historical evidence,
not current host fact.

No environment may be created from this plan until read-only host access is
restored and the host audit is rerun.

## Evidence classes

- **SOURCE REQUIREMENT**: stated or encoded by a pinned upstream source.
- **EXISTING HOST FACT**: observed live during M6D0. There are no remote host
  facts because authentication failed.
- **HISTORICAL HOST EVIDENCE**: M5 observations from 2026-09-21; useful for
  planning but requiring re-verification.
- **PROPOSED COMPATIBILITY CHOICE**: a candidate to probe. It is not a pin.

## Proposed isolation matrix

| Environment | Methods | Source-grounded baseline | Compiled surface | Existing M5 reuse | Gate before creation |
|---|---|---|---|---|---|
| `env-m6-core` | E01, E02, common audit/runtime | E01 production pixel adapter needs NumPy, Pillow with ImageCms, OpenCV; upstream requirement versions unspecified. E02 needs NumPy. PyYAML supports config tooling. | None in the selected E01/E02 paths | **UNVERIFIED**; historical M5 packages are newer than source declarations but E01/E02 need an import + synthetic operator/FFT probe | Restore SSH; inventory env; verify Pillow/ImageCms, OpenCV and NumPy; run existing synthetic E01/E02 tests without benchmark pixels |
| `env-m6-stdn` | E03 only | README: Python 3.6 and TensorFlow 1.13.0 tested; source uses TF1 graph APIs and `tensorflow.contrib`. README simultaneously says `>1.8,<1.13`, so exact TF pin is unresolved | TensorFlow GPU wheel/runtime; no repository custom extension | **NO** for historical M5 modern PyTorch env | Resolve/test the README contradiction; choose a TF1/Python/CUDA/cuDNN combination that supports RTX 3090 and source APIs; import/static smoke before any model construction |
| `env-m6-physics-std` | E04 only | Paper/framework metadata says TensorFlow but no official PhySTD code exists; controlled training graph dependency versions are unspecified. Geometry basis 3DDFA_V2 requires Python 3, PyTorch/torchvision, NumPy, OpenCV, SciPy, scikit-image, PyYAML and Cython | `Sim3DR_Cython` C++11/Cython extension; 3DDFA PyTorch regressor | **UNVERIFIED**, and sharing training + geometry in one environment is **not proven** | Reverify compilers/assets on GPU; resolve future TensorFlow training implementation boundary; decide, by probe, whether geometry preprocessing must be isolated from training; compile/import Sim3DR and run synthetic-only smoke |
| `env-m6-pcgan` | E05 only | README: author used Python 3.6/PyTorch 1.7.1; CUDA >=10.1. Named packages: dominate, torchgeometry, func-timeout, tqdm, matplotlib, OpenCV, lmdb, NumPy, GPUtil, Pillow, scikit-learn, visdom, ninja | JIT `fused_bias_act` and `upfirdn2d` C++/CUDA extensions | **NO until proved**; PyTorch 2.12/CUDA 13 compatibility is not source-grounded | Reverify nvcc/GCC/ninja; create isolated candidate based on source evidence; compile/import custom ops and run static/synthetic architecture smoke |
| `env-m6-dsdg` | E06c only | DSDG README: Python 3.8, PyTorch 1.6.0, torchvision 0.7.0. Main generator path also imports NumPy, OpenCV and Pillow; source uses CUDA-specific tensors/DataParallel | No repository custom extension for the E06c generator path | **NO** until compatibility probe; historical M5 stack differs substantially | Reverify GPU asset mapping; prove RTX 3090 support for candidate CUDA/runtime; import source safely and test constructor only after separately authorized environment work; physical batch 240 remains mandatory |
| `env-m6-difffas` | E07c only | Shipped environment pins Python 3.6.13, PyTorch 1.10.2 CUDA 11.3, torchvision 0.11.3, NumPy 1.19.2, tensorfn 0.1.28 and its recorded ecosystem | No repository custom extension, but CUDA-only allocations and old APIs occur in source | **NO**; historical M5 Python/PyTorch/CUDA stack conflicts with shipped environment | Determine whether the exact environment can be solved on the host/driver and RTX 3090; import-only probe tensorfn/config/custom_rn/diffusion; auxiliary checkpoint remains NOT_YET_TRAINED |

`env-m6-physics-std` may ultimately require two cooperating environments, one
for fixed 3DDFA geometry and one for TensorFlow training. That is an unresolved
execution-engineering choice, not authorization to precompute or alter the
scientific depth contract. No sharing with `env-m6-stdn` is approved without an
actual compatibility probe.

## Method dependency evidence

### E01 FAS-Aug

The upstream `requirements.txt` names torch, torchvision, torchaudio, NumPy,
Pillow, OpenCV, pandas, tqdm, scikit-learn, regex, pickle5 and yacs but gives no
versions; it also includes entries that are standard-library names or appear
mistyped. The benchmark's selective official pixel backend compiles the pinned
operator/helper AST and imports only NumPy, Pillow/ImageCms and OpenCV at pixel
execution. Therefore the selected E01 path is `VERSION_UNSPECIFIED` plus
`COMPATIBILITY_REQUIRES_PROBE`; the full upstream training dependency list is
not silently imposed on this generation-only adapter.

### E02 Frequency Substitution

The benchmark implementation is NumPy/FFT only. NumPy's resolved version must
be captured from the real environment because PCG64 and FFT numerical evidence
is logged. Status: `VERSION_UNSPECIFIED`, then `COMPATIBILITY_REQUIRES_PROBE`.

### E03 STDN

Pinned source uses `tf.Session`, `tf.ConfigProto`, `tf.py_func`,
`tf.image.resize_images`, `tf.train.AdamOptimizer` and `tf.contrib.layers`.
This is TensorFlow 1 generation. The README's “tested on Python 3.6 & TensorFlow
1.13.0” conflicts with its proposed range “>1.8.0 and <1.13.0”; M6D0 does not
choose between them. Python 3.6 is source-tested. CUDA/cuDNN compatibility is
not stated and must be resolved against the RTX 3090 driver. Status:
`SOURCE_RANGE_ONLY` and `COMPATIBILITY_REQUIRES_PROBE`.

### E04 Physics-STD controlled reconstruction

The paper records TensorFlow but no official implementation or version. The
A4 optimizer semantics reference TF `tf.train.AdamOptimizer`; that is not a
complete framework pin. Pinned 3DDFA_V2 has an unversioned `requirements.txt`
and builds `Sim3DR_Cython` from Cython + C++ with `-std=c++11`; its source also
uses removed NumPy alias `np.long`, so modern NumPy is not presumed compatible.
Production Sim3DR availability remains a separate hard gate. Status:
`VERSION_UNSPECIFIED` and `COMPATIBILITY_REQUIRES_PROBE`.

### E05 PCGAN controlled architecture resolution

Pinned Swapping Autoencoder README records Python 3.6, PyTorch 1.7.1 and CUDA
10.1 or newer. On supported CUDA it JIT-compiles fused activation and upfirdn2d
from C++/CUDA using `torch.utils.cpp_extension.load`; ninja and a compatible
host compiler/nvcc pair are required. PyTorch 2.12/CUDA 13 is not presumed
compatible. Status: source-tested versions are `SOURCE_PINNED` evidence, while
host compatibility remains `COMPATIBILITY_REQUIRES_PROBE`.

### E06c DSDG-BIN-IDFREE

The DSDG README explicitly lists Python 3.8, PyTorch 1.6.0 and torchvision
0.7.0. The used source path has CUDA-specific tensors and DataParallel. The
LightCNN asset is external and must be verified on the GPU host without loading
it. The E06c full physical batch of 240 cannot be replaced by naive gradient
accumulation. Status: `SOURCE_PINNED` plus `COMPATIBILITY_REQUIRES_PROBE` for
RTX 3090/CUDA and the controlled adapter.

### E07c DiffFAS-BIN-IDFREE

The pinned `environment.yml` is the strongest source evidence: Python 3.6.13,
PyTorch 1.10.2 CUDA 11.3, torchvision 0.11.3, NumPy 1.19.2,
tensorfn 0.1.28, pydantic 1.9.2, pyhocon 0.3.60, OpenCV 4.5.4.58 and the full
recorded ecosystem. The file also contains CUDA 11.7 meta/runtime entries next
to the CUDA 11.3 PyTorch build; this inconsistency must be solved and tested,
not normalized by guess. Source contains CUDA-only allocations and legacy
whole-module load semantics. Status: `SOURCE_PINNED` with
`COMPATIBILITY_REQUIRES_PROBE`.

## External assets

Laptop-side read-only verification passed for all four E04 3DDFA assets and the
E06c LightCNN checkpoint at their frozen `/media/cong/Data/...` paths. Size and
SHA256 match the scientific records. These paths are laptop paths. Their
existence and mapping under the GPU runtime root are **UNVERIFIED** because SSH
authentication failed. E07c's auxiliary encoder remains **NOT_YET_TRAINED**;
there is no hash or checkpoint to verify.

## Historical M5 evidence, not current fact

M5 recorded RTX 3090 (24,576 MiB, compute capability 8.6), driver 595.84,
`nvidia-smi` maximum CUDA 13.2, and an environment at
`/home/student20261/miniconda3/envs/gpat-m5/bin/python` with Python 3.11.16,
PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130 and cuDNN 92000. It also recorded
Linux 7.0.0-29 and NumPy 2.4.6. None was reverified in M6D0. System CUDA/nvcc
was not established by that evidence and must not be equated with PyTorch CUDA.

## Freeze procedure after access is restored

1. Re-run the M6D0 read-only host/repository/platform/environment/asset audit.
2. Resolve each environment's explicit blockers without changing scientific
   configs or source caches.
3. Create one candidate environment at a time in a later authorized milestone.
4. Run import-only and required synthetic/static method probes first; then the
   separately authorized model/runtime smoke, still without benchmark pixels.
5. Capture `conda list --explicit`, `conda env export`, `pip freeze --all`,
   Python/framework/CUDA/cuDNN/compiler facts and SHA256s from the actual
   passing environment.
6. Only then call the captured lock an execution pin and bind its hash into run
   manifests. Failed candidates remain audit evidence, never scientific pins.
