# M6 environment live addendum — M6D0b

**PROPOSED — NOT AN EXECUTION LOCK. Operational facts only.**

Authority: laptop `a6f78e9ba403148cb92e7f46ad4c72037dfc0499`, branch `m6-baselines`, clean at start.
This addendum supplements the unchanged `m6_environment_plan.md` and historical
M6D0 audits. Detailed evidence: `outputs/audit/M6D0B_LIVE_GPU_VERIFICATION.json`
and its Markdown companion. No scientific requirement has been changed.

Public-key SSH works to `student20261@100.121.84.44`; host
`nd-System-Product-Name`. GPU checkout is clean at
`89ea43e3d32d87d0641ec57bb62e6c72694dbe8a`, 15 commits behind current remote
authority. It was not synchronized and its source_cache is absent.

RTX 3090, 24576 MiB, capability 8.6, driver 595.84. NVIDIA's reported CUDA
compatibility is 13.2; system nvcc is 12.0 (V12.0.140); gpat-m5 bundles CUDA
13.0. These are distinct. gcc/g++ 13.3.0; cmake/ninja absent from shell PATH.
31Gi RAM and approximately 461G filesystem space available at audit.

Conda 26.7.2 is `/home/student20261/miniconda3/bin/conda`.
Existing environments: base, gpat-m5, grbkd, llm-movielens, stdn. The direct
gpat-m5 path is `/home/student20261/miniconda3/envs/gpat-m5`; Python 3.11.16,
pip 26.2.1, torch 2.12.1+cu130, torchvision 0.27.1+cu130, cuDNN 92000.
All five historical M5 version facts match. CUDA availability is true.
NumPy 2.4.6, Pillow/ImageCms 12.3.0, OpenCV 5.0.0 and PyYAML 6.0.3 import.
This supports gpat-m5 as an E01/E02 candidate only.

The existing stdn environment is Python 3.10.21, TensorFlow 2.21.0, torch
2.5.1+cu121 and torchvision 0.20.1+cu121 by installed package metadata. It
does not satisfy the isolated legacy TF1/contrib requirement.

Both expected runtime directories exist, owned by student20261:student20261,
mode 775; faces_256 has exactly 20615 PNG filenames. No pixels were decoded.
The four E04 assets and LightCNN were absent from established candidate path
conventions; no host-wide absence claim or invented storage binding is made.
E07c encoder remains **NOT_YET_TRAINED**.

## Updated isolation decisions

| Proposed environment | Decision | Existing candidate | Reason |
| --- | --- | --- | --- |
| env-m6-core | REUSE_EXISTING_CANDIDATE | gpat-m5 | All selected-path imports available. Official operator/PCG64/FFT compatibility evidence remains required. |
| env-m6-stdn | CREATE_NEW_CANDIDATE | None approved | Existing stdn is TensorFlow 2.21.0 / Python 3.10.21. Legacy TF1/contrib stack must remain isolated; version ambiguity unresolved. |
| env-m6-physics-std | STILL_UNRESOLVED | None approved | Training framework boundary unresolved. Geometry/training split may be necessary, but SPLIT_REQUIRED is not established by import inventory alone. |
| env-m6-pcgan | CREATE_NEW_CANDIDATE | None approved | Isolate source-era stack and future StyleGAN2 JIT compiler/toolkit/framework compatibility work. |
| env-m6-dsdg | CREATE_NEW_CANDIDATE | None approved | Isolate source stack and assess RTX 3090 physical batch 240 without changing the objective. |
| env-m6-difffas | CREATE_NEW_CANDIDATE | None approved | Isolate pinned source ecosystem; resolve mixed CUDA declarations before selecting versions. |

No method is EXECUTION_READY. All require future GPU repository synchronization
outside this milestone. The full method matrix follows so the distinction between
source requirements, live host facts and unresolved probes stays explicit.

| Method | Source requirements | Live statuses | Remaining gates |
| --- | --- | --- | --- |
| E01 | NumPy, Pillow/ImageCms and OpenCV for the selected official pixel path; versions unspecified. | EXISTING_ENV_CANDIDATE; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | Official operator compatibility probe remains pending; only imports checked. GPU source_cache is absent. |
| E02 | NumPy with explicit PCG64 semantics and FFT; final NumPy pin requires deterministic compatibility evidence. | EXISTING_ENV_CANDIDATE; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | PCG64 selection and FFT deterministic compatibility probe remains pending; no numerical operator executed. |
| E03 | Isolated Python 3.6-era TensorFlow 1 graph stack with tf.contrib. README tested TF 1.13.0 but also requires >1.8 and <1.13; unresolved contradiction. | NEW_ENV_REQUIRED; BLOCKED_BY_VERSION_AMBIGUITY; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | Existing stdn is Python 3.10.21 / TensorFlow 2.21.0, not the required TF1/contrib stack.; Resolve TF 1.13 versus <1.13 and legacy CUDA/cuDNN support on RTX 3090. |
| E04 | Training framework boundary remains unresolved (paper TensorFlow, no released implementation/version). 3DDFA geometry requires PyTorch plus Cython/C++11 Sim3DR. Training and geometry may need separate environments. | BLOCKED_BY_VERSION_AMBIGUITY; BLOCKED_BY_MISSING_ASSET; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | All four frozen GPU assets missing in inspected conventions.; gpat-m5 lacks TensorFlow, Cython, SciPy and Sim3DR_Cython; GPU source_cache is absent.; Resolve training framework/version boundary and possible geometry/training split; production Sim3DR compatibility remains pending. |
| E05 | Source reference Python 3.6 / PyTorch 1.7.1, CUDA >=10.1; StyleGAN2 JIT C++/CUDA fused_bias_act and upfirdn2d extensions. | NEW_ENV_REQUIRED; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | Modern gpat-m5 differs from source reference; no environment approved.; System nvcc 12.0 versus framework CUDA 13.0 requires a compatible future compiler/toolkit/framework combination.; gcc/g++ 13.3.0 observed; ninja and cmake absent from shell PATH and ninja module missing. JIT build/runtime compatibility not probed. |
| E06c | Source specifies Python 3.8, PyTorch 1.6.0 and torchvision 0.7.0. Full physical/effective batch 240, accumulation 1, replica factor 1; no naive accumulation fallback. | NEW_ENV_REQUIRED; BLOCKED_BY_MISSING_ASSET; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | LightCNN missing in inspected GPU conventions.; Existing environments do not match the Python 3.8 / torch 1.6.0 / torchvision 0.7.0 source stack.; RTX 3090 compatibility and physical-batch-240 VRAM feasibility unproven; no model or allocation feasibility probe performed. |
| E07c | Source environment pins Python 3.6.13, PyTorch 1.10.2 CUDA 11.3, torchvision 0.11.3, NumPy 1.19.2 and tensorfn 0.1.28; also declares CUDA 11.7. No version resolution selected. | NEW_ENV_REQUIRED; BLOCKED_BY_VERSION_AMBIGUITY; BLOCKED_BY_MISSING_ASSET; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | No existing environment matches the source stack; tensorfn is missing from gpat-m5.; Mixed CUDA 11.3/11.7 declarations remain unresolved.; Auxiliary conditioning encoder remains NOT_YET_TRAINED; required future asset unavailable. |

E04's Python.h exists in gpat-m5, but Cython, SciPy and Sim3DR are missing.
Possible geometry/training separation is unresolved, not a scientific substitution.
E05 requires future JIT compatibility work; the system/framework CUDA versions
differ. No extension was compiled. E06c full physical/effective batch 240 remains
mandatory; memory feasibility is unproven. No accumulation fallback is introduced.
E03 TF 1.13 versus <1.13 and E07c CUDA 11.3 versus 11.7 remain unresolved.

Future compatibility probes and actual environment resolution must precede a
final execution lock. No environment creation, package installation, scientific
execution, checkpoint, source/config modification, GPU synchronization, commit
or push occurred. The M6D0 freeze policy remains unchanged.
