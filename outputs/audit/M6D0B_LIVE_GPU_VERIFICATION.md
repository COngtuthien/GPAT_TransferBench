# M6D0b — live GPU verification after SSH recovery

**COMPLETED WITH BLOCKERS. Read-only infrastructure/environment audit. No method is execution-ready.**

This additive audit resolves operational unknowns in historical M6D0 evidence.
It does not revise scientific contracts or freeze environment versions.
Current laptop authority is `a6f78e9ba403148cb92e7f46ad4c72037dfc0499`. All M6D0 artifacts and the original
environment plan remain unchanged. Live inventory timestamp:
`2026-09-22T17:09:24.929042+00:00`; final remote check: `2026-09-22T17:11:10Z`.

## Laptop, SSH and GPU repository

- Laptop HEAD and `origin/m6-baselines`: `a6f78e9ba403148cb92e7f46ad4c72037dfc0499`.
- Branch `m6-baselines`; divergence `0 0`; worktree **clean at start**.
  Final worktree contains only the four additive deliverables and ledger/index updates.
- SSH: **PUBLIC_KEY_WORKING**, with BatchMode, publickey-only preference and
  password authentication disabled. Initial sandbox socket denial was resolved
  with network escalation; actual public-key authentication succeeded.
- Host `nd-System-Product-Name`; user `student20261`.
- GPU repository `/home/student20261/workdir/GPAT_TransferBench`:
  HEAD `89ea43e3d32d87d0641ec57bb62e6c72694dbe8a`, branch `m6-baselines`,
  **clean at start and end**. Its `third_party/source_cache` directory is absent.
- Origin `git@github.com:COngtuthien/GPAT_TransferBench.git`;
  live `git ls-remote origin refs/heads/m6-baselines` returned `a6f78e9ba403148cb92e7f46ad4c72037dfc0499`.
- The GPU HEAD is an ancestor **15 commits behind**. Derived from existing
  laptop Git objects: ancestor check exit 0; left/right count `0 15`.
  No GPU fetch, pull, checkout, reset, merge or synchronization occurred.

## Platform and three distinct CUDA layers

| Fact | Live observation |
| --- | --- |
| OS / kernel | Ubuntu 24.04.4 LTS (Noble Numbat) / 7.0.0-29-generic |
| GPU / VRAM / capability | NVIDIA GeForce RTX 3090 / 24576 MiB / 8.6 |
| Driver / NVIDIA CUDA compatibility | 595.84 / 13.2 |
| System CUDA toolkit | /usr/bin/nvcc; CUDA 12.0; V12.0.140 |
| Framework-bundled CUDA runtime | gpat-m5 torch CUDA 13.0 |
| gcc | /usr/bin/gcc; Ubuntu 13.3.0-6ubuntu2~24.04.1 |
| g++ | /usr/bin/g++; Ubuntu 13.3.0-6ubuntu2~24.04.1 |
| cmake / ninja executables | Neither found on audited shell PATH |
| RAM | 31Gi total; 4.3Gi used; 3.0Gi free; 26Gi available |
| Swap | 8.0Gi total; 527Mi used |
| Disk | /dev/nvme0n1p3 mounted /; 916G total, 410G used, 461G available, 48% used |
| Exact final disk bytes | 983366017024 total; 439225942016 used; 494140010496 available |

The same filesystem backs `/home/student20261`, its `workdir`, and the runtime
root. Driver compatibility **13.2**, system toolkit **12.0**, and framework
runtime **13.0** are separate observations. Raw `uname -a`, OS release,
`nvidia-smi`, compiler, memory and filesystem output is retained in the JSON.

## Conda and existing environments

Conda **26.7.2**, executable `/home/student20261/miniconda3/bin/conda`,
base `/home/student20261/miniconda3`. The noninteractive login shell did not
expose the conda function or CONDA_EXE. The environments registry and existing
installation directory resolved the executable; subsequent calls used it directly.
No shell initialization or environment activation was required.

| Name | Absolute path | Python | Installed framework metadata |
| --- | --- | --- | --- |
| base | /home/student20261/miniconda3 | Python 3.14.6 | torch 2.14.0 |
| gpat-m5 | /home/student20261/miniconda3/envs/gpat-m5 | Python 3.11.16 | torch 2.12.1+cu130, torchvision 0.27.1+cu130 |
| grbkd | /home/student20261/miniconda3/envs/grbkd | Python 3.12.13 | torch 2.13.0, torchvision 0.28.0 |
| llm-movielens | /home/student20261/miniconda3/envs/llm-movielens | Python 3.13.15 | torch 2.14.0 |
| stdn | /home/student20261/miniconda3/envs/stdn | Python 3.10.21 | tensorflow 2.21.0, torch 2.5.1+cu121, torchvision 0.20.1+cu121 |

Other environments were inspected through `python --version` and installed
dist-info metadata, without heavyweight imports. The environment named `stdn`
contains TensorFlow **2.21.0**, so its name does not establish legacy STDN
compatibility. No existing legacy stack is approved for reuse.

## gpat-m5 imports and historical comparison

Environment: `/home/student20261/miniconda3/envs/gpat-m5`.
Direct Python executable: that path plus `/bin/python`; Python **3.11.16**.
Direct `/bin/pip --version`: **pip 26.2.1**, Python 3.11.

| Import | Status | Exact version or error |
| --- | --- | --- |
| torch | AVAILABLE | 2.12.1+cu130 |
| torchvision | AVAILABLE | 0.27.1+cu130 |
| tensorflow | MISSING | ModuleNotFoundError: No module named 'tensorflow' |
| cv2 | AVAILABLE | 5.0.0 |
| numpy | AVAILABLE | 2.4.6 |
| scipy | MISSING | ModuleNotFoundError: No module named 'scipy' |
| sklearn | MISSING | ModuleNotFoundError: No module named 'sklearn' |
| pandas | AVAILABLE | 3.0.6 |
| pyarrow | AVAILABLE | 25.0.1 |
| PIL | AVAILABLE | 12.3.0 |
| yaml | AVAILABLE | 6.0.3 |
| tensorfn | MISSING | ModuleNotFoundError: No module named 'tensorfn' |
| Cython | MISSING | ModuleNotFoundError: No module named 'Cython' |
| onnxruntime | MISSING | ModuleNotFoundError: No module named 'onnxruntime' |
| lpips | MISSING | ModuleNotFoundError: No module named 'lpips' |
| PIL.ImageCms | AVAILABLE | 12.3.0 |
| Sim3DR_Cython | MISSING | ModuleNotFoundError: No module named 'Sim3DR_Cython' |
| ninja | MISSING | ModuleNotFoundError: No module named 'ninja' |

PyTorch CUDA available **true**; device **NVIDIA GeForce RTX 3090**;
capability **(8, 6)**; cuDNN **92000**. No model was constructed.

| Historical M5 fact | Historical | Live | Classification |
| --- | --- | --- | --- |
| python | 3.11.16 | 3.11.16 | MATCHES_HISTORICAL_M5 |
| torch | 2.12.1+cu130 | 2.12.1+cu130 | MATCHES_HISTORICAL_M5 |
| torchvision | 0.27.1+cu130 | 0.27.1+cu130 | MATCHES_HISTORICAL_M5 |
| torch_cuda_runtime | 13.0 | 13.0 | MATCHES_HISTORICAL_M5 |
| cudnn | 92000 | 92000 | MATCHES_HISTORICAL_M5 |

Fresh-process flags, read without modification: matmul `allow_tf32=False`,
cuDNN `allow_tf32=True`, `benchmark=False`, `deterministic=False`.
Historical M5 run-configured cuDNN TF32 was false and deterministic was true;
fresh import defaults are not execution settings or a scientific-contract change.

## Runtime and external assets

`/home/student20261/workdir/GPAT_TransferBench_runtime` and its `data/processed/faces_256` resolve to themselves.
Both are owned by `student20261:student20261`, mode **775 / drwxrwxr-x**.
Available filesystem bytes at final capture: **494140010496**.
PNG filename metadata count: **20615**, matching the expected count. No image
pixels were opened or decoded. Runtime root contains only the `data` child.

Asset identity authority is the **current laptop** configs and prior committed
provenance, not the lagging GPU checkout. Inspected conventions were the exact
recorded laptop paths, their runtime-relative `third_party_weights` mappings
under the established GPU root, and pinned source-relative paths under the GPU
repository. These are candidate checks, not newly asserted storage bindings.
All inspected candidates were absent; no asset was available to stat/hash.
Absence is scoped to these paths, not an exhaustive claim about the host disk.

| Method / asset | Frozen bytes | Frozen SHA256 | GPU status |
| --- | --- | --- | --- |
| E04 / configs/bfm_noneck_v3.pkl | 24393598 | 89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca | MISSING_IN_INSPECTED_CONVENTIONS |
| E04 / configs/tri.pkl | 913040 | 0562a594d8369f7d1c86306522ff76fd3a2c81cf90ba9443893c1191ad0abbb8 | MISSING_IN_INSPECTED_CONVENTIONS |
| E04 / configs/param_mean_std_62d_120x120.pkl | 713 | 090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795 | MISSING_IN_INSPECTED_CONVENTIONS |
| E04 / weights/mb1_120x120.pth | 13755269 | a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79 | MISSING_IN_INSPECTED_CONVENTIONS |
| E06c / LightCNN_29Layers_V2_checkpoint.pth.tar | 123844849 | d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964 | MISSING_IN_INSPECTED_CONVENTIONS |

E07c auxiliary conditioning encoder: **NOT_YET_TRAINED**;
SHA remains `RECORDED_AFTER_TRAINING`. Nothing was created or loaded.

## Compiler and extension readiness

E04: gcc/g++ 13.3.0 are present. `gpat-m5` has Python development header
`/home/student20261/miniconda3/envs/gpat-m5/include/python3.11/Python.h`.
Cython and `Sim3DR_Cython` imports are missing; GPU source_cache is absent.
The 3DDFA source requires Cython/C++11 and NumPy. No Sim3DR build occurred.
Training framework/version and potential training/geometry separation remain unresolved.

E05: gcc/g++ 13.3.0, nvcc 12.0 and torch 2.12.1+cu130 / CUDA 13.0 are live
facts. Ninja is missing from gpat-m5 and shell PATH. Cmake is absent from PATH.
The observed compiler/toolkit/framework combination is not approved for the
source's StyleGAN2 JIT extensions. No extension was compiled.

## Live method readiness matrix

| Method | Source requirements | Live statuses | Remaining gates |
| --- | --- | --- | --- |
| E01 | NumPy, Pillow/ImageCms and OpenCV for the selected official pixel path; versions unspecified. | EXISTING_ENV_CANDIDATE; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | Official operator compatibility probe remains pending; only imports checked. GPU source_cache is absent. |
| E02 | NumPy with explicit PCG64 semantics and FFT; final NumPy pin requires deterministic compatibility evidence. | EXISTING_ENV_CANDIDATE; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | PCG64 selection and FFT deterministic compatibility probe remains pending; no numerical operator executed. |
| E03 | Isolated Python 3.6-era TensorFlow 1 graph stack with tf.contrib. README tested TF 1.13.0 but also requires >1.8 and <1.13; unresolved contradiction. | NEW_ENV_REQUIRED; BLOCKED_BY_VERSION_AMBIGUITY; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | Existing stdn is Python 3.10.21 / TensorFlow 2.21.0, not the required TF1/contrib stack.; Resolve TF 1.13 versus <1.13 and legacy CUDA/cuDNN support on RTX 3090. |
| E04 | Training framework boundary remains unresolved (paper TensorFlow, no released implementation/version). 3DDFA geometry requires PyTorch plus Cython/C++11 Sim3DR. Training and geometry may need separate environments. | BLOCKED_BY_VERSION_AMBIGUITY; BLOCKED_BY_MISSING_ASSET; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | All four frozen GPU assets missing in inspected conventions.; gpat-m5 lacks TensorFlow, Cython, SciPy and Sim3DR_Cython; GPU source_cache is absent.; Resolve training framework/version boundary and possible geometry/training split; production Sim3DR compatibility remains pending. |
| E05 | Source reference Python 3.6 / PyTorch 1.7.1, CUDA >=10.1; StyleGAN2 JIT C++/CUDA fused_bias_act and upfirdn2d extensions. | NEW_ENV_REQUIRED; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | Modern gpat-m5 differs from source reference; no environment approved.; System nvcc 12.0 versus framework CUDA 13.0 requires a compatible future compiler/toolkit/framework combination.; gcc/g++ 13.3.0 observed; ninja and cmake absent from shell PATH and ninja module missing. JIT build/runtime compatibility not probed. |
| E06c | Source specifies Python 3.8, PyTorch 1.6.0 and torchvision 0.7.0. Full physical/effective batch 240, accumulation 1, replica factor 1; no naive accumulation fallback. | NEW_ENV_REQUIRED; BLOCKED_BY_MISSING_ASSET; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | LightCNN missing in inspected GPU conventions.; Existing environments do not match the Python 3.8 / torch 1.6.0 / torchvision 0.7.0 source stack.; RTX 3090 compatibility and physical-batch-240 VRAM feasibility unproven; no model or allocation feasibility probe performed. |
| E07c | Source environment pins Python 3.6.13, PyTorch 1.10.2 CUDA 11.3, torchvision 0.11.3, NumPy 1.19.2 and tensorfn 0.1.28; also declares CUDA 11.7. No version resolution selected. | NEW_ENV_REQUIRED; BLOCKED_BY_VERSION_AMBIGUITY; BLOCKED_BY_MISSING_ASSET; REQUIRES_COMPATIBILITY_PROBE; BLOCKED_BY_GPU_REPO_NOT_SYNCED | No existing environment matches the source stack; tensorfn is missing from gpat-m5.; Mixed CUDA 11.3/11.7 declarations remain unresolved.; Auxiliary conditioning encoder remains NOT_YET_TRAINED; required future asset unavailable. |

Every row retains `BLOCKED_BY_GPU_REPO_NOT_SYNCED`. E01/E02 are only
`EXISTING_ENV_CANDIDATE`; imports do not prove operator or deterministic PCG64
compatibility. E03's TF contradiction and E07c's CUDA contradiction are preserved,
not resolved by choosing versions. E06c retains full physical/effective batch 240.

## Updated environment isolation matrix

| Proposed environment | Decision | Existing candidate | Reason |
| --- | --- | --- | --- |
| env-m6-core | REUSE_EXISTING_CANDIDATE | gpat-m5 | All selected-path imports available. Official operator/PCG64/FFT compatibility evidence remains required. |
| env-m6-stdn | CREATE_NEW_CANDIDATE | None approved | Existing stdn is TensorFlow 2.21.0 / Python 3.10.21. Legacy TF1/contrib stack must remain isolated; version ambiguity unresolved. |
| env-m6-physics-std | STILL_UNRESOLVED | None approved | Training framework boundary unresolved. Geometry/training split may be necessary, but SPLIT_REQUIRED is not established by import inventory alone. |
| env-m6-pcgan | CREATE_NEW_CANDIDATE | None approved | Isolate source-era stack and future StyleGAN2 JIT compiler/toolkit/framework compatibility work. |
| env-m6-dsdg | CREATE_NEW_CANDIDATE | None approved | Isolate source stack and assess RTX 3090 physical batch 240 without changing the objective. |
| env-m6-difffas | CREATE_NEW_CANDIDATE | None approved | Isolate pinned source ecosystem; resolve mixed CUDA declarations before selecting versions. |

These are proposals, not locks or permission to create environments in this
milestone. The existence of gpat-m5 does not justify sharing legacy method stacks.
No versions have been guessed or frozen.

## Validation, ledger and firewall

Preflight: `python tools/m6d0b_live_gpu_preflight.py --final` —
**LIVE_AUDIT_VALIDATION_PASS**. Validates recorded live evidence, seven
config/snapshot pairs, six source pins, input hashes, additive file scope,
ledger prefix and finalized index; executes no scientific method or tests.

Ledger prefix: **95 committed rows**, **236873 bytes**,
SHA256 `8855f49e9474f23aa7059a9b461db21e940f3a205ecb3dc219d98eda94169979` calculated before modification.
Exactly one `M6D0B_LIVE_GPU_VERIFICATION` record is appended: **96 final rows**.
The first 95 rows remain byte-identical to the committed prefix.
The canonical artifact index is rebuilt only at finalization, with **465 unique
paths**, current sizes and hashes, and CRLF line endings. It excludes itself
and the ledger by existing policy. Final six file hashes are reported in the
completion message; the ledger binds the four new deliverables without a hash cycle.

Initial overstrict import guarding prevented PyTorch's temporary-file checks;
that harness failure was not classified as a missing installed package in the
final matrix. The successful fresh-process probe disabled bytecode and blocked
persistent Python file mutations while allowing `/dev/null` and `/tmp`.
It observed an `os.mkdir` attempt for `/tmp/torchinductor_student20261` during
import. This is outside the repository, runtime and environment; no JIT extension
was built. The guard is Python-level, not a syscall audit. Full probe source and
observed operations are retained in the JSON.

Benchmark image opens/decodes, manifest interpretation/referenced-image reads,
weight loads and model constructions: **zero**. The mandated index builder and
index verifier stream tracked file bytes, including opaque manifest bytes, solely
for hashes; they do not interpret manifests or read their referenced images.

Protected configs/methods, configs/frozen, configs/amendments,
frozen_config_snapshot, run_logging_v1.yaml, manifests, gpatbench, docs/spec,
methods, source_pins.json and source_cache have zero changes. Historical M6D0
artifacts and the environment plan remain byte-identical. GPU checkout remained
clean and unchanged; no GPU runtime write command was run.

**NO GPU REPO SYNC; NO PACKAGE INSTALL; NO ENV CREATION; NO MODEL CONSTRUCTION;
NO WEIGHT LOADING; NO BENCHMARK IMAGE DECODING; NO TRAINING; NO DIFFUSION SAMPLING;
NO SYNTHETIC BANK; NO SCIENTIFIC CHECKPOINT; NO SCIENTIFIC CONFIG CHANGE;
NO COMMIT; NO PUSH.** No optimizer, backward pass, TEST execution, shell config
modification or final environment lock occurred.
