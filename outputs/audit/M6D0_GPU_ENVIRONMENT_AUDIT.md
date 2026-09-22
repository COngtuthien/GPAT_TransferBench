# M6D0 — GPU execution environment audit and freeze plan

Status: **COMPLETED WITH BLOCKERS**. This is audit/planning evidence only, not
an execution-environment freeze and not method execution.

Starting HEAD `f2a625b1b2ee832e3517561e8f64a8cde3ae9d0c`, branch
`m6-baselines`; the local worktree was clean at preflight.

## Central finding

The live GPU audit is blocked by SSH authentication. The requested read-only
connection to `student20261@100.121.84.44` reached SSH but returned
`Permission denied (publickey,password)` before any remote command ran. Current
GPU repository, runtime root, host platform, compilers, conda environments,
packages and external assets therefore remain **UNVERIFIED**.

M5 evidence is retained as historical evidence only: on 2026-09-21 it recorded
an RTX 3090 with 24,576 MiB, compute capability 8.6, driver 595.84,
NVIDIA-reported maximum CUDA 13.2, and `gpat-m5` at Python 3.11.16,
PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130 and cuDNN 92000. Those facts were
not reverified during M6D0. System CUDA is not inferred from PyTorch's bundled
CUDA runtime.

The operational config names repository
`/home/student20261/workdir/GPAT_TransferBench` and runtime root
`/home/student20261/workdir/GPAT_TransferBench_runtime`. README still names the
older `/home/sparc/workdir/longnm/GPAT_TransferBench` path. README is not
scientific or operational authority and was left unchanged for a later
documentation-only correction. Whether the actual GPU filesystem matches the
recorded paths could not be determined.

## Dependency conclusions

| Method | Source-grounded requirement | M5 reuse decision | Remaining gate |
|---|---|---|---|
| E01 | Selected official pixel path: NumPy, Pillow/ImageCms, OpenCV; versions unspecified | UNVERIFIED | Import plus synthetic official-operator smoke |
| E02 | NumPy PCG64/FFT; version unspecified | UNVERIFIED | Repeat deterministic selection/FFT smoke and record resolved NumPy |
| E03 | Python 3.6; TensorFlow 1 graph/`tf.contrib`; README says both “tested 1.13.0” and “>1.8,<1.13” | NO | Resolve contradiction and TF1/CUDA/cuDNN/RTX3090 compatibility |
| E04 | Paper framework TensorFlow, version unspecified; 3DDFA geometry uses unversioned PyTorch stack plus Cython/C++11 Sim3DR | NOT APPROVED | Resolve training implementation boundary; build/import Sim3DR on actual host |
| E05 | Source authors used Python 3.6/PyTorch 1.7.1; CUDA >=10.1; JIT StyleGAN2 C++/CUDA ops | NO until probe | Actual nvcc/GCC/ninja and extension build/runtime compatibility |
| E06c | Python 3.8, PyTorch 1.6.0, torchvision 0.7.0; CUDA-specific source | NO until probe | RTX3090 runtime compatibility, GPU asset, and physical batch 240 feasibility |
| E07c | Source environment pins Python 3.6.13, PyTorch 1.10.2/cu113, torchvision 0.11.3, NumPy 1.19.2, tensorfn 0.1.28; also includes CUDA 11.7 meta/runtime entries | NO | Solve mixed CUDA declarations and import old tensorfn ecosystem |

Detailed package lists, exact evidence classes, deprecated API surfaces and
blockers are recorded in the companion JSON and
`environments/m6_environment_plan.md`.

STDN is isolated because it uses `tf.Session`, `tf.ConfigProto`, `tf.py_func`,
`tf.image.resize_images`, `tf.train.AdamOptimizer` and `tf.contrib`. It must not
be forced into the modern PyTorch environment.

E04 has two independently old surfaces: an unspecified TensorFlow training
stack and pinned 3DDFA geometry. Its `Sim3DR/setup.py` compiles a Cython/C++11
extension against NumPy. The source uses removed `np.long`. Production
`Sim3DR_Cython` availability remains a separate gate. M6D0 does not decide
whether two cooperating environments are scientifically permissible; the
future execution boundary must be resolved first.

E05's pinned source uses `torch.utils.cpp_extension.load` for
`fused_bias_act` and `upfirdn2d`. The README's CUDA >=10.1 statement and
PyTorch 1.7.1 reference do not establish compatibility with historical
PyTorch 2.12/CUDA 13.

E06c's exact source requirements are Python 3.8/PyTorch 1.6.0/torchvision
0.7.0. The generator uses CUDA-specific allocation and DataParallel. Its
batch-statistical objective still requires physical batch 240; environment
work may not silently replace that with accumulation.

E07c supplies the strongest lock-like source evidence, but it is upstream
evidence rather than a GPAT execution lock. Its environment mixes a cu113
PyTorch build with CUDA 11.7 runtime packages and old tensorfn/pydantic. The
exact solve must be tested rather than normalized.

## External assets

Read-only laptop-side streaming SHA256 verified all frozen E04 3DDFA assets and
the E06c LightCNN checkpoint without deserialization:

- E04 `bfm_noneck_v3.pkl`: 24,393,598 bytes,
  `89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca`.
- E04 `tri.pkl`: 913,040 bytes,
  `0562a594d8369f7d1c86306522ff76fd3a2c81cf90ba9443893c1191ad0abbb8`.
- E04 `param_mean_std_62d_120x120.pkl`: 713 bytes,
  `090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795`.
- E04 `mb1_120x120.pth`: 13,755,269 bytes,
  `a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79`.
- E06c LightCNN: 123,844,849 bytes,
  `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964`.

These are `/media/cong/Data/...` laptop paths. No claim is made that the bytes
exist on the GPU host. E07c's auxiliary encoder correctly remains
**NOT_YET_TRAINED**; no checkpoint or SHA was fabricated.

## Proposed isolation and freeze policy

Six proposed groups are retained: `env-m6-core` (E01/E02), `env-m6-stdn`,
`env-m6-physics-std`, `env-m6-pcgan`, `env-m6-dsdg`, and `env-m6-difffas`.
No environment sharing beyond E01/E02 is authorized by current evidence. No
existing GPU environment is approved for reuse because no live inventory or
compatibility probe was possible.

The proposed matrix is not a lock. A version becomes an execution pin only
after the environment exists, imports pass, the required static/runtime smoke
passes, and real `conda list --explicit`, environment export and `pip freeze`
outputs are captured and hashed. M6D0 created no lock or conda YAML.

## Validation and access disclosure

`tools/m6d0_gpu_environment_preflight.py` validates the seven frozen
config/snapshot pairs, pinned checkout commits, evidence-status vocabulary,
plan groups, E07c pending checkpoint state and safety flags. Its success means
the static plan is internally consistent; execution readiness remains blocked.

M6D0 opened zero benchmark images and zero manifests. It did not load any
model weight. Existing asset checks used stat and streaming SHA256 only. The
failed SSH attempt executed zero remote commands. No GPU query, framework
import, model construction or model runtime occurred.

The committed ledger prefix has 94 records and SHA256
`b9bb74471d362af6b162505b8ebbe6bf1799ed5ed40d470cd70a37d6f9b0d561`.
M6D0 appended exactly one `M6D0_GPU_ENVIRONMENT_AUDIT` record, producing 95
records. The first 94 committed records remain byte-identical. The canonical
artifact index contains 461 unique current paths with current sizes/SHA256s and
CRLF line endings.

No package installation, environment creation, training, model construction,
weight loading, benchmark image execution, diffusion sampling, synthetic bank,
GPU training job, scientific checkpoint, scientific-config change, commit or
push occurred.
