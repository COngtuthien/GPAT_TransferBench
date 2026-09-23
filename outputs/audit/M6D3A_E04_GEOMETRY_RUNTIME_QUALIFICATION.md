# M6D3a — E04 fixed geometry/depth runtime qualification

Decision: **E04_GEOMETRY_RUNTIME_QUALIFIED**. Scope: **E04_FIXED_GEOMETRY_DEPTH_ONLY**.
Reporting label: **Physics-STD (controlled geometry/depth reconstruction)**.
Fidelity: **CONTROLLED_ADAPTATION**.

- geometry_runtime = QUALIFIED
- training_runtime = NOT_YET_QUALIFIED
- training_graph = NOT_YET_EXECUTED
- E04 REMAINS IMPLEMENTED_NOT_EXECUTED

This qualifies the fixed auxiliary pipeline on synthetic inputs only. No full E04
training-environment or benchmark crop/ROI integration claim is made.
Historical package versions are not mandatory; compatibility is established by the probes.

## Repository synchronization

Laptop HEAD = origin/m6-baselines = `17c46f833a31e0a8a038cd013d12bbd888f2e18f`, branch `m6-baselines`,
initial worktree clean and divergence `0 0`. GPU advanced from
`6aebf148bceb5e4d3692326dbabc6b3e2390fb9d` to the same target.
Ancestry was proved and distance was exactly **1**. Only the specified branch fetch
and `git merge --ff-only origin/m6-baselines` performed the sync. GPU was clean
with divergence `0 0` immediately afterward. Both HEADs remain unchanged;
M6D3a changes are left uncommitted in the worktrees.

## Source, assets and compatibility

3DDFA_V2 source pin: `1b6c67601abffc1e9f248b291708aef0e43b55ae`.
Tree: `28492fd307da6644c3e2c21d4b9a97a1f646c65f`. All **113** source-release files match pinned
Git blobs before/after on laptop and GPU. Required source SHA256s are in the JSON.
The GPU source-only deployment excluded all nine upstream external assets, including
their Git blobs; exactly the four E04 assets were deployed separately under
`/home/student20261/workdir/GPAT_TransferBench_runtime/third_party_weights/tddfa_v2`. No source bytes were patched, and no external model asset or
compiled extension was put in source_cache.

| Relative asset | Bytes | SHA256 |
|---|---:|---|
| `configs/bfm_noneck_v3.pkl` | 24393598 | `89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca` |
| `configs/param_mean_std_62d_120x120.pkl` | 713 | `090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795` |
| `configs/tri.pkl` | 913040 | `0562a594d8369f7d1c86306522ff76fd3a2c81cf90ba9443893c1191ad0abbb8` |
| `weights/mb1_120x120.pth` | 13755269 | `a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79` |

Compatibility changes are disclosed in
`environments/e04_geometry.compatibility_patch_manifest.json`: verified location
resolution for GPU assets and external compiled extension; unchanged upstream model,
loader and normalization bindings; checkpoint-key diagnostics; expanded build-input
verification. No NumPy alias, setup.py, Cython, compiler or scientific-source patch
was required. Geometry math, projection, rasterization and model topology are unchanged.

The first diagnostic attempt rejected `fc_lm.bias` and `fc_lm.weight` before inference.
Inspection of the pinned loader established that it already ignores those unused
landmark-head tensors. The guard was aligned to that exact behavior. Final missing
keys: `[]`; unexpected raw checkpoint keys: `['fc_lm.bias', 'fc_lm.weight']`,
shapes `[136]` and `[136,1024]`; shape mismatches: `[]`. All **164** inference-state
tensors exactly equal their checkpoint tensors (166 checkpoint entries total).

## Isolated environment and build

Environment: `gpat-m6-e04-geometry` at `/home/student20261/miniconda3/envs/gpat-m6-e04-geometry`.
Python: **3.12.14**; ABI `cpython-312-x86_64-linux-gnu`; NumPy C ABI `33554432`.

| Package | Version |
|---|---|
| Cython | 3.3.0 |
| PyYAML | 6.0.3 |
| numpy | 2.5.3 |
| opencv-python-headless | 5.0.0.93 |
| pytest | 9.1.1 |
| scikit-image | 0.26.0 |
| scipy | 1.18.1 |
| setuptools | 83.0.0 |
| torch | 2.14.0 |
| torchvision | 0.29.0 |

PyTorch CUDA **13.0**; cuDNN API **92400** (distribution 9.24.0.43).
GPU: `NVIDIA GeForce RTX 3090, 595.84, GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`. Sim3DR is a CPU extension; the regressor was exercised on CPU
and CUDA. `pip check`: no broken requirements. Only the new environment received
packages; gpat-m5, gpat-m6-e03, stdn and base were not install targets.
Post-task protected package inventories are retained; all committed E03 locked
package versions still match. M6D3a did not collect pre-task filesystem fingerprints
for protected environments and does not claim that stronger measurement.

Build working directory:
`/home/student20261/workdir/GPAT_TransferBench_runtime/builds/e04_geometry/tddfa_v2/Sim3DR`.

```sh
/home/student20261/miniconda3/envs/gpat-m6-e04-geometry/bin/python setup.py build_ext -i
```

Compiler: `/usr/bin/gcc` and `/usr/bin/g++`, Ubuntu **13.3.0-6ubuntu2~24.04.1**.
The built ELF `.comment` independently identifies GCC 13.3.0. Cython **3.3.0**.
Build exit code: **0**. Original setup.py, rasterize.pyx, rasterize.h and
rasterize_kernel.cpp were copied unchanged and their hashes checked.

Extension: `/home/student20261/workdir/GPAT_TransferBench_runtime/builds/e04_geometry/tddfa_v2/Sim3DR/Sim3DR_Cython.cpython-312-x86_64-linux-gnu.so`.
SHA256: `7fb03434a0e1ce4f86e248f87d462d775c24c40aad843724880bb21341c27aa4`.
Actual `import Sim3DR_Cython` succeeds. `DepthRenderer()` loads it through
`ExtensionFileLoader`; no fallback or injected kernel supplies production evidence.

## Synthetic qualification

The input is an in-memory uint8 BGR `[120,120,3]` arange/modulo-256 pattern.
MobileNet v1 uses widen_factor 1.0, 62 outputs, shape_dim 40 and exp_dim 10.
Only the frozen mb1_120x120 checkpoint is loaded. The original normalization and
parameter mean/std are used. The synthetic reconstruction ROI is `[0,0,256,256]`.
No detector, benchmark image, dataset or decoder is invoked.

CPU and CUDA model construction, checkpoint loading and finite `[1,62]` inference
pass. Fitted vectors are float32 `[62]`. Same-input repeats are byte-identical.
Synthetic fitted-parameter reconstruction gives dense `[38365,3]`, triangles
`[76073,3]`, and Q140 `[140,3]`. Exactly 140 unique frozen indices are consumed;
the first 68 match the BFM iBUG anchors in exact order. Q140 is never rederived.
Q140 SHA256: `1b884401377f5aadd3d56857f05fddf70e2c54a52a9eb2160cebc06a74031a1f`.
`recon_vers`, `_parse_param` and `similar_transform` remain unchanged.

Real Sim3DR synthetic-mesh probes pass: larger z wins; triangle-order invariance;
uint8 `[256,256]` intermediate; captured face mask; zero background; exact
**uint8 INTER_AREA resize → float32 → /255 → clip [0,1]**; float32 `[32,32]` output.
The end-to-end regressor → GeometryAdapter → real Sim3DR → live depth path passes
with finite values in `[0, 0.9882352948188782]`, **18,983** written mask pixels and
repeatable outputs. A second clean CUDA process produces identical fitted vectors,
mesh and final depth bytes. CPU/CUDA meshes differ slightly; no cross-device mesh
bitwise claim is made. Their final depth bytes match for this fixture:
`bf442bca634087075cd27082eef172d39576006efcfc54a2595b3ad613099531`.
Spoof output is exactly float32 zeros `[32,32]`, with zero regressor,
reconstruction and renderer calls.

## Tests and training assessment

**61/61 PASS; zero failures, errors or skips** across the three requested files
and `tests/test_m6d3a_e04_geometry_runtime.py`. The unchanged
`test_official_extension_smoke_when_available` now **PASSes**. Existing C++ shim
unit tests remain regression coverage and are not production qualification evidence.
Four supplementary laptop runtime-identity rejection tests also pass.
Probe/test Python open audits report zero denied scientific accesses, zero
benchmark images decoded and no TEST scientific data access.

Future training assessment: **E03_RUNTIME_IS_CANDIDATE_FOR_E04_TRAINING**.
The existing gpat-m6-e03 NVIDIA TensorFlow **1.15.5+nv22.12** supplies the
`tf.train.AdamOptimizer` and exponential-decay APIs used by the frozen A4 hook.
There is no actual E04 training graph in this adapter, so the candidate is not
qualified for E04 training. Its Python 3.8 TF1 process and this Python 3.12
geometry process require explicit future integration. No optimizer step ran.

Input 256, batch 8, budget 150000, LR 5e-5, /10 every 45000, initialization
normal [0,0.02], alpha1..6 = [100,5,1,1e-4,10,1], depth K=32 and terminal
iteration_150000 remain unchanged. Frozen config, snapshot, A3, A4 and Q140
bytes are verified against the authoritative commit.

## Freeze, ledger and reproduction

All five requested environment files are present. Lock SHA256:
`9426be37044ae7121fb22a757e5483850e02bf68461b1eabb5421fae164fbc3d`.
The lock records source, assets, full package artifacts, compiler/ABI, extension,
CUDA/cuDNN, GPU/driver and qualification conditions.

Example GPU synthetic-only replay:

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export GPAT_E04_GEOMETRY_RUNTIME=/home/student20261/workdir/GPAT_TransferBench/environments/e04_geometry.runtime.json
/home/student20261/miniconda3/envs/gpat-m6-e04-geometry/bin/python tools/m6d3a_e04_geometry_preflight.py --qualify --device cuda
```

Ledger: **100 → 101**, exactly one M6D3A_E04_GEOMETRY_RUNTIME_QUALIFICATION
record; first 100 rows byte-identical (256699 bytes; prefix SHA256
`d975323203172b6f21e4075286cf5a577a0c5fc8f86de6610fded028fd441b21`). Artifact index is rebuilt **LAST** with
`python tools/build_artifact_index.py`, retaining CRLF. Final verification is
read-only and checks hashes, source integrity, ledger prefix and index coverage.

Source integrity checks hash upstream example files without decoding them.
The required artifact index hashes tracked project/manifest bytes; it does not
execute scientific records. Probe/test auditing covers Python opens; native
inference receives only generated arrays. No broader operating-system tracing
claim is made.

E04_GEOMETRY_RUNTIME_QUALIFIED. E04_TRAINING_RUNTIME_NOT_YET_QUALIFIED.
E04 REMAINS IMPLEMENTED_NOT_EXECUTED. CONTROLLED_ADAPTATION DISCLOSURE PRESERVED.
HISTORICAL PACKAGE VERSIONS NOT MANDATORY. PINNED SOURCE CACHE UNMODIFIED.
NO BENCHMARK IMAGE DECODED. NO TEST SCIENTIFIC DATA ACCESSED.
NO TRAIN/VAL/TEST SCIENTIFIC EXECUTION. NO PHY-STD TRAINING. NO E04 OPTIMIZER STEP.
NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK. NO Q140 REDERIVATION.
NO FROZEN-CONFIG MODIFICATION. NO COMMIT. NO PUSH.
