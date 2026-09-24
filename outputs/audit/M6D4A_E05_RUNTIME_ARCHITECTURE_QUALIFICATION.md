# M6D4a — E05 runtime and architecture qualification

**E05_ARCHITECTURE_RUNTIME_QUALIFIED**. **E05_TRAINING_RUNNER_NOT_YET_QUALIFIED**.
E05 remains **IMPLEMENTED_NOT_EXECUTED**, **CONTROLLED_ADAPTATION**.
Reporting label: PCGAN (controlled architecture resolution).
Architecture/runtime only; PCGAN five-loss runner NOT YET QUALIFIED; benchmark training NOT STARTED.

## Authority and GPU synchronization

Laptop HEAD/origin `05e90efa6c36e11405822c79fed8511c2179f043`, initially clean, divergence 0 0.
GPU started at `90263169f0731b397dfa459d7ac94eaafd2c69ec` with 19 E04 paths.
Owner-authorized exact-path cleanup preserved and verified all 19 under
`/home/student20261/workdir/GPAT_TransferBench_runtime/recovery/M6D3e_pre_sync_20260924T082949_365687Z`.
Clean-worktree gate passed before fetch. Ancestry proved exactly one commit;
`git merge --ff-only origin/m6-baselines` reached authority, clean, divergence 0 0.
All 19 pre-cleanup, archived and post-merge file SHA256 identities match.
No stash, reset, clean, force checkout, commit or push. Recovery is engineering evidence only.

## Source and compatibility

Architecture basis: pinned Swapping Autoencoder, `taesungp/swapping-autoencoder-pytorch`
at `6baa180f1184ee79a6b967f9d80ee0e02a979ac7`. This is an executable architecture basis,
not an official PCGAN implementation. No StyleGAN-v1 AdaIN substitution.
Source commit, remote, tree, 23 required/cited files,
and 67 code-file Git blobs/SHA256 identities verified before/after.
The source was absent on GPU and acquired at the exact pin. Source cache remained clean and unchanged after acquisition.

Only compatibility change: generated-copy `util/util.py` CUDA predicate changes
`major >= 10 and minor >= 1` to tuple comparison `(major, minor) >= (10, 1)`.
CUDA 13.0 otherwise incorrectly selects the upstream fallback. This enables the real
pinned kernels; no tensor arithmetic, normalization, modulation, demodulation or
sampling changes. Original SHA256 `1877a389417a0c6f746006da852d09b3109ecfed2b119d5dc692519aa848b3eb`;
generated SHA256 `bd5346192b482d20c00233252b0fdcc3aadbf8d34f901fbb9af258ee80d50b6b`.
Exact diff and complete copy closure are in `environments/e05.compatibility_patch_manifest.json`.
C++ wrappers and CUDA kernels compile unchanged. `new_demodulation` and ToRGB
`demodulate=False` remain pinned. No source-cache patch or pretrained weights.

## Runtime and build

Created isolated **gpat-m6-e05** from gpat-m5; no existing environment reused as the final runtime.
gpat-m5 remains unchanged by before/after Python/framework and all 45 package-version checks.
Added source/import dependencies and CUDA build tools only to gpat-m6-e05.
Python 3.11.16; PyTorch 2.12.1+cu130; torchvision 0.27.1+cu130; CUDA 13.0;
cuDNN 9.20.0; nvcc 13.0.88; C++ Ubuntu GCC 13.3.0; RTX 3090; driver 595.84.
Historical Python 3.6/PyTorch 1.7.1 are provenance, not execution pins.
FP32; TF32, AMP and autocast disabled; cuDNN benchmark off, deterministic flag on.
Pinned stochastic noise remains enabled; qualification-only seed 60401.

All generated source/build artifacts remain under runtime `builds/e05_pcgan`.
`TORCH_EXTENSIONS_DIR` points to its `torch_extensions` subdirectory.
Build command, Ninja commands, input hashes, extension paths/hashes, controls and
complete package freezes are recorded in the environment artifacts.
Both extensions import and execute on real CUDA tensors in both accepted processes:

| Extension | SHA256 |
|---|---|
| fused_bias_act (`fused.so`) | `5afdb2369817ffea68530b1e88d75596a77ed08a397f83122b618a18225718ab` |
| upfirdn2d (`upfirdn2d.so`) | `40ba9db73e1790cd3d904a1fed09c5441180c6995a9a396030cfaa52efd1d081` |

Forward outputs and backward gradients are finite; independent Torch references pass.
Environment lock SHA256: `849a100430e36048acde80d58801ae8908c15ae43719e97f4dcb90d77564ac50`.

## Measured architecture and tensor paths

| Pinned class | Trainable parameters |
|---|---:|
| StyleGAN2ResnetEncoder | 929,960 |
| StyleGAN2ResnetGenerator | 3,699,233 |
| StyleGAN2Discriminator | 28,859,521 |
| StyleGAN2PatchDiscriminator | 24,522,145 |

Every exact inherited option, named parameter shape, nontrainable buffer and module
tree is retained in each process JSON. No new widths/defaults were invented.
Source and target: independent analytic FP32 `[1,3,256,256]` inputs.
Encoder spatial `[1,8,128,128]`, measured global `[1,2048]`.
Both reconstructions and `G(spatial_src, global_tgt)` return `[1,3,256,256]`.
Image-D real/reconstruction/mixed outputs are `[1,1]`.
Pinned random crops `[1,8,3,128,128]`, features `[8,384,2,2]`, PatchD scores `[8,1]`.
The active upstream crop/extract/discriminate interface is exercised, with inherited
scale range, eight crops and reference aggregation. The unused inherited forward
references an absent `sample_patches`; it is not the upstream active path.

A5 independently pools target/mixed `[1,3,256,256]` to `[1,3,128,128]` using the
exact frozen avg_pool2d arguments. Direct 2x2-block means agree. Target sum-gradient
is exactly 0.25; generated blur connects to generator parameters, both code inputs
and source/target input paths. No detach/fallback. All required model probes have
finite backward gradients; parameter hashes are identical before/after.

Scalar means/squares are diagnostic probes only. None of the five PCGAN losses,
new reductions, upstream training loop, Adam construction/step or final-state
checkpoint was implemented or executed. No scientific interface choice was needed.
Frozen batch 1, 4000 iterations, lr 1e-6, betas (0.9,0.999), five unit weights,
alpha/beta PMN scope, seeds, iteration_4000 and BASELINE_FINAL_STATE_V1 are preserved.

## Repeatability and validation

Two final clean GPU processes pass. Source/options/classes/parameter counts,
initial parameter hashes and extension identities match. All forward diagnostics
are bitwise equal. 173 of 408 forward/gradient tensor diagnostics have
different hashes, all in backward probes. Largest differences between scalar
summary statistics: min 9.31322575e-10,
max 1.86264515e-09,
mean 3.49245965e-10,
L2 6.98933099e-10.
These are not elementwise error bounds. Full per-tensor evidence is retained.
Native CUDA accumulation in generator bilinear-interpolation backward and patch
crop grid_sample backward are plausible sources; exact kernels were not profiled.
No scientific operation was changed to force determinism.

Two exploratory full synthetic attempts reached the final environment check and
were rejected because setuptools changes sys.path metadata visibility at import.
The final harness inventories fixed installed-package directories. No packages
changed during execution. Including exploration, four model diagnostic processes
ran; two final accepted processes establish qualification. Build/attempt logs retained.

**GPU: 65/65 pass**, zero skips. **Laptop: 64 pass, 1 skip** (pre-existing real
Torch pooling test; Torch absent). Includes all 43 existing E05/A5 tests and 22 new
focused runtime/evidence/guard tests. Data firewall recorded zero denied real I/O
attempts in both qualification and test processes. Explicit firewall rejection
unit tests use simulated events and do not access prohibited paths.
Full repository suite not run. Firewall covers Python file opens/enumeration;
compiler child I/O is constrained by explicit source/build commands, not claimed
as Python-hook coverage.

## Finalization and scope

Ledger **105 -> 106**, exactly one `M6D4A_E05_RUNTIME_ARCHITECTURE_QUALIFICATION`
record; first 105 rows byte-identical (prefix SHA256
`10d52a4eb584d0dc200fe0e890794073861e484971f8131e023b9b5f13e6540f`).
Artifact index rebuilt last from committed metadata plus 14 explicit new
artifacts: **544 rows**, CRLF preserved; no historical manifest or data targets opened.
Source/config/historical tests unchanged. All milestone changes remain uncommitted.

**NO BENCHMARK DATA ACCESS. NO BENCHMARK IMAGE DECODE/FILENAME ENUMERATION.
NO TRAIN/VAL/TEST SAMPLE ACCESS. NO E05 BENCHMARK TRAINING. NO OPTIMIZER STEP.
NO CHECKPOINT. NO SYNTHETIC BANK. NO PRETRAINED MODEL WEIGHTS. NO COMMIT. NO PUSH.**
