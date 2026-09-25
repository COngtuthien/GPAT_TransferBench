# M6D5a — E06c DSDG-BIN-IDFREE runtime environment and executable architecture qualification

**E06c_RUNTIME_ENVIRONMENT_QUALIFIED**. **E06c_ARCHITECTURE_RUNTIME_QUALIFIED**.
**E06c_TRAINING_GRAPH_NOT_YET_QUALIFIED**. **E06c_PHYSICAL_BATCH_240_NOT_YET_QUALIFIED**.
E06c remains **IMPLEMENTED_NOT_EXECUTED**, fidelity **CONTROLLED_ADAPTATION**
(Amendment A1 identity-free adaptation, DEV-020). Reporting label:
DSDG-BIN-IDFREE (Amendment A1 identity-free controlled adaptation).
This is runtime/architecture evidence only; no E06c training of any kind has occurred.

## Authority and GPU synchronization

Laptop HEAD = origin/m6-baselines = `13fc86c15ecb9303e4ec678ab3f0f55ff61a51d8`
("M6D4e: cleanly requalify E05 training runner"), divergence 0 0, clean worktree.
`/home/cong/GPAT_TransferBench/AGENTS.md` tested with `test -f`: absent; no other instruction discovery.

GPU started at `32c8f979e14d1ea37ee2047cbd5489cb83c64b72` = its origin, divergence 0 0, clean.
Fetched only `refs/heads/m6-baselines`; ancestry proved (distance 1);
`git merge --ff-only origin/m6-baselines` reached authority, divergence 0 0, clean.
No reset, clean, stash, force checkout, commit or push.

## Pinned source

`JDAI-CV/FaceX-Zoo@16b793a7564a4b9308cf94e62bdb2ffacb3a725a`, tree
`0d2216bdbdd9977130db1100641abf336f46ac9d`, `addition_module/DSDG`.
The source cache was absent on GPU. It was acquired at the exact pin in the laptop
cache's recorded mode (sparse `addition_module/DSDG`, `blob:none`, depth 1), and the
one recorded upstream weight `DUM/checkpoint/CDCN_U_P1.pkl` was removed from the worktree,
exactly as `third_party/source_pins.json` records. It was never read.
GPU and laptop status are identical (only that deletion). The 13-file closure
(12 required + cited `make_train_list.py`) matches pinned Git blobs and SHA256 on both hosts,
before and after both processes. `light_cnn.py` SHA256
`a66539b53cd9d8a081fd0a4e523b61aa39d9d0a42b2f68ad1146cfdd3e8eddba`;
`generator.py` `c8516e2a…d303`; `misc/util.py` `64aa2d09…41c7`.
`third_party/source_pins.json` and the source cache were not modified.

## Runtime environment

New isolated **gpat-m6-e06c** (`/home/student20261/miniconda3/envs/gpat-m6-e06c`),
cloned offline from gpat-m5 with **no packages added**. gpat-m5 pip-freeze, conda-explicit
and interpreter hashes are identical before and after. gpat-m6-e03, gpat-m6-e04-geometry,
gpat-m6-e05, stdn and base were not touched.

| Item | Value |
|---|---|
| Python | 3.11.16 |
| PyTorch / torchvision | 2.12.1+cu130 / 0.27.1+cu130 |
| CUDA runtime / cuDNN | 13.0 / 9.20.0 (92000) |
| NumPy / Pillow | 2.4.6 / 12.3.0 |
| GPU / driver | NVIDIA GeForce RTX 3090 (sm_86) / 595.84 |
| Compiler | not used (no extension build) |

Historical upstream Python 3.8 / PyTorch 1.6.0 / torchvision 0.7.0 are provenance, not pins:
PyTorch 1.6 CUDA 10.x builds ship no sm_86 kernels. DSDG uses only standard `torch.nn`
operators, so the modern stack already qualified for gpat-m5 is the minimal compatible runtime.
FP32; TF32/autocast off; cuDNN benchmark off and deterministic on (frozen M6C2a seed plan).
Environment lock SHA256: `91416a20fef6eb4bbe550dc0ccdc703163f51d8df9168c1418f7a2de48e64e95`.

**compatibility_patch = NONE.** No generated source copy and no shim. Pinned `networks` and
`misc.util` are imported unchanged via `methods.common.upstream.upstream_modules`.
Engineering observations with no semantic effect:

- `reparameterize` still uses `torch.cuda.FloatTensor(std.size()).normal_()`. It executes
  unchanged and only emits a deprecation `UserWarning`. The N(0,1) distribution and the global
  CUDA generator are unchanged.
- The upstream `F.interpolate(..., mode='bilinear')` passes no `align_corners`. Its default is
  `False` in both 1.6 and 2.12, so the call is identical.
- The upstream LightCNN loader sits inside `main()` (train_generator.py:75-84). The harness
  mirrors those statements exactly: filter `k in model_dict`, update, `load_state_dict`, then
  `requires_grad=False`. PyTorch ≥2.6 defaults `torch.load` to `weights_only=True`; the harness
  passes it explicitly. The file needs only `OrderedDict`/`FloatStorage`/`_rebuild_tensor`, and
  the loaded values equal the checkpoint tensors (`torch.equal`).

## LightCNN-29 v2 official checkpoint

| Host | Path | Bytes | SHA256 |
|---|---|---:|---|
| Laptop | `/media/cong/Data/GPAT_TransferBench_runtime/third_party_weights/lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar` | 123844849 | `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964` |
| GPU | `/home/student20261/workdir/GPAT_TransferBench_runtime/third_party_weights/lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar` | 123844849 | `d0750746622270548137b092700811b00dc64d67ca8df042c6cf48cb3b1b9964` |

The file was absent on GPU. The laptop source was verified first, then only this file was
copied and re-verified. It is not committed, not in source_cache, and was not re-downloaded.
Live runtime compatibility: checkpoint tensors **61**; `define_IP(is_train=False)` model tensors **60**;
matched **60**; missing **0**; shape mismatches **0**; filtered extra key
**`module.fc2.weight` [80013, 256]**. This reproduces M6A4 exactly.
Checkpoint metadata: arch `LightCNN`, epoch 50, prec1 99.46.

## Exact construction and live parameter counts

`networks.define_G(hdim=128, attack_type=1)` and `networks.define_IP(is_train=False)`,
each wrapped by the upstream `torch.nn.DataParallel(...).cuda()` (device_ids `[0]`).
Class objects are resolved from the verified source root.

| Component | Pinned class | Parameters | Tensors |
|---|---|---:|---:|
| netE_nir | Encoder_s(128) | 13,790,048 | 17 |
| netE_vis | Encoder(128) | 11,692,640 | 17 |
| netG | Decoder_s(128) | 19,887,494 | 21 |
| netCls | Cls(128, 1) | 129 | 2 |
| **Optimizer-owned (netE_nir+netE_vis+netG)** | | **45,370,182** | **55** |
| netIP | network_29layers_v2(is_train=False) | 10,475,872 | 60 |

netCls is excluded from optimizer ownership (train_generator.py:87-88). All netIP parameters
have `requires_grad=False` and netIP is in eval mode. netE_nir/netE_vis/netG are in train mode,
as at train_generator.py:105-108. All counts match the contract; no architecture was edited.
The optimizer contract (Adam, lr 2e-4, scope netE_nir+netE_vis+netG) was inspected statically only.

## Synthetic batch-one forward — DIAGNOSTIC_ARCHITECTURE_BATCH_ONLY

Distinct deterministic analytic FP32 in-memory inputs in [0,1]. No file, decoder or benchmark data.
The forward ran under `torch.no_grad()`.

| Tensor | Shape |
|---|---|
| x_spoof, x_live | [1,3,256,256] |
| Encoder_s → mu_nir, logvar_nir, mu_a, logvar_a | [1,128] each |
| Encoder → mu_vis, logvar_vis | [1,128] each |
| pinned `reparameterize` → z_cls, z_nir, z_vis | [1,128] each |
| netCls(z_cls) → pre_spoof | [1,1] |
| cat(z_cls, z_nir, z_vis) | [1,384] |
| netG → rec | [1,6,256,256] |
| rec_spoof = rec[:,0:3], rec_live = rec[:,3:6] | [1,3,256,256] each |
| bilinear 128 (x_spoof, x_live, rec_spoof, rec_live) | [1,3,128,128] each |
| pinned `rgb2gray` → LightCNN inputs | [1,1,128,128] each |
| netIP features | [1,256] each |

All 28 recorded tensors are finite FP32 CUDA tensors. Single-logit classifier:
`CrossEntropyLoss(pre_spoof, index 0) = 0.0`, recorded as **EXPECTED_DEGENERACY_CONFIRMED**.
It is kept as one logit, per A1.

## Diagnostic CUDA memory — DIAGNOSTIC_BATCH1_ONLY

Allocated 267,419,648 B; reserved 369,098,752 B; peak allocated 353,616,896 B;
peak reserved 369,098,752 B. Peaks cover model construction, LightCNN loading and a no-grad forward.
**This does NOT establish physical-batch-240 feasibility.** No batch-240 attempt was made
and nothing was extrapolated. `E06C_REQUIRES_PHYSICAL_BATCH_240_FOR_OBJECTIVE_EQUIVALENCE` is
retained. The adapter still refuses 120×2, 60×4 and 1×240. Batch 240 belongs to M6D5b.

## Repeatability

Two fresh final GPU processes (`PYTHONHASHSEED`/diagnostic seed 60501, qualification only;
not experiment seed 42/1337/2026) both PASS. Their JSON evidence is byte-identical
(SHA256 `05de38af9561dd065311acaa80933127af734fbd5c4f851d32e94093d4e9874a`).
Initial parameter hashes, all 28 forward tensors and the reparameterized latents are bitwise equal.
VAE reparameterization is stochastic; under the fixed seed, both processes drew identical ε
from the unchanged global CUDA generator. Equality was observed, not forced.
Memory values are also equal.

Attempts before the final pair: **attempt 1 stopped** at import. The harness's own write firewall
refused torchvision's temp-dir probe under `/tmp`. Source verification and the LightCNN
byte hash had already run; no model was constructed and nothing was deserialized. The harness
then had SHA256 `0090d7da…120c`. The fix is
engineering-only: `TMPDIR` now points inside the dedicated build root, with no `/tmp` exception.
Attempt 2 passed with the same harness bytes used for the final pair. There were four harness
launches in total (attempt 1, attempt 2, process 1, process 2), all logged. Before the harness ran, a separate disclosed
API probe checked only the legacy CUDA tensor constructor and bilinear interpolation.

## Tests and preflight

Focused `tests/test_m6d5a_e06c_runtime.py`: **22/22 PASS on GPU** (gpat-m6-e06c, pytest) and
**22/22 PASS on laptop** (unittest, no Torch). They cover frozen config and A1 identity,
source identity and immutability, contract values, class binding, live counts, optimizer
ownership, LightCNN bytes, SHA256, 60/60 and fc2 filtering, netIP eval/frozen, every interface
shape, finiteness, the one-logit degeneracy, zero optimizer/backward/checkpoint, batch 240 not
qualified, the data/TEST firewall and CONTROLLED_ADAPTATION wording.
`tools/m6d5a_e06c_runtime_preflight.py` is static: no Torch, CUDA, model or optimizer.
The full repository suite was not run. Existing tests were not modified.

## Finalization and scope

Ledger **110 → 111**, exactly one `M6D5A_E06C_RUNTIME_ARCHITECTURE_QUALIFICATION` row; the first
110 rows are byte-identical (prefix SHA256
`ee11822a9d0d45c0952ffc4af180772b897ce0c47825ffbb2ab6df42633dd434`).
Artifact index rebuilt last from committed metadata plus 12 explicit new artifacts:
**572 → 584 rows**, CRLF preserved. No historical target was opened.
The staged harness/test/evidence copies on the GPU worktree were removed after use;
the GPU worktree is clean at authority.

training_launched=false · optimizer_constructed=false · optimizer_applications=0 ·
backward_passes=0 · physical_batch_240_qualified=false · benchmark_data_access=false ·
TEST_access=false · checkpoint_created=false · synthetic_bank=false.

The E06c method is not described as trained, faithful, native or an official reproduction.

**NO RECURSIVE FILESYSTEM SEARCH. NO BENCHMARK DATA ACCESS. NO TEST ACCESS. NO BENCHMARK
TRAINING. NO SCIENTIFIC CHECKPOINT. NO SYNTHETIC BANK. NO COMMIT. NO PUSH.**
