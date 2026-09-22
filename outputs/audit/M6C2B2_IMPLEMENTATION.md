# M6C2b2 — E05 PCGAN implementation

**PCGAN (controlled architecture resolution)** — `CONTROLLED_ADAPTATION`, provenance `AMENDMENT_A2_A2_07_ARCHITECTURE_RESOLUTION`. Status: **IMPLEMENTED_NOT_EXECUTED**.

Starting HEAD `2b38b3a6d97afbac8e79c844bdf3e2c1880495bf`, branch `m6-baselines`, clean preflight.

## Implementation and authority

Implemented source/contract/architecture mappings, symbolic five-loss and Adam mappings, three seed plans, lazy A5 pooling hook, and iteration-based checkpoint metadata support in the existing common learned runtime. This milestone does not construct a model or supply/execute a complete training runner. No official or faithful PCGAN reproduction is claimed.

Base E05 config SHA256: `478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de`. Byte-identical frozen snapshot verified.
A5 overlay SHA256: `af28a48b43828652a7423235aba6be239325e3fbfeeaf8fded9bbb57c11a7ca1`; bound to committed authority `2b38b3a6d97afbac8e79c844bdf3e2c1880495bf`. A2/A5 document hashes are included in each plan. No frozen input or existing amendment changed.

## Pinned executable architecture

Repository `https://github.com/taesungp/swapping-autoencoder-pytorch`; commit `6baa180f1184ee79a6b967f9d80ee0e02a979ac7`; tree `25a434f1282a0177d6afdcc68ebfcf40bd805621`. Repository/commit/tree and all required/cited file bytes verified fail-closed. No pretrained weights loaded or copied.

Encoder `StyleGAN2ResnetEncoder` uses frozen sp downsampling=1 and spatial channels=8: input 256 → z_pat 8×128×128. `StyleGAN2ResnetGenerator` returns full 3×256×256 RGB. Inherited defaults are read from pinned source AST, with provenance, including global_code_ch=2048. This follows A2 executable architecture, not every prose detail.

`stylegan2_layers.py:ModulatedConv2d.__init__/forward` supplies EqualLinear modulation and reciprocal-square-root demodulation, including its pinned new_demodulation path; `StyledConv` connects it to the generator. `ToRGB` preserves demodulate=False. The image discriminator is `discriminator.py:StyleGAN2Discriminator`, wrapping `stylegan2_layers.py:Discriminator`; the patch discriminator is `StyleGAN2PatchDiscriminator`. Exact file hashes/symbol lines appear in JSON and plans. StyleGAN-v1 AdaIN is not substituted.

## A5 blur, losses and optimization

`torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)` maps N×C×256×256 → N×C×128×128. The SAME operator applies independently to target and mixed generated tensors; no detach or fallback. Provenance: `BENCHMARK_DEFINED_CONTROLLED_RECONSTRUCTION`.

The five unit-weighted terms remain `L_rec + L_recblur + L_advrec + L_advmix + L_pat`. Symbolic mappings preserve paper Eq.1–5, with A5 supplying blur only. Alpha=0.2 and beta=1e-6 are retained as frozen metadata; paper Eq.10 assigns them to PMN, so they do not reweight PCGAN Eq.5. No numerical loss reduction is implemented or changed.

Adam maps frozen lr=1e-6, betas=(0.9,0.999). Batch=1; budget=4000 iterations; seeds=42,1337,2026. The upstream architecture is reused, but its even-image-batch training swap, L1 objective and lazy R1 lr/beta scaling are not launched as PCGAN. Source/target inputs and frozen loss integration belong to the future runner.

Supervision remains pooled TRAIN live/attack labels. Attack-type labels and GPAT identity/landmark losses are prohibited. VAL is diagnostic only. TEST never trains/selects. Every seed has its own authoritative iteration-4000 state under BASELINE_FINAL_STATE_V1; terminal save adds zero optimizer steps. No best-seed selection.

## Validation and environment

E05: **29 pass, 1 skip (30 tests)**. Real Torch CPU pooling/autograd test skipped precisely because PyTorch is absent; static two-branch dispatch and no-detach behavior verified. A5 mathematical NumPy regression: **13/13 pass**. No fake Torch/model execution is reported.
M6B: **PASS, 0 failures**. E04: **49 pass, 1 skip** (Sim3DR Cython). M6A6: **4/4**. M6C2a: **72/72**. M6C1: **83/83**.
M0: **16/17 pass**; sole historical failure `test_manifests_are_stage_appropriate` on `manifests/artifact_probe_classes_v1.json`. No historical test was changed.

Static preflight: **PASS** for all three seeds. File-open auditing: E05 focused tests **0 benchmark-image opens, 0 manifest opens**; E05 preflight **0 benchmark-image opens, 0 manifest opens**. Counts apply to those processes. Historical M0 examines metadata, not method execution or benchmark evidence. No full suite run.

Observed laptop: Python 3.12.3, NumPy 2.5.3, Pillow 12.3.0, OpenCV headless 5.0.0.93; PyTorch, torchvision and SciPy absent. No dependencies installed or GPU probed. These are not final execution pins. Future execution requires compatible framework/custom-op environment and training-runner integration.

## Files and integrity

- `methods/pcgan/__init__.py` — `efe5e8e011798e47a04e72305f80e787409722f392ccb47e8454e243d007552e`
- `methods/pcgan/adapter.py` — `7a4119fb1dac33aaee6d5e454f88dd7212b121808da3500c3c6d044de500fb44`
- `methods/pcgan/architecture.py` — `d0da76078283c82549250b7d0c8f02cc6057f7e091f4095cacc1cc7c02551826`
- `methods/pcgan/blur.py` — `ec16f8620534d7908ba23bb459ffdf5844d9d85a0ed8f6ad8668b094c67cdde9`
- `methods/pcgan/contract.py` — `2578d6e41bd9c86800a553ee57d8c469efa7f401e7f52bb1dcfa41200a9a64fe`
- `methods/pcgan/source.py` — `9c46f8869ce87124a9ecbcf5d03717b9cfc0f872b2ffb2b1c14822f36d9017b3`
- `methods/pcgan/source_traceability.md` — `61645843f65a51e7858787cfbc7cf30f1ff75763c8ded6da41d96410d4c5c1bd`
- `tests/test_m6c2b2_pcgan.py` — `8f34f784069f60f732f30585c3c4de55e27cd1f136cfaedfce21c21838d00316`
- `tests/test_m6c2b2_pcgan_architecture.py` — `c7f2939a8782bb2f50d6f3667b1a3aa16090d8e91f7b5b01c3877d72c6358936`
- `tools/m6c2b2_preflight.py` — `7279754b195c8535bcd5d8a97f805547f678f7cbcfa5aa9e0eacdd9b80d3624f`
- `methods/common/learned.py` — `0634752ab0b1abbf90faccf82f19db974263ce1f6a74f16eee5f787d26a47602`
- `configs/CONFIG_STATUS.md` — `a84cd8afe3e8030e121f403534b0eada98b0763d079db77c104516df6550e25e`

Existing historical audit artifacts remain unchanged. Append exactly one schema-complete M6C2B2_E05_IMPLEMENTATION record: 91 committed rows → 92. First 91 rows remain byte-identical.
Committed ledger prefix SHA256: `349c2f85124e57cb78bb8d51dc60cb07908be17ced6c8cabf071dbb0198131d7`.
Canonical artifact index includes unique current paths/sizes/SHA256 and CRLF. Audit/ledger/index final hashes are reported externally to avoid self-reference.

NO TRAINING; NO BENCHMARK IMAGE EXECUTION; NO SYNTHETIC BANK GENERATION; NO TEST DATA ACCESS FOR METHOD EXECUTION; NO GPU JOB; NO SCIENTIFIC CHECKPOINT; NO FROZEN CONFIG CHANGE; NO E07c IMPLEMENTATION; NO COMMIT; NO PUSH.
