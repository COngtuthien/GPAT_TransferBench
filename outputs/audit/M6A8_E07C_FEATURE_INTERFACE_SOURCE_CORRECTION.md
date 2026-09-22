# M6A8 — E07c feature-interface source correction

**OWNER-APPROVED · ADDITIVE · NON-DESTRUCTIVE**. Provenance: `SOURCE_GROUNDED_CONTRACT_CORRECTION`.

Starting HEAD: `4389578fa547d9589bb58bf8dff3221617288259`; branch `m6-baselines`; clean preflight.

M6C2b3 stopped before implementation because A3/M6B described feature channels as 128/256/512, but exact pinned source produces 256/512/512. The required source-preservation rule and the frozen descriptions could not both be obeyed.

| Output (H×W×C) | Superseded description | Corrected interface |
| --- | --- | --- |
| x32x32 | 32×32×128 | 32×32×256 |
| x16x16 | 16×16×256 | 16×16×512 |
| x8x8 | 8×8×512 | 8×8×512 |
| embg | B×K | B×7, unchanged A3 head contract |

Source: `https://github.com/murphytju/DiffFAS` at `23f40519ec25a833ebc06842aa6fbab74fad4d15`.
Independently recomputed custom_rn SHA256: `fa788c4d2453b40585b295a8292eaf1afce7a0c622eb8ecc2adec5453a9a64c3`. Both architecture and consumer files match pinned Git objects byte-for-byte.

Source evidence: `custom_rn.py:124` BasicBlock expansion=1; `ResNet.__init__` lines 267/268/269 sets stage channels 256/512/512; `_forward_impl` lines 329–346 returns layer2/3/4 directly. `resnet18` lines 367–369 preserves `_resnet("resnet18", BasicBlock, [3,4,6,3], ...)`. Stem/stage stride propagation confirms spatial sizes 32/16/8 at input 256, without framework execution.

`custom_rn.py:271` preserves head input 512×expansion=512. A3 head remains `nn.Linear(512,7)`. `unet_autoenc.py:78` returns four encoder outputs; line 178 discards the fourth, and lines 186–188 consume the first three. K affects only the discarded output dimension and does not explain/change consumed channels.

A6 restores consistency with A3’s intent to preserve the exact pinned architecture. The previous dimensions were a benchmark documentation/static-analysis error. This is not retuning, architecture redesign, improved DiffFAS, or an official author correction. No projection, fixed channel adapter, BasicBlock/stage modification or torchvision substitution is authorized.

Fidelity remains CONTROLLED_ADAPTATION for the existing A1/A3 reasons, with no new downgrade. All other contracts remain unchanged. E07c is CONTRACT_RESOLVED_NOT_IMPLEMENTED; M6C2b3 has not resumed.

## Contract identities

- `configs/methods/e07c_difffas_bin_idfree.yaml` — SHA256 `dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c`
- `frozen_config_snapshot/configs/methods/e07c_difffas_bin_idfree.yaml` — SHA256 `dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c`
- `configs/frozen/difffas_bin_idfree_v1.yaml` — SHA256 `aa9e984166db3854bba4221098afaef1898474e2e3f1f08a3f80cab0035cf3eb`
- `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A3_Controlled_Reconstruction_E04_E07c.md` — SHA256 `b12451537bcc3bc14e96e5bcd2ce390b5fc0c8a4b5c60bff7f40a2333665a67a`
- `docs/spec/amendments/GPAT_TransferBench_v1_0_Amendment_A6_E07c_Feature_Interface_Source_Correction.md` — SHA256 `759f72d860ccf49927c214bb4ce84e87bcedd98eb8f06ef5fe7361a758d6bc81`
- `configs/amendments/e07c_a6_feature_interface_source_correction.yaml` — SHA256 `dd3f29aa8ff96de9c0e2d504e6d07f4788d8fb8d030ffa26ef51443104ca971b`
- `tests/test_m6a8_e07c_feature_interface_contract.py` — SHA256 `14f690900fa6d4772d576fb8109f118f1b91eb0e275344b4bdb3ae060a47d919`

## Validation and integrity

**13/13 static tests PASS**, zero failures/errors/skips. M6B validator **PASS, 0 failures**. No full-suite run. No PyTorch dependency, model construction, inference or dependency installation.
Existing StaticAccessAudit on the focused test process reports **0 benchmark-image opens, 0 manifest opens, 0 rejected data opens**. Work consists of source/config/metadata reads, with no benchmark pixels.
Append exactly one schema-complete M6A8_E07C_FEATURE_INTERFACE_SOURCE_CORRECTION record: **92 committed rows → 93 total**, preserving the first 92 byte-for-byte.
Committed ledger prefix SHA256: `2e46e5e562a0bf526d6fa154f135581c94a5d94f5707a60f1a29d8f5487ee16d`.
Rebuild the artifact index canonically at finalization; verify unique paths/current sizes/hashes/CRLF. Final audit/ledger/index hashes are reported externally to avoid self-reference.

NO TRAINING; NO E07c IMPLEMENTATION; NO SOURCE MODIFICATION; NO CHANNEL PROJECTION ADAPTER; NO BENCHMARK IMAGE EXECUTION; NO DIFFUSION SAMPLING; NO SYNTHETIC BANK GENERATION; NO GPU JOB; NO SCIENTIFIC CHECKPOINT; NO FROZEN E07c CONFIG CHANGE; NO EXISTING AMENDMENT REWRITE; NO COMMIT; NO PUSH.
