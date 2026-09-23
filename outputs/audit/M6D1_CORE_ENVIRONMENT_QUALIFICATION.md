# M6D1 core environment qualification

Decision: **QUALIFIED_EXECUTION_ENVIRONMENT_FOR_E01_E02_ONLY**. E01/E02 remain **IMPLEMENTED_NOT_EXECUTED**. E03, E04, E05, E06c and E07c are not qualified.

The existing `/home/student20261/miniconda3/envs/gpat-m5` environment originally served M5 and pre-existed M6D1. M6D1 did not create or mutate it and installed no package. No scientific execution occurred.

## Starting state

Laptop and GPU: clean `m6-baselines` at `7387a25f62b50ad748edd1a69e21616af44501b9`. Laptop origin matches, divergence 0/0. Public-key SSH verified. No fetch, synchronization or source deployment performed.

Existing FAS-Aug pin `0da1dd79bad00e225b8cb6977c7f3f06ee8f7517`: all 470 tracked deployment files verified against Git blobs, recorded SHA256 for each, cited-file SHA256 checks passed, no extra/missing files. Assets: background 90, noiseTexture 48, MPTexture 190. Selected ICC profiles: RGB 11, CMYK 7. Final verification matches initial evidence.

## Direct focused tests

The prior incident is **NON_SCIENTIFIC_INVOCATION_ERROR_RESOLVED_BY_DIRECT_FILE_EXECUTION**. The user-provided resume record reports zero tests ran under failed absolute-directory discovery. This session independently reproduced successful direct execution; no test, import structure or source edits were made.

Working directory: `/home/student20261/workdir/GPAT_TransferBench`; `PYTHONDONTWRITEBYTECODE=1`.

```bash
PY=/home/student20261/miniconda3/envs/gpat-m5/bin/python
"$PY" tests/test_m6c1_e01.py -v
"$PY" tests/test_m6c1_e02.py -v
"$PY" tests/test_m6c1_runtime.py -v
```

| File | Run | Pass | Fail | Error | Skip |
|---|---:|---:|---:|---:|---:|
| tests/test_m6c1_e01.py | 26 | 26 | 0 | 0 | 0 |
| tests/test_m6c1_e02.py | 29 | 29 | 0 | 0 | 0 |
| tests/test_m6c1_runtime.py | 28 | 28 | 0 | 0 | 0 |
| Total | 83 | 83 | 0 | 0 | 0 |

Static review: synthetic inputs only; pinned upstream assets may be decoded. TEST rejection tests submit synthetic arguments and refuse them before scientific execution. Runtime checkpoint-index tests create temporary metadata only. Seven-method config loading does not qualify other environments or methods.

## Synthetic contracts and dry run

E01: known PTR000001/42 seed 795981663; all eight source-bound official operators on 256x256 RGB arrays at seeds 42, 1337 and 2026 passed 24 byte-identical repeats and 24 level-zero no-op checks.

E02: NumPy 2.4.6; n_eligible 159, k 40; pair seed `19354000309184887456743517179028239418116648094833629759175866349341962868907`; first ten selected indices `[2, 4, 16, 22, 25, 30, 33, 39, 42, 44]`. Six noise/smooth cases across three seeds passed Hermitian-mask symmetry, uint8 output and deterministic repeats. Maximum imaginary residual 5.301936406639086e-12 < 1e-8. This is the fresh-session measured maximum, not the prior-session value.

`"$PY" tools/m6c1_dry_run.py` (without --keep): ok=true; E01/E02 deterministic repeats=true; official eight-operator smoke=PASS; overwrite refusal=true; append/history preservation=true. Benchmark image opens=0; manifest opens=0; TEST split accessed=false. Temporary root `/tmp/m6c1-dryrun-4f47s6yk` confirmed removed. Production E01/E02 runtime paths absent before and after. Temporary run manifests are metadata, not benchmark manifests.

## Existing environment capture

Python 3.11.16; pip 26.2.1; NumPy 2.4.6; Pillow/ImageCms 12.3.0 (available); OpenCV 5.0.0; PyYAML 6.0.3. Required E01/E02 imports passed. Unrelated missing packages TensorFlow, SciPy, sklearn, tensorfn, Cython, onnxruntime and lpips do not affect qualification.

Inventory only: torch 2.12.1+cu130; torchvision 0.27.1+cu130; torch CUDA 13.0; cuDNN 92000; NVIDIA GeForce RTX 3090, compute capability 8.6, driver 595.84; system nvcc 12.0.140. No learned model was constructed or run.

Conda explicit uses actual `/home/student20261/miniconda3/bin/conda` (26.7.2) with the target prefix. Pip uses the target Python and `-m pip freeze --all`. Runtime JSON is exact GPU probe stdout. All three capture files preserve the received GPU stdout bytes. Conda and pip are complementary inventories; no reinstall was attempted. The JSON audit embeds the capture probe source and complete test output.

| Capture | SHA256 |
|---|---|
| environments/m6_core_gpu.conda-explicit.txt | `2c5a2bbcacf767c54eb2f21ce2648a8bd265dbb57f6d0e0cdc01839d8b42b864` |
| environments/m6_core_gpu.pip-freeze.txt | `527a5ec89fc9fb951130b7cc2a9a9b0bbf77c653a93b8aee35ff2c01cd27fec2` |
| environments/m6_core_gpu.runtime.json | `5da1b42bd5b5c3db4e43c865512507635cf9fad2c8b1bccc65f027c38132d25e` |
| environments/m6_core_gpu.lock.json | `53d4910a17c541ad7ac57a1ac410c0fef8c2d1c63399b423c97e490817b335eb` |

Future E01/E02 `environment_lock_sha256`: `53d4910a17c541ad7ac57a1ac410c0fef8c2d1c63399b423c97e490817b335eb`. No hash was written into frozen method configs. Lock qualification evidence is linked by a canonical JSON digest to avoid a circular file-hash dependency.

## Ledger, preflight and firewall

Append exactly one M6D1_CORE_ENVIRONMENT_QUALIFICATION record: 97 -> 98 rows. First 97 rows remain byte-identical; prefix SHA256 `a6cb6a8c6fef626cf4e8bd929b27d4f6634bc7829c77b511afe81dc70394820e`. Ledger stores hashes of finalized M6D1 artifacts. The artifact index is rebuilt last using the established builder and CRLF policy; its final SHA256 is reported by preflight.

Run `python tools/m6d1_core_environment_preflight.py --live-gpu` after rebuilding the index. It validates evidence and current identity without scientific execution, package operations or environment changes. Default mode validates recorded GPU evidence; --live-gpu additionally verifies current GPU identity, versions, pinned deployment bytes and absent production paths over public-key SSH.

Only the nine authorized paths may change. Configs, frozen snapshots, manifests, gpatbench, methods, docs/spec, tests and source_pins.json stay unchanged. The index builder hashes tracked manifest bytes opaquely; qualification and dry-run probes do not open benchmark manifests.

UNITTEST DISCOVERY ERROR WAS INVOCATION-ONLY. DIRECT TEST-FILE EXECUTION USED. GPAT-M5 WAS NOT MUTATED. NO PACKAGE INSTALLED. NO ENVIRONMENT CREATED. ONLY E01/E02 QUALIFIED. E03-E07c NOT QUALIFIED. NO BENCHMARK IMAGE DECODED. NO TEST EXECUTION (scientific TEST split). NO SCIENTIFIC BANK GENERATED. NO TRAINING. NO CHECKPOINT (scientific). NO COMMIT. NO PUSH.
