# E04: owner-resolution investigation (D1–D9)

**Status:** `E04_OWNER_RULING_READY_FOR_FINAL_APPROVAL`. The decision package is complete and **DRAFT**; nothing in it is approved. There is no E04 scientific authority and execution is not authorized. Draft addendum: `docs/spec/amendments/GPAT_TransferBench_v1_0_E04_Production_Owner_Resolution_Addendum_DRAFT.md`; draft companion: `configs/amendments/e04_production_resolution.DRAFT.yaml`.

## D2 TRAIN pre-pass population (manifest-only read)

- `pairs_train_v1` `a5e4fdaef236f157…`: 8838 rows, 8838 unique pair_ids, stored sorted by pair_id: True.
- 3830 unique live targets (split/label [['TRAIN', 0, 3830]]; reuse 1–23, mean 2.308) and 8838 unique spoof sources ([['TRAIN', 1, 8838]]); overlap 0. Landmarks in `e03_feed_pack_v1` cover 3830 live and 8838 spoof.
- The manifest carries `attack_macro`, which the E04 feed must never read.

## D3 CPU geometry repeatability (synthetic)

- 3 clean CPU processes are byte-identical (True). Fitted parameters, normalised input, mesh and depth equal the committed M6D3a values (recorded under driver 595.84): True. CUDA differs from CPU in the fit and mesh, but its values also equal the committed CUDA values.

## D4 / D7 RNG test vectors

- Identical across numpy 2.5.3 (laptop) / 1.23.5 (gpat-m6-e03) / 2.5.3 (gpat-m6-e04-geometry): True. The seed-314159 realization reproduces the M6D3e hard-sample fixture hashes: True.

| seed | pair seed | hard seed | pair stream sha256 | iteration 0 rows | hard stream sha256 | keep rate | straddling / repeated |
|---|---|---|---|---|---|---|---|
| 42 | 679869707 | 320493565 | `0dea19a0889cdf6c…` | [4937, 1260, 4990, 7755] | `ed4f32ed21e819f0…` | 0.50016 | 34 / 0 |
| 1337 | 1349475396 | 35554372 | `5c18cc0823127f18…` | [3963, 5112, 660, 2748] | `2cdc01ca64445e2a…` | 0.50019 | 34 / 0 |
| 2026 | 692890966 | 1053992267 | `dc0513240d641ce4…` | [733, 1023, 2899, 1486] | `563b431dc3d28925…` | 0.49986 | 34 / 0 |

## D9 BN investigation (synthetic, qualification seed 314159, 4 clean processes)

- Identical to history: GraphDef True, fixtures True, CUDA/cuDNN/TF True, BN update sets True, changed-trainable sets True, losses before every step True, final losses True.
- Step 1: all current processes and the readiness run are identical (0 varying tensors); they differ from history only in `D/D3/conv1/BN/moving_mean:0`, `D/D3/conv2/BN/moving_mean:0`, `D/D3/conv3/BN/moving_mean:0`, `D/D3/conv4/BN/moving_mean:0`. Historical processes 1 and 2 were identical at step 1. The difference is therefore deterministic within each runtime and differs between the runtimes.

The historical raw tensors were not retained, so the comparison with history uses their recorded min/max/mean/L2 summaries:

| tensor | shape | |max| | float32 ulp there | Δmax | Δmin | Δ in ulps | relative ΔL2 |
|---|---|---|---|---|---|---|---|
| `D/D3/conv1/BN/moving_mean:0` | [32] | 7.499e-04 | 5.82e-11 | 0.00e+00 | 0.00e+00 | 0 | -6.52e-10 |
| `D/D3/conv2/BN/moving_mean:0` | [64] | 2.544e-03 | 2.33e-10 | 0.00e+00 | 4.66e-10 | 2 | 4.80e-09 |
| `D/D3/conv3/BN/moving_mean:0` | [96] | 3.789e-03 | 2.33e-10 | 0.00e+00 | 0.00e+00 | 0 | 4.81e-09 |
| `D/D3/conv4/BN/moving_mean:0` | [128] | 4.782e-03 | 4.66e-10 | -4.66e-10 | 0.00e+00 | 1 | -8.78e-09 |

- Steps 2/3: history varied in 51 tensors (13 trainable, all D3). Every current process matches historical process 2 on 47/47/47/47 of them; the only current variation is the same 4 D3 moving means (2 variants, max |Δ| 4.66e-10, max relative 9.26e-08).
- The moving statistics are read only by `Assign`, `AssignSub`, `Identity`, `Sub`; FusedBatchNorm never reads them (training=True). They are write-only during training and cannot reach any loss, gradient or trainable variable.

Interpretation: last-bit float32 rounding in the D3 batch-mean reduction that feeds the moving average. The update semantics (ops, order, sets) are unchanged. The exact kernel is not identified (CUPTI unavailable), the same limitation M6D3e disclosed.

## Boundary

No benchmark image or landmark payload was read and no TEST was accessed. The pre-pass did not start, no experiment seed was executed, no environment or GPU checkout was modified, and no training was launched.
