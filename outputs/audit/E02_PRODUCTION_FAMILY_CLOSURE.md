# E02 production family closure

Status: `E02_PRODUCTION_FAMILY_COMPLETED_AND_AUDITED`

Storage: `/home/student20261/workdir/GPAT_TransferBench_runtime` on /dev/nvme0n1p3 (ext4 UUID `227ce2f6-373e-4f13-9cdb-84b911775706`, mount `/`)
Launch commit `0ca57e3bcd77e1a0c54004c7fd4c21e5fe101822` (candidate `d995f556a8adcc5a032ba266a022b470117d1c1c`), config `fdf74776cc100d5b5c1ed5f58b84ebb0dfd534782fbb7fc34382919d38d67bf0`

| seed | run_id | outputs | bank manifest SHA256 | audit |
|---|---|---|---|---|
| 42 | `77dce92a0f1b6751` | 8838 | `0b0aa41d714316601f0d9e2120a7d107f5a58d414a9475ab9f0e83feac284d1b` | 55/55 |
| 1337 | `47663d317a0046f9` | 8838 | `cd34bc02e94ddcb7ef6097260d1d22d1492b9b5b0a3afab9a46e5557e8628cd8` | 55/55 |
| 2026 | `8f698e0a2e72ae85` | 8838 | `9eb09bae542d37b769da25e1216031c7bcd4b365983ec4699532d6304c6fa485` | 55/55 |

Disclosures:
- Spec 8.2 frequency substitution: source-spoof amplitude in 40 of 159 eligible 16x16 blocks (0.20 <= r <= 0.50, >= 75 % coverage) with pixel-wise conjugate closure, target-live phase preserved, PCG64 selection from the A2-04 namespace seed.
- uint8 quantization is the implementation-defined clip [0,255] -> numpy.rint (owner ruling Q3), not an explicit spec instruction. Every output was recomputed with an independent float64 FFT and was identical.
- Edit maps (L PNG, 255 = edited) record the applied selection plus conjugates for every output.
- Pixel identity is shown for the gpat-m5 stack only (and the laptop in M6). It is not an unrestricted cross-stack guarantee.

Remaining:
- M8+ downstream evaluation of these banks (ResNet-18 per the approved evaluator contract; DINOv3 needs its frozen config and pinned weights): separate owner authorization required
- TEST evaluation: not authorized; no TEST row or TEST face was read
- outputs/audit/method_status.csv is unchanged; updating it needs a separately verified registry-authority decision
- learned-method banks (E03-E07) and GPAT are outside this work
