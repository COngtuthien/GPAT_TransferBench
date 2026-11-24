# M8 learned-bank S7 toy acceptance: E05

**Status:** `S7_ACCEPTANCE_PASSED`, 145/145 gates passing; `s7_acceptance_passed` = true.

Label: S7_TOY (S7_NON_SCIENTIFIC toy acceptance; never a production bank or benchmark evidence).

Authority `configs/amendments/learned_bank_s7_launch_authority.yaml` (`0be8f0e14578…`), contract `3554577185a9…`, E04 seed-2026 record `47a4e4974d86…`, execution commit `7ff4d36f74cd64d6c405944d3b9fa4aee1fae521`.

| seed | run A | run B | A/B bytes | manifest sha256 |
|---|---|---|---|---|
| 42 | PASS | PASS | identical | `8c622e2a0d037e41…` |
| 1337 | PASS | PASS | identical | `6ad2de476fb3daba…` |
| 2026 | PASS | PASS | identical | `522d6fd1ad83f017…` |

Failed gates: none.

## Execution

- Environment `gpat-m6-e05` (`/home/student20261/miniconda3/envs/gpat-m6-e05/bin/python`), framework torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04.
- One OS process per seed and run (owner ruling B2, P2), strictly sequential: seed 42 run A 2026-11-24T19:27:38Z → 2026-11-24T19:27:48Z exit 0; seed 42 run B 2026-11-24T19:28:05Z → 2026-11-24T19:28:13Z exit 0; seed 1337 run A 2026-11-24T19:28:32Z → 2026-11-24T19:28:40Z exit 0; seed 1337 run B 2026-11-24T19:28:49Z → 2026-11-24T19:28:57Z exit 0; seed 2026 run A 2026-11-24T19:29:16Z → 2026-11-24T19:29:25Z exit 0; seed 2026 run B 2026-11-24T19:29:33Z → 2026-11-24T19:29:42Z exit 0.
- Population: 186 toy pairs of outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json (sha256 50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690) at their canonical global pairs_train_v1 positions; faces TRAIN-only via TrainFaceReader (O_NOATIME, M2 sha256).
- RNG: E05: torch.manual_seed(e05_pair_seed(seed, pair_id)) before each mix forward.
- Conversion: `util.tensor2im(mix[0]): clip((x+1)/2*255, 0, 255).astype(uint8) [swapping_autoencoder util/util.py:125-131]`.
- Output root (GPU host): `/home/student20261/workdir/m8_learned_banks/s7/7ff4d36f74cd64d6c405944d3b9fa4aee1fae521/E05`.
- Peak CUDA memory per seed (bytes): {"A": {"42": 471932928, "1337": 471932928, "2026": 471932928}, "B": {"42": 471932928, "1337": 471932928, "2026": 471932928}}.
- atime ledger: pre 2026-11-24T19:27:36Z, post 2026-11-24T19:29:58Z; no TRAIN/VAL/TEST face atime or mtime changed.

## Scope

S7 establishes method-specific readiness for a separately authorized production launch only. No production bank, retraining, downstream training, TEST evaluation or GPAT work was performed.
