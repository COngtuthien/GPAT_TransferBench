# M8 learned-bank S7 toy acceptance: E03

**Status:** `S7_ACCEPTANCE_PASSED`, 98/98 gates passing; `s7_acceptance_passed` = true.

Label: S7_TOY (S7_NON_SCIENTIFIC toy acceptance; never a production bank or benchmark evidence).

Authority `configs/amendments/learned_bank_s7_launch_authority.yaml` (`0be8f0e14578…`), contract `3554577185a9…`, E04 seed-2026 record `47a4e4974d86…`, execution commit `a5e31322fd99bb6c4b6361b8b6a96790bb0d8f18`.

| seed | run A | run B | A/B bytes | manifest sha256 |
|---|---|---|---|---|
| 42 | PASS | PASS | identical | `5522f105975a2e95…` |
| 1337 | PASS | PASS | identical | `39a10b84e9fa3977…` |
| 2026 | PASS | PASS | identical | `cd4568a4189e69eb…` |

Failed gates: none.

## Execution

- Environment `gpat-m6-e03` (`/home/student20261/miniconda3/envs/gpat-m6-e03/bin/python`), framework tensorflow 1.15.5, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04.
- Run A 2026-11-24T18:52:51Z → 2026-11-24T18:53:28Z; run B 2026-11-24T18:53:55Z → 2026-11-24T18:54:32Z; exit status A 0, B 0.
- Population: 186 toy pairs of outputs/audit/M8_SMOKE_EVIDENCE/run_a/select/toy_manifest.json (sha256 50ac8fb8b9fc6938ba630d0853523d358250137c8fbbc62704d2a73830f7f690) at their canonical global pairs_train_v1 positions; faces TRAIN-only via TrainFaceReader (O_NOATIME, M2 sha256).
- RNG: E03: deterministic (no sampling at inference).
- Conversion: `tf.cast(tf.clip_by_value(synth1, 0.0, 1.0) * 255, tf.uint8) [stdn model/utils.py:24,37]`.
- Output root (GPU host): `/home/student20261/workdir/m8_learned_banks/s7/a5e31322fd99bb6c4b6361b8b6a96790bb0d8f18/E03`.
- Peak CUDA memory per seed (bytes): {"A": {"42": null, "1337": null, "2026": null}, "B": {"42": null, "1337": null, "2026": null}}.
- atime ledger: pre 2026-11-24T18:52:49Z, post 2026-11-24T18:54:45Z; no TRAIN/VAL/TEST face atime or mtime changed.

## Scope

S7 establishes method-specific readiness for a separately authorized production launch only. No production bank, retraining, downstream training, TEST evaluation or GPAT work was performed.
