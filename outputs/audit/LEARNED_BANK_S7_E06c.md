# M8 learned-bank S7 toy acceptance: E06c

**Status:** `S7_ACCEPTANCE_PASSED`, 98/98 gates passing; `s7_acceptance_passed` = true.

Label: S7_TOY (S7_NON_SCIENTIFIC toy acceptance; never a production bank or benchmark evidence).

Authority `configs/amendments/learned_bank_s7_launch_authority.yaml` (`0be8f0e14578…`), contract `3554577185a9…`, E04 seed-2026 record `47a4e4974d86…`, execution commit `a5e31322fd99bb6c4b6361b8b6a96790bb0d8f18`.

| seed | run A | run B | A/B bytes | manifest sha256 |
|---|---|---|---|---|
| 42 | PASS | PASS | identical | `57a4d9faedd1503f…` |
| 1337 | PASS | PASS | identical | `afa28386da1c1f53…` |
| 2026 | PASS | PASS | identical | `52099a8b3ce4d066…` |

Failed gates: none.

## Execution

- Environment `gpat-m6-e06c` (`/home/student20261/miniconda3/envs/gpat-m6-e06c/bin/python`), framework torch 2.12.1+cu130, GPU NVIDIA GeForce RTX 3090 `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7`, driver 580.178.04.
- Run A 2026-11-24T18:46:43Z → 2026-11-24T18:47:05Z; run B 2026-11-24T18:47:18Z → 2026-11-24T18:47:40Z; exit status A 0, B 0.
- Population: synthetic stream indices 0..185 (E06c_S<seed>_000000..000185); 2 official batches of 100, 14 discarded; stream seeds {42: 1660457813, 1337: 203456175, 2026: 1718284219}; no face read.
- RNG: DSDG: torch.Generator().manual_seed(dsdg_bank_seed(method, seed)); per batch of 100 randn noise, randn noise_s, z = cat(noise_s, noise, noise); 2 batches, 14 discarded.
- Conversion: `np.transpose((255 * spoof).astype('uint8'), (1, 2, 0)) [DSDG generated.py:63]`.
- Output root (GPU host): `/home/student20261/workdir/m8_learned_banks/s7/a5e31322fd99bb6c4b6361b8b6a96790bb0d8f18/E06c`.
- Peak CUDA memory per seed (bytes): {"A": {"42": 2634086400, "1337": 2634479616, "2026": 2634479616}, "B": {"42": 2634086400, "1337": 2634479616, "2026": 2634479616}}.
- atime ledger: pre 2026-11-24T18:46:41Z, post 2026-11-24T18:48:06Z; no TRAIN/VAL/TEST face atime or mtime changed.

## Scope

S7 establishes method-specific readiness for a separately authorized production launch only. No production bank, retraining, downstream training, TEST evaluation or GPAT work was performed.
