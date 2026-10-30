# E04 D12 requalification: execution notes and disclosures

These notes accompany `E04_PRODUCTION_QUALIFICATION_D12.json` (`E04_PRODUCTION_QUALIFIED_READY_FOR_SCIENTIFIC_EXECUTION`,
52/52). They change no gate.

## Execution

- **Commits:**
  - owner ruling: c60481b91922c1c72c73a603219585ec4a78c1e4;
  - gate activation: c676459a3d4b1ee64f5348bf84d44b72a36fb685.
  The GPU checkout was clean at c676459 at the start and the end of the run.
- **Environment:**
  - GPU host RTX 3090 GPU-b722cd9d, driver 580.178.04 (D9);
  - gpat-m6-e03, `/tmp/gpat_e04_d10_env.sh` (the original precision variable set), PYTHONHASHSEED=60904.
- **Run:** one sequence, `run.sh` in the archive, 2026-10-29T23:58:53Z to 2026-10-30T00:28:59Z, every step exit 0.
  1. atime pre.
  2. Two fresh CPU geometry pre-pass runs (`<rt>/qualification/e04_prepass_d12/run_{a,b}`, read-only). They ran in the background, in parallel with the following steps.
  3. Feed-pack verification.
  4. det_a, det_b, terminal_probe, failure_probe (fresh run id 0f86725250f57541).
  5. D9 re-probe.
  6. atime post.
  7. rdp 1–3 and the L_G probe.
  8. D12 production replay P1–P3 (3 iterations).
  9. d10_trace2 P1–P2.
  10. Pre-pass compare (run_a vs run_b, and run_a vs the frozen `e04_geometry_prepass_v1`).
  11. atime post2.
- **Raw outputs:** `<rt>/qualification/e04_production/d12_requalification_c676459`, read-only, 82 files.
  SHA256SUMS `3742793b…` = `E04_PRODUCTION_QUALIFICATION_D12_RAW_SHA256SUMS.txt`.
- **Re-derivation:** `build-d12` was run twice on the collected evidence and produced byte-identical outputs.

## Disclosures

- **Timing is not representative.** The recorded 3.10 / 3.20 s per iteration (projected 129 / 133 h) was measured while the two CPU pre-pass processes ran concurrently. The original qualification measured about 0.94 s per iteration without that load. No gate depends on timing.
- **Superseded K gate.** `K.real_data_gradient_repeatability` was recomputed and FAILS again in the same way: L_G only, GPU ResizeBilinearGrad 8 and ScatterNd 16, median relative L2 4.2e-7. It is recorded as superseded, not gating and never counted as passed. The original 41/42 record (`d158a1ca…`) is unchanged.
- **D12 diagnostics (not gating).**
  - The forward-variation root is `T_A:0` again, with maximum forward relative L2 5.7e-7.
  - Maximum functional step-gradient relative L2:
    - step 1: 1.8e-7;
    - step 2: 5.2e-5;
    - step 3: 0 (bitwise repeatable).
  - Post-iteration states differ across the 3 processes. In det_a vs det_b, 11 loss values differ per iteration from iteration 1. Iteration-0 losses are identical.
- **Executed-placement records.** The two traces' step-stats node sets differ in membership (4073 vs 4072 entries: constants, shape ops and CPU warp index arithmetic recorded in one trace and not the other). The D12 investigation saw the same (4072–4083 entries). Ops present in both have equal device types, and the canonical partition-graph set and fetched-output hash equal the pinned values. This is criterion R1.
- **No claims.** No bitwise reproducibility, identical weights for the same seed, or bound on long-run divergence is claimed.

## Not done

- No scientific seed (42, 1337, 2026) was launched and none is authorized.
- No TEST/VAL use, bank, E01/E02/GPAT or downstream evaluation.
- No graph, optimizer, data, environment or package change.
