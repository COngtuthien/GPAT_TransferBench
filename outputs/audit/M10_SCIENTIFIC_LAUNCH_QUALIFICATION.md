# M10 scientific launch qualification

**Status:** `M10_DOWNSTREAM_SCIENTIFIC_READY_AUTHORITY_ACTIVE`

Qualified source commit (clean GPU-host clone): `a4dd841b4b2a3e636af3bad2818a0effdc87a544`.

## Gates

- Q01_owner_decisions_recorded: pass
- Q02_frozen_files_unchanged: pass
- Q03_precision_pin_O1: pass
- Q04_storage_root_verified: pass
- Q05_environment_and_gpu_verified: pass
- Q06_qualified_commit_clean_clone: pass
- Q07_fixture_runs_O1: pass
- Q08_bitwise_replay: pass
- Q09_gradscaler_skip_and_fail_closed: pass
- Q10_auditor_pass_on_good_runs: pass
- Q11_auditor_detects_injected_failures: pass
- Q12_test_firewall: pass
- Q13_scientific_dry_preflight_24: pass
- Q14_host_tests_cuda: pass
- Q15_laptop_tests: pass
- Q16_full_regression_no_new_failures: pass
- Q17_resources: pass

## Plan

24 runs, one at a time, seed-major (O5): E00/42, E01/42, E02/42, E03/42, E04/42, E05/42, E06b/42, E06c/42, E00/1337, E01/1337, E02/1337, E03/1337, E04/1337, E05/1337, E06b/1337, E06c/1337, E00/2026, E01/2026, E02/2026, E03/2026, E04/2026, E05/2026, E06b/2026, E06c/2026.

Provisional throughput: {"cpu_decode_augment_batch64_seconds": 0.0672017460456118, "fixture_epoch_wall_seconds": [1.6851489150431007, 2.3610346590867266, 3.028917800053023], "gpu_eval_batch64_seconds": 0.010855840391013772, "gpu_peak_allocated_mib": 999, "gpu_train_step_seconds_median": 0.03498220653273165, "matrix_24_hours_estimate": 5.788536916427935, "note": "fixture epochs are tiny (3 steps + 32 VAL frames); see the benchmark for per-step cost", "per_run_seconds_estimate": 868.2805374641903, "status": "PROVISIONAL (random tensors / noise PNGs on a shared GPU; real throughput is measured by the first scientific run's audit)", "val_epoch_seconds_estimate": 3.8248217353946528}.

Storage: {"basis": "measured fixture checkpoint (same ResNet-18 state); VAL parquet scaled per row to 3121 frames; steps.jsonl scaled to 7000 lines", "checkpoint_bytes": 44782603, "matrix_24_bytes_estimate": 1898954508, "per_run_bytes_estimate": 79123104}.

## Not done

- no scientific detector training
- no TEST access or evaluation
- no DINOv3
- no GPAT
- no generator or bank change
