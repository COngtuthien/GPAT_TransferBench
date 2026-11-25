# M10 downstream ResNet-18 scientific readiness

**Status:** `M8_DOWNSTREAM_BLOCKED_BY_SCIENTIFIC_CONTRACT`.

## Gates

- G01_protocol_authorities_verified: pass
- G02_seven_families_21_banks_reverified: pass
- G03_bank_manifests_equal_audits: pass
- G04_e05_low_contrast_and_e04_clipped_included: pass
- G05_populations_and_split_firewall: pass
- G06_augmentation_equivalence: pass
- G07_runner_bitwise_equal_to_qualified_smoke_runner: pass
- G08_cuda_policies_replay_and_checkpoint_reload: pass
- G09_plots: pass
- G10_laptop_tests: pass
- G11_threshold_rule_complete: **NOT MET**
- G12_science_precision_pinned: **NOT MET**
- G13_checkpoint_retention_and_run_root_approved: **NOT MET**
- G14_science_environment_approved: **NOT MET**
- G15_launcher_and_independent_run_auditor: **NOT MET**
- G16_resources: pass

## Owner decisions

- **O1_science_precision** (blocking): D13 leaves the science AMP dtype and GradScaler policy (and the CUDA determinism settings) to be pinned before M10. Which precision policy do all 24 runs use? Recommended `A_CUDA_AMP_FP16_GRADSCALER_DETERMINISTIC`.
- **O2_threshold_tie_rule** (blocking): Spec 15 breaks ACER ties only by 'closest to 0.5'. When two minimum-ACER thresholds are equally close to 0.5 (always the case for constant or fully inverted VAL scores: 0 and 1 at ACER 1/2) the frozen rule selects nothing and the implementation aborts the whole run. Which completion applies? Recommended `C_EPOCH_INELIGIBLE`.
- **O3_checkpoint_retention_and_run_root** (blocking): Which checkpoints are kept, and where do detector run roots live? The system SSD holding the runtime root is ~97% full (~27 GB free); the approved HDD has ~3.0 TB free but only the bank root is registered. Recommended `A_SELECTED_ONLY_HDD_NEW_ROOT`.
- **O4_science_environment** (blocking): The evaluator contract authorized gpat-m8-downstream (environments/m8_downstream.lock.json) for the smoke only. Approve it (unchanged) for the M10 scientific runs? Recommended `A_APPROVE_UNCHANGED`.
- **O5_execution_order_and_gpu_sharing**: Execution order and GPU policy on the shared RTX 3090 (an unrelated user's jobs held ~10-12 GB and ~93% utilization during this assessment). Recommended `A_SEED_MAJOR`.
- **O6_primary_table_view**: Spec 7: state the primary table (RAW-MATCHED or QUALITY-MATCHED) before inspecting TEST. All seven previous-work banks have 8838 valid outputs; N_common across the final comparison depends on later GPAT banks. Not needed before training; needed before any TEST evaluation. Recommended `A_RAW_MATCHED_PRIMARY`.
- **O7_track_id_labels**: Prediction rows carry track_id. Confirm the A1 track per method (method_status.csv) and 'binary' for E00 (spec 17), replacing the smoke runner's fixed 'binary'. Recommended `A_CONFIRM`.

## Plan (provisional)

24 runs, one at a time, seed-major (O5). Estimated 0.165 s/step, ~1419 s/run, ~9.5 h for 24 runs (unrelated GPU jobs were running; real throughput may differ (measure in the first scientific run's audit, never extrapolate from generator jobs)).
Storage ~57 MB/run with selected-only retention (~1.4 GB total).

## Not done

- no scientific detector training
- no TEST access or evaluation
- no DINOv3
- no GPAT
- no generator or bank change
- no launch authority activated
