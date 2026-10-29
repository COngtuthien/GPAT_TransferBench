# E03: family host-state owner ruling (`H.host_final_state`)

**Status:** `OWNER_RULING_RECORDED_CONDITIONS_VERIFIED`. Ruling `OWNER_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE_SUBSTITUTION_FAMILY_HOST_STATE` (2026-10-29). Ruling text `docs/spec/amendments/GPAT_TransferBench_v1_0_E03_Family_Host_State_Reboot_Owner_Ruling.md`, machine-readable record `configs/amendments/e03_family_host_state_reboot_resolution.yaml`.

This is an audit-tool amendment record, not a scientific authority. It changes neither the trained E03 models nor the procedure, and no E03 evidence file was modified. The E03 training authority stays `d547fc9f8932a273d719ae589aade85a9d989b85`; the family closure is `374de40c66818bf27cf3bd98cb1871046bd57c2a`.

A fresh read-only host recheck at 2026-10-29T08:44:28Z re-builds the family closure to `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`, 74/74 gates.

| check | result |
|---|---|
| `1_runs_completed_before_verified_reboot` | pass |
| `2_original_evidence_valid` | pass |
| `3_no_e03_tmux_session` | pass |
| `4_no_e03_runner_process` | pass |
| `5_exact_verified_boot` | pass |
| `6_boot_after_every_seed` | pass |
| `7_all_other_gates_pass` | pass |
| `gate_probes_fail_closed_and_live_path_unchanged` | pass |
| `gate_result_committed_and_fresh` | pass |
| `earlier_disclosures_preserved` | pass |
| `ruling_record_consistent` | pass |

Gate probes, run in memory on the committed host recheck:

| probe | expected gate | gate | tmux sub-result |
|---|---|---|---|
| `other_gate_fails` | fail | fail | FAIL_POST_COMPLETION_REBOOT_EVIDENCE |
| `different_boot` | fail | fail | FAIL |
| `boot_before_last_seed_end` | fail | fail | FAIL |
| `e03_tmux_session_present` | fail | fail | FAIL |
| `runner_process_present` | fail | fail | PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE |
| `live_panes_dead_status_0` | pass | pass | LIVE_PANES_DEAD_STATUS_0 |
| `live_pane_nonzero_status` | fail | fail | FAIL |

The tmux sub-result covers only the pane condition; the gate also requires the hostname, the GPU identity and no method process, so `runner_process_present` fails the gate with a passing pane condition.
