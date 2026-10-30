# E04 Physics-STD: production requalification under approved D12

**Status:** `E04_PRODUCTION_QUALIFIED_READY_FOR_SCIENTIFIC_EXECUTION`, 52/52 gates. Qualification seed 60904 only. **No E04 scientific seed was launched**, no scientific seed is authorized by this record, and no E04 scientific result exists. Machine-readable record: `E04_PRODUCTION_QUALIFICATION_D12.json`.

- Approved amendment `configs/amendments/e04_d12_procedural_reproducibility.yaml` (`OWNER_APPROVED`), criteria `9f1a2471f0c9e6ad24b9cc4727779361b763ec84b7e43dc221fc6b411c199d0d`.
- Original record `outputs/audit/E04_PRODUCTION_QUALIFICATION.json` (`d158a1ca773cfae2…`) is unchanged: 41/42, `K.real_data_gradient_repeatability` FAIL. It is not re-counted.
- Pre-pass `274a4b0a52077cae18a5477565429410b24479236e8835415cf0eb783f026279` (fresh repeat byte-identical and equal to the frozen pre-pass); feed pack `db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b`.
- Production construction `61f1ed5868d6774a…` (11315 ops).
- 3.099 / 3.200 s per iteration, projected 129.1 / 133.4 h for 150000 iterations.

## Gates (41 retained + D12 R1-R10 + evidence)

| gate | result |
|---|---|
| `A.frozen_contract_and_owner_resolution` | pass |
| `A.pinned_manifests` | pass |
| `A.same_production_identity_all_cases` | pass |
| `A.qualification_seed_only` | pass |
| `B.prepass_repeat_byte_identical` | pass |
| `B.prepass_population` | pass |
| `B.prepass_cpu_qualified_settings` | pass |
| `B.prepass_firewall` | pass |
| `B.prepass_samples` | pass |
| `C.feed_pack_verified_gpu_and_laptop` | pass |
| `C.feed_pack_binds_prepass` | pass |
| `D.det_a_rows_and_hard_draws_equal_replay` | pass |
| `D.det_b_rows_and_hard_draws_equal_replay` | pass |
| `D.det_inputs_identical` | pass |
| `D.seed42_full_streams_equal_approved_vectors` | pass |
| `D.seed1337_full_streams_equal_approved_vectors` | pass |
| `D.seed2026_full_streams_equal_approved_vectors` | pass |
| `E.construction_equals_qualified_m6d3e` | pass |
| `E.graph_inventory` | pass |
| `E.initial_state_identical` | pass |
| `F.det_a_step_transitions_equal_m6d3e` | pass |
| `F.det_a_pure_fetch_nonmutating_and_equal` | pass |
| `F.det_b_step_transitions_equal_m6d3e` | pass |
| `F.det_b_pure_fetch_nonmutating_and_equal` | pass |
| `F.iteration0_losses_identical_across_processes` | pass |
| `G.losses_finite` | pass |
| `H.det_a_periodic_checkpoints` | pass |
| `H.det_b_periodic_checkpoints` | pass |
| `H.terminal_probe_selected_semantics` | pass |
| `H.failure_probe` | pass |
| `I.det_a_summary_and_records` | pass |
| `I.det_b_summary_and_records` | pass |
| `I.terminal_probe_summary_and_records` | pass |
| `I.failure_probe_summary_and_records` | pass |
| `J.firewall_all_cases` | pass |
| `J.atime_test_unchanged` | pass |
| `J.manifests_unmodified` | pass |
| `J.atime_test_unchanged_after_diagnostics` | pass |
| `J.no_scientific_run_root` | pass |
| `K.environment_lock_except_driver` | pass |
| `K.d9_compatibility_reprobe_matches_accepted_variant` | pass |
| `D12.R10_firewall` | pass |
| `D12.R1_graph_identity` | pass |
| `D12.R2_deterministic_inputs_and_initial_state` | pass |
| `D12.R3_gradient_connectivity` | pass |
| `D12.R4_optimizer_application` | pass |
| `D12.R5_state_transitions` | pass |
| `D12.R6_finite` | pass |
| `D12.R7_forward_integrity` | pass |
| `D12.R8_gradient_integrity` | pass |
| `D12.R9_retained_gates` | pass |
| `D12.evidence_complete_and_hashed` | pass |

## Superseded gate (recorded, not gating, never counted as passed)

- `K.real_data_gradient_repeatability`: computed result FAIL on this requalification.

## Disclosed numerical behavior (D12)

- GPU computation is not bitwise reproducible. Parameters, losses and checkpoints of two runs with the same seed may diverge and are not guaranteed identical. D12 does not bound long-run trajectory divergence.
- Diagnostics (not gating): max functional step-gradient relative L2 5.25e-05; varying forward tensors per iteration [19, 0, 0, 19, 16, 56, 55, 0, 0]; distinct post-state hashes across processes per iteration [3, 3, 3].

## Remaining before scientific execution

- A separate owner launch decision for scientific seeds 42, 1337 and 2026.
