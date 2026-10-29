# E04 Physics-STD: production qualification

**Status:** `E04_PRODUCTION_QUALIFICATION_INCOMPLETE`, 41/42 gates. Qualification seed 60904 only; **no E04 scientific seed was launched** and no E04 scientific authority exists. Machine-readable record: `E04_PRODUCTION_QUALIFICATION.json`.

- Frozen geometry pre-pass `274a4b0a52077cae18a5477565429410b24479236e8835415cf0eb783f026279`: 12668 TRAIN faces (3830 live, 8838 spoof), two independent CPU runs byte-identical.
- Feed pack `db1908d56efb23758fee49f88d2158374b85572a213a269419881bd02037520b`, verified on the GPU host and on the laptop.
- Production construction `61f1ed5868d6774a…` (11315 ops) equals the qualified M6D3e construction.
- 0.944 / 0.942 s per iteration, projected 39.4 / 39.2 h for 150000 iterations.

## Gates

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
| `K.real_data_gradient_repeatability` | **FAIL** |
| `K.d9_compatibility_reprobe_matches_accepted_variant` | pass |

## Crop boundary (official landmark ROI vs the 256 frame)

| dataset/role | faces | ROI outside frame | zero-padded faces | median padded px | max padded px |
|---|---|---|---|---|---|
| casia_fasd / live | 681 | 681 | 681 | 35588 | 69888 |
| casia_fasd / spoof | 2520 | 2520 | 2520 | 37505 | 72848 |
| msu_mfsd / live | 272 | 271 | 271 | 3276 | 17696 |
| msu_mfsd / spoof | 1200 | 1197 | 1197 | 3224 | 11536 |
| siwmv2 / live | 2877 | 2842 | 2842 | 3304 | 22673 |
| siwmv2 / spoof | 5118 | 5078 | 5078 | 3775 | 34953 |

## Determinism

data, RNG draws, input tensors, initial state and iteration-0 losses are identical across processes; later loss values may differ through the disclosed D3 GPU nondeterminism (M6D3e, D9).

## Blocker: real-data adversarial gradient is not bit-repeatable (fail closed, D9 condition 6)

- With identical initial state and identical real TRAIN inputs (iteration 0), every per-loss gradient is bit-repeatable in-process and across 3 processes except `L_G`.
- L_G (adversarial generator loss) backpropagates through the four discriminators into 58 generator variables. Its backward path alone contains GPU `ResizeBilinearGrad` (8) and `ScatterNd` (16; the gradient of the CPU-forward trace warp `GatherNd`), op types without deterministic TF 1.15 GPU kernels.
- Size: median per-variable relative L2 difference 4.20e-07; the largest relative values (1.21) occur only for biases of convolutions followed by training-mode BN, whose exact gradient is zero; max |diff| 1.11e-04.
- Consequence: E04 real-data training is not bitwise reproducible across processes; the generator state differs from Step 1 of iteration 0 and later losses diverge at ~1e-7 to 1e-6 relative within 2 iterations. Data, RNG draws, inputs, initial state and iteration-0 losses stay identical.
- M6D3e recorded bit-repeatable gradients for every loss on synthetic data, so this is a new gradient-level observation; the M6D3e statement that the warp has a deterministic backward path holds for its forward pass only (the gradient ScatterNd is placed on /gpu:0). It is not a driver difference (in-process).
- No graph, placement or optimizer change was made. An owner decision is required.

