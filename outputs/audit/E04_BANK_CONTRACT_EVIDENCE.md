# E04 bank-contract evidence (read-only CPU probe)

**Status:** `E04_BANK_CONTRACT_EVIDENCE_COLLECTED` (27/27 gates).

Host clone `0728cb1b2be0` (clean True), Python 3.8.20, TensorFlow 1.15.5, CPU only True, 2026-11-25T12:19:52Z.

## Final checkpoints (BASELINE_FINAL_STATE_V1)

| seed | run_id | file | bytes | sha256 |
|---|---|---|---|---|
| 42 | `40befa6004aee85c` | `iteration_150000.data-00000-of-00001` | 21185796 | `55891ee1fd3a46ea3c451680d9a1a67998ede8246d754b6ea92aaf4bc45d71c6` |
| 42 | `40befa6004aee85c` | `iteration_150000.index` | 17435 | `f83983429ee5f7f8b82e8802439173f54e6d7684b4ffab4ad9f6425fb40dcc02` |
| 42 | `40befa6004aee85c` | `iteration_150000.meta` | 2827829 | `fc441707a1300b74a1eaea53f1265bbbf05cb25b7204d7457925cbea66b24933` |
| 1337 | `a42fdd2a2accaf0a` | `iteration_150000.data-00000-of-00001` | 21185796 | `fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5` |
| 1337 | `a42fdd2a2accaf0a` | `iteration_150000.index` | 17435 | `1f284d8be5a4234bc4a7cdead2aa49b00ee413907c83c2d962ac1cbc06fea560` |
| 1337 | `a42fdd2a2accaf0a` | `iteration_150000.meta` | 2827870 | `bade2ca14ae01e67384705543789bf39d4baabd8411f7253fbbda7ba139af233` |
| 2026 | `e7100eb9a5ab0537` | `iteration_150000.data-00000-of-00001` | 21185796 | `089c5e01da28c086e3660d4ba1ae4248cbf80cae907a163a0282f4af6812e0b2` |
| 2026 | `e7100eb9a5ab0537` | `iteration_150000.index` | 17435 | `af333ff6094c178ad08140ebc5a6a42824835eba62b1d722fe924dfac7bc2b5a` |
| 2026 | `e7100eb9a5ab0537` | `iteration_150000.meta` | 2827870 | `b0285d65ad41d75979ea5b16ec3e9a54cb422c348975bcd7ad9292b3060d2090` |

| seed | variables | classes | G BN moving stats | min moving_variance | max abs moving_mean |
|---|---|---|---|---|---|
| 42 | 500 | D_bn_moving 32, D_trainable 72, G_bn_moving 30, G_trainable 72, counter 4, optimizer_slot 288, other 2 | 30 | 0.000262 | 1.32 |
| 1337 | 500 | D_bn_moving 32, D_trainable 72, G_bn_moving 30, G_trainable 72, counter 4, optimizer_slot 288, other 2 | 30 | 0.000127 | 1.78 |
| 2026 | 500 | D_bn_moving 32, D_trainable 72, G_bn_moving 30, G_trainable 72, counter 4, optimizer_slot 288, other 2 | 30 | 0.000237 | 1.86 |

## Frozen M6D3e graph: the `synthesized_spoof` tensor

- Tensor `add_6:0`, shape [4, 256, 256, 3]; 850 ancestor ops; inputs only `sampling_offsets`, `synthetic_rgb`.
- 14 FusedBatchNorm ancestors, `is_training` = [True]: the frozen graph normalises with **batch statistics** of the 8-row batch (4 live targets + 4 spoof sources); no moving statistic is read.
- `T_A` (the AddN that D12 found non-bitwise on GPU) is an ancestor: False. AddN ancestors: none.
- The original-pass and hard-pass BN update ops assign the same 30 generator moving statistics (update counts {'D': 32, 'hard': 30, 'original': 30}).
- 576 ancestor ops request `/cpu:0` (bilinear resize and warp, as qualified).

## Gates

| gate | pass |
|---|---|
| `H.cpu_only_clean_clone` | PASS |
| `C.closure_status` | PASS |
| `C.final_checkpoint_bytes_seed42` | PASS |
| `C.final_checkpoint_table_seed42` | PASS |
| `C.final_state_counters_seed42` | PASS |
| `C.no_ema_shadow_seed42` | PASS |
| `C.generator_bn_moving_statistics_finite_seed42` | PASS |
| `C.final_checkpoint_bytes_seed1337` | PASS |
| `C.final_checkpoint_table_seed1337` | PASS |
| `C.final_state_counters_seed1337` | PASS |
| `C.no_ema_shadow_seed1337` | PASS |
| `C.generator_bn_moving_statistics_finite_seed1337` | PASS |
| `C.final_checkpoint_bytes_seed2026` | PASS |
| `C.final_checkpoint_table_seed2026` | PASS |
| `C.final_state_counters_seed2026` | PASS |
| `C.no_ema_shadow_seed2026` | PASS |
| `C.generator_bn_moving_statistics_finite_seed2026` | PASS |
| `G.generator_variables_in_checkpoint_seed42` | PASS |
| `G.generator_variables_in_checkpoint_seed1337` | PASS |
| `G.generator_variables_in_checkpoint_seed2026` | PASS |
| `G.synthesized_spoof_inputs` | PASS |
| `G.synthesized_spoof_generator_only` | PASS |
| `G.frozen_graph_bn_uses_batch_statistics` | PASS |
| `G.synthesized_spoof_independent_of_T_A_AddN` | PASS |
| `G.bn_moving_statistics_shared_original_and_hard` | PASS |
| `D.feed_pack_manifest` | PASS |
| `D.prepass_manifest` | PASS |

Scope: read-only CPU evidence: no Session on data, no face read, no checkpoint restored into a graph, no GPU, no VAL/TEST, no write outside this record.
