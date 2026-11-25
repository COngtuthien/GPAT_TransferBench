# E04 Physics-STD bank adapter: synthetic-fixture qualification

**Status:** `E04_BANK_ADAPTER_SYNTHETIC_QUALIFIED_S7_PENDING` (7/7 record gates).

Adapter commit `92ef74c99765`; contract `configs/amendments/e04_bank_generation_contract.yaml` (`45240cd25e840c16`).

## CPU fixture qualification: PASSED

- Host CPU (gpat-m6-e03, TF 1.15.5, CUDA hidden): **45/45** gates, 15.9 s, started 2026-11-25T12:54:44Z.
- Laptop unit tests: 33 ran, OK True (`python3 -m unittest -v tests.test_e04_bank_adapter tests.test_e04_bank_readiness`).

| category | passed | meaning |
|---|---|---|
| A | 8/8 | governance and frozen sources |
| C | 6/6 | graph integrity and restore |
| D | 12/12 | bank infrastructure |
| E | 5/5 | synthesis integrity (numpy references, roles, offsets) |
| F | 1/1 | frozen-graph comparison |
| G | 1/1 | BN inference-mode isolation |
| H | 6/6 | conversion, non-finite accounting, firewall |
| I | 3/3 | real E04 checkpoints (hash, index, moving statistics; no forward) |
| environment | 3/3 | environment / data firewall |

## Key results

- Frozen-graph comparison: the adapter composition fed the same 8-image batch reproduces `TrainingGraph.synthesized_spoof` and every warped component **bitwise** (max abs 0.0).
- BN mode: with moving statistics set to the image's own batch statistics, inference mode equals the training-mode forward to max abs 2.88e-05 (synthesis); with the approved stored statistics the difference is 1.05. The statistics source is the only difference.
- Independent numpy references (heads, sigmoid P, low32/low128, five warps, synthesis equation): max abs 8.41e-06.
- Conversion `(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)`: inputs [-1.0, -0.0, 0.0, 0.0019607844296842813, 0.003921568859368563, 0.5, 0.5, 0.998039186000824, 0.9999999403953552, 1.0, 1.5, 3.0] -> [0, 0, 0, 0, 1, 127, 127, 254, 254, 255, 255, 255].
- Inference graph: 102 variables (72 trainable, 30 moving), 15 FusedBatchNorm all `is_training=False`, 0 update/gradient/optimizer/random ops, batch 1.

## Real E04 checkpoints (hash, index, moving statistics; no forward)

| seed | run_id | identity | restored / in checkpoint | moving statistics finite | unchanged |
|---|---|---|---|---|---|
| 42 | `40befa6004aee85c` | `96e8963015077984` | 102 / 500 | True | True |
| 1337 | `a42fdd2a2accaf0a` | `4f183d2143d27d1d` | 102 / 500 | True | True |
| 2026 | `e7100eb9a5ab0537` | `9b0b391bacfece7b` | 102 / 500 | True | True |

## Real-device qualification: PENDING_E04_S7

- not shown by CPU: GPU kernel outputs of the real checkpoints
- not shown by CPU: GPU bitwise A/B replay (E04-Q5)
- not shown by CPU: real TRAIN faces, feed pack and Q140 geometry through load_e04_inputs
- not shown by CPU: environment-lock verification (production_runner.verify_environment) on the GPU
- not shown by CPU: production throughput and memory

## Record gates

| gate | pass |
|---|---|
| `host_run_at_qualification_commit` | PASS |
| `host_cpu_only_locked_interpreter` | PASS |
| `host_all_gates_passed` | PASS |
| `host_stdout_matches` | PASS |
| `no_gpu_claim` | PASS |
| `real_checkpoints_equal_contract` | PASS |
| `laptop_tests_passed` | PASS |

Not done: E04 S7 GPU inference, E04 production banks, <bank_root>/E04, downstream evaluation, TEST access, DINOv3, GPAT.
