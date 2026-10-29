# E04 Physics-STD: production-readiness audit

**Status:** `E04_BLOCKED_BY_SOURCE_GAP` (readiness audit complete; `IMPLEMENTED_NOT_EXECUTED`, `CONTROLLED_ADAPTATION`). E04 scientific execution is **not** authorized and no E04 authority exists. The machine-readable record is `E04_PRODUCTION_READINESS_AUDIT.json`.

The training graph and the fixed-geometry runtime re-qualify on synthetic data under the current host driver. The real-data path cannot be built yet: nine production decisions (D1–D9) are not fixed by any frozen artifact, and spec §8.4 requires `BLOCKED_BY_SOURCE_GAP` rather than a choice.

## Qualification stages

| stage | status |
|---|---|
| A static_source_and_contract | `PASS` |
| B environment_and_dependencies | `PASS_SYNTHETIC_PENDING_D9_DRIVER_RULING` |
| C scientific_graph_construction | `PASS_SYNTHETIC` |
| D synthetic_forward_and_gradient | `PASS_SYNTHETIC_WITH_DISCLOSED_D3_VARIANCE` |
| E deterministic_data_path | `BLOCKED_D4_D5_D7_D8` |
| F real_train_val_feed | `BLOCKED_D4_D5_D6_D8` |
| G depth_m0_integrity | `BLOCKED_D1_D2_D3` |
| H checkpoint_and_lifecycle | `NOT_IMPLEMENTED` |
| I test_firewall | `NOT_IMPLEMENTED (pattern reusable from E03)` |
| J real_data_production_qualification | `BLOCKED` |
| K independent_evidence_verification | `BLOCKED` |

## Synthetic re-qualification (driver 580.178.04)

- Geometry (M6D3a worker), CPU and CUDA: repeat byte-identical; final depth `bf442bca63408707…` on both devices, equal to the committed M6D3a value; spoof exact zeros; no file-access denial.
- Training graph (M6D3e worker, qualification seed 314159, 3 applies): `PASS`, 33/33 tests; GraphDef `7e3e84ccc40f8390…` equal to the committed hash; 9143 common fields, differing: `.firewall.events.open`, `.firewall.events.os.listdir`, `.operation_structure_sha256`, `.steps`, `.tests.transcript`.
- Post-step state differs from committed process 1 only in D3: 4/51/51 tensors after steps 1/2/3 (committed process pair: 0/51/51). The 4 Step-1 differences (`D/D3/conv1/BN/moving_mean:0`, `D/D3/conv2/BN/moving_mean:0`, `D/D3/conv3/BN/moving_mean:0`, `D/D3/conv4/BN/moving_mean:0`) are new relative to M6D3e and are disclosed under D9.

## Decisions required from the owner

| id | decision | recommendation | blocks |
|---|---|---|---|
| D1 | 3DDFA crop/ROI policy for canonical 256 faces | A | depth-M0 pre-pass, Q140 correspondences, stages E-K |
| D2 | geometry pre-pass population and outputs | A (B only with D6 = B) | depth-M0 pre-pass, trace transfer |
| D3 | pre-pass device (engineering) | A | depth-M0 pre-pass determinism record |
| D4 | TRAIN pair sampling over pairs_train_v1 | A | real TRAIN feed, stages F, J |
| D5 | TRAIN augmentation | A | real TRAIN feed |
| D6 | VAL diagnostics (engineering) | A | lifecycle definition |
| D7 | hard-sample RNG stream | A | production controller |
| D8 | graph input encoding | A (confirm) | real TRAIN feed |
| D9 | driver-version tolerance for E04 (engineering) | A, with the Step-1 D3 variance disclosed | stage B acceptance, launch |

Options and reasons for each decision are in the JSON (`decisions_required`).

## Production implementation

| missing file | waits on |
|---|---|
| `tools/e04_geometry_prepass.py` | D1, D2, D3 |
| `methods/physics_std/feed_pack.py (hash-verified TRAIN pack: pair rows + pre-pass M0/mask/Q140, no TEST)` | D1, D2, D4 |
| `methods/physics_std/production_io.py (contract/source/env/pack verifiers, resolved config)` | D4-D9 |
| `methods/physics_std/production_runner.py (controller: feed, 3-step loop, checkpoints, lifecycle)` | D4, D5, D6, D7, D8 |
| `tools/run_e04.py (fail-closed CLI, fresh only)` | D4-D9 |
| `tools/e04_production_qualification_evidence.py + tests` | all of the above |
| `tools/e04_seed_completion_audit.py` | after qualification |

## Boundary

No benchmark image was decoded and no TEST, VAL or TRAIN sample was accessed. No E04 run root was created, no experiment seed was used, no environment was modified, and no E03 artifact changed.

**Next milestone:** owner production resolution for E04 (D1-D9); then the geometry pre-pass, the feed pack, the production controller and CLI, and real-TRAIN production qualification.
