# E05 three-seed scientific training closure

**Status:** `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`, 83/83 family gates passing. The machine-readable record is `E05_THREE_SEED_TRAINING_CLOSURE.json`, built by `tools/e05_three_seed_training_closure.py` from the three committed seed closures and a read-only host recheck (`tools/e05_family_host_recheck.py`, embedded in the JSON). The per-seed table is `E05_THREE_SEED_FINAL_CHECKPOINTS.csv`.

**Scope:** this closes E05 generator **scientific training** only.

- The synthetic bank has **not started**, downstream FAS evaluation has **not started**, and the benchmark is **not complete**.
- No performance metric exists or is reported here.
- Future E05 metrics use `MEAN_AND_STD_OVER_ALL_THREE_SEEDS`.

| | |
|---|---|
| method | E05, **PCGAN (controlled architecture resolution)**, CONTROLLED_ADAPTATION, track A_FAIR_COMMON_IDENTITY_FREE |
| scientific training authority (all seeds) | **`efdf753156045dd2bed52229d9eac869b18d72f8`**. The seed closure commits and this commit are evidence only, and the GPU checkout is clean at efdf753. |
| seed set | exactly **{42, 1337, 2026}**, 3/3 completed. There is no fourth, replacement, preferred, dropped or best seed, and no cross-seed choice. |

## Final checkpoints (one per seed, BASELINE_FINAL_STATE_V1)

| seed | run_id | final checkpoint | sha256 | bytes | metrics.jsonl sha256 | run-root listing digest | seed closure |
|---|---|---|---|---|---|---|---|
| 42 | `18463bf5cd471e55` | `checkpoints/iteration_4000.pth` | `4ad80b35e759154f741d249db8c6e43622cab916e13dc785ad9273bd65113cb1` | 232,113,504 | `4b2fc19c…` | `526419f0…` | `072ce7d` |
| 1337 | `036c806bb2f339b7` | `checkpoints/iteration_4000.pth` | `b6a4cb2c61b1b582f1acd1eb3514f6a15ea49e544e1dac48415004c6ce4f7b34` | 232,113,504 | `2c6a4746…` | `638c777f…` | `023beb4` |
| 2026 | `b739351daf9779bc` | `checkpoints/iteration_4000.pth` | `b7f0575f1e061936175ac04d2ac333e1008577bdd3654566c009a2f72bf0b166` | 232,113,504 | `aeb1e5c6…` | `1667ccc2…` | `21d47c2` |

The full hashes are in the CSV. The seed42 and seed1337 values are taken from their committed closure evidence, and the seed2026 values from its completion audit. The read-only host recheck confirms them against the run roots, including both checkpoints of each seed and the listing digest, which stayed stable during the recheck.

The final-state rule is applied independently within each seed: the selected state is iteration 4000, the only selected checkpoint of that run.

## Per-seed results (identical in every seed)

- **Completion:** status `completed`, 4000 contiguous iterations (1…4000).
- **Optimizer applications:** D = G = hook_D = hook_G = t and total = 2t at every iteration, so the final counts are D = 4000, G = 4000, total 8000.
- **Pair order:** 4000/4000 sampler positions equal the independent seed prediction.
- **Pairs:**
  - 4000 distinct TRAIN pairs, each matching its frozen `pairs_train_v1` row.
  - The source is a spoof face and the target a live face, both TRAIN; the face PNG digests equal the M2 accounting.
  - The source and target sets are disjoint across the manifest.
- **Inputs:** [1,3,256,256] float32 in [-1, 1].
- **Numerics:** all losses finite, the five G terms active, `L_G_total` equal to their unit sum, and 80/80 D and 59/59 G gradients finite. stderr is empty.
- **Checkpoints:** exactly 2 events, iteration_0001 (periodic) and iteration_4000 (selected), with all floating tensors finite. There is no `.partial`, runner_state or resume sidecar, and the run was fresh (`NOT_QUALIFIED_FRESH_ONLY`).
- **Gate semantics:** each seed closure is rebuilt with the current seed tool and with the copy in its own closure commit. The two copies differ only in the run_id table, and every seed has the same 65 gates.

## Cross-seed consistency

**Identical in all three runs:**

- config sha `478756e1…` (= snapshot), the spec, A2/A5/M6D4c/M6D4e identities, the logging contract, and the source pin `6baa180f…`/tree `25a434f1…`;
- the pair manifest `a5e4fdae…`, environment lock `849a1004…`, Python `d3255ce6…`, the 54 packages, torch 2.12.1+cu130, torchvision 0.27.1+cu130, CUDA 13.0 and cuDNN 92000;
- the extensions `5afdb236…` and `40ba9db7…`, with no rebuild;
- FP32 precision flags, batch 1, 0 workers, seeded shuffle with drop_last false, and the D → G order;
- Adam with lr 1e-6 and betas (0.9, 0.999), and the five unit-weighted G losses;
- the [-1, 1] input mapping with no augmentation, the 1/4000 checkpoint cadence, BASELINE_FINAL_STATE_V1, fresh-only runs and the VAL/TEST firewall;
- every common run_manifest field. The command lines differ only by `--seed`.

**Different by design:**

- the run_manifest seed, run_id, run_uuid and timestamps;
- three resolved_config lines (`experiment_seed`, `run_dir`, `dataloader.generator_seed`);
- the run_start `seeding.seed` and `loader.generator_seed`;
- seed-derived pair order and model initialisation, which give distinct sampler orders, metrics and learned checkpoint bytes.

## Firewall

- **VAL = 0 and TEST = 0** in every run. Each seed's own pre-launch and post-run stat-only snapshots show 3121/3121 VAL and 3027/3027 TEST faces atime-sensitive and unchanged.
- No consumed sample is VAL or TEST, and `test_split_accessed` and `val_split_accessed` are false.
- TRAIN scope is proven by the pair manifest, the independent sampler order and the M2 face digests.

## Driver ruling (applied consistently)

`lock_driver_version = 595.84`, `observed_driver_version = 580.178.04`, `driver_policy = OWNER_APPROVED_DRIVER_VERSION_ONLY_RUNTIME_COMPATIBILITY` in all three runs.

The GPU model (RTX 3090), UUID `GPU-b722cd9d-c9fa-223c-915e-cc09d9b862b7` and every other lock field are exact. This is not evidence that arbitrary driver changes are acceptable.

## Host final state (read-only recheck)

- **Run roots:** `runs/m6/E05` holds exactly `seed_42`, `seed_1337` and `seed_2026`, each unchanged against its closure.
- **Processes:** the three E05 tmux panes are dead with status 0. There is no E05 or other method process and no GPU compute process.
- **GPU checkout:** clean on `m6-baselines` at efdf753.
- **Leftover bundle:** `/tmp/e05_candidate.bundle` (193,640 bytes) is still present. It is no longer needed for scientific identity and was not deleted.

## Confirmations

- No best, preferred, dropped or replacement seed.
- No VAL checkpoint selection and no TEST use.
- No bank generation and no downstream evaluation.
- No E03 or other method started, and no GPAT work.
- No run root was mutated.

The production resolutions (pair roles, seeded order, [-1, 1] mapping, checkpoint cadence) are owner-approved controlled implementation resolutions, not native, faithful or official PCGAN choices.
