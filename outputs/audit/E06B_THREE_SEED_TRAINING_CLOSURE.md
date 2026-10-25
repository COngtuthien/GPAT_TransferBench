# E06b DSDG-NATIVE: three-seed scientific training closed

**Status:** `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`, with 72/72 gates passing (`tools/e06b_three_seed_training_closure.py`, modelled on the E06c family tool). The machine-readable record is `E06B_THREE_SEED_TRAINING_CLOSURE.json`, and the authoritative final checkpoint table is `E06B_THREE_SEED_FINAL_CHECKPOINTS.csv`. The build re-derives from the committed per-seed closures and the embedded read-only host recheck (`tools/e06b_family_host_recheck.py`, 2026-10-25T03:41:13Z).

| | |
|---|---|
| method | E06b DSDG-NATIVE, track B_NATIVE_FULL_SECONDARY, fidelity **FAITHFUL_OFFICIAL_WITH_RUNTIME_COMPATIBILITY** |
| successful seeds | exactly **{42, 1337, 2026}**: 3 scientific runs, 3/3 completed |
| training authority | **`d9ecbef7295fc6b41ced6a4966b49810d36a8c87`** for all three (the checkpoint-context fix) |
| config sha256 | `9d665dc2c909d421b8e54964407e27bb40d133f11bceec2e2cb29810f268417f` |
| excluded history | seed42 attempt 1 `d3c0ddd5dc82a4a1` @ 24f7049, `FAILED_TECHNICAL_CHECKPOINT_WRITER_BUG`: not a family member, not counted, not selectable; still archived under `diagnostics/m6/` outside `runs/`, untouched (digest `8ce8d5a8…`, 14 files) |

This closes E06b **generator scientific training only**. The seed closure commits and this family closure are evidence/reporting only.

## Final checkpoints

Each final checkpoint is the official epoch-200 generator `checkpoints/netG_model_epoch_200_iter_0.pth` (79 577 600 bytes), selected by OFFICIAL_GENERATOR_EPOCH_200 independently within its seed. There is no cross-seed choice.

| seed | run_id | final netG sha256 | metrics.jsonl sha256 | run-root sha listing | seed closure |
|---|---|---|---|---|---|
| 42 (attempt 2) | `aa96129289d080ba` | `d56122e0cf363a6c870e29b394998732f4bcdaf35a96a6110e937905b64ff8aa` | `bcdbefd8…` | `71f8f357…` | 1f546c9 |
| 1337 | `9ffda3ea3f6cc767` | `645538e866a6d835a1bc35240c2f15c27296742850df96d722fd4d38609b4f53` | `4ee2b616…` | `f2cc290e…` | a8a7b5d |
| 2026 | `ec71778b28479de5` | `e8d0d01cfdb59b92d0f24318ab4fec042188bf0a72f9c59054ea10204a326cad` | `0442a4ad…` | `7f03af43…` | 1729333 |

All three are `completed` with VAL access 0, TEST access 0 and SiW access 0.

## Cross-seed consistency

- **Per-seed closures:** each `E06B_SEED<s>_COMPLETION.json` re-derives with its own tool, and its evidence, report, SHA manifest and tool are byte-equal to its closure commit. Each passed 84/84 gates, and the gate sets are identical apart from the seed-named parent and attempt-1 gate names.
- **Identity:** the run manifests are equal in every field except `command_line` (the `--seed` value only), `experiment_seed`, `run_id`, `run_uuid`, `start_utc` and `end_utc`. Every run_id follows `sha256("E06b|seed|config|d9ecbef")[:16]`.
- **Shared identities:** the config, freeze record, contract, logging contract, M6D5c/M6D5d overlays, M6F-C evidence, source pin (FaceX-Zoo `16b793a7`), environment lock (`91416a20…`), LightCNN (`d0750746…`), split manifest and torch 2.12.1+cu130 are the same for all three runs.
- **Resolved config:** `resolved_config.yaml` differs only in `experiment_seed` and `run_dir`.
- **Run-start blocks:** binding, loader, population, precision, execution mode (V2) and fidelity are identical across seeds; seeding differs only by its seed value.
- **Population and loader:** the TRAIN population is 3720 CASIA+MSU spoof rows, 1240 live and 60 subjects. The loader is batch 240, workers 8, persistent_workers false, shuffle true, drop_last false.
- **Protocol:** every seed ran 200 epochs and 3200 contiguous optimizer steps of 15 × 240 + 120. The independent predictor matches 3200/3200 pair hashes and 200/200 epoch orders in each seed. All 744 000 native pairs are same-subject TRAIN live partners (print=0: 416 000, replay=1: 328 000). Each run was fresh, with no resume.
- **Numerics:** all losses are finite, `loss_cls` and `loss_pair` are active, the netCls gradient is nonzero and every official float storage is finite.
- **Checkpoints:** every seed has 21 checkpoint events, 63 official files and 0 runner_state sidecars, with exactly one selected final netG.
- **What differs:** only seed-derived randomness (shuffle, partner and noise draws) and the learned model and checkpoint bytes. The three metrics streams and the three final netG hashes are distinct.

## Firewall

The firewall evidence is a stat-only relatime atime comparison for each seed. In every run the 992 CASIA+MSU TEST faces and the 1088 VAL faces were all atime-sensitive and none changed. No SiW face changed in any run, and SiW TEST/VAL were fully sensitive throughout. There was no VAL pass. The upstream "test" block writes TRAIN visualizations only and is not benchmark TEST.

**VAL 0, TEST 0, SiW 0.**

## Run roots at family closure

A read-only host recheck confirms that `runs/m6/E06b/` holds exactly `seed_42`, `seed_1337` and `seed_2026`. For each seed the final netG hash and size, the metrics hash, the path/size/mtime and path/size/sha256 listing digests and the 197-file count all equal the per-seed evidence, and the listing stayed stable during the recheck.

## Semantics

- There is no best, preferred, dropped, replacement or fourth seed, and no best-seed field or selection.
- Future metrics use **MEAN_AND_STD_OVER_ALL_THREE_SEEDS**. No downstream metric is computed here, and the training losses in the JSON are descriptive only.
- Training speed is not an acceptance gate.
- **Not started:** synthetic bank, downstream FAS evaluation. The whole benchmark is not complete. E05, other methods and GPAT were not started.
