# E04 Physics-STD: three-seed scientific training closure

**Status:** `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`, 74/74 gates passing (`E04_THREE_SEED_TRAINING_CLOSURE.json`).

Training authority for all three seeds: **`18ba00e94813f005876c8d6235b0b848226b2c1d`** (production qualification `b0f0ffd`, D12 ruling `c60481b`, criteria `9f1a2471…`). The per-seed closure commits and this closure are evidence only.

Execution fidelity: `CONTROLLED_ADAPTATION` (reporting label "Physics-STD (controlled geometry/depth reconstruction)").

| seed | run_id | iterations | G / D | final iteration_150000 data sha256 | run-root listing | TEST | VAL | closure commit |
|---|---|---|---|---|---|---|---|---|
| 42 | `40befa6004aee85c` | 150000 | 300000 / 150000 | `55891ee1fd3a46ea3c451680d9a1a67998ede8246d754b6ea92aaf4bc45d71c6` | `c50d18e832557c70…` | 0 | 0 | `ea533db` |
| 1337 | `a42fdd2a2accaf0a` | 150000 | 300000 / 150000 | `fffa5f252c6dde2e1c5a49b380cec0bffe5d13e27ab711cbb3983188897ae0c5` | `6a8f471effd41d0e…` | 0 | 0 | `6da6719` |
| 2026 | `e7100eb9a5ab0537` | 150000 | 300000 / 150000 | `089c5e01da28c086e3660d4ba1ae4248cbf80cae907a163a0282f4af6812e0b2` | `3503f2d15e21a7c2…` | 0 | 0 | `0e3f1d1` |

**Family totals:** 450000 iterations, 900000 G and 450000 D optimizer applications, 30 checkpoint events (90 files; inventory `E04_THREE_SEED_CHECKPOINT_INVENTORY.csv`).

## What is common and what differs

- Common to all three seeds:
  - frozen config `41861794…` and every identity (pre-pass, feed pack, pairs, split, ruling, lock);
  - the qualified graph structure and session configuration;
  - the environment re-check and the D9 driver ruling (580.178.04 vs lock 595.84, version only);
  - 150000 iterations of G → D → G, batch 8 (4 live targets + 4 spoof sources), 10 checkpoints every 15000 iterations.
- Differs by design only:
  - the seed, the run_id, the timestamps and the wrapper command;
  - the derived PAIR_ORDER / HARD_SAMPLE seeds and the TF graph seed;
  - the resulting streams and learned weights.
- Disclosed, not part of the E04 lock: seed 42 ran on host kernel 7.0.0-38, seeds 1337 and 2026 on 7.0.0-43.
- Each seed's pair and hard streams equal an independent record-by-record numpy replay.
- iteration_150000 is the final checkpoint of each seed by `BASELINE_FINAL_STATE_V1`. There is no cross-seed choice and no best seed, and no seed was dropped, replaced or retrained.
- TEST: 0 accesses in every seed. VAL: 0 atime changes in every seed (all VAL faces atime-sensitive).
- D12: the runs are independent GPU executions that are not bitwise reproducible; none is compared bitwise to another.

## Seed-tool amendments (disclosed)

1. `seed1337_uninterrupted_boot_branch` (`ea533db` → `6da6719`): B.no_reboot_during_run also accepts an uninterrupted boot (no wtmp boot after launch, run boot "still running", /proc btime before launch and equal to that record within 120 s, same-boot_id journal boot spanning launch..end); a reboot, shutdown or journal boot inside the window still fails and seed42 re-derives byte-identically.
2. `seed2026_finished_sibling_session` (`6da6719` → `0e3f1d1`): J.no_other_e04_runner accepts a leftover e04_* tmux session only when it is another registered seed's session with one dead pane (status 0), exit marker "0" and an end marker before this seed's launch (the finished seed-1337 session); no runner process is still required, and seed42/seed1337 re-derive byte-identically.

## Host recheck

Read-only at 2026-11-24T18:34:07Z: all 90 checkpoint files, metrics.jsonl and the run-root listing digests equal the per-seed evidence. The three exit markers are `0`, no E04 or other method process is running, and the GPU checkout is at `18ba00e`, clean. The seed-42 tmux session is absent because the host rebooted after that run had ended. The seed-1337 and seed-2026 panes are dead with status 0.

## Descriptive training losses

Block means of 1500 iterations, for orientation only. Nothing here is aggregated, ranked or used for a choice.

| seed | step1/loss_G_step1 first → last | step1/L_depth first → last | step2/loss_D first → last | step3/loss_G_step3 first → last |
|---|---|---|---|---|
| 42 | 2.335e+04 → 8848 | 0.1971 → 0.1142 | 529.1 → 829.2 | 1.169e+05 → 1.955e+04 |
| 1337 | 2.848e+04 → 5879 | 0.1982 → 0.1127 | 486.9 → 807.8 | 1.219e+05 → 1.026e+04 |
| 2026 | 2.566e+04 → 6863 | 0.1952 → 0.1141 | 512.1 → 795.8 | 1.177e+05 → 8767 |

## Reproduction

`E04_THREE_SEED_REPRODUCTION_MANIFEST.json` records, per seed, the source commit, config, identities, environment, RNG derivation, exact launch environment and command, window, exit status, outputs and audit record. Checkpoint bytes stay in the run roots on the GPU host; git holds their hashes only.

## Not completed

The E04 synthetic bank, downstream FAS evaluation and the whole benchmark have not started. This closes Physics-STD training only.
