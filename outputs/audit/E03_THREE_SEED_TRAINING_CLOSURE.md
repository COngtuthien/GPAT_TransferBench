# E03 STDN: three-seed scientific training closure

**Status:** `SCIENTIFIC_TRAINING_COMPLETED_3_OF_3_SEEDS`, 74/74 gates passing (`E03_THREE_SEED_TRAINING_CLOSURE.json`).

Training authority for all three seeds: **`d547fc9f8932a273d719ae589aade85a9d989b85`**. The per-seed closure commits and this closure are evidence only.

Execution fidelity: `FAITHFUL_OFFICIAL_CORE_WITH_TRACK_A_CONTROLLED_DATA_ADAPTATION_AND_RUNTIME_COMPATIBILITY` (not a fully native reproduction).

| seed | run_id | G / D | global_step | ckpt-50 data sha256 | run-root listing | TEST | closure commit |
|---|---|---|---|---|---|---|---|
| 42 | `a565b76bc7349532` | 100000 / 50000 | 150000 | `5b337a63db63e405bf35bceefdacc101060cb7d1625b5a4d2d389b963e311753` | `a7bed081022c09ab…` | 0 | `b95c127` |
| 1337 | `e8c85cc378e778f5` | 100000 / 50000 | 150000 | `316c94c066c32314c6819bc105d0043e1ddc7b27329afd48ddcfb8bc8d181551` | `ad748419b53d4c07…` | 0 | `5cdb65d` |
| 2026 | `237887eebab18637` | 100000 / 50000 | 150000 | `9dbd463cb3eb56c9ecc578dd275e0c2507a1e6d5f8b9b4876406d5011bd52cb3` | `f42708341b91e6b6…` | 0 | `15bfc84` |

## What is common and what differs

- Common to all three seeds:
  - frozen config `08dde851…` and every identity;
  - the official graph structure and session configuration;
  - the feed pack `c8c9850e…` and the environment re-check;
  - the driver ruling (580.178.04 vs lock 595.84, version only);
  - 50 × (2000 TRAIN, ckpt-e, 500 VAL);
  - G 100000 / D 50000 / global_step 150000.
- Differs by design only:
  - the seed, the run_id and the timestamps;
  - the TRAIN seed = experiment seed and the VAL seed from ruling V;
  - the resulting streams and learned weights.
- Every seed's TRAIN/VAL streams equal an independent shuffle replay and its flips equal a numpy prediction.
- ckpt-50 is the final checkpoint of each seed by `OFFICIAL_LATEST_FINAL_CKPT_50`. There is no cross-seed choice and no best seed, and no seed was dropped or replaced.
- TEST: 0 accesses in every seed.

## Disclosures

- seed1337: `pane_log_keyboard_echo`: the last pane-log line is the runner completion record preceded only by caret-notation cursor-key echo (^[[A..^[[D) from keys pressed in an attached tmux client; it is tty echo, not runner output, and no E03 production or pinned-source code path reads stdin; fail-pattern hits are all zero
- seed2026: `post_completion_reboot`: the run completed at 2026-10-28T18:52:21Z (exit marker 0, runner completion record, tmux pane scope stopped 2026-10-28T18:52:21Z); the host was then cleanly shut down (wtmp shutdown record, run-boot journal ends 2026-10-29T02:44:03Z) and rebooted (kernel boot 2026-10-29T02:44:20Z), which removed the tmux server, so B.process_exited_zero and J.single_e03_session pass by owner-approved durable post-completion-reboot evidence instead of live pane state; this is an audit-tool amendment only and changes neither the trained model nor the scientific procedure; the first 86/88 NEEDS_REVIEW build is kept at outputs/audit/history/E03_SEED2026_AUDIT1_NEEDS_REVIEW/E03_SEED2026_COMPLETION.json
- Seed tool amendment 1, `seed1337_pane_log_keyboard_echo` (`b95c127` → `5cdb65d`): B.pane_log_clean strips only a leading ^[[A..^[[D cursor-key tty echo prefix from the last pane-log line and discloses it (seed1337: keys pressed in an attached tmux client); the gate set is unchanged and seed42 re-derives byte-identically with both tool versions.
- Seed tool amendment 2, `seed2026_post_completion_reboot_evidence` (`5cdb65d` → `15bfc84`): owner ruling 2026-10-29 (OWNER_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE_SUBSTITUTION): B.process_exited_zero and J.single_e03_session may pass as PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE on durable evidence when a clean host reboot after the run completed removed the tmux server (seed2026); the live tmux path and every other gate are unchanged and seed42/seed1337 re-derive byte-identically with all three tool versions.
- Gates passed by the owner-approved post-completion-reboot evidence: seed2026 `B.process_exited_zero`, `J.single_e03_session`. The first seed2026 build (86/88, NEEDS_REVIEW) is kept unchanged at `outputs/audit/history/E03_SEED2026_AUDIT1_NEEDS_REVIEW/E03_SEED2026_COMPLETION.json`. At this family recheck, the E03 tmux panes are gone because the host is still in the boot that followed the verified reboot (`H.host_final_state`: `PASS_BY_APPROVED_POST_COMPLETION_REBOOT_EVIDENCE`).

## Aggregation

The frozen primary result rule is `MEAN_AND_STD_OVER_ALL_THREE_SEEDS`. It applies to downstream evaluation metrics, and none has been computed. The losses below are descriptive only: VAL is diagnostics, and nothing here is aggregated or ranked.

| seed | TRAIN epoch 50 (a / d / g) | VAL epoch 50 (a / d / g) |
|---|---|---|
| 42 | 5.594 / 0.4138 / 32.09 | 6.957 / 0.5904 / 47.51 |
| 1337 | 5.513 / 0.4228 / 31.69 | 6.935 / 0.5529 / 49.93 |
| 2026 | 5.286 / 0.4149 / 29.65 | 6.741 / 0.5958 / 45.55 |

## Not completed

The E03 synthetic bank, downstream FAS evaluation and the whole benchmark have not started. This closes STDN training only.
