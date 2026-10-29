# Broad test-suite reconciliation (2026-10-29)

**The repository-wide suite is not all-green.** 63 failures/errors remain on the laptop. None is in the E03 suite (91/91 pass) and none comes from a scientific result. No test and no scientific file was changed.

- The `Data` partition was identified by its filesystem label and mounted **read-only** with udisks; that fixed exactly 7 drive-dependent tests, with no new failures (70 → 63).
- The scipy- and torch-dependent modules were re-run on the GPU host under existing environments (test files byte-identical at `d547fc9`): every algorithmic test passes, and only milestone-state assertions remain.
- No environment on either machine has `zstandard`, so those 23 tests cannot run without an install (out of scope).

| category | count |
|---|---|
| `ENV_MISSING_PACKAGE_scipy` | 24 |
| `ENV_MISSING_PACKAGE_torch` | 3 |
| `ENV_MISSING_PACKAGE_zstandard` | 23 |
| `GPAT_M7D1_FLOAT_REPR_REDERIVATION_PAUSED_SCOPE` | 2 |
| `HISTORICAL_MILESTONE_STATE_ASSERTION` | 10 |
| `NON_AUTHORITATIVE_GITIGNORED_EXPLORATORY_ASSET` | 1 |

`NON_AUTHORITATIVE_GITIGNORED_EXPLORATORY_ASSET`: `test_smoke_shards_pass_the_same_audit` reads the git-ignored `outputs/exploratory/m2a_smoke/runC/logit_shards` that exists only in the laptop working tree. It skips when the directory is absent. It is not part of any mandatory qualification contract. The exploratory data was not touched.

`HISTORICAL_MILESTONE_STATE_ASSERTION`: tests written at M0–M5 that assert a later milestone has not started or that no later artifact exists (for example `STAGE_STATE` `NOT_STARTED`). They have been false since those milestones ran and need a separate history-scope reconciliation.

Commands, exit codes and every failing test with its error line are in `BASELINE_EXEC_BROAD_TEST_SUITE_RECONCILIATION.json`.
