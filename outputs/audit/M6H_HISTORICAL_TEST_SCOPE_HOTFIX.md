# M6H — Historical test-scoping hotfix: no-A9 assertions read their own milestone trees

| Field | Value |
|---|---|
| Milestone | M6H, classification `M6H_HISTORICAL_TEST_SCOPE_HOTFIX` (a test-scoping correction, not a scientific amendment) |
| Authority | `e936ec252de20ac9dd28bcc8bff0282b5b65d2da` (M6F-A), branch `m6-baselines` |
| Correction | `CURRENT_HEAD_TREE -> OWN_HISTORICAL_MILESTONE_TREE` |
| Verifier | `tools/m6h_historical_test_scope_preflight.py` (stdlib only) |

## Defect

The two affected tests are:
- `tests/test_m6d6ir_e07c_checkpoint_retention.py::TestM6D6iRDecision::test_06_no_a9`
- `tests/test_m6d6jr_e07c_science_launch_decision.py::TestM6D6jRDecision::test_14_no_a9`

Both asserted "no A9 path" using `git ls-files docs configs`, which lists the **current HEAD** index. That contradicts
the tests' own design: they read everything else at the commit that added them.

- M6A9 (`8e9ccc3`) legitimately tracked the Amendment A9 document, so both tests have failed since that commit.
- This was reproduced at authority `e936ec2` in a clean detached worktree.
- The defect **pre-existed M6F-B**, and M6F-B did not cause it.

## Correction

Each test file gains one helper, `tracked_at_m6d6ir` or `tracked_at_m6d6jr`. It returns
`git ls-tree -r --name-only <commit that added the test> <paths>`, falling back to the candidate index only if the test
was never committed. The no-A9 line now uses that helper.

Nothing else changes. The tests still assert all of the following:
- the records' `amendment_created = false` and `a9_created = false`;
- M6D6iR (`39508719`) and M6D6jR (`8358d8b6`) did not create A9;
- their own trees contain no A9 path;
- the authority bindings are unchanged.

The preflight proves that each file differs from the authority by exactly the helper and the one assertion line.

`tests/test_m6h_historical_test_scope.py` proves two things:
- **(A)** each historical tree has no A9, and the historical test passes;
- **(B)** a later tree contains the legitimate A9 (document and record) without invalidating the historical assertion.

## Unchanged

- Scientific state, A9, E07c, E06b and every record or document are unchanged.
- The parked M6F-B candidate is not mixed in.
- No training, GPU use, model run, data read, image read or TEST access.

**M6_CLOSED = false. M7 HAS NOT STARTED.**
