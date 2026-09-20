# M4 — Deterministic Rerun Report

**Date:** 2026-09-20 · **Result: PASS** · Machine-readable: `M4_DETERMINISM_COMPARE.json`
Tool: `tools/m4_determinism.py`

## 1. Runs

Each run is an **independent fresh Python process** that rebuilds all three artifacts from the
frozen inputs into its own clean temporary directory. Nothing is written to `manifests/`.

| run | input order | `PYTHONHASHSEED` | stats bytes | TRAIN bytes | VAL bytes | rows identical | schema identical | membership identical | `pair_id` identical |
|---|---|---|---|---|---|---|---|---|---|
| A | canonical | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| B | shuffled (seed 20260814) | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C | canonical | 424242 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| D | shuffled (seed 7) | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Frozen hashes reproduced by every run:

```
pair_train_stats_v1.json  a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50
pairs_train_v1.parquet    a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243
val_pairs_v1.parquet      84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872
```

Byte equality is checked on the files themselves; row lists, Parquet schemas, source→target
membership and the `pair_id` sequence are compared separately, so a byte match cannot hide a
schema-level coincidence.

## 2. A real defect this rerun caught

The **first** determinism attempt failed runs B and D: shuffling the input population changed pair
membership in both splits.

- **Cause:** `fit_pose_stats` stacked the TRAIN rows in arrival order. Floating-point accumulation
  is not associative, so the **population standard deviation** differed in the last ulp between
  orderings (the mean happened to agree; the std did not). That perturbation propagates through the
  per-axis z-scores into `d_pose`, and a near-tie in `d_pair` can then resolve to a different
  target.
- **Fix:** `fit_pose_stats` now stacks rows in canonical `sample_id` order, so the fit is a function
  of the TRAIN row *set* and never of its arrival order. This implements the frozen
  `determinism_contract.invariant_to: [input_row_order, …]`; it does not change any Q-25 decision
  (per-dataset TRAIN z-score, population std, `ddof=0`, L2 are untouched).
- **Blast radius:** none. The defect was found *before* any manifest was frozen — the authoritative
  artifacts were regenerated with the corrected fit and are the only ones ever recorded. Verified
  directly: three different shuffles now produce identical statistics.

This is exactly what the rerun gate exists for, and it is recorded rather than quietly patched.

## 3. Scope of reproducibility

`pair_train_stats_v1.json` embeds the SHA-256 of the two modules that produced it
(`gpatbench/pairs/common.py`, `gpatbench/pairs/execute.py`), so byte-exact reproduction of the
**stats file** requires those identical module bytes. The pose statistics themselves depend only on
the TRAIN row set. The two Parquet manifests carry no module hash and reproduce from the frozen
inputs alone. This is deliberate: a frozen scientific artifact should pin the code that made it, and
`reproduction_note` inside the JSON says so.

## 4. Native manifests

No native manifest was materialized (see `M4_NATIVE_PAIR_SOURCE_AUDIT.md`), so there is nothing to
rerun. This is reported as a blocker, not as a determinism pass.
