# M4 — Track-A Identity-Free Determinism Report

**Date:** 2026-09-20 · **Result: PASS** · Machine-readable: `M4_IDFREE_DETERMINISM_COMPARE.json`
Tool: `tools/m4_idfree_determinism.py`

Four independent **fresh Python processes** rebuild both Track-A relations from the frozen inputs,
the DiffFAS manifest into its own clean directory. Nothing is written to `manifests/`.

| run | input order | `PYTHONHASHSEED` | DiffFAS bytes | rows | schema | membership | `track_pair_id` | DSDG relation |
|---|---|---|---|---|---|---|---|---|
| A | canonical | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| B | shuffled (seed 20260814) | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C | canonical | 424242 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| D | shuffled (seed 7) | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Frozen values reproduced by every run:

```
manifests/difffas_bin_idfree_train_v1.parquet
    0d4c0ab435a258be51577aec17d9ecea27785354c900d0f7ab863d2bb2924fd6
DSDG-BIN-IDFREE adapter relation (pair_id|dataset|source_spoof_id|target_live_id)
    403469859598fce129f4ac8dcacbfa9bcdad4d58065dbc00177c90dc92c4f142
```

The shuffled runs permute both the TRAIN row list and the common pair list before building, so the
result is proven independent of arrival order rather than assumed to be.

## Why DSDG has a relation hash and not a manifest

DSDG-BIN-IDFREE reuses `manifests/pairs_train_v1.parquet` directly — a frozen, explicit decision in
`configs/frozen/dsdg_bin_idfree_v1.yaml`. A separate manifest would have been a redundant projection
of the common one plus three constants (`binary_spoof_label = 1`, `lambda_pair = 0`, the adaptation
config hash), and a second copy could drift from the frozen original. What is hashed and rerun here
is the **identity-free adapter projection** of that relation, which is what a DSDG adapter would
actually consume: it contains no subject column at all.

## Determinism of the guide rule

The DiffFAS style guide is chosen by raw-byte SHA-256 ranking (source digest carried as 32 raw
bytes, never hex), with lexical `guide_spoof_sample_id` as a defensive tie-break, over a pool sorted
in canonical lexical order. No Python `random`, no NumPy random, no filesystem order, no
`PYTHONHASHSEED` dependence — which the C and D runs confirm empirically.
