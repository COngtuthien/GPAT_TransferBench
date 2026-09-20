# M4 — Common Pair Manifest Audit

**Date:** 2026-09-20 · **Result: PASS — 0 failures** · Machine-readable: `M4_COMMON_PAIR_AUDIT.json`
Tool: `tools/m4_audit.py` (read-only; it re-derives every pair independently and compares).

| artifact | rows | sha256 |
|---|---|---|
| `manifests/pairs_train_v1.parquet` | 8,838 | `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243` |
| `manifests/val_pairs_v1.parquet` | 1,905 | `84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872` |
| `manifests/pair_train_stats_v1.json` | — | `a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50` |

Inputs: `split_v1.parquet` `fb9aeb36…`, `pairs_v1.yaml` `f243fdfa…`.
Schema signature `13061103a24366ca722c6cc256377ccce7dc7604882dc0b4a3f9202b3da6271b` (25 columns).

## 1. Structure (§21)

| check | TRAIN | VAL |
|---|---|---|
| rows == spoof-source count | 8,838 == 8,838 | 1,905 == 1,905 |
| unique `pair_id` == rows | PASS | PASS |
| unique `source_spoof_id` == rows | PASS | PASS |
| source set == split's spoof set (no missing, no extra) | PASS | PASS |
| written manifest == independent rebuild (all 25 columns, every row) | PASS | PASS |
| canonical row order (`dataset`, `source_spoof_id`) | PASS | PASS |
| `pair_id` sequence == canonical assignment (`PTR`/`PVA`, from 1) | PASS | PASS |
| every source SPOOF, every target LIVE | PASS | PASS |
| same dataset, same split | PASS | PASS |
| both endpoints M2 COMPLETE | PASS | PASS |
| TEST references | **0** | **0** |
| `seed` == 20260814 everywhere | PASS | PASS |
| `split_manifest_sha256` / `pairs_config_sha256` correct on every row | PASS | PASS |
| `source_sha256` / `target_sha256` match the split lineage | PASS | PASS |
| `attack_macro` is the source's | PASS | PASS |
| all distances finite and ≥ 0 | PASS | PASS |
| `d_pair` == 0.50·d_pose + 0.30·d_scale + 0.20·d_luma, **exactly** | PASS | PASS |
| every component recomputes exactly from the frozen primitives | PASS | PASS |
| winner is the minimum over the evaluated candidate set (exhaustive, all 10,743 sources) | PASS | PASS |
| `candidate_count_evaluated` | 64 for every row | 64 for every row |
| `candidate_count_eligible` | min 384, max 4,389 | min 64, max 944 |
| exact `d_pair` ties at the top of the evaluated set | 0 | 0 |

The lexical tie-break is implemented and unit-tested, but no real pair needed it: no source produced
two evaluated candidates with bit-identical `d_pair`.

## 2. Eligibility (§22, §23)

**CASIA-FASD and MSU-MFSD — different subject:** every pair has non-null `source_subject` and
`target_subject` and `source_subject != target_subject`. **PASS** (0 violations, 0 nulls).

**SiW-Mv2 — DEV-013:**

| constraint | result |
|---|---|
| `source_subject` is null | **PASS** (all rows) |
| `target_subject` is null | **PASS** (all rows) |
| `source_video_id != target_video_id` | **PASS** (all rows) |
| `source_content_group_id != target_content_group_id` | **PASS** (all rows) |

No SiW pair is described as different subject or different person anywhere in the manifests or the
audit outputs. DEV-013 is a different-video / different-exact-content guarantee and nothing more.

## 3. Candidate selection (§9, §26)

Every source in both splits had more than 64 eligible targets **except** MSU VAL, where every
source has exactly 64 — there the cap is reached but the ranking is not selective, because all
eligible candidates are kept. Everywhere else the frozen Q-24 ranking chose 64 of a larger pool.

Deterministic evidence for 25 sources per dataset × split (150 rows,
`M4_CANDIDATE_SELECTION_AUDIT.csv`): eligible-ID-list SHA-256, selected-64-ID-list SHA-256, winner
id, winner `d_pair`. The winner lies inside the selected set for every audited row. Full
64-candidate dumps for all 10,743 sources are deliberately not persisted (§26, §44).

## 4. Distances (§25)

`M4_PAIR_DISTANCE_DISTRIBUTION.csv` holds min / median / p95 / max of all four quantities per
dataset × split. Medians:

| dataset | split | d_pose | d_scale | d_luma | d_pair |
|---|---|---|---|---|---|
| casia_fasd | TRAIN | 0.5554 | **0.0000** | 0.0835 | 0.2988 |
| casia_fasd | VAL | 0.7651 | **0.0000** | 0.0819 | 0.4030 |
| msu_mfsd | TRAIN | 0.6255 | 0.2729 | 0.0780 | 0.4289 |
| msu_mfsd | VAL | 1.0942 | 0.3248 | 0.0977 | 0.6957 |
| siwmv2 | TRAIN | 0.5073 | 0.1971 | 0.0631 | 0.3491 |
| siwmv2 | VAL | 0.4967 | 0.1905 | 0.0669 | 0.3393 |

**CASIA `d_scale` unique values = `[0.0]` in both splits — exactly zero for all 3,096 CASIA pairs**,
as DEV-018 requires. The 0.50/0.20 weights are not renormalised, so the 0.30 term is simply inert
for CASIA and its pairs are ranked on pose and luminance alone. This is a disclosed limitation, not
a tuning choice, and nothing in this section was allowed to change a formula.

## 5. Target reuse (§24 — diagnostic only)

| split | dataset | pairs | live targets available | unique used | max reuse | mean reuse |
|---|---|---|---|---|---|---|
| TRAIN | casia_fasd | 2,520 | 840 | 681 | 13 | 3.70 |
| TRAIN | msu_mfsd | 1,200 | 400 | 272 | 23 | 4.41 |
| TRAIN | siwmv2 | 5,118 | 4,389 | 2,877 | 7 | 1.78 |
| VAL | casia_fasd | 576 | 192 | 130 | 25 | 4.43 |
| VAL | msu_mfsd | 240 | 80 | 21 | 29 | 11.43 |
| VAL | siwmv2 | 1,089 | 944 | 597 | 7 | 1.82 |

Reuse is permitted by the frozen contract and **no pair was altered to reduce it**. MSU VAL is the
most concentrated cell (21 distinct targets, max reuse 29), which follows directly from its small
live pool (80) and its exactly-64 eligible set. Full histograms are in
`M4_PAIR_TARGET_REUSE.csv`.
