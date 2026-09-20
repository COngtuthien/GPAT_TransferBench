# M4 — Native Pair Coverage (DSDG / DiffFAS)

**Date:** 2026-09-20 · Machine-readable table: `M4_NATIVE_PAIR_COVERAGE.csv`
Source-of-truth audit: `M4_NATIVE_PAIR_SOURCE_AUDIT.md`

Measured on `manifests/split_v1.parquet` TRAIN rows only. **No native manifest was materialized in
M4**: the dataset scope is settled (Q-28/Q-29) but the official construction is not pinned, so
every supported cell is `SUPPORTED_BUT_BLOCKED_BY_SOURCE_GAP`.

## Coverage

| method | dataset | status | TRAIN rows | live / spoof | identities | with live | with spoof | **eligible (both)** | coverage | style_ids | rows written |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DSDG | casia_fasd | SUPPORTED_BUT_BLOCKED_BY_SOURCE_GAP | 3,360 | 840 / 2,520 | 35 | 35 | 35 | **35** | 100.0 % | 9 | 0 |
| DSDG | msu_mfsd | SUPPORTED_BUT_BLOCKED_BY_SOURCE_GAP | 1,600 | 400 / 1,200 | 25 | 25 | 25 | **25** | 100.0 % | 3 | 0 |
| DSDG | siwmv2 | NOT_INSTANTIABLE_MISSING_SUBJECT_ID | 9,507 | 4,389 / 5,118 | 0 | 0 | 0 | **0** | 0 % | 14 | 0 |
| DiffFAS | casia_fasd | SUPPORTED_BUT_BLOCKED_BY_SOURCE_GAP | 3,360 | 840 / 2,520 | 35 | 35 | 35 | **35** | 100.0 % | 9 | 0 |
| DiffFAS | msu_mfsd | SUPPORTED_BUT_BLOCKED_BY_SOURCE_GAP | 1,600 | 400 / 1,200 | 25 | 25 | 25 | **25** | 100.0 % | 3 | 0 |
| DiffFAS | siwmv2 | NOT_INSTANTIABLE_MISSING_SUBJECT_ID | 9,507 | 4,389 / 5,118 | 0 | 0 | 0 | **0** | 0 % | 14 | 0 |

"identities" counts distinct non-null `subject_id_global`; SiW-Mv2 has none at all, so the column is
0 rather than small.

## Reasons

- **CASIA-FASD, MSU-MFSD — blocked, not unsupported.** Both datasets fully satisfy the row
  semantics: every TRAIN identity carries both live and spoof samples, so a same-identity live/spoof
  pair exists for each. What is missing is the *official construction*: how many rows, which frames,
  and whether the official loader enumerates a finite set at all. `third_party/registry.yaml`
  records `pinned_commit: null` and `url_verification: UNVERIFIED` for both E06b (DSDG-NATIVE) and
  E07b (DiffFAS-NATIVE), and `methods/dsdg/` and `methods/difffas/` contain only `.gitkeep`.
- **SiW-Mv2 — not instantiable, permanently under Q-14.** No trustworthy subject identity exists in
  the local copy, so no same-identity pair can be formed. This is a property of the data, not a
  policy preference, and it would not change if the repositories were pinned tomorrow.

## Identity-fabrication guards

For every cell above:

- no pseudo or invented SiW subject id exists anywhere;
- `content_group_id` is **not** used as person identity;
- `video_id` is **not** used as person identity;
- **DEV-013 is not used as same-identity evidence** — it resolves the common source→target rule for
  SiW (different video AND different exact-content group) and is never a same-person claim;
- SiW absence is represented by these coverage rows, **never** by manifest rows.

`DIFFFAS-BIN` does not change any of this: collapsing `style_id` to a binary label preserves the
same-identity reconstruction structure, and the missing information for SiW is identity, not style.
Likewise `DSDG-BIN` collapses only the spoof-type target.
