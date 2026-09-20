# M4 — Authoritative Pair Manifest Execution Report

> **SUPERSEDED IN PART by Owner Protocol Amendment A1 (2026-09-20).** Everything below about the
> **common** pair artifacts stands unchanged and their hashes are still current. What changed is the
> milestone outcome: the native manifests are no longer part of the main-track M4 dependency, both
> official sources were pinned, and two identity-free Track-A adaptations were frozen. **M4 is now
> COMPLETE under the amended main-track rule** — not under the originally specified rule, and the
> originally specified native manifests were never created. Current status:
> `M4_FAIR_TRACK_AMENDMENT_REPORT.md`. Nothing below was deleted or rewritten.

**Date:** 2026-09-20 · **Base commit:** `443614b10d5f54e34ff8451b54fde99ca6fcc8a7`
**Outcome:** common pair manifests **COMPLETE and frozen**; native pair manifests
**BLOCKED_BY_NATIVE_PAIR_CONSTRUCTION_SOURCE_GAP**.
**Milestone status: M4 = IN_PROGRESS**, phase `COMMON_PAIRS_COMPLETE_NATIVE_PAIR_SOURCE_BLOCKED`.
M4 is deliberately **not** marked COMPLETE: the acceptance gate requires the supported native
manifests to be either faithfully materialized or explicitly blocked, and a blocker means the
milestone is not done.

## 1. Frozen inputs (verified before execution)

| input | sha256 |
|---|---|
| `manifests/split_v1.parquet` | `fb9aeb369a124fc96ba855ef2ce269236c4a743fe960e73ab739412c9cb5092d` |
| `configs/frozen/pairs_v1.yaml` | `f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985` |
| `frozen_config_snapshot/.../pairs_v1.yaml` | byte-identical to the above |

Population recomputed from the manifest, not taken from the preflight report: 20,615 rows
(TRAIN 14,467 · VAL 3,121 · TEST 3,027); TRAIN spoof 2,520 + 1,200 + 5,118 = 8,838; VAL spoof
576 + 240 + 1,089 = 1,905. All match. Baseline 359 tests PASS.

## 2. Artifacts created

| artifact | rows | sha256 |
|---|---|---|
| `manifests/pair_train_stats_v1.json` | — | `a7ccabb0956f5121eabb5b4f7e85dfe49668469d8d3ee77cd88b24fc54bd3a50` |
| `manifests/pairs_train_v1.parquet` | 8,838 | `a5e4fdaef236f15730c7e3885b537e08e995faffbe654167fc940f44bbc75243` |
| `manifests/val_pairs_v1.parquet` | 1,905 | `84d124919a52cd8d84a89766f464a4dcde1aeaa7791218f6356813a40904f872` |

All written atomically (tmp → fsync → rename → directory fsync). No native manifest was created —
not even an empty one. No TEST pair manifest exists; the spec does not require one.

## 3. TRAIN statistics (Q-25)

Fitted **per dataset on TRAIN rows only**, population std, `ddof = 0`, float64:

| dataset | n TRAIN | mean (pitch, yaw, roll) | population std |
|---|---|---|---|
| casia_fasd | 3,360 | −0.044414, 0.003671, −0.015623 | 0.077981, 0.051887, 0.056370 |
| msu_mfsd | 1,600 | −0.014761, 0.025940, −0.015947 | 0.102100, 0.057406, 0.043483 |
| siwmv2 | 9,507 | −0.016929, 0.007284, −0.016149 | 0.175045, 0.283055, 0.095737 |

VAL reuses these; TEST never contributed and was never read. The zero-variance guard
(`std <= 1e-12` is a hard error) was armed and never triggered.

Canonical JSON: UTF-8, LF, sorted keys, 2-space indent, ASCII-escaped, NaN/Inf rejected, floats via
Python's shortest round-trip `repr` of the IEEE-754 double, exactly one trailing newline. The policy
is recorded inside the document as `json_policy`.

## 4. Audit

`M4_COMMON_PAIR_AUDIT.md` / `.json` — **PASS, 0 failures**, exhaustive over all 10,743 pairs. Every
§21 check, the CASIA/MSU different-subject rule, the SiW DEV-013 rule, the candidate rule, the
distance recomputation and the winner-is-the-minimum property were re-derived independently and
matched. CASIA `d_scale` unique values = `[0.0]` in both splits. 0 exact `d_pair` ties occurred, so
the lexical tie-break was never needed on real data (it remains implemented and unit-tested).

## 5. Determinism

`M4_DETERMINISM_REPORT.md` — **PASS**. Four independent fresh processes (canonical and shuffled
input order × `PYTHONHASHSEED` ∈ {0, 1, 424242}) reproduced all three files **byte-identically**,
with row lists, schemas, source→target membership and `pair_id` sequences compared separately.

**A real defect was caught here and is recorded, not hidden:** the first attempt failed the shuffled
runs because `fit_pose_stats` stacked TRAIN rows in arrival order, making the population std differ
in the last ulp and occasionally flipping a near-tied target. The fit now stacks rows in canonical
`sample_id` order. No Q-25 decision changed, and nothing was invalidated because the defect was
found *before* any manifest was frozen; the authoritative artifacts were regenerated with the
corrected fit.

## 6. Native manifests — blocked

`M4_NATIVE_PAIR_SOURCE_AUDIT.md` gives the full audit. In short: `third_party/registry.yaml` records
`pinned_commit: null` and `url_verification: UNVERIFIED` for both DSDG-NATIVE (E06b) and
DiffFAS-NATIVE (E07b); `methods/dsdg/` and `methods/difffas/` contain only `.gitkeep`; no checkout
exists anywhere on the machine. The source-of-truth hierarchy therefore has nothing at level 1, and
the open questions are all data-loader behaviour that papers do not specify — whether pairing is
materialized or sampled online, how many tuples exist, whether frames repeat, what seeds apply, and
for DiffFAS how the guide is drawn and what the official fallback is when no different-subject guide
exists.

Per the hard rule, nothing was invented: no first-lexical pick, no seeded random draw, no Cartesian
product, no one-to-one or cyclic matching, and the common-pair Q-24 ranking was not repurposed.

Dataset support is not the blocker — CASIA (35/35 identities) and MSU (25/25) fully satisfy the row
semantics, and SiW-Mv2 is `NOT_INSTANTIABLE_MISSING_SUBJECT_ID` with coverage 0 under Q-14.
Connectivity is not the blocker either; outbound HTTPS works. The blocker is that pinning these
repositories is an M6 first-setup action under the project's own registry and spec §8.1, so it needs
an owner decision rather than a silent unilateral pin.

## 7. Integrity

M2 and M3 artifacts untouched: `split_v1.parquet` still `fb9aeb36…`, `split_groups_v1.parquet`,
`m2_sample_accounting.parquet` and every frozen M2/M3 config unchanged; 0 files modified under the
M2 runtime root, the raw dataset roots or the model cache. M4 wrote only three small metadata files
(no frames, faces, geometry or identity data was duplicated) and no all-candidate dump.

## 8. What M4 still needs

Exactly one thing: an owner decision on the native pair construction source (authorize pinning now,
or defer both native manifests to M6). Everything else in the M4 acceptance gate passes.
