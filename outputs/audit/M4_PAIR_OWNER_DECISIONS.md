# M4 — Common Pair Owner Decisions (Q-24 … Q-27, and Q-28/Q-29 manifest scope)

> **Revised 2026-09-20 — `PRE_EXECUTION_CONTRACT_IMPLEMENTATION_CORRECTION`.** The Q-24 candidate
> preimage is now stated in unambiguous byte terms and Q-28/Q-29 are resolved for M4 native manifest
> scope. The owner's Q-24 contract did not change and `gpatbench/pairs/common.py` needed no change:
> it already hashed the RAW 32-byte source digest (0 of 240 candidate-64 sets changed, measured
> against the code committed at `e4d167b`). The config SHA256 below is therefore the
> **pre-correction** value, retained as history and marked
> `SUPERSEDED_BY_Q24_PREIMAGE_CORRECTION`; the current value is
> `f243fdfaab2904b41aea3a09bbb3aa4bf5ddb55221db0cfaae328cab6b918985`. Full record:
> `M4_Q24_PREIMAGE_CORRECTION.md`. Q-25, Q-26, Q-27 and DEV-018 were not reopened.

Frozen contract: `configs/frozen/pairs_v1.yaml` (sha256 `16ff68036a9ed3a315df6944697394359cb2ea53e0b5623721d036a3cf170398`, SUPERSEDED_BY_Q24_PREIMAGE_CORRECTION), implemented in
`gpatbench/pairs/common.py`, tested in `tests/test_m4_pair_preflight.py`. The pre-owner-decision
proposal is preserved unchanged at `configs/proposed/pairs_v1.proposed.yaml`.

These are **benchmark-design / controlled-adaptation decisions** taken where spec §6 was silent.
They are not literature-derived facts and must not be presented as such. The spec's own settled
elements — splits, one pair per spoof source, the 64 cap, the 0.50/0.30/0.20 weights, minimum
`d_pair`, the lexical tie-break and the seed — are unchanged and were never tuned.

**Still pre-M4:** no pair manifest and no `pair_train_stats_v1.json` exist. M4 is NOT_STARTED.

## Q-24 — deterministic 64-candidate selection · RESOLVED_BY_OWNER_HASH_RANKING

Eligibility is computed first under the already-frozen dataset rules, then the eligible set is put in
lexical `target_sample_id` order **before any hashing**. If at most 64 remain, all are evaluated;
otherwise exactly 64 are chosen by a two-stage SHA-256:

```
source_seed_digest   = SHA256(UTF8("gpatbench.pair.source_seed.v1|" + source_sample_id + "|" + split_seed))
candidate_rank_digest = SHA256(source_seed_digest_bytes || UTF8("|gpatbench.pair.candidate.v1|" + target_sample_id))
```

Candidates rank by `candidate_rank_digest` read as an **unsigned big-endian 256-bit integer**, with
lexical `target_sample_id` as a defensive collision tie-break; the first 64 are kept. No PRNG, no
sampling with replacement, and no dependence on input order, filesystem order, `hash()`,
`PYTHONHASHSEED`, worker count or wall-clock time.

`d_pair` is then computed **only over the selected set**; the winner is the minimum, exact ties break
on lexical `target_sample_id`, and there is no fuzzy epsilon window. All arithmetic is float64.

This matters more than it looks: every source in every dataset and both splits has at least 64
eligible targets, so the cap binds for **every** pair.

### pair_id

`pair_id` is load-bearing downstream — spec §8.1 derives FAS-Aug operator randomness from
`SHA256(pair_id + global_seed)` — so it is frozen explicitly rather than left as bookkeeping.
Canonical source order inside each manifest is `dataset` ascending then `source_spoof_id` ascending;
TRAIN ids are `PTR000001…` and VAL ids `PVA000001…`, numbered from 1 independently per manifest so
the two can never collide.

**Pair membership is decided first; `pair_id` is assigned only afterwards**, so candidate selection
can never depend on it. That ordering is what prevents a circular dependence.

## Q-25 — pose · RESOLVED_BY_OWNER_DATASET_TRAIN_ZSCORE_L2

Frozen FaceXFormer pitch/yaw/roll from M2 (never rerun). Statistics are fitted **per dataset**, on
M2 COMPLETE **TRAIN rows only** — CASIA, MSU and SiW are never pooled, and VAL and TEST never
contribute. Each component is normalised independently with the population mean and population
standard deviation (`ddof = 0`) in float64.

```
z = (value − mean_dataset_train) / std_dataset_train
d_pose = sqrt(Δz_pitch² + Δz_yaw² + Δz_roll²)      # Euclidean L2, no division by sqrt(3)
```

A fitted `std <= 1e-12` is a **hard error** — never set to 1, never dropped, never given a silent
epsilon. The real data has non-zero variance on every axis, so this is an integrity guard rather
than a live branch. VAL pairing reuses that dataset's TRAIN statistics, so validation never defines
its own representation.

Fitted on the real split:

| dataset | TRAIN rows | mean (pitch, yaw, roll) | population std |
|---|---|---|---|
| casia_fasd | 3,360 | [-0.0444, 0.0037, -0.0156] | [0.078, 0.0519, 0.0564] |
| msu_mfsd | 1,600 | [-0.0148, 0.0259, -0.0159] | [0.1021, 0.0574, 0.0435] |
| siwmv2 | 9,507 | [-0.0169, 0.0073, -0.0161] | [0.175, 0.2831, 0.0957] |

The per-dataset spread differs substantially (SiW yaw std 0.2831
vs CASIA 0.0519), which is exactly why pooling was
rejected.

## Q-26 — scale · RESOLVED_BY_OWNER_NORMALIZED_VISIBLE_BBOX_LOGRATIO

The scientific quantity is a **dimensionless visible face-box area fraction**. For MSU and SiW the
selected SCRFD box from the frozen M2 metadata is clipped to the original frame and divided by the
frame area:

```
cx1 = max(0, min(W, x1)) ;  cy1 = max(0, min(H, y1))
cx2 = max(0, min(W, x2)) ;  cy2 = max(0, min(H, y2))
face_area_fraction = ((cx2 − cx1) · (cy2 − cy1)) / (W · H)
d_scale = |ln f_target − ln f_source|                      # natural log, no epsilon, no clipping
```

The box is the **original detector box**, before the 1.25× expansion, the zero padding and the 256
resize — the requested crop square and the canonical 256 area are explicitly not used. Normalising by
frame area removes raw-resolution dependence, so MSU (640×480) and SiW (1920×1080) become comparable.
`W > 0`, `H > 0`, positive visible width and height and `0 < f <= 1` are all required; anything else
is a hard error rather than silently repaired geometry.

### CASIA controlled adaptation — DEV-018

CASIA has **no SCRFD box at all** (0 of 4,800 rows) under the approved DEV-011 pre-cropped route.
SCRFD is not rerun, no bbox is invented and no pseudo-box is derived from landmarks. The owner fixed

```
face_area_fraction = 1.0   for every usable CASIA sample  ⇒  d_scale = 0.0 exactly
```

and the 0.50/0.20 weights are deliberately **not** renormalised, so the frozen `d_pair` formula is
unchanged. The honest consequence: for CASIA the scale term carries no discriminative information,
and candidate ranking is governed by pose and luminance alone with the 0.30-weighted term identically
zero. Because pairing is always within a dataset, no `d_pair` value is ever compared across datasets
as a physical distance.

Verified on real data: CASIA `d_scale` unique values = **[0.0]**.

## Q-27 — luminance · RESOLVED_BY_OWNER_BT601_UNIT_MEAN_ABSDIFF

Input is the frozen canonical **256×256 RGB uint8** face from M2 — never the raw frame, the SCRFD
crop before the canonical resize, the FaceXFormer or AdaFace input, or any generated image.
Channels are scaled to `[0, 1]` in float64 and

```
Y = 0.299·R + 0.587·G + 0.114·B          # BT.601, no offset, no studio range
Y_mean = mean over all 256×256 pixels     # including canonical zero-padded pixels
d_luma = |Y_mean_target − Y_mean_source|
```

No OpenCV integer YCrCb path and no BT.709. "Normalized" fixes the range and nothing else: `Y_mean`
is not z-scored and no luminance statistic is fitted from TRAIN. No background masking and no
parsing-region restriction.

## Final distance (unchanged)

```
d_pair = 0.50·d_pose + 0.30·d_scale + 0.20·d_luma
```

float64 throughout, no post-hoc normalisation, no weight tuning, no dataset-specific weights, no
threshold. Winner is the minimum; an exact tie breaks on lexical `target_sample_id`.

## Real-data diagnostic (evaluated, not used to choose anything)

Deterministic hash-selected sample, 40 sources per dataset × split,
all candidates evaluated. Every component finite and non-negative in every cell.

| dataset | split | d_pose (med) | d_scale (med) | d_luma (med) | d_pair (med) |
|---|---|---|---|---|---|
| casia_fasd | TRAIN | 2.1666 | 0.0000 | 0.0900 | 1.1059 |
| casia_fasd | VAL | 2.5880 | 0.0000 | 0.0986 | 1.3128 |
| msu_mfsd | TRAIN | 2.2053 | 0.4574 | 0.0922 | 1.2720 |
| msu_mfsd | VAL | 2.1403 | 0.4502 | 0.0947 | 1.2538 |
| siwmv2 | TRAIN | 1.6693 | 0.5500 | 0.0782 | 1.0723 |
| siwmv2 | VAL | 1.7053 | 0.5892 | 0.0767 | 1.1046 |

No formula was changed on the basis of these distributions, and TEST was never inspected.

## Q-28 — DSDG native manifest scope · RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE

*(Owner decision, 2026-09-20. Scope: which datasets may appear in the M4 native manifest. Nothing
else.)*

`manifests/dsdg_identity_pairs_v1.parquet` (TRAIN, one same-identity live/spoof pair per row):

| dataset | status |
|---|---|
| casia_fasd | SUPPORTED |
| msu_mfsd | SUPPORTED |
| siwmv2 | NOT_INSTANTIABLE_MISSING_SUBJECT_ID |

The manifest contains **CASIA + MSU only**. For SiW: `native_identity_pair_coverage = 0`,
`status = NOT_INSTANTIABLE_MISSING_SUBJECT_ID`, represented in the audit coverage table and never as
manifest rows. Forbidden: invented or pseudo SiW subject ids, `content_group_id` or `video_id` used
as person identity, and DEV-013 used as a same-identity substitute.

**This does not decide the later M6 DSDG training-adaptation strategy for SiW.** That is a separate
question and is deliberately left open.

## Q-29 — DiffFAS native manifest scope · RESOLVED_FOR_M4_NATIVE_MANIFEST_SCOPE

*(Owner decision, 2026-09-20. Same scope limit.)*

DiffFAS native reconstruction pairs require **same dataset + same trustworthy identity + LIVE/SPOOF**.
`manifests/difffas_recon_pairs_v1.parquet` (TRAIN):

| dataset | status |
|---|---|
| casia_fasd | SUPPORTED |
| msu_mfsd | SUPPORTED |
| siwmv2 | NOT_INSTANTIABLE_MISSING_SUBJECT_ID |

The manifest contains **CASIA + MSU only**. Forbidden for SiW: pairing by same video, pairing by
content group, inferring identity, invoking DEV-013, and pairing arbitrary live/spoof rows to
imitate same-ID reconstruction. **DIFFFAS-BIN does not rescue SiW** — collapsing `style_id` to a
binary label leaves the reconstruction-pair structure intact, and the missing information is
identity, not style.

**This does not silently settle every later DiffFAS M6 adaptation question.**

## DEV-013 boundary (reasserted)

DEV-013 resolves the **COMMON** source→target pairing rule for SiW: different canonical video **AND**
different exact-content group. That is a different-video / different-exact-content guarantee. It is
**not** a same-person claim and not a different-person claim. It does **not** apply to DSDG
same-identity native pairing or to DiffFAS same-identity reconstruction pairing.
