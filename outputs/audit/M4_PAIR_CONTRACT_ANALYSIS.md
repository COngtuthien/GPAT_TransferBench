# M4 — Pair Contract Analysis (pre-flight)

> **RESOLVED 2026-09-19.** All four questions below were answered by the owner and frozen in
> `configs/frozen/pairs_v1.yaml`: Q-24 `RESOLVED_BY_OWNER_HASH_RANKING`, Q-25
> `RESOLVED_BY_OWNER_DATASET_TRAIN_ZSCORE_L2`, Q-26
> `RESOLVED_BY_OWNER_NORMALIZED_VISIBLE_BBOX_LOGRATIO` (CASIA via **DEV-018**), Q-27
> `RESOLVED_BY_OWNER_BT601_UNIT_MEAN_ABSDIFF`. The decision record is
> `M4_PAIR_OWNER_DECISIONS.md`. **The text below is preserved unchanged as the analysis that
> motivated those decisions**, so the gaps and their measured sensitivity stay on record.

Status at the time of writing: **OWNER_DECISION_REQUIRED**. Four execution-affecting details of the
common pairing rule were not fixed by the frozen specification and had no approved project decision.
The proposal lived in `configs/proposed/pairs_v1.proposed.yaml`; `configs/frozen/pairs_v1.yaml` did
not yet exist, and no pair manifest was created.

## 1. What the spec settles

| item | source | value |
|---|---|---|
| build from | §6, App. A | TRAIN and VAL (`pairs_val = build_common_pairs(split.VAL)`; §19 needs `val_pairs_v1.parquet` for generator checkpoint selection) |
| TEST pairs | — | **not required**; none will be built |
| source row | §6 | every spoof frame of the split, exactly one pair each |
| N_syn,intended | §6/§7 | number of TRAIN spoof frames = **8,838** |
| target | §6 | LIVE, same dataset, different subject |
| candidate cap | §6 | 64 |
| weights | §6 | 0.50 pose / 0.30 scale / 0.20 luma — never tuned |
| selection | §6 | minimum `d_pair`, tie-break by lexical `sample_id` |
| seed | §3.5 | 20260814 |
| target reuse | — | no constraint forbids it, so no one-to-one matching is imposed without evidence |

Only M2 `COMPLETE` rows exist in `split_v1.parquet`, so the 25 `SCRFD_NO_FACE` rows cannot become a
source or a target; nothing about them is regenerated or substituted.

## 2. What the spec does not settle

### Q-24 — the 64-candidate sampler

§6 says the candidates are sampled "using a hash of (source_sample_id, split_seed)" but not how.
Per-candidate hash ranking, a seeded PRNG shuffle and reservoir sampling all satisfy that wording and
select **different subsets**.

This is not a corner case: every source in every dataset and both splits has at least 64 eligible
targets (tightest: MSU VAL at exactly 64), so the sampler decides the evaluated candidate set of
*every* pair.

It also reaches further than it looks. Spec §8.1 derives FAS-Aug operator parameters from
`SHA256(pair_id + global_seed)`, so `pair_id` — ordinarily bookkeeping — becomes scientifically
load-bearing downstream and its assignment rule must be frozen alongside.

### Q-25 — pose distance

"z-normalization inside TRAIN" leaves four things open: pooled vs per-dataset statistics,
population vs sample standard deviation, the distance norm, and zero-variance handling. Measured:
per-dataset std differs sharply from the pooled std (SiW yaw 0.283055
vs pooled 0.231691; CASIA yaw 0.051887),
and the L1/L2 median ratio is 1.4662, which rescales the
0.50-weighted term against the other two. "Distance" alone does not name a norm.

### Q-26 — scale distance (the hardest gap)

"log face-box area ratio" presumes a face box. **CASIA-FASD has none**: the approved DEV-011 route
runs no SCRFD, so 0 of 4,800 CASIA rows carry a
detector bbox. This is a missing quantity, not a choice between conventions, and CASIA supplies
2,520 of the 8,838 TRAIN
sources.

Every conceivable resolution changes the metric for that share of the benchmark — treating the whole
112×112 frame as the box makes `d_scale` identically zero for all CASIA pairs, deriving a box from
FaceXFormer landmarks invents a new geometry definition, dropping the term renormalises the weights,
and excluding CASIA changes the population. None may be adopted silently.

Even where a box exists the definition is open: raw pixel area is not comparable across MSU (640×480)
and SiW (1920×1080) frames, so bbox-pixels and bbox-as-fraction-of-frame are genuinely different
metrics (measured medians 0.2703
vs 0.5731).

### Q-27 — luminance distance

"normalized Y-channel mean" names neither the Y standard nor what "normalized" means. BT.601 and
BT.709 differ by up to 0.0132 per image on a quantity whose
observed range is about 0.55; OpenCV's
`BGR2YCrCb` matches BT.601 to ~1e-4, so the real fork is 601 vs 709.

## 3. Normalization must not leak

Any statistic described as "inside TRAIN" is fitted on TRAIN rows only. VAL pairing should reuse the
TRAIN-fitted statistics rather than fitting its own, otherwise validation would define its own
representation — the proposal states this explicitly. TEST never contributes to any statistic or to
any pair. The preflight fitted the pose normalizer on all 14,467 TRAIN
rows and on nothing else.

## 4. Geometry inputs are reused, never recomputed

Pose comes from the frozen M2 FaceXFormer cache (present for all 20,615 usable rows), the face box
from the M2 accounting, and luminance from the frozen canonical 256×256 faces. SCRFD and FaceXFormer
are not rerun, and no generated image is involved.

## 5. Native methods

DSDG and DiffFAS need identity-based native pairs that the common manifest cannot provide. CASIA and
MSU have full coverage (35/35 and 25/25 TRAIN identities carry both live and spoof); SiW has none,
because it has no trustworthy subject identity. Details and the two scope questions (Q-28, Q-29) are
in `M4_NATIVE_PAIR_REQUIREMENTS.md`. DEV-013 resolved the *common* pairing rule for SiW and is not
stretched to cover a same-identity requirement.

## 6. Why this pass stops here

The feasibility side is clean — 0 sources lack a candidate, and
the row counts are known (8,838 TRAIN, 1,905 VAL).
What is missing is the definition of three of the four quantities the selection rule minimises, plus
the sampler that chooses what is minimised over. Building the manifests now would bake four
undocumented scientific choices into every downstream method, so the contract is proposed rather than
frozen and the decision is returned to the owner.
