"""Common source->target pair construction (spec §6) — FROZEN contract.

The spec fixes the shape of the rule: pairs are built per split, every spoof frame is one source
row, the target is a LIVE frame of the same dataset under a dataset-specific different-identity
constraint, at most 64 candidates are evaluated, and

    d_pair = 0.50*d_pose + 0.30*d_scale + 0.20*d_luma

is minimised with a lexical tie-break. It did **not** fix the candidate sampler or the three
distance formulas; the owner resolved those as Q-24..Q-27 and they are frozen in
`configs/frozen/pairs_v1.yaml`. This module implements exactly that contract:

* **Q-24** candidates are ranked by a two-stage SHA-256 (a per-source seed digest, then a per
  candidate digest), read as an unsigned big-endian integer, with lexical `target_sample_id` as a
  defensive tie-break. No PRNG, no replacement, no dependence on input order or `PYTHONHASHSEED`.
  `pair_id` is assigned only after membership is decided, so selection can never depend on it —
  which matters because spec §8.1 seeds FAS-Aug from `pair_id`.
* **Q-25** pose is z-scored per dataset on TRAIN rows only (population std, ddof=0, float64) and
  compared with Euclidean L2. VAL reuses the TRAIN statistics; TEST never contributes. A degenerate
  standard deviation is a hard error rather than a silent epsilon.
* **Q-26** scale uses the *visible* (frame-clipped) SCRFD box as a fraction of the original frame
  area, so raw resolution drops out, and `d_scale = |ln f_t - ln f_s|`. CASIA has no detector box at
  all under DEV-011, so by owner adaptation DEV-018 its fraction is exactly 1.0 and `d_scale` is
  exactly 0 for every CASIA pair. The weights are **not** renormalised.
* **Q-27** luminance is the full-image mean of BT.601 Y over the frozen canonical 256x256 RGB face
  with channels scaled to [0, 1], and `d_luma` is the absolute difference.

Everything is computed in float64 and no fuzzy tie tolerance is used anywhere.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

SPLITS_WITH_PAIRS = ("TRAIN", "VAL")   # spec App. A; TEST pairs are not required and are never built
CANDIDATE_CAP = 64                     # spec §6
SPLIT_SEED = 20260814                  # spec §3.5
WEIGHTS = {"pose": 0.50, "scale": 0.30, "luma": 0.20}   # spec §6; never tuned
SOURCE_SEED_NAMESPACE = "gpatbench.pair.source_seed.v1"
CANDIDATE_NAMESPACE = "gpatbench.pair.candidate.v1"
PAIR_ID_PREFIX = {"TRAIN": "PTR", "VAL": "PVA"}
PAIR_ID_PREFIX_FORMAT = {"TRAIN": "PTR%06d", "VAL": "PVA%06d"}
MIN_POSE_STD = 1e-12
BT601 = (0.299, 0.587, 0.114)          # Q-27: R, G, B
CASIA_FACE_AREA_FRACTION = 1.0         # Q-26 / DEV-018: no detector box exists for CASIA


class PairPolicyError(RuntimeError):
    """The frozen pair contract was violated or required metadata is missing."""


class PairFeasibilityError(RuntimeError):
    """A spoof source has no eligible live target; a source may never be silently skipped."""


@dataclass(frozen=True)
class Sample:
    """One usable (M2 COMPLETE) split row plus the frozen M2 measurements a pair needs."""
    sample_id: str
    dataset: str
    split: str
    label_binary: int
    video_id: str
    subject_id_global: str | None
    content_group_id: str | None
    attack_macro: str
    attack_raw: str | None
    sha256: str
    pose: tuple | None = None               # (pitch, yaw, roll) radians, frozen FaceXFormer output
    face_area_fraction: float | None = None  # Q-26; exactly 1.0 for CASIA (DEV-018)
    luma_mean: float | None = None           # Q-27; BT.601 Y mean of the canonical face, in [0, 1]


# ------------------------------------------------------------------ eligibility
def eligible_targets(source: Sample, live: list) -> list:
    """Live candidates allowed for this source, in canonical (lexical sample_id) order.

    CASIA/MSU use the trustworthy subject id. SiW has no trustworthy subject identity (Q-14), so
    DEV-013 applies: different canonical video AND different exact-content group. That is a
    different-video / different-exact-content guarantee and must never be described as
    different-person.
    """
    if source.label_binary != 1:
        raise PairPolicyError(f"source {source.sample_id} is not a spoof row")
    if source.split not in SPLITS_WITH_PAIRS:
        raise PairPolicyError(f"pairs are built for {SPLITS_WITH_PAIRS}, not {source.split}")
    out = []
    for t in live:
        if t.label_binary != 0 or t.dataset != source.dataset or t.split != source.split:
            continue
        if source.dataset == "siwmv2":
            if t.video_id == source.video_id or t.content_group_id == source.content_group_id:
                continue
        else:
            if not source.subject_id_global or not t.subject_id_global:
                raise PairPolicyError(f"{source.dataset} requires subject ids for pairing")
            if t.subject_id_global == source.subject_id_global:
                continue
        out.append(t)
    return sorted(out, key=lambda t: t.sample_id)


# ------------------------------------------------------------------ Q-24 candidate selection
def source_seed_digest(source_sample_id: str, split_seed: int = SPLIT_SEED) -> bytes:
    return hashlib.sha256(
        f"{SOURCE_SEED_NAMESPACE}|{source_sample_id}|{split_seed}".encode("utf-8")).digest()


def candidate_rank_digest(seed_digest: bytes, target_sample_id: str) -> bytes:
    """SHA-256 over the raw source-seed digest concatenated with the namespaced target id."""
    return hashlib.sha256(
        seed_digest + f"|{CANDIDATE_NAMESPACE}|{target_sample_id}".encode("utf-8")).digest()


def candidate_rank_key(source_sample_id: str, target_sample_id: str,
                       split_seed: int = SPLIT_SEED) -> tuple:
    """(digest as unsigned big-endian 256-bit int, target_sample_id) — the frozen ranking key."""
    d = candidate_rank_digest(source_seed_digest(source_sample_id, split_seed), target_sample_id)
    return (int.from_bytes(d, "big", signed=False), target_sample_id)


def select_candidates(source: Sample, eligible: list, split_seed: int = SPLIT_SEED,
                      cap: int = CANDIDATE_CAP) -> list:
    """<= cap eligible -> evaluate all; otherwise exactly `cap`, by frozen SHA-256 ranking."""
    if not eligible:
        raise PairFeasibilityError(
            f"source {source.sample_id} ({source.dataset}/{source.split}) has no eligible live target")
    canonical = sorted(eligible, key=lambda t: t.sample_id)    # canonical order before any hashing
    if len(canonical) <= cap:
        return canonical
    ranked = sorted(canonical, key=lambda t: candidate_rank_key(source.sample_id, t.sample_id, split_seed))
    return sorted(ranked[:cap], key=lambda t: t.sample_id)


# ------------------------------------------------------------------ Q-25 pose
@dataclass(frozen=True)
class PoseStats:
    """Per-dataset pose statistics fitted on TRAIN rows only (population std, ddof=0)."""
    dataset: str
    n_train_complete: int
    mean: tuple       # (pitch, yaw, roll)
    std: tuple

    def z(self, pose) -> tuple:
        return tuple((float(p) - m) / s for p, m, s in zip(pose, self.mean, self.std))


def fit_pose_stats(samples: list, dataset: str) -> PoseStats:
    """Fit on this dataset's TRAIN rows only. VAL and TEST never contribute a statistic."""
    import numpy as np
    rows = [s for s in samples if s.dataset == dataset and s.split == "TRAIN" and s.pose is not None]
    if not rows:
        raise PairPolicyError(f"{dataset}: no TRAIN rows with pose to fit the normalizer")
    P = np.asarray([s.pose for s in rows], dtype=np.float64)
    mean = P.mean(axis=0)
    std = P.std(axis=0, ddof=0)                                  # population std, frozen
    for i, v in enumerate(std):
        if v <= MIN_POSE_STD:
            raise PairPolicyError(
                f"{dataset}: pose component {i} has std {v} <= {MIN_POSE_STD}; the frozen contract "
                "treats a degenerate pose axis as a hard error (never epsilon or a dropped axis)")
    return PoseStats(dataset, len(rows), tuple(float(x) for x in mean), tuple(float(x) for x in std))


def d_pose(source: Sample, target: Sample, stats: PoseStats) -> float:
    if source.pose is None or target.pose is None:
        raise PairPolicyError("pose missing; the frozen M2 geometry cache must supply it")
    if stats.dataset != source.dataset or source.dataset != target.dataset:
        raise PairPolicyError("pose statistics must belong to the pair's own dataset")
    zs, zt = stats.z(source.pose), stats.z(target.pose)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(zt, zs)))


# ------------------------------------------------------------------ Q-26 scale
def face_area_fraction(dataset: str, bbox, frame_w: float, frame_h: float) -> float:
    """Visible (frame-clipped) detector box area as a fraction of the original frame area.

    Uses the ORIGINAL SCRFD box, before the 1.25x expansion, the zero padding and the 256 resize, so
    the value is dimensionless and independent of raw frame resolution. CASIA never reaches here:
    it has no detector box and takes the frozen adaptation value instead (DEV-018).
    """
    if dataset == "casia_fasd":
        raise PairPolicyError("CASIA has no detector box; use CASIA_FACE_AREA_FRACTION (DEV-018)")
    if not (frame_w > 0 and frame_h > 0):
        raise PairPolicyError(f"invalid frame dimensions {frame_w}x{frame_h}")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx1, cx2 = max(0.0, min(frame_w, x1)), max(0.0, min(frame_w, x2))
    cy1, cy2 = max(0.0, min(frame_h, y1)), max(0.0, min(frame_h, y2))
    w, h = cx2 - cx1, cy2 - cy1
    if not (w > 0 and h > 0):
        raise PairPolicyError(f"non-positive visible bbox {w}x{h}; geometry is never repaired silently")
    frac = (w * h) / (frame_w * frame_h)
    if not (0.0 < frac <= 1.0):
        raise PairPolicyError(f"face_area_fraction {frac} outside (0, 1]")
    return frac


def d_scale(source: Sample, target: Sample) -> float:
    """|ln(f_target) - ln(f_source)| with natural log, no epsilon and no clipping.

    For CASIA both fractions are exactly 1.0, so this is exactly 0.0 for every CASIA pair: the scale
    term carries no discriminative information there because the dataset does not provide the
    measurement. The 0.50/0.20 weights are deliberately NOT renormalised (DEV-018).
    """
    for s in (source, target):
        if s.face_area_fraction is None:
            raise PairPolicyError(f"{s.dataset}/{s.sample_id}: no face_area_fraction")
        if not (0.0 < s.face_area_fraction <= 1.0):
            raise PairPolicyError(f"{s.sample_id}: face_area_fraction {s.face_area_fraction} outside (0, 1]")
    return abs(math.log(target.face_area_fraction) - math.log(source.face_area_fraction))


# ------------------------------------------------------------------ Q-27 luminance
def luma_mean_from_rgb(rgb_uint8) -> float:
    """Full-image mean of BT.601 Y over a canonical 256x256 RGB uint8 face, channels in [0, 1].

    Every pixel counts, including canonical pixels that came from the Q-22 zero padding: no mask, no
    parsing region, no z-normalisation and no TRAIN-fitted luminance statistic.
    """
    import numpy as np
    a = np.asarray(rgb_uint8)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise PairPolicyError(f"expected an RGB uint8 image, got {a.shape} {a.dtype}")
    f = a.astype(np.float64) / 255.0
    y = BT601[0] * f[:, :, 0] + BT601[1] * f[:, :, 1] + BT601[2] * f[:, :, 2]
    return float(y.mean())


def d_luma(source: Sample, target: Sample) -> float:
    for s in (source, target):
        if s.luma_mean is None:
            raise PairPolicyError(f"{s.sample_id}: no luminance mean")
        if not 0.0 <= s.luma_mean <= 1.0:
            raise PairPolicyError(f"{s.sample_id}: luma {s.luma_mean} outside [0, 1]")
    return abs(target.luma_mean - source.luma_mean)


# ------------------------------------------------------------------ combination
def d_pair(source: Sample, target: Sample, stats: PoseStats) -> dict:
    dp = d_pose(source, target, stats)
    ds = d_scale(source, target)
    dl = d_luma(source, target)
    return {"d_pose": dp, "d_scale": ds, "d_luma": dl,
            "d_pair": WEIGHTS["pose"] * dp + WEIGHTS["scale"] * ds + WEIGHTS["luma"] * dl}


def choose_target(source: Sample, candidates: list, stats: PoseStats) -> dict:
    """Minimum d_pair; exact ties break on lexical target sample_id. No fuzzy tolerance."""
    best = None
    for t in sorted(candidates, key=lambda x: x.sample_id):
        d = d_pair(source, t, stats)
        key = (d["d_pair"], t.sample_id)
        if best is None or key < best[0]:
            best = (key, t, d)
    if best is None:
        raise PairFeasibilityError(f"source {source.sample_id}: no candidate to choose from")
    return {"target": best[1], **best[2]}


# ------------------------------------------------------------------ pair ids
def canonical_source_order(sources: list) -> list:
    """Frozen ordering used only for pair_id assignment: dataset, then source sample_id."""
    return sorted(sources, key=lambda s: (s.dataset, s.sample_id))


def assign_pair_ids(split: str, ordered_sources: list) -> list:
    """`PTR`/`PVA` + 6-digit index, numbered from 1 per manifest.

    Assigned only after pair membership is final, so candidate selection can never depend on it —
    which matters because spec §8.1 derives FAS-Aug operator randomness from `pair_id`.
    """
    if split not in PAIR_ID_PREFIX:
        raise PairPolicyError(f"no pair_id prefix for split {split}")
    return [f"{PAIR_ID_PREFIX[split]}{i:06d}" for i in range(1, len(ordered_sources) + 1)]
