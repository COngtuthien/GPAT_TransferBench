"""Common source->target pair construction (spec §6) — PROPOSED, not frozen.

What the spec fixes
-------------------
* pairs are built from one split only; each spoof frame is exactly one source row;
* the target is a LIVE frame of the **same dataset**, from a different subject;
* up to 64 eligible live candidates per source, sampled deterministically "using a hash of
  (source_sample_id, split_seed)"; fewer than 64 eligible -> use all;
* `d_pair = 0.50*d_pose + 0.30*d_scale + 0.20*d_luma`, minimum wins, lexical sample_id breaks ties;
* pose distance from FaceXFormer yaw/pitch/roll "after z-normalization inside TRAIN", scale distance
  from "log face-box area ratio", luminance distance from "normalized Y-channel mean".

What the spec does NOT fix
--------------------------
The three distance formulas are named but not defined to the precision execution needs, and the
64-candidate sampler is described only as "a hash of (source_sample_id, split_seed)". Those gaps are
open owner questions (Q-24..Q-27). This module therefore takes them as **required** policy fields
with no defaults: a caller cannot obtain a pair without stating which convention it used, so no
silent choice can leak into M4. `PairMetricPolicy.frozen` stays False until the owner decides.

Determinism
-----------
Candidate ordering, the 64-subset and the final choice depend only on sample ids, the split seed and
the stored metadata — never on input row order, dict iteration, PYTHONHASHSEED or worker count.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

CANDIDATE_CAP = 64                     # spec §6
SPLIT_SEED = 20260814                  # spec §3.5
WEIGHTS = {"pose": 0.50, "scale": 0.30, "luma": 0.20}   # spec §6; never tuned
CANDIDATE_NAMESPACE = "gpatbench.pairs_v1.candidate.v1"


class PairPolicyError(RuntimeError):
    """A pairing detail that the spec does not fix was left unspecified or is unsupported."""


class PairFeasibilityError(RuntimeError):
    """A spoof source has no eligible live target; the source may never be silently skipped."""


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
    pose: tuple | None = None          # (pitch, yaw, roll) radians, FaceXFormer
    bbox_area: float | None = None     # face-box area, definition = policy.scale_box
    frame_area: float | None = None    # original frame area, for the normalised variant
    luma_mean: float | None = None     # mean Y of the canonical face, definition = policy.luma_standard


@dataclass(frozen=True)
class PairMetricPolicy:
    """Every contested convention, stated explicitly. No field has a default (Q-24..Q-27)."""
    candidate_selection: str           # 'per_candidate_hash_rank'
    pose_stat_scope: str               # 'train_pooled' | 'train_per_dataset'
    pose_std: str                      # 'population' | 'sample'
    pose_norm: str                     # 'l2' | 'l1'
    pose_zero_variance: str            # 'error' | 'treat_as_one'
    scale_box: str                     # 'scrfd_bbox' | 'requested_square' | 'bbox_frame_fraction'
    scale_formula: str                 # 'abs_log_ratio'
    luma_standard: str                 # 'bt601' | 'bt709'
    luma_range: str                    # 'unit_interval'
    luma_formula: str                  # 'abs_difference'
    tie_break: str = "lexical_target_sample_id"
    frozen: bool = False               # flipped only by an explicit owner decision

    ALLOWED = {
        "candidate_selection": {"per_candidate_hash_rank"},
        "pose_stat_scope": {"train_pooled", "train_per_dataset"},
        "pose_std": {"population", "sample"},
        "pose_norm": {"l2", "l1"},
        "pose_zero_variance": {"error", "treat_as_one"},
        "scale_box": {"scrfd_bbox", "requested_square", "bbox_frame_fraction"},
        "scale_formula": {"abs_log_ratio"},
        "luma_standard": {"bt601", "bt709"},
        "luma_range": {"unit_interval"},
        "luma_formula": {"abs_difference"},
    }

    def validate(self) -> None:
        for k, allowed in self.ALLOWED.items():
            v = getattr(self, k)
            if v not in allowed:
                raise PairPolicyError(f"{k}={v!r} is not one of {sorted(allowed)}")


# ------------------------------------------------------------------ eligibility
def eligible_targets(source: Sample, live: list) -> list:
    """Live candidates allowed for this source. Same dataset, same split, different identity.

    CASIA/MSU use the trustworthy subject id. SiW has no trustworthy subject identity (Q-14), so
    DEV-013 applies: different canonical video AND different exact-content group. That is a
    different-video / different-exact-content guarantee and must never be described as
    different-person.
    """
    if source.label_binary != 1:
        raise PairPolicyError(f"source {source.sample_id} is not a spoof row")
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
    return out


def candidate_rank(source_sample_id: str, target_sample_id: str, seed: int = SPLIT_SEED) -> str:
    """Per-candidate deterministic rank key (PROPOSED, Q-24).

    The spec says the 64 candidates are sampled "using a hash of (source_sample_id, split_seed)" but
    does not say how. This proposal hashes each (source, candidate, seed) triple and keeps the 64
    smallest digests: it is order-independent, needs no PRNG state, and gives each source its own
    pseudo-random candidate subset. It is NOT frozen.
    """
    return hashlib.sha256(
        f"{CANDIDATE_NAMESPACE}|{source_sample_id}|{target_sample_id}|{seed}".encode("utf-8")
    ).hexdigest()


def select_candidates(source: Sample, eligible: list, policy: PairMetricPolicy,
                      seed: int = SPLIT_SEED, cap: int = CANDIDATE_CAP) -> list:
    """<= cap eligible -> all of them; otherwise exactly `cap`, chosen deterministically."""
    policy.validate()
    if not eligible:
        raise PairFeasibilityError(
            f"source {source.sample_id} ({source.dataset}/{source.split}) has no eligible live target")
    if len(eligible) <= cap:
        return sorted(eligible, key=lambda t: t.sample_id)
    ranked = sorted(eligible, key=lambda t: (candidate_rank(source.sample_id, t.sample_id, seed),
                                             t.sample_id))
    return sorted(ranked[:cap], key=lambda t: t.sample_id)


# ------------------------------------------------------------------ distances
@dataclass
class PoseNormalizer:
    """z-normalization fitted on TRAIN rows only (spec §6: 'inside TRAIN')."""
    mean: tuple
    std: tuple
    scope: str
    n_rows: int
    std_convention: str
    zero_variance_axes: tuple = field(default_factory=tuple)

    @staticmethod
    def fit(train_samples: list, policy: PairMetricPolicy, dataset: str | None = None) -> "PoseNormalizer":
        import numpy as np
        policy.validate()
        rows = [s for s in train_samples if s.split == "TRAIN" and s.pose is not None]
        if dataset is not None:
            rows = [s for s in rows if s.dataset == dataset]
        if not rows:
            raise PairPolicyError("no TRAIN rows with pose to fit the normalizer")
        P = np.asarray([s.pose for s in rows], dtype=np.float64)
        mean = P.mean(axis=0)
        std = P.std(axis=0, ddof=1 if policy.pose_std == "sample" else 0)
        zero = tuple(int(i) for i in range(P.shape[1]) if std[i] == 0.0)
        if zero:
            if policy.pose_zero_variance == "error":
                raise PairPolicyError(f"zero-variance pose axes {zero}; no owner rule to rescale them")
            std = std.copy()
            for i in zero:
                std[i] = 1.0
        return PoseNormalizer(tuple(mean), tuple(std), policy.pose_stat_scope, len(rows),
                              policy.pose_std, zero)

    def z(self, pose):
        return tuple((p - m) / s for p, m, s in zip(pose, self.mean, self.std))


def d_pose(a: Sample, b: Sample, norm: PoseNormalizer, policy: PairMetricPolicy) -> float:
    if a.pose is None or b.pose is None:
        raise PairPolicyError("pose missing; M2 geometry cache must supply it")
    za, zb = norm.z(a.pose), norm.z(b.pose)
    diffs = [abs(x - y) for x, y in zip(za, zb)]
    if policy.pose_norm == "l2":
        return math.sqrt(sum(d * d for d in diffs))
    return sum(diffs)


def d_scale(a: Sample, b: Sample, policy: PairMetricPolicy) -> float:
    """|log(area_target / area_source)| — area definition selected by the policy (Q-26).

    CASIA carries no detector box at all (DEV-011 route, SCRFD N/A), so every `scale_box` variant
    that needs one is undefined there. That is raised rather than silently substituted.
    """
    if policy.scale_formula != "abs_log_ratio":
        raise PairPolicyError(policy.scale_formula)
    for s in (a, b):
        if s.bbox_area is None:
            raise PairPolicyError(
                f"{s.dataset}/{s.sample_id}: no face-box area (CASIA has no SCRFD bbox; Q-26 open)")
        if s.bbox_area <= 0:
            raise PairPolicyError(f"{s.sample_id}: non-positive face-box area")
    if policy.scale_box == "bbox_frame_fraction":
        for s in (a, b):
            if not s.frame_area:
                raise PairPolicyError(f"{s.sample_id}: no frame area for the normalised variant")
        ra, rb = a.bbox_area / a.frame_area, b.bbox_area / b.frame_area
        return abs(math.log(rb / ra))
    return abs(math.log(b.bbox_area / a.bbox_area))


def d_luma(a: Sample, b: Sample, policy: PairMetricPolicy) -> float:
    if policy.luma_formula != "abs_difference":
        raise PairPolicyError(policy.luma_formula)
    for s in (a, b):
        if s.luma_mean is None:
            raise PairPolicyError(f"{s.sample_id}: no luminance mean")
        if not 0.0 <= s.luma_mean <= 1.0:
            raise PairPolicyError(f"{s.sample_id}: luma {s.luma_mean} outside the unit interval")
    return abs(b.luma_mean - a.luma_mean)


def d_pair(source: Sample, target: Sample, norm: PoseNormalizer, policy: PairMetricPolicy) -> dict:
    dp = d_pose(source, target, norm, policy)
    ds = d_scale(source, target, policy)
    dl = d_luma(source, target, policy)
    return {"d_pose": dp, "d_scale": ds, "d_luma": dl,
            "d_pair": WEIGHTS["pose"] * dp + WEIGHTS["scale"] * ds + WEIGHTS["luma"] * dl}


def choose_target(source: Sample, candidates: list, norm: PoseNormalizer,
                  policy: PairMetricPolicy) -> dict:
    """Minimum d_pair; exact ties broken by lexical target sample_id (no fuzzy tolerance)."""
    best = None
    for t in sorted(candidates, key=lambda x: x.sample_id):
        d = d_pair(source, t, norm, policy)
        key = (d["d_pair"], t.sample_id)
        if best is None or key < best[0]:
            best = (key, t, d)
    return {"target": best[1], **best[2]}


def pair_id(index: int) -> str:
    """`P` + 6-digit index over the canonical source order (spec table shows `P000001`).

    Numbering is bookkeeping, but it is **not** inert: spec §8.1 derives FAS-Aug operator parameters
    from `SHA256(pair_id + global_seed)`, so the assignment must be frozen. PROPOSED (Q-24 scope).
    """
    return f"P{index:06d}"


def canonical_source_order(sources: list) -> list:
    """Frozen source ordering for pair_id assignment: dataset, then sample_id."""
    return sorted(sources, key=lambda s: (s.dataset, s.sample_id))
