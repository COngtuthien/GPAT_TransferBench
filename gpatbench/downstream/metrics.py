"""Downstream metrics resolved by the frozen spec (§15, §16, §18). Pure functions; stdlib only.

Conventions fixed by the spec or by the frozen data contract:
- y_true is `label_binary` from split_v1: 1 = spoof, 0 = live (attack_macro 'live' <=> 0).
- frame probability = sigmoid(binary logit) (§15);
- video score = arithmetic mean of that canonical video's frame probabilities (§15, §26); a canonical video is
  (dataset, video_id);
- APCER = spoof videos predicted live / spoof videos; BPCER = live videos predicted spoof / live videos;
  ACER = (APCER + BPCER) / 2 (§16);
- threshold on VAL video scores by minimizing ACER over all unique score midpoints plus 0 and 1, tie-break the
  threshold closest to 0.5 (§15).

The scores passed here are SPOOF scores (higher = more spoof-like). Which network output is the spoof score is part
of the evaluator contract, not of this module. The comparison at the threshold (>= or >) is not fixed by the spec:
every function that turns a score into a decision takes `rule` explicitly and has no default (the owner-selected D4
value is supplied by the approved evaluator contract, never by this module). ACER comparisons use exact rationals so
that ties are decided exactly; a tie the spec does not break raises UnresolvedThresholdTie. EER (D7), spoof F1 (D8)
and frame-level rates at the video threshold (D9), TPR@FPR (D8) and the SiW-Mv2 raw-type table (D10) follow the
owner-approved evaluator contract (configs/amendments/m8_downstream_evaluator_contract.yaml).
"""
from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Iterable, Sequence

SPOOF, LIVE = 1, 0
DECISION_RULES = ("SPOOF_IF_SCORE_GE_THRESHOLD", "SPOOF_IF_SCORE_GT_THRESHOLD")


class MetricError(ValueError):
    """Invalid metric input."""


class UnresolvedThresholdTie(MetricError):
    """Two thresholds have the minimum ACER and are equally close to 0.5; the spec gives no further tie-break."""


def _check_rule(rule: str) -> None:
    if rule not in DECISION_RULES:
        raise MetricError(f"decision rule must be one of {DECISION_RULES}; the spec fixes none, so it has no default")


def _check_labels(y_true: Sequence[int]) -> None:
    if any(y not in (0, 1) or isinstance(y, bool) for y in y_true):
        raise MetricError("y_true must be label_binary values 0 (live) or 1 (spoof)")


def _check_scores(scores: Sequence[float]) -> None:
    for s in scores:
        if not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s) or not 0.0 <= s <= 1.0:
            raise MetricError(f"score {s!r} is not a finite probability in [0, 1]")


def sigmoid(logit: float) -> float:
    """Frame probability = sigmoid(binary logit) (§15), numerically stable."""
    if not math.isfinite(logit):
        raise MetricError("non-finite logit")
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


def aggregate_videos(frames: Iterable[dict]) -> list[dict]:
    """Frame rows {dataset, video_id, y_true, p_frame, ...} -> one row per canonical video, sorted by key.

    video_score is the arithmetic mean of that video's frame probabilities only (math.fsum, so the result does not
    depend on frame order). Every frame of a video must carry the same y_true; extra keys that are constant within the
    video (e.g. attack_macro, subject_id_global) are carried over, varying ones are refused.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in frames:
        groups[(f["dataset"], f["video_id"])].append(f)
    out = []
    for (dataset, video_id), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        labels = {r["y_true"] for r in rows}
        _check_labels(list(labels))
        if len(labels) != 1:
            raise MetricError(f"video {dataset}/{video_id} mixes labels {sorted(labels)}")
        probs = [r["p_frame"] for r in rows]
        _check_scores(probs)
        video = {"dataset": dataset, "video_id": video_id, "y_true": labels.pop(), "n_frames": len(rows),
                 "video_score": math.fsum(probs) / len(probs)}
        for k in ("attack_macro", "attack_raw", "subject_id_global"):
            if k in rows[0]:
                vals = {r.get(k) for r in rows}
                if len(vals) != 1:
                    raise MetricError(f"video {dataset}/{video_id}: {k} varies within the video")
                video[k] = vals.pop()
        out.append(video)
    return out


def predict_spoof(scores: Sequence[float], threshold: float, *, rule: str) -> list[int]:
    _check_rule(rule)
    _check_scores(scores)
    if rule == "SPOOF_IF_SCORE_GE_THRESHOLD":
        return [SPOOF if s >= threshold else LIVE for s in scores]
    return [SPOOF if s > threshold else LIVE for s in scores]


def confusion_counts(y_true: Sequence[int], pred: Sequence[int]) -> dict:
    _check_labels(y_true)
    _check_labels(pred)
    if len(y_true) != len(pred):
        raise MetricError("length mismatch")
    return {"n_live": sum(1 for y in y_true if y == LIVE), "n_spoof": sum(1 for y in y_true if y == SPOOF),
            "live_pred_spoof": sum(1 for y, p in zip(y_true, pred) if y == LIVE and p == SPOOF),
            "spoof_pred_live": sum(1 for y, p in zip(y_true, pred) if y == SPOOF and p == LIVE)}


def rates_from_counts(c: dict) -> dict:
    """APCER/BPCER/ACER (exact Fractions) from confusion counts (§16). A rate whose class is absent is None (N/A)."""
    apcer = Fraction(c["spoof_pred_live"], c["n_spoof"]) if c["n_spoof"] else None
    bpcer = Fraction(c["live_pred_spoof"], c["n_live"]) if c["n_live"] else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    n = c["n_live"] + c["n_spoof"]
    correct = (c["n_spoof"] - c["spoof_pred_live"]) + (c["n_live"] - c["live_pred_spoof"])
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer,
            "accuracy": Fraction(correct, n) if n else None,
            "balanced_accuracy": 1 - acer if acer is not None else None}


def threshold_candidates(scores: Sequence[float]) -> list[float]:
    """All unique score midpoints plus 0 and 1 (§15), ascending and unique."""
    _check_scores(scores)
    u = sorted(set(float(s) for s in scores))
    mids = [(a + b) / 2.0 for a, b in zip(u, u[1:])]
    return sorted(set(mids) | {0.0, 1.0})


def select_threshold(y_true: Sequence[int], scores: Sequence[float], *, rule: str) -> dict:
    """VAL threshold: minimum video ACER; tie-break the threshold closest to 0.5 (§15). Both classes required."""
    _check_labels(y_true)
    if len(y_true) != len(scores):
        raise MetricError("length mismatch")
    if not (LIVE in y_true and SPOOF in y_true):
        raise MetricError("threshold selection needs live and spoof videos")
    evaluated = []
    for t in threshold_candidates(scores):
        r = rates_from_counts(confusion_counts(y_true, predict_spoof(scores, t, rule=rule)))
        evaluated.append((r["ACER"], t))
    best_acer = min(a for a, _ in evaluated)
    best = [t for a, t in evaluated if a == best_acer]
    closest = min(abs(t - 0.5) for t in best)
    chosen = [t for t in best if abs(t - 0.5) == closest]
    if len(chosen) != 1:
        raise UnresolvedThresholdTie(f"thresholds {chosen} tie on ACER {best_acer} and distance to 0.5")
    return {"threshold": chosen[0], "val_acer": best_acer, "candidates": len(evaluated),
            "min_acer_thresholds": best, "rule": rule}


def auc(y_true: Sequence[int], scores: Sequence[float]) -> Fraction:
    """Area under the empirical ROC curve with spoof as the positive class (exact Mann-Whitney form, ties = 1/2)."""
    _check_labels(y_true)
    _check_scores(scores)
    pos = sorted(s for y, s in zip(y_true, scores) if y == SPOOF)
    neg = sorted(s for y, s in zip(y_true, scores) if y == LIVE)
    if not pos or not neg:
        raise MetricError("AUC needs live and spoof samples")
    import bisect
    twice = 0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        twice += 2 * lo + (hi - lo)
    return Fraction(twice, 2 * len(pos) * len(neg))


def curve_points(y_true: Sequence[int], scores: Sequence[float], *, rule: str) -> list[dict]:
    """ROC/DET operating points at every §15 threshold candidate: APCER, BPCER, and the ROC pair
    (FPR = BPCER, TPR = 1 - APCER) with spoof as the positive class."""
    pts = []
    for t in threshold_candidates(scores):
        r = rates_from_counts(confusion_counts(y_true, predict_spoof(scores, t, rule=rule)))
        pts.append({"threshold": t, "APCER": r["APCER"], "BPCER": r["BPCER"],
                    "FPR": r["BPCER"], "TPR": None if r["APCER"] is None else 1 - r["APCER"]})
    return pts


def apply_frozen_threshold(videos: Sequence[dict], threshold: float, *, rule: str) -> dict:
    """Apply one frozen threshold to video rows and report the §16 rates, plus dataset-wise ACER/AUC and
    attack_macro-wise APCER (§16 subgroups) under that same threshold."""
    y = [v["y_true"] for v in videos]
    s = [v["video_score"] for v in videos]
    pred = predict_spoof(s, threshold, rule=rule)
    out = {"overall": rates_from_counts(confusion_counts(y, pred)), "by_dataset": {}, "apcer_by_attack_macro": {}}
    for ds in sorted({v["dataset"] for v in videos}):
        idx = [i for i, v in enumerate(videos) if v["dataset"] == ds]
        yi, si, pi = [y[i] for i in idx], [s[i] for i in idx], [pred[i] for i in idx]
        r = rates_from_counts(confusion_counts(yi, pi))
        r["AUC"] = auc(yi, si) if (LIVE in yi and SPOOF in yi) else None
        out["by_dataset"][ds] = r
    for m in sorted({v.get("attack_macro") for v in videos if v["y_true"] == SPOOF}, key=str):
        idx = [i for i, v in enumerate(videos) if v["y_true"] == SPOOF and v.get("attack_macro") == m]
        out["apcer_by_attack_macro"][m] = Fraction(sum(1 for i in idx if pred[i] == LIVE), len(idx))
    return out


def eer_linear_interpolation(y_true: Sequence[int], scores: Sequence[float], *, rule: str) -> Fraction:
    """EER, owner-selected D7 LINEAR_INTERPOLATION (final contract approval pending).

    Over the section-15 threshold candidates in ascending order, APCER is non-decreasing and BPCER non-increasing,
    so d = APCER - BPCER is non-decreasing. If some candidate has d = 0, EER = APCER there (all such candidates
    share that value). Otherwise take the adjacent candidates i, i+1 with d_i < 0 < d_{i+1}, interpolate APCER and BPCER
    linearly between the two operating points and return the value at their crossing. Exact rationals."""
    pts = curve_points(y_true, scores, rule=rule)
    if any(p["APCER"] is None or p["BPCER"] is None for p in pts):
        raise MetricError("EER needs live and spoof samples")
    d = [p["APCER"] - p["BPCER"] for p in pts]
    if any(b < a for a, b in zip(d, d[1:])):
        raise MetricError("APCER - BPCER is not monotone over the candidates")
    zero = {p["APCER"] for p, di in zip(pts, d) if di == 0}
    if zero:
        if len(zero) != 1:
            raise MetricError("several EER values at d = 0")
        return zero.pop()
    for (p0, d0), (p1, d1) in zip(zip(pts, d), zip(pts[1:], d[1:])):
        if d0 < 0 < d1:
            alpha = (p0["BPCER"] - p0["APCER"]) / ((p1["APCER"] - p0["APCER"]) - (p1["BPCER"] - p0["BPCER"]))
            return p0["APCER"] + alpha * (p1["APCER"] - p0["APCER"])
    raise MetricError("APCER and BPCER do not cross over the candidates")


def f1_spoof(y_true: Sequence[int], pred: Sequence[int]):
    """F1 of the spoof class (owner-selected D8 SPOOF_POSITIVE): 2TP / (2TP + FP + FN); None (N/A) if undefined."""
    c = confusion_counts(y_true, pred)
    tp = c["n_spoof"] - c["spoof_pred_live"]
    denom = 2 * tp + c["live_pred_spoof"] + c["spoof_pred_live"]
    return Fraction(2 * tp, denom) if denom else None


def frame_level_rates(frames: Sequence[dict], video_threshold: float, *, rule: str) -> dict:
    """Frame-level diagnostics (T08) at the frozen VIDEO threshold (owner-selected D9 VIDEO_THRESHOLD)."""
    y = [f["y_true"] for f in frames]
    pred = predict_spoof([f["p_frame"] for f in frames], video_threshold, rule=rule)
    out = rates_from_counts(confusion_counts(y, pred))
    out["F1_spoof"] = f1_spoof(y, pred)
    return out


def tpr_at_fpr(y_true: Sequence[int], scores: Sequence[float], target: str, *, rule: str):
    """D8 (owner-approved): max TPR = 1 - APCER over the section-15 candidates with BPCER <= target, no interpolation.
    `target` is a decimal string ("0.01", "0.05") so the comparison is exact. None (N/A) if no candidate qualifies."""
    x = Fraction(target)
    ok = [1 - p["APCER"] for p in curve_points(y_true, scores, rule=rule)
          if p["APCER"] is not None and p["BPCER"] is not None and p["BPCER"] <= x]
    return max(ok) if ok else None


def siw_raw_type_table(videos: Sequence[dict], threshold: float, token_to_macro: dict, *, rule: str) -> dict:
    """D10 RAW_TYPE_COVERAGE_V1 (owner-approved) over the evaluated siwmv2 spoof videos at the frozen threshold.

    Gate: every siwmv2 spoof video has one non-null attack_raw that is a known token mapping to its attack_macro
    (constancy within a video is enforced by aggregate_videos). Any violation -> the whole table is N/A with reason
    RAW_LABEL_COVERAGE_INVALID. Otherwise every token is listed: APCER with numerator/denominator, or N/A
    ABSENT_IN_SPLIT. No pooling, imputation or minimum-support filter."""
    tokens = sorted(token_to_macro)
    spoof = [v for v in videos if v["dataset"] == "siwmv2" and v["y_true"] == SPOOF]
    bad = [v["video_id"] for v in spoof if v.get("attack_raw") not in token_to_macro
           or token_to_macro[v["attack_raw"]] != v.get("attack_macro")]
    if bad:
        return {"status": "N/A", "reason": "RAW_LABEL_COVERAGE_INVALID", "invalid_videos": len(bad),
                "rows": [{"attack_raw": t, "APCER": None, "numerator": None, "denominator": None,
                          "reason": "RAW_LABEL_COVERAGE_INVALID"} for t in tokens]}
    pred = predict_spoof([v["video_score"] for v in spoof], threshold, rule=rule)
    rows = []
    for t in tokens:
        idx = [i for i, v in enumerate(spoof) if v["attack_raw"] == t]
        if not idx:
            rows.append({"attack_raw": t, "APCER": None, "numerator": None, "denominator": 0, "reason": "ABSENT_IN_SPLIT"})
        else:
            num = sum(1 for i in idx if pred[i] == LIVE)
            rows.append({"attack_raw": t, "APCER": Fraction(num, len(idx)), "numerator": num, "denominator": len(idx),
                         "reason": None})
    return {"status": "VALID", "reason": None, "invalid_videos": 0, "rows": rows}


def seed_mean_std(values: Sequence[float]) -> dict:
    """Mean and sample standard deviation (n - 1) over seeds (§0, §18). std is None for fewer than 2 values."""
    if not values or any(not math.isfinite(v) for v in values):
        raise MetricError("seed values must be finite and non-empty")
    n = len(values)
    mean = math.fsum(values) / n
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sample_std": std}
