"""M10 VAL threshold and checkpoint selection with the owner-approved O2 completion (stdlib only).

Authority: configs/amendments/m10_resnet18_owner_decisions.yaml O2_threshold_tie_rule = A_SMALLER_THRESHOLD, a
scientific-protocol completion of spec section 15 (not a bug fix). gpatbench/downstream/metrics.py is unchanged and
still documents the incomplete frozen rule (its select_threshold raises UnresolvedThresholdTie); this module adds the
completed hierarchy on top of its primitives:

  candidates  every unique VAL video-score midpoint plus 0 and 1 (metrics.threshold_candidates, unchanged)
  decision    spoof iff score >= threshold (D4, unchanged)
  1. minimum VAL video ACER (exact rationals)
  2. among those, minimum distance to 0.5, computed exactly as the approved implementation does (float abs(t - 0.5),
     compared for exact float equality) -- so steps 1-2 are byte-for-byte the frozen metrics.select_threshold rule
  3. among those, the numerically smallest threshold (O2 A_SMALLER_THRESHOLD, the additive final tie-break)

Consequence: select_threshold_o2 returns exactly what metrics.select_threshold returns whenever the latter returns, and
resolves every case in which it raises UnresolvedThresholdTie. Thresholds that are equidistant from 0.5 as real
numbers but whose float midpoints differ in the last bit (e.g. 0.025 and 0.975, whose float distances 0.5 - 0.025 and
0.975 - 0.5 both round to 0.475) are a tie under this rule and resolve to the smaller threshold.
Every validly evaluated epoch stays eligible (C_EPOCH_INELIGIBLE is not implemented).

Checkpoint selection (D5, unchanged): minimum VAL video ACER, then maximum VAL video AUC, then the earliest epoch.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from . import metrics as M

RULE = "SPOOF_IF_SCORE_GE_THRESHOLD"
THRESHOLD_RULE_ID = "SPEC15_MIN_ACER__NEAREST_0_5__O2_A_SMALLER_THRESHOLD"
THRESHOLD_RULE_TEXT = ("spec 15 + O2 A_SMALLER_THRESHOLD: minimum VAL video ACER over all unique video-score midpoints "
                       "plus 0 and 1; then minimum abs(threshold - 0.5) (float, as the approved implementation); then the "
                       "numerically smallest threshold; "
                       "D4 spoof iff score >= threshold")
CHECKPOINT_RULE_TEXT = ("D5 EPOCH_OWN_VAL_THRESHOLD: minimum VAL video ACER, then maximum VAL video AUC, then the "
                        "earliest epoch (selection.checkpoint min_val_video_acer_then_max_auc)")


def select_threshold_o2(y_true: Sequence[int], scores: Sequence[float]) -> dict:
    """VAL video threshold under the completed hierarchy. Refuses invalid input exactly as metrics does (labels 0/1,
    finite scores in [0, 1], equal lengths, both classes present)."""
    M._check_labels(y_true)
    M._check_scores(scores)
    if len(y_true) != len(scores):
        raise M.MetricError("length mismatch")
    if not (M.LIVE in y_true and M.SPOOF in y_true):
        raise M.MetricError("threshold selection needs live and spoof videos")
    evaluated = []
    for t in M.threshold_candidates(scores):
        r = M.rates_from_counts(M.confusion_counts(y_true, M.predict_spoof(scores, t, rule=RULE)))
        evaluated.append((r["ACER"], t))
    best_acer = min(a for a, _ in evaluated)
    min_acer = [t for a, t in evaluated if a == best_acer]
    dist = {t: abs(t - 0.5) for t in min_acer}
    d = min(dist.values())
    nearest = [t for t in min_acer if dist[t] == d]
    chosen = min(nearest)
    tie_break = "NONE" if len(min_acer) == 1 else ("NEAREST_0_5" if len(nearest) == 1 else "O2_SMALLER_THRESHOLD")
    return {"threshold": chosen, "val_acer": best_acer, "candidates": len(evaluated), "min_acer_thresholds": min_acer,
            "nearest_thresholds": nearest, "tie_break": tie_break, "rule": RULE, "threshold_rule": THRESHOLD_RULE_ID}


def checkpoint_rank(e: dict) -> tuple:
    """D5 sort key of an epoch record {'epoch', 'val_acer', 'val_auc'} (rationals as strings or Fractions)."""
    return (Fraction(e["val_acer"]), -Fraction(e["val_auc"]), int(e["epoch"]))


def select_epoch(epochs: Sequence[dict]) -> dict:
    """The D5 winner over every evaluated epoch (never the final epoch by default)."""
    if not epochs:
        raise M.MetricError("no evaluated epoch")
    if len({int(e["epoch"]) for e in epochs}) != len(epochs):
        raise M.MetricError("duplicate epoch records")
    return min(epochs, key=checkpoint_rank)
