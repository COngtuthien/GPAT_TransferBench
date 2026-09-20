"""Fixed-class macro-F1 for ArtifactProbeNet (owner decision D-M5-05).

Deterministic, integer-accumulated, and independent of scikit-learn. The class set is always the
frozen seven; a class that receives no prediction is NOT dropped -- it contributes F1 = 0, which is
what makes the metric comparable across epochs.
"""
from __future__ import annotations

import numpy as np

from .contract import CLASSES, K, ProbeContractViolation


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """7x7 int64 matrix, rows = true class index, cols = predicted class index."""
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if t.shape != p.shape or t.ndim != 1:
        raise ProbeContractViolation(f"shape mismatch {t.shape} vs {p.shape}")
    if t.size and (t.min() < 0 or t.max() >= K or p.min() < 0 or p.max() >= K):
        raise ProbeContractViolation("class index outside the frozen 0..6 range")
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 per frozen class, with 0 substituted for every zero denominator."""
    if cm.shape != (K, K):
        raise ProbeContractViolation(f"confusion matrix must be {K}x{K}, got {cm.shape}")
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    precision = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1.0), 0.0)
    recall = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1.0), 0.0)
    denom = precision + recall
    return np.where(denom > 0, 2.0 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)


def macro_f1(cm: np.ndarray) -> float:
    """Arithmetic mean of F1 over ALL seven frozen classes. Never weighted, never micro."""
    return float(per_class_f1(cm).mean())


def report(cm: np.ndarray) -> dict:
    f1 = per_class_f1(cm)
    tp = np.diag(cm).astype(np.int64)
    return {"classes": list(CLASSES),
            "support": [int(x) for x in cm.sum(axis=1)],
            "predicted": [int(x) for x in cm.sum(axis=0)],
            "true_positives": [int(x) for x in tp],
            "f1": [float(x) for x in f1],
            "macro_f1": macro_f1(cm),
            "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0}


def is_better(new_macro_f1: float, best_macro_f1: float) -> bool:
    """D-M5-06: strictly greater. An exact tie keeps the EARLIER epoch. No epsilon window."""
    return new_macro_f1 > best_macro_f1
