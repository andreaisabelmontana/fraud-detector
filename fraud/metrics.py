"""Honest evaluation metrics for imbalanced classification.

Headline metrics here are precision, recall, F1, ROC-AUC and PR-AUC (average
precision). Accuracy is deliberately *not* a headline number on a ~1.5%
positive problem, where predicting "never fraud" already scores ~98.5%.

Also provides threshold selection along the precision-recall curve: pick the
operating point that meets a target precision (or recall) while maximising the
other, which is the "threshold-driven" core of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)


def confusion_counts(y_true, y_pred) -> ConfusionCounts:
    """Four-cell confusion counts from hard 0/1 predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def precision_recall_f1(y_true, y_pred) -> dict:
    """Precision, recall, F1 and accuracy from hard predictions.

    Computed directly from confusion counts so the arithmetic is transparent
    and testable against hand-built cases.
    """
    c = confusion_counts(y_true, y_pred)
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    total = c.tp + c.fp + c.tn + c.fn
    accuracy = (c.tp + c.tn) / total if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion": c.as_dict(),
    }


def ranking_metrics(y_true, y_score) -> dict:
    """Threshold-independent ranking quality: ROC-AUC and PR-AUC.

    PR-AUC (average precision) is the more informative summary on imbalanced
    data because it ignores the large true-negative mass.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def select_threshold(
    y_true,
    y_score,
    target_precision: float = 0.5,
    target_recall: float | None = None,
) -> dict:
    """Choose an operating threshold along the precision-recall curve.

    Default mode (``target_precision`` set): among all thresholds whose
    precision is at least ``target_precision``, pick the one with the highest
    recall. This is the canonical "meet a precision floor, then catch as much
    fraud as possible" rule.

    Alternate mode (``target_recall`` set): among thresholds whose recall is at
    least ``target_recall``, pick the one with the highest precision.

    Returns the chosen threshold and the precision/recall/F1 at that point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns arrays of length len(thresholds)+1;
    # the last precision/recall pair (1.0 / 0.0) has no threshold. Align by
    # dropping that trailing point.
    precision = precision[:-1]
    recall = recall[:-1]

    if target_recall is not None:
        mask = recall >= target_recall
        objective = precision
        target_desc = f"recall>={target_recall}"
    else:
        mask = precision >= target_precision
        objective = recall
        target_desc = f"precision>={target_precision}"

    if not mask.any():
        # No threshold meets the constraint; fall back to the F1-optimal point.
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0.0,
        )
        idx = int(np.argmax(f1))
        chosen = float(thresholds[idx])
        return {
            "threshold": chosen,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "target": target_desc,
            "met_target": False,
        }

    # Among feasible points, maximise the objective; break ties toward the
    # higher threshold (more conservative).
    feasible_idx = np.where(mask)[0]
    best = feasible_idx[np.argmax(objective[feasible_idx])]
    p = float(precision[best])
    r = float(recall[best])
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "threshold": float(thresholds[best]),
        "precision": p,
        "recall": r,
        "f1": float(f1),
        "target": target_desc,
        "met_target": True,
    }
