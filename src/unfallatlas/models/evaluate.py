"""Evaluation metrics and the Q-phase §8 acceptance gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

MACRO_F1_THRESHOLD = 0.55
RECALL_CLASS_1_THRESHOLD = 0.50
BINARY_MACRO_F1_THRESHOLD = 0.55
BINARY_RECALL_KSI_THRESHOLD = 0.50


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def recall_for_class(y_true, y_pred, target_class: int) -> float:
    return float(recall_score(y_true, y_pred, labels=[target_class], average="macro"))


def evaluate_predictions(y_true, y_pred) -> dict:
    """Metrics reported for every model/strategy row in the A³ comparison table."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "recall_class_1": recall_for_class(y_true, y_pred, target_class=1),
        "recall_class_2": recall_for_class(y_true, y_pred, target_class=2),
        "recall_class_3": recall_for_class(y_true, y_pred, target_class=3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3]).tolist(),
    }


def meets_acceptance_criteria(metrics: dict) -> bool:
    """Q-phase §8 acceptance gate: macro-F1 >= 0.55 AND recall(class 1) >= 0.50."""
    return (
        metrics["macro_f1"] >= MACRO_F1_THRESHOLD
        and metrics["recall_class_1"] >= RECALL_CLASS_1_THRESHOLD
    )


def evaluate_binary_predictions(y_true, y_pred) -> dict:
    """Metrics for the binary KSI (label=1) vs. slight (label=0) model."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "recall_ksi": recall_for_class(y_true, y_pred, target_class=1),
        "recall_slight": recall_for_class(y_true, y_pred, target_class=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 0]).tolist(),
    }


def meets_binary_acceptance_criteria(metrics: dict) -> bool:
    """Revised gate: binary macro-F1 >= 0.55 AND Recall(KSI) >= 0.50."""
    return (
        metrics["macro_f1"] >= BINARY_MACRO_F1_THRESHOLD
        and metrics["recall_ksi"] >= BINARY_RECALL_KSI_THRESHOLD
    )


def find_best_binary_threshold(
    y_true,
    scores: np.ndarray,
    recall_gate: float = BINARY_RECALL_KSI_THRESHOLD,
    n_steps: int = 81,
) -> tuple[float, dict]:
    """Sweep a decision threshold over ``scores`` to maximise macro-F1 subject
    to Recall(KSI) >= recall_gate.

    ``scores`` may be ``predict_proba(X)[:, 1]`` (range [0, 1]) or
    ``decision_function(X)`` (unbounded) - both are monotonic in "how KSI-like
    is this row", so the sweep range is derived from ``scores.min()``/
    ``scores.max()`` rather than assumed to be [0, 1]. This lets SVM
    estimators (LinearSVC, SGDClassifier, SVC), which expose
    decision_function but not predict_proba, reuse the same gate-optimal
    thresholding logic as the LightGBM champion's predict_proba sweep.

    Returns (best_threshold, best_metrics). If no threshold satisfies the
    recall gate, returns the unconstrained macro-F1-maximising threshold
    instead - callers must check meets_binary_acceptance_criteria(best_metrics)
    to distinguish the two cases.
    """
    scores = np.asarray(scores)
    best_threshold_gated, best_f1_gated, best_metrics_gated = 0.0, -1.0, None
    best_threshold_free, best_f1_free, best_metrics_free = 0.0, -1.0, None

    for threshold in np.linspace(scores.min(), scores.max(), n_steps):
        y_pred = (scores >= threshold).astype(int)
        metrics = evaluate_binary_predictions(y_true, y_pred)
        if metrics["macro_f1"] > best_f1_free:
            best_f1_free = metrics["macro_f1"]
            best_metrics_free = metrics
            best_threshold_free = float(threshold)
        if metrics["recall_ksi"] >= recall_gate and metrics["macro_f1"] > best_f1_gated:
            best_f1_gated = metrics["macro_f1"]
            best_metrics_gated = metrics
            best_threshold_gated = float(threshold)

    if best_metrics_gated is not None:
        return best_threshold_gated, best_metrics_gated
    return best_threshold_free, best_metrics_free


def select_best_candidate(
    rows: pd.DataFrame,
    recall_threshold: float = RECALL_CLASS_1_THRESHOLD,
    recall_col: str = "recall_class_1",
) -> pd.Series:
    """Pick the best row from a (family, strategy) comparison table.

    Gate-aware: prefers the highest macro_f1 among rows whose `recall_col`
    clears `recall_threshold`. Falls back to the highest combined
    (macro_f1 + recall_col) / 2 score if no row clears the gate.

    `recall_col` defaults to 'recall_class_1' (the 3-class column name,
    from evaluate_predictions) for backward compatibility with every
    existing call site. Pass recall_col='recall_ksi' for binary-KSI
    comparisons (the column name from evaluate_binary_predictions).
    """
    passing = rows[rows[recall_col] >= recall_threshold]
    if len(passing) > 0:
        return passing.sort_values("macro_f1", ascending=False).iloc[0]
    combined = rows.assign(_combined_score=(rows["macro_f1"] + rows[recall_col]) / 2)
    return combined.sort_values("_combined_score", ascending=False).iloc[0]
