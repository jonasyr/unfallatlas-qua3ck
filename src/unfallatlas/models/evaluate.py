"""Evaluation metrics and the Q-phase §8 acceptance gate."""

from __future__ import annotations

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


def select_best_candidate(
    rows: pd.DataFrame, recall_threshold: float = RECALL_CLASS_1_THRESHOLD
) -> pd.Series:
    """Pick the best row from a (family, strategy) comparison table.

    Rule: highest ``macro_f1`` among rows clearing ``recall_class_1 >=
    recall_threshold`` (the harder Q-phase gate). If no row clears it,
    fall back to the highest ``(macro_f1 + recall_class_1) / 2`` combined
    score across all rows, so there is always a well-defined winner even
    when nothing meets the gate yet.

    This directly encodes "both acceptance criteria must pass" instead of
    optimising macro-F1 alone and hoping recall follows — the mistake that
    picked an unweighted-recall Random Forest as champion in the original
    A³ selection rule.
    """
    passing = rows[rows["recall_class_1"] >= recall_threshold]
    if len(passing) > 0:
        return passing.sort_values("macro_f1", ascending=False).iloc[0]
    combined = rows.assign(_combined_score=(rows["macro_f1"] + rows["recall_class_1"]) / 2)
    return combined.sort_values("_combined_score", ascending=False).iloc[0]
