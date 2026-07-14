"""Class-imbalance mitigation strategies compared on the A³ champion model.

U-phase §10 menu: class_weight='balanced' is handled directly inside the
model constructors in baseline.py/boosting.py. This module implements the
three remaining strategies: SMOTE, ADASYN, and threshold moving.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight


def resample_smote(X, y, random_state: int = 42):
    """SMOTE (U-phase §10 menu item) — synthesises minority-class samples."""
    return SMOTE(random_state=random_state).fit_resample(X, y)


def resample_adasyn(X, y, random_state: int = 42):
    """ADASYN (U-phase §10 menu item) — adaptive synthetic oversampling."""
    return ADASYN(random_state=random_state).fit_resample(X, y)


def balanced_sample_weight(y) -> np.ndarray:
    """Per-row weights making every class contribute equally to the loss.

    Used for XGBoost's ``sample_weight=`` argument at fit time, since
    XGBClassifier has no ``class_weight`` parameter.
    """
    return compute_sample_weight(class_weight="balanced", y=y)


def find_best_threshold_for_class(
    y_true, y_proba: np.ndarray, classes: list[int], target_class: int
) -> float:
    """Threshold-moving (U-phase §10 menu item): sweep the decision threshold
    for ``target_class`` and return the value maximising macro-F1.

    ``y_proba`` columns must align with ``classes`` order (as from a
    fitted estimator's ``.classes_``). Predictions falling below the swept
    threshold fall back to whichever of the *other* classes has the
    highest probability among themselves.
    """
    classes = list(classes)
    target_idx = classes.index(target_class)
    other_classes = np.array([c for i, c in enumerate(classes) if i != target_idx])
    other_proba = np.delete(y_proba, target_idx, axis=1)

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        fallback = other_classes[np.argmax(other_proba, axis=1)]
        y_pred = np.where(y_proba[:, target_idx] >= threshold, target_class, fallback)
        score = f1_score(y_true, y_pred, average="macro")
        if score > best_score:
            best_score, best_threshold = score, threshold
    return float(best_threshold)


def find_gate_optimal_offsets(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[int],
    recall_gate_class: int = 1,
    recall_gate: float = 0.50,
    n_steps_o1: int = 13,
    n_steps_o2: int = 11,
) -> tuple[tuple[float, float] | None, float]:
    """2D additive log-prob offset sweep over the two minority classes.

    Maximises macro-F1 subject to recall(recall_gate_class) >= recall_gate.
    Returns ((o1, o2), best_f1) when feasible; (None, best_unconstrained_f1)
    when no sweep point satisfies the constraint.

    Apply the returned offsets to new data:
        logit = np.log(np.clip(y_proba, 1e-9, 1)).copy()
        logit[:, classes.index(recall_gate_class)] += o1
        logit[:, minority2_idx] += o2   # minority2_idx = classes index of the second minority class
        y_pred = np.array(classes)[logit.argmax(1)]
    """
    classes = list(classes)
    gate_idx = classes.index(recall_gate_class)
    minority2_idx = next(i for i, c in enumerate(classes) if i != gate_idx and c != max(classes))
    # Assumes max(classes) identifies the majority class (holds for UKATGEORIE {1,2,3})

    best_constrained: tuple[tuple[float, float] | None, float] = (None, -1.0)
    best_unconstrained: tuple[tuple[float, float], float] = ((0.0, 0.0), -1.0)

    for o1 in np.linspace(0.0, 3.0, n_steps_o1):
        for o2 in np.linspace(0.0, 2.0, n_steps_o2):
            logit = np.log(np.clip(y_proba, 1e-9, 1)).copy()
            logit[:, gate_idx] += o1
            logit[:, minority2_idx] += o2
            y_pred = np.array(classes)[logit.argmax(1)]
            r = recall_score(y_true, y_pred, labels=[recall_gate_class], average="macro")
            f = f1_score(y_true, y_pred, average="macro")
            if f > best_unconstrained[1]:
                best_unconstrained = ((o1, o2), f)
            if r >= recall_gate and f > best_constrained[1]:
                best_constrained = ((o1, o2), f)

    if best_constrained[0] is not None:
        return best_constrained
    return (None, best_unconstrained[1])
