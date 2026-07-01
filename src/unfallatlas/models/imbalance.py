"""Class-imbalance mitigation strategies compared on the A³ champion model.

U-phase §10 menu: class_weight='balanced' is handled directly inside the
model constructors in baseline.py/boosting.py. This module implements the
three remaining strategies: SMOTE, ADASYN, and threshold moving.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import f1_score
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
