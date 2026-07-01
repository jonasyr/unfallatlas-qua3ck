import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from unfallatlas.models.imbalance import (
    balanced_sample_weight,
    find_best_threshold_for_class,
    resample_adasyn,
    resample_smote,
)


def _imbalanced_toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([1, 2, 3], n, p=[0.05, 0.25, 0.70])
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    return X, y


def test_resample_smote_balances_class_counts():
    X, y = _imbalanced_toy_data()
    X_res, y_res = resample_smote(X, y)
    counts = pd.Series(y_res).value_counts()
    # SMOTE (default 'auto' strategy) balances every class to the majority count.
    assert counts.nunique() == 1
    assert len(y_res) >= len(y)


def test_resample_adasyn_increases_minority_share():
    X, y = _imbalanced_toy_data()
    original_minority_share = (y == 1).mean()
    X_res, y_res = resample_adasyn(X, y)
    resampled_minority_share = (np.asarray(y_res) == 1).mean()
    assert resampled_minority_share > original_minority_share


def test_balanced_sample_weight_upweights_rare_classes():
    y = np.array([1, 2, 2, 3, 3, 3])
    weights = balanced_sample_weight(y)
    assert len(weights) == len(y)
    # Class 1 (rarest) must receive a strictly higher weight than class 3 (most common).
    assert weights[0] > weights[-1]


def test_find_best_threshold_for_class_improves_on_default_half():
    # 10 rows; class 1 is rare (2 rows) with high probability but a naive
    # argmax at threshold 0.5 for the *other* classes would bury it.
    y_true = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    y_proba = np.array(
        [
            [0.4, 0.35, 0.25],
            [0.45, 0.3, 0.25],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
        ]
    )
    threshold = find_best_threshold_for_class(y_true, y_proba, classes=[1, 2, 3], target_class=1)
    assert 0.0 < threshold < 1.0

    # Guard against an index-misalignment bug: after deleting the target class's
    # column, the remaining "other classes" argmax indices must map back to the
    # correct real class labels (2, 3), not a shifted/swapped pair. A misaligned
    # mapping still returns *some* threshold in (0, 1) but produces near-random
    # fallback predictions, so we check macro-F1 directly instead.
    target_idx = [1, 2, 3].index(1)
    other_classes = np.array([2, 3])
    other_proba = np.delete(y_proba, target_idx, axis=1)
    fallback = other_classes[np.argmax(other_proba, axis=1)]
    y_pred = np.where(y_proba[:, target_idx] >= threshold, 1, fallback)
    assert f1_score(y_true, y_pred, average="macro") > 0.9
