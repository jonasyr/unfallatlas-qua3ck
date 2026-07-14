import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from unfallatlas.models.ordinal import OrdinalClassifier


def _ordered_toy_data(n_per_class=40, seed=0):
    """1-D feature strongly separating the three ordered classes."""
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, (n_per_class, 1))
    X2 = rng.normal(5, 1, (n_per_class, 1))
    X3 = rng.normal(10, 1, (n_per_class, 1))
    X = np.vstack([X1, X2, X3])
    y = np.array([1] * n_per_class + [2] * n_per_class + [3] * n_per_class)
    return X, y


def test_ordinal_classifier_requires_at_least_three_classes():
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    with pytest.raises(ValueError):
        clf.fit(np.array([[0], [1]]), np.array([1, 2]))


def test_ordinal_classifier_fits_k_minus_1_binary_estimators():
    X, y = _ordered_toy_data()
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    clf.fit(X, y)
    assert len(clf.binary_estimators_) == 2  # 3 classes -> 2 thresholds
    assert list(clf.classes_) == [1, 2, 3]


def test_ordinal_classifier_predict_proba_rows_sum_to_one():
    X, y = _ordered_toy_data()
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_ordinal_classifier_recovers_well_separated_classes():
    X, y = _ordered_toy_data()
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.9  # classes are trivially separable by construction
