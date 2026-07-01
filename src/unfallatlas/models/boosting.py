"""Stufe 1 tree ensembles — docs/project/PROJEKTPLAN_SETUP.md ML-Roadmap."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class _ZeroIndexedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Shifts UKATGEORIE's {1,2,3} labels to XGBoost's required {0,1,2} and back.

    XGBClassifier with an explicit ``num_class``/``multi:softprob`` objective
    requires zero-indexed labels and raises ``ValueError: Invalid classes
    inferred from unique values of y`` otherwise (confirmed empirically for
    the pinned xgboost version — labels are not remapped internally). This
    wrapper is the fix noted as a possible requirement in the plan.
    """

    def __init__(self, estimator: XGBClassifier):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        y_arr = np.asarray(y)
        self.classes_ = np.sort(np.unique(y_arr))
        label_to_zero_indexed = {label: i for i, label in enumerate(self.classes_)}
        y_zero_indexed = np.array([label_to_zero_indexed[label] for label in y_arr])
        if "sample_weight" in fit_params:
            self.estimator_.fit(X, y_zero_indexed, sample_weight=fit_params["sample_weight"])
        else:
            self.estimator_.fit(X, y_zero_indexed)
        return self

    def predict(self, X) -> np.ndarray:
        zero_indexed_preds = self.estimator_.predict(X)
        return self.classes_[zero_indexed_preds]

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def build_random_forest_pipeline(
    preprocessor,
    class_weight: str | dict | None = "balanced",
    max_depth: int | None = 20,
    min_samples_leaf: int = 5,
) -> Pipeline:
    """Random Forest with bounded tree size.

    ``max_depth=None`` (sklearn's default) grows every tree to full purity,
    which on ~1.6M training rows produces extremely large trees that overfit
    (especially harmful for the ~1%-share class 1) and are prohibitively slow
    to train. Bounding depth and minimum leaf size is standard practice at
    this data volume and does not weaken validation macro-F1 in practice —
    it typically improves it by preventing leaves fit to a handful of
    training rows.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(preprocessor) -> Pipeline:
    """XGBoost has no ``class_weight``; the class-weighted configuration is
    applied via ``sample_weight`` at ``.fit()`` time in the notebook
    (computed by ``unfallatlas.models.imbalance.balanced_sample_weight``),
    not inside this pipeline builder.

    ``subsample``/``colsample_bytree`` < 1.0 (row/column subsampling per
    boosting round) is standard anti-overfitting regularisation for gradient
    boosting at this data volume — each of the 300 trees sees a different
    80% slice of rows and features, reducing variance without needing a
    validation-based early-stopping loop inside the Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                _ZeroIndexedXGBClassifier(
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="multi:softprob",
                        num_class=3,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="mlogloss",
                    )
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(preprocessor, class_weight: str | dict | None = "balanced") -> Pipeline:
    """Same row/column subsampling rationale as ``build_xgboost_pipeline``.

    LightGBM requires ``bagging_freq`` set alongside ``subsample`` for the
    row-subsampling to actually take effect every boosting round (otherwise
    ``subsample`` is silently ignored).
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                LGBMClassifier(
                    n_estimators=300,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_catboost_pipeline(preprocessor, class_weights: list[float] | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    class_weights=class_weights,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )
