"""SVM candidate models for the A³ binary-KSI algorithm-selection comparison.

Course reference: docs/course-material/007_Support_Vector_Machines.md.
SVMs require scaled features (§8 "Feature scaling", §21 "Common mistakes") -
always pass a preprocessor built with build_preprocessor(scale_for_linear=True),
never the default tree-oriented build_preprocessor().
"""

from __future__ import annotations

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC


def build_linear_svm_binary_pipeline(
    preprocessor,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    dual: str = "auto",
) -> Pipeline:
    """Linear SVM via LinearSVC (liblinear, squared-hinge loss by default).

    O(m x n) - scales to the full training set (§19 complexity table). No
    kernel-trick support; this is the "fast baseline" SVM candidate, the
    one §9's tip says to try first.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    dual=dual,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_sgd_hinge_binary_pipeline(
    preprocessor,
    alpha: float = 1e-4,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Linear SVM approximation via SGDClassifier(loss="hinge").

    O(m x n), incremental/out-of-core capable (§19 complexity table) - the
    only SVM variant that comfortably trains on the full 1.55M-row training
    set in this project without subsampling.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                SGDClassifier(
                    loss="hinge",
                    alpha=alpha,
                    class_weight=class_weight,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_rbf_svm_binary_pipeline(
    preprocessor,
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Kernel SVM via SVC(kernel="rbf") - the gaussian RBF kernel.

    O(m^2) to O(m^3) in fit time (§19 complexity table) - only feasible on a
    small stratified subsample (thousands, not millions, of rows). Callers
    are responsible for subsampling before calling .fit().
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )
