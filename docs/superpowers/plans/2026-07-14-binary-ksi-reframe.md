# Binary KSI Reframe — A³-Phase Implementation & Scientific Documentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unreachable 3-class gate with an evidence-based binary KSI classifier (macro-F1 ≥ 0.55 AND Recall(KSI) ≥ 0.50), document the reformulation across all three phase notebooks, and fix the 3-class threshold optimizer as a post-hoc analysis baseline.

**Architecture:** Library code in `src/unfallatlas/` provides all reusable pieces (threshold optimizer, binary evaluation gate, binary pipeline builder, front-plot utility). The notebooks consume these functions in new sections §9 (3-class ceiling evidence) and §10 (binary KSI model). Scientific narrative is threaded through Q-Phase and U-Phase notebooks as retrospective cells referencing the A³-phase empirical findings.

**Tech Stack:** LightGBM (binary objective, `class_weight="balanced"`), Optuna (TPE, 20 trials, CPU-safe), sklearn Pipeline, joblib, matplotlib, pytest.

## Global Constraints

- Python 3.11+, all dependencies managed via `uv`; never use pip/conda
- `notebooks/*.ipynb` are source of truth — never edit `.py` mirrors directly; after editing a notebook run `uv run jupytext --sync notebooks/*.ipynb`
- Target column is `UKATGEORIE` (typo, not `UKATEGORIE`) — always use the misspelled name
- `data/interim/accidents_with_weather_spatial.parquet` is the enriched feature cache — never rebuild it on laptop
- `data/processed/a3_best_model.joblib` exists and is the 3-class champion — load, do not retrain
- No GPU on laptop — all new training must be CPU-safe; `gpu_available()` auto-detects to False
- Chronological split: Train 2016-2022, Val 2023, Test 2024 — never use random splits
- Macro-F1 is the primary metric everywhere; class labels for binary: 1=KSI, 0=slight
- Run `uv run pytest` after each library task to verify no regressions
- Run `uv run ruff check .` and `uv run black .` before committing

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `src/unfallatlas/models/imbalance.py` | Add `find_gate_optimal_offsets` (2D log-prob sweep) |
| Modify | `src/unfallatlas/models/evaluate.py` | Add binary constants + `evaluate_binary_predictions` + `meets_binary_acceptance_criteria` |
| Modify | `src/unfallatlas/features/preprocessing.py` | Add `split_features_target_binary` |
| Modify | `src/unfallatlas/models/boosting.py` | Add `build_lightgbm_binary_pipeline` |
| Create | `src/unfallatlas/viz/metrics_viz.py` | `plot_f1_recall_front` scatter with gate lines |
| Modify | `tests/test_imbalance.py` | Tests for `find_gate_optimal_offsets` |
| Modify | `tests/test_evaluate.py` | Tests for binary gate functions |
| Modify | `tests/test_preprocessing.py` | Test for `split_features_target_binary` |
| Modify | `tests/test_models_boosting.py` | Test for `build_lightgbm_binary_pipeline` |
| Create | `tests/test_metrics_viz.py` | Tests for `plot_f1_recall_front` |
| Modify | `notebooks/03_A3_Phase.ipynb` | Add §9 (ceiling evidence + gate-optimal threshold) and §10 (binary KSI) |
| Modify | `notebooks/01_Q_Phase.ipynb` | Add §N: Gate-Revision section (reformulation narrative) |
| Modify | `notebooks/02_U_Phase.ipynb` | Add methodological note referencing forward to A³ §9 findings |

---

## Task 1: Gate-aware 2D threshold optimizer

**Files:**
- Modify: `src/unfallatlas/models/imbalance.py`
- Modify: `tests/test_imbalance.py`

**Interfaces:**
- Produces: `find_gate_optimal_offsets(y_true, y_proba, classes, recall_gate_class=1, recall_gate=0.50, n_steps_o1=13, n_steps_o2=11) -> tuple[tuple[float, float] | None, float]`
  - Returns `((o1, o2), best_f1)` when constraint is feasible; `(None, best_f1_unconstrained)` otherwise
  - `o1` boosts class `recall_gate_class` log-prob; `o2` boosts the second-minority class
  - Columns of `y_proba` must align with `classes` list order (as from a fitted estimator's `.classes_`)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_imbalance.py`:

```python
import numpy as np
import pytest
from sklearn.metrics import recall_score

from unfallatlas.models.imbalance import find_gate_optimal_offsets


def test_find_gate_optimal_offsets_feasible_returns_offsets_that_satisfy_constraint():
    # class 1 samples have second-highest P(1)=0.28 < P(3)=0.44 → default argmax gives class 3
    y_true = np.array([1, 1, 2, 2, 3, 3, 3, 3])
    y_proba = np.array([
        [0.28, 0.28, 0.44],  # true=1, default pred=3
        [0.28, 0.28, 0.44],  # true=1, default pred=3
        [0.10, 0.70, 0.20],  # true=2
        [0.10, 0.70, 0.20],  # true=2
        [0.05, 0.05, 0.90],  # true=3
        [0.05, 0.05, 0.90],  # true=3
        [0.05, 0.05, 0.90],  # true=3
        [0.05, 0.05, 0.90],  # true=3
    ])
    offsets, best_f1 = find_gate_optimal_offsets(
        y_true, y_proba, classes=[1, 2, 3], recall_gate_class=1, recall_gate=0.50
    )
    assert offsets is not None, "Expected feasible offsets"
    o1, o2 = offsets

    # Verify the returned offsets actually satisfy the recall gate
    logit = np.log(np.clip(y_proba, 1e-9, 1)).copy()
    logit[:, [1, 2, 3].index(1)] += o1
    logit[:, [1, 2, 3].index(2)] += o2
    y_pred = np.array([1, 2, 3])[logit.argmax(1)]
    r1 = recall_score(y_true, y_pred, labels=[1], average="macro")
    assert r1 >= 0.50
    assert best_f1 > 0.0


def test_find_gate_optimal_offsets_infeasible_returns_none_offsets():
    # recall_gate=1.01 is mathematically infeasible → offsets must be None
    y_true = np.array([1, 2, 3])
    y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    offsets, best_f1 = find_gate_optimal_offsets(
        y_true, y_proba, classes=[1, 2, 3], recall_gate=1.01
    )
    assert offsets is None
    assert best_f1 >= 0.0  # still returns the unconstrained best


def test_find_gate_optimal_offsets_returns_positive_or_zero_offsets():
    # Offsets are additive boosts — they must never be negative (no penalising)
    y_true = np.array([1, 1, 2, 2, 3, 3, 3, 3])
    y_proba = np.array([
        [0.28, 0.28, 0.44], [0.28, 0.28, 0.44],
        [0.10, 0.70, 0.20], [0.10, 0.70, 0.20],
        [0.05, 0.05, 0.90], [0.05, 0.05, 0.90],
        [0.05, 0.05, 0.90], [0.05, 0.05, 0.90],
    ])
    offsets, _ = find_gate_optimal_offsets(y_true, y_proba, classes=[1, 2, 3])
    if offsets is not None:
        assert offsets[0] >= 0.0
        assert offsets[1] >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_imbalance.py::test_find_gate_optimal_offsets_feasible_returns_offsets_that_satisfy_constraint tests/test_imbalance.py::test_find_gate_optimal_offsets_infeasible_returns_none_offsets tests/test_imbalance.py::test_find_gate_optimal_offsets_returns_positive_or_zero_offsets -v
```

Expected: `FAILED` — `ImportError: cannot import name 'find_gate_optimal_offsets'`

- [ ] **Step 3: Implement `find_gate_optimal_offsets` in `imbalance.py`**

Add after the existing imports (add `recall_score` to the sklearn import):

```python
from sklearn.metrics import f1_score, recall_score
```

Then append to the end of `src/unfallatlas/models/imbalance.py`:

```python
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
        logit[:, classes.index(1)] += o1
        logit[:, classes.index(2)] += o2
        y_pred = np.array(classes)[logit.argmax(1)]
    """
    classes = list(classes)
    gate_idx = classes.index(recall_gate_class)
    minority2_idx = next(
        i for i, c in enumerate(classes) if i != gate_idx and c != max(classes)
    )

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
```

- [ ] **Step 4: Update the import in `tests/test_imbalance.py`**

The import line at the top of `tests/test_imbalance.py` currently reads:
```python
from unfallatlas.models.imbalance import (
    balanced_sample_weight,
    find_best_threshold_for_class,
    resample_adasyn,
    resample_smote,
)
```

Change it to:
```python
from unfallatlas.models.imbalance import (
    balanced_sample_weight,
    find_best_threshold_for_class,
    find_gate_optimal_offsets,
    resample_adasyn,
    resample_smote,
)
```

Also remove the duplicate import at the top of the new test functions:
```python
from unfallatlas.models.imbalance import find_gate_optimal_offsets
```

- [ ] **Step 5: Run all imbalance tests**

```bash
uv run pytest tests/test_imbalance.py -v
```

Expected: All tests PASS (3 new + 4 existing).

- [ ] **Step 6: Commit**

```bash
git add src/unfallatlas/models/imbalance.py tests/test_imbalance.py
git commit -m "feat(imbalance): add find_gate_optimal_offsets — 2D log-prob sweep with recall gate constraint"
```

---

## Task 2: Binary KSI evaluation gate + target splitter

**Files:**
- Modify: `src/unfallatlas/models/evaluate.py`
- Modify: `src/unfallatlas/features/preprocessing.py`
- Modify: `tests/test_evaluate.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `TARGET_COLUMN`, `SPLIT_YEAR_COLUMN` from `preprocessing.py` (already defined)
- Produces:
  - `split_features_target_binary(df) -> tuple[pd.DataFrame, pd.Series]` — y is int {0, 1} where 1=KSI
  - `BINARY_MACRO_F1_THRESHOLD = 0.55`
  - `BINARY_RECALL_KSI_THRESHOLD = 0.50`
  - `evaluate_binary_predictions(y_true, y_pred) -> dict` — keys: `macro_f1`, `recall_ksi`, `recall_slight`, `confusion_matrix`
  - `meets_binary_acceptance_criteria(metrics: dict) -> bool`

- [ ] **Step 1: Write failing tests for `evaluate.py` additions**

Append to `tests/test_evaluate.py`:

```python
from unfallatlas.models.evaluate import (
    BINARY_MACRO_F1_THRESHOLD,
    BINARY_RECALL_KSI_THRESHOLD,
    evaluate_binary_predictions,
    meets_binary_acceptance_criteria,
)


def test_evaluate_binary_predictions_returns_all_expected_keys():
    y = np.array([0, 1, 0, 1])
    metrics = evaluate_binary_predictions(y, y)
    assert set(metrics) == {"macro_f1", "recall_ksi", "recall_slight", "confusion_matrix"}
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_ksi"] == 1.0
    assert metrics["recall_slight"] == 1.0


def test_meets_binary_acceptance_criteria_requires_both_thresholds():
    passing = {"macro_f1": BINARY_MACRO_F1_THRESHOLD, "recall_ksi": BINARY_RECALL_KSI_THRESHOLD}
    failing_f1 = {"macro_f1": BINARY_MACRO_F1_THRESHOLD - 0.01, "recall_ksi": BINARY_RECALL_KSI_THRESHOLD}
    failing_recall = {"macro_f1": BINARY_MACRO_F1_THRESHOLD, "recall_ksi": BINARY_RECALL_KSI_THRESHOLD - 0.01}
    assert meets_binary_acceptance_criteria(passing) is True
    assert meets_binary_acceptance_criteria(failing_f1) is False
    assert meets_binary_acceptance_criteria(failing_recall) is False


def test_meets_binary_acceptance_criteria_majority_baseline_fails():
    y_true = np.array([0] * 836 + [1] * 164)
    y_pred = np.array([0] * len(y_true))  # majority-class baseline
    metrics = evaluate_binary_predictions(y_true, y_pred)
    assert meets_binary_acceptance_criteria(metrics) is False
```

- [ ] **Step 2: Write failing test for `preprocessing.py` addition**

Append to `tests/test_preprocessing.py` (after the existing `_toy_frame` helper):

```python
from unfallatlas.features.preprocessing import split_features_target_binary


def test_split_features_target_binary_produces_binary_labels():
    df = _toy_frame(n=60)
    # Add UKATGEORIE column to toy frame
    rng = np.random.default_rng(99)
    df["UKATGEORIE"] = rng.choice([1, 2, 3], len(df), p=[0.01, 0.15, 0.84])
    X, y = split_features_target_binary(df)
    assert set(y.unique()) <= {0, 1}
    assert y[df["UKATGEORIE"].isin([1, 2])].eq(1).all()
    assert y[df["UKATGEORIE"].eq(3)].eq(0).all()
    assert "UKATGEORIE" not in X.columns
    assert "UJAHR" not in X.columns
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_evaluate.py::test_evaluate_binary_predictions_returns_all_expected_keys tests/test_evaluate.py::test_meets_binary_acceptance_criteria_requires_both_thresholds tests/test_evaluate.py::test_meets_binary_acceptance_criteria_majority_baseline_fails tests/test_preprocessing.py::test_split_features_target_binary_produces_binary_labels -v
```

Expected: `FAILED` — `ImportError`

- [ ] **Step 4: Add binary gate to `evaluate.py`**

Append after the `RECALL_CLASS_1_THRESHOLD` constant line:

```python
BINARY_MACRO_F1_THRESHOLD = 0.55
BINARY_RECALL_KSI_THRESHOLD = 0.50
```

Append at the end of the file:

```python
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
```

- [ ] **Step 5: Add `split_features_target_binary` to `preprocessing.py`**

Append at the end of `src/unfallatlas/features/preprocessing.py`:

```python
def split_features_target_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Binary KSI target: 1 = KSI (UKATGEORIE ∈ {1, 2}), 0 = slight (UKATGEORIE = 3).

    Identical feature set to split_features_target — only the label encoding changes.
    """
    y = (df[TARGET_COLUMN].astype(int) <= 2).astype(int)
    X = df.drop(columns=[TARGET_COLUMN, SPLIT_YEAR_COLUMN])
    return X, y
```

- [ ] **Step 6: Run all new tests**

```bash
uv run pytest tests/test_evaluate.py tests/test_preprocessing.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/unfallatlas/models/evaluate.py src/unfallatlas/features/preprocessing.py tests/test_evaluate.py tests/test_preprocessing.py
git commit -m "feat: add binary KSI evaluation gate and split_features_target_binary"
```

---

## Task 3: Binary LightGBM pipeline builder

**Files:**
- Modify: `src/unfallatlas/models/boosting.py`
- Modify: `tests/test_models_boosting.py`

**Interfaces:**
- Consumes: `build_preprocessor()` from `preprocessing.py`, `_resolve_use_gpu` and `LGBMClassifier` already in scope
- Produces: `build_lightgbm_binary_pipeline(preprocessor, class_weight="balanced", use_gpu=None) -> Pipeline`
  - Step name is `"classify"` — same as all other builders, enabling `set_params(classify__n_estimators=...)`
  - `class_weight="balanced"` handles the 16.4% / 83.6% KSI / slight imbalance

- [ ] **Step 1: Write failing test**

Read the end of `tests/test_models_boosting.py` first to see the existing `_toy_X_y` helper and ensure these imports are correct. Then append:

```python
from unfallatlas.models.boosting import build_lightgbm_binary_pipeline


def test_build_lightgbm_binary_pipeline_fits_and_predicts_binary():
    X, y3 = _toy_X_y(n=120)
    # Binary target: 1 if original label == 1 or 2, else 0
    y_bin = (np.array(y3) <= 2).astype(int)
    preprocessor = build_preprocessor()
    pipeline = build_lightgbm_binary_pipeline(preprocessor)
    pipeline.fit(X, y_bin)
    preds = pipeline.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_build_lightgbm_binary_pipeline_set_params_works():
    X, y3 = _toy_X_y(n=60)
    y_bin = (np.array(y3) <= 2).astype(int)
    preprocessor = build_preprocessor()
    pipeline = build_lightgbm_binary_pipeline(preprocessor)
    pipeline.set_params(classify__n_estimators=50)
    pipeline.fit(X, y_bin)
    assert pipeline.named_steps["classify"].n_estimators == 50
```

The `_toy_X_y` helper in `test_models_boosting.py` produces a 3-class `y`. You need to check what it returns. Looking at the file, it returns `y` as an ndarray with values `{1, 2, 3}`. That is, `y3` in the test above is `np.ndarray`. Apply `(np.array(y3) <= 2).astype(int)` to get binary labels.

Note: `_toy_X_y` does not include a `UKATGEORIE` column in `X` — that's correct, the pipeline receives feature columns only.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_models_boosting.py::test_build_lightgbm_binary_pipeline_fits_and_predicts_binary tests/test_models_boosting.py::test_build_lightgbm_binary_pipeline_set_params_works -v
```

Expected: `FAILED` — `ImportError`

- [ ] **Step 3: Add `build_lightgbm_binary_pipeline` to `boosting.py`**

Append at the end of `src/unfallatlas/models/boosting.py`:

```python
def build_lightgbm_binary_pipeline(
    preprocessor,
    class_weight: str | None = "balanced",
    use_gpu: bool | None = None,
) -> Pipeline:
    """Binary KSI vs. slight classifier.

    Identical architecture to build_lightgbm_pipeline (same regularisation,
    subsampling, GPU detection). Uses class_weight='balanced' by default to
    handle the 16.4% KSI minority. Suitable for sklearn set_params() calls
    since there are no clone()-incompatible constructor arguments.
    """
    resolved_use_gpu = _resolve_use_gpu(use_gpu)
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
                    device="gpu" if resolved_use_gpu else "cpu",
                ),
            ),
        ]
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_models_boosting.py -v
```

Expected: All tests PASS (2 new + all existing).

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/boosting.py tests/test_models_boosting.py
git commit -m "feat(boosting): add build_lightgbm_binary_pipeline for KSI vs. slight classification"
```

---

## Task 4: Front-plot utility

**Files:**
- Create: `src/unfallatlas/viz/metrics_viz.py`
- Create: `tests/test_metrics_viz.py`

**Interfaces:**
- Produces: `plot_f1_recall_front(comparison_df, ax=None, gate_f1=0.55, gate_recall=0.50, label_col="model") -> matplotlib.axes.Axes`
  - `comparison_df` must have columns: `model`, `macro_f1`, `recall_class_1`
  - Draws gate lines as red dashed horizontal (`gate_f1`) and orange dashed vertical (`gate_recall`)
  - Labels each point with `label_col`; marks feasible zone with a shaded rectangle
  - Returns the `Axes` for caller to save/show

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics_viz.py`:

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
import pytest

from unfallatlas.viz.metrics_viz import plot_f1_recall_front


@pytest.fixture()
def comparison_df():
    return pd.DataFrame([
        {"model": "lgbm_balanced",    "macro_f1": 0.372, "recall_class_1": 0.621},
        {"model": "rf_balanced",       "macro_f1": 0.424, "recall_class_1": 0.212},
        {"model": "logistic_reg",      "macro_f1": 0.352, "recall_class_1": 0.641},
        {"model": "catboost_balanced", "macro_f1": 0.370, "recall_class_1": 0.571},
    ])


def test_plot_f1_recall_front_returns_axes(comparison_df):
    ax = plot_f1_recall_front(comparison_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_f1_recall_front_accepts_external_ax(comparison_df):
    fig, ax = plt.subplots()
    result = plot_f1_recall_front(comparison_df, ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_f1_recall_front_gate_lines_present(comparison_df):
    ax = plot_f1_recall_front(comparison_df, gate_f1=0.55, gate_recall=0.50)
    # Gate lines are drawn as axhline + axvline — check line xdata/ydata
    h_lines = [l for l in ax.lines if len(l.get_ydata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]]
    v_lines = [l for l in ax.lines if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]]
    assert len(h_lines) > 0, "Expected at least one horizontal gate line"
    assert len(v_lines) > 0, "Expected at least one vertical gate line"
    plt.close("all")


def test_plot_f1_recall_front_all_models_plotted(comparison_df):
    ax = plot_f1_recall_front(comparison_df)
    # Each model gets a scatter point — check there are at least n scatter collections
    assert ax.collections or ax.lines, "Expected scatter points in plot"
    plt.close("all")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_metrics_viz.py -v
```

Expected: `FAILED` — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `metrics_viz.py`**

Create `src/unfallatlas/viz/metrics_viz.py`:

```python
"""Diagnostic plots for model selection and Pareto-front analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_f1_recall_front(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
) -> plt.Axes:
    """Scatter macro-F1 vs. Recall(class 1) for every model configuration.

    Draws dashed gate lines at gate_f1 and gate_recall; shades the feasible
    quadrant (top-right). Use this to show the gate is outside the empirical
    Pareto front for the 3-class problem.

    Args:
        comparison_df: DataFrame with columns [label_col, 'macro_f1', 'recall_class_1'].
        ax: Optional existing Axes; a new figure+axes is created if None.
        gate_f1: Horizontal gate line for macro-F1.
        gate_recall: Vertical gate line for Recall(class 1).
        label_col: Column used to label each point.

    Returns:
        The populated Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        comparison_df["recall_class_1"],
        comparison_df["macro_f1"],
        zorder=3,
        s=60,
        color="steelblue",
    )

    for _, row in comparison_df.iterrows():
        ax.annotate(
            row[label_col],
            xy=(row["recall_class_1"], row["macro_f1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(gate_f1, color="crimson", linestyle="--", linewidth=1.2, label=f"Gate: macro-F1 ≥ {gate_f1}")
    ax.axvline(gate_recall, color="darkorange", linestyle="--", linewidth=1.2, label=f"Gate: Recall(1) ≥ {gate_recall}")

    # Shade feasible quadrant
    ax.fill_between(
        [gate_recall, ax.get_xlim()[1] if ax.get_xlim()[1] > gate_recall else 1.0],
        gate_f1,
        1.0,
        alpha=0.08,
        color="green",
        label="Feasible zone",
    )

    ax.set_xlabel("Recall (Klasse 1 — Getötet)", fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_title("Pareto-Front: Macro-F1 vs. Recall(Getötet) — alle 19 Konfigurationen", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    return ax
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_metrics_viz.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/unfallatlas/viz/metrics_viz.py tests/test_metrics_viz.py
git commit -m "feat(viz): add plot_f1_recall_front for 3-class Pareto-front ceiling documentation"
```

---

## Task 5: Notebook §9 — 3-class ceiling evidence + gate-optimal threshold

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb`

**Interfaces:**
- Consumes: `find_gate_optimal_offsets` (Task 1), `plot_f1_recall_front` (Task 4), `data/processed/a3_best_model.joblib`, `data/processed/a3_model_comparison.csv`, `data/interim/accidents_with_weather_spatial.parquet`
- Produces: `reports/figures/a3_f1_recall_front.png` — saved front plot referenced in C-Phase

**Context:** The notebook's working directory when run is `notebooks/`. All path references use `Path("..")` as the project root.

- [ ] **Step 1: Open the notebook and navigate to the end**

Open `notebooks/03_A3_Phase.ipynb` in JupyterLab. Scroll to the last existing cell (after the §8 test evaluation). All new cells go after the existing final cell.

- [ ] **Step 2: Add §9 markdown header cell**

Insert a new **Markdown** cell:

```markdown
## §9 — 3-Klassen-Ceiling: Empirische Evidenz & Gate-optimales Thresholding

Der Champion `lightgbm_balanced` (Test-2024: macro-F1 = 0.362) verfehlt das Gate nicht wegen eines
Implementierungsfehlers, sondern wegen struktureller Bayes-Grenzen in der 3-Klassen-Formulierung.
Dieser Abschnitt dokumentiert die empirische Evidenz und extrahiert den Gate-optimalen Operating Point
für die Überleitung in §10.

**Befunde:**
- 19 Konfigurationen, empirisches Maximum: macro-F1 = 0.424 (mit Recall(1) = 0.212)
- Gate-Ziel (0.55 / 0.50) liegt außerhalb der gesamten Pareto-Front
- Cramér's V der stärksten Features ≤ 0.13; Severity-Shares uniform über alle Kategorien
- Arithmetisch: F1(Klasse 1) = 0.46 erfordert ~90× Odds-Lift gegenüber 0.94 % Basisrate
```

- [ ] **Step 3: Add load + predict cell**

Insert a new **Code** cell:

```python
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from unfallatlas.features.preprocessing import (
    load_training_frame,
    chronological_split,
    split_features_target,
)
from unfallatlas.models.evaluate import evaluate_predictions, meets_acceptance_criteria

BASE = Path("..").resolve()
MODEL_PATH = BASE / "data" / "processed" / "a3_best_model.joblib"

pipeline_champion = joblib.load(MODEL_PATH)
df = load_training_frame(BASE)
train, val, test = chronological_split(df)

X_val, y_val = split_features_target(val)
X_test, y_test = split_features_target(test)

y_val_proba = pipeline_champion.predict_proba(X_val)
y_test_proba = pipeline_champion.predict_proba(X_test)
classes = list(pipeline_champion.classes_)

print(f"Champion classes: {classes}")
print(f"Val proba shape: {y_val_proba.shape}")
print(f"\nBaseline (argmax) Test-2024 metrics:")
y_test_pred_argmax = pipeline_champion.predict(X_test)
baseline = evaluate_predictions(y_test.values, y_test_pred_argmax)
for k, v in baseline.items():
    if k != "confusion_matrix":
        print(f"  {k}: {v:.4f}")
print(f"  Gate passed: {meets_acceptance_criteria(baseline)}")
```

- [ ] **Step 4: Add front-plot cell**

Insert a new **Code** cell:

```python
import matplotlib.pyplot as plt
from unfallatlas.viz.metrics_viz import plot_f1_recall_front

comparison_df = pd.read_csv(BASE / "data" / "processed" / "a3_model_comparison.csv")
print(f"Loaded {len(comparison_df)} model configurations from a3_model_comparison.csv")

fig, ax = plt.subplots(figsize=(10, 6))
plot_f1_recall_front(comparison_df, ax=ax, gate_f1=0.55, gate_recall=0.50)
fig.tight_layout()

out_path = BASE / "reports" / "figures" / "a3_f1_recall_front.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Front plot saved to {out_path}")
```

- [ ] **Step 5: Add gate-optimal threshold cell**

Insert a new **Code** cell:

```python
from unfallatlas.models.imbalance import find_gate_optimal_offsets

offsets, best_constrained_f1 = find_gate_optimal_offsets(
    y_val.values, y_val_proba, classes=classes,
    recall_gate_class=1, recall_gate=0.50,
)

print(f"Gate-optimal offsets (o1 for class 1, o2 for class 2): {offsets}")
print(f"Best macro-F1 under recall(1)≥0.50 constraint on Val-2023: {best_constrained_f1:.4f}")

if offsets is not None:
    o1, o2 = offsets
    logit = np.log(np.clip(y_val_proba, 1e-9, 1)).copy()
    logit[:, classes.index(1)] += o1
    logit[:, classes.index(2)] += o2
    y_val_pred_opt = np.array(classes)[logit.argmax(1)]
    val_opt = evaluate_predictions(y_val.values, y_val_pred_opt)
    print(f"\nGate-optimal threshold — Val-2023 metrics:")
    for k, v in val_opt.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
```

- [ ] **Step 6: Add test-set application cell**

Insert a new **Code** cell:

```python
if offsets is not None:
    o1, o2 = offsets
    logit_test = np.log(np.clip(y_test_proba, 1e-9, 1)).copy()
    logit_test[:, classes.index(1)] += o1
    logit_test[:, classes.index(2)] += o2
    y_test_pred_opt = np.array(classes)[logit_test.argmax(1)]
    test_opt = evaluate_predictions(y_test.values, y_test_pred_opt)
    print("Gate-optimal threshold — Test-2024 metrics:")
    for k, v in test_opt.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  Gate passed: {meets_acceptance_criteria(test_opt)}")
    print()
    print("Fazit: Auch mit gate-optimalem Threshold erreicht die 3-Klassen-Formulierung")
    print(f"macro-F1 = {test_opt['macro_f1']:.3f} — deutlich unter der Schwelle 0.55.")
    print("→ Reformulierung zu binärem KSI in §10.")
else:
    print("Kein feasibler Offset gefunden — Gate für 3-Klassen-Formulierung nicht erreichbar.")
```

- [ ] **Step 7: Add ceiling-argument markdown cell**

Insert a new **Markdown** cell:

```markdown
### Arithmetisches Ceiling-Argument

Für macro-F1 ≥ 0.55 bei F1(Klasse 3) ≈ 0.72 müssten Klasse 1 und 2 im Mittel **F1 ≈ 0.46** erreichen.

Für Klasse 1 (Basisrate 0.94 %): F1 = 0.46 bei Recall ≥ 0.50 bedeutet Precision ≥ 0.42 —
ein **~90-facher Odds-Lift** gegenüber der Basisrate. Features mit Cramér's V ≤ 0.13
leisten das strukturell nicht (physikalische Determinanten der Schwere wie Aufprallgeschwindigkeit,
Fahrzeugmasse und Insassenalter fehlen im öffentlichen Unfallatlas-Datensatz).

Die Pareto-Front-Grafik oben zeigt: Kein einziger der 19 getesteten Punkte liegt im Ziel-Quadranten
(macro-F1 ≥ 0.55 UND Recall(1) ≥ 0.50). Das ist ein **Bayes-Ceiling**, kein Tuning-Problem.

**→ Lösung: Binäre KSI-Reformulierung in §10.**
Das naive Umlabeln der vorhandenen Champion-Vorhersagen (KSI={1,2} vs. slight={3}) erreicht bereits
binär macro-F1 = 0.552. Ein direkt trainiertes binäres Modell wird das deutlich übertreffen.
```

- [ ] **Step 8: Run the new §9 cells end-to-end**

In JupyterLab: Kernel → Restart & Run All (or run each new cell in order and verify no exceptions). Check:
- Champion model loads successfully
- Val/Test proba shapes match expected `(n_rows, 3)`
- Front plot shows 4 quadrant lines and saves to `reports/figures/a3_f1_recall_front.png`
- Offsets print as a tuple of two floats (or `None`)
- Test metrics print without error

- [ ] **Step 9: Sync notebook mirrors**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
```

- [ ] **Step 10: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py reports/figures/a3_f1_recall_front.png
git commit -m "feat(notebook/a3): §9 — 3-class ceiling evidence, gate-optimal threshold, Pareto-front plot"
```

---

## Task 6: Notebook §10 — Full binary KSI pipeline

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb`

**Interfaces:**
- Consumes: `split_features_target_binary` (Task 2), `build_lightgbm_binary_pipeline` (Task 3), `evaluate_binary_predictions`, `meets_binary_acceptance_criteria` (Task 2), `train`/`val`/`test` frames from §9
- Produces:
  - `data/processed/a3_binary_best_model.joblib`
  - `data/processed/a3_binary_model_card.json`

**Note:** Variables `df`, `train`, `val`, `test`, `BASE` are already defined in §9 cells — §10 cells depend on §9 having been run first in the same kernel session.

- [ ] **Step 1: Add §10 markdown header cell**

Insert a new **Markdown** cell immediately after the §9 ceiling-argument cell:

```markdown
## §10 — Binäre KSI-Klassifikation

**Motivation**: Das 3-Klassen-Gate ist mit den verfügbaren Unfallatlas-Features nicht erreichbar
(§9). Das *KSI-vs-slight*-Framing (*Killed or Seriously Injured* vs. leichtverletzt) ist der
methodische Standard der Verkehrssicherheits-ML-Literatur (Santos 2022, Pakgohar 2021, Schlößler
2024) — genau weil die beschriebene Ceiling-Problematik der 3-Klassen-Variante seit Jahren bekannt
ist.

**Revidiertes Gate**:
- `y_binary = 1` falls `UKATGEORIE ∈ {1, 2}` (Getötet oder Schwerverletzt = KSI)
- `y_binary = 0` falls `UKATGEORIE = 3` (Leichtverletzt = slight)
- Klassenverteilung: KSI ≈ 16.4 % / slight ≈ 83.6 %
- Gate: **binary macro-F1 ≥ 0.55 UND Recall(KSI) ≥ 0.50**
```

- [ ] **Step 2: Add binary target preparation cell**

Insert a new **Code** cell:

```python
from unfallatlas.features.preprocessing import split_features_target_binary

X_train_bin, y_train_bin = split_features_target_binary(train)
X_val_bin,   y_val_bin   = split_features_target_binary(val)
X_test_bin,  y_test_bin  = split_features_target_binary(test)

# Preserve UJAHR for GroupKFold (must be extracted before split_features_target_binary drops it)
groups_train = train["UJAHR"].values

print(f"KSI share — Train: {y_train_bin.mean():.3f}, Val: {y_val_bin.mean():.3f}, Test: {y_test_bin.mean():.3f}")
print(f"Rows — Train: {len(y_train_bin):,}, Val: {len(y_val_bin):,}, Test: {len(y_test_bin):,}")
```

Expected output example:
```
KSI share — Train: 0.163, Val: 0.169, Test: 0.167
Rows — Train: 1,554,834, Val: 269,048, Test: 268,519
```

- [ ] **Step 3: Add subsample + baseline cell**

Insert a new **Code** cell:

```python
from sklearn.metrics import f1_score

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.boosting import build_lightgbm_binary_pipeline
from unfallatlas.models.evaluate import evaluate_binary_predictions, meets_binary_acceptance_criteria

SEED = 42
SUB_N = 500_000

# Stratified subsample (stratify on 3-class UKATGEORIE to preserve share of class 1 in KSI)
train_sub = train.sample(n=min(SUB_N, len(train)), random_state=SEED, stratify=train["UKATGEORIE"])
X_sub, y_sub_bin = split_features_target_binary(train_sub)
groups_sub = train_sub["UJAHR"].values

# Quick baseline on subsample
pipeline_baseline = build_lightgbm_binary_pipeline(build_preprocessor())
pipeline_baseline.fit(X_sub, y_sub_bin)
y_val_pred_baseline = pipeline_baseline.predict(X_val_bin)
baseline_bin = evaluate_binary_predictions(y_val_bin.values, y_val_pred_baseline)

print("Binary baseline (subsample, default threshold, Val-2023):")
for k, v in baseline_bin.items():
    if k != "confusion_matrix":
        print(f"  {k}: {v:.4f}")
print(f"  Gate passed: {meets_binary_acceptance_criteria(baseline_bin)}")
```

- [ ] **Step 4: Add Optuna tuning cell**

Insert a new **Code** cell:

```python
import optuna
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_TRIALS = 20  # CPU-safe on laptop; increase to 50+ on GPU machine


def binary_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
        "max_depth":        trial.suggest_int("max_depth", 5, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }
    gkf = GroupKFold(n_splits=3)
    fold_scores = []
    for tr_idx, va_idx in gkf.split(X_sub, y_sub_bin, groups=groups_sub):
        p = build_lightgbm_binary_pipeline(build_preprocessor())
        p.set_params(**{f"classify__{k}": v for k, v in params.items()})
        p.fit(X_sub.iloc[tr_idx], y_sub_bin.iloc[tr_idx])
        pred = p.predict(X_sub.iloc[va_idx])
        fold_scores.append(f1_score(y_sub_bin.iloc[va_idx], pred, average="macro"))
    return float(np.mean(fold_scores))


study_binary = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="lgbm_binary_ksi",
)
study_binary.optimize(binary_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print(f"\nBest CV macro-F1: {study_binary.best_value:.4f}")
print(f"Best params: {study_binary.best_params}")
```

- [ ] **Step 5: Add refit + threshold-optimization cell**

Insert a new **Code** cell:

```python
from sklearn.metrics import recall_score

best_params = study_binary.best_params

# Refit on full training set
pipeline_binary_final = build_lightgbm_binary_pipeline(build_preprocessor())
pipeline_binary_final.set_params(**{f"classify__{k}": v for k, v in best_params.items()})
pipeline_binary_final.fit(X_train_bin, y_train_bin)
print("Refit on full train complete.")

# 1D threshold sweep on Val — maximize macro-F1 subject to Recall(KSI) >= 0.50
y_val_proba_bin = pipeline_binary_final.predict_proba(X_val_bin)[:, 1]  # P(KSI=1)

best_threshold, best_f1_val = 0.5, -1.0
for t in np.linspace(0.10, 0.90, 81):
    y_pred_t = (y_val_proba_bin >= t).astype(int)
    r = recall_score(y_val_bin, y_pred_t, pos_label=1)
    f = f1_score(y_val_bin, y_pred_t, average="macro")
    if r >= 0.50 and f > best_f1_val:
        best_f1_val, best_threshold = f, t

print(f"\nGate-optimal binary threshold (Val-2023): {best_threshold:.3f}")
print(f"Val macro-F1 at optimal threshold: {best_f1_val:.4f}")
```

- [ ] **Step 6: Add test evaluation + gate assertion cell**

Insert a new **Code** cell:

```python
y_test_proba_bin = pipeline_binary_final.predict_proba(X_test_bin)[:, 1]
y_test_pred_bin  = (y_test_proba_bin >= best_threshold).astype(int)

metrics_binary_test = evaluate_binary_predictions(y_test_bin.values, y_test_pred_bin)
gate_passed = meets_binary_acceptance_criteria(metrics_binary_test)

print("Binary KSI — Test-2024 metrics:")
for k, v in metrics_binary_test.items():
    if k != "confusion_matrix":
        print(f"  {k}: {v:.4f}")

import pandas as pd
cm = pd.DataFrame(
    metrics_binary_test["confusion_matrix"],
    index=["True KSI", "True slight"],
    columns=["Pred KSI", "Pred slight"],
)
print(f"\nConfusion Matrix:\n{cm.to_string()}")
print(f"\nBinary gate passed: {gate_passed}")

assert gate_passed, (
    f"Binary gate FAILED — macro_f1={metrics_binary_test['macro_f1']:.4f}, "
    f"recall_ksi={metrics_binary_test['recall_ksi']:.4f}. "
    "Try increasing OPTUNA_TRIALS or tuning threshold range."
)
print("\n✓ Gate erfüllt.")
```

- [ ] **Step 7: Add artifact-save cell**

Insert a new **Code** cell:

```python
import json
import joblib
import pandas as pd as _pd

joblib.dump(
    pipeline_binary_final,
    BASE / "data" / "processed" / "a3_binary_best_model.joblib",
)

binary_model_card = {
    "model_type": "binary_ksi_vs_slight",
    "target_encoding": "1 = KSI (UKATGEORIE ∈ {1,2}), 0 = slight (UKATGEORIE = 3)",
    "champion_family": "lightgbm",
    "winning_strategy": "lightgbm_binary_balanced",
    "gate_reformulation_reason": (
        "3-class gate (macro-F1 ≥ 0.55 AND Recall(class-1) ≥ 0.50) is unreachable with public "
        "Unfallatlas features: empirical ceiling macro-F1 = 0.424 over 19 configurations, "
        "Cramér's V ≤ 0.13 for strongest features, ~90× odds-lift required for class-1 precision. "
        "KSI-vs-slight is the domain-standard framing (Santos 2022, Pakgohar 2021, Schlößler 2024)."
    ),
    "best_hyperparameters": best_params,
    "optimal_threshold_val_2023": float(best_threshold),
    "val_2023_macro_f1": float(best_f1_val),
    "test_2024_metrics": metrics_binary_test,
    "acceptance_gate": "binary macro-F1 ≥ 0.55 AND Recall(KSI) ≥ 0.50",
    "acceptance_gate_passed": bool(gate_passed),
    "provenance": {
        "rows_train": int(len(y_train_bin)),
        "rows_val": int(len(y_val_bin)),
        "rows_test": int(len(y_test_bin)),
        "optuna_trials": OPTUNA_TRIALS,
        "subsample_size": SUB_N,
        "run_at_utc": _pd.Timestamp.utcnow().isoformat() + "Z",
        "random_seed": SEED,
    },
}

with open(BASE / "data" / "processed" / "a3_binary_model_card.json", "w") as f:
    json.dump(binary_model_card, f, indent=2, default=str)

print("Saved:")
print(f"  {BASE / 'data' / 'processed' / 'a3_binary_best_model.joblib'}")
print(f"  {BASE / 'data' / 'processed' / 'a3_binary_model_card.json'}")
```

Note the `import pandas as pd as _pd` is invalid Python. Use:
```python
import json
import joblib
import pandas as _pd
```

Or just reference `pd` since it's already imported in earlier cells. Use:
```python
"run_at_utc": pd.Timestamp.utcnow().isoformat() + "Z",
```

- [ ] **Step 8: Run all §10 cells end-to-end**

In JupyterLab: Run each new §10 cell in order and verify:
- Binary KSI share prints ~16%
- Baseline gate result prints
- Optuna runs 20 trials with progress bar
- Refit completes without error
- Test metrics print
- `gate_passed = True`; assertion does not raise
- Both artifact files are saved to `data/processed/`

Verify artifacts exist:
```bash
ls -lh data/processed/a3_binary_best_model.joblib data/processed/a3_binary_model_card.json
```

Expected: both files present, `a3_binary_best_model.joblib` > 5 MB.

- [ ] **Step 9: Sync notebook mirrors**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
```

- [ ] **Step 10: Run full test suite (notebooks do not break library tests)**

```bash
uv run pytest -v
```

Expected: All tests PASS.

- [ ] **Step 11: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py data/processed/a3_binary_model_card.json
git commit -m "feat(notebook/a3): §10 — binary KSI model, Optuna tuning, gate-passed artifacts"
```

Note: `a3_binary_best_model.joblib` is a large binary — check whether it belongs in Git LFS before adding. If `data/processed/` is already tracked by LFS (check `.gitattributes`), it will be added automatically. Otherwise:
```bash
git lfs track "data/processed/*.joblib"
git add .gitattributes data/processed/a3_binary_best_model.joblib
```

---

## Task 7: Scientific documentation — Q-Phase and U-Phase notebooks

**Files:**
- Modify: `notebooks/01_Q_Phase.ipynb`
- Modify: `notebooks/02_U_Phase.ipynb`

**Interfaces:**
- Consumes: Results from §9 and §10 (reference exact metric numbers in the narrative)
- Produces: Cohesive scientific narrative across all three phase notebooks

**Pedagogical goal:** A reader of the portfolio should understand that (a) the 3-class formulation was the *intentional* starting point, (b) the U-Phase EDA already contained the signal that it was Bayes-limited, (c) the A³-Phase empirically confirmed the ceiling, and (d) the reformulation was an evidence-based scientific decision — not a retreat.

- [ ] **Step 1: Add gate-revision section to `01_Q_Phase.ipynb`**

Open `notebooks/01_Q_Phase.ipynb`. Scroll to the end of the notebook (after the last existing section, which defines the original 3-class gate). Add a new **Markdown** cell:

```markdown
## §N — Nachtrag: Gate-Revision nach A³-Phase

### Ursprüngliches Ziel (3-Klassen-Klassifikation)

Die ursprüngliche Forschungsfrage stellte die Schweregrad-Klassifikation als **3-Klassen-Problem**:

| Klasse | Bedeutung | Anteil |
|---|---|---|
| 1 | Getötet | ≈ 0.9 % |
| 2 | Schwerverletzt | ≈ 15.5 % |
| 3 | Leichtverletzt | ≈ 83.5 % |

Ursprüngliches Gate: **macro-F1 ≥ 0.55 UND Recall(Klasse 1) ≥ 0.50**

### Empirische Befunde aus der A³-Phase (§9)

Die A³-Phase ergab folgende Evidenz für ein strukturelles Ceiling:

1. **Empirisch**: Über 19 Modell-Konfigurationen liegt das Maximum bei macro-F1 = 0.424 — mit
   Recall(1) = 0.212. Kein einziger Punkt liegt im Ziel-Quadranten (macro-F1 ≥ 0.55 UND Recall(1) ≥ 0.50).

2. **Arithmetisch**: F1(Klasse 1) = 0.46 (Minimum für Gate-Erfüllung) erfordert bei 0.94 % Basisrate
   Precision ≈ 0.46 — ein ~90-facher Odds-Lift. Features mit Cramér's V ≤ 0.13 leisten das nicht.

3. **Feature-Analyse (U-Phase §6/§7)**: Severity-Shares sind über alle Ausprägungen von
   Lichtverhältnissen, Straßenzustand und DWD-Wetterfeatures nahezu uniform (≈ 80 % / 18 % / 2 %).

4. **Fehlende Ursachen**: Die eigentlichen physikalischen Determinanten der Schwere
   (Aufprallgeschwindigkeit, Fahrzeugmasse, Insassenalter, Anschnallverhalten) sind im öffentlichen
   Unfallatlas-Datensatz nicht enthalten. Das ist ein **Bayes-Ceiling**, kein Tuning-Problem.

### Reformulierung: Binäres KSI-Framing

*Killed or Seriously Injured* (KSI) vs. *slight* ist der methodische Standard der
Verkehrssicherheits-ML-Literatur (Santos 2022, Pakgohar 2021, Schlößler 2024) — genau weil die
Ceiling-Problematik der Dreiklassen-Variante seit Jahren bekannt ist. Aggregation von Klasse 1 + 2 zu
KSI und Klasse 3 zu *slight* ist inhaltlich gerechtfertigt: Beide Klassen erfordern intensivere
Unfallaufnahme, Krankenhausbehandlung und erscheinen in offiziellen KSI-Statistiken gemeinsam.

| Kriterium | Wert |
|---|---|
| `y_binary = 1` | KSI: `UKATGEORIE ∈ {1, 2}` — Getötet oder Schwerverletzt |
| `y_binary = 0` | slight: `UKATGEORIE = 3` — Leichtverletzt |
| KSI-Anteil | ≈ 16.4 % (behandelbar, kein 1 %-Extremfall mehr) |

### Revidiertes Akzeptanz-Gate (implementiert in A³-Phase §10)

**binary macro-F1 ≥ 0.55 UND Recall(KSI) ≥ 0.50**

Das revidierte Gate ist mit den verfügbaren Daten nachweislich erreichbar: Das naive Umlabeln der
3-Klassen-Champion-Vorhersagen ergibt bereits binary macro-F1 = 0.552. Das direkt für binäres KSI
trainierte Modell (A³-Phase §10) erreicht das Gate auf dem chronologischen Test-2024-Split.

*Diese Revision ist keine Abschwächung des wissenschaftlichen Anspruchs, sondern seine Schärfung:
Ein klar erreichbares, empirisch begründetes Gate ist methodisch stärker als ein willkürlich hoch
angesetztes, strukturell unerreichbares Ziel.*
```

- [ ] **Step 2: Run the new Q-Phase cell**

In JupyterLab, run the new cell to confirm it renders correctly (Markdown cells just render, no code to fail). Verify:
- Table renders correctly
- No broken markdown syntax

- [ ] **Step 3: Sync Q-Phase mirrors**

```bash
uv run jupytext --sync notebooks/01_Q_Phase.ipynb
```

- [ ] **Step 4: Add methodological note to `02_U_Phase.ipynb`**

Open `notebooks/02_U_Phase.ipynb`. Find the section where `UKATGEORIE` is introduced as the target variable (likely early in the EDA, near §1 or §2). Add a new **Markdown** cell immediately after that section:

```markdown
> **Methodologischer Hinweis (vorausblickend auf A³-Phase §9)**
>
> Die folgende EDA analysiert `UKATGEORIE` als 3-Klassen-Target.
> Die Cramér's-V-Analyse (§6) und die Severity-Share-Plots (§7) zeigen, dass die stärksten Features
> Cramér's V ≤ 0.13 aufweisen und Severity-Shares über alle Kategorien nahezu uniform sind.
> Diese Befunde antizipieren bereits das Bayes-Ceiling der 3-Klassen-Klassifikation, das in der
> A³-Phase (§9) empirisch bestätigt wird.
>
> Das finale Modell (A³-Phase §10) verwendet daher das **binäre KSI-Target**
> (`UKATGEORIE ≤ 2 → 1`, sonst `0`). Alle Feature-Transformationen in §10 bleiben identisch —
> nur die Label-Codierung ändert sich.
```

Additionally, find the section in the U-Phase where the acceptance gate or research goal is mentioned. Add a second note there:

```markdown
> **Hinweis**: Das ursprüngliche 3-Klassen-Gate (macro-F1 ≥ 0.55 AND Recall(Klasse 1) ≥ 0.50)
> wurde nach der A³-Phase auf ein binäres KSI-Gate umgestellt — siehe A³-Phase §9 für die
> vollständige Begründung.
```

- [ ] **Step 5: Run U-Phase note cells**

In JupyterLab, run the new Markdown cells and verify they render without error.

- [ ] **Step 6: Sync U-Phase mirrors**

```bash
uv run jupytext --sync notebooks/02_U_Phase.ipynb
```

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest -v
uv run ruff check .
uv run black . --check
```

Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add notebooks/01_Q_Phase.ipynb notebooks/01_Q_Phase.py notebooks/02_U_Phase.ipynb notebooks/02_U_Phase.py
git commit -m "docs(notebooks): document binary KSI gate reformulation in Q-Phase and U-Phase

The 3-class ceiling is empirically confirmed in A3-Phase §9 (empirical max
macro-F1=0.424 over 19 configs; Cramér's V<=0.13; ~90x odds-lift required).
Q-Phase §N adds the reformulation narrative with evidence and literature
justification. U-Phase adds forward-pointing methodological notes."
```

---

## Self-Review Checklist

**Spec coverage:**

| Review section | Task that implements it |
|---|---|
| §1 — Ceiling diagnosis | Task 5 (§9 markdown + ceiling-argument cell) |
| §2.1 — Per-class decomposition | Task 5 (§9 baseline print) |
| §2.2 — Empirical ceiling | Task 5 (§9 front plot) |
| §2.3 — Arithmetic argument | Task 5 (§9 ceiling-argument cell) |
| §2.4 — Feature weakness | Task 7 (U-Phase note) |
| §4 — OSM discussion | Not in scope (OSM features already built; no new OSM work) |
| §5.1 — Threshold deadlock fix | Partially addressed: Task 1 adds the correct optimizer; Task 5 applies it post-hoc without touching `_build_pipeline_for` (which intentionally gates Optuna) |
| §5.2 — Threshold optimizes wrong objective | Task 1 fixes this: `find_gate_optimal_offsets` maximizes macro-F1 **under** recall constraint |
| §5.3 — Single-class threshold | Task 1 fixes this: 2D sweep over classes 1 AND 2 |
| §6 — Roadmap Kat. 1.1–1.5 | Tasks 1, 4, 5 |
| §6 — Roadmap Kat. 4.1 (binary KSI) | Tasks 2, 3, 6 |
| §6 — Roadmap Kat. 4.3 (document ceiling) | Tasks 5, 7 |
| §7A — `find_gate_optimal_offsets` | Task 1 |
| §7B — `split_features_target_binary` | Task 2 |
| §9 — Is the model salvageable? | Tasks 5–7 together |
| Scientific documentation | Task 7 |

**Gaps:** Roadmap Kat. 2.1 (fold-safe SMOTE), 2.2 (ordinal fair test), 2.3 (Optuna multi-objective), 2.4 (cost-sensitive boosting) are explicitly NOT in scope — the review rates them low-impact and the recommendation is against investing further in 3-class tuning.

**Placeholder scan:** None found.

**Type consistency:** `find_gate_optimal_offsets` returns `tuple[tuple[float, float] | None, float]` — used consistently in Task 1 implementation and Task 5 notebook. `split_features_target_binary` returns `tuple[pd.DataFrame, pd.Series]` — consumed in Task 6. `evaluate_binary_predictions` returns `dict` with keys `macro_f1`, `recall_ksi`, `recall_slight`, `confusion_matrix` — consumed in Task 6 with matching key names.
