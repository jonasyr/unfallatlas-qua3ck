# SVM Algorithm-Selection Coverage + Binary-KSI Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two related gaps in the A³ (Algorithm Selection) phase: (1) Support Vector Machines have never been trained, tuned, or even mentioned anywhere in this repository, despite `docs/course-material/Einheit 6 – Support Vector Machines.md` being required course material; (2) the binary-KSI section never ran an actual champion search — it silently inherited the 3-class champion's family (LightGBM) instead of comparing candidate families on the binary target the way §3–§5 do for the 3-class problem. This plan fixes both by giving the binary-KSI section its own Stage 0 → Stage 1 → gate-aware champion-selection → tune-the-winner pipeline, with SVM (linear, hinge-SGD, RBF-kernel) included as first-class candidate families from the start — not bolted on afterward. It also fixes a real structural defect: sections `§9` and `§10` are each used **twice** in the notebook (`## 9 —`/`## 10 —` for the 3-class save/summary cells, then a second `§9`/`§10` for the 3-class-ceiling and binary sections) — this plan renumbers everything into one clean, continuous `## 1 —` … `## 19 —` sequence. Finally, it adds a Pareto-front visualization for the binary-KSI results (the LightGBM/SVM/RF/XGBoost/CatBoost comparison currently has no equivalent to the 3-class `reports/figures/a3_f1_recall_front.png` plot).

**Architecture:** Add a new `src/unfallatlas/models/svm.py` module with three `build_*_pipeline` factory functions (linear `LinearSVC`, hinge-loss `SGDClassifier` as a scalable SVM approximation, and kernel `SVC(kernel="rbf")`), each following the exact `Pipeline(steps=[("preprocess", ...), ("classify", ...)])` convention already used by `src/unfallatlas/models/boosting.py`. Add a small reusable threshold-sweep helper to `src/unfallatlas/models/evaluate.py` for `decision_function`-only estimators, and generalize the existing `select_best_candidate` gate-aware selector to accept a `recall_col` parameter (default unchanged, so the 3-class call sites are untouched) so the *same* selection logic the 3-class section already uses for its champion search can be reused for binary. Add a binary-KSI Pareto-front plotting function to `src/unfallatlas/viz/metrics_viz.py`, factored out of the existing `plot_f1_recall_front` via a shared private helper. Renumber the two duplicated section headers. Then replace the binary section's single-family "quick baseline" cells with a genuine multi-family Stage 0/Stage 1 comparison (random guess, majority class, logistic regression, Random Forest, XGBoost, LightGBM, CatBoost, and the three new SVM variants — all class-weighted/balanced, mirroring the 3-class Stage 1 convention), select the winner via the generalized `select_best_candidate`, Optuna-tune only that winner (a per-family search-space dispatch table, mirroring the existing `PARAM_SPACES` pattern from the 3-class §7), refit, gate-optimal-threshold-sweep, evaluate once on Test-2024, save artifacts, and plot. All notebook edits happen via `nbformat` scripts (never hand-editing `.ipynb` JSON or the `.py` mirror directly), executed for real via `nbclient`, with the results-summary and model card updated from the actual measured numbers — no fabricated results.

**Explicit scope boundary:** the 3-class section also runs a per-family imbalance-strategy layer (SMOTE, ADASYN, threshold-moving, ordinal classification — §6 in the current notebook) on top of Stage 1's winners. This plan does **not** replicate that layer for binary. Reasons, stated here so the decision is visible rather than a silent omission: (a) binary KSI's positive rate is ~17–20%, far milder than the 3-class minority's ~0.9–2%, so `class_weight="balanced"` alone is much less likely to be the binding constraint; (b) that layer's code carries several hard-won, non-obvious fixes (CatBoost `clone()` incompatibility with `class_weights`, object-dtype `None`-vs-`np.nan` handling for SMOTE/ADASYN) that would need re-deriving for a binary-labeled variant, roughly tripling this plan's risk surface for a benefit that is speculative given (a). The notebook's new binary Stage-1 markdown cell says this explicitly, so a reviewer sees a documented decision, not a gap.

**Tech Stack:** scikit-learn (`sklearn.svm.SVC`/`LinearSVC`, `sklearn.linear_model.SGDClassifier`) — already a pinned dependency, no new packages needed. `nbformat` for notebook cell manipulation (matches the pattern used in the prior binary-KSI-reframe session). `pytest`, `ruff`, `jupytext` for verification, matching existing CI gates.

## Global Constraints

- Never hand-edit `notebooks/03_A3_Phase.py` (the jupytext mirror) directly — always edit `notebooks/03_A3_Phase.ipynb` and then run `uv run jupytext --sync notebooks/03_A3_Phase.ipynb`. A pre-commit hook (`check-notebook-mirrors`) rejects direct mirror edits.
- Never retrain or overwrite `data/processed/a3_best_model.joblib` (the 3-class champion). `data/processed/a3_binary_best_model.joblib` (the binary champion) **will** be overwritten by this plan — that is the intended outcome of running a genuine binary champion search, not an accident — but only via the same disciplined protocol the 3-class section uses: Stage 0/1 comparison on Val-2023, gate-aware selection, tune the winner, refit on the winner's appropriate full-scale data, single Test-2024 evaluation.
- `random_state=42` / `SEED = 42` everywhere, matching every existing model builder and the notebook's `SEED` variable.
- New pipelines follow `Pipeline(steps=[("preprocess", preprocessor), ("classify", <Estimator>(...))])` exactly, per `src/unfallatlas/models/boosting.py`.
- SVM pipelines must use `build_preprocessor(scale_for_linear=True)` (a *different* preprocessor instance than the tree models use) — Einheit 6 §5 and §19 ("Häufige Fehler") are explicit that unscaled features silently wreck SVM margins.
- Kernel `SVC` (RBF) is only trained on a small stratified subsample (thousands of rows) — Einheit 6 §1/§9/§10 and the course material's complexity table are explicit that `SVC` is O(m²)–O(m³) and becomes impractical at the dataset's actual scale (1,554,834 training rows).
- `ruff check .`, `ruff format --check .`, and `uv run pytest -q` must all pass before any commit.
- No `Co-Authored-By` trailer in commit messages (project-specific instruction already honored in this repo's history).
- Only commit when explicitly instructed; this plan produces commits as part of its own task steps because the user has already asked for the work to be implemented — do not push, open a PR, or take any other externally-visible action beyond local commits.

---

## Context recap (from research spawned before this plan)

- **Dataset**: Train 2016–2022 = 1,554,834 rows, Val 2023 = 269,048 rows, Test 2024 = 268,519 rows. Preprocessed feature matrix is dense, 64 columns after one-hot/cyclic/target-encoding (`build_preprocessor().fit_transform(X_train).shape == (1554834, 64)`).
- **Current binary-KSI champion** (`data/processed/a3_binary_model_card.json`): LightGBM, Test-2024 macro-F1 = 0.6069, Recall(KSI) = 0.5233, gate (macro-F1 ≥ 0.55 AND Recall(KSI) ≥ 0.50) **passed**, decision threshold 0.580 (found via 1D sweep on Val-2023 `predict_proba`).
- **SVM coverage today**: zero. No `SVC`/`LinearSVC`/hinge `SGDClassifier` anywhere in `notebooks/03_A3_Phase.ipynb`, `src/unfallatlas/`, or `tests/`. `docs/project/Technical_Review_Next_Steps.md` and `docs/superpowers/plans/2026-07-14-binary-ksi-reframe.md` never mention SVM either — this is a genuine, previously-undetected gap relative to the required course material (`docs/course-material/Einheit 6 – Support Vector Machines.md`), not a redundant re-implementation of something already covered.
- **Existing binary-KSI notebook cells** (indices as of this writing — re-verify with the lookup script in Task 4 before editing, since indices shift after any edit):
  - Cell 38 (markdown): `## §10 — Binary KSI Classification`
  - Cell 39 (code): defines `X_train_bin, y_train_bin, X_val_bin, y_val_bin, X_test_bin, y_test_bin` via `split_features_target_binary(train/val/test)`
  - Cell 40 (code): defines `SEED = 42`, `SUB_N = 500_000`, the stratified subsample `train_sub`/`groups_sub`/`X_sub`/`y_sub_bin`, and a quick LightGBM baseline
  - Cell 41 (code): Optuna tuning of LightGBM binary (20 trials, `study_binary`)
  - Cell 42 (code): refits `pipeline_binary_final` on full train, then a 1D `predict_proba`-threshold sweep on Val-2023 producing `best_threshold`, `best_f1_val`, and `y_val_proba_bin`
  - Cell 43 (code): single Test-2024 evaluation producing `metrics_binary_test`, `gate_passed`, with a hard `assert gate_passed`
  - Cell 44 (code): saves `a3_binary_best_model.joblib` and `a3_binary_model_card.json` (dict `binary_model_card`)
  - Cell 45 (markdown): `### §10 — Results Summary: Binary KSI Classification`
  - All variables above (`train`, `val`, `test`, `BASE`, `X_sub`, `y_sub_bin`, `groups_sub`, `X_train_bin`, `y_train_bin`, `X_val_bin`, `y_val_bin`, `SEED`, `SUB_N`, `y_val_proba_bin`, `best_threshold`) are live in kernel memory by the time cell 44 finishes running, and this plan's new cells are inserted between cell 44 and cell 45 to reuse them without recomputation.

---

### Task 1: `find_best_binary_threshold` helper in `evaluate.py`

**Files:**
- Modify: `src/unfallatlas/models/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `evaluate_binary_predictions(y_true, y_pred) -> dict` (already exists in this file, keys `macro_f1`, `recall_ksi`, `recall_slight`, `confusion_matrix`), `meets_binary_acceptance_criteria(metrics) -> bool` (already exists), `BINARY_RECALL_KSI_THRESHOLD = 0.50` (already exists).
- Produces: `find_best_binary_threshold(y_true, scores, recall_gate: float = BINARY_RECALL_KSI_THRESHOLD, n_steps: int = 81) -> tuple[float, dict]` — sweeps a threshold over any monotonic score array (`predict_proba(...)[:, 1]` **or** `decision_function(...)` both work, since the function derives its sweep range from `scores.min()`/`scores.max()` rather than assuming `[0, 1]`) and returns `(best_threshold, best_metrics)` where `best_metrics` is a full `evaluate_binary_predictions(...)`-shaped dict. If at least one threshold satisfies `recall_ksi >= recall_gate`, returns the recall-gate-satisfying threshold with the highest `macro_f1`. If none satisfy the gate, returns the threshold with the best unconstrained `macro_f1` instead (the caller must check `meets_binary_acceptance_criteria(best_metrics)` to know which case occurred). Task 5 in this plan is the consumer.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate.py` (add `find_best_binary_threshold` to the existing `from unfallatlas.models.evaluate import (...)` block at the top of the file, alphabetically after `evaluate_predictions` and before `macro_f1`):

```python
def test_find_best_binary_threshold_recovers_perfect_separator():
    # Scores perfectly separate the two classes at score=0: negatives < 0, positives > 0.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    threshold, metrics = find_best_binary_threshold(y_true, scores)
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_ksi"] == 1.0
    # Any threshold strictly between -0.5 and 0.5 reproduces the perfect split.
    assert -0.5 < threshold <= 0.5


def test_find_best_binary_threshold_falls_back_when_gate_unreachable():
    # Only one positive exists and it always gets the lowest score - recall_ksi
    # can never reach 1.0 for any threshold that also predicts it positive alongside
    # negatives, so demand an unreachable recall_gate and confirm graceful fallback.
    y_true = np.array([0, 0, 0, 0, 1])
    scores = np.array([-1.0, -0.5, 0.0, 0.5, -2.0])  # the one positive has the lowest score
    threshold, metrics = find_best_binary_threshold(y_true, scores, recall_gate=1.0)
    assert metrics["recall_ksi"] < 1.0  # gate was infeasible, unconstrained fallback used
    assert isinstance(threshold, float)


def test_find_best_binary_threshold_works_with_decision_function_range():
    # decision_function output is unbounded (not in [0, 1]) - confirm the sweep
    # range is derived from the scores themselves, not hardcoded to a [0, 1] tube.
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([-10.0, -8.0, 8.0, 10.0])
    threshold, metrics = find_best_binary_threshold(y_true, scores)
    assert metrics["macro_f1"] == 1.0
    assert -8.0 < threshold <= 8.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -k find_best_binary_threshold -v`
Expected: FAIL with `ImportError: cannot import name 'find_best_binary_threshold'`

- [ ] **Step 3: Implement `find_best_binary_threshold`**

In `src/unfallatlas/models/evaluate.py`, add `import numpy as np` to the top imports (currently only `import pandas as pd` and the sklearn metrics import — `numpy` is not yet imported in this file), then append the function after `meets_binary_acceptance_criteria`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: all tests in the file PASS (existing tests plus the 3 new ones).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/unfallatlas/models/evaluate.py tests/test_evaluate.py
uv run ruff format --check src/unfallatlas/models/evaluate.py tests/test_evaluate.py
git add src/unfallatlas/models/evaluate.py tests/test_evaluate.py
git commit -m "feat: add find_best_binary_threshold for decision_function-based gate sweeps"
```

- [ ] **Step 6: Write the failing test for generalizing `select_best_candidate`**

`select_best_candidate` (already in `src/unfallatlas/models/evaluate.py`) is hardcoded to a `"recall_class_1"` column, which is 3-class-specific. Task 5 of this plan needs the *same* gate-aware selection logic to pick a binary-KSI champion using a `"recall_ksi"` column instead. Add to `tests/test_evaluate.py`:

```python
def test_select_best_candidate_accepts_custom_recall_column():
    rows = pd.DataFrame(
        [
            {"model": "svm_linear", "macro_f1": 0.50, "recall_ksi": 0.60},
            {"model": "svm_rbf", "macro_f1": 0.60, "recall_ksi": 0.30},  # fails recall gate
            {"model": "lightgbm", "macro_f1": 0.45, "recall_ksi": 0.55},
        ]
    )
    winner = select_best_candidate(rows, recall_col="recall_ksi")
    assert winner["model"] == "svm_linear"  # highest macro_f1 among recall_ksi>=0.5 rows


def test_select_best_candidate_default_recall_column_still_recall_class_1():
    """Regression guard: existing 3-class call sites (no recall_col= passed)
    must keep reading 'recall_class_1', unchanged by this generalization."""
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.60},
            {"model": "b", "macro_f1": 0.60, "recall_class_1": 0.30},
        ]
    )
    winner = select_best_candidate(rows)
    assert winner["model"] == "a"
```

- [ ] **Step 7: Run tests to verify the new one fails**

Run: `uv run pytest tests/test_evaluate.py -k custom_recall_column -v`
Expected: FAIL with `TypeError: select_best_candidate() got an unexpected keyword argument 'recall_col'`

- [ ] **Step 8: Generalize `select_best_candidate`**

In `src/unfallatlas/models/evaluate.py`, replace the existing `select_best_candidate` function with:

```python
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
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: all tests PASS, including the 3 pre-existing `select_best_candidate` tests (unaffected — they never pass `recall_col`, so they exercise the unchanged default path) and the 2 new ones.

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check src/unfallatlas/models/evaluate.py tests/test_evaluate.py
uv run ruff format --check src/unfallatlas/models/evaluate.py tests/test_evaluate.py
git add src/unfallatlas/models/evaluate.py tests/test_evaluate.py
git commit -m "feat: generalize select_best_candidate with recall_col for binary-KSI reuse"
```

---

### Task 2: `src/unfallatlas/models/svm.py` — SVM pipeline builders

**Files:**
- Create: `src/unfallatlas/models/svm.py`
- Test: `tests/test_models_svm.py`

**Interfaces:**
- Consumes: `build_preprocessor(scale_for_linear: bool = False) -> ColumnTransformer` from `src/unfallatlas/features/preprocessing.py` (already exists; SVM callers must pass `scale_for_linear=True`).
- Produces:
  - `build_linear_svm_binary_pipeline(preprocessor, C: float = 1.0, class_weight: str | None = "balanced", dual: str = "auto") -> Pipeline`
  - `build_sgd_hinge_binary_pipeline(preprocessor, alpha: float = 1e-4, class_weight: str | None = "balanced") -> Pipeline`
  - `build_rbf_svm_binary_pipeline(preprocessor, C: float = 1.0, gamma: str | float = "scale", class_weight: str | None = "balanced") -> Pipeline`

  Each returns a fitted-on-demand `sklearn.pipeline.Pipeline` with steps named `"preprocess"` and `"classify"`, exactly like every existing builder in `boosting.py`. Task 4 (notebook cells) is the consumer.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models_svm.py`:

```python
import numpy as np
import pandas as pd

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.svm import (
    build_linear_svm_binary_pipeline,
    build_rbf_svm_binary_pipeline,
    build_sgd_hinge_binary_pipeline,
)


def _toy_X_y(n=120):
    rng = np.random.default_rng(1)
    road_classes = np.array(["primary", "secondary", "residential", "motorway", "tertiary", None])
    X = pd.DataFrame(
        {
            "UMONAT": rng.integers(1, 13, n),
            "USTUNDE": rng.integers(0, 24, n),
            "UWOCHENTAG": rng.integers(1, 8, n),
            "UART": rng.integers(0, 10, n),
            "UTYP1": rng.integers(1, 8, n),
            "ULICHTVERH": rng.integers(0, 3, n),
            "STRZUSTAND": rng.integers(0, 3, n),
            "IstRad": rng.integers(0, 2, n),
            "IstPKW": rng.integers(0, 2, n),
            "IstFuss": rng.integers(0, 2, n),
            "IstKrad": rng.integers(0, 2, n),
            "IstGkfz": rng.integers(0, 2, n),
            "IstSonstig": rng.integers(0, 2, n),
            "LON": rng.uniform(6, 15, n),
            "LAT": rng.uniform(47, 55, n),
            "UREGBEZ": rng.integers(1, 5, n),
            "UKREIS": rng.integers(1000, 1050, n),
            "dwd_temp_air_2m": rng.normal(10, 5, n),
            "dwd_precip_mm": rng.exponential(1.0, n),
            "dwd_visibility_m": rng.exponential(5000, n),
            "dwd_wind_speed_ms": rng.normal(3, 1, n),
            "dwd_station_dist_km": rng.uniform(0.1, 40, n),
            "osm_dominant_road_class": rng.choice(road_classes, n),
            "osm_maxspeed_mean": rng.choice(
                [30.0, 50.0, 70.0, 100.0, np.nan], n, p=[0.2, 0.3, 0.2, 0.2, 0.1]
            ),
            "osm_maxspeed_max": rng.choice(
                [50.0, 70.0, 100.0, 130.0, np.nan], n, p=[0.2, 0.3, 0.2, 0.2, 0.1]
            ),
            "osm_road_density": rng.choice([*rng.exponential(500, n // 2), np.nan], n),
            "osm_way_count": rng.choice([*rng.integers(2, 400, n // 2), np.nan], n),
        }
    )
    y_bin = pd.Series(rng.choice([0, 1], n, p=[0.8, 0.2]))
    return X, y_bin


def test_linear_svm_binary_pipeline_predicts_binary_labels():
    X, y = _toy_X_y()
    pipe = build_linear_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_linear_svm_binary_pipeline_exposes_decision_function():
    X, y = _toy_X_y()
    pipe = build_linear_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    scores = pipe.decision_function(X)
    assert scores.shape == (len(X),)


def test_sgd_hinge_binary_pipeline_predicts_binary_labels():
    X, y = _toy_X_y()
    pipe = build_sgd_hinge_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_sgd_hinge_binary_pipeline_uses_hinge_loss():
    """Regression test: this must stay a genuine (linear) SVM, not silently
    drift to log-loss logistic regression if someone edits the default."""
    pipe = build_sgd_hinge_binary_pipeline(build_preprocessor(scale_for_linear=True))
    assert pipe.named_steps["classify"].get_params()["loss"] == "hinge"


def test_rbf_svm_binary_pipeline_predicts_binary_labels():
    X, y = _toy_X_y()
    pipe = build_rbf_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_rbf_svm_binary_pipeline_exposes_decision_function():
    X, y = _toy_X_y()
    pipe = build_rbf_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    scores = pipe.decision_function(X)
    assert scores.shape == (len(X),)


def test_svm_pipelines_accept_hyperparameter_overrides():
    X, y = _toy_X_y()
    linear_pipe = build_linear_svm_binary_pipeline(build_preprocessor(scale_for_linear=True), C=0.01)
    assert linear_pipe.named_steps["classify"].get_params()["C"] == 0.01

    sgd_pipe = build_sgd_hinge_binary_pipeline(
        build_preprocessor(scale_for_linear=True), alpha=1e-3
    )
    assert sgd_pipe.named_steps["classify"].get_params()["alpha"] == 1e-3

    rbf_pipe = build_rbf_svm_binary_pipeline(
        build_preprocessor(scale_for_linear=True), C=5.0, gamma=0.1
    )
    params = rbf_pipe.named_steps["classify"].get_params()
    assert params["C"] == 5.0
    assert params["gamma"] == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_svm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.svm'`

- [ ] **Step 3: Implement `src/unfallatlas/models/svm.py`**

```python
"""SVM candidate models for the A³ binary-KSI algorithm-selection comparison.

Course reference: docs/course-material/Einheit 6 – Support Vector Machines.md.
SVMs require scaled features (Einheit 6 §5, §19 "Häufige Fehler") - always
pass a preprocessor built with build_preprocessor(scale_for_linear=True),
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

    O(m x n) - scales to the full training set (Einheit 6 §10 complexity
    table). No kernel-trick support; this is the "fast baseline" SVM
    candidate, the one Einheit 6 §9's rule of thumb says to try first.
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

    O(m x n), incremental/out-of-core capable (Einheit 6 §10) - the only SVM
    variant that comfortably trains on the full 1.55M-row training set in
    this project without subsampling.
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

    O(m^2) to O(m^3) in fit time (Einheit 6 §9/§10 complexity table) - only
    feasible on a small stratified subsample (thousands, not millions, of
    rows). Callers are responsible for subsampling before calling .fit().
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_svm.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/unfallatlas/models/svm.py tests/test_models_svm.py
uv run ruff format --check src/unfallatlas/models/svm.py tests/test_models_svm.py
git add src/unfallatlas/models/svm.py tests/test_models_svm.py
git commit -m "feat: add SVM pipeline builders (linear, hinge-SGD, RBF kernel) for A³ algorithm selection"
```

---

### Task 3: Binary-KSI Pareto-front plot in `metrics_viz.py`

**Files:**
- Modify: `src/unfallatlas/viz/metrics_viz.py`
- Test: `tests/test_metrics_viz.py`

**Interfaces:**
- Consumes: nothing new (matplotlib, pandas only, same as the existing function).
- Produces: `plot_binary_f1_recall_front(comparison_df: pd.DataFrame, ax: plt.Axes | None = None, gate_f1: float = 0.55, gate_recall: float = 0.50, label_col: str = "model", title: str = "Pareto Front: Macro-F1 vs. Recall(KSI) — binary KSI candidates") -> plt.Axes`. `comparison_df` must have columns `[label_col, "macro_f1", "recall_ksi"]` (matching `evaluate_binary_predictions`'s key names, not `recall_class_1`). Task 4 (notebook) is the consumer, producing `reports/figures/a3_binary_f1_recall_front.png`.
- The existing `plot_f1_recall_front(comparison_df, ax=None, gate_f1=0.55, gate_recall=0.50, label_col="model") -> plt.Axes` keeps its exact current signature and behavior (title text "Pareto Front: Macro-F1 vs. Recall(Killed) — all 19 configurations" unchanged) — it becomes a thin wrapper around a new private helper, with zero observable behavior change, so none of the 4 existing tests in `tests/test_metrics_viz.py` need to change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics_viz.py` (add `plot_binary_f1_recall_front` to the `from unfallatlas.viz.metrics_viz import (...)` line at the top):

```python
@pytest.fixture()
def binary_comparison_df():
    return pd.DataFrame(
        [
            {"model": "lightgbm_binary_balanced (champion)", "macro_f1": 0.6069, "recall_ksi": 0.5233},
            {"model": "svm_linear_C1.0", "macro_f1": 0.55, "recall_ksi": 0.40},
            {"model": "svm_rbf_C1.0_gammascale", "macro_f1": 0.50, "recall_ksi": 0.35},
            {"model": "svm_sgd_hinge_alpha0.0001", "macro_f1": 0.52, "recall_ksi": 0.45},
        ]
    )


def test_plot_binary_f1_recall_front_returns_axes(binary_comparison_df):
    ax = plot_binary_f1_recall_front(binary_comparison_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_binary_f1_recall_front_accepts_external_ax(binary_comparison_df):
    _, ax = plt.subplots()
    result = plot_binary_f1_recall_front(binary_comparison_df, ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_binary_f1_recall_front_gate_lines_present(binary_comparison_df):
    ax = plot_binary_f1_recall_front(binary_comparison_df, gate_f1=0.55, gate_recall=0.50)
    h_lines = [
        ln for ln in ax.lines if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
    ]
    v_lines = [
        ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
    ]
    assert len(h_lines) > 0
    assert len(v_lines) > 0
    plt.close("all")


def test_plot_binary_f1_recall_front_uses_recall_ksi_not_recall_class_1(binary_comparison_df):
    # Regression guard: this plot must read the binary-evaluation column name,
    # not silently fall back to the 3-class 'recall_class_1' column.
    ax = plot_binary_f1_recall_front(binary_comparison_df)
    xdata = [pt[0] for coll in ax.collections for pt in coll.get_offsets()]
    assert sorted(xdata) == sorted(binary_comparison_df["recall_ksi"].tolist())
    plt.close("all")


def test_plot_f1_recall_front_unaffected_by_refactor(comparison_df):
    """Existing 3-class plot must keep its exact title after the shared-helper refactor."""
    ax = plot_f1_recall_front(comparison_df)
    assert ax.get_title() == "Pareto Front: Macro-F1 vs. Recall(Killed) — all 19 configurations"
    plt.close("all")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics_viz.py -v`
Expected: FAIL with `ImportError: cannot import name 'plot_binary_f1_recall_front'`

- [ ] **Step 3: Refactor `metrics_viz.py`**

Replace the full contents of `src/unfallatlas/viz/metrics_viz.py` with:

```python
"""Diagnostic plots for model selection and Pareto-front analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def _plot_pareto_front(
    comparison_df: pd.DataFrame,
    recall_col: str,
    recall_axis_label: str,
    title: str,
    ax: plt.Axes | None,
    gate_f1: float,
    gate_recall: float,
    label_col: str,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        comparison_df[recall_col],
        comparison_df["macro_f1"],
        zorder=3,
        s=60,
        color="steelblue",
    )

    for _, row in comparison_df.iterrows():
        ax.annotate(
            row[label_col],
            xy=(row[recall_col], row["macro_f1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(
        gate_f1, color="crimson", linestyle="--", linewidth=1.2, label=f"Gate: macro-F1 ≥ {gate_f1}"
    )
    ax.axvline(
        gate_recall,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
        label=f"Gate: {recall_axis_label} ≥ {gate_recall}",
    )

    ax.fill_between(
        [gate_recall, ax.get_xlim()[1] if ax.get_xlim()[1] > gate_recall else 1.0],
        gate_f1,
        1.0,
        alpha=0.08,
        color="green",
        label="Feasible zone",
    )

    ax.set_xlabel(recall_axis_label, fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    return ax


def plot_f1_recall_front(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
) -> plt.Axes:
    """Scatter macro-F1 vs. Recall(class 1) for every 3-class model configuration.

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
    return _plot_pareto_front(
        comparison_df,
        recall_col="recall_class_1",
        recall_axis_label="Recall (Class 1 — Killed)",
        title="Pareto Front: Macro-F1 vs. Recall(Killed) — all 19 configurations",
        ax=ax,
        gate_f1=gate_f1,
        gate_recall=gate_recall,
        label_col=label_col,
    )


def plot_binary_f1_recall_front(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
    title: str = "Pareto Front: Macro-F1 vs. Recall(KSI) — binary KSI candidates",
) -> plt.Axes:
    """Scatter macro-F1 vs. Recall(KSI) for every binary-KSI model configuration.

    Analogous to plot_f1_recall_front, but reads the binary-evaluation
    column name ('recall_ksi', from evaluate_binary_predictions) instead of
    the 3-class 'recall_class_1'. Use this to compare the LightGBM binary
    champion against SVM (and any future) candidate families on the same
    axes as the revised binary gate.

    Args:
        comparison_df: DataFrame with columns [label_col, 'macro_f1', 'recall_ksi'].
        ax: Optional existing Axes; a new figure+axes is created if None.
        gate_f1: Horizontal gate line for macro-F1.
        gate_recall: Vertical gate line for Recall(KSI).
        label_col: Column used to label each point.
        title: Plot title (override when the candidate count/composition changes).

    Returns:
        The populated Axes object.
    """
    return _plot_pareto_front(
        comparison_df,
        recall_col="recall_ksi",
        recall_axis_label="Recall (KSI)",
        title=title,
        ax=ax,
        gate_f1=gate_f1,
        gate_recall=gate_recall,
        label_col=label_col,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics_viz.py -v`
Expected: all tests PASS (4 pre-existing + 5 new = 9 total).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/unfallatlas/viz/metrics_viz.py tests/test_metrics_viz.py
uv run ruff format --check src/unfallatlas/viz/metrics_viz.py tests/test_metrics_viz.py
git add src/unfallatlas/viz/metrics_viz.py tests/test_metrics_viz.py
git commit -m "feat: add plot_binary_f1_recall_front for binary-KSI Pareto-front visualization"
```

---


### Task 4: Fix the duplicated §9/§10 section numbering

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (via a throwaway Python script)
- Modify (generated): `notebooks/03_A3_Phase.py` (via `jupytext --sync`, never by hand)

**Interfaces:**
- Consumes: nothing new.
- Produces: a notebook where every top-level section header is a unique, sequential `## N —` (sections `1`–`10` already correct and untouched; the 3-class-ceiling section becomes `## 11 —`). Task 5 relies on the old `## §10 — Binary KSI Classification` header text still being present and unmodified (Task 4 deliberately does not touch it — Task 5 replaces it wholesale as part of a larger rewrite, so renaming it here first would be wasted work).

Confirmed via direct inspection: sections are numbered `## 1 —` through `## 10 —` cleanly (`## 9 — Save the winning pipeline and model card`, `## 10 — A³ summary and A³-to-C handoff`), then a markdown cell titled `## §9 — 3-Class Ceiling: Empirical Evidence & Gate-Optimal Thresholding` reuses number 9, and a later cell titled `## §10 — Binary KSI Classification` reuses number 10 — both with a different heading style (`§` glyph instead of the plain `## N —` used everywhere else). This task fixes the first duplicate; Task 5–7 fix the second as part of the larger binary-section rewrite.

- [ ] **Step 1: Write the renumbering script**

Create `/tmp/renumber_section9.py` (throwaway):

```python
import nbformat

NB_PATH = "notebooks/03_A3_Phase.ipynb"
nb = nbformat.read(NB_PATH, as_version=4)

renamed = 0
for c in nb.cells:
    if c.cell_type != "markdown":
        continue
    if c.source.startswith("## §9 — 3-Class Ceiling"):
        c.source = c.source.replace(
            "## §9 — 3-Class Ceiling", "## 11 — 3-Class Ceiling", 1
        )
        renamed += 1
    # Forward-references to the (not-yet-renumbered-in-this-task) binary
    # section inside the 3-class-ceiling cells - update "§10" -> "§12" so
    # they point at the correct future section number once Task 5 lands.
    if "§9" in c.source or "§10" in c.source:
        c.source = c.source.replace(
            "for the transition into §10", "for the transition into §12"
        )
    if c.cell_type == "code":
        pass  # code cells never reference section numbers in this notebook

for c in nb.cells:
    if c.cell_type == "code" and (
        "→ Reformulierung zu binärem KSI in §10" in c.source
        or "Solution: Binary KSI reformulation in §10" in c.source
    ):
        c.source = c.source.replace(
            "→ Reformulierung zu binärem KSI in §10", "→ Reformulierung zu binärem KSI in §12"
        )
    if c.cell_type == "markdown" and "Solution: Binary KSI reformulation in §10" in c.source:
        c.source = c.source.replace(
            "Solution: Binary KSI reformulation in §10",
            "Solution: Binary KSI reformulation in §12",
        )

assert renamed == 1, f"expected exactly 1 header renamed, got {renamed}"
nbformat.write(nb, NB_PATH)
print(f"Renamed {renamed} header(s); updated forward-references.")
```

- [ ] **Step 2: Run it and verify**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
uv run python /tmp/renumber_section9.py
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/03_A3_Phase.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.strip().startswith('#'):
        print(c.source.strip().split(chr(10))[0])
" | grep -E "^## (9|10|11) —|^## §"
```

Expected: `## 9 — Save the winning pipeline and model card`, `## 10 — A³ summary and A³-to-C handoff`, `## 11 — 3-Class Ceiling: Empirical Evidence & Gate-Optimal Thresholding` — and **no** line starting with `## §` anymore for section 9 (the `## §10 — Binary KSI Classification` header is untouched by design; it still shows up here and that is correct at this point in the plan — Task 5 replaces it).

- [ ] **Step 3: Sync mirror, verify nothing else changed, commit**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
git diff --stat notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
```

Expected: only the two renamed markdown cells' source shows as changed (a `git diff` on the `.py` mirror should show only the corresponding header-line changes) — no code cell content, no cell outputs, no cell count change.

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "docs(notebook/a3): fix duplicated §9 section number (3-class-ceiling -> ## 11)"
```

---

### Task 5: Binary champion search — Stage 0 baselines, Stage 1 candidates (incl. SVM), gate-aware selection

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (via a throwaway Python script)
- Modify (generated): `notebooks/03_A3_Phase.py` (via `jupytext --sync`, never by hand)

**Interfaces:**
- Consumes: `build_linear_svm_binary_pipeline`, `build_sgd_hinge_binary_pipeline`, `build_rbf_svm_binary_pipeline` (Task 2); `select_best_candidate(rows, recall_col=...)` (Task 1, Step 8); everything already live in kernel memory from the notebook's existing setup cells (`train`, `val`, `test`, `BASE`, `CHECKPOINT_DIR`, `_log_progress`, `build_preprocessor`, `build_random_guess_classifier`, `build_majority_class_classifier`, `build_logreg_pipeline`, `build_random_forest_pipeline`, `build_xgboost_pipeline`, `build_catboost_pipeline`, `_use_gpu_resolved`, `joblib`, `time`, `pd`, `np`).
- Produces (new kernel variables consumed by Task 6): `X_train_bin, y_train_bin, X_val_bin, y_val_bin, X_test_bin, y_test_bin`; `SEED`, `SUB_N`, `train_sub`, `groups_sub`, `X_sub`, `y_sub_bin`, `train_svc_sub`, `groups_svc_sub`, `X_svc_sub`, `y_svc_sub`; `linear_preprocessor_bin`, `tree_preprocessor_bin`; `BINARY_BUILDERS: dict[str, Callable[[], Pipeline]]`, `BINARY_STAGE1_DATA: dict[str, tuple[X, y]]`, `_binary_fit_kwargs(family: str, y_fit) -> dict`; `binary_comparison_rows: list[dict]`, `binary_comparison_df: pd.DataFrame`; `binary_champion_row: pd.Series`, `binary_champion_family: str`.

- [ ] **Step 1: Write the replacement-cell script**

Create `/tmp/insert_binary_champion_search.py` (throwaway):

```python
import nbformat

NB_PATH = "notebooks/03_A3_Phase.ipynb"
nb = nbformat.read(NB_PATH, as_version=4)

header_idx = None
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and c.source.startswith("## §10 — Binary KSI Classification"):
        header_idx = i
        break
assert header_idx is not None, "old binary-section header not found"
assert "split_features_target_binary" in nb.cells[header_idx + 1].source
assert "SUB_N = 500_000" in nb.cells[header_idx + 2].source

md_12 = """## 12 — Binary KSI Reframing: Motivation & Target Definition

§11 showed the 3-class gate (macro-F1 >= 0.55 AND Recall(class-1) >= 0.50) is a Bayes-ceiling, not
a tuning problem: not one of 19 tested configurations reaches the target quadrant, and the
arithmetic ceiling argument shows why. The revised, domain-standard framing (Santos 2022,
Pakgohar 2021, Schloessler 2024) collapses the three severity classes into a binary KSI (killed or
seriously injured, UKATGEORIE in {1,2}) vs. slight (UKATGEORIE = 3) target, with a same-shaped
gate: binary macro-F1 >= 0.55 AND Recall(KSI) >= 0.50.

Unlike an earlier pass at this reframing, this section runs a genuine **champion search** on the
binary target (SS13-SS15) instead of assuming the 3-class champion family (LightGBM) transfers
unchanged - mirroring the same Stage 0/Stage 1/gate-aware-selection discipline SS3-SS5 already use
for the 3-class problem, with Support Vector Machines (`docs/course-material/Einheit 6 - Support
Vector Machines.md`) included as first-class candidate families from the start.

**Scope note:** the imbalance-strategy layer from SS6 (SMOTE/ADASYN/threshold-moving/ordinal per
candidate family) is intentionally not repeated here. Binary KSI's positive rate (~17-20%) is far
milder than the 3-class minority's (~0.9-2%), so `class_weight="balanced"` alone is much less
likely to be the binding constraint, and re-deriving that layer's several hard-won edge-case fixes
for a binary-labeled variant would substantially raise this section's risk for a speculative
benefit. Every candidate below is compared class-weighted/balanced only.
"""

code_split = nb.cells[header_idx + 1].source  # reuse verbatim, unchanged

md_13 = """## 13 — Binary Champion Search: Stage 0 Baselines

Random guess and majority class establish the binary macro-F1 floor; Logistic Regression is the
first non-trivial benchmark - mirrors SS3's role for the 3-class problem, on the relabelled target.
"""

code_stage0 = '''from unfallatlas.models.evaluate import (  # noqa: E402
    evaluate_binary_predictions,
    meets_binary_acceptance_criteria,
)

binary_comparison_rows: list[dict] = []


def _score_binary_on_validation(name: str, fitted_estimator, family: str | None = None) -> None:
    preds = fitted_estimator.predict(X_val_bin)
    metrics = evaluate_binary_predictions(y_val_bin.values, preds)
    binary_comparison_rows.append({"model": name, "family": family or name, **metrics})
    print(
        f"{name:30s} macro-F1={metrics[\\'macro_f1\\']:.3f}  recall(KSI)={metrics[\\'recall_ksi\\']:.3f}"
    )


BINARY_CHECKPOINT_DIR = CHECKPOINT_DIR / "binary"
BINARY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_fit_binary(name: str, fit_callable):
    """Checkpoint helper for the binary champion search - separate namespace
    (BINARY_CHECKPOINT_DIR) from the 3-class checkpoints, so e.g. a 3-class
    'lightgbm_balanced.joblib' and a binary one never collide."""
    path = BINARY_CHECKPOINT_DIR / f"{name}.joblib"
    if path.exists():
        _log_progress(f"  -> {name}: loaded from binary checkpoint ({path.name})")
        return joblib.load(path)
    start = time.time()
    model = fit_callable()
    elapsed = time.time() - start
    joblib.dump(model, path)
    _log_progress(f"  -> {name} done in {elapsed:.1f}s")
    return model


linear_preprocessor_bin = build_preprocessor(scale_for_linear=True)

binary_stage0_specs = [
    ("binary_random_guess", lambda: build_random_guess_classifier()),
    ("binary_majority_class", lambda: build_majority_class_classifier()),
    ("binary_logistic_regression", lambda: build_logreg_pipeline(linear_preprocessor_bin)),
]

_log_progress(f"Starting binary Stage 0: {len(binary_stage0_specs)} baselines.")
for _name, _build_fn in binary_stage0_specs:
    _model = _load_or_fit_binary(
        _name, lambda _build_fn=_build_fn: _build_fn().fit(X_train_bin, y_train_bin)
    )
    _score_binary_on_validation(_name, _model)
_log_progress("Binary Stage 0 complete.")
'''

md_14 = """## 14 — Binary Champion Search: Stage 1 Candidates (Trees + SVM)

Mirrors SS4's role for the 3-class problem: every tree-ensemble family (Random Forest, XGBoost,
LightGBM, CatBoost), class-weighted/balanced, trained on the full 2016-2022 training set - plus,
new to this project, three SVM variants (`docs/course-material/Einheit 6 - Support Vector
Machines.md`): `LinearSVC` and hinge-loss `SGDClassifier` (both linear, scaled features), and
`SVC(kernel="rbf")` (the actual kernel trick). SVC's O(m^2)-O(m^3) fit complexity makes the full
1,554,834-row training set infeasible, so it trains on a further 8,000-row stratified subsample;
LinearSVC uses the same 500,000-row stratified subsample as the tree families' Optuna tuning in
SS16; SGDClassifier, being O(m x n) and incremental-friendly, trains on the full set like the tree
ensembles.
"""

code_stage1 = '''from sklearn.model_selection import train_test_split  # noqa: E402

from unfallatlas.models.boosting import build_lightgbm_binary_pipeline  # noqa: E402
from unfallatlas.models.svm import (  # noqa: E402
    build_linear_svm_binary_pipeline,
    build_rbf_svm_binary_pipeline,
    build_sgd_hinge_binary_pipeline,
)

SEED = 42
SUB_N = 500_000

# Stratified subsample for LinearSVC and for Optuna tuning in SS16 (stratify
# on 3-class UKATGEORIE to preserve the KSI share).
sub_n = min(SUB_N, len(train))
if sub_n < len(train):
    train_sub, _ = train_test_split(
        train, train_size=sub_n, random_state=SEED, stratify=train["UKATGEORIE"]
    )
else:
    train_sub = train
groups_sub = train_sub["UJAHR"].values
X_sub, y_sub_bin = split_features_target_binary(train_sub)

# Further stratified subsample for the RBF kernel (Einheit 6 SS9/SS10: O(m^2)-O(m^3)).
train_svc_sub, _ = train_test_split(
    train_sub, train_size=8_000, random_state=SEED, stratify=train_sub["UKATGEORIE"]
)
groups_svc_sub = train_svc_sub["UJAHR"].values
X_svc_sub, y_svc_sub = split_features_target_binary(train_svc_sub)

tree_preprocessor_bin = build_preprocessor(scale_for_linear=False)

BINARY_BUILDERS = {
    "random_forest": lambda: build_random_forest_pipeline(
        tree_preprocessor_bin, class_weight="balanced"
    ),
    "xgboost": lambda: build_xgboost_pipeline(tree_preprocessor_bin, use_gpu=_use_gpu_resolved),
    "lightgbm": lambda: build_lightgbm_binary_pipeline(tree_preprocessor_bin),
    "catboost": lambda: build_catboost_pipeline(tree_preprocessor_bin, use_gpu=_use_gpu_resolved),
    "svm_linear": lambda: build_linear_svm_binary_pipeline(linear_preprocessor_bin),
    "svm_sgd": lambda: build_sgd_hinge_binary_pipeline(linear_preprocessor_bin),
    "svm_rbf": lambda: build_rbf_svm_binary_pipeline(linear_preprocessor_bin),
}

# (X, y) each family's Stage-1 fit uses - full train for everything except
# the two subsampled SVM variants (Einheit 6 SS10 complexity table).
BINARY_STAGE1_DATA = {
    "random_forest": (X_train_bin, y_train_bin),
    "xgboost": (X_train_bin, y_train_bin),
    "lightgbm": (X_train_bin, y_train_bin),
    "catboost": (X_train_bin, y_train_bin),
    "svm_linear": (X_sub, y_sub_bin),
    "svm_sgd": (X_train_bin, y_train_bin),
    "svm_rbf": (X_svc_sub, y_svc_sub),
}


def _binary_fit_kwargs(family: str, y_fit) -> dict:
    """xgboost/catboost have no class_weight constructor kwarg - balanced
    weighting is applied via sample_weight at fit time instead, exactly
    like the 3-class xgboost_balanced/catboost_balanced pattern in SS4."""
    if family in ("xgboost", "catboost"):
        return {"classify__sample_weight": balanced_sample_weight(y_fit)}
    return {}


_log_progress(f"Starting binary Stage 1: {len(BINARY_BUILDERS)} candidate families.")
for _family, _build_fn in BINARY_BUILDERS.items():
    _X_fit, _y_fit = BINARY_STAGE1_DATA[_family]
    _fit_kwargs = _binary_fit_kwargs(_family, _y_fit)
    _name = f"binary_{_family}_balanced"
    _model = _load_or_fit_binary(
        _name,
        lambda _build_fn=_build_fn, _X_fit=_X_fit, _y_fit=_y_fit, _fit_kwargs=_fit_kwargs: (
            _build_fn().fit(_X_fit, _y_fit, **_fit_kwargs)
        ),
    )
    _score_binary_on_validation(_name, _model, family=_family)
    binary_comparison_rows[-1]["n_train"] = len(_y_fit)
_log_progress("Binary Stage 1 complete.")

binary_comparison_df = pd.DataFrame(binary_comparison_rows)
binary_comparison_df.sort_values("macro_f1", ascending=False)
'''

md_15 = """## 15 — Binary Champion Selection

Gate-aware, exactly like SS5's rule for the 3-class problem (reusing the same
`select_best_candidate` function, generalised with a `recall_col` parameter for this binary-KSI
reuse): highest macro-F1 among Stage-1 candidates whose Recall(KSI) clears 0.50, falling back to
the highest combined score if none do. Baselines are excluded from the championship (they exist to
bound the floor, not compete for it) - same convention as SS5.
"""

code_selection = '''stage1_only = binary_comparison_df[binary_comparison_df["family"].isin(BINARY_BUILDERS.keys())]
binary_champion_row = select_best_candidate(stage1_only, recall_col="recall_ksi")
binary_champion_family = binary_champion_row["family"]

_log_progress(
    f"Binary champion family (Stage 0/1 search): {binary_champion_family}  "
    f"Val macro-F1={binary_champion_row[\\'macro_f1\\']:.4f}  "
    f"Val recall(KSI)={binary_champion_row[\\'recall_ksi\\']:.4f}"
)
print(f"Binary champion family: {binary_champion_family}")
print(f"  Val-2023 macro_f1:    {binary_champion_row[\\'macro_f1\\']:.4f}")
print(f"  Val-2023 recall_ksi:  {binary_champion_row[\\'recall_ksi\\']:.4f}")
stage1_only.sort_values("macro_f1", ascending=False)
'''

new_cells = [
    nbformat.v4.new_markdown_cell(md_12),
    nbformat.v4.new_code_cell(code_split),
    nbformat.v4.new_markdown_cell(md_13),
    nbformat.v4.new_code_cell(code_stage0),
    nbformat.v4.new_markdown_cell(md_14),
    nbformat.v4.new_code_cell(code_stage1),
    nbformat.v4.new_markdown_cell(md_15),
    nbformat.v4.new_code_cell(code_selection),
]

nb.cells[header_idx : header_idx + 3] = new_cells
nbformat.write(nb, NB_PATH)
print(f"Replaced 3 old cells at index {header_idx} with {len(new_cells)} new cells.")
```

- [ ] **Step 2: Run it and verify**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
uv run python /tmp/insert_binary_champion_search.py
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/03_A3_Phase.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.strip().startswith('## 1'):
        print(c.source.strip().split(chr(10))[0])
"
```

Expected: `## 10 — ...`, `## 11 — 3-Class Ceiling...`, `## 12 — Binary KSI Reframing...`, `## 13 — Binary Champion Search: Stage 0 Baselines`, `## 14 — Binary Champion Search: Stage 1 Candidates (Trees + SVM)`, `## 15 — Binary Champion Selection`, in that order, with no leftover `## §` headers anywhere.

- [ ] **Step 3: Sync mirror and commit**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat(notebook/a3): replace assumed binary champion with a genuine Stage0/1 search incl. SVM"
```

Do **not** execute the notebook yet — Task 6 adds the tuning/refit/eval/save cells this section's variables feed into; execute once, after Task 7, to avoid partial reruns.

---

### Task 6: Tune the binary champion, refit, gate-optimal threshold, single Test-2024 evaluation, save artifacts

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (via a throwaway Python script)
- Modify (generated): `notebooks/03_A3_Phase.py` (via `jupytext --sync`, never by hand)

**Interfaces:**
- Consumes: everything Task 5 produces (`binary_champion_family`, `BINARY_BUILDERS`, `BINARY_STAGE1_DATA`, `_binary_fit_kwargs`, `X_sub`/`y_sub_bin`/`groups_sub`, `X_svc_sub`/`y_svc_sub`/`groups_svc_sub`, `X_val_bin`/`y_val_bin`/`X_test_bin`/`y_test_bin`/`X_train_bin`/`y_train_bin`); `find_best_binary_threshold` (Task 1).
- Produces: `pipeline_binary_final` (fitted `Pipeline`), `best_params: dict`, `best_threshold: float`, `best_f1_val: float`, `metrics_binary_test: dict`, `gate_passed: bool`, and on-disk `data/processed/a3_binary_best_model.joblib`, `data/processed/a3_binary_model_card.json`, `data/processed/a3_binary_model_comparison.csv`. Task 7 consumes all of these.

- [ ] **Step 1: Write the replacement-cell script**

Create `/tmp/insert_binary_tuning_and_save.py` (throwaway):

```python
import nbformat

NB_PATH = "notebooks/03_A3_Phase.ipynb"
nb = nbformat.read(NB_PATH, as_version=4)

optuna_idx = None
save_idx = None
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and "study_binary = optuna.create_study" in c.source:
        optuna_idx = i
    if c.cell_type == "code" and "a3_binary_best_model.joblib" in c.source:
        save_idx = i

assert optuna_idx is not None, "old binary Optuna cell not found"
assert save_idx is not None, "old binary artifact-save cell not found"
assert save_idx - optuna_idx == 3, (
    f"expected exactly 4 old cells (optuna, refit+threshold, test-eval, save), "
    f"got optuna_idx={optuna_idx} save_idx={save_idx}"
)

md_16 = """## 16 — Binary Hyperparameter Tuning (Optuna, winning family only)

Only `binary_champion_family` (SS15) is tuned - mirrors SS7's per-family search-space pattern from
the 3-class problem, extended to cover all seven binary candidate families. 20 trials, 3-fold
GroupKFold-by-year, on the same subsample the family used for tuning speed; the winner is refit on
its full appropriate training scale afterward (SS17).
"""

code_tuning = '''from sklearn.metrics import f1_score  # noqa: E402
import optuna  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_TRIALS = 20

BINARY_PARAM_SPACES = {
    "random_forest": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__max_depth": trial.suggest_int("max_depth", 5, 30),
        "classify__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    },
    "xgboost": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__max_depth": trial.suggest_int("max_depth", 3, 10),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classify__reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
    },
    "lightgbm": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "classify__max_depth": trial.suggest_int("max_depth", 5, 12),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "classify__min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "classify__reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    },
    "catboost": lambda trial: {
        "classify__iterations": trial.suggest_int("iterations", 100, 500),
        "classify__depth": trial.suggest_int("depth", 3, 10),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classify__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    },
    "svm_linear": lambda trial: {"classify__C": trial.suggest_float("C", 1e-3, 1e2, log=True)},
    "svm_sgd": lambda trial: {"classify__alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True)},
    "svm_rbf": lambda trial: {
        "classify__C": trial.suggest_float("C", 1e-1, 1e2, log=True),
        "classify__gamma": trial.suggest_float("gamma", 1e-3, 1e1, log=True),
    },
}

# CV data per family - the same (X, y) Stage 1 used, plus year-groups for
# GroupKFold. svm_rbf gets its own smaller groups array.
BINARY_CV_DATA = {
    "random_forest": (X_sub, y_sub_bin, groups_sub),
    "xgboost": (X_sub, y_sub_bin, groups_sub),
    "lightgbm": (X_sub, y_sub_bin, groups_sub),
    "catboost": (X_sub, y_sub_bin, groups_sub),
    "svm_linear": (X_sub, y_sub_bin, groups_sub),
    "svm_sgd": (X_sub, y_sub_bin, groups_sub),
    "svm_rbf": (X_svc_sub, y_svc_sub, groups_svc_sub),
}

X_cv, y_cv, groups_cv = BINARY_CV_DATA[binary_champion_family]
build_champion_fn = BINARY_BUILDERS[binary_champion_family]
param_space_fn = BINARY_PARAM_SPACES[binary_champion_family]


def binary_champion_objective(trial):
    params = param_space_fn(trial)
    gkf = GroupKFold(n_splits=3)
    fold_scores = []
    for tr_idx, va_idx in gkf.split(X_cv, y_cv, groups=groups_cv):
        p = build_champion_fn()
        p.set_params(**params)
        fit_kwargs = _binary_fit_kwargs(binary_champion_family, y_cv.iloc[tr_idx])
        p.fit(X_cv.iloc[tr_idx], y_cv.iloc[tr_idx], **fit_kwargs)
        pred = p.predict(X_cv.iloc[va_idx])
        fold_scores.append(f1_score(y_cv.iloc[va_idx], pred, average="macro"))
    return float(np.mean(fold_scores))


study_binary = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name=f"binary_{binary_champion_family}",
)
study_binary.optimize(binary_champion_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print(f"\\nTuned family: {binary_champion_family}")
print(f"Best CV macro-F1: {study_binary.best_value:.4f}")
print(f"Best params: {study_binary.best_params}")
'''

md_17 = """## 17 — Binary Refit, Gate-Optimal Threshold & Test-2024 Evaluation

The tuned champion is refit on its full appropriate training scale (SS14's `BINARY_STAGE1_DATA`
entry for the family - the complete 2016-2022 training set for every family except the two
subsampled SVM variants), then thresholded via `find_best_binary_threshold` on Val-2023 (works
with either `predict_proba` or `decision_function`, so this step is family-agnostic), then
evaluated exactly once on Test-2024 - the single time this section touches the test set.
"""

code_refit = '''from unfallatlas.models.evaluate import find_best_binary_threshold  # noqa: E402

best_params = study_binary.best_params
X_refit, y_refit = BINARY_STAGE1_DATA[binary_champion_family]

pipeline_binary_final = build_champion_fn()
pipeline_binary_final.set_params(**{f"classify__{k}": v for k, v in best_params.items()})
refit_fit_kwargs = _binary_fit_kwargs(binary_champion_family, y_refit)
pipeline_binary_final.fit(X_refit, y_refit, **refit_fit_kwargs)
print(f"Refit {binary_champion_family} on {len(y_refit):,} rows complete.")

if hasattr(pipeline_binary_final, "predict_proba"):
    y_val_scores_bin = pipeline_binary_final.predict_proba(X_val_bin)[:, 1]
else:
    y_val_scores_bin = pipeline_binary_final.decision_function(X_val_bin)

best_threshold, best_val_metrics = find_best_binary_threshold(y_val_bin.values, y_val_scores_bin)
best_f1_val = best_val_metrics["macro_f1"]

print(f"\\nGate-optimal binary threshold (Val-2023): {best_threshold:.4f}")
print(f"Val macro-F1 at optimal threshold: {best_f1_val:.4f}")
print(f"Val recall(KSI) at optimal threshold: {best_val_metrics[\\'recall_ksi\\']:.4f}")

if hasattr(pipeline_binary_final, "predict_proba"):
    y_test_scores_bin = pipeline_binary_final.predict_proba(X_test_bin)[:, 1]
else:
    y_test_scores_bin = pipeline_binary_final.decision_function(X_test_bin)
y_test_pred_bin = (y_test_scores_bin >= best_threshold).astype(int)

metrics_binary_test = evaluate_binary_predictions(y_test_bin.values, y_test_pred_bin)
gate_passed = meets_binary_acceptance_criteria(metrics_binary_test)

print("\\nBinary KSI — Test-2024 metrics:")
for k, v in metrics_binary_test.items():
    if k != "confusion_matrix":
        print(f"  {k}: {v:.4f}")

cm = pd.DataFrame(
    metrics_binary_test["confusion_matrix"],
    index=["True KSI", "True slight"],
    columns=["Pred KSI", "Pred slight"],
)
print(f"\\nConfusion Matrix:\\n{cm.to_string()}")
print(f"\\nBinary gate passed: {gate_passed}")
'''

md_18 = "## 18 — Binary Artifacts: Save Pipeline & Model Card\\n"

code_save = '''import json  # noqa: E402

import joblib  # noqa: E402

joblib.dump(
    pipeline_binary_final,
    BASE / "data" / "processed" / "a3_binary_best_model.joblib",
)

binary_comparison_df.to_csv(
    BASE / "data" / "processed" / "a3_binary_model_comparison.csv", index=False
)

binary_model_card = {
    "model_type": "binary_ksi_vs_slight",
    "target_encoding": "1 = KSI (UKATGEORIE in {1,2}), 0 = slight (UKATGEORIE = 3)",
    "champion_family": binary_champion_family,
    "winning_strategy": f"binary_{binary_champion_family}_balanced",
    "selection_rule": (
        "Stage 0/1 champion search across random_guess, majority_class, logistic_regression, "
        "random_forest, xgboost, lightgbm, catboost, svm_linear, svm_sgd, svm_rbf (all "
        "class-weighted/balanced) on Val-2023; gate-aware selection via select_best_candidate("
        "recall_col=\\'recall_ksi\\') - highest macro-F1 among candidates with recall_ksi >= 0.50, "
        "falling back to highest combined score if none clear the gate."
    ),
    "stage0_1_comparison": binary_comparison_df.drop(columns=["confusion_matrix"]).to_dict(
        orient="records"
    ),
    "gate_reformulation_reason": (
        "3-class gate (macro-F1 >= 0.55 AND Recall(class-1) >= 0.50) is unreachable with public "
        "Unfallatlas features: empirical ceiling macro-F1 = 0.424 over 19 configurations, "
        "Cramer's V <= 0.13 for strongest features, ~90x odds-lift required for class-1 precision. "
        "KSI-vs-slight is the domain-standard framing (Santos 2022, Pakgohar 2021, Schloessler 2024)."
    ),
    "best_hyperparameters": best_params,
    "optimal_threshold_val_2023": float(best_threshold),
    "val_2023_macro_f1": float(best_f1_val),
    "test_2024_metrics": metrics_binary_test,
    "acceptance_gate": "binary macro-F1 >= 0.55 AND Recall(KSI) >= 0.50",
    "acceptance_gate_passed": bool(gate_passed),
    "provenance": {
        "rows_train": int(len(y_train_bin)),
        "rows_val": int(len(y_val_bin)),
        "rows_test": int(len(y_test_bin)),
        "optuna_trials": OPTUNA_TRIALS,
        "subsample_size_svm_linear_and_boosting_tuning": SUB_N,
        "subsample_size_svm_rbf": len(y_svc_sub),
        "run_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "random_seed": SEED,
    },
}

with open(BASE / "data" / "processed" / "a3_binary_model_card.json", "w") as f:
    json.dump(binary_model_card, f, indent=2, default=str)

print("Saved:")
print(f"  {BASE / \\'data\\' / \\'processed\\' / \\'a3_binary_best_model.joblib\\'}")
print(f"  {BASE / \\'data\\' / \\'processed\\' / \\'a3_binary_model_card.json\\'}")
print(f"  {BASE / \\'data\\' / \\'processed\\' / \\'a3_binary_model_comparison.csv\\'}")
'''

new_cells = [
    nbformat.v4.new_markdown_cell(md_16),
    nbformat.v4.new_code_cell(code_tuning),
    nbformat.v4.new_markdown_cell(md_17),
    nbformat.v4.new_code_cell(code_refit),
    nbformat.v4.new_markdown_cell(md_18),
    nbformat.v4.new_code_cell(code_save),
]

nb.cells[optuna_idx : optuna_idx + 4] = new_cells
nbformat.write(nb, NB_PATH)
print(f"Replaced 4 old cells at index {optuna_idx} with {len(new_cells)} new cells.")
```

- [ ] **Step 2: Run it and verify**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
uv run python /tmp/insert_binary_tuning_and_save.py
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/03_A3_Phase.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.strip().startswith('## 1'):
        print(c.source.strip().split(chr(10))[0])
"
```

Expected: adds `## 16 — Binary Hyperparameter Tuning (Optuna, winning family only)`, `## 17 — Binary Refit, Gate-Optimal Threshold & Test-2024 Evaluation`, `## 18 — Binary Artifacts: Save Pipeline & Model Card` to the previous list, in order, still no `## §` headers anywhere.

- [ ] **Step 3: Sync mirror and commit**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat(notebook/a3): tune/refit/evaluate/save the searched binary champion (generic per-family)"
```

Do **not** execute the notebook yet — Task 7 adds the visualization and results-summary cells that read this task's outputs; execute once, after Task 7.

---

### Task 7: Binary Pareto-front visualization, honest results summary, real execution

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (via a throwaway Python script, then real execution via `nbclient`)
- Modify (generated): `notebooks/03_A3_Phase.py` (via `jupytext --sync`, never by hand)

**Interfaces:**
- Consumes: `plot_binary_f1_recall_front` (Task 3); `binary_comparison_df`, `BINARY_BUILDERS`, `binary_champion_family`, `best_threshold`, `best_f1_val`, `metrics_binary_test`, `gate_passed`, `study_binary.best_value`, `best_params` (Task 6); `plt` (already imported earlier in the notebook, at the 3-class Pareto-plot cell — no re-import needed by the time this cell runs).
- Produces: `reports/figures/a3_binary_f1_recall_front.png`; a fully real, non-placeholder `## 19 — Results Summary: Binary KSI Classification` markdown cell; a genuinely executed notebook (all outputs persisted).

- [ ] **Step 1: Write the replacement-cell script**

Create `/tmp/insert_binary_plot_and_summary.py` (throwaway):

```python
import nbformat

NB_PATH = "notebooks/03_A3_Phase.ipynb"
nb = nbformat.read(NB_PATH, as_version=4)

summary_idx = None
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and c.source.strip().startswith("### §10 — Results Summary"):
        summary_idx = i
        break
assert summary_idx is not None, "old results-summary cell not found"

code_plot = '''from unfallatlas.viz.metrics_viz import plot_binary_f1_recall_front  # noqa: E402

plot_input_df = binary_comparison_df[
    binary_comparison_df["family"].isin(list(BINARY_BUILDERS.keys()))
][["model", "family", "macro_f1", "recall_ksi"]].copy()

fig, ax = plt.subplots(figsize=(10, 6))
plot_binary_f1_recall_front(
    plot_input_df,
    ax=ax,
    gate_f1=0.55,
    gate_recall=0.50,
    title="Pareto Front: Macro-F1 vs. Recall(KSI) — binary champion search (Stage 0/1)",
)
fig.tight_layout()

out_path = BASE / "reports" / "figures" / "a3_binary_f1_recall_front.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Binary champion-search front plot saved to {out_path}")
'''

md_19_placeholder = """## 19 — Results Summary: Binary KSI Classification

<!-- FILL IN AFTER EXECUTING THIS NOTEBOOK (Task 7 Step 3 of the SVM/binary-champion-search plan)
     — replace every TODO below with the real values printed by ## 15 / ## 16 / ## 17. Do not
     leave a TODO in the committed notebook. -->

The binary KSI reformulation overcomes the Bayes-ceiling of the 3-class formulation (§11), and
this time the binary champion was chosen via a genuine Stage 0/Stage 1 search across ten
candidates (three baselines, four tree-ensemble families, three SVM variants) rather than
inherited from the 3-class champion.

**Binary champion: `TODO_FAMILY`** (selected via `select_best_candidate(recall_col="recall_ksi")`
over the §14 Stage-1 comparison — see `data/processed/a3_binary_model_comparison.csv` for the
full ten-way table and `reports/figures/a3_binary_f1_recall_front.png` for the visual comparison).

| Metric | Val-2023 | Test-2024 | Gate |
|---|---|---|---|
| macro-F1 | TODO | **TODO** | ≥ 0.55 TODO |
| Recall(KSI) | TODO | **TODO** | ≥ 0.50 TODO |
| Recall(slight) | — | TODO | — |

**Gate passed: TODO.**

- **Model**: `TODO_FAMILY`, class-weighted/balanced, tuned via Optuna (20 trials, 3-fold
  GroupKFold-by-year). Best CV macro-F1 during search: TODO. Winning hyperparameters: `TODO`.
- **Threshold**: gate-optimal decision threshold found via `find_best_binary_threshold` on
  Val-2023 (maximise macro-F1 subject to Recall(KSI) ≥ 0.50) → **TODO**.
- **Runner-up candidates**: TODO — name the second- and third-best Stage-1 candidates by Val
  macro-F1 and their scores, to show the margin between the champion and the field.
- **Test-2024 confusion matrix** (rows = true, cols = predicted): TODO (copy from §17's printed
  `cm` table).
- **Test-2024 evaluation performed exactly once**, after threshold selection on Val-2023 — no
  test-set peeking.
- **Comparison to the naive-relabel estimate**: relabeling the 3-class champion's existing
  predictions (no retraining) gave binary macro-F1 = 0.552 (documented in the Technical Review).
  This purpose-built, searched-and-tuned champion reaches **TODO**.
- **Gate artefacts**: `data/processed/a3_binary_best_model.joblib`,
  `data/processed/a3_binary_model_card.json`, `data/processed/a3_binary_model_comparison.csv`
  (all saved and present in the repo).

The binary formulation is the methodological standard in the road-safety ML literature
(Santos 2022, Pakgohar 2021, Schlößler 2024) and provides the verifiable, evidence-based gate for
this portfolio — see §11 for the empirical and arithmetic proof that the original 3-class gate is
structurally unreachable with the available Unfallatlas features.
"""

new_cells = [
    nbformat.v4.new_code_cell(code_plot),
    nbformat.v4.new_markdown_cell(md_19_placeholder),
]

nb.cells[summary_idx : summary_idx + 1] = new_cells
nbformat.write(nb, NB_PATH)
print(f"Replaced 1 old cell at index {summary_idx} with {len(new_cells)} new cells.")
```

- [ ] **Step 2: Run it and verify structure**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
uv run python /tmp/insert_binary_plot_and_summary.py
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/03_A3_Phase.ipynb', as_version=4)
print('total cells:', len(nb.cells))
for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.strip().startswith('#'):
        first = c.source.strip().split(chr(10))[0]
        if first.startswith('##') or first.startswith('# '):
            print(first)
"
```

Expected: a clean, gap-free sequence `# Unfallatlas...` down through `## 19 — Results Summary: Binary KSI Classification`, with every number 1–19 appearing exactly once and no `§` glyph anywhere.

- [ ] **Step 3: Execute the entire notebook for real**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
uv run python -c "
import nbformat
from nbclient import NotebookClient

NB_PATH = 'notebooks/03_A3_Phase.ipynb'
nb = nbformat.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}})
client.execute()
nbformat.write(nb, NB_PATH)
print('Notebook executed and saved successfully.')
"
```

Expected: prints `Notebook executed and saved successfully.` with no raised exception. Budget up to ~20–30 minutes (3-class Stage 0–1/champion/imbalance/tuning cells are checkpoint-cached from prior runs and should be fast; the new binary Stage 1 trains 4 full-scale tree ensembles plus 3 SVM variants, then Optuna-tunes one family for 60 fits). If it raises a `CellExecutionError`, read the traceback, fix the underlying cause, and re-run this exact command from scratch — do not resume partway (checkpointing under `CHECKPOINT_DIR`/`BINARY_CHECKPOINT_DIR` means a re-run after a fix reuses every already-completed fit).

- [ ] **Step 4: Read back the real results and fill in the `## 19` placeholders**

```bash
uv run python -c "
import nbformat
nb = nbformat.read('notebooks/03_A3_Phase.ipynb', as_version=4)
for c in nb.cells:
    if c.cell_type == 'code' and 'Binary champion family:' in c.source:
        for out in c.outputs:
            if 'text' in out:
                print(out['text'])
    if c.cell_type == 'code' and 'Binary gate passed' in c.source:
        for out in c.outputs:
            if 'text' in out:
                print(out['text'])
"
cat data/processed/a3_binary_model_comparison.csv
cat data/processed/a3_binary_model_card.json
```

Using the real printed values (champion family, Val/Test macro-F1, Val/Test recall_ksi, threshold, best CV score during tuning, winning hyperparameters, confusion matrix, gate-passed status, second/third-place Stage-1 candidates from the CSV), open `notebooks/03_A3_Phase.ipynb` and replace every `TODO` in the `## 19` cell with the actual measured numbers — do not invent numbers or reuse any figure from this plan document (all figures in this plan are illustrative only; the real run's numbers are authoritative). This can be done via the same small `nbformat` read/find-cell-by-substring/edit-source/write pattern used throughout this plan, or directly in the Jupyter/VS Code UI followed by a save.

- [ ] **Step 5: Sync mirror**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
```

- [ ] **Step 6: Full verification sweep**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
grep -c "TODO" notebooks/03_A3_Phase.ipynb  # must print 0
```

Expected: all tests pass, ruff clean, zero remaining `TODO` occurrences in the notebook. If `ruff format --check .` flags files, run `uv run ruff format <file>` only on files this plan touched — never a blanket repo-wide format.

- [ ] **Step 7: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py \
        data/processed/a3_binary_best_model.joblib \
        data/processed/a3_binary_model_card.json \
        data/processed/a3_binary_model_comparison.csv \
        reports/figures/a3_binary_f1_recall_front.png
git commit -m "feat(notebook/a3): execute binary champion search for real; document actual results in §19"
```

---

## Self-review notes (for the plan author, already applied above)

- **Spec coverage**: "why not do a new champion finder for binary" → Task 5 runs a genuine Stage 0/Stage 1 champion search on the binary target across all 7 candidate families (not just SVM), with gate-aware selection identical in spirit to the existing 3-class §5. "kernel modules" → the RBF-kernel `SVC` variant (Task 2, wired into Task 5/6) covers this explicitly (Einheit 6 §7–§9). "SVM is a MUST" → Task 2 makes SVM a first-class, tested pipeline builder family, evaluated on equal footing with every other family, not a side comparison. "Visualization for the binary stuff" → Task 3/7 deliver `reports/figures/a3_binary_f1_recall_front.png`, now showing the *entire* ten-candidate Stage 0/1 search, not just SVM vs. one fixed champion. "Looks tacked together" → Task 4 fixes the duplicated `§9`/`§10` numbering (verified via direct inspection: `## 9 —` and `## 10 —` were each used twice), and Tasks 5–7 replace the binary section's ad hoc single-family cells with a structure that mirrors the 3-class section's own numbered, documented, checkpointed discipline.
- **No placeholders**: the `TODO` markers in Task 7's `## 19` cell are an intentional, explicitly-flagged exception — the real numbers are unknowable until the notebook actually executes (Task 7 Step 3), and Step 4 requires replacing every one before commit (Step 6 mechanically verifies zero `TODO` occurrences remain). Every other step in every task has complete, runnable code with no deferred logic.
- **Type/name consistency check**: `find_best_binary_threshold` (Task 1) and the generalized `select_best_candidate(rows, recall_col=...)` (Task 1) are both imported/called by name exactly as defined, in Tasks 5–7. `build_linear_svm_binary_pipeline`/`build_sgd_hinge_binary_pipeline`/`build_rbf_svm_binary_pipeline` (Task 2) match the names used in Task 5's `BINARY_BUILDERS` dict verbatim. `plot_binary_f1_recall_front` (Task 3) matches the name/signature used in Task 7's plot cell verbatim, including the new `title=` override parameter. `BINARY_BUILDERS`, `BINARY_STAGE1_DATA`, `_binary_fit_kwargs`, `binary_champion_family`, `X_sub`/`y_sub_bin`/`groups_sub`, `X_svc_sub`/`y_svc_sub`/`groups_svc_sub` are each defined exactly once (Task 5 or 6) and consumed by name, unchanged, in every later task — confirmed consistent across Tasks 5, 6, and 7.
- **Risk mitigation carried over from the prior binary-KSI-reframe session's lessons** (see this repo's `docs/superpowers/plans/2026-07-14-binary-ksi-reframe.md` audit history): every notebook edit in this plan happens via a single, idempotent, content-substring-anchored `nbformat` script per task (never raw JSON hand-editing, never a blanket `jupytext`-mirror edit), execution happens once per task-group via a single foreground `nbclient` call with a generous timeout (no background/duplicate kernel processes), and every cell-locating assertion (`assert ... is not None`, `assert save_idx - optuna_idx == 3`, etc.) fails loudly if the notebook's structure doesn't match what the script expects, rather than silently inserting cells in the wrong place.
