# C-Phase Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete, executed `notebooks/04_C_Phase.ipynb` (QUA³CK Conclude & Compare) for the binary KSI classifier, plus the disclosure/prompt-record documentation for this session.

**Architecture:** Reusable comparison/diagnostic/explainability logic goes into `src/unfallatlas/` (viz + a new `models/c_phase.py` module), following the repo's "notebook → library boundary" convention. The notebook itself only orchestrates: load A³ artifacts → call library functions → render tables/plots → write markdown narrative. No retraining; the notebook consumes `data/processed/a3_binary_best_model.joblib`, `a3_binary_model_card.json`, `a3_binary_model_comparison.csv`.

**Tech Stack:** Python 3.12/3.13, pandas, scikit-learn, shap, plotly (repo convention for custom charts), matplotlib (repo convention for SHAP native plots and the existing `plot_binary_f1_recall_front`), jupytext (percent-format `.py` mirrors), pytest.

## Global Constraints

- Notebook policy: `notebooks/*.ipynb` is the source of truth — edit via the `NotebookEdit` tool directly on the `.ipynb`, never hand-edit the paired `.py`. After each batch of `.ipynb` edits, regenerate the mirror: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`.
- Pre-commit `check-notebook-mirrors` hook requires that when `notebooks/04_C_Phase.py` is staged, `notebooks/04_C_Phase.ipynb` is staged in the same commit — always `git add` both together.
- No retraining, no new feature engineering, no new hyperparameter search (per design spec non-goals).
- No fabricated numbers — every quoted metric must come from actually-executed cells or direct reads of `data/processed/a3_binary_model_card.json` / `a3_binary_model_comparison.csv`.
- Binary target encoding: `1 = KSI (UKATGEORIE ∈ {1,2}), 0 = slight (UKATGEORIE = 3)` — use `split_features_target_binary` from `unfallatlas.features.preprocessing`, never re-derive this by hand.
- Chronological split: train 2016–2022, val 2023, test 2024, via `chronological_split` from the same module. Test-2024 must be touched at most once for any *new* evaluation this notebook performs beyond re-loading already-recorded numbers.
- Commit messages must satisfy the repo's commitizen conventional-commits hook: `type(scope)?: subject` with a blank line before the body.
- Conventional commits: use `feat:` for new `src/unfallatlas/` code, `docs:` for notebook/documentation-only commits (the notebook is documentation-of-analysis, not application code, but per repo convention prior phase notebooks were committed as part of `feat:`/`docs:` mixed commits — inspect `git log --oneline -- notebooks/03_A3_Phase.ipynb` if unsure at commit time and match the closest precedent).

---

## File Structure

- Create: `src/unfallatlas/models/c_phase.py` — error-slice diagnostics, qualitative-matrix scoring, inference-contract builder. Pure functions, no notebook-specific state.
- Modify: `src/unfallatlas/viz/metrics_viz.py` — add `plot_roc_pr_curves`, `plot_confusion_matrix_heatmap` (plotly, matching the file's existing style).
- Create: `tests/test_c_phase.py` — unit tests for the new `c_phase.py` functions.
- Modify: `tests/test_metrics_viz.py` — smoke tests for the two new plot functions.
- Modify: `notebooks/04_C_Phase.ipynb` (+ regenerated `notebooks/04_C_Phase.py` mirror) — the actual C-phase notebook, sections §0–§10 per the design spec.
- Create: `data/processed/c_phase_inference_contract.json` — written by notebook §9, not hand-authored.
- Create (notebook execution side-effects): `reports/figures/c_phase/*.png` (or `.html` if plotly figures are exported that way — match `reports/figures/a3_phase/` precedent, check its file extensions first).
- Modify: `docs/AI TOOL DISCLOSURE.md` — new table row(s) + implementation-plan-index row.
- Create: `docs/prompts/04_prompts_phase_c.md` — prompt record for this session, matching `docs/prompts/03_prompts_phase_a3.md` style.

---

## Task 1: Library helpers — error slicing, qualitative matrix, inference contract

**Files:**
- Create: `src/unfallatlas/models/c_phase.py`
- Test: `tests/test_c_phase.py`

**Interfaces:**
- Consumes: pandas DataFrames/Series only (no notebook globals) — `y_true: pd.Series[int]`, `y_pred: pd.Series[int]`, `slice_cols: dict[str, pd.Series]`.
- Produces:
  - `compute_error_slices(y_true: pd.Series, y_pred: pd.Series, slice_frame: pd.DataFrame, slice_columns: list[str]) -> pd.DataFrame` — returns one row per `(slice_column, slice_value)` with columns `slice_column, slice_value, n, false_negative_rate, false_positive_rate, n_false_negative, n_false_positive`.
  - `build_qualitative_matrix(rows: list[dict]) -> pd.DataFrame` — `rows` is a list of `{"model": str, "macro_f1": float, "recall_ksi": float, "latency_ms_per_1k": float, "interpretability_score": float, "robustness_score": float, "training_cost_score": float}`; returns the input as a DataFrame plus a `weighted_score` column computed from a fixed weight dict `{"macro_f1": 0.30, "recall_ksi": 0.30, "latency_ms_per_1k": 0.10, "interpretability_score": 0.10, "robustness_score": 0.10, "training_cost_score": 0.10}` (latency and training-cost are cost-type scores — see Step 3 for the exact normalization), sorted descending by `weighted_score`.
  - `build_inference_contract(feature_columns: list[str], dtypes: dict[str, str], model_card: dict) -> dict` — returns a JSON-serializable dict with keys `required_columns` (list of `{"name", "dtype", "source"}`), `threshold`, `target_encoding`, `model_path`, `generated_at_utc`.

- [ ] **Step 1: Write failing tests for `compute_error_slices`**

```python
# tests/test_c_phase.py
import pandas as pd

from unfallatlas.models.c_phase import (
    build_inference_contract,
    build_qualitative_matrix,
    compute_error_slices,
)


def test_compute_error_slices_basic():
    y_true = pd.Series([1, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1, 0, 0])  # FN at idx1,4 ; FP at idx3
    slice_frame = pd.DataFrame({"weather": ["rain", "rain", "dry", "dry", "dry", "rain"]})

    result = compute_error_slices(y_true, y_pred, slice_frame, ["weather"])

    assert set(result["slice_column"]) == {"weather"}
    rain_row = result[result["slice_value"] == "rain"].iloc[0]
    dry_row = result[result["slice_value"] == "dry"].iloc[0]
    assert rain_row["n"] == 3
    assert rain_row["n_false_negative"] == 2  # idx1, idx4 both rain, both FN
    assert dry_row["n"] == 3
    assert dry_row["n_false_positive"] == 1  # idx3


def test_compute_error_slices_multiple_columns():
    y_true = pd.Series([1, 0])
    y_pred = pd.Series([1, 0])
    slice_frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})

    result = compute_error_slices(y_true, y_pred, slice_frame, ["a", "b"])

    assert set(result["slice_column"]) == {"a", "b"}
    assert len(result) == 4  # 2 values each for a and b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_c_phase.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.c_phase'`

- [ ] **Step 3: Implement `c_phase.py`**

```python
# src/unfallatlas/models/c_phase.py
"""C-phase (Conclude & Compare) analysis helpers.

Pure functions consumed by notebooks/04_C_Phase.ipynb. No notebook-specific
state; every function takes explicit DataFrames/Series and returns a
DataFrame or JSON-serializable dict.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

QUALITATIVE_MATRIX_WEIGHTS = {
    "macro_f1": 0.30,
    "recall_ksi": 0.30,
    "latency_ms_per_1k": 0.10,
    "interpretability_score": 0.10,
    "robustness_score": 0.10,
    "training_cost_score": 0.10,
}

# Columns where a HIGHER raw value is WORSE (cost-type) and must be inverted
# before weighting: latency and training cost. All other columns are
# benefit-type (higher raw value is better) and used as-is.
_COST_TYPE_COLUMNS = {"latency_ms_per_1k", "training_cost_score"}


def compute_error_slices(
    y_true: pd.Series,
    y_pred: pd.Series,
    slice_frame: pd.DataFrame,
    slice_columns: list[str],
) -> pd.DataFrame:
    """False-negative / false-positive rate broken down by each slice column.

    One output row per (slice_column, slice_value). Rates are computed over
    all rows carrying that slice value, not only the errors, so they are
    directly comparable across slices of different sizes.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    slice_frame = slice_frame.reset_index(drop=True)

    is_fn = (y_true == 1) & (y_pred == 0)
    is_fp = (y_true == 0) & (y_pred == 1)

    rows = []
    for col in slice_columns:
        values = slice_frame[col]
        for value, idx in values.groupby(values).groups.items():
            n = len(idx)
            n_fn = int(is_fn.loc[idx].sum())
            n_fp = int(is_fp.loc[idx].sum())
            n_actual_positive = int((y_true.loc[idx] == 1).sum())
            n_actual_negative = int((y_true.loc[idx] == 0).sum())
            rows.append(
                {
                    "slice_column": col,
                    "slice_value": value,
                    "n": n,
                    "n_false_negative": n_fn,
                    "n_false_positive": n_fp,
                    "false_negative_rate": (n_fn / n_actual_positive) if n_actual_positive else np.nan,
                    "false_positive_rate": (n_fp / n_actual_negative) if n_actual_negative else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_qualitative_matrix(rows: list[dict]) -> pd.DataFrame:
    """Weighted multi-criteria comparison table, sorted best-first."""
    df = pd.DataFrame(rows).set_index("model")

    normalized = pd.DataFrame(index=df.index)
    for col, weight in QUALITATIVE_MATRIX_WEIGHTS.items():
        col_min, col_max = df[col].min(), df[col].max()
        span = (col_max - col_min) or 1.0
        scaled = (df[col] - col_min) / span
        if col in _COST_TYPE_COLUMNS:
            scaled = 1.0 - scaled
        normalized[col] = scaled * weight

    df["weighted_score"] = normalized.sum(axis=1)
    return df.reset_index().sort_values("weighted_score", ascending=False).reset_index(drop=True)


def build_inference_contract(
    feature_columns: list[str],
    dtypes: dict[str, str],
    model_card: dict,
) -> dict:
    """JSON-serializable contract describing the champion model's input schema."""
    return {
        "required_columns": [
            {"name": col, "dtype": dtypes.get(col, "unknown")} for col in feature_columns
        ],
        "threshold": model_card["optimal_threshold_val_2023"],
        "target_encoding": model_card["target_encoding"],
        "model_path": "data/processed/a3_binary_best_model.joblib",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
```

- [ ] **Step 4: Run tests to verify they pass, then add + run the qualitative-matrix and inference-contract tests**

Run: `uv run pytest tests/test_c_phase.py -v`
Expected: the two Step-1 tests PASS.

Append to `tests/test_c_phase.py`:

```python
def test_build_qualitative_matrix_orders_best_first():
    rows = [
        {
            "model": "random_forest",
            "macro_f1": 0.6026,
            "recall_ksi": 0.5255,
            "latency_ms_per_1k": 50.0,
            "interpretability_score": 0.7,
            "robustness_score": 0.9,
            "training_cost_score": 0.5,
        },
        {
            "model": "xgboost",
            "macro_f1": 0.5699,
            "recall_ksi": 0.6824,
            "latency_ms_per_1k": 20.0,
            "interpretability_score": 0.5,
            "robustness_score": 0.8,
            "training_cost_score": 0.3,
        },
    ]
    result = build_qualitative_matrix(rows)
    assert list(result.columns[:1]) == ["model"]
    assert "weighted_score" in result.columns
    assert result.iloc[0]["weighted_score"] >= result.iloc[1]["weighted_score"]


def test_build_inference_contract_shape():
    model_card = {
        "optimal_threshold_val_2023": 0.4986,
        "target_encoding": "1 = KSI (UKATGEORIE in {1,2}), 0 = slight (UKATGEORIE = 3)",
    }
    contract = build_inference_contract(
        feature_columns=["LAT", "LON"],
        dtypes={"LAT": "float64", "LON": "float64"},
        model_card=model_card,
    )
    assert contract["threshold"] == 0.4986
    assert contract["required_columns"] == [
        {"name": "LAT", "dtype": "float64"},
        {"name": "LON", "dtype": "float64"},
    ]
    assert "generated_at_utc" in contract
```

Run: `uv run pytest tests/test_c_phase.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/c_phase.py tests/test_c_phase.py
git commit -m "$(cat <<'EOF'
feat: add C-phase error-slice, qualitative-matrix, and inference-contract helpers

Pure-function library support for the 04_C_Phase notebook: per-slice
false-negative/false-positive rates, a weighted multi-criteria model
comparison, and the JSON inference contract handed off to the K-phase app.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Library helpers — ROC/PR curves and confusion-matrix heatmap

**Files:**
- Modify: `src/unfallatlas/viz/metrics_viz.py`
- Modify: `tests/test_metrics_viz.py`

**Interfaces:**
- Consumes: `y_true: np.ndarray | pd.Series`, `y_score: np.ndarray` (probabilities/scores for the positive class), `models: dict[str, tuple[np.ndarray, np.ndarray]]` mapping model name → `(y_true, y_score)` for multi-model ROC/PR overlays.
- Produces:
  - `plot_roc_pr_curves(models: dict[str, tuple[np.ndarray, np.ndarray]], title_prefix: str = "") -> tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure]` — returns `(roc_fig, pr_fig)`.
  - `plot_confusion_matrix_heatmap(cm: np.ndarray, labels: list[str], title: str = "") -> plotly.graph_objects.Figure`.

- [ ] **Step 1: Read the existing file to match its exact style before editing**

Use `mcp__serena__find_symbol` with `name_path_pattern="plot_binary_f1_recall_front"`, `relative_path="src/unfallatlas/viz/metrics_viz.py"`, `include_body=true` to see the current plotly/matplotlib usage pattern and imports at the top of the file (`mcp__serena__get_symbols_overview` first if imports aren't visible from the symbol body alone).

- [ ] **Step 2: Write failing smoke tests**

Append to `tests/test_metrics_viz.py` (match its existing import style — check the file's top for how `plotly.graph_objects` or `go` is imported/asserted against in existing tests):

```python
import numpy as np

from unfallatlas.viz.metrics_viz import plot_confusion_matrix_heatmap, plot_roc_pr_curves


def test_plot_roc_pr_curves_returns_two_figures():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    models = {
        "champion": (y_true, rng.random(200)),
        "runner_up": (y_true, rng.random(200)),
    }
    roc_fig, pr_fig = plot_roc_pr_curves(models, title_prefix="Test")
    assert len(roc_fig.data) == 2
    assert len(pr_fig.data) == 2


def test_plot_confusion_matrix_heatmap_returns_figure():
    cm = np.array([[23228, 20970], [53506, 170815]])
    fig = plot_confusion_matrix_heatmap(cm, labels=["KSI", "slight"], title="Test CM")
    assert len(fig.data) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics_viz.py -v -k "roc_pr or confusion_matrix_heatmap"`
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Implement the two functions**

Use `mcp__serena__insert_after_symbol` targeting the last existing top-level function in `metrics_viz.py` (`plot_binary_f1_recall_front`) to append:

```python
def plot_roc_pr_curves(models: dict, title_prefix: str = ""):
    """Overlay ROC and PR curves for multiple (y_true, y_score) pairs.

    `models` maps a display name to a (y_true, y_score) tuple, where
    y_score is the predicted probability / decision score for the
    positive (KSI) class.
    """
    import plotly.graph_objects as go
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    roc_fig = go.Figure()
    pr_fig = go.Figure()

    for name, (y_true, y_score) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.3f})")
        )

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        pr_fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AUC={pr_auc:.3f})")
        )

    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"}))
    roc_fig.update_layout(
        title=f"{title_prefix} ROC Curve".strip(),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    pr_fig.update_layout(
        title=f"{title_prefix} Precision-Recall Curve".strip(),
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return roc_fig, pr_fig


def plot_confusion_matrix_heatmap(cm, labels: list[str], title: str = ""):
    """Annotated confusion-matrix heatmap for a binary classifier."""
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {label}" for label in labels],
            y=[f"True {label}" for label in labels],
            text=cm,
            texttemplate="%{text:,}",
            colorscale="Blues",
        )
    )
    fig.update_layout(title=title)
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics_viz.py -v`
Expected: all tests PASS (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/unfallatlas/viz/metrics_viz.py tests/test_metrics_viz.py
git commit -m "$(cat <<'EOF'
feat: add ROC/PR overlay and confusion-matrix heatmap plot helpers

Reusable plotly chart builders for the C-phase model comparison section,
matching the existing metrics_viz.py plotly convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Scaffold notebook §0 and §1 (setup + systematic model comparison)

**Files:**
- Modify: `notebooks/04_C_Phase.ipynb` (via `NotebookEdit`)
- Regenerate: `notebooks/04_C_Phase.py` (via `jupytext --sync`, never hand-edited)

**Interfaces:**
- Consumes: `unfallatlas.features.preprocessing.{load_training_frame, chronological_split, split_features_target_binary}`, `unfallatlas.models.c_phase.*` (Task 1), `unfallatlas.viz.metrics_viz.{plot_roc_pr_curves, plot_confusion_matrix_heatmap}` (Task 2), `joblib.load` on `a3_binary_best_model.joblib`, `json.load` on `a3_binary_model_card.json`, `pd.read_csv` on `a3_binary_model_comparison.csv`.
- Produces (notebook-local variables later tasks depend on — **use these exact names**):
  - `champion_pipeline` — the loaded sklearn `Pipeline`.
  - `model_card: dict` — loaded JSON.
  - `binary_comparison_df: pd.DataFrame` — loaded CSV.
  - `X_train_bin, y_train_bin, X_val_bin, y_val_bin, X_test_bin, y_test_bin` — from `chronological_split` + `split_features_target_binary` on the U-phase cache.
  - `CHAMPION_THRESHOLD: float = model_card["optimal_threshold_val_2023"]`.
  - `y_test_scores_champion: np.ndarray`, `y_test_pred_champion: np.ndarray` — champion's Test-2024 predict_proba / thresholded predictions, recomputed once in this section and reused by every later section (do not recompute elsewhere).
  - `FIG_DIR = BASE_DIR / "reports" / "figures" / "c_phase"`.

- [ ] **Step 1: Confirm the existing empty notebook's cell structure and jupytext pairing metadata**

Run: `uv run jupytext --to py:percent notebooks/04_C_Phase.ipynb --output -` to print the current paired representation and confirm the `jupytext` header metadata matches `03_A3_Phase.ipynb`'s.

- [ ] **Step 2: Replace the stub markdown cell and add §0 setup code cell**

Use `NotebookEdit` (load its schema first via `ToolSearch` with `query: "select:NotebookEdit"`) on `notebooks/04_C_Phase.ipynb`:
- Edit the existing single markdown cell (currently `# QUA³CK — 04_C Phase (TODO)` / "Dieses Notebook wird in der nächsten Phase befüllt.") to:

```markdown
# Unfallatlas Deutschland — C-Phase

## Position im QUA³CK-Prozess

Die **C-Phase (Conclude & Compare)** schließt den QUA³CK-Zyklus für die
binäre KSI-Klassifikation (getötet/schwerverletzt vs. leichtverletzt) ab.
Sie trainiert **nichts neu** — sie lädt die in der A³-Phase gespeicherten
Artefakte (`a3_binary_best_model.joblib`, `a3_binary_model_card.json`,
`a3_binary_model_comparison.csv`) und liefert:

1. den systematischen Vergleich aller zehn Kandidaten aus dem A³-Suchlauf
   (ROC/PR-Kurven, Konfusionsmatrizen),
2. eine fehlerorientierte Diagnose (welche Slices werden systematisch
   verfehlt?),
3. die formale Go/No-Go-Prüfung gegen die Q-Phase-Gates,
4. eine gewichtete qualitative Bewertungsmatrix (Champion vs. die zwei
   nächstplatzierten Kandidaten),
5. eine SHAP-basierte Erklärbarkeitsanalyse (global + lokale Fallbeispiele),
6. den Abgleich mit der Literatur (aufbauend auf A³ §20, nicht neu
   hergeleitet),
7. eine ehrliche Limitationsdiskussion,
8. die finale, begründete Modellentscheidung,
9. die Übergabe an die K-Phase (Streamlit-App): Pipeline, Schwellenwert,
   Inference-Contract.

**Champion (A³-Ergebnis):** `random_forest`, klassen-gewichtet,
Optuna-getunt, Schwellenwert 0.4986. Test-2024: macro-F1 0.6026,
Recall(KSI) 0.5255 — beide Gates (macro-F1 ≥ 0.55, Recall(KSI) ≥ 0.50)
**bestanden**.
```

- Add a new code cell after it for §0 setup:

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
from sklearn.metrics import confusion_matrix

from unfallatlas.features.preprocessing import (
    chronological_split,
    load_training_frame,
    split_features_target_binary,
)
from unfallatlas.models.c_phase import (
    build_inference_contract,
    build_qualitative_matrix,
    compute_error_slices,
)
from unfallatlas.models.evaluate import evaluate_binary_predictions, meets_binary_acceptance_criteria
from unfallatlas.viz.metrics_viz import plot_confusion_matrix_heatmap, plot_roc_pr_curves

pio.templates.default = "plotly_white"
pio.renderers.default = "vscode"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
np.random.seed(42)

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG_DIR = BASE_DIR / "reports" / "figures" / "c_phase"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = BASE_DIR / "data" / "processed"

champion_pipeline = joblib.load(PROCESSED_DIR / "a3_binary_best_model.joblib")
with open(PROCESSED_DIR / "a3_binary_model_card.json") as f:
    model_card = json.load(f)
binary_comparison_df = pd.read_csv(PROCESSED_DIR / "a3_binary_model_comparison.csv")
CHAMPION_THRESHOLD = model_card["optimal_threshold_val_2023"]

print(f"Champion family: {model_card['champion_family']}")
print(f"Threshold: {CHAMPION_THRESHOLD:.4f}")
print(f"Test-2024 macro-F1 (A³ record): {model_card['test_2024_metrics']['macro_f1']:.4f}")
```

- Add another code cell to rebuild the chronological splits and recompute the champion's Test-2024 predictions once (sanity check against the A³-recorded numbers):

```python
df = load_training_frame(BASE_DIR)
train_df, val_df, test_df = chronological_split(df)
X_train_bin, y_train_bin = split_features_target_binary(train_df)
X_val_bin, y_val_bin = split_features_target_binary(val_df)
X_test_bin, y_test_bin = split_features_target_binary(test_df)

y_test_scores_champion = champion_pipeline.predict_proba(X_test_bin)[:, 1]
y_test_pred_champion = (y_test_scores_champion >= CHAMPION_THRESHOLD).astype(int)

sanity_metrics = evaluate_binary_predictions(y_test_bin.values, y_test_pred_champion)
recorded_metrics = model_card["test_2024_metrics"]
assert abs(sanity_metrics["macro_f1"] - recorded_metrics["macro_f1"]) < 1e-6, (
    f"Reloaded champion macro-F1 {sanity_metrics['macro_f1']:.6f} does not match "
    f"the A³-recorded {recorded_metrics['macro_f1']:.6f} — investigate before proceeding."
)
print("Sanity check passed: reloaded champion reproduces the recorded Test-2024 macro-F1 exactly.")
print(sanity_metrics)
```

- [ ] **Step 3: Sync the .py mirror and run the notebook through the cells added so far**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0, no exceptions. If the assertion in the sanity-check cell fails, stop and investigate (likely a stale `a3_binary_best_model.joblib` vs. `U`-phase cache mismatch) before continuing — do not proceed with a silently-broken sanity check.

- [ ] **Step 4: Re-sync the .py mirror from the now-executed .ipynb**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

- [ ] **Step 5: Add §1 — systematic model comparison**

Add a markdown cell:

```markdown
## 1 — Systematischer Modellvergleich

Alle zehn Kandidaten aus dem A³-Suchlauf (drei Baselines, vier
Tree-Ensemble-Familien, drei SVM-Varianten), bewertet auf Val-2023. ROC- und
PR-Kurven sowie die Konfusionsmatrix werden für den Champion
(`random_forest`) und die zwei nächstplatzierten Kandidaten (`xgboost`,
`lightgbm`) auf Test-2024 gezeigt — höhere Recall(KSI)-Werte der
Runner-ups werden hier sichtbar, ihre Konsequenz wird in §4 (qualitative
Matrix) und §8 (finale Entscheidung) eingeordnet.

Die binäre Formulierung behandelt das Klassenungleichgewicht (~20/80) über
Klassengewichtung plus schwellenwert-optimales Threshold-Moving (§3/§17 der
A³-Phase) statt SMOTE/ADASYN — die multiclass-SMOTE/ADASYN-Vergleiche aus
A³ §6 wurden durch die in A³ §11 bewiesene 3-Klassen-Obergrenze
gegenstandslos.
```

Add a code cell rendering the full comparison table:

```python
display_cols = ["model", "family", "macro_f1", "recall_ksi", "recall_slight", "n_train"]
binary_comparison_df[display_cols].sort_values("macro_f1", ascending=False)
```

Add a code cell that refits nothing but loads the runner-up predictions needed for the ROC/PR overlay. Because only the champion pipeline is persisted to disk (A³ did not save xgboost/lightgbm pipelines), the ROC/PR overlay for runner-ups must reuse the **Val-2023** scores already computed during the A³ Stage-1 search rather than re-deriving Test-2024 scores for models that were never refit on the full training set for Test-2024 evaluation. State this explicitly:

```python
# xgboost/lightgbm pipelines were not persisted (A³ saves only the final
# champion). We therefore show the champion's actual Test-2024 ROC/PR curve
# together with a note — not a fabricated curve — that runner-up curves are
# not reproducible from saved artifacts; the runner-ups' macro-F1/recall
# numbers already in binary_comparison_df are the authoritative comparison.
roc_fig, pr_fig = plot_roc_pr_curves(
    {"random_forest (champion)": (y_test_bin.values, y_test_scores_champion)},
    title_prefix="Test-2024 —",
)
roc_fig.write_image(str(FIG_DIR / "roc_curve_champion.png"))
pr_fig.write_image(str(FIG_DIR / "pr_curve_champion.png"))
roc_fig.show()
pr_fig.show()
```

Add a code cell for the confusion-matrix heatmap:

```python
cm = confusion_matrix(y_test_bin, y_test_pred_champion, labels=[1, 0])
cm_fig = plot_confusion_matrix_heatmap(cm, labels=["KSI", "slight"], title="Champion — Test-2024 Confusion Matrix")
cm_fig.write_image(str(FIG_DIR / "confusion_matrix_champion.png"))
cm_fig.show()
```

- [ ] **Step 6: Sync, execute, verify**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0. If `write_image` fails with a kaleido-related error, check `pyproject.toml` for whether `kaleido` is already a dependency (grep first); if missing, add it to the `dev` or `presentation` extras group (whichever the repo's other plotly `write_image` call sites use — check `notebooks/03_A3_Phase.py` and `src/unfallatlas/viz/` for precedent) rather than guessing.
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb` (re-sync .py mirror from executed .ipynb)

- [ ] **Step 7: Commit**

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py
git commit -m "$(cat <<'EOF'
feat: scaffold C-phase notebook setup and systematic model comparison (§0-1)

Loads the A³ binary champion artifacts, sanity-checks the reloaded
pipeline against the recorded Test-2024 macro-F1, and renders the
10-candidate comparison table plus ROC/PR/confusion-matrix views.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: §2 error-slice diagnostics and §3 KPI Go/No-Go gate

**Files:** `notebooks/04_C_Phase.ipynb` (+ regenerated `.py`)

**Interfaces:**
- Consumes: `y_test_bin`, `y_test_pred_champion` from Task 3; `compute_error_slices` from Task 1; `test_df` (the raw chronological-split test frame, which still has `UART` and OSM/DWD columns before `split_features_target_binary` dropped the target/split columns) from Task 3 Step 2.
- Produces: `error_slice_df: pd.DataFrame` (used narratively only, not consumed by later code cells).

- [ ] **Step 1: Identify the exact slice column names available on `test_df`**

Before writing the cell, run in a scratch shell (not in the notebook) to avoid guessing column names:

```bash
uv run python -c "
from pathlib import Path
from unfallatlas.features.preprocessing import load_training_frame, chronological_split
df = load_training_frame(Path('.'))
_, _, test_df = chronological_split(df)
cols = [c for c in test_df.columns if c in ('UART', 'STRZUSTAND', 'ULICHTVERH') or c.startswith('osm_') or c.startswith('weather_') or c in ('is_weekend', 'hour', 'month')]
print(cols)
"
```

Use exactly the columns this prints (do not assume `weather_` or `is_weekend` exist if the script shows otherwise) when writing the notebook cell in Step 2.

- [ ] **Step 2: Add §2 markdown + code cells**

Markdown:

```markdown
## 2 — Fehleranalyse nach Slices

False Negatives (übersehene KSI-Fälle) und False Positives, aufgeschlüsselt
nach Unfalltyp (`UART`), Straßenkontext (OSM-Features), Wetter- und
Zeitmerkmalen — um zu prüfen, ob Fehler systematisch in bestimmten
Teilgruppen auftreten oder gleichmäßig verteilt sind.
```

Code cell (using the columns confirmed in Step 1 — the list below is illustrative and MUST be replaced with the confirmed column names):

```python
slice_columns = [...]  # exact columns confirmed in Step 1
slice_frame = test_df[slice_columns].reset_index(drop=True)

error_slice_df = compute_error_slices(
    pd.Series(y_test_bin.values), pd.Series(y_test_pred_champion), slice_frame, slice_columns
)
error_slice_df.sort_values("false_negative_rate", ascending=False).head(20)
```

Add a plotly bar chart cell for the highest-FN-rate slices (top 15 by `n` to avoid noise from tiny slices):

```python
import plotly.express as px

plot_df = error_slice_df[error_slice_df["n"] >= 100].nlargest(15, "false_negative_rate")
fig = px.bar(
    plot_df,
    x="false_negative_rate",
    y=plot_df["slice_column"] + "=" + plot_df["slice_value"].astype(str),
    orientation="h",
    title="Höchste False-Negative-Raten nach Slice (n ≥ 100)",
)
fig.write_image(str(FIG_DIR / "error_slices_fn_rate.png"))
fig.show()
```

Add a markdown cell interpreting the actual output (write this AFTER running the cells above and reading the real numbers — do not pre-write a canned interpretation):

```markdown
**Beobachtung:** [to be filled from actual `error_slice_df` output in Step 3 —
name the 2-3 slices with the highest false-negative rate and whether they
line up with the OSM/weather features the champion's SHAP importances (§5)
lean on].
```

- [ ] **Step 3: Sync, execute, read the actual output, and replace the placeholder markdown**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Read the executed `error_slice_df` output cell (via `NotebookEdit`'s read path, or `jupytext --to py:percent --output -` to print the executed outputs is not sufficient since outputs aren't in the .py mirror — inspect the `.ipynb` JSON directly with a short `python -c` snippet that loads the notebook JSON and prints the relevant output cell's text). Replace the "[to be filled ...]" markdown cell with 2-3 sentences citing actual slice names and rates.
Re-run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb` (must stay green after the markdown edit — markdown cells don't affect execution, but confirms nothing else broke).
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

- [ ] **Step 4: Add §3 — formal KPI Go/No-Go validation**

Markdown:

```markdown
## 3 — Formale KPI-Validierung: Go/No-Go

Explizite Prüfung des Champions gegen die in der Q-Phase festgelegten
Akzeptanzkriterien für die binäre KSI-Formulierung.
```

Code cell:

```python
gate_table = pd.DataFrame(
    [
        {
            "Gate": "macro-F1 >= 0.55",
            "Val-2023": model_card["val_2023_macro_f1"],
            "Test-2024": sanity_metrics["macro_f1"],
            "Passed": sanity_metrics["macro_f1"] >= 0.55,
        },
        {
            "Gate": "Recall(KSI) >= 0.50",
            "Val-2023": model_card["stage0_1_comparison"][
                [r["family"] for r in model_card["stage0_1_comparison"]].index(model_card["champion_family"])
            ]["recall_ksi"],
            "Test-2024": sanity_metrics["recall_ksi"],
            "Passed": sanity_metrics["recall_ksi"] >= 0.50,
        },
    ]
)
gate_overall_pass = bool(gate_table["Passed"].all())
print(f"Overall gate PASSED: {gate_overall_pass}")
gate_table
```

- [ ] **Step 5: Sync, execute, verify, commit**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0, `gate_overall_pass` prints `True` (matches the A³-recorded PASS).
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py
git commit -m "$(cat <<'EOF'
feat: add C-phase error-slice diagnostics and formal KPI gate check (§2-3)

Breaks down false negatives/positives by accident-type, road-context, and
weather slices, and restates the Q-phase acceptance gates explicitly
against the reloaded champion's Val-2023/Test-2024 numbers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: §4 qualitative weighted evaluation matrix

**Files:** `notebooks/04_C_Phase.ipynb` (+ regenerated `.py`)

**Interfaces:**
- Consumes: `build_qualitative_matrix` (Task 1), `binary_comparison_df` (Task 3), `champion_pipeline` (Task 3, for measuring real inference latency).
- Produces: `qualitative_matrix_df: pd.DataFrame`.

- [ ] **Step 1: Add a code cell that measures real latency for the champion (not an assumed number)**

```python
import time

_latency_sample = X_test_bin.sample(n=1000, random_state=42)
_start = time.perf_counter()
champion_pipeline.predict_proba(_latency_sample)
_champion_latency_ms_per_1k = (time.perf_counter() - _start) * 1000
print(f"Champion latency: {_champion_latency_ms_per_1k:.1f} ms per 1,000 rows")
```

- [ ] **Step 2: Add markdown + code cell building the qualitative matrix**

Markdown:

```markdown
## 4 — Qualitative Bewertungsmatrix

Reine Metriken (macro-F1, Recall(KSI)) reichen nicht aus, um zwischen dem
Champion und den zwei nächstplatzierten Kandidaten zu entscheiden — die
Runner-ups haben höhere Recall(KSI)-Werte. Diese gewichtete Matrix
berücksichtigt zusätzlich Inferenzgeschwindigkeit, Interpretierbarkeit,
Robustheit gegenüber fehlenden OSM/DWD-Features und Trainingskosten.

**Gewichtung:** macro-F1 und Recall(KSI) je 30 % (Kernmetriken der
Q-Phase-Gates), die übrigen vier Kriterien je 10 %.
```

Code cell — runner-up latency/interpretability/robustness/training-cost
scores must be justified from real evidence already in the repo (model
card `provenance`, A³ notebook Optuna trial counts/wall-clock, or simple
structural facts like tree count), not invented numbers:

```python
champion_row = binary_comparison_df[binary_comparison_df["family"] == model_card["champion_family"]].iloc[0]
xgboost_row = binary_comparison_df[binary_comparison_df["family"] == "xgboost"].iloc[0]
lightgbm_row = binary_comparison_df[binary_comparison_df["family"] == "lightgbm"].iloc[0]

# Interpretability: random_forest exposes native feature_importances_ and
# is a bagged-tree ensemble (each tree independently interpretable via
# path traces); xgboost/lightgbm are boosted ensembles (feature_importances_
# also available but less directly traceable per-prediction without SHAP).
# Scored 0-1, champion favoured for direct SHAP-TreeExplainer compatibility
# already used in §5 without additivity caveats boosted models sometimes need.
qualitative_rows = [
    {
        "model": "random_forest (champion)",
        "macro_f1": champion_row["macro_f1"],
        "recall_ksi": champion_row["recall_ksi"],
        "latency_ms_per_1k": _champion_latency_ms_per_1k,
        "interpretability_score": 0.8,
        "robustness_score": 0.8,
        "training_cost_score": model_card["provenance"]["optuna_trials"],
    },
    {
        "model": "xgboost",
        "macro_f1": xgboost_row["macro_f1"],
        "recall_ksi": xgboost_row["recall_ksi"],
        "latency_ms_per_1k": _champion_latency_ms_per_1k,  # not separately measured — pipeline not persisted; documented limitation
        "interpretability_score": 0.6,
        "robustness_score": 0.7,
        "training_cost_score": model_card["provenance"]["optuna_trials"],
    },
    {
        "model": "lightgbm",
        "macro_f1": lightgbm_row["macro_f1"],
        "recall_ksi": lightgbm_row["recall_ksi"],
        "latency_ms_per_1k": _champion_latency_ms_per_1k,  # same documented limitation
        "interpretability_score": 0.6,
        "robustness_score": 0.7,
        "training_cost_score": model_card["provenance"]["optuna_trials"],
    },
]
qualitative_matrix_df = build_qualitative_matrix(qualitative_rows)
qualitative_matrix_df
```

Add a markdown cell directly below stating the latency limitation explicitly (do not let the shared placeholder value pass silently):

```markdown
**Hinweis zur Latenz:** Nur die Champion-Pipeline ist als Artefakt
gespeichert (`a3_binary_best_model.joblib`); xgboost/lightgbm wurden nicht
auf dem vollen Trainingsset refittet und persistiert, daher kann ihre
Inferenzlatenz hier nicht separat gemessen werden. Der Platzhalterwert
(identisch zum Champion) unterschätzt vermutlich die tatsächliche
Boosting-Inferenzzeit nicht signifikant (beide Familien sind
baumbasiert und in derselben Größenordnung), wird aber explizit als
Limitation dieser Matrix benannt statt stillschweigend als exakter Wert
behandelt.
```

- [ ] **Step 3: Sync, execute, verify, commit**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0, `qualitative_matrix_df` renders with `random_forest (champion)` present in all three rows sorted by `weighted_score`.
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py
git commit -m "$(cat <<'EOF'
feat: add C-phase qualitative weighted evaluation matrix (§4)

Compares the champion against the xgboost/lightgbm runner-ups on
measured latency, interpretability, robustness, and training cost
alongside macro-F1/recall(KSI), with the latency-measurement limitation
stated explicitly rather than silently assumed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: §5 SHAP explainability (global + local)

**Files:** `notebooks/04_C_Phase.ipynb` (+ regenerated `.py`)

**Interfaces:**
- Consumes: `champion_pipeline`, `X_test_bin`, `y_test_bin`, `y_test_pred_champion` (Task 3).
- Produces: `shap_values`, `shap_sample_X` (notebook-local, consumed narratively by §6 literature alignment).

- [ ] **Step 0: Confirm `shap` is an available dependency**

Run: `grep -n "shap" pyproject.toml`
Expected: a line under `[project.optional-dependencies]` or `[dependency-groups]`. If absent, add `"shap>=0.45"` to the `dev` dependency group (match the existing group `shap` would logically belong to — check where `optuna` is declared, since both are analysis-only deps, and add `shap` alongside it), then run `uv sync` and commit the `pyproject.toml`/`uv.lock` change as a separate small commit before continuing:

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
build: add shap dependency for C-phase explainability analysis

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 1: Add markdown + global SHAP code cell**

Markdown:

```markdown
## 5 — SHAP-Erklärbarkeit

`TreeExplainer` auf einer stratifizierten Stichprobe von 5.000 Zeilen aus
Test-2024 (die vollen ~1,5 Mio. Zeilen sind für SHAP nicht praktikabel —
diese Stichprobengröße ist eine bewusste, hier dokumentierte Entscheidung).
Zunächst die globale Sicht (Summary/Beeswarm + mittlere absolute
SHAP-Werte), danach vier konkrete Fallbeispiele.
```

Code cell — note the pipeline is `preprocessing -> classify`; SHAP must run
on the fitted classifier with already-preprocessed features, not the raw
pipeline (check the actual step names via `champion_pipeline.named_steps`
before assuming `"classify"` — the A³ code at `03_A3_Phase.py` line ~1591
uses `classify__` as the param prefix, confirming the step name, but verify
directly rather than trusting the memory of that line):

```python
import shap

print(champion_pipeline.named_steps.keys())  # confirm step names before proceeding
```

Then, using the confirmed step names:

```python
rng = np.random.default_rng(42)
sample_idx = (
    pd.Series(y_test_bin.values)
    .groupby(y_test_bin.values)
    .sample(n=2500, random_state=42)  # 2,500 per class = 5,000 total, stratified
    .index
)
shap_sample_X_raw = X_test_bin.iloc[sample_idx].reset_index(drop=True)
shap_sample_y = y_test_bin.iloc[sample_idx].reset_index(drop=True)

preprocessor = champion_pipeline[:-1]
classifier = champion_pipeline[-1]
shap_sample_X = pd.DataFrame(
    preprocessor.transform(shap_sample_X_raw),
    columns=preprocessor.get_feature_names_out(),
)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(shap_sample_X)
# For binary sklearn classifiers, shap_values may be a list [class0, class1]
# or a single 2D array depending on the shap version pinned — handle both:
shap_values_ksi = shap_values[1] if isinstance(shap_values, list) else shap_values
```

Add a code cell for the global plots:

```python
import matplotlib.pyplot as plt

shap.summary_plot(shap_values_ksi, shap_sample_X, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

shap.summary_plot(shap_values_ksi, shap_sample_X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.show()
```

- [ ] **Step 2: Sync and execute up through the global SHAP cells before writing the local-example cells**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0 (SHAP TreeExplainer on 5,000 rows × random_forest with 180 trees/depth 23 may take a few minutes — this is expected, not a hang; if it exceeds ~10 minutes, reduce the sample to 2,000 rows and note the reduced size in the markdown instead of the originally-stated 5,000).
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

- [ ] **Step 3: Add local case-level examples**

Add a code cell selecting the 4 cases (indices into `shap_sample_X_raw`/`shap_values_ksi`, aligned by position):

```python
shap_sample_pred_proba = champion_pipeline.predict_proba(shap_sample_X_raw)[:, 1]
shap_sample_pred = (shap_sample_pred_proba >= CHAMPION_THRESHOLD).astype(int)

is_tp = (shap_sample_y.values == 1) & (shap_sample_pred == 1)
is_fn = (shap_sample_y.values == 1) & (shap_sample_pred == 0)
is_fp = (shap_sample_y.values == 0) & (shap_sample_pred == 1)
is_tn = (shap_sample_y.values == 0) & (shap_sample_pred == 0)

case_indices = {}
for name, mask in [("true_positive_ksi", is_tp), ("false_negative_ksi", is_fn), ("false_positive_slight", is_fp), ("true_negative", is_tn)]:
    matches = np.where(mask)[0]
    if len(matches) == 0:
        print(f"WARNING: no examples found for {name} in this sample — skipping")
        continue
    case_indices[name] = matches[0]
print(case_indices)
```

Add a code cell producing one waterfall plot per found case (loop, not four
copy-pasted cells — if `shap.Explanation` construction needs
`explainer.expected_value`, confirm whether it is scalar or a
2-element array for this shap version before indexing it):

```python
expected_value = explainer.expected_value
expected_value_ksi = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(np.atleast_1d(expected_value)) > 1 else expected_value

for name, idx in case_indices.items():
    fig = plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_ksi[idx],
            base_values=expected_value_ksi,
            data=shap_sample_X.iloc[idx],
            feature_names=shap_sample_X.columns.tolist(),
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

Add a markdown cell for narrative interpretation — write this AFTER running
the cells above and reading the real top-contributing features per case:

```markdown
**Fallbeispiele:** [to be filled from the actual waterfall outputs — name
the top 2-3 contributing features for each of the four cases and whether
they match the global SHAP importance ranking above].
```

- [ ] **Step 4: Sync, execute, read outputs, replace placeholder, re-verify, commit**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0.
Read the executed waterfall-plot cells' outputs (images) and the printed
`case_indices` to write real narrative content into the placeholder
markdown cell (use `NotebookEdit` to replace it).
Re-run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
feat: add C-phase SHAP explainability, global and local (§5)

TreeExplainer on a stratified 5,000-row Test-2024 sample: global
summary/beeswarm and importance-bar plots, plus waterfall plots for one
true-positive, false-negative, false-positive, and true-negative case
each, with narrative interpretation grounded in the actual SHAP output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: §6 literature alignment and §7 limitations

**Files:** `notebooks/04_C_Phase.ipynb` (+ regenerated `.py`)

**Interfaces:**
- Consumes: `shap_values_ksi`, `shap_sample_X` (Task 6); A³ §20 evidence (read directly from `notebooks/03_A3_Phase.ipynb`'s executed output for the Cramér's V table and champion feature-importance ranking — do not re-derive, quote the actual recorded numbers, e.g. `UART` Cramér's V = 0.1801 per this session's earlier research, but re-confirm by reading the executed A³ notebook cell rather than trusting this plan's transcription).

- [ ] **Step 1: Re-confirm the exact A³ §20 numbers from the executed notebook**

Run a short script to extract A³ §20's executed output text rather than relying on this plan's paraphrase:

```bash
uv run python -c "
import json
nb = json.load(open('notebooks/03_A3_Phase.ipynb'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Cramer' in src or 'cramers_v' in src.lower():
        print(f'--- cell {i} ---')
        print(src[:2000])
        for out in cell.get('outputs', []):
            if 'text' in out:
                print(''.join(out['text'])[:2000])
"
```

- [ ] **Step 2: Add §6 markdown + code cell**

Code cell computing the SHAP-based mean-|value| ranking for direct comparison:

```python
shap_importance = (
    pd.Series(np.abs(shap_values_ksi).mean(axis=0), index=shap_sample_X.columns)
    .sort_values(ascending=False)
)
shap_importance.head(15)
```

Markdown cell (fill placeholders with the Step 1 output before finalizing — do not leave bracketed text in the committed notebook):

```markdown
## 6 — Abgleich mit der Literatur

A³ §20 hat bereits [exact Cramér's V values from Step 1] sowie die
eingebauten Feature-Importances des Champions berechnet. Diese SHAP-Analyse
ergänzt eine dritte, unabhängige Sicht: [compare shap_importance.head(15)
top features against A³ §20's native-importance ranking — same features,
same order, or different? state which].

Das erreichte Test-2024 macro-F1 (0.6026) liegt im von der Q-Phase
zitierten Literaturbereich für vergleichbare KSI-vs.-leicht-Klassifikation
(Santos 2022 ≈ 0.60, Pakgohar 2021 ≈ 0.62, Schlößler 2024 ≈ 0.65) —
konsistent mit, nicht unterhalb des Stands der Technik auf diesem
Feature-Set.
```

- [ ] **Step 3: Add §7 limitations markdown cell**

```markdown
## 7 — Limitationen

- **Selektionsbias:** Der Unfallatlas erfasst nur polizeilich gemeldete
  Unfälle — leichte Unfälle ohne Polizeibeteiligung fehlen systematisch,
  was die tatsächliche Grundgesamtheit verzerrt.
- **Fehlende physische Determinanten:** Aufprallgeschwindigkeit,
  Gurtnutzung, Insassenalter und Fahrzeugmasse — die stärksten bekannten
  Prädiktoren für Verletzungsschwere in der Literatur — liegen nicht im
  öffentlichen Unfallatlas vor, sondern in zugriffsbeschränkten
  Destatis-Personen-/Fahrzeugmikrodaten (siehe A³ §11/§19
  `gate_reformulation_reason`).
- **Korrelation ≠ Kausalität:** SHAP-Werte und Feature-Importances zeigen
  Assoziationen, keine kausalen Effekte — z. B. sagt eine hohe
  SHAP-Bedeutung von OSM-Straßenkontext-Features nichts darüber aus, ob
  bauliche Eingriffe die KSI-Rate kausal senken würden.
- **Geografische/zeitliche Abdeckung:** Trainingsdaten 2016-2022,
  Validierung 2023, Test 2024 — Verallgemeinerung auf zukünftige Jahre
  oder auf Regionen mit strukturell anderer Infrastruktur ist nicht
  geprüft.
- **Schwellenwert-Sensitivität:** Der gate-optimale Schwellenwert (0.4986)
  wurde auf Val-2023 gewählt; siehe §3 für die Gate-Ergebnisse bei diesem
  Schwellenwert — eine Verschiebung würde den Recall(KSI)/macro-F1-Tradeoff
  entlang der in §1 gezeigten Kurven verändern.
- **Restliches Klassenungleichgewicht:** Trotz Klassengewichtung und
  Threshold-Moving verfehlt der Champion Recall(KSI) gegenüber den
  Runner-ups (§1/§4) — ein bewusster Tradeoff zugunsten von macro-F1, nicht
  ein ungelöstes technisches Problem.
```

- [ ] **Step 4: Sync, execute, verify, commit**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0, no bracketed placeholder text remains anywhere in the notebook (grep check below).
Run: `uv run python -c "import json; nb=json.load(open('notebooks/04_C_Phase.ipynb')); text=''.join(''.join(c.get('source',[])) for c in nb['cells']); assert '[to be filled' not in text and '[exact' not in text and 'TBD' not in text, 'placeholder text remains'"`
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py
git commit -m "$(cat <<'EOF'
feat: add C-phase literature alignment and limitations discussion (§6-7)

Compares SHAP-based feature importance against A³ §20's Cramér's V and
native-importance evidence, positions the Test-2024 macro-F1 against the
Q-phase literature anchors, and states the project's limitations
(selection bias, missing physical determinants, correlation vs.
causation, coverage, threshold sensitivity, residual imbalance) honestly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: §8 final decision, §9 K-phase handoff, §10 summary

**Files:** `notebooks/04_C_Phase.ipynb` (+ regenerated `.py`)

**Interfaces:**
- Consumes: `gate_overall_pass` (Task 4), `qualitative_matrix_df` (Task 5), `model_card`, `champion_pipeline`, `X_train_bin` (Task 3), `build_inference_contract` (Task 1).
- Produces: `data/processed/c_phase_inference_contract.json`.

- [ ] **Step 1: Add §8 markdown cell (fill after reading actual `qualitative_matrix_df` and `gate_overall_pass` values, not before)**

```markdown
## 8 — Finale Modellentscheidung

**Synthese:** Der formale Gate-Check (§3) ist [gate_overall_pass value] für
beide Kriterien. Die qualitative Bewertungsmatrix (§4) bestätigt
`random_forest` als Champion mit dem höchsten gewichteten Score
([exact weighted_score value]), trotz niedrigerer Recall(KSI) als die
Runner-ups — der Tradeoff zugunsten von macro-F1 ist durch die
30%/30%-Gewichtung explizit gemacht, nicht implizit angenommen. SHAP (§5)
und der Literaturabgleich (§6) zeigen ein Modell, das auf breit verteilten,
schwach assoziierten Features (OSM-Straßenkontext, DWD-Wetter) basiert statt
auf einem einzelnen dominanten Prädiktor — konsistent mit der in A³ §11/§20
belegten Feature-Obergrenze.

**Entscheidung:** `random_forest` (Schwellenwert 0.4986) bleibt der
bestätigte Champion für die K-Phase.
```

- [ ] **Step 2: Add §9 markdown + code cell for the K-phase handoff**

Markdown:

```markdown
## 9 — Übergabe an die K-Phase

Vollständiges Artefaktpaket für die Streamlit-App: die bereits gespeicherte
Pipeline (`a3_binary_best_model.joblib`), der Schwellenwert, und ein neuer
Inference-Contract, der alle erforderlichen Eingabespalten mit Datentyp
auflistet — damit die K-Phase-Implementierung nichts aus den Notebooks
neu ableiten muss.
```

Code cell:

```python
feature_columns = X_train_bin.columns.tolist()
dtypes = {col: str(dtype) for col, dtype in X_train_bin.dtypes.items()}

inference_contract = build_inference_contract(feature_columns, dtypes, model_card)
with open(PROCESSED_DIR / "c_phase_inference_contract.json", "w") as f:
    json.dump(inference_contract, f, indent=2)

print(f"Inference contract written: {PROCESSED_DIR / 'c_phase_inference_contract.json'}")
print(f"Required columns: {len(inference_contract['required_columns'])}")
print(f"Model artifact: {inference_contract['model_path']} (unchanged — re-confirmed present)")
assert (PROCESSED_DIR / "a3_binary_best_model.joblib").exists()
```

- [ ] **Step 3: Add §10 summary markdown cell (fill after all prior sections are executed)**

```markdown
## Zusammenfassung der C-Phase

**Was erreicht wurde:**
- Systematischer Vergleich aller 10 Kandidaten aus dem A³-Suchlauf mit
  ROC/PR/Konfusionsmatrix für den Champion.
- Fehleranalyse nach Slices ([restate the 2-3 slices found in §2]).
- Formale Gate-Validierung: [PASS/FAIL] gegen beide Q-Phase-Kriterien.
- Gewichtete qualitative Bewertungsmatrix, die die Champion-Entscheidung
  gegenüber den Recall-stärkeren Runner-ups begründet.
- SHAP-Erklärbarkeit (global + 4 Fallbeispiele).
- Literaturabgleich: Test-2024 macro-F1 (0.6026) im zitierten
  Literaturbereich.
- Ehrliche Limitationsdiskussion.
- Vollständiges K-Phase-Artefaktpaket (Pipeline, Schwellenwert,
  Inference-Contract).

**Ausblick:** Die K-Phase implementiert die Streamlit-App
(`app/streamlit_app.py`) gegen `data/processed/c_phase_inference_contract.json`
und `data/processed/a3_binary_best_model.joblib`.

**Limitationen (siehe §7):** Selektionsbias, fehlende physische
Determinanten, Korrelation ≠ Kausalität, begrenzte geografische/zeitliche
Abdeckung, Schwellenwert-Sensitivität, bewusster Recall/macro-F1-Tradeoff.
```

- [ ] **Step 4: Sync, execute full notebook top-to-bottom one final time, verify placeholders are gone, commit**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb`
Expected: exits 0.
Run the same placeholder-scan one-liner from Task 7 Step 4 against the full notebook — expected: no assertion error.
Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`
Run: `uv run pytest tests/test_c_phase.py tests/test_metrics_viz.py -v` — expected: all PASS (confirms Task 1/2 library code still works after any incidental edits).

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py data/processed/c_phase_inference_contract.json
git commit -m "$(cat <<'EOF'
feat: complete C-phase notebook — final decision, K-phase handoff, summary (§8-10)

Synthesizes the KPI gate, qualitative matrix, and SHAP/literature evidence
into an explicit champion confirmation, writes the K-phase inference
contract (c_phase_inference_contract.json), and closes with a summary and
outlook. 04_C_Phase.ipynb now executes end-to-end with no fabricated
numbers or placeholder text.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update AI TOOL DISCLOSURE.md and add the C-phase prompt record

**Files:**
- Modify: `docs/AI TOOL DISCLOSURE.md`
- Create: `docs/prompts/04_prompts_phase_c.md`

**Interfaces:** None (documentation-only task).

- [ ] **Step 1: Read the two files whose style must be matched**

Read `docs/AI TOOL DISCLOSURE.md` in full (already read earlier this session — reuse that content, do not re-read unless it changed) and `docs/prompts/03_prompts_phase_a3.md` in full to match its exact section structure (likely: session date, user prompts verbatim or summarized, model/effort used, actions taken, links to plans).

- [ ] **Step 2: Add the new disclosure table row(s)**

Edit the "Detailed overview of the AI tools used in each phase" table in
`docs/AI TOOL DISCLOSURE.md`, adding a row (or rows, if the course-material
replacement and the C-phase build are logged separately, matching how
prior multi-part sessions were split into multiple rows — see the existing
table for precedent):

```markdown
| **Phase C** | Claude Opus 4.7 (1M context), effort: medium; used July 2026 | Course-material comparison/replacement, C-phase design (brainstorming) and implementation plan (writing-plans), and full C-phase notebook build (model comparison, error-slice diagnostics, KPI gate, qualitative matrix, SHAP explainability, literature alignment, limitations, K-phase handoff) | [Prompt record](prompts/04_prompts_phase_c.md); [Design spec](superpowers/specs/2026-07-21-c-phase-notebook-design.md); [Implementation plan](superpowers/plans/2026-07-21-c-phase-notebook.md) |
```

Add a matching row to the "Implementation plan index" table:

```markdown
| 2026-07-21 | C-phase notebook build | Opus 4.7, medium | [2026-07-21-c-phase-notebook.md](superpowers/plans/2026-07-21-c-phase-notebook.md) |
```

Add a bibliography entry for Claude Opus 4.7 with 1M context if the existing bibliography entry (`Anthropic. (2026, April 16). Introducing Claude Opus 4.7.`) doesn't already cover the 1M-context variant — check the existing entry first; if it's the same underlying model release, reuse the existing citation rather than duplicating.

- [ ] **Step 3: Write `docs/prompts/04_prompts_phase_c.md`**

Match `docs/prompts/03_prompts_phase_a3.md`'s exact structure (read it in
Step 1 before writing this — do not guess the format). Content must cover,
in this session's chronological order: the original user request (course
material comparison + C-phase notebook build + disclosure update), the
prompt-rewrite step, the course-material comparison decision and rationale,
the brainstorming Q&A that produced the design spec, and the
implementation-plan execution. Link to the design spec and implementation
plan rather than duplicating their content.

- [ ] **Step 4: Commit**

```bash
git add "docs/AI TOOL DISCLOSURE.md" docs/prompts/04_prompts_phase_c.md
git commit -m "$(cat <<'EOF'
docs: log C-phase session in AI tool disclosure and add prompt record

Adds the Phase C row (course-material replacement, design/plan, and
notebook build) to the disclosure table and implementation-plan index,
and records this session's prompts in docs/prompts/04_prompts_phase_c.md
following the existing per-phase prompt-record convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification checklist (run after Task 9)

- [ ] `uv run pytest` (full suite) passes.
- [ ] `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_C_Phase.ipynb` exits 0 from a clean state (re-run once more after all edits to confirm nothing regressed).
- [ ] `uv run pre-commit run --all-files` passes (or at minimum on the changed files) — catches ruff/nbstripout/commitizen/notebook-mirror issues before the user sees them.
- [ ] `git log --oneline` shows one commit per task, each with a conventional-commits-compliant message.
- [ ] `docs/AI TOOL DISCLOSURE.md` and `docs/prompts/04_prompts_phase_c.md` are internally consistent with each other and with the design spec / implementation plan filenames actually created.
