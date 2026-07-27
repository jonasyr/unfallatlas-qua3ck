# Phase K Streamlit App ("Risk Explainer & Model Console") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the empty `app/streamlit_app.py` / `src/unfallatlas/viz/streamlit_app.py` stubs with a working, English-language, native-multipage Streamlit app (Overview, Risk Predictor, Why This Prediction, Model Comparison) that runs fully from artifacts already committed to the repo, with no notebook execution required after `git lfs pull`.

**Architecture:** A thin `app/streamlit_app.py` entry point wires four `st.Page` pages under `app/pages/`. All data loading, model inference, and pure logic lives in `src/unfallatlas/viz/streamlit_app.py` (unit-testable without a Streamlit runtime); pages only call `st.*` widgets and the shared module's functions. Existing `src/unfallatlas/viz/metrics_viz.py` Plotly functions are reused as-is, unmodified.

**Tech Stack:** Streamlit 1.57.0 (already installed, `st.navigation`/`st.Page` supported), pandas, joblib, DuckDB, scikit-learn Pipeline (already-fitted, loaded not trained), pytest.

## Global Constraints

- Python 3.11+; ruff line-length 100; no `print()` — use `logging` for load failures (repo convention).
- No `pyproject.toml` changes: `streamlit>=1.35` and all other needed libraries (`pandas`, `joblib`, `duckdb`, `scikit-learn`, `plotly`) are already base dependencies or resolve transitively (verified in `uv.lock`).
- English UI (per user decision).
- Native multipage navigation via `st.navigation`/`st.Page` (per user decision), not `st.tabs`.
- No SHAP anywhere; the Why-This-Prediction page must explicitly label its chart as global permutation importance, never a per-instance explanation.
- No live ROC/PR curve recomputation from `data/processed/c_phase_candidate_scores.parquet` (no `y_true` column; out of scope per design).
- No retraining, no new model artifacts, no changes to `src/unfallatlas/models/`, `src/unfallatlas/features/`, or any notebook.
- No Docker/cloud deployment — local `uv run streamlit run app/streamlit_app.py` only (already documented in `AGENTS.md`).
- Severity colors: KSI = `#E63946` (red), slight = `#2A9D8F` (green). Amber (`#F4A261`) is reserved but unused (binary model has only two output classes).
- Every artifact read by the app must already be committed and either plain-tracked or Git-LFS-tracked (verified for all files used: `data/processed/*`, `data/accidents.parquet`). Never write to any file under `data/`.
- `src/unfallatlas/viz/streamlit_app.py` contains zero `st.*` widget calls (loaders and pure functions only) so it stays importable and testable without a Streamlit runtime.

---

### Task 1: Core artifact loaders

**Files:**
- Create: `src/unfallatlas/viz/streamlit_app.py`
- Test: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `data/processed/c_phase_inference_contract.json`, `data/processed/a3_binary_model_card.json`, `data/processed/a3_binary_model_comparison.csv`, `data/processed/a3_model_comparison.csv`, `data/processed/c_phase_candidate_metrics.csv`, `data/processed/c_phase_permutation_importance.csv` (all committed, verified present via `git ls-files`).
- Produces: `load_inference_contract() -> dict`, `load_model_card() -> dict`, `load_binary_comparison() -> pd.DataFrame`, `load_3class_comparison() -> pd.DataFrame`, `load_candidate_metrics() -> pd.DataFrame`, `load_permutation_importance(model_name: str = "binary_random_forest_balanced", top_n: int = 15) -> pd.DataFrame` — all consumed by later tasks' pages.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_streamlit_app.py`:

```python
import pandas as pd

from unfallatlas.viz.streamlit_app import (
    load_3class_comparison,
    load_binary_comparison,
    load_candidate_metrics,
    load_inference_contract,
    load_model_card,
    load_permutation_importance,
)


def test_load_inference_contract_has_required_keys():
    contract = load_inference_contract()
    assert contract["model_path"] == "data/processed/a3_binary_best_model.joblib"
    assert contract["threshold"] == 0.49860217036273086
    assert len(contract["required_columns"]) == 30
    assert contract["required_columns"][0]["name"] == "UREGBEZ"


def test_load_model_card_has_test_2024_metrics():
    card = load_model_card()
    assert card["optimal_threshold_val_2023"] == 0.49860217036273086
    metrics = card["test_2024_metrics"]
    assert metrics["macro_f1"] == 0.6038956179272812
    assert metrics["confusion_matrix"] == [[22767, 21431], [51887, 172434]]


def test_load_binary_comparison_has_ten_candidates():
    df = load_binary_comparison()
    assert len(df) == 10
    assert {"model", "macro_f1", "recall_ksi"}.issubset(df.columns)


def test_load_3class_comparison_has_nineteen_configs():
    df = load_3class_comparison()
    assert len(df) == 19
    assert {"model", "macro_f1", "recall_class_1"}.issubset(df.columns)


def test_load_candidate_metrics_parses_confusion_matrix_as_list():
    df = load_candidate_metrics()
    assert len(df) == 10
    first_cm = df.loc[df["model"] == "binary_random_guess", "confusion_matrix"].iloc[0]
    assert first_cm == [[22976, 23053], [111714, 111305]]


def test_load_permutation_importance_returns_top_15_sorted_by_rank():
    df = load_permutation_importance()
    assert len(df) == 15
    assert (df["model"] == "binary_random_forest_balanced").all()
    assert df["rank"].is_monotonic_increasing


def test_load_permutation_importance_respects_top_n():
    df = load_permutation_importance(top_n=3)
    assert len(df) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.viz.streamlit_app'` (or `ImportError` since the file exists but is empty).

- [ ] **Step 3: Implement the loaders**

Create `src/unfallatlas/viz/streamlit_app.py`:

```python
"""Cached data loaders and pure helpers for the Phase K Streamlit app.

No Streamlit widget calls live here - only `st.cache_data`/`st.cache_resource`
decorated loaders and plain functions, so this module stays importable and
unit-testable without a Streamlit runtime. Widget code lives in app/pages/.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")


@st.cache_data
def load_inference_contract() -> dict:
    """Load the C-phase deployment contract: model path, threshold, required_columns schema."""
    with open(DATA_PROCESSED / "c_phase_inference_contract.json") as f:
        return json.load(f)


@st.cache_data
def load_model_card() -> dict:
    """Load the binary champion's model card (val/test metrics, confusion matrices)."""
    with open(DATA_PROCESSED / "a3_binary_model_card.json") as f:
        return json.load(f)


@st.cache_data
def load_binary_comparison() -> pd.DataFrame:
    """Load the 10-candidate binary-KSI comparison table."""
    return pd.read_csv(DATA_PROCESSED / "a3_binary_model_comparison.csv")


@st.cache_data
def load_3class_comparison() -> pd.DataFrame:
    """Load the 19-configuration 3-class comparison table (the pre-reframe ceiling evidence)."""
    return pd.read_csv(DATA_PROCESSED / "a3_model_comparison.csv")


@st.cache_data
def load_candidate_metrics() -> pd.DataFrame:
    """Load the C-phase candidate metrics table with confusion matrices parsed to lists."""
    df = pd.read_csv(DATA_PROCESSED / "c_phase_candidate_metrics.csv")
    df["confusion_matrix"] = df["confusion_matrix"].apply(ast.literal_eval)
    return df


@st.cache_data
def load_permutation_importance(
    model_name: str = "binary_random_forest_balanced", top_n: int = 15
) -> pd.DataFrame:
    """Load global permutation importance for one model, sorted by rank ascending."""
    df = pd.read_csv(DATA_PROCESSED / "c_phase_permutation_importance.csv")
    df = df[df["model"] == model_name].sort_values("rank").head(top_n)
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py`
Expected: no errors.

```bash
git add src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat(streamlit): add cached artifact loaders for Phase K app"
```

---

### Task 2: Categorical options, column spec, and shared UI constants

**Files:**
- Modify: `src/unfallatlas/viz/streamlit_app.py`
- Modify: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `load_inference_contract()` (Task 1), `data/accidents.parquet` (committed, LFS-tracked).
- Produces: `load_categorical_options(column: str) -> list[str]`, `get_column_spec(contract: dict, name: str) -> dict`, constants `DEFAULT_DWD_STATION_ID`, `DEFAULT_H3_CELL`, `DEFAULT_DWD_STATION_DIST_KM`, `SEVERITY_COLORS`, `LIMITATIONS_TEXT`, `WEEKDAY_LABELS`, `LICHTVERH_LABELS`, `STRZUSTAND_LABELS`, `DEFAULT_WIDGET_VALUES` — all consumed by Task 3 and the page tasks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streamlit_app.py`:

```python
from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
    get_column_spec,
    load_categorical_options,
)


def test_load_categorical_options_ukreis_has_87_sorted_values():
    options = load_categorical_options("UKREIS")
    assert len(options) == 87
    assert options[0] == "01"
    assert options == sorted(options)


def test_get_column_spec_returns_matching_entry():
    contract = load_inference_contract()
    spec = get_column_spec(contract, "UMONAT")
    assert spec["name"] == "UMONAT"
    assert spec["min"] == 1.0
    assert spec["max"] == 12.0


def test_get_column_spec_raises_on_unknown_column():
    contract = load_inference_contract()
    try:
        get_column_spec(contract, "NOT_A_COLUMN")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_default_widget_values_cover_every_required_column():
    contract = load_inference_contract()
    required_names = {col["name"] for col in contract["required_columns"]}
    assert required_names.issubset(DEFAULT_WIDGET_VALUES.keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "categorical_options or column_spec or default_widget" -v`
Expected: FAIL with `ImportError` (names not defined yet).

- [ ] **Step 3: Implement**

Add to `src/unfallatlas/viz/streamlit_app.py` (after the `DATA_PROCESSED` constant, before the loader functions):

```python
ACCIDENTS_PARQUET = Path("data/accidents.parquet")

DEFAULT_DWD_STATION_ID = "01975"
DEFAULT_H3_CELL = "881f15ad31fffff"
DEFAULT_DWD_STATION_DIST_KM = 9.51

SEVERITY_COLORS = {"KSI": "#E63946", "slight": "#2A9D8F"}

LIMITATIONS_TEXT = (
    "- The strongest available feature (accident type, `UART`) has a Cramer's V "
    "of only 0.18 against the target - even the best predictors have weak "
    "individual association with severity.\n"
    "- No demographic data (age, seatbelt use) or impact-speed data is in the "
    "Unfallatlas dataset, even though the literature identifies these as the "
    "strongest severity predictors.\n"
    "- Correlation is not causation: the model finds statistical association, "
    "not a causal mechanism.\n"
    "- The 'Why This Prediction' page shows global permutation importance, "
    "not a per-instance explanation - no SHAP was computed for this project."
)

WEEKDAY_LABELS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
LICHTVERH_LABELS = {0: "Daylight", 1: "Dusk/Dawn", 2: "Darkness"}
STRZUSTAND_LABELS = {0: "Dry", 1: "Wet/Slippery", 2: "Wintry"}

DEFAULT_WIDGET_VALUES = {
    "UREGBEZ": "1",
    "UKREIS": "01",
    "UMONAT": 6,
    "USTUNDE": 12,
    "UWOCHENTAG": 2,
    "UART": 0,
    "UTYP1": 1,
    "ULICHTVERH": 0,
    "STRZUSTAND": 0,
    "IstRad": False,
    "IstPKW": True,
    "IstFuss": False,
    "IstKrad": False,
    "IstGkfz": False,
    "IstSonstig": False,
    "LON": 9.67,
    "LAT": 50.85,
    "dwd_station_id": DEFAULT_DWD_STATION_ID,
    "dwd_station_dist_km": DEFAULT_DWD_STATION_DIST_KM,
    "dwd_temp_air_2m": 10.0,
    "dwd_precip_mm": 0.0,
    "dwd_visibility_m": 10000.0,
    "dwd_wind_speed_ms": 3.0,
    "_precip_bucket": "dry (0 mm)",
    "h3_cell": DEFAULT_H3_CELL,
    "osm_dominant_road_class": "residential",
    "osm_maxspeed_mean": 50.0,
    "osm_maxspeed_max": 50.0,
    "osm_road_density": 100.0,
    "osm_way_count": 50.0,
}
```

Add near the top of the file (with the other imports):

```python
import duckdb
```

Add these two functions after `load_inference_contract`:

```python
@st.cache_data
def load_categorical_options(column: str) -> list[str]:
    """Load sorted distinct values for a high-cardinality column with no fixed
    category list in the inference contract (currently only 'UKREIS').

    Reads data/accidents.parquet directly via DuckDB (a single-column
    columnar scan, not a full-dataset load); this file is committed and
    Git-LFS-tracked, so no notebook execution is required.
    """
    con = duckdb.connect()
    query = f"SELECT DISTINCT {column} FROM '{ACCIDENTS_PARQUET}' ORDER BY {column}"  # noqa: S608
    return con.execute(query).df()[column].astype(str).tolist()


def get_column_spec(contract: dict, name: str) -> dict:
    """Return the required_columns entry for one column name."""
    for col in contract["required_columns"]:
        if col["name"] == name:
            return col
    raise KeyError(f"Column {name!r} not found in inference contract required_columns")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: 11 passed.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py`

```bash
git add src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat(streamlit): add categorical options loader and shared UI constants"
```

---

### Task 3: Input-row assembly and prediction (with real-model integration test)

**Files:**
- Modify: `src/unfallatlas/viz/streamlit_app.py`
- Modify: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `load_inference_contract()`, `DEFAULT_WIDGET_VALUES` (Tasks 1-2), `data/processed/a3_binary_best_model.joblib` (committed, LFS-tracked).
- Produces: `load_champion_model() -> Pipeline`, `build_input_row(widget_values: dict, contract: dict) -> pd.DataFrame`, `predict_ksi(model, row: pd.DataFrame, threshold: float) -> tuple[float, int]` — consumed by the Risk Predictor page (Task 6).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streamlit_app.py`:

```python
import joblib

from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
    build_input_row,
    load_champion_model,
    predict_ksi,
)


class _FakePipeline:
    """Tiny stand-in for the real 407 MB joblib pipeline, for fast unit tests."""

    def predict_proba(self, row):
        return [[0.3, 0.7]]


def test_build_input_row_keeps_istgkfz_as_real_bool_like_its_siblings():
    """IstGkfz's contract dtype metadata ("object"/["False","True"]) looks like
    it wants a string, but the real committed model requires a raw bool
    (verified: string input raises ValueError inside the fitted pipeline)."""
    contract = load_inference_contract()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    assert row.loc[0, "IstGkfz"] is False


def test_build_input_row_keeps_real_bools_for_other_ist_columns():
    contract = load_inference_contract()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    assert row.loc[0, "IstPKW"] is True
    assert row.loc[0, "IstRad"] is False


def test_build_input_row_has_all_30_required_columns_in_contract_order():
    contract = load_inference_contract()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    expected_order = [col["name"] for col in contract["required_columns"]]
    assert list(row.columns) == expected_order


def test_build_input_row_raises_clear_keyerror_on_missing_widget_value():
    contract = load_inference_contract()
    incomplete_values = {k: v for k, v in DEFAULT_WIDGET_VALUES.items() if k != "UMONAT"}
    try:
        build_input_row(incomplete_values, contract)
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "UMONAT" in str(exc)


def test_predict_ksi_applies_threshold_not_default_half():
    row = build_input_row(DEFAULT_WIDGET_VALUES, load_inference_contract())
    proba, prediction = predict_ksi(_FakePipeline(), row, threshold=0.75)
    assert proba == 0.7
    assert prediction == 0  # 0.7 < 0.75 threshold, even though 0.7 > sklearn's default 0.5

    proba2, prediction2 = predict_ksi(_FakePipeline(), row, threshold=0.5)
    assert prediction2 == 1  # 0.7 >= 0.5


def test_load_champion_model_predicts_on_real_contract_row():
    """End-to-end check that the committed joblib model, the committed
    inference contract, and build_input_row/predict_ksi all agree - this is
    the concrete proof that the app needs no notebook execution."""
    contract = load_inference_contract()
    model = load_champion_model()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    proba, prediction = predict_ksi(model, row, contract["threshold"])
    assert 0.0 <= proba <= 1.0
    assert prediction in (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "build_input_row or predict_ksi or champion_model" -v`
Expected: FAIL with `ImportError` (`build_input_row`, `predict_ksi`, `load_champion_model` not defined).

- [ ] **Step 3: Implement**

Add near the top of `src/unfallatlas/viz/streamlit_app.py`, with the other imports:

```python
import joblib
from sklearn.pipeline import Pipeline
```

Add after `load_categorical_options`/`get_column_spec`:

```python
@st.cache_resource
def load_champion_model() -> Pipeline:
    """Load the fitted champion pipeline referenced by the inference contract.

    The returned Pipeline already includes preprocessing (encoding, scaling),
    so callers only need to pass a raw-feature row matching required_columns.
    """
    contract = load_inference_contract()
    return joblib.load(contract["model_path"])


def build_input_row(widget_values: dict, contract: dict) -> pd.DataFrame:
    """Assemble a one-row DataFrame matching the contract's required_columns.

    Note on IstGkfz: the inference contract's required_columns entry for
    IstGkfz declares dtype "object" with string categories ["False", "True"],
    unlike its five Ist* sibling columns (dtype "bool"). This looks like it
    should require a string cast, but verified empirically against the real
    committed data/processed/a3_binary_best_model.joblib, it does not: the
    fitted ColumnTransformer routes all six Ist* columns through the same
    'passthrough' group into RandomForestClassifier, which casts the whole
    array to float32 - a real bool converts cleanly (True/False -> 1.0/0.0),
    but the string "False" raises "ValueError: could not convert string to
    float: 'False'". The contract's recorded deployment_model_sha256 does not
    match the actual committed joblib's sha256 either, so the contract's
    dtype metadata for this column likely describes a different training
    run/metadata-extraction artifact than the model actually deployed here.
    IstGkfz is therefore treated identically to its five Ist* siblings: no
    cast, pass the raw bool through unchanged.

    dtype=object is passed to pd.DataFrame to preserve genuine Python bool
    identity for the Ist* columns (without it, pandas infers numpy bool_ for
    an all-bool column, and numpy.bool_(True) is True evaluates to False).
    """
    row = {}
    for col in contract["required_columns"]:
        name = col["name"]
        if name not in widget_values:
            raise KeyError(
                f"Missing value for required column {name!r} - the predictor form did not "
                "supply this input (contract/widget schema drift)."
            )
        row[name] = widget_values[name]
    ordered_columns = [col["name"] for col in contract["required_columns"]]
    return pd.DataFrame([row], dtype=object)[ordered_columns]


def predict_ksi(model: Pipeline, row: pd.DataFrame, threshold: float) -> tuple[float, int]:
    """Predict KSI probability and thresholded label for one input row.

    Uses the contract's tuned decision threshold (0.4986), not sklearn's
    default 0.5 - the champion was selected and evaluated at this threshold.
    """
    proba = float(model.predict_proba(row)[0][1])
    prediction = int(proba >= threshold)
    return proba, prediction
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: 17 passed. (`test_load_champion_model_predicts_on_real_contract_row` will take a few seconds to load the 407 MB joblib - this is expected and acceptable.)

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py`

```bash
git add src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat(streamlit): add input-row assembly and threshold-aware KSI prediction"
```

---

### Task 4: Streamlit entry point and native multipage navigation

**Files:**
- Modify: `app/streamlit_app.py`
- Create: `app/pages/` (directory, populated by Tasks 5-8)

**Interfaces:**
- Consumes: nothing (wiring only).
- Produces: the running app shell; Tasks 5-8 each add one file this task's `st.Page` list references.

- [ ] **Step 1: Create the pages directory and a placeholder so navigation has something to point at**

```bash
mkdir -p app/pages
```

- [ ] **Step 2: Write the entry point**

Replace the contents of `app/streamlit_app.py`:

```python
"""Phase K Streamlit entry point: page config and navigation wiring only.

All data loading, model inference, and plotting logic lives in
src/unfallatlas/viz/streamlit_app.py and the individual page modules under
app/pages/ - this file stays a thin wiring layer.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Unfallatlas KSI Risk Console",
    layout="wide",
    page_icon="\U0001f6a6",
)

PAGES_DIR = Path(__file__).parent / "pages"

pages = [
    st.Page(str(PAGES_DIR / "overview.py"), title="Overview", icon=":material/bar_chart:"),
    st.Page(str(PAGES_DIR / "risk_predictor.py"), title="Risk Predictor", icon=":material/query_stats:"),
    st.Page(
        str(PAGES_DIR / "why_this_prediction.py"),
        title="Why This Prediction",
        icon=":material/search_insights:",
    ),
    st.Page(str(PAGES_DIR / "model_comparison.py"), title="Model Comparison", icon=":material/balance:"),
]

nav = st.navigation(pages)
nav.run()
```

- [ ] **Step 3: Add a temporary placeholder page so the app can be smoke-tested before Tasks 5-8 land**

Create `app/pages/overview.py` (will be replaced with real content in Task 5):

```python
import streamlit as st

st.title("Unfallatlas KSI Risk Console")
st.info("Overview page under construction.")
```

Create identical placeholders `app/pages/risk_predictor.py`, `app/pages/why_this_prediction.py`, `app/pages/model_comparison.py` with the same two lines (titles adjusted to match).

- [ ] **Step 4: Manual smoke test**

Run: `uv run streamlit run app/streamlit_app.py --server.headless true &`
then `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501`
Expected: `200`. Stop the server afterward: `kill %1`.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py app/pages/
git commit -m "feat(streamlit): wire native multipage navigation with placeholder pages"
```

---

### Task 5: Overview page

**Files:**
- Modify: `app/pages/overview.py`

**Interfaces:**
- Consumes: `load_model_card()`, `load_3class_comparison()`, `load_binary_comparison()`, `LIMITATIONS_TEXT` (Tasks 1-2); `plot_f1_recall_front`, `plot_binary_f1_recall_front` from `src/unfallatlas/viz/metrics_viz.py` (existing, unmodified).
- Produces: nothing consumed by later tasks (leaf page).

- [ ] **Step 1: Replace the placeholder**

Replace `app/pages/overview.py`:

```python
"""Overview page: champion headline metrics and the 3-class vs. binary ceiling story."""

import streamlit as st

from unfallatlas.viz.metrics_viz import plot_binary_f1_recall_front, plot_f1_recall_front
from unfallatlas.viz.streamlit_app import (
    LIMITATIONS_TEXT,
    load_3class_comparison,
    load_binary_comparison,
    load_model_card,
)

st.title("Unfallatlas KSI Risk Console")
st.caption("Binary KSI (killed or seriously injured) vs. slight-injury severity classification")

card = load_model_card()
test_metrics = card["test_2024_metrics"]

col1, col2, col3 = st.columns(3)
col1.metric("Macro-F1 (test 2024)", f"{test_metrics['macro_f1']:.3f}")
col2.metric("Recall (KSI)", f"{test_metrics['recall_ksi']:.3f}")
col3.metric("Decision threshold", f"{card['optimal_threshold_val_2023']:.4f}")

st.markdown("---")
st.subheader("Why the target was reframed: the 3-class ceiling vs. the binary reframe")
st.markdown(
    "The original 3-class target (killed / seriously injured / slightly injured) has an "
    "empirical ceiling of macro-F1 = 0.424 across 19 configurations, well below the 0.55 "
    "acceptance gate. Reframing as binary KSI (killed-or-seriously-injured vs. slight) "
    "clears both acceptance gates on the held-out 2024 test set."
)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_f1_recall_front(load_3class_comparison()), use_container_width=True)
with col_b:
    st.plotly_chart(plot_binary_f1_recall_front(load_binary_comparison()), use_container_width=True)

with st.expander("Limitations"):
    st.markdown(LIMITATIONS_TEXT)
```

- [ ] **Step 2: Manual verification**

Run: `uv run streamlit run app/streamlit_app.py`
Open the Overview page in the browser. Confirm: three metrics render with real numbers (macro-F1 ≈ 0.604, recall-KSI ≈ 0.515, threshold ≈ 0.4986), both Pareto-front charts render and are interactive (hover tooltips work), and the Limitations expander opens with four bullet points.

- [ ] **Step 3: Commit**

```bash
git add app/pages/overview.py
git commit -m "feat(streamlit): implement Overview page with ceiling-vs-reframe narrative"
```

---

### Task 6: Risk Predictor page

**Files:**
- Modify: `app/pages/risk_predictor.py`

**Interfaces:**
- Consumes: `load_inference_contract()`, `load_categorical_options()`, `get_column_spec()`, `load_champion_model()`, `build_input_row()`, `predict_ksi()`, `DEFAULT_WIDGET_VALUES`, `SEVERITY_COLORS`, `LIMITATIONS_TEXT`, `WEEKDAY_LABELS`, `LICHTVERH_LABELS`, `STRZUSTAND_LABELS` (Tasks 1-3).
- Produces: `st.session_state["last_prediction"] = {"inputs": dict, "proba": float, "prediction": int}` - consumed by Task 7 (Why This Prediction).

- [ ] **Step 1: Replace the placeholder**

Replace `app/pages/risk_predictor.py`:

```python
"""Risk Predictor page: interactive KSI-risk prediction form."""

import streamlit as st

from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
    LICHTVERH_LABELS,
    LIMITATIONS_TEXT,
    SEVERITY_COLORS,
    STRZUSTAND_LABELS,
    WEEKDAY_LABELS,
    build_input_row,
    get_column_spec,
    load_categorical_options,
    load_champion_model,
    load_inference_contract,
    predict_ksi,
)

st.title("Risk Predictor")
st.caption(
    "Estimate the probability that a described accident is KSI "
    "(killed or seriously injured) vs. slight."
)

contract = load_inference_contract()
defaults = DEFAULT_WIDGET_VALUES
ukreis_options = load_categorical_options("UKREIS")

with st.form("risk_predictor_form"):
    st.subheader("When and where")
    c1, c2, c3 = st.columns(3)
    with c1:
        uregbez_categories = get_column_spec(contract, "UREGBEZ")["categories"]
        uregbez = st.selectbox(
            "Regierungsbezirk (UREGBEZ)",
            options=uregbez_categories,
            index=uregbez_categories.index(defaults["UREGBEZ"]),
        )
        ukreis = st.selectbox(
            "Kreis (UKREIS)", options=ukreis_options, index=ukreis_options.index(defaults["UKREIS"])
        )
    with c2:
        umonat_spec = get_column_spec(contract, "UMONAT")
        umonat = st.slider(
            "Month (UMONAT)",
            min_value=int(umonat_spec["min"]),
            max_value=int(umonat_spec["max"]),
            value=defaults["UMONAT"],
        )
        ustunde_spec = get_column_spec(contract, "USTUNDE")
        ustunde = st.slider(
            "Hour (USTUNDE)",
            min_value=int(ustunde_spec["min"]),
            max_value=int(ustunde_spec["max"]),
            value=defaults["USTUNDE"],
        )
    with c3:
        uwochentag_label = st.selectbox(
            "Weekday (UWOCHENTAG)",
            options=list(WEEKDAY_LABELS.values()),
            index=defaults["UWOCHENTAG"] - 1,
        )
        uwochentag = {v: k for k, v in WEEKDAY_LABELS.items()}[uwochentag_label]

    st.subheader("Accident characteristics")
    c4, c5, c6 = st.columns(3)
    with c4:
        uart_spec = get_column_spec(contract, "UART")
        uart_options = list(range(int(uart_spec["min"]), int(uart_spec["max"]) + 1))
        uart = st.selectbox(
            "Accident type (UART)", options=uart_options, index=uart_options.index(defaults["UART"])
        )
        utyp1_spec = get_column_spec(contract, "UTYP1")
        utyp1_options = list(range(int(utyp1_spec["min"]), int(utyp1_spec["max"]) + 1))
        utyp1 = st.selectbox(
            "Accident category (UTYP1)", options=utyp1_options, index=utyp1_options.index(defaults["UTYP1"])
        )
    with c5:
        ulichtverh_label = st.selectbox(
            "Light conditions (ULICHTVERH)",
            options=list(LICHTVERH_LABELS.values()),
            index=defaults["ULICHTVERH"],
        )
        ulichtverh = {v: k for k, v in LICHTVERH_LABELS.items()}[ulichtverh_label]
        strzustand_label = st.selectbox(
            "Road condition (STRZUSTAND)",
            options=list(STRZUSTAND_LABELS.values()),
            index=defaults["STRZUSTAND"],
        )
        strzustand = {v: k for k, v in STRZUSTAND_LABELS.items()}[strzustand_label]
    with c6:
        ist_rad = st.checkbox("Cyclist involved (IstRad)", value=defaults["IstRad"])
        ist_pkw = st.checkbox("Car involved (IstPKW)", value=defaults["IstPKW"])
        ist_fuss = st.checkbox("Pedestrian involved (IstFuss)", value=defaults["IstFuss"])
        ist_krad = st.checkbox("Motorcycle involved (IstKrad)", value=defaults["IstKrad"])
        ist_gkfz = st.checkbox("Heavy goods vehicle involved (IstGkfz)", value=defaults["IstGkfz"])
        ist_sonstig = st.checkbox("Other vehicle involved (IstSonstig)", value=defaults["IstSonstig"])

    st.subheader("Location")
    c7, c8 = st.columns(2)
    with c7:
        lon_spec = get_column_spec(contract, "LON")
        lon = st.number_input(
            "Longitude (LON)",
            min_value=float(lon_spec["min"]),
            max_value=float(lon_spec["max"]),
            value=defaults["LON"],
        )
    with c8:
        lat_spec = get_column_spec(contract, "LAT")
        lat = st.number_input(
            "Latitude (LAT)",
            min_value=float(lat_spec["min"]),
            max_value=float(lat_spec["max"]),
            value=defaults["LAT"],
        )

    st.subheader("Weather")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        temp_spec = get_column_spec(contract, "dwd_temp_air_2m")
        dwd_temp_air_2m = st.slider(
            "Air temperature (C)",
            min_value=float(temp_spec["min"]),
            max_value=float(temp_spec["max"]),
            value=defaults["dwd_temp_air_2m"],
        )
    with c10:
        precip_spec = get_column_spec(contract, "dwd_precip_mm")
        dwd_precip_mm = st.slider(
            "Precipitation (mm)",
            min_value=float(precip_spec["min"]),
            max_value=float(precip_spec["max"]),
            value=defaults["dwd_precip_mm"],
        )
    with c11:
        vis_spec = get_column_spec(contract, "dwd_visibility_m")
        dwd_visibility_m = st.slider(
            "Visibility (m)",
            min_value=float(vis_spec["min"]),
            max_value=float(vis_spec["max"]),
            value=defaults["dwd_visibility_m"],
        )
    with c12:
        wind_spec = get_column_spec(contract, "dwd_wind_speed_ms")
        dwd_wind_speed_ms = st.slider(
            "Wind speed (m/s)",
            min_value=float(wind_spec["min"]),
            max_value=float(wind_spec["max"]),
            value=defaults["dwd_wind_speed_ms"],
        )
    precip_bucket_categories = get_column_spec(contract, "_precip_bucket")["categories"]
    precip_bucket = st.selectbox(
        "Precipitation bucket",
        options=precip_bucket_categories,
        index=precip_bucket_categories.index(defaults["_precip_bucket"]),
    )

    st.subheader("Road context (OpenStreetMap)")
    c13, c14, c15, c16 = st.columns(4)
    with c13:
        road_class_categories = get_column_spec(contract, "osm_dominant_road_class")["categories"]
        osm_dominant_road_class = st.selectbox(
            "Dominant road class",
            options=road_class_categories,
            index=road_class_categories.index(defaults["osm_dominant_road_class"]),
        )
    with c14:
        maxspeed_mean_spec = get_column_spec(contract, "osm_maxspeed_mean")
        osm_maxspeed_mean = st.slider(
            "Mean speed limit (km/h)",
            min_value=float(maxspeed_mean_spec["min"]),
            max_value=float(maxspeed_mean_spec["max"]),
            value=defaults["osm_maxspeed_mean"],
        )
    with c15:
        maxspeed_max_spec = get_column_spec(contract, "osm_maxspeed_max")
        osm_maxspeed_max = st.slider(
            "Max speed limit (km/h)",
            min_value=float(maxspeed_max_spec["min"]),
            max_value=float(maxspeed_max_spec["max"]),
            value=defaults["osm_maxspeed_max"],
        )
    with c16:
        density_spec = get_column_spec(contract, "osm_road_density")
        osm_road_density = st.slider(
            "Road density (H3 cell)",
            min_value=float(density_spec["min"]),
            max_value=float(density_spec["max"]),
            value=defaults["osm_road_density"],
        )
        way_count_spec = get_column_spec(contract, "osm_way_count")
        osm_way_count = st.slider(
            "Road way count (H3 cell)",
            min_value=float(way_count_spec["min"]),
            max_value=float(way_count_spec["max"]),
            value=defaults["osm_way_count"],
        )

    submitted = st.form_submit_button("Predict KSI risk")

if submitted:
    widget_values = {
        "UREGBEZ": uregbez,
        "UKREIS": ukreis,
        "UMONAT": umonat,
        "USTUNDE": ustunde,
        "UWOCHENTAG": uwochentag,
        "UART": uart,
        "UTYP1": utyp1,
        "ULICHTVERH": ulichtverh,
        "STRZUSTAND": strzustand,
        "IstRad": ist_rad,
        "IstPKW": ist_pkw,
        "IstFuss": ist_fuss,
        "IstKrad": ist_krad,
        "IstGkfz": ist_gkfz,
        "IstSonstig": ist_sonstig,
        "LON": lon,
        "LAT": lat,
        "dwd_station_id": defaults["dwd_station_id"],
        "dwd_station_dist_km": defaults["dwd_station_dist_km"],
        "dwd_temp_air_2m": dwd_temp_air_2m,
        "dwd_precip_mm": dwd_precip_mm,
        "dwd_visibility_m": dwd_visibility_m,
        "dwd_wind_speed_ms": dwd_wind_speed_ms,
        "_precip_bucket": precip_bucket,
        "h3_cell": defaults["h3_cell"],
        "osm_dominant_road_class": osm_dominant_road_class,
        "osm_maxspeed_mean": osm_maxspeed_mean,
        "osm_maxspeed_max": osm_maxspeed_max,
        "osm_road_density": osm_road_density,
        "osm_way_count": osm_way_count,
    }

    try:
        model = load_champion_model()
        row = build_input_row(widget_values, contract)
        proba, prediction = predict_ksi(model, row, contract["threshold"])
    except KeyError as exc:
        st.error(f"Could not build the model input row: {exc}")
        st.stop()

    st.session_state["last_prediction"] = {
        "inputs": widget_values,
        "proba": proba,
        "prediction": prediction,
    }

    label = "KSI (killed or seriously injured)" if prediction == 1 else "Slight injury"
    color = SEVERITY_COLORS["KSI"] if prediction == 1 else SEVERITY_COLORS["slight"]
    st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    st.metric(
        "KSI probability", f"{proba:.1%}", help=f"Decision threshold: {contract['threshold']:.1%}"
    )

with st.expander("Limitations"):
    st.markdown(LIMITATIONS_TEXT)
```

- [ ] **Step 2: Manual verification**

Run: `uv run streamlit run app/streamlit_app.py`, open Risk Predictor. Submit the form with defaults unchanged. Confirm: a colored prediction result appears (green "Slight injury" or red "KSI" - both are valid outcomes depending on model behavior on the defaults), the probability metric shows a percentage with the threshold in its tooltip, and no exception is raised. Then change several categorical widgets (weekday, light conditions, road class, `IstGkfz` checkbox) and resubmit - confirm no exception (this specifically exercises the `IstGkfz` checkbox, which build_input_row passes through as a raw bool, same as its Ist* siblings).

- [ ] **Step 3: Commit**

```bash
git add app/pages/risk_predictor.py
git commit -m "feat(streamlit): implement Risk Predictor page with full input form"
```

---

### Task 7: Why This Prediction page

**Files:**
- Modify: `app/pages/why_this_prediction.py`

**Interfaces:**
- Consumes: `st.session_state["last_prediction"]` (set by Task 6), `load_permutation_importance()` (Task 1).
- Produces: nothing consumed by later tasks (leaf page).

- [ ] **Step 1: Replace the placeholder**

Replace `app/pages/why_this_prediction.py`:

```python
"""Why This Prediction page: global permutation importance + user-input context."""

import pandas as pd
import streamlit as st

from unfallatlas.viz.streamlit_app import load_permutation_importance

st.title("Why This Prediction")

last_prediction = st.session_state.get("last_prediction")
if last_prediction is None:
    st.info("No prediction yet. Go to the Risk Predictor page and submit a prediction first.")
    st.stop()

st.warning(
    "This shows global, model-level permutation importance from the C-phase analysis, "
    "not a per-instance SHAP explanation. No SHAP was computed for this project."
)

importance_df = load_permutation_importance()
st.bar_chart(importance_df.set_index("feature")["importance_mean"])

st.subheader("Your inputs for the globally most influential features")
st.caption(
    "These are the values you submitted for the model's top globally-important "
    "features. This is context, not a causal explanation of this specific prediction."
)
top_features = importance_df["feature"].tolist()
user_values = {
    feature: last_prediction["inputs"][feature]
    for feature in top_features
    if feature in last_prediction["inputs"]
}
st.table(pd.DataFrame({"feature": list(user_values.keys()), "your value": list(user_values.values())}))
```

- [ ] **Step 2: Manual verification**

Run: `uv run streamlit run app/streamlit_app.py`. Navigate directly to "Why This Prediction" without having submitted a prediction first - confirm the empty-state info message appears and no exception is raised. Then go to Risk Predictor, submit a prediction, return to "Why This Prediction" - confirm the bar chart and the "your inputs" table both render, and the global-importance warning is visible above the fold.

- [ ] **Step 3: Commit**

```bash
git add app/pages/why_this_prediction.py
git commit -m "feat(streamlit): implement Why This Prediction page with honest global-importance framing"
```

---

### Task 8: Model Comparison page

**Files:**
- Modify: `app/pages/model_comparison.py`

**Interfaces:**
- Consumes: `load_candidate_metrics()`, `load_binary_comparison()`, `load_model_card()`, `load_inference_contract()` (Task 1); `plot_binary_f1_recall_front`, `plot_confusion_matrix_heatmap` from `src/unfallatlas/viz/metrics_viz.py` (existing, unmodified).
- Produces: nothing consumed by later tasks (leaf page).

- [ ] **Step 1: Replace the placeholder**

Replace `app/pages/model_comparison.py`:

```python
"""Model Comparison page: candidate table, Pareto front, confusion matrix, robustness."""

import pandas as pd
import streamlit as st

from unfallatlas.viz.metrics_viz import plot_binary_f1_recall_front, plot_confusion_matrix_heatmap
from unfallatlas.viz.streamlit_app import (
    load_binary_comparison,
    load_candidate_metrics,
    load_inference_contract,
    load_model_card,
)

st.title("Model Comparison")

st.subheader("All 10 candidates (binary KSI)")
candidate_df = load_candidate_metrics()
st.dataframe(
    candidate_df[
        ["model", "family", "evaluation_role", "macro_f1", "recall_ksi", "recall_slight", "latency_ms_per_1k"]
    ],
    use_container_width=True,
)

st.subheader("Pareto front: macro-F1 vs. Recall(KSI)")
st.plotly_chart(plot_binary_f1_recall_front(load_binary_comparison()), use_container_width=True)

st.subheader("Champion confusion matrix (test 2024)")
card = load_model_card()
confusion_matrix = card["test_2024_metrics"]["confusion_matrix"]
st.plotly_chart(
    plot_confusion_matrix_heatmap(confusion_matrix, labels=["KSI", "slight"]),
    use_container_width=True,
)

st.subheader("Finalist comparison: macro-F1, latency, robustness")
contract = load_inference_contract()
finalists_df = pd.DataFrame(contract["decision_evidence"]["finalist_measurements"])
st.dataframe(
    finalists_df[
        ["model", "macro_f1", "recall_ksi", "latency_ms_per_1k", "robustness_score", "robustness_status"]
    ],
    use_container_width=True,
)

st.info(contract["decision_evidence"]["preference_conclusion"]["statement"])
```

- [ ] **Step 2: Manual verification**

Run: `uv run streamlit run app/streamlit_app.py`, open Model Comparison. Confirm: the 10-row candidate table renders and is sortable by column header click, the Pareto-front chart renders interactively, the confusion-matrix heatmap shows the four cells with real counts (22767/21431/51887/172434), the 4-row finalist table renders, and the honest preference-conclusion statement (mentioning XGBoost leading measured criteria while Random Forest remains deployment champion) is visible.

- [ ] **Step 3: Commit**

```bash
git add app/pages/model_comparison.py
git commit -m "feat(streamlit): implement Model Comparison page"
```

---

### Task 9: End-to-end fresh-environment verification

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Confirm every artifact the app reads is actually committed (not just present locally)**

Run:
```bash
git ls-files data/processed/a3_binary_best_model.joblib data/processed/a3_binary_model_card.json \
  data/processed/a3_binary_model_comparison.csv data/processed/a3_model_comparison.csv \
  data/processed/c_phase_candidate_metrics.csv data/processed/c_phase_permutation_importance.csv \
  data/processed/c_phase_inference_contract.json data/accidents.parquet app/streamlit_app.py \
  app/pages/overview.py app/pages/risk_predictor.py app/pages/why_this_prediction.py \
  app/pages/model_comparison.py src/unfallatlas/viz/streamlit_app.py
```
Expected: every path listed back (no missing paths, no error).

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass, including the new `tests/test_streamlit_app.py` tests, with no regressions in existing suites.

- [ ] **Step 3: Lint the whole app**

Run: `uv run ruff check app/ src/unfallatlas/viz/streamlit_app.py`
Expected: no errors.

- [ ] **Step 4: Full manual walkthrough**

Run: `uv run streamlit run app/streamlit_app.py`. Walk all four pages in order (Overview -> Risk Predictor -> Why This Prediction -> Model Comparison). On Risk Predictor, submit at least three predictions covering different combinations of: `IstGkfz` checked and unchecked, at least two different `osm_dominant_road_class` values, at least two different `UWOCHENTAG` values, and a `UKREIS` value other than the default "01". Confirm no exception is ever raised and each submission updates the Why-This-Prediction page's "your inputs" table.

- [ ] **Step 5: Confirm this closes the user's fresh-clone requirement**

No code changes in this step - this is a documentation checkpoint. Everything the app reads (verified Step 1) is committed and Git-LFS-tracked, so `git clone` + `git lfs pull` + `uv sync` + `uv run streamlit run app/streamlit_app.py` is sufficient with no notebook execution, matching the user's explicit requirement.

---

### Task 10: Fix stale AGENTS.md auto-managed sections

**Files:**
- Modify: `AGENTS.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Fix the stale Overview line**

In `AGENTS.md`, inside the `<!-- AUTO-MANAGED: project-description -->` block, replace:

```
- Phase status: Q/U/A³ notebooks done; `notebooks/04_C_Phase.ipynb` (C phase) and `app/streamlit_app.py` / `src/unfallatlas/viz/streamlit_app.py` (K phase) are still empty stubs — SHAP/comparison and the Streamlit demo are not yet implemented
```

with:

```
- Phase status: Q/U/A³/C notebooks all done (`notebooks/04_C_Phase.ipynb` implements the full binary-champion comparison, permutation importance, and inference-contract handoff — not SHAP, which was dropped in favor of permutation importance); the Phase K Streamlit app (`app/streamlit_app.py`, `src/unfallatlas/viz/streamlit_app.py`) is implemented and runs fully from committed `data/processed/` artifacts with no notebook execution required
```

- [ ] **Step 2: Fix the stale Architecture section's C-phase and app/ lines**

In the `<!-- AUTO-MANAGED: architecture -->` block, replace:

```
│   └── 04_C_Phase.ipynb    # Comparison, SHAP, conclusions (TODO — empty stub)
```

with:

```
│   └── 04_C_Phase.ipynb    # Comparison, permutation importance, inference-contract handoff (done)
```

And replace:

```
│   ├── viz/                # geo.py (stub, empty), shap_plots.py (stub, empty), streamlit_app.py (stub, empty), metrics_viz.py
```

with:

```
│   ├── viz/                # geo.py (stub, empty), shap_plots.py (stub, empty — permutation importance used instead), streamlit_app.py (Phase K app loaders/helpers, done), metrics_viz.py
```

And replace:

```
├── app/                    # Streamlit demo entry point (stub, empty — K phase not implemented)
```

with:

```
├── app/                    # Streamlit demo: entry point + pages/ (Overview, Risk Predictor, Why This Prediction, Model Comparison) — implemented, K phase done
```

And replace the launch-command comment in the build-commands block:

```
# Launch Streamlit demo (K phase — app/streamlit_app.py is currently an empty
# stub, so this command does not yet produce a working app)
uv run streamlit run app/streamlit_app.py
```

with:

```
# Launch the Phase K Streamlit demo (Overview, Risk Predictor, Why This
# Prediction, Model Comparison — runs fully from committed data/processed/
# artifacts, no notebook execution required)
uv run streamlit run app/streamlit_app.py
```

- [ ] **Step 3: Verify no other stale references remain**

Run: `grep -n "empty stub\|not yet implemented\|TODO" AGENTS.md`
Expected: no remaining hits referring to the C-phase or the Streamlit app (other unrelated stubs like `geo.py`/`shap_plots.py`/`enrich.py`/`download.py` are still genuinely empty and may still be described as stubs - only the Streamlit-app and C-phase lines needed fixing).

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): fix stale C-phase and Streamlit-app status in AGENTS.md"
```

---

### Task 11: Update AI tool disclosure and Phase K prompt log

**Files:**
- Modify: `docs/AI TOOL DISCLOSURE.md`
- Modify: `docs/prompts/04_prompts_phase_k.md` (create if it doesn't already exist, matching the naming/format of `docs/prompts/01_prompts_phase_q.md` etc.)

**Interfaces:** none (documentation only). This task runs last, after Task 9's manual verification confirms the app works end-to-end.

- [ ] **Step 1: Check the existing format of a sibling prompts file**

Run: `cat docs/prompts/03_prompts_phase_a3.md | head -40` (or the closest existing phase file) to confirm the exact heading structure and metadata-bullet style already in use, so the new file matches it exactly rather than inventing a new format.

- [ ] **Step 2: Create `docs/prompts/04_prompts_phase_k.md`**

Follow the exact section structure found in Step 1 (typically: a top-level heading naming the phase, then one subsection per major prompt/response exchange with a `**Kontext**:` metadata bullet, then a fenced quote block of the verbatim prompt). Populate it with:
- The brainstorming exchange that produced the concept selection (Concept A: Risk Explainer & Model Console) and the four follow-up clarifying decisions (English UI, native multipage navigation, ceiling story included, local-only deployment).
- The design-doc creation prompt and a summary of what `docs/superpowers/specs/2026-07-27-phase-k-streamlit-app-design.md` locked in.
- The implementation-plan creation prompt and a summary of the 11 tasks in `docs/superpowers/plans/2026-07-27-phase-k-streamlit-app.md`.
- Note explicitly: no SHAP was computed in this project (a decision made in the C-phase, reaffirmed here); the Streamlit app's "Why This Prediction" page uses permutation importance instead, and this is disclosed to the end user directly in the app UI.

- [ ] **Step 3: Update `docs/AI TOOL DISCLOSURE.md`**

Read the file first (`cat "docs/AI TOOL DISCLOSURE.md"`) to find its existing per-phase structure, then add a Phase K entry following the same structure as the existing Q/U/A³/C entries, covering: which AI tool/model was used, that the Streamlit app design was produced via a structured brainstorming-then-planning workflow (concept selection among 5 proposed options, followed by a written design spec and a task-by-task implementation plan), what was human-verified (the manual walkthrough in Task 9 - every page, multiple predictions, confirmed no notebook execution was needed), and the explicit SHAP-to-permutation-importance substitution disclosed above.

- [ ] **Step 4: Commit**

```bash
git add "docs/AI TOOL DISCLOSURE.md" docs/prompts/04_prompts_phase_k.md
git commit -m "docs: disclose AI-assisted Phase K Streamlit app design and implementation"
```
