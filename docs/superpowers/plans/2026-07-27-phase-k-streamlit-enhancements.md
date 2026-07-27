# Phase K Streamlit App Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the already-shipped Phase K Streamlit app's usability (readable category labels, an interactive map for picking a location, an interactive severity map on the Overview page, readable feature names on the Why This Prediction page, an inline German explanation of the Random-Forest-vs-XGBoost finalist decision) and prepare it for a live deployment on Streamlit Community Cloud with cross-links to the existing GitHub Pages notebook exports.

**Architecture:** All new logic lives in `src/unfallatlas/viz/streamlit_app.py` (pure, testable helpers — no widget calls), consumed by `app/pages/*.py` (widget code only), matching the existing Phase K module split. The interactive maps use `folium` + `streamlit-folium`, which are already base dependencies (not optional extras) — no new dependencies are introduced anywhere in this plan. All accidents.parquet queries go through DuckDB, never a full pandas load.

**Tech Stack:** Streamlit 1.35+, folium, streamlit-folium, DuckDB, pandas, pytest, `streamlit.testing.v1.AppTest`.

## Global Constraints

- Fresh clone + `git lfs pull` + `uv sync` (no extras) + `uv run streamlit run app/streamlit_app.py` must fully run the app with every feature in this plan working, zero notebook execution, zero manual steps beyond those two commands.
- No new dependencies: `folium` and `streamlit-folium` are already in `pyproject.toml`'s base `dependencies` list (not `optional-dependencies.geo`, which holds `geopandas`/`h3`/`osmnx` — those are NOT available after a plain `uv sync` and must not be imported by app code).
- No `Co-Authored-By` trailer in any commit (binding instruction for this session).
- Work happens directly on `main` (user has explicitly consented to this for Phase K work this session).
- Use `uv` exclusively: `uv sync`, `uv run pytest`, `uv run streamlit run ...`, `uv run ruff check ...`.
- All DuckDB queries against `data/accidents.parquet` (2,092,401 rows) must aggregate/filter in SQL — never `SELECT *` into pandas.
- All 18 existing tests in `tests/test_streamlit_app.py` must keep passing; new tests follow the same plain-function-call pattern (most new logic is testable without `AppTest` at all) or the `AppTest` pattern for widget-level behavior.
- KSI severity definition (verified against `notebooks/01_Q_Phase.py` lines 156-157): `UKATGEORIE IN (1, 2)` = KSI, `UKATGEORIE = 3` = slight. `data/accidents.parquet` has no invalid `UKATGEORIE` values (confirmed via `notebooks/02_U_Phase.py`'s data-quality check), so `ksi_count + slight_count == total` always holds for any grouping of this column.
- `UREGBEZ`/`UKREIS` cannot be decoded to real German administrative names: the inference contract's `required_columns` (`data/processed/c_phase_inference_contract.json`) does not include `ULAND` (Bundesland), and `docs/dataset/DSB_Unfallatlas.md` lines 8-11 confirm the official Gemeindeschlüssel requires `ULAND` + `UREGBEZ` + `UKREIS` together. There is no offline lookup table in this repo and no permitted new dependency to build one. This plan's honest fix is a clarifying caption, not invented names — do not add a fake name table.

---

### Task 1: Display-name and label helpers, and the severity-grid loader

**Files:**
- Modify: `src/unfallatlas/viz/streamlit_app.py`
- Test: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: nothing new (uses existing `DATA_PROCESSED`, `ACCIDENTS_PARQUET`, `UART_LABELS`, `UTYP1_LABELS`, `STRZUSTAND_LABELS`, `LICHTVERH_LABELS`, `WEEKDAY_LABELS`, `SEVERITY_COLORS`).
- Produces: `FEATURE_DISPLAY_NAMES: dict[str, str]`, `display_feature_name(name: str) -> str`, `COLUMN_LABEL_MAPS: dict[str, dict]`, `decode_feature_value(feature: str, value) -> object`, `load_severity_grid(precision: float = 0.1) -> pd.DataFrame` (columns `lat_bin`, `lon_bin`, `ksi_count`, `slight_count`, `total`) — all consumed by Tasks 3, 4, 5.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streamlit_app.py`:

```python
from unfallatlas.viz.streamlit_app import (
    FEATURE_DISPLAY_NAMES,
    UART_LABELS,
    COLUMN_LABEL_MAPS,
    decode_feature_value,
    display_feature_name,
    load_severity_grid,
)


def test_display_feature_name_maps_known_column():
    assert display_feature_name("dwd_wind_speed_ms") == "Wind Speed (m/s)"


def test_display_feature_name_falls_back_to_raw_name_for_unknown_column():
    assert display_feature_name("some_future_column") == "some_future_column"


def test_feature_display_names_covers_every_permutation_importance_feature():
    importance_df = pd.read_csv("data/processed/c_phase_permutation_importance.csv")
    all_features = set(importance_df["feature"].unique())
    assert all_features.issubset(FEATURE_DISPLAY_NAMES.keys())


def test_decode_feature_value_maps_uart_code_to_label():
    assert decode_feature_value("UART", 6) == UART_LABELS[6]


def test_decode_feature_value_passes_through_unmapped_column():
    assert decode_feature_value("dwd_wind_speed_ms", 3.0) == 3.0


def test_load_severity_grid_has_expected_columns_and_counts_agree():
    df = load_severity_grid()
    assert {"lat_bin", "lon_bin", "ksi_count", "slight_count", "total"}.issubset(df.columns)
    assert len(df) > 0
    assert (df["ksi_count"] + df["slight_count"] == df["total"]).all()


def test_load_severity_grid_respects_precision_parameter():
    coarse = load_severity_grid(precision=0.5)
    fine = load_severity_grid(precision=0.1)
    assert len(coarse) < len(fine)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "display_feature_name or decode_feature_value or severity_grid" -v`
Expected: FAIL with `ImportError` (names not defined yet).

- [ ] **Step 3: Implement in `src/unfallatlas/viz/streamlit_app.py`**

Add after the existing `UTYP1_LABELS` dict (after line 76, before `DEFAULT_WIDGET_VALUES`):

```python
FEATURE_DISPLAY_NAMES = {
    "IstFuss": "Pedestrian Involved",
    "IstGkfz": "Heavy Goods Vehicle Involved",
    "IstKrad": "Motorcycle Involved",
    "IstPKW": "Car Involved",
    "IstRad": "Cyclist Involved",
    "IstSonstig": "Other Vehicle Involved",
    "LAT": "Latitude",
    "LON": "Longitude",
    "STRZUSTAND": "Road Condition",
    "UART": "Accident Type",
    "UKREIS": "Kreis Code",
    "ULICHTVERH": "Light Conditions",
    "UMONAT": "Month",
    "UREGBEZ": "Regierungsbezirk Code",
    "USTUNDE": "Hour of Day",
    "UTYP1": "Accident Category",
    "UWOCHENTAG": "Weekday",
    "_precip_bucket": "Precipitation Bucket",
    "dwd_precip_mm": "Precipitation (mm)",
    "dwd_station_dist_km": "Weather Station Distance (km)",
    "dwd_station_id": "Weather Station ID",
    "dwd_temp_air_2m": "Air Temperature (C)",
    "dwd_visibility_m": "Visibility (m)",
    "dwd_wind_speed_ms": "Wind Speed (m/s)",
    "h3_cell": "H3 Location Cell",
    "osm_dominant_road_class": "Dominant Road Class",
    "osm_maxspeed_max": "Max Speed Limit (km/h)",
    "osm_maxspeed_mean": "Mean Speed Limit (km/h)",
    "osm_road_density": "Road Density",
    "osm_way_count": "Road Way Count",
}


def display_feature_name(name: str) -> str:
    """Return a human-readable label for a contract column name, or the raw name if unmapped."""
    return FEATURE_DISPLAY_NAMES.get(name, name)


COLUMN_LABEL_MAPS = {
    "UART": UART_LABELS,
    "UTYP1": UTYP1_LABELS,
    "STRZUSTAND": STRZUSTAND_LABELS,
    "ULICHTVERH": LICHTVERH_LABELS,
    "UWOCHENTAG": WEEKDAY_LABELS,
}


def decode_feature_value(feature: str, value):
    """Return a human-readable value for a coded column (e.g. UART code -> label), else value unchanged."""
    return COLUMN_LABEL_MAPS.get(feature, {}).get(value, value)
```

Modify the import block near the top of the file (after `import logging`) to add `math`:

```python
import logging
import math
```

Add after `load_categorical_options` (after line 138):

```python
def precision_decimals(precision: float) -> int:
    """Convert a grid precision like 0.1 or 0.5 into a ROUND() decimal-places argument."""
    if precision <= 0:
        raise ValueError(f"precision must be positive, got {precision}")
    return max(0, -int(round(math.log10(precision))))


@st.cache_data
def load_severity_grid(precision: float = 0.1) -> pd.DataFrame:
    """Aggregate accidents.parquet into a lat/lon grid with per-cell KSI/slight counts.

    UKATGEORIE in (1, 2) is KSI, UKATGEORIE == 3 is slight (verified against
    notebooks/01_Q_Phase.py lines 156-157; no invalid UKATGEORIE values exist
    in this file). Aggregation happens entirely in DuckDB so the ~2.09M-row
    parquet is never loaded into pandas row-by-row - only the grouped result
    (a few thousand rows at precision=0.1) crosses into pandas.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT
                ROUND(LAT, {precision_decimals(precision)}) AS lat_bin,
                ROUND(LON, {precision_decimals(precision)}) AS lon_bin,
                SUM(CASE WHEN UKATGEORIE IN (1, 2) THEN 1 ELSE 0 END) AS ksi_count,
                SUM(CASE WHEN UKATGEORIE = 3 THEN 1 ELSE 0 END) AS slight_count,
                COUNT(*) AS total
            FROM '{ACCIDENTS_PARQUET}'
            GROUP BY 1, 2
        """  # noqa: S608
        return con.execute(query).df()
    except Exception:
        logger.exception("Failed to load severity grid")
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -k "display_feature_name or decode_feature_value or severity_grid" -v`
Expected: 6 passed. (`load_severity_grid` reads the real 407MB `data/accidents.parquet`, so this step needs `git lfs pull` already done, same as the existing `test_load_champion_model_predicts_on_real_contract_row` integration test.)

- [ ] **Step 5: Run the full test suite and lint**

Run: `uv run pytest tests/test_streamlit_app.py -v && uv run ruff check src/unfallatlas/viz/streamlit_app.py`
Expected: 24 passed (18 existing + 6 new), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add feature-label helpers and severity-grid loader for Streamlit app"
```

---

### Task 2: Honest UREGBEZ/UKREIS labeling on the Risk Predictor page

**Files:**
- Modify: `app/pages/risk_predictor.py:32-44`

**Interfaces:**
- Consumes: existing `get_column_spec`, `load_categorical_options`, `defaults` (unchanged).
- Produces: nothing new for later tasks (self-contained UI text change).

**Why not a real name lookup:** `UREGBEZ`/`UKREIS` require `ULAND` (Bundesland) to resolve to an official Gemeindeschlüssel/region name (see Global Constraints); `ULAND` isn't part of the model's feature set or the inference contract, so no real name can be shown without inventing one. The fix is transparency: replace the bare `(UREGBEZ)`/`(UKREIS)` labels with explicit "code, not a name" wording and a caption, so a user doesn't mistake the digits for something decoded (as they might reasonably assume after seeing `UWOCHENTAG` show "Monday").

- [ ] **Step 1: Edit the widget labels and add a caption**

In `app/pages/risk_predictor.py`, replace lines 36-44:

```python
        uregbez_categories = get_column_spec(contract, "UREGBEZ")["categories"]
        uregbez = st.selectbox(
            "Regierungsbezirk (UREGBEZ)",
            options=uregbez_categories,
            index=uregbez_categories.index(defaults["UREGBEZ"]),
        )
        ukreis = st.selectbox(
            "Kreis (UKREIS)", options=ukreis_options, index=ukreis_options.index(defaults["UKREIS"])
        )
```

with:

```python
        uregbez_categories = get_column_spec(contract, "UREGBEZ")["categories"]
        uregbez = st.selectbox(
            "Regierungsbezirk code (UREGBEZ)",
            options=uregbez_categories,
            index=uregbez_categories.index(defaults["UREGBEZ"]),
        )
        ukreis = st.selectbox(
            "Kreis code (UKREIS)",
            options=ukreis_options,
            index=ukreis_options.index(defaults["UKREIS"]),
        )
        st.caption(
            "Dataset-internal codes, not official region names: the model's "
            "feature set doesn't include the Bundesland key (ULAND) needed to "
            "resolve these to an official Gemeindeschlüssel/region name."
        )
```

- [ ] **Step 2: Manually verify the page renders**

Run: `uv run streamlit run app/streamlit_app.py` (or, for a scripted check, run the existing `AppTest`-based test in Task 3's test file once it exists). Visually confirm the caption appears under the two selectboxes on the Risk Predictor page.

- [ ] **Step 3: Run the existing test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: 24 passed, no regressions (this task changes only widget label strings, not any tested function).

- [ ] **Step 4: Commit**

```bash
git add app/pages/risk_predictor.py
git commit -m "fix: clarify UREGBEZ/UKREIS as internal codes, not region names"
```

---

### Task 3: Interactive map picker for LON/LAT on the Risk Predictor page

**Files:**
- Modify: `app/pages/risk_predictor.py:110-127` (Location section), `:206-243` (submit-time `widget_values` assembly)
- Test: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `get_column_spec(contract, "LAT"/"LON")` (existing, gives `min`/`max` bounds), `DEFAULT_WIDGET_VALUES["LAT"/"LON"]` (existing).
- Produces: `st.session_state["picked_lat"]`, `st.session_state["picked_lon"]` (floats) — read by the form's submit handler in this same file; no other task depends on these keys.

**Design decision (why outside the form):** `st.form` only propagates widget values to the script on submit; a `streamlit_folium.st_folium` map needs an immediate rerun on click to show the picked marker, which a form batches away. So the map picker lives **outside** `st.form("risk_predictor_form")`, above it, writing the click to `st.session_state`; the form reads those session-state values at submit time instead of holding its own LON/LAT widgets.

**Known AppTest limitation (documented up front, not discovered late):** `streamlit_folium.st_folium` is a bidirectional custom component (iframe + JS map library). `streamlit.testing.v1.AppTest` can render the component's presence but cannot simulate a real map click (no `last_clicked` payload is produced without the actual JS map runtime) — the same category of gap Task 9 of the original Phase K plan hit with Playwright's Chromium in a sandboxed subagent. The test in this task therefore covers: (a) the page renders without exception when no click has occurred yet (default LAT/LON path), and (b) `build_input_row`/`predict_ksi` still work end-to-end using the session-state defaults. Real click-driven map interaction is out of automated test reach here and is called out explicitly rather than silently skipped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_risk_predictor_page.py`:

```python
from streamlit.testing.v1 import AppTest


def test_risk_predictor_page_loads_with_default_location_and_can_predict():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=30)
    at.run()
    assert not at.exception

    submit_buttons = [b for b in at.button if "Predict KSI risk" in b.label]
    assert len(submit_buttons) == 1
    submit_buttons[0].click().run()
    assert not at.exception
    assert any("Prediction:" in md.value for md in at.markdown)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_risk_predictor_page.py -v`
Expected: currently PASSES already (the page works today) — this confirms the baseline before the map-picker change; re-run after Step 3 to confirm it still passes with the map in place. (This is a regression guard around the refactor, not a new-feature test — the map's own click behavior is untestable per the AppTest limitation noted above.)

- [ ] **Step 3: Replace the LON/LAT number inputs with a map picker**

In `app/pages/risk_predictor.py`, add to the imports at the top:

```python
import folium
from streamlit_folium import st_folium
```

Before the `with st.form("risk_predictor_form"):` line (before line 32), insert:

```python
lon_spec = get_column_spec(contract, "LON")
lat_spec = get_column_spec(contract, "LAT")
st.session_state.setdefault("picked_lat", defaults["LAT"])
st.session_state.setdefault("picked_lon", defaults["LON"])

st.subheader("Pick a location")
st.caption(
    "Click a point on the map to set the accident's longitude/latitude. "
    "Clicks outside Germany's covered area are ignored with a warning below."
)
picker_map = folium.Map(
    location=[st.session_state["picked_lat"], st.session_state["picked_lon"]], zoom_start=6
)
folium.Marker(
    [st.session_state["picked_lat"], st.session_state["picked_lon"]], tooltip="Selected location"
).add_to(picker_map)
map_state = st_folium(picker_map, height=350, width=None, key="location_picker")

if map_state and map_state.get("last_clicked"):
    clicked_lat = map_state["last_clicked"]["lat"]
    clicked_lon = map_state["last_clicked"]["lng"]
    if lat_spec["min"] <= clicked_lat <= lat_spec["max"] and lon_spec["min"] <= clicked_lon <= lon_spec["max"]:
        st.session_state["picked_lat"] = clicked_lat
        st.session_state["picked_lon"] = clicked_lon
    else:
        st.warning(
            f"Clicked point ({clicked_lat:.4f}, {clicked_lon:.4f}) is outside the "
            f"covered range (lat {lat_spec['min']:.2f}-{lat_spec['max']:.2f}, "
            f"lon {lon_spec['min']:.2f}-{lon_spec['max']:.2f}) and was ignored."
        )

st.caption(
    f"Selected: lat {st.session_state['picked_lat']:.4f}, "
    f"lon {st.session_state['picked_lon']:.4f}"
)
```

Now remove the old "Location" subsection inside the form. Delete lines 110-127 (the `st.subheader("Location")` block with `c7`/`c8` and the `lon`/`lat` number inputs) entirely — the map picker above the form replaces it.

In the submit-time `widget_values` dict (originally lines 228-229), replace:

```python
        "LON": lon,
        "LAT": lat,
```

with:

```python
        "LON": st.session_state["picked_lon"],
        "LAT": st.session_state["picked_lat"],
```

- [ ] **Step 4: Run the test again to verify it still passes**

Run: `uv run pytest tests/test_risk_predictor_page.py -v`
Expected: 1 passed (form still submits and predicts using the session-state-backed default location, since no click event occurs under `AppTest`).

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests/test_streamlit_app.py tests/test_risk_predictor_page.py -v`
Expected: 25 passed (24 from Task 1 + this page test), no regressions.

- [ ] **Step 6: Manual check**

Run: `uv run streamlit run app/streamlit_app.py`, open Risk Predictor, click a point inside Germany on the map, confirm the marker moves and the caption updates; click a point clearly outside the bbox (e.g. mid-Atlantic) and confirm the warning appears and the selection doesn't change.

- [ ] **Step 7: Commit**

```bash
git add app/pages/risk_predictor.py tests/test_risk_predictor_page.py
git commit -m "feat: replace LON/LAT number inputs with an interactive map picker"
```

---

### Task 4: Interactive severity map on the Overview page

**Files:**
- Modify: `app/pages/overview.py`
- Test: `tests/test_overview_page.py` (new)

**Interfaces:**
- Consumes: `load_severity_grid()` (Task 1), `SEVERITY_COLORS` (existing).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_overview_page.py`:

```python
from streamlit.testing.v1 import AppTest


def test_overview_page_loads_and_renders_severity_map():
    at = AppTest.from_file("app/pages/overview.py", default_timeout=60)
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run the test to verify current baseline**

Run: `uv run pytest tests/test_overview_page.py -v`
Expected: currently PASSES (the page already works before this change) — confirms baseline, then re-run after Step 3.

- [ ] **Step 3: Add the severity map**

In `app/pages/overview.py`, add to imports:

```python
import folium
from streamlit_folium import st_folium
```

and add `load_severity_grid` to the existing `from unfallatlas.viz.streamlit_app import (...)` block:

```python
from unfallatlas.viz.streamlit_app import (
    LIMITATIONS_TEXT,
    SEVERITY_COLORS,
    load_3class_comparison,
    load_binary_comparison,
    load_model_card,
    load_severity_grid,
)
```

After the existing `col_a`/`col_b` Pareto-front block (after line 37, before the `Limitations` expander), add:

```python
st.markdown("---")
st.subheader("Where accidents happen: severity by location")
st.caption(
    "Each marker aggregates accidents within a ~0.1 degree (~11 km) grid cell. "
    "Color shows the dominant severity in that cell; size scales with accident count."
)
grid_df = load_severity_grid()
severity_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)
for _, cell in grid_df.iterrows():
    ksi_share = cell["ksi_count"] / cell["total"]
    color = SEVERITY_COLORS["KSI"] if ksi_share >= 0.5 else SEVERITY_COLORS["slight"]
    folium.CircleMarker(
        location=[cell["lat_bin"], cell["lon_bin"]],
        radius=min(15, 3 + cell["total"] / 500),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=(
            f"KSI: {int(cell['ksi_count'])}, slight: {int(cell['slight_count'])}, "
            f"total: {int(cell['total'])}"
        ),
    ).add_to(severity_map)
st_folium(severity_map, height=450, width=None, key="overview_severity_map", returned_objects=[])
```

Note `returned_objects=[]`: this page only displays the map, so no click/zoom state needs to round-trip back to Python, keeping the component lightweight.

- [ ] **Step 4: Run the test to verify it still passes**

Run: `uv run pytest tests/test_overview_page.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest tests/ -v && uv run ruff check app/ src/unfallatlas/viz/streamlit_app.py`
Expected: all passing, ruff clean. (One pre-existing unrelated failure, `test_notebook_presentation_contract.py`, was already known before this plan — confirm it's the same failure, not a new one, via `uv run pytest tests/test_notebook_presentation_contract.py -v` output matching what was seen at the end of the original Phase K plan.)

- [ ] **Step 6: Commit**

```bash
git add app/pages/overview.py tests/test_overview_page.py
git commit -m "feat: add interactive severity map to the Overview page"
```

---

### Task 5: Human-readable feature names and values on Why This Prediction

**Files:**
- Modify: `app/pages/why_this_prediction.py`
- Test: `tests/test_why_this_prediction_page.py` (new)

**Interfaces:**
- Consumes: `display_feature_name`, `decode_feature_value` (Task 1).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_why_this_prediction_page.py`:

```python
from streamlit.testing.v1 import AppTest


def test_why_this_prediction_shows_readable_labels_after_a_prediction():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=30)
    at.run()
    submit_buttons = [b for b in at.button if "Predict KSI risk" in b.label]
    submit_buttons[0].click().run()

    at2 = AppTest.from_file("app/pages/why_this_prediction.py", default_timeout=30)
    at2.session_state["last_prediction"] = at.session_state["last_prediction"]
    at2.run()
    assert not at2.exception

    table_values = [t.value for t in at2.table]
    assert len(table_values) == 1
    feature_column = table_values[0]["feature"].tolist()
    assert "Wind Speed (m/s)" in feature_column
    assert "UART" not in feature_column  # raw code name should not appear, only the label
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_why_this_prediction_page.py -v`
Expected: FAIL — `feature_column` currently contains raw names like `"UART"`, not `"Wind Speed (m/s)"`.

- [ ] **Step 3: Implement the label decoding**

Replace the full contents of `app/pages/why_this_prediction.py` with:

```python
"""Why This Prediction page: global permutation importance + user-input context."""

import pandas as pd
import streamlit as st

from unfallatlas.viz.streamlit_app import (
    decode_feature_value,
    display_feature_name,
    load_permutation_importance,
)

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
importance_df["display_name"] = importance_df["feature"].apply(display_feature_name)
st.bar_chart(importance_df.set_index("display_name")["importance_mean"])

st.subheader("Your inputs for the globally most influential features")
st.caption(
    "These are the values you submitted for the model's top globally-important "
    "features. This is context, not a causal explanation of this specific prediction."
)
top_features = importance_df["feature"].tolist()
rows = [
    {
        "feature": display_feature_name(feature),
        "your value": decode_feature_value(feature, last_prediction["inputs"][feature]),
    }
    for feature in top_features
    if feature in last_prediction["inputs"]
]
st.table(pd.DataFrame(rows))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_why_this_prediction_page.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest tests/ -v && uv run ruff check app/`
Expected: all passing (except the pre-existing unrelated `test_notebook_presentation_contract.py` failure noted in Task 4), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add app/pages/why_this_prediction.py tests/test_why_this_prediction_page.py
git commit -m "feat: show human-readable feature names and decoded values on Why This Prediction"
```

---

### Task 6: German explanation of the Random Forest vs. XGBoost finalist decision

**Files:**
- Modify: `app/pages/model_comparison.py:60-61`

**Interfaces:**
- Consumes: `contract["decision_evidence"]["preference_conclusion"]["statement"]` (existing, unchanged).
- Produces: nothing consumed by later tasks.

**Background (verified against `data/processed/c_phase_inference_contract.json`'s `decision_evidence.finalist_measurements`):** on the measured validation-set comparison, XGBoost's `macro_f1` (0.570) and `recall_ksi` (0.682) beat Random Forest's (0.601 macro_f1 is actually higher for RF — re-check: RF macro_f1=0.6011 > XGBoost macro_f1=0.5699, but XGBoost recall_ksi=0.6824 > RF recall_ksi=0.5401 and XGBoost is much faster, 12.2ms vs 43.6ms per 1k). The contract's own `preference_conclusion.statement` says XGBoost "leads the measured validation matrix" on the combined criteria despite RF's higher raw macro-F1, because the combined criteria weigh recall and latency alongside macro-F1. Random Forest remains the deployment champion regardless, because Random Forest's champion status was already locked in using the 2023 validation set before this additional finalist robustness/latency comparison was run, and the 2024 test set (used only to report Random Forest's final, already-decided metrics) cannot be reused to re-select between candidates without invalidating the independence of those test metrics (data leakage / test-set overfitting on the model-selection process itself).

- [ ] **Step 1: Add the explanation expander**

In `app/pages/model_comparison.py`, replace line 61:

```python
st.info(contract["decision_evidence"]["preference_conclusion"]["statement"])
```

with:

```python
st.info(contract["decision_evidence"]["preference_conclusion"]["statement"])

with st.expander("Warum bleibt Random Forest der Champion? (German explanation)"):
    st.markdown(
        "**Deutsch:** Auf den hier gemessenen Validierungsdaten (2023) schneidet "
        "XGBoost über die kombinierten Kriterien (Recall, Latenz, Robustheit) besser "
        "ab als Random Forest — Random Forest wäre also nicht der bevorzugte Finalist, "
        "wenn man ausschließlich diese Kennzahlen vergleicht. Random Forest bleibt "
        "trotzdem das ausgelieferte Champion-Modell: Die Modellauswahl wurde bereits "
        "vorher auf Basis der Validierungsdaten getroffen, und der separate "
        "Testdatensatz (2024) darf danach nicht mehr benutzt werden, um zwischen "
        "Kandidaten zu wählen - sonst waeren die fuer Random Forest bereits "
        "berichteten Testmetriken nicht mehr unabhaengig (Data Leakage durch "
        "Overfitting auf den Testdatensatz). Die urspruengliche Entscheidung fuer "
        "Random Forest bleibt deshalb bestehen, auch wenn dieser spaetere "
        "Robustheits-/Latenzvergleich XGBoost knapp vorne sieht.\n\n"
        "**English gloss:** On the measured validation data, XGBoost beats Random "
        "Forest on the combined criteria (recall, latency, robustness) - so Random "
        "Forest isn't the preferred finalist by these numbers alone. It remains the "
        "deployment champion anyway because model selection was already locked in "
        "earlier using validation data; the held-out 2024 test set can't be reused "
        "afterward to re-pick between candidates without invalidating the "
        "independence of the test metrics already reported for Random Forest "
        "(that would be data leakage from overfitting the model-selection process "
        "to the test set). So the original Random Forest choice stands, even though "
        "this later robustness/latency comparison shows XGBoost slightly ahead."
    )
```

- [ ] **Step 2: Manual check**

Run: `uv run streamlit run app/streamlit_app.py`, open Model Comparison, expand the new expander, confirm both language blocks render.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: no regressions (this page has no dedicated test file yet in the existing suite; confirm via `uv run ruff check app/pages/model_comparison.py` that the new markdown string is syntactically fine).

- [ ] **Step 4: Commit**

```bash
git add app/pages/model_comparison.py
git commit -m "docs: add German explanation of the Random Forest vs XGBoost finalist decision"
```

---

### Task 7: Streamlit Community Cloud deployment prep

**Files:**
- Create: `requirements.txt`
- Modify: `AGENTS.md`, `README.md`

**Interfaces:**
- Consumes: `pyproject.toml`'s base `dependencies` list (lines 21-44) — the app only imports from base deps (`duckdb`, `folium`, `joblib`, `pandas`, `scikit-learn` via `sklearn.pipeline`, `streamlit`, `streamlit-folium`); none of the `geo`/`dev`/`presentation` extras are imported by `app/` or `src/unfallatlas/viz/streamlit_app.py`.
- Produces: nothing consumed by later tasks except the manual-step instructions Task 8's README section links to.

**Research finding (verified via web search this session):** Streamlit Community Cloud (1) supports Git LFS automatically with no special configuration, but (2) does not understand `uv.lock`, and if it finds a `pyproject.toml` it assumes Poetry's format — this repo's `pyproject.toml` uses the PEP 621 + hatchling format, not Poetry, so Community Cloud would very likely fail to parse it correctly. The reliable path is a plain `requirements.txt` at the repo root (Community Cloud's documented, recommended format).

- [ ] **Step 1: Generate `requirements.txt` from the base dependencies**

Create `requirements.txt` at the repo root with exactly the base `dependencies` array from `pyproject.toml` (lines 21-44), pinned to the same floors, one per line, no extras:

```
catboost>=1.2
duckdb>=1.1
folium>=0.17
holidays>=0.48
imbalanced-learn>=0.12
ipykernel>=6
lightgbm>=4.3
nbformat>=4.2
optuna>=3.6
pandas>=2.2
pip>=23
plotly>=5.22
polars>=1
pyarrow>=16
requests>=2.32
scikit-learn>=1.4
scipy>=1.13
shap>=0.45
streamlit>=1.35
streamlit-folium>=0.20
tqdm>=4.66
xgboost>=2
```

- [ ] **Step 2: Verify it installs cleanly in isolation**

Run: `python3 -m venv /tmp/streamlit-cloud-check && /tmp/streamlit-cloud-check/bin/pip install -r requirements.txt -q && /tmp/streamlit-cloud-check/bin/python -c "import streamlit, streamlit_folium, duckdb, folium, joblib, pandas, sklearn; print('ok')"`
Expected: `ok` printed, no errors. Then remove the throwaway venv: `rm -rf /tmp/streamlit-cloud-check`.

- [ ] **Step 3: Document the manual deployment step**

In `AGENTS.md`, after the existing launch-command section (near line 63, after the `uv run streamlit run app/streamlit_app.py` line), add:

```markdown
### Deploying to Streamlit Community Cloud (manual, one-time)

This is a human action through Streamlit's web UI — no script or agent can complete it:

1. Go to https://share.streamlit.io and sign in with the GitHub account that owns/has access to this repo.
2. Click "New app", select this repo, branch `main`, main file path `app/streamlit_app.py`.
3. Deploy. Streamlit Community Cloud auto-detects `requirements.txt` at the repo root and Git LFS objects (`data/accidents.parquet`, `data/processed/a3_binary_best_model.joblib`) automatically - no extra LFS configuration is needed.
4. Once live, update the "Live app" link in `README.md` with the real `https://<app-name>.streamlit.app` URL (see the "Live Deployment" section).
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt AGENTS.md
git commit -m "chore: add requirements.txt and manual deployment instructions for Streamlit Community Cloud"
```

---

### Task 8: README and GitHub Pages cross-linking

**Files:**
- Modify: `README.md`
- Modify: `src/unfallatlas/presentation/templates/site_index.html.j2`
- Test: existing `tests/presentation/` suite (no new test needed — this is a template text/link change; verify via the existing presentation contract test)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add a "Live Deployment" section to README.md**

In `README.md`, after the existing Phase K table row (near line 64, `| **K** | Knowledge Transfer | ... |`), add a new section (place it near the top-level intro, right after the phase table):

```markdown
## Live Deployment

- **Streamlit app:** https://REPLACE-WITH-REAL-STREAMLIT-CLOUD-URL.streamlit.app (deployed per `AGENTS.md`'s "Deploying to Streamlit Community Cloud" section) - interactive risk predictor, model comparison, and severity map.
- **Notebook presentations (GitHub Pages):** the full Q/U/A³/C phase notebooks, rendered as static HTML, are linked from the Streamlit app's Overview page and from the notebook-presentation site itself (see below).
```

- [ ] **Step 2: Add the reciprocal link in the notebook-presentation site template**

In `src/unfallatlas/presentation/templates/site_index.html.j2`, inside the `<header class="presentation-header index-header">` block, after the `<h1>Notebook presentations</h1>` line, add:

```html
      <p class="source-path"><a href="https://REPLACE-WITH-REAL-STREAMLIT-CLOUD-URL.streamlit.app">Open the interactive Streamlit app -&gt;</a></p>
```

So the block reads:

```html
  <header class="presentation-header index-header">
    <div class="header-content">
      <p class="document-kind">Unfallatlas QUA³CK</p>
      <h1>Notebook presentations</h1>
      <p class="source-path"><a href="https://REPLACE-WITH-REAL-STREAMLIT-CLOUD-URL.streamlit.app">Open the interactive Streamlit app -&gt;</a></p>
      {% if generated_at_local %}<p class="source-path">Updated: <time datetime="{{ manifest.generated_at | e }}">{{ generated_at_local }}</time></p>{% endif %}
    </div>
  </header>
```

- [ ] **Step 3: Add a reciprocal link from the Overview page back to GitHub Pages**

In `app/pages/overview.py`, at the end of the file (after the `Limitations` expander added originally, and after Task 4's severity-map section), add:

```python
st.markdown(
    "[View the full Q/U/A3/C phase notebooks (GitHub Pages)]"
    "(https://REPLACE-WITH-OWNER.github.io/unfallatlas-qua3ck/)"
)
```

Use the real GitHub Pages URL from the repo's Pages settings (`https://<github-username-or-org>.github.io/unfallatlas-qua3ck/`) in place of the placeholder — confirm the exact owner/org by running `git remote get-url origin` and deriving the Pages URL from it before committing.

- [ ] **Step 4: Run the presentation test suite to confirm the template still renders**

Run: `uv run pytest tests/presentation/ -v`
Expected: all passing, confirming the Jinja2 template change didn't break rendering (the existing suite exercises `site_index.html.j2` rendering via `manifest.py`'s `render_site_index`).

- [ ] **Step 5: Run the full app test suite**

Run: `uv run pytest tests/ -v`
Expected: all passing except the pre-existing unrelated `test_notebook_presentation_contract.py` failure already noted in Task 4.

- [ ] **Step 6: Commit**

```bash
git add README.md src/unfallatlas/presentation/templates/site_index.html.j2 app/pages/overview.py
git commit -m "docs: cross-link the Streamlit app and the GitHub Pages notebook presentations"
```

---

## Notes for the implementer of Task 8 and Task 7

The exact Streamlit Community Cloud URL and the exact GitHub Pages URL are not known until Task 7's manual deployment step (Task 7 Step 3) actually happens and until the repo's Pages settings are confirmed. Both tasks use an explicit `REPLACE-WITH-...` placeholder string, not a guessed URL — this is intentionally visible (grep-able) rather than a silently wrong link, and should be flagged back to the user as a follow-up once the real URLs are known, rather than left as a silent TODO.
