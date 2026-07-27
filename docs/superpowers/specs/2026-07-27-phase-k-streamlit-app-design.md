# Phase K Streamlit App — "Risk Explainer & Model Console" — Design

**Status:** Approved by user, ready for `writing-plans`.

## Context

`app/streamlit_app.py`, `src/unfallatlas/viz/streamlit_app.py`, and
`src/unfallatlas/viz/geo.py` are all 0-byte stubs. Phase K
("Knowledge Transfer") per `docs/GLOSSARY.md` requires "an interactive
Streamlit app, final report, and this documentation." No further rubric
constraints exist beyond that; `docs/project/PROJEKTPLAN_SETUP.md`'s
Wiesbaden-focused mockup is early brainstorming with no code/data trace and
is not binding.

`AGENTS.md`'s auto-managed sections are stale: they describe
`notebooks/04_C_Phase.ipynb` as an empty TODO stub. It is not — it is a
58.8 MB executed notebook, and the C-phase produced a complete,
already-committed artifact set in `data/processed/` (verified via
`git ls-files` + `git lfs ls-files`, all tracked, no local-only files):

- `c_phase_inference_contract.json` — the deployment contract: model path
  (`a3_binary_best_model.joblib`, a full `sklearn.Pipeline` including
  preprocessing), decision threshold (`0.49860217036273086`), target
  encoding (`1 = KSI {UKATGEORIE ∈ {1,2}}, 0 = slight {UKATGEORIE = 3}`),
  and `required_columns`: 28 entries, each with `name`/`dtype` and, for
  categoricals, an exact `categories` list (verified via
  `Read` of the JSON — see column table below).
- `c_phase_candidate_metrics.csv` (10 rows: one per candidate model family,
  `model, family, evaluation_role, macro_f1, recall_ksi, recall_slight,
  confusion_matrix, latency_ms_per_1k` — `confusion_matrix` is a
  Python-list-literal string, parseable with `ast.literal_eval`).
- `c_phase_candidate_robustness.csv` (50 rows = 10 models × 5 perturbed
  features: `model, feature, prediction_failed, mean_abs_score_drift,
  changed_class_share, error`) — raw per-feature probe detail; the
  **aggregated** per-finalist robustness (`robustness_score`,
  `robustness_status`) already lives in the inference contract's
  `decision_evidence.finalist_measurements` and is preferred for the UI.
- `c_phase_permutation_importance.csv` (120 rows = 10 models × 12 features:
  `model, feature, importance_mean, importance_std, rank`).
- `a3_binary_model_comparison.csv` / `a3_model_comparison.csv` — full binary
  (10 rows) and 3-class (19 rows) candidate tables, columns matching
  `plot_binary_f1_recall_front`/`plot_f1_recall_front`'s expected
  `{model, macro_f1, recall_ksi}` / `{model, macro_f1, recall_class_1}`.
- `a3_binary_model_card.json` — champion metrics incl. `val_2023_metrics`
  and `test_2024_metrics`, each with a `confusion_matrix: [[tp, fn], [fp, tn]]`
  2×2 list (row 0 = actual KSI: [predicted KSI correctly, predicted slight
  incorrectly]; row 1 = actual slight: [predicted KSI incorrectly, predicted
  slight correctly] — verified via direct read — no recomputation needed).
- `data/processed/c_phase_candidate_scores.parquet` (2.69M rows × 10 models,
  columns `{model, row, score, prediction}`, **no `y_true`**) — deliberately
  **not used live**: recomputing ROC/PR curves would require re-deriving
  ground truth at runtime for a 2.69M-row file, for no benefit over the
  already-computed aggregate metrics above.
- `src/unfallatlas/viz/metrics_viz.py` already implements
  `plot_binary_f1_recall_front`, `plot_f1_recall_front`, and
  `plot_confusion_matrix_heatmap` — all confirmed (via `find_symbol`) to
  return **interactive `plotly.graph_objects.Figure`** objects (not
  matplotlib), so they render via `st.plotly_chart` directly.
- No SHAP was ever computed (`viz/shap_plots.py` is a stub) — the C-phase
  used **permutation importance** instead. The app must not imply SHAP or
  per-instance causal attribution anywhere.
- **Dependencies are already sufficient**: `streamlit>=1.35` and
  `streamlit-folium>=0.20` are base (non-optional) dependencies in
  `pyproject.toml`; `matplotlib`/`joblib` resolve transitively (verified in
  `uv.lock`). **No `pyproject.toml` changes are required for this app.**
- Reference project (`Degrees-of-No-Return-App/app.py`) patterns adapted:
  `st.cache_resource` for the loaded model, `st.cache_data` for
  JSON/CSV reads, `st.metric` with `delta` for confidence framing,
  `st.expander` for a limitations disclaimer, `src/` holding all
  non-UI logic with `app.py` as a thin entry point.

## Goal

A user who freshly clones the repo and runs `git lfs pull` must be able to
run `uv sync && uv run streamlit run app/streamlit_app.py` and use every
page and feature of the app immediately, with **no notebook execution, no
retraining, and no network access** required. Every artifact the app reads
already exists under `data/processed/` and is already committed
(confirmed: not `.gitignore`d, and the two binaries are Git LFS pointers
that resolve after `git lfs pull`).

## Non-goals

- No geospatial map (Concept E's hybrid atlas is explicitly deferred; no
  `viz/geo.py` work in this plan).
- No live SHAP computation, no per-instance local explanations.
- No live ROC/PR curve recomputation from `c_phase_candidate_scores.parquet`.
- No retraining, no Optuna, no new model artifacts.
- No Docker/cloud deployment (local `uv run streamlit run` only, per
  AGENTS.md's already-documented launch command).
- No changes to `src/unfallatlas/models/`, `features/`, or notebooks.

## Architecture

```
app/
├── streamlit_app.py          # Entry point: st.set_page_config + st.navigation
└── pages/
    ├── overview.py            # Champion metrics + 3-class vs. binary ceiling story
    ├── risk_predictor.py      # Interactive KSI-risk prediction form
    ├── why_this_prediction.py # Global permutation importance + user-input context
    └── model_comparison.py    # Candidate table, Pareto front, confusion matrix, robustness

src/unfallatlas/viz/streamlit_app.py   # All cached loaders + pure helper functions
tests/test_streamlit_app.py            # Unit tests for the pure helpers (no Streamlit runtime)
```

`app/streamlit_app.py` stays thin: page config + `st.navigation()` wiring
only. Each page file is a thin script that calls into
`src/unfallatlas/viz/streamlit_app.py` for all data loading, model
inference, and DataFrame assembly — matching the repo's existing
notebook-to-library boundary convention. `src/unfallatlas/viz/streamlit_app.py`
contains no Streamlit widget calls itself (so it stays unit-testable
without a Streamlit runtime); only the page files call `st.*` UI functions.

## Data flow

1. `load_inference_contract() -> dict` — `st.cache_data`, reads
   `c_phase_inference_contract.json`.
2. `load_champion_model() -> Pipeline` — `st.cache_resource`, `joblib.load`
   on the contract's `model_path`.
3. `load_model_card() -> dict` — `st.cache_data`, reads
   `a3_binary_model_card.json`.
4. `load_binary_comparison() -> pd.DataFrame` / `load_3class_comparison() -> pd.DataFrame`
   — `st.cache_data`, read the two `*_model_comparison.csv` files.
5. `load_candidate_metrics() -> pd.DataFrame` — `st.cache_data`, reads
   `c_phase_candidate_metrics.csv`, parses `confusion_matrix` via
   `ast.literal_eval`.
6. `load_permutation_importance(model_name: str) -> pd.DataFrame` —
   `st.cache_data`, reads `c_phase_permutation_importance.csv`, filters to
   one model, sorts by `rank`.
7. `build_input_row(widget_values: dict, contract: dict) -> pd.DataFrame` —
   pure function: assembles a one-row `DataFrame` from the Risk Predictor
   widget values, applying the exact `dtype` per `required_columns` entry
   (categorical columns get their declared pandas dtype; booleans map
   `True/False`; `IstGkfz` is the one column whose contract `categories`
   are the strings `["False","True"]` under `dtype: object`, not real
   Python bools — must be built as literal strings, not `bool`).
8. `predict_ksi(model, row, threshold) -> tuple[float, int]` — pure
   function: `model.predict_proba(row)[0, 1]`, then `int(proba >= threshold)`
   (not `model.predict`, since the champion needs the contract's tuned
   threshold, not sklearn's default 0.5).

### Required-columns reference (from `c_phase_inference_contract.json`, verified)

| Column | dtype | Widget |
|---|---|---|
| `UREGBEZ` | str, categories `["0","1","2","3","4","5","6","7","9"]` (fixed, from contract) | `st.selectbox` |
| `UKREIS` | str, high-cardinality (87 values, **no fixed list in the contract**) | `st.selectbox`, options sourced from `load_categorical_options("UKREIS")` (see below), not the contract |
| `UMONAT` | int8, 1-12 | `st.slider` |
| `USTUNDE` | int8, 0-23 | `st.slider` |
| `UWOCHENTAG` | int8, 1-7 | `st.selectbox` (labeled Sun..Sat per `UWOCHENTAG` convention: 1=Sunday) |
| `UART` | int8, 0-9 | `st.selectbox` |
| `UTYP1` | int8, 1-7 | `st.selectbox` |
| `ULICHTVERH` | int8, 0-2 | `st.selectbox` (0=daylight,1=dusk,2=darkness) |
| `STRZUSTAND` | int8, 0-2 | `st.selectbox` (0=dry,1=wet,2=wintry) |
| `IstRad`, `IstPKW`, `IstFuss`, `IstKrad`, `IstSonstig` | bool | `st.checkbox` |
| `IstGkfz` | object, categories `["False","True"]` (string, not bool) | `st.checkbox`, cast to `str(bool_value)` when building the row |
| `LON`, `LAT` | float64, DE bounding box (5.87-15.03 / 47.32-55.04) | `st.number_input` bounded to contract min/max, default = dataset centroid |
| `dwd_station_id` | str, high-cardinality (528 values) | not user-facing; default to a representative fixed value (documented in the widget layer) since asking a user to pick a weather-station ID has no UX value |
| `dwd_station_dist_km` | float | derived/defaulted alongside `dwd_station_id` |
| `dwd_temp_air_2m`, `dwd_precip_mm`, `dwd_visibility_m`, `dwd_wind_speed_ms` | float | `st.slider`, bounded to contract min/max |
| `_precip_bucket` | category, `["dry (0 mm)", "light (0-5 mm)"]` | `st.selectbox`, or derived from `dwd_precip_mm` |
| `h3_cell` | str, high-cardinality (200,656 values) | not user-facing; default to a fixed representative cell (same rationale as `dwd_station_id`) |
| `osm_dominant_road_class` | str, 15 categories | `st.selectbox` |
| `osm_maxspeed_mean`, `osm_maxspeed_max`, `osm_road_density`, `osm_way_count` | float | `st.slider`, bounded to contract min/max |

Exact category lists, bounds, and the small set of non-user-facing columns
(`dwd_station_id`, `h3_cell`, `dwd_station_dist_km`) are read programmatically
from the contract at implementation time, not hardcoded twice — the table
above is the design-level summary confirmed against the live JSON.

`UKREIS` is the one **user-facing** column the contract deliberately leaves
without a fixed category list (87 real district codes, too many to hardcode
and liable to drift). Its widget options come from
`load_categorical_options("UKREIS") -> list[str]` — a `st.cache_data`
function that runs `SELECT DISTINCT UKREIS FROM 'data/accidents.parquet'
ORDER BY UKREIS` via DuckDB (a single-column columnar scan, not a full
dataset load; `data/accidents.parquet` is already a committed, Git-LFS
tracked asset per the Context section, so this stays within the
"no notebooks, LFS pull is enough" constraint). `dwd_station_id` and
`h3_cell` are deliberately **not** given this treatment — they are
technical join keys with no meaningful user interpretation, so they stay
non-user-facing and default to one fixed representative value each (picked
as the mode of the training data, computed once at implementation time and
hardcoded as a constant, not re-queried per app run).

## Pages

**Overview** (`app/pages/overview.py`)
- `st.metric` row: champion macro-F1 (test-2024), recall(KSI), decision
  threshold.
- Ceiling narrative: `st.plotly_chart(plot_f1_recall_front(load_3class_comparison()))`
  next to `st.plotly_chart(plot_binary_f1_recall_front(load_binary_comparison()))`,
  with a short caption: empirical 3-class ceiling (max macro-F1 0.424 over
  19 configs) vs. the binary reframe clearing both gates.
- `st.expander("Limitations")`: Cramér's V ≤ 0.13 for the strongest
  features, no demographic/impact-speed data, correlation ≠ causation,
  permutation importance is global not local. This same expander content
  is reused verbatim on the Risk Predictor page (shared constant in
  `streamlit_app.py`, not duplicated prose).

**Risk Predictor** (`app/pages/risk_predictor.py`)
- Sidebar or main-column form built from the required-columns table above.
- On submit: `build_input_row` → `predict_ksi` → result panel using the
  reserved severity colors (red `#E63946` for KSI, green `#2A9D8F` for
  slight; amber is not used since the binary model only distinguishes two
  buckets) plus the raw probability and the 0.4986 threshold shown
  alongside it so the user sees the operating point, not just a bare
  percentage.
- Stores `{inputs: dict, proba: float, prediction: int}` in
  `st.session_state["last_prediction"]`.
- Includes the same Limitations expander as Overview.

**Why This Prediction** (`app/pages/why_this_prediction.py`)
- If `st.session_state.get("last_prediction")` is empty: empty-state
  message directing the user to the Risk Predictor page first (no crash,
  no silent default).
- Global permutation importance for `binary_random_forest_balanced`
  (top 15 by `rank`, bar chart via `st.bar_chart` or a small Plotly bar).
- A **prominent, unmissable** notice: "This is global, model-level
  permutation importance from the C-phase analysis, not a per-instance
  SHAP explanation — none was computed for this project."
- Below the chart: a table of the user's current input values restricted
  to those top-15 features, captioned "your inputs for the globally most
  influential features" — explicitly not framed as causal attribution for
  this specific prediction.

**Model Comparison** (`app/pages/model_comparison.py`)
- Full 10-candidate table from `load_candidate_metrics()` (baselines
  through champion), sortable.
- `st.plotly_chart(plot_binary_f1_recall_front(...))` (same figure as
  Overview's binary panel, or shared via cache so it's not recomputed).
- `st.plotly_chart(plot_confusion_matrix_heatmap(model_card["test_2024_metrics"]["confusion_matrix"], labels=["KSI","slight"]))`.
- Finalist comparison table from the contract's
  `decision_evidence.finalist_measurements` (Random Forest, XGBoost,
  LightGBM, CatBoost): macro_f1, recall_ksi, latency_ms_per_1k,
  robustness_score, robustness_status.
- The contract's own honest conclusion, quoted directly:
  "XGBoost leads the measured validation matrix... Random Forest remains
  the deployment champion because the test set cannot be reused for
  challenger selection."

## Caching & session state

- `st.cache_resource`: the joblib-loaded `Pipeline` only.
- `st.cache_data`: every JSON/CSV read (all small: largest is the 120-row
  permutation-importance CSV and the 19-row 3-class comparison CSV).
- `st.session_state["last_prediction"]`: set by Risk Predictor, read by
  Why This Prediction. No other cross-page state.

## Error handling

- If a widget-built row's dtypes don't match `required_columns`
  (e.g. contract schema drifts in a future C-phase re-run), raise a clear
  `st.error` naming the offending column rather than letting sklearn throw
  an opaque exception.
- `logging` (not `print`) for any load failure, per repo convention.
- Missing `last_prediction` on the Why-This-Prediction page is a normal
  empty state, not an error.

## Testing & verification

- `tests/test_streamlit_app.py` (pytest, no Streamlit runtime): tests for
  `build_input_row` (correct dtypes incl. the `IstGkfz` string-bool edge
  case), `predict_ksi` (threshold applied correctly, not sklearn default),
  and each `load_*` function (reads real committed fixtures under
  `data/processed/`, or a tiny synthetic fixture mirroring the same schema
  if reading the 407 MB joblib in unit tests proves too slow — decided at
  implementation time).
- Manual verification: fresh-clone simulation is not literally re-cloned,
  but `git lfs pull` + `uv sync` + `uv run streamlit run app/streamlit_app.py`
  is run and every page is exercised with at least one prediction covering
  each categorical branch, confirming the AGENTS.md-documented launch
  command (currently non-functional against the stub) now works.

## Documentation updates (after implementation + verification)

- Fix `AGENTS.md`'s auto-managed Overview/Architecture sections (C-phase
  and Streamlit-stub descriptions are currently stale).
- `docs/AI TOOL DISCLOSURE.md` and `docs/prompts/04_prompts_phase_k.md`
  updated per the user's original Step 5 instructions, once the app is
  verified working end-to-end.
