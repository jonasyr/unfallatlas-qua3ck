# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

<!-- AUTO-MANAGED: project-description -->
## Overview

**unfallatlas-qua3ck** — ML portfolio project: multiclass classification of German traffic accident severity (`UKATGEORIE`: 1=Getötet, 2=Schwerverletzt, 3=Leichtverletzt) on the federal Unfallatlas dataset 2016–2024.

- Dataset: ~2.09M rows, 21 columns — primary file `data/accidents.parquet`
- Process model: **QUA³CK** (Frage → Untersuchung → Analyse → Auswertung → Kommunikation)
- Class imbalance: ~1% / 18% / 81% — primary metric is macro-F1
- Chronological split: Train 2016–2022, Val 2023, Test 2024 (no random splits)

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Build & Development Commands

```bash
# Install all dependencies (including dev and geo extras)
uv sync --all-extras

# Install only geo extras (geopandas, h3, osmnx) without dev tools
uv sync --extra geo

# Run tests (coverage report auto-generated via pyproject.toml addopts)
uv run pytest

# Lint
uv run ruff check .

# Format
uv run black .

# Sync Jupytext notebook mirrors (after editing .ipynb files)
uv run jupytext --sync notebooks/*.ipynb

# Re-index Serena after notebook sync
serena project index

# Launch Streamlit demo
uv run streamlit run app/streamlit_app.py

# Install package in editable mode
uv pip install -e .
```

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Architecture

```
unfallatlas-qua3ck/
├── notebooks/              # QUA³CK phase notebooks (source of truth)
│   ├── 01_Q_Phase.ipynb    # Research question & hypotheses (done)
│   ├── 02_U_Phase.ipynb    # EDA & feature engineering (in progress)
│   ├── 03_A3_Phase.ipynb   # Modelling & tuning (done)
│   └── 04_C_Phase.ipynb    # Comparison, SHAP, conclusions (TODO)
├── src/unfallatlas/        # Reusable production library
│   ├── data/               # download.py, dwd.py (weather), osm.py (road network)
│   ├── features/           # enrich.py, spatial.py (H3/OSM aggregation), temporal.py, preprocessing.py
│   ├── models/             # baseline.py, boosting.py, evaluate.py, ordinal.py, imbalance.py
│   └── viz/                # geo.py, shap_plots.py, streamlit_app.py, metrics_viz.py (planned)
├── app/                    # Streamlit demo entry point
├── data/
│   ├── accidents.parquet   # Main dataset (Git LFS)
│   └── raw/                # Local-only raw CSVs (not committed)
├── tests/                  # pytest test suite
├── scripts/                # Utility scripts
├── docs/                   # Disclosure + glossary (hard requirements) and supporting docs
│   ├── prompts/            # AI prompts used per QUA³CK phase (01_..., 02_..., referenced by AI TOOL DISCLOSURE.md)
│   ├── course-material/    # Lecture notes used as AI context (Einheit 1/2, Data Analytics und Big Data, ChatGPT best-practice notes)
│   ├── dataset/            # Unfallatlas dataset description (DSB_Unfallatlas.md/.pdf), used for citing + coded-label lookups
│   ├── project/            # Repo/process docs (ConventionalCommitsGuide.md, PROJEKTPLAN_SETUP.md)
│   └── superpowers/plans/  # Implementation plans (local dev artefact, not committed)
├── reports/figures/        # Generated output figures
└── pyproject.toml          # Project config (hatchling, ruff, black, jupytext, pytest-cov)
```

### Optional dependency groups

- `dev`: pytest, pytest-cov, ruff, black, jupytext, jupyter, ipywidgets
- `geo`: geopandas, h3, osmnx — required for OSM road network features; install with `uv sync --extra geo`

### Test coverage

Configured via `[tool.pytest.ini_options]` in `pyproject.toml` (`--cov=src/unfallatlas --cov-report=xml --cov-report=term-missing`); runs automatically with `uv run pytest`.

**Notebook → library boundary**: Reusable logic moves from notebook cells into `src/unfallatlas/` and is imported back into the notebook.

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## Code Conventions

**Formatting**
- Formatter: `ruff` + `black`, line-length 100
- Python target: 3.11+
- Ruff rules: E, F, I (isort), UP (pyupgrade); E501 ignored

**Coding rules**
- No `print()` in modules — use `logging`
- All paths via `pathlib.Path`, never raw strings
- Model artefacts saved to `data/processed/`
- Notebook outputs stripped before commits via `nbstripout`

**Notebook policy**
- `notebooks/*.ipynb` are the **source of truth** — edit these, not the `.py` mirrors
- `notebooks/*.py` are Jupytext/Serena mirrors for symbolic navigation — read-only for agents
- After editing a notebook, regenerate mirrors: `uv run jupytext --sync notebooks/*.ipynb`
- Never commit a changed `.py` mirror without the matching `.ipynb` also being updated

**Data loading**
```python
import duckdb
from pathlib import Path

DATA = Path("data/accidents.parquet")

# Preferred for large queries
con = duckdb.connect()
df = con.execute(f"SELECT * FROM '{DATA}' WHERE UJAHR = 2024").df()

# Raw CSVs (decimal comma, UTF-8-BOM)
df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
```

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Detected Patterns

**Target variable**
- Column is `UKATGEORIE` (typo, not `UKATEGORIE`) — always use the misspelled name
- Derive Bundesland: `df["ULAND"] = df["UKREIS"].str[:2].astype(int)`

**Train/val/test split**
```python
train = df[df.UJAHR <= 2022]
val   = df[df.UJAHR == 2023]
test  = df[df.UJAHR == 2024]
```

**Evaluation**
- Always use stratified splits and macro-F1 as primary metric
- No `ULAND` column in parquet — derive from `UKREIS` prefix

**ML stack**
- Boosting: LightGBM, XGBoost, CatBoost
- Imbalance handling: SMOTE via `imbalanced-learn`
- Hyperparameter tuning: Optuna
- Explainability: SHAP
- Spatial enrichment: DWD weather data, OSM road features, optional H3/osmnx

**OSM road network pipeline** (`src/unfallatlas/data/osm.py`, `src/unfallatlas/features/spatial.py`)

- `download_road_network(state, cache_dir, force_refresh)`: tiled fetch at 0.2° tiles (0.4° tried — caused >180s server timeouts, reverted) with per-tile parquet cache at `cache_dir/{state_slug}_tiles/tile_{i:04d}.parquet`; state-level retry up to `max_state_retries=5` per call; raises `_TransientFetchError` on network failure — failed tiles are never cached to avoid recording transient outages as "no roads"; raises `RuntimeError("No road data found")` when all retries exhausted with zero tile results; does NOT write whole-state cache if any tiles remain incomplete; uses `ox.graph_from_bbox` with `useful_tags_way=["highway","maxspeed"]` (NOT `features_from_place` — confirmed OOM at ~79 GiB); `overpass-api.de/api` endpoint; `ox.settings.log_file=True` (not `log_console`, which bypasses Jupyter via `sys.__stdout__`)
- `build_spatial_features(accidents_df, raw_cache_dir, interim_cache_dir, resolution=8, force_refresh=False)`: joins OSM features onto accident frame; requires `LAT`/`LON` columns (raises `RuntimeError` if missing); short-circuits to `interim_cache_dir/accidents_with_weather_spatial.parquet` if present; sizes overall-run ETA upfront via geocode-only `_state_total_tiles()` before first Overpass fetch; cross-state retry pass after main loop (up to `max_retry_passes=3`); H3 cell dedup across state boundaries keeps higher-`osm_way_count` version
- `_clean_road_gdf(gdf)`: filters to `_VEHICLE_HIGHWAY_VALUES` allow-list (motorway, trunk, primary, secondary, tertiary, unclassified, residential, living_street, service, track) — unknown highway values default to dropped, not included; normalizes list-valued `highway`/`maxspeed` OSM tags to first element (pyarrow rejects list-valued columns)
- `_fetch_tile_edges(bbox, custom_filter, max_retries=2)`: `truncate_by_edge=True` required — default False drops boundary-crossing edges producing grid-aligned data gaps; tile-boundary duplicate edges deduped on `(highway, maxspeed, geometry)` via `_geom_wkb` after combining all tiles
- `_TransientFetchError`: raised (not `None`) on network exhaustion — prevents caching transient outages as permanent "no roads" (Bayern/Brandenburg incident: 70 tiles silently zeroed before this fix)
- `aggregate_roads_to_h3(gdf, resolution=8)`: aggregates road GeoDataFrame to H3 cells; output columns: `{h3_cell, osm_dominant_road_class, osm_maxspeed_mean, osm_maxspeed_max, osm_road_density, osm_way_count}`; `osm_way_count` counts distinct ways, not vertices
- `parse_maxspeed(value)`: converts OSM maxspeed strings to km/h float; handles numeric strings, `"N mph"` (×1.60934), `DE:urban`→50, `DE:rural`→100; semicolon-separated lists take first value; returns `None` for unparseable values
- `assign_h3_cell(lat, lon, resolution=8)`: returns stable H3 cell string ID
- `ROAD_CLASS_RANK`: dict ranking highway types (motorway > primary > residential > …)

**Binary KSI reformulation** (planned — `feature/binary_ksi_reframe`)

- 3-class gate (macro-F1 ≥ 0.55 AND Recall(1) ≥ 0.50) is a **Bayes ceiling**: empirical max macro-F1 = 0.424 over 19 configs; Cramér's V ≤ 0.13 for strongest features; ~90× odds-lift required for class-1 precision at 0.94% base rate
- Binary target: `y = (UKATGEORIE <= 2).astype(int)` — 1=KSI ({1,2}), 0=slight ({3}); KSI share ≈ 16.4%
- Revised gate: **binary macro-F1 ≥ 0.55 AND Recall(KSI) ≥ 0.50**
- New library functions (pending implementation):
  - `find_gate_optimal_offsets(y_true, y_proba, classes, recall_gate_class=1, recall_gate=0.50, n_steps_o1=13, n_steps_o2=11) -> tuple[tuple[float, float] | None, float]` in `imbalance.py` — 2D log-prob sweep; returns `(None, best_unconstrained_f1)` when infeasible
  - `split_features_target_binary(df) -> tuple[pd.DataFrame, pd.Series]` in `preprocessing.py`
  - `build_lightgbm_binary_pipeline(preprocessor, class_weight="balanced", use_gpu=None) -> Pipeline` in `boosting.py`
  - `evaluate_binary_predictions(y_true, y_pred) -> dict` + `meets_binary_acceptance_criteria(metrics) -> bool` in `evaluate.py`; constants `BINARY_MACRO_F1_THRESHOLD=0.55`, `BINARY_RECALL_KSI_THRESHOLD=0.50`
  - `plot_f1_recall_front(comparison_df, ax=None, gate_f1=0.55, gate_recall=0.50, label_col="model") -> Axes` in `viz/metrics_viz.py`
- Notebook additions: A³-Phase §9 (ceiling evidence + Pareto-front plot) and §10 (binary KSI model, Optuna, gate assertion); Q-Phase §N (gate-revision narrative); U-Phase (forward-pointing methodological notes)
- Output artefacts: `data/processed/a3_binary_best_model.joblib`, `data/processed/a3_binary_model_card.json`, `reports/figures/a3_f1_recall_front.png`

**A³-phase modelling pipeline** (`notebooks/03_A3_Phase.ipynb`, `src/unfallatlas/models/`)

- **Champion selection**: `select_best_candidate()` (`unfallatlas.models.evaluate`) applies a recall gate — recall(class 1) >= 0.50 must be met before comparing macro-F1; `random_forest_balanced` was excluded despite highest raw macro-F1 (0.410) because recall(1)=0.229
- **Two candidate families**: `catboost_balanced` and `lightgbm_balanced` both advance to §6/§7; Random Forest/XGBoost stay in comparison table only
- **CatBoost clone() incompatibility**: `CatBoostClassifier` with non-None `class_weights` cannot survive any `clone()` call (sklearn's `cross_validate` clones internally per fold); fix: `class_weights` removed from `build_catboost_pipeline()` constructor; balanced weighting supplied via `sample_weight` through `cross_validate(params=...)` — sklearn slices `sample_weight` to fold indices automatically
- **Imbalance strategies in §6**: SMOTE/ADASYN/threshold-moving/ordinal are scored for information only; only balanced-weighted configurations (`{family}_balanced`) are selectable for Optuna tuning — `_build_pipeline_for()` raises `NotImplementedError` for resampling/ordinal strategies
- **SMOTE/ADASYN NaN issue**: `tree_preprocessor`'s passthrough branch leaves `IstGkfz` as Python `None` (not `float NaN`); `SimpleImputer` silently misses `None`; workaround: coerce through `pd.to_numeric(errors="coerce")` first, then `SimpleImputer`
- **Subsample cap**: §6/§7 use a 500k-row stratified subsample of the training set
- **Commit-scoped checkpoints**: fitted pipelines saved to `data/processed/a3_checkpoints/<git-sha>/` via `joblib`; committed hyperparameter changes auto-invalidate the cache; Optuna study persisted alongside at `optuna_study.db` (per-family `study_name`)
- **Optuna tuning**: 9 trials per family (18 total), TPE sampler, `GroupKFold` on subsample years; `recall(1)` stored via `trial.set_user_attr`; gate-aware `select_best_candidate()` picks winner (not Optuna's `study.best_trial`)
- **Output artefacts**: `data/processed/a3_best_model.joblib`, `a3_model_card.json`, `a3_model_comparison.csv`; progress log at `reports/a3_progress.log`

**Key column reference**

| Column       | Type     | Meaning                                           |
|--------------|----------|---------------------------------------------------|
| `UKATGEORIE` | TINYINT  | Target: 1=Getötet, 2=Schwerverletzt, 3=Leicht     |
| `UJAHR`      | SMALLINT | Year 2016–2024                                    |
| `UMONAT`     | TINYINT  | Month 1–12                                        |
| `USTUNDE`    | TINYINT  | Hour 0–23                                         |
| `UWOCHENTAG` | TINYINT  | 1=Sunday … 7=Saturday                             |
| `UART`       | TINYINT  | Accident type 0–9                                 |
| `UTYP1`      | TINYINT  | Accident category 1–7                             |
| `ULICHTVERH` | TINYINT  | 0=daylight, 1=dusk, 2=darkness                    |
| `STRZUSTAND` | TINYINT  | 0=dry, 1=wet/slippery, 2=wintry                   |
| `IstRad`     | BOOLEAN  | Bicycle involved                                  |
| `IstPKW`     | BOOLEAN  | Car involved                                      |
| `IstFuss`    | BOOLEAN  | Pedestrian involved                               |
| `LON`/`LAT`  | DOUBLE   | WGS84 coordinates                                 |

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: git-insights -->
## Git Insights

- Serena MCP + Jupytext workflow configured for symbolic notebook inspection
- Notebook `.py` mirrors regenerated and committed alongside `.ipynb` changes
- Notebook outputs stripped pre-commit via `nbstripout` hook
- `docs/superpowers/` excluded from version control (local dev artefact)
- Raw CSV data is local-only; `data/accidents.parquet` tracked via Git LFS
- AI prompts per QUA³CK phase live at `docs/prompts/` (corrected from earlier `docs/docs/prompts/` typo)
- U-Phase plotting conventions documented in `docs/prompts/02_prompts_phase_u.md`: human-readable label dicts (`FEATURE_LABELS`, `UKATGEORIE_LABELS`, `COL_CODE_LABELS`, etc.) + helpers (`feature_label()`, `severity_label()`, `apply_code_labels()`) sourced from `docs/dataset/DSB_Unfallatlas.md`; consistent `sns.set_theme(style="whitegrid", palette="colorblind")` styling
- `docs/` reorganized into `prompts/`, `course-material/`, `dataset/`, `project/`; `GLOSSARY.md` and `AI TOOL DISCLOSURE.md` stay at `docs/` top level (hard requirements)
- CI (`.github/workflows/ci.yml`): GitHub Actions on ubuntu-latest; installs `uv sync --extra dev --extra geo`, runs `ruff check .` then `uv run pytest`; uploads `coverage.xml` to Codecov via `codecov-action@v5` authenticated with `secrets.CODECOV_TOKEN`
- A³-phase CatBoost fix (commits e7cf9ec/4677517): `class_weights` removed from `build_catboost_pipeline()` constructor to fix `clone()` incompatibility; balanced weighting now applied via `sample_weight` at fit time through `cross_validate(params=...)`
- A³-phase checkpoint pattern: fitted pipelines cached under `data/processed/a3_checkpoints/<git-sha>/` (joblib); Optuna study persisted alongside at `optuna_study.db`; committed hyperparameter changes automatically land in a fresh, empty directory
- A³-phase §6 filter (commit 22d84a3): §2 GroupKFold cell is a standalone sanity check only — §7 Optuna builds its own `GroupKFold` from subsample years; `_build_pipeline_for()` raises `NotImplementedError` for SMOTE/ADASYN/ordinal/threshold strategies so only `{family}_balanced` configs enter Optuna; full comparison table persisted to `data/processed/a3_model_comparison.csv`
- Binary KSI reframe (`feature/binary_ksi_reframe`): 7-task plan at `docs/superpowers/plans/2026-07-14-binary-ksi-reframe.md`; adds §9/§10 to A³ notebook, new library functions across `imbalance.py`, `evaluate.py`, `preprocessing.py`, `boosting.py`, `viz/metrics_viz.py`; `docs/superpowers/` is excluded from version control

<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: best-practices -->
## Best Practices

- Keep `CLAUDE.md` / `AGENTS.md` current — auto-memory updates them on file changes
- Never edit `notebooks/*.py` mirror files directly; they are regenerated by Jupytext
- Avoid data leakage: features must be derivable at accident-report time; no post-hoc columns
- Prefer DuckDB for queries over loading the full parquet into pandas
- Use `uv` (not pip/conda) for all dependency management in this project

<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Custom Notes

Add project-specific notes here. This section is never auto-modified.

<!-- END MANUAL -->
