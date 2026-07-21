# Core — unfallatlas-qua3ck

ML-Portfolio: multiclass classification of German traffic-accident severity (`UKATGEORIE` 1/2/3) on the Unfallatlas dataset 2016–2024 (~2.09M rows, 21 cols). Follows the QUA³CK data-science process framework.

## Source map

```
data/accidents.parquet               Main dataset (gitignored large file)
data/interim/accidents_with_weather.parquet   Weather-enriched interim data
data/raw/unfalldaten/                Raw CSVs from BASt
data/raw/dwd/                        Cached DWD weather downloads
data/processed/                      Model artefacts (gitignored)
reports/figures/                     Saved plots

src/unfallatlas/                     Installed Python package
  data/dwd.py                        DWD CDC weather (done)
  data/download.py                   Stub (0 lines)
  data/osm.py                        OSM road-network fetch/cache/tiling pipeline (done, 753 lines)
  features/enrich.py                 Stub (0 lines)
  features/temporal.py               Small helper module (26 lines)
  features/spatial.py                H3/OSM aggregation (done, 184 lines)
  features/preprocessing.py          Feature/target split incl. binary variant (done, 223 lines)
  models/baseline.py                 Done (38 lines)
  models/boosting.py                 LightGBM/XGBoost/CatBoost pipelines incl. binary (done, 309 lines)
  models/evaluate.py                 Champion selection + binary metrics (done, 123 lines)
  models/imbalance.py                SMOTE/ADASYN/offset-sweep gate logic (done, 105 lines)
  models/ordinal.py                  Done (65 lines)
  models/svm.py                      SVM candidate family added for binary champion search (done, 101 lines)
  viz/geo.py                         Stub (0 lines)
  viz/metrics_viz.py                 Done (142 lines)
  viz/shap_plots.py                  Stub (0 lines)
  viz/streamlit_app.py               Stub (0 lines)
  presentation/                      Notebook→HTML exporter package (~2500 lines): assets.py, cli.py,
                                      discovery.py, manifest.py, metadata.py, models.py, rendering.py,
                                      validation.py — see docs/presentation-export.md; CLI entry via
                                      `scripts/export_notebooks.py`

notebooks/
  01_Q_Phase.ipynb                   Research question + hypotheses (DONE)
  02_U_Phase.ipynb                   EDA + feature engineering (DONE)
  03_A3_Phase.ipynb                  Modelling + tuning, incl. binary KSI reframe (DONE)
  04_C_Phase.ipynb                   Comparison + SHAP + conclusion (TODO)

app/streamlit_app.py                 Streamlit demo (still TODO — 0 bytes, verified)
tests/test_features.py               Nearly empty (stub, 0 lines) — but tests/ overall has grown to
                                      ~25 files incl. tests/presentation/ (12 files, ~4000 lines) and
                                      per-module test files for osm/spatial/imbalance/svm/etc.
```

## Critical data invariants

- **Typo**: target column is `UKATGEORIE` (not `UKATEGORIE`) — matches stored Parquet schema.
- `ULAND` absent in Parquet; derive: `df["ULAND"] = df["UKREIS"].str[:2].astype(int)`.
- Raw CSVs: `sep=";"`, `decimal=","`, `encoding="utf-8-sig"`.
- Class imbalance: 1% / 18% / 81% → always stratified splits, **macro-F1** as primary metric.
- **Chronological split only** — never random: train ≤ 2022, val = 2023, test = 2024.
- `UWOCHENTAG`: 1=Sunday, 2=Monday, …, 7=Saturday.

## Loading data

```python
import duckdb
from pathlib import Path
DATA = Path("data/accidents.parquet")
df = duckdb.connect().execute(f"SELECT * FROM '{DATA}' WHERE UJAHR = 2024").df()
```

## Notebook policy

`notebooks/*.ipynb` are the **source of truth**. Paired `notebooks/*.py` files are generated Jupytext/Serena mirrors — read-only for symbolic inspection. **Never edit `.py` mirrors directly.**

- Edit the `.ipynb`; then sync: `uv run jupytext --sync notebooks/*.ipynb && serena project index`
- The pre-commit hook `scripts/check_notebook_mirrors.py` blocks commits where a `.py` mirror changed without its `.ipynb` counterpart.
- Reusable logic extracted from notebooks belongs in `src/unfallatlas/`, imported back into the notebook.

See `mem:tech_stack` for dependencies, `mem:conventions` for code style, `mem:suggested_commands` for CLI usage, `mem:task_completion` for done checklist, `mem:documentation` for the `docs/` folder layout.
