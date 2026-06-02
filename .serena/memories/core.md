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
  data/dwd.py                        DWD CDC weather — only fully implemented module
  data/download.py                   Stub
  data/osm.py                        Stub
  features/enrich.py                 Stub
  features/temporal.py               Stub
  features/spatial.py                Stub
  models/baseline.py                 Stub
  models/boosting.py                 Stub
  models/evaluate.py                 Stub
  models/ordinal.py                  Stub
  viz/geo.py                         Stub
  viz/shap_plots.py                  Stub
  viz/streamlit_app.py               Stub

notebooks/
  01_Q_Phase.ipynb                   Research question + hypotheses (DONE)
  02_U_Phase.ipynb                   EDA + feature engineering (TODO)
  03_A3_Phase.ipynb                  Modelling + tuning (TODO)
  04_C_Phase.ipynb                   Comparison + SHAP + conclusion (TODO)

app/streamlit_app.py                 Streamlit demo (TODO)
tests/test_features.py               Nearly empty (stub)
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

See `mem:tech_stack` for dependencies, `mem:conventions` for code style, `mem:suggested_commands` for CLI usage, `mem:task_completion` for done checklist.
