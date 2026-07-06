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
# Install all dependencies (including dev extras)
uv sync --all-extras

# Run tests
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
│   ├── data/               # download.py, dwd.py (weather), osm.py
│   ├── features/           # enrich.py, spatial.py, temporal.py, preprocessing.py
│   ├── models/             # baseline.py, boosting.py, evaluate.py, ordinal.py, imbalance.py
│   └── viz/                # geo.py, shap_plots.py, streamlit_app.py
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
│   └── project/            # Repo/process docs (ConventionalCommitsGuide.md, PROJEKTPLAN_SETUP.md)
├── reports/figures/        # Generated output figures
└── pyproject.toml          # Project config (hatchling, ruff, black, jupytext)
```

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
