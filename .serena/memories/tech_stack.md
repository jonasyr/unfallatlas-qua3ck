# Tech Stack

- **Language**: Python 3.11+ (3.13 in active venv)
- **Package manager**: `uv` (lockfile: `uv.lock`); venv at `.venv/`
- **Build backend**: hatchling; installed as editable package `unfallatlas`
- **Notebooks**: JupyterLab 4 via `uv run jupyter lab`

## Core libraries

| Purpose | Library |
|---------|---------|
| Tabular data | pandas 2.2+, polars 1+ |
| Fast queries | duckdb 1.1+ |
| ML | scikit-learn 1.4+, xgboost 2+, lightgbm 4.3+, catboost 1.2+ |
| Imbalance | imbalanced-learn 0.12+ |
| Tuning | optuna 3.6+ |
| Explainability | shap 0.45+ |
| Geospatial | folium 0.17+, (h3, osmnx in optional `geo` extra) |
| App | streamlit 1.35+, streamlit-folium 0.20+ |
| Weather data | requests, scipy (cKDTree for nearest station) |

## Tooling

| Tool | Version pinned | Purpose |
|------|---------------|---------|
| ruff | 0.4.x | lint + format |
| black | 24+ | secondary formatter |
| pytest | 8+ | tests |
| jupytext | 1.19+ | sync `.ipynb` ↔ `.py` percent mirrors for Serena |
| nbstripout | 0.7.x | strip notebook outputs pre-commit |
| pre-commit | — | hooks (ruff, nbstripout, commitizen, detect-private-key, check-large-files≤5MB, pyproject-fmt, nbqa-ruff) |
| commitizen | 3.27.x | conventional commits enforcement |
