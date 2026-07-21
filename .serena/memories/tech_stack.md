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
| Plotting | plotly 5.22+ (used by presentation exporter) |

## Tooling

| Tool | Version pinned | Purpose |
|------|---------------|---------|
| ruff | 0.4+ (pre-commit pins ruff-pre-commit v0.15.20) | lint + primary formatter (`ruff format`) |
| black | 24+ | vestigial — `[tool.black]` config remains in pyproject.toml but ruff format is the formatter actually run in pre-commit/CI |
| pytest | 8+ | tests; `browser` marker (opt-in Playwright checks) excluded by default via `-m "not browser"` in addopts |
| jupytext | 1.19+ | sync `.ipynb` ↔ `.py` percent mirrors for Serena |
| nbstripout | 0.7.1 | strip notebook outputs pre-commit (excludes `tests/presentation/fixtures/`) |
| pre-commit | — | hooks: ruff --fix, ruff-format, nbstripout, detect-private-key, check-added-large-files (≤5MB), pyproject-fmt, commitizen (commit-msg), nbqa-ruff, local `check-notebook-mirrors` |
| commitizen | 3.27.x | conventional commits enforcement |
| playwright | 1.52+ | optional `presentation-test` extra; drives `browser`-marked tests against exported presentation HTML |
| nbconvert / beautifulsoup4 | 7.17+ / 4.12+ | optional `presentation` extra; notebook→HTML export pipeline |
