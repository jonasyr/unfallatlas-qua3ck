<!-- Design tokens (aligned with reports/presentation UI, see presentation.css):
     steel blue #315F7D (accent) · dark steel #24475F (badge base) · matching notebook charts (COLOR_PRIMARY).
     Reserved for Streamlit K-Phase severity encoding: red #E63946 (Killed) · amber #F4A261 (Seriously injured) · green #2A9D8F (Slightly injured) -->

<div align="center">

# 🚦 Unfallatlas Germany

**Traffic accident severity classification using the QUA³CK process model**

Multiclass classification of traffic accident severity in Germany based on the official
Unfallatlas (GovData / Mobilithek), 2016-2024. University portfolio project following **QUA³CK**
(Question → Understanding → Algorithm/Adapt/Adjust → Conclude & Compare → Knowledge Transfer).

[![Live-Report](https://img.shields.io/badge/Live--Report-jonasyr.github.io%2Funfallatlas--qua3ck-315F7D?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=24475F)](https://jonasyr.github.io/unfallatlas-qua3ck/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-315F7D?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=24475F)](https://unfallatlas-qua3ck-fg5thv4lthkbhw86er3kfk.streamlit.app)

[![Python](https://img.shields.io/badge/Python-3.11%2B-315F7D?style=plastic&logo=python&logoColor=white&labelColor=24475F)](pyproject.toml)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-315F7D?style=plastic&logo=jupyter&logoColor=white&labelColor=24475F)](notebooks/)
[![CI](https://img.shields.io/github/actions/workflow/status/jonasyr/unfallatlas-qua3ck/ci.yml?branch=main&style=plastic&logo=githubactions&logoColor=white&label=CI&labelColor=24475F)](https://github.com/jonasyr/unfallatlas-qua3ck/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/jonasyr/unfallatlas-qua3ck?style=plastic&logo=codecov&logoColor=white&label=Coverage&labelColor=24475F)](https://codecov.io/gh/jonasyr/unfallatlas-qua3ck)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=plastic&labelColor=24475F)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=plastic&labelColor=24475F)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-315F7D?style=plastic&logo=opensourceinitiative&logoColor=white&labelColor=24475F)](LICENSE)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-jonasyr%2Funfallatlas--qua3ck-315F7D?style=plastic&logo=readthedocs&logoColor=white&labelColor=24475F)](https://deepwiki.com/jonasyr/unfallatlas-qua3ck)
<!-- DeepWiki: simple-icons has no dedicated DeepWiki glyph, so readthedocs (closest "docs" icon) stands in -->

</div>

<!-- TODO(visual): add a hero image once a result is worth leading with --
     e.g. SHAP summary plot or geo accident-density map from A³/C phase,
     exported as static PNG/SVG to reports/figures/ (not the interactive
     .html exports, those can't be embedded). Drop it here below the badges. -->

> **Research question:** Which spatiotemporal, infrastructural, and meteorological factors
> determine the severity of a traffic accident in Germany, and can this severity be reliably
> predicted from publicly available data using interpretable machine learning models?

The [Live Report](https://jonasyr.github.io/unfallatlas-qua3ck/) renders all notebooks in full
(code + interactive Plotly outputs) as offline-capable HTML - no local setup required.

## Contents

- [QUA³CK Phases](#qua³ck-phases)
- [Dataset](#dataset)
- [Goals](#goals)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Tests & Quality Assurance](#tests--quality-assurance)
- [Notebook Presentations](#notebook-presentations)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [Related Projects](#related-projects)
- [License & Data Sources](#license--data-sources)

## QUA³CK Phases

| | Phase | Notebook | Content |
|---|-------|----------|--------|
| **Q** | Question | [`01_Q_Phase.ipynb`](notebooks/01_Q_Phase.ipynb) · [HTML](https://jonasyr.github.io/unfallatlas-qua3ck/notebooks/01_Q_Phase.html) | Research question, hypotheses, success metrics, literature |
| **U** | Understanding the Data | [`02_U_Phase.ipynb`](notebooks/02_U_Phase.ipynb) · [HTML](https://jonasyr.github.io/unfallatlas-qua3ck/notebooks/02_U_Phase.html) | DIG description, EDA, geo-visualization, feature engineering |
| **A³** | Algorithm / Adapt / Adjust | [`03_A3_Phase.ipynb`](notebooks/03_A3_Phase.ipynb) · [HTML](https://jonasyr.github.io/unfallatlas-qua3ck/notebooks/03_A3_Phase.html) | Baselines, boosting models, imbalance strategies, Optuna tuning |
| **C** | Conclude & Compare | [`04_C_Phase.ipynb`](notebooks/04_C_Phase.ipynb) · [HTML](https://jonasyr.github.io/unfallatlas-qua3ck/notebooks/04_C_Phase.html) | SHAP, model comparison, limitations |
| **K** | Knowledge Transfer | [`app/streamlit_app.py`](app/streamlit_app.py) | Interactive risk-profile app (Streamlit) |

## Live Deployment

- **Streamlit app:** https://unfallatlas-qua3ck-fg5thv4lthkbhw86er3kfk.streamlit.app (deployed per `AGENTS.md`'s "Deploying to Streamlit Community Cloud" section) - interactive risk predictor, model comparison, and severity map.
- **Notebook presentations (GitHub Pages):** the full Q/U/A³/C phase notebooks, rendered as static HTML, are linked from the Streamlit app's Overview page. The notebook-presentation site links back to the Streamlit app once the next notebook re-export picks up the updated template (`src/unfallatlas/presentation/templates/site_index.html.j2`) - this has not happened yet as of this commit.

## Dataset

| | |
|---|---|
| **Source** | [Unfallatlas on GovData](https://www.govdata.de/suche/daten/unfallatlas) (Mobilithek), Data License Germany 2.0 |
| **Period** | 2016-2024 (9 years) |
| **Scope** | ~2.09M police-recorded accidents involving personal injury |
| **Target variable** | `UKATGEORIE` - 1 = Killed (1%), 2 = Seriously injured (18%), 3 = Slightly injured (81%) |
| **Format** | Parquet (`data/accidents.parquet`, ~65 MB) - consolidated from the district-level CSV files, tracked via **Git LFS** |

## Goals

| Metric | Target |
|--------|----------|
| macro-F1 (held-out 2024) | ≥ 0.55 |
| Recall class 1 (killed) | ≥ 0.50 |
| Baseline macro-F1 | ~0.30 (majority class) |

**Test strategy:** Chronological split - Train 2016-2022 · Val 2023 · Test 2024
(no random splits, since the data is time-series-like).

## Project Structure

```
unfallatlas-qua3ck/
├── notebooks/          # QUA³CK phase notebooks (source of truth)
├── src/unfallatlas/    # Reusable library (data/, features/, models/, viz/, presentation/)
├── app/                # Streamlit demo (K phase)
├── data/               # accidents.parquet (Git LFS) + interim/processed artifacts
├── tests/              # pytest suite
├── docs/               # Disclosure, glossary, dataset description, process documentation
├── reports/            # Generated figures + notebook presentations
└── pyproject.toml      # Project configuration (hatchling, ruff, pytest, jupytext)
```

## Setup

Local setup on a new machine (e.g. for review):

```bash
git lfs install && git lfs pull   # 1 - fetch the dataset via Git LFS
curl -LsSf https://astral.sh/uv/install.sh | sh   # 2 - install uv
uv sync --all-extras              # 3 - Python dependencies (Python >= 3.11, managed by uv)
uv run jupyter lab                # 4a - launch the notebooks (optional)
uv run streamlit run app/streamlit_app.py   # 4b - launch the Streamlit app (no notebook run needed)
```

Step 4b opens the interactive K-phase app (Risk Predictor, Model Comparison,
Overview map) at `http://localhost:8501` - fully self-contained, with no notebook
needing to run first. Every artifact it needs (models, contract, map data) is
already committed to the repo via Git LFS/Git.

> [!IMPORTANT]
> `data/accidents.parquet` is tracked via Git LFS. Without LFS the file only contains a
> 133-byte pointer, and DuckDB raises `No magic bytes found`.

<details>
<summary><strong>Install Git LFS</strong> (platform-specific)</summary>

| Platform | Command |
|-----------|--------|
| Arch Linux | `sudo pacman -S git-lfs` |
| Ubuntu / Debian | `sudo apt install git-lfs` |
| macOS (Homebrew) | `brew install git-lfs` |
| Windows (winget) | `winget install GitHub.GitLFS` |

</details>

<details>
<summary><strong>Pre-commit hooks</strong> (optional, for contributors)</summary>

```bash
uv run pre-commit install
```

</details>

## Tests & Quality Assurance

```bash
uv run pytest              # Test suite incl. coverage report (see pyproject.toml)
uv run ruff check .        # Linting
uv run ruff format .       # Formatting
pre-commit run --all-files # All hooks (Ruff, nbstripout, Commitizen, etc.)
```

CI runs the same steps on every push/PR via GitHub Actions (badge above).

## Notebook Presentations

Already-executed notebooks can be exported as offline-capable HTML snapshots under
`reports/presentation/` without re-running anything - in the same design as the
[Live Report](https://jonasyr.github.io/unfallatlas-qua3ck/) (IBM Plex, light/dark mode,
page zoom, interactive Plotly charts):

```bash
uv sync --extra presentation
uv run python scripts/export_notebooks.py --all
```

The export uses only saved outputs. Installation, validation, strict mode, the offline
copy, PDF, and GitHub Pages are described in the
[presentation export guide](docs/presentation-export.md).

## Tech Stack

| Area | Libraries |
|---------|--------------|
| Data | pandas, polars, duckdb, pyarrow |
| ML | scikit-learn, xgboost, lightgbm, catboost |
| Imbalance & Tuning | imbalanced-learn, optuna |
| Explainability | shap |
| Geospatial | folium, streamlit-folium (optional: geopandas, h3, osmnx) |
| App | streamlit |
| Tooling | uv, ruff, pytest, jupytext, pre-commit |

## Documentation

- [GLOSSARY.md](docs/GLOSSARY.md) - glossary of technical terms and column names
- [AI TOOL DISCLOSURE.md](docs/AI%20TOOL%20DISCLOSURE.md) - disclosure of AI tools used per QUA³CK phase
- [docs/prompts/](docs/prompts/) - full AI prompt transcripts per phase
- [docs/dataset/](docs/dataset/) - dataset description (DSB_Unfallatlas)
- [docs/course-material/](docs/course-material/) - course materials as AI context
- [docs/project/](docs/project/) - repo/process documentation (Conventional Commits, project plan)

## Related Projects

- [EnergyCast-App](https://github.com/NiklasSkulll/EnergyCast-App) - related notebook portfolio project:
  forecasting electricity demand from weather data, solar/wind feed-in, and temporal patterns
- [Degrees-of-No-Return-App](https://github.com/noahrsn/Degrees-of-No-Return-App) - related notebook
  portfolio project: translating global climate models into local risk profiles, predicting
  local heat-day counts and flood risk through 2050

## License & Data Sources

- **Code:** [MIT](LICENSE)
- **Accident data:** [Data License Germany – Attribution – Version 2.0](https://www.govdata.de/dl-de/by-2-0) · Source: Mobilithek / Federal Statistical Office (Destatis), Unfallatlas Germany
- **Weather data:** German Weather Service (DWD), CDC Open Data
- **Road network:** © OpenStreetMap contributors, ODbL

---

<div align="center">

[@jonasyr](https://github.com/jonasyr) · University portfolio project (Data Analytics / Big Data)

</div>
