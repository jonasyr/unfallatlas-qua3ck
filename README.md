# 🚦 Unfallatlas Deutschland — QUA³CK ML-Portfolio

<!-- Design tokens: navy #14213D (badge base) · blue #3776AB · orange #F37626 · mint #06D6A0 · indigo #6D5DFC.
     Reserved for Streamlit K-Phase severity encoding: red #E63946 (Getötet) · amber #F4A261 (Schwerverletzt) · green #2A9D8F (Leicht verletzt) -->

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=plastic&logo=python&logoColor=white&labelColor=14213D)](pyproject.toml)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=plastic&logo=jupyter&logoColor=white&labelColor=14213D)](notebooks/)
[![CI](https://img.shields.io/github/actions/workflow/status/jonasyr/unfallatlas-qua3ck/ci.yml?branch=main&style=plastic&logo=githubactions&logoColor=white&label=CI&labelColor=14213D)](https://github.com/jonasyr/unfallatlas-qua3ck/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/jonasyr/unfallatlas-qua3ck?style=plastic&logo=codecov&logoColor=white&label=Coverage&labelColor=14213D)](https://codecov.io/gh/jonasyr/unfallatlas-qua3ck)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=plastic&labelColor=14213D)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=plastic&labelColor=14213D)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-06D6A0?style=plastic&logo=opensourceinitiative&logoColor=white&labelColor=14213D)](LICENSE)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-jonasyr%2Funfallatlas--qua3ck-6D5DFC?style=plastic&logo=readthedocs&logoColor=white&labelColor=14213D)](https://deepwiki.com/jonasyr/unfallatlas-qua3ck)
<!-- DeepWiki: simple-icons has no dedicated DeepWiki glyph, so readthedocs (closest "docs" icon) stands in -->

<!-- TODO(visual): add a hero image once a result is worth leading with --
     e.g. SHAP summary plot or geo accident-density map from A³/C phase,
     exported as static PNG/SVG to reports/figures/ (not the interactive
     .html exports, those can't be embedded). Drop it here below the badges. -->

Multiclass-Klassifikation der Verkehrsunfallschwere in Deutschland auf Basis des offiziellen Unfallatlas (GovData / Mobilithek), 2016–2024. Universitäts-Portfolioprojekt nach dem **QUA³CK**-Prozessmodell (Question → Understanding → Algorithm/Adapt/Adjust → Conclude & Compare → Knowledge Transfer).

> **Forschungsfrage:** Welche raumzeitlichen, infrastrukturellen und meteorologischen Faktoren bestimmen die Schwere eines Verkehrsunfalls in Deutschland, und lässt sich diese Schwere mit interpretierbaren Machine-Learning-Modellen aus öffentlich verfügbaren Daten zuverlässig vorhersagen?

**📊 Live-Report:** [jonasyr.github.io/unfallatlas-qua3ck](https://jonasyr.github.io/unfallatlas-qua3ck/) — alle Notebooks vollständig gerendert (Code + Outputs) als statisches HTML, ohne lokales Setup.

## Inhalt

- [QUA³CK-Phasen](#qua³ck-phasen)
- [Datensatz](#datensatz)
- [Ziele](#ziele)
- [Projektstruktur](#projektstruktur)
- [Setup](#setup)
- [Tests & Qualitätssicherung](#tests--qualitätssicherung)
- [Notebook-Präsentationen](#notebook-präsentationen)
- [Tech-Stack](#tech-stack)
- [Dokumentation](#dokumentation)
- [Lizenz & Datenquellen](#lizenz--datenquellen)

## QUA³CK-Phasen

| Phase | Notebook | Inhalt |
|-------|----------|--------|
| **Q** — Question | `notebooks/01_Q_Phase.ipynb` | Forschungsfrage, Hypothesen, Erfolgsmetriken, Literatur |
| **U** — Understanding | `notebooks/02_U_Phase.ipynb` | DIG-Description, EDA, Geo-Visualisierung, Feature Engineering |
| **A³** — Algorithm/Adapt/Adjust | `notebooks/03_A3_Phase.ipynb` | Baselines, Boosting-Modelle, Imbalance-Strategien, Optuna-Tuning |
| **C** — Conclude & Compare | `notebooks/04_C_Phase.ipynb` | SHAP, Modellvergleich, Limitationen |
| **K** — Knowledge Transfer | `app/streamlit_app.py` | Interaktive Risikoprofil-App (Streamlit) |

## Datensatz

- **Quelle:** [Unfallatlas auf GovData](https://www.govdata.de/suche/daten/unfallatlas) (Mobilithek), Datenlizenz Deutschland 2.0
- **Zeitraum:** 2016–2024 (9 Jahrgänge)
- **Umfang:** ~2,09 Mio. polizeilich aufgenommene Unfälle mit Personenschaden
- **Zielvariable:** `UKATGEORIE` — 1 = Getötet (1 %), 2 = Schwer verletzt (18 %), 3 = Leicht verletzt (81 %)
- **Format:** Parquet (`data/accidents.parquet`, ~65 MB) — konsolidiert aus den district-level CSV-Dateien, verwaltet via **Git LFS**

## Ziele

| Metrik | Zielwert |
|--------|----------|
| macro-F1 (Held-Out 2024) | ≥ 0.55 |
| Recall Klasse 1 (Getötete) | ≥ 0.50 |
| Basis-Baseline macro-F1 | ~0.30 (Majority Class) |

**Test-Strategie:** Chronologischer Split — Train 2016–2022 · Val 2023 · Test 2024 (keine Zufalls-Splits, da zeitreihenähnliche Daten).

## Projektstruktur

```
unfallatlas-qua3ck/
├── notebooks/          # QUA³CK-Phasennotebooks (Source of Truth)
├── src/unfallatlas/    # Wiederverwendbare Bibliothek (data/, features/, models/, viz/, presentation/)
├── app/                # Streamlit-Demo (K-Phase)
├── data/                # accidents.parquet (Git LFS) + interim/processed Artefakte
├── tests/               # pytest-Suite
├── docs/                # Disclosure, Glossar, Datensatzbeschreibung, Prozessdokumentation
├── reports/             # Generierte Figures + Notebook-Präsentationen
└── pyproject.toml       # Projektkonfiguration (hatchling, ruff, pytest, jupytext)
```

## Setup

Für die lokale Einrichtung auf einem neuen Rechner (z. B. zur Begutachtung) in vier Schritten:

### 1 — Git LFS installieren

`data/accidents.parquet` wird über Git LFS verwaltet. Ohne LFS enthält die Datei nur einen 133-Byte-Pointer und DuckDB wirft `No magic bytes found`.

| Plattform | Befehl |
|-----------|--------|
| Arch Linux | `sudo pacman -S git-lfs` |
| Ubuntu / Debian | `sudo apt install git-lfs` |
| macOS (Homebrew) | `brew install git-lfs` |
| Windows (winget) | `winget install GitHub.GitLFS` |

```bash
# LFS einmalig im Git-Profil registrieren
git lfs install

# Datei herunterladen (nach dem Klonen oder wenn data/ nur einen Pointer enthält)
git lfs pull
```

### 2 — uv installieren

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3 — Python-Abhängigkeiten

```bash
uv sync --all-extras
```

Voraussetzung: Python ≥ 3.11 (wird von `uv` automatisch verwaltet).

### 4 — Notebooks starten

```bash
uv run jupyter lab
```

### Pre-commit Hooks (optional, für Contributor)

```bash
uv run pre-commit install
```

## Tests & Qualitätssicherung

```bash
uv run pytest              # Testsuite inkl. Coverage-Report (siehe pyproject.toml)
uv run ruff check .        # Linting
uv run ruff format .       # Formatierung
pre-commit run --all-files # Alle Hooks (Ruff, nbstripout, Commitizen, u. a.)
```

CI führt dieselben Schritte auf jedem Push/PR über GitHub Actions aus (Badge oben).

## Notebook-Präsentationen

Bereits ausgeführte Notebooks lassen sich ohne erneute Berechnung als offlinefähige
HTML-Snapshots unter `reports/presentation/` exportieren:

```bash
uv sync --extra presentation
uv run python scripts/export_notebooks.py --all
```

Der Export verwendet nur gespeicherte Outputs. Installation, Validierung, Strict-Modus,
Offline-Kopie, PDF und GitHub Pages beschreibt der
[Leitfaden zum Präsentationsexport](docs/presentation-export.md).

## Tech-Stack

| Bereich | Bibliotheken |
|---------|--------------|
| Daten | pandas, polars, duckdb, pyarrow |
| ML | scikit-learn, xgboost, lightgbm, catboost |
| Imbalance & Tuning | imbalanced-learn, optuna |
| Explainability | shap |
| Geospatial | folium, streamlit-folium (optional: geopandas, h3, osmnx) |
| App | streamlit |
| Tooling | uv, ruff, pytest, jupytext, pre-commit |

## Dokumentation

- [GLOSSARY.md](docs/GLOSSARY.md) — Glossar der Fachbegriffe und Spaltenbezeichnungen
- [AI TOOL DISCLOSURE.md](docs/AI%20TOOL%20DISCLOSURE.md) — Offenlegung der verwendeten KI-Tools je QUA³CK-Phase
- [docs/prompts/](docs/prompts/) — vollständige KI-Prompt-Transkripte je Phase
- [docs/dataset/](docs/dataset/) — Datensatzbeschreibung (DSB_Unfallatlas)
- [docs/course-material/](docs/course-material/) — Kursunterlagen als KI-Kontext
- [docs/project/](docs/project/) — Repo-/Prozessdokumentation (Conventional Commits, Projektplan)

## Lizenz & Datenquellen

- **Code:** [MIT](LICENSE)
- **Unfalldaten:** [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0) · Quelle: Mobilithek / Statistisches Bundesamt, Unfallatlas Deutschland
- **Wetterdaten:** Deutscher Wetterdienst (DWD), CDC Open Data
- **Straßennetz:** © OpenStreetMap-Mitwirkende, ODbL

---

Autor: [@jonasyr](https://github.com/jonasyr) · Universitäts-Portfolioprojekt (Data Analytics / Big Data)
