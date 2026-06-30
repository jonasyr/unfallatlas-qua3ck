# Unfallatlas Deutschland — QUA³CK ML-Portfolio

[![CI](https://img.shields.io/github/actions/workflow/status/jonasyr/unfallatlas-qua3ck/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/jonasyr/unfallatlas-qua3ck/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/jonasyr/unfallatlas-qua3ck?style=flat-square&logo=codecov&logoColor=white&label=coverage)](https://codecov.io/gh/jonasyr/unfallatlas-qua3ck)
[![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square)](https://github.com/astral-sh/uv)

<!-- TODO(visual): add a hero image once a result is worth leading with --
     e.g. SHAP summary plot or geo accident-density map from A³/C phase,
     exported as static PNG/SVG to reports/figures/ (not the interactive
     .html exports, those can't be embedded). Drop it here below the badges. -->

Multiclass-Klassifikation der Verkehrsunfallschwere in Deutschland auf Basis des offiziellen Unfallatlas (GovData / Mobilithek), 2016–2024.

> **Forschungsfrage:** Welche raumzeitlichen, infrastrukturellen und meteorologischen Faktoren bestimmen die Schwere eines Verkehrsunfalls in Deutschland, und lässt sich diese Schwere mit interpretierbaren Machine-Learning-Modellen aus öffentlich verfügbaren Daten zuverlässig vorhersagen?

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
- **Zielvariable:** `UKATGEORIE` — 1 = Getötet (1%), 2 = Schwer verletzt (18%), 3 = Leicht verletzt (81%)
- **Format:** Parquet (`data/accidents.parquet`, ~65 MB) — konsolidiert aus den district-level CSV-Dateien, gespeichert via **Git LFS**

## Setup

### 1 — Git LFS installieren

`data/accidents.parquet` wird über Git LFS verwaltet. Ohne LFS enthält die Datei nur einen 133-Byte-Pointer und DuckDB wirft `No magic bytes found`.

| Plattform | Befehl |
|-----------|--------|
| Arch Linux | `sudo pacman -S git-lfs` |
| Ubuntu / Debian | `sudo apt install git-lfs` |
| macOS (Homebrew) | `brew install git-lfs` |
| Windows (winget) | `winget install GitHub.GitLFS` |

```bash
# LFS einmalig in deinem Git-Profil registrieren
git lfs install

# Datei herunterladen (nach dem Klonen oder wenn data/ nur Pointer enthält)
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

### 4 — Notebooks starten

```bash
uv run jupyter lab
```

Voraussetzungen: Python ≥ 3.11 (wird von uv automatisch verwaltet) · git-lfs (siehe Schritt 1)

### Pre-commit Hooks

```bash
uv run pre-commit install
```

## Ziele

| Metrik | Zielwert |
|--------|----------|
| macro-F1 (Held-Out 2024) | ≥ 0.55 |
| Recall Klasse 1 (Getötete) | ≥ 0.50 |
| Basis-Baseline macro-F1 | ~0.30 (Majority Class) |

**Test-Strategie:** Chronologischer Split — Train 2016–2022 · Val 2023 · Test 2024.

## Dokumentation

- [GLOSSARY.md](docs/GLOSSARY.md) — Glossar der Fachbegriffe und Spaltenbezeichnungen
- [AI TOOL DISCLOSURE.md](docs/AI%20TOOL%20DISCLOSURE.md) — Offenlegung der verwendeten KI-Tools je QUA³CK-Phase
- [docs/prompts/](docs/prompts/) — vollständige KI-Prompt-Transkripte je Phase
- [docs/dataset/](docs/dataset/) — Datensatzbeschreibung (DSB_Unfallatlas)
- [docs/course-material/](docs/course-material/) — Kursunterlagen als KI-Kontext
- [docs/project/](docs/project/) — Repo-/Prozessdokumentation (Conventional Commits, Projektplan)

## Lizenz

Code: [MIT](LICENSE). Daten: [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0). Quellenangabe: Mobilithek / Statistisches Bundesamt, Unfallatlas Deutschland.
