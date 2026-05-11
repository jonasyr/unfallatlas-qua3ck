# Unfallatlas Deutschland — QUA³CK ML-Portfolio

Multiclass-Klassifikation der Verkehrsunfallschwere in Deutschland auf Basis des offiziellen Unfallatlas (GovData / Mobilithek), 2016–2024.

> **Forschungsfrage:** Welche raumzeitlichen, infrastrukturellen und meteorologischen Faktoren bestimmen die Schwere eines Verkehrsunfalls in Deutschland, und lässt sich diese Schwere mit interpretierbaren Machine-Learning-Modellen aus öffentlich verfügbaren Daten zuverlässig vorhersagen?

---

## QUA³CK-Phasen

| Phase | Notebook | Inhalt |
|-------|----------|--------|
| **Q** — Question | `notebooks/01_Q_Phase.ipynb` | Forschungsfrage, Hypothesen, Erfolgsmetriken, Literatur |
| **U** — Understanding | `notebooks/02_U_Phase.ipynb` | DIG-Description, EDA, Geo-Visualisierung, Feature Engineering |
| **A³** — Algorithm/Adapt/Adjust | `notebooks/03_A3_Phase.ipynb` | Baselines, Boosting-Modelle, Imbalance-Strategien, Optuna-Tuning |
| **C** — Conclude & Compare | `notebooks/04_C_Phase.ipynb` | SHAP, Modellvergleich, Limitationen |
| **K** — Knowledge Transfer | `app/streamlit_app.py` | Interaktive Risikoprofil-App (Streamlit) |

---

## Datensatz

- **Quelle:** [Unfallatlas auf GovData](https://www.govdata.de/suche/daten/unfallatlas) (Mobilithek), Datenlizenz Deutschland 2.0
- **Zeitraum:** 2016–2024 (9 Jahrgänge)
- **Umfang:** ~2,09 Mio. polizeilich aufgenommene Unfälle mit Personenschaden
- **Zielvariable:** `UKATGEORIE` — 1 = Getötet (1%), 2 = Schwer verletzt (18%), 3 = Leicht verletzt (81%)
- **Format:** Parquet (`data/body.parquet`) — konsolidiert aus den district-level CSV-Dateien

Die Rohdaten (`data/`) sind gitignored. Download über die offizielle Mobilithek-API oder MFDZ-Mirror.

---

## Setup

```bash
# Abhängigkeiten installieren (uv empfohlen)
uv sync

# Mit optionalen Geo-Packages
uv sync --extra geo

# Jupyter starten
uv run jupyter lab
```

### Voraussetzungen

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) oder pip

### Pre-commit Hooks

```bash
uv run pre-commit install
```

---

## Ziele

| Metrik | Zielwert |
|--------|----------|
| macro-F1 (Held-Out 2024) | ≥ 0.55 |
| Recall Klasse 1 (Getötete) | ≥ 0.50 |
| Basis-Baseline macro-F1 | ~0.30 (Majority Class) |

**Test-Strategie:** Chronologischer Split — Train 2016–2022 · Val 2023 · Test 2024.

---

## Lizenz

Code: MIT. Daten: [Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0). Quellenangabe: Mobilithek / Statistisches Bundesamt, Unfallatlas Deutschland.
