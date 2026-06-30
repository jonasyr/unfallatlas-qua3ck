---
title: Projektplan – Unfallatlas Deutschland (QUA³CK ML-Portfolio)
date: 05-05-2026
category:
  - project-plan
  - qua3ck
  - ml-portfolio
status:
  - active
---

# Projektplan: Unfallatlas Deutschland

## Bestätigung des Datensatzes

Ja, exakt der richtige. Dein Screenshot zeigt den offiziellen Eintrag auf GovData (`govdata.de/suche/daten/unfallatlas`), bereitgestellt von Mobilithek, mit den Ressourcen für 2024:

- `Download Unfallorte 2024 - shapefile (zip)`
- `Download Unfallorte 2024 - CSV-Format (zip)`
- WMS-Service für direkte Karteneinbindung
- Metadaten als PDF
- Letzte Änderung 31.10.2025 (also brandaktuell)

GovData ist Deutschlands nationales Open-Data-Portal und wird vollständig von data.europa.eu indexiert — damit erfüllt der Datensatz dein Quellen-Kriterium aus den Kursnotizen vollständig. Lizenz: Datenlizenz Deutschland 2.0 (Namensnennung), kompatibel zu CC-BY.

Zusätzlich verifiziert über die Spaltendokumentation aus mehreren GovData-Mirrors: das Schema ist seit 2018 stabil mit kleinen Änderungen, wir wissen also genau, mit was wir arbeiten.

---

## Übergeordnete Forschungsfrage

> **Welche raumzeitlichen, infrastrukturellen und meteorologischen Faktoren bestimmen die Schwere eines Verkehrsunfalls in Deutschland, und lässt sich diese Schwere mit interpretierbaren Machine-Learning-Modellen aus öffentlich verfügbaren Daten zuverlässig vorhersagen?**

Diese Frage ist deshalb gut, weil sie:

- ein klares ML-Target hat (Unfallkategorie 1/2/3 — getötet/schwer/leicht),
- wissenschaftlich anschlussfähig ist (peer-reviewed Literatur 2022–2025 nutzt exakt diesen Ansatz: XGBoost/LightGBM/CatBoost + SHAP + SMOTE),
- gesellschaftliche Relevanz hat (EU "Vision Zero" 2050),
- regional zugespitzt werden kann (Wiesbaden/Hessen als Fokus),
- nicht trivial mit "letzter Wert" oder Mittelwert-Baseline zu schlagen ist.

---

## Datenarchitektur: vier Quellen, klare Joins

### Quelle 1: Unfallatlas (Hauptdatensatz)

- **Plattform:** GovData (`govdata.de`), indexiert auf data.europa.eu
- **Mirror für Multi-Year-Konsolidierung:** MITFAHR|DE|ZENTRALE (`data.mfdz.de/destatis_Unfalldaten/`)
- **Direkt-URL Statistikportal:** `unfallatlas.statistikportal.de`
- **Format:** Shapefile + CSV pro Jahr, 2016–2024 (9 Jahre)
- **Volumen:** ca. 250.000–300.000 Unfälle/Jahr × 9 = ~2,5 Mio. Records
- **Schlüsselspalten (verifiziert):**

| Spalte | Bedeutung | Codierung |
|---|---|---|
| `OBJECTID` | eindeutige Unfall-ID | Integer |
| `ULAND` | Bundesland | 01–16 |
| `UREGBEZ`, `UKREIS`, `UGEMEINDE` | Verwaltungsebenen | numerisch |
| `UJAHR`, `UMONAT`, `USTUNDE`, `UWOCHENTAG` | Zeit | 2016–2024 / 1–12 / 0–23 / 1–7 |
| `UKATEGORIE` | **Zielvariable** | 1=Getötet, 2=Schwer, 3=Leicht |
| `UART` | Unfallart | 0–9 (10 Klassen) |
| `UTYP1` | Unfalltyp | 1–7 |
| `ULICHTVERH` | Lichtverhältnisse | 0=Tageslicht, 1=Dämmerung, 2=Dunkelheit |
| `STRZUSTAND` | Straßenzustand | 0=trocken, 1=nass/feucht, 2=winterglatt |
| `IstRad`, `IstPKW`, `IstFuss`, `IstKrad`, `IstGkfz`, `IstSonstig` | Beteiligte Verkehrsmittel | binär |
| `XGCSWGS84`, `YGCSWGS84` | Geokoordinaten WGS84 | Dezimalgrad (mit Komma als Trennzeichen!) |
| `LINREFX`, `LINREFY` | UTM-Koordinaten | ETRS89 Zone 32N |

**Bekannte Stolpersteine, die du in der U-Phase adressieren musst:**

- Geokoordinaten verwenden **Komma** als Dezimaltrennzeichen → `pd.read_csv(..., sep=";", decimal=",")`.
- Spalte `STRZUSTAND` hieß bis 2017 `IstStrasse`, `ULICHTVERH` hieß `LICHT` — beim Multi-Year-Join harmonisieren.
- Nur Unfälle mit ≥92 % Geokoordinaten-Trefferquote werden veröffentlicht (NRW-Pressemeldung 2024) — Selektionsbias ist im Datensatz eingebaut, das musst du in der DIG-Introspektion explizit dokumentieren.
- Nur Unfälle mit Personenschaden enthalten — Sachschadensunfälle fehlen vollständig.

### Quelle 2: DWD Climate Data Center (Wetter)

- **Plattform:** `opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/`
- **Auffindbarkeit:** indirekt über Google Dataset Search; auch als HVD ("High-Value Dataset") auf GovData gelistet
- **Parameter, die du brauchst:**
  - `air_temperature` (Stündliche Werte)
  - `precipitation`
  - `visibility`
  - `wind`
  - `weather_phenomena` (Glatteis, Nebel, Schnee, Sturm)
- **Stationsabdeckung:** ~400 aktive Klimastationen + ~2000 Niederschlagsstationen
- **Volumen:** historische Stundenwerte gehen bis 1937 zurück; für 2016–2024 leicht handhabbar
- **Lizenz:** GeoNutzV (frei mit Quellenangabe)
- **Wichtig:** Daten sind nach Station versioniert in `*_hist.zip`-Archiven — du brauchst einen Download-Loop mit `requests` + Stations-Metadaten-Liste

### Quelle 3: OpenStreetMap via Overpass API (Straßenkontext)

- **Plattform:** `overpass-api.de/api/interpreter`, indirekt über data.europa.eu listbar
- **Zweck:** Anreicherung jeder Unfall-Koordinate mit Straßenattributen
- **Was du holst:**
  - `highway`-Tag (residential / primary / motorway / cycleway …) → Straßenkategorie
  - `maxspeed` → erlaubte Geschwindigkeit
  - `surface` → Belag
  - `lit` → Straßenbeleuchtung ja/nein
  - `junction` → Kreuzung/Kreisverkehr
- **Tooling:** `osmnx` Python-Paket für nearest-edge-Lookup
- **Performance-Hinweis:** Für 2,5 Mio. Punkte Naive-Lookup unpraktisch — extrahiere ein Hessen-/DE-Subgraph einmal, baue einen Spatial-Index (R-tree), dann Batch-Lookup. Oder Reduktion auf Wiesbaden/Hessen für Performance.

### Quelle 4: Eurostat Transport Statistics (Vergleichskontext, optional für K-Phase)

- **Plattform:** Eurostat über data.europa.eu
- **Zweck:** EU-Benchmark in der Knowledge-Transfer-Phase, "Wo steht Deutschland im Vergleich?"
- **Tooling:** `eurostat`-Python-Paket

### Join-Strategie

```text
Unfallatlas (Hauptfakt-Tabelle, 2,5 Mio. Rows)
   ├─ JOIN auf nächstgelegene DWD-Station × Stunde
   │     (Spatial-Lookup auf Stationsliste, dann Time-Join)
   ├─ JOIN auf OSM-Way (nearest edge zur Unfall-Koordinate)
   │     (osmnx.nearest_edges)
   └─ Optional: JOIN auf Eurostat-Aggregate für Plausibilitätschecks
         (Land × Jahr)
```

**Performance-Empfehlung:** Verarbeite das Ganze in **DuckDB** statt pandas. DuckDB unterstützt direkt CSV/Parquet, hat einen Spatial-Extension-Modul und ist für deine Datenmenge (2,5 Mio. Rows × ~30 Spalten nach Joins ≈ ein paar GB) trivial schnell. Auf einem normalen Laptop sind alle Aggregationen Sekundensache.

---

## DIG-Framework angewendet

### D — Description (das WAS)

Was beobachtet der Datensatz?

- 2,5 Mio. polizeilich aufgenommene Verkehrsunfälle mit Personenschaden in Deutschland 2016–2024.
- Jeder Eintrag: ein Unfall, geokodiert, mit Zeitstempel auf Stundenebene, Verwaltungsbezug, Schweregrad, Unfallart, Lichtverhältnissen, Straßenzustand und beteiligten Verkehrsmitteln (binär).
- Stichprobenwahrscheinlichkeit ist nicht uniform: nur Unfälle, deren Koordinaten eindeutig auf eine Straße gemappt werden konnten (~92 %), sind enthalten.

Konkrete Sub-Aufgaben in der D-Phase:

1. Daten laden, Schema validieren, Spaltenharmonisierung über Jahre hinweg.
2. Verteilung der Zielvariable `UKATEGORIE` global und pro Bundesland.
3. Räumliche Verteilung als Heatmap (Folium) — Sichtprüfung, ob Daten plausibel verteilt sind.
4. Missing-Values-Heatmap pro Spalte und Jahr.
5. Range-Checks für Geokoordinaten (Plausibilität: alles innerhalb DE-Bounding-Box?).

### I — Introspection (das WIE und WARUM)

Welche Fragen *kann* ich mit diesem Datensatz beantworten?

- Welche Faktoren korrelieren mit Unfallschwere? **Ja**, mit Vorsicht (Korrelation ≠ Kausalität).
- Wo sind Hotspots schwerer Unfälle? **Ja**, sehr gut.
- Wie wirkt Wetter auf Unfallhäufigkeit? **Ja**, mit DWD-Join.
- Wie unterscheiden sich Hessen-Hotspots von NRW-Hotspots? **Ja**.
- Können wir die Schwere eines hypothetischen Unfalls vorhersagen? **Ja, mit Einschränkungen** — siehe nächste Sektion.

Welche Fragen *kann ich nicht* beantworten?

- Wie viele Sachschadensunfälle gab es? Nicht im Datensatz.
- Was war der konkrete Unfallhergang? Nur kategoriale Klassifikation, keine Freitexte.
- Wer war schuld? Verkehrsmittel-Beteiligung ja, aber keine Schuldzuweisung.
- Welche Geschwindigkeit hatte das Fahrzeug? Nicht im Datensatz (nur über OSM `maxspeed` als erlaubte Geschwindigkeit annäherbar).
- Welche Demografie hatten die Beteiligten (Alter, Geschlecht)? Nicht enthalten — und genau das ist laut Literatur einer der stärksten Prädiktoren. **Diese Limitation ist im Notebook offen anzusprechen, das adressiert direkt eine zentrale DIG-Frage.**
- Wie viele *nicht*-polizeilich gemeldete Unfälle gibt es? Nicht beobachtbar.

### G — Goal Setting

Konkrete Ziele für das Projekt:

1. **Primärziel:** Multiclass-Klassifikator für `UKATEGORIE` mit macro-F1 ≥ 0.55 auf Held-Out-Test 2024 (Naive-Baseline = 0.33 / Majority-Class-Baseline = 0.30).
2. **Sekundärziel:** SHAP-basierte Erklärung der Top-10 Risikofaktoren mit visualisierter Validierung gegen wissenschaftliche Literatur.
3. **Tertiärziel:** Streamlit-App "Risikoprofil-Karte für Wiesbaden" mit interaktiver Folium-Karte plus Vorhersage-Modul für hypothetische Konstellationen.

---

## ML-Roadmap: Baselines bis State-of-the-Art

### Stufe 0: Baselines (Pflicht — gegen die du schlagen musst)

| Modell | Was es tut | Zweck |
|---|---|---|
| Random Guess | gleichverteilte Wahl | unteres Bound (macro-F1 ≈ 0.33) |
| Majority Class | immer "leicht" | zeigt das Imbalance-Problem (macro-F1 ≈ 0.30, Accuracy 80 %) |
| Logistic Regression | lineare Klassifikation | erste echte Baseline |

### Stufe 1: Klassische Modelle

| Modell | Stärke | Wann wählen |
|---|---|---|
| Random Forest | robust, weniger Tuning | guter erster Tree-Ansatz |
| **XGBoost** | State-of-the-Art für tabular | Hauptbenchmark |
| **LightGBM** | schneller als XGBoost auf vielen Kategorien | wahrscheinlich bestes Modell |
| **CatBoost** | nativ kategorische Features | besonders sinnvoll, weil `UART`, `UTYP1`, `ULAND` kategorisch sind |

Nach der Literatur (MDPI 2024, ScienceDirect 2025) liefern LightGBM und CatBoost typischerweise die besten Ergebnisse für genau diese Art von Daten.

### Stufe 2: Klassen-Imbalance-Strategien

`UKATEGORIE` ist stark imbalanced (~3 % Getötete, ~25 % Schwer, ~72 % Leicht). Du musst mehrere Strategien vergleichen:

1. **Class Weights** (XGBoost `scale_pos_weight`, LightGBM `class_weight='balanced'`)
2. **SMOTE** (Synthetic Minority Over-sampling)
3. **ADASYN** (Adaptive Synthetic Sampling)
4. **Threshold Moving** (Optimierung der Klassengrenze nach Training) — laut ScienceDirect 2025 Paper das beste Ergebnis bei XGBoost
5. **Ordinale Klassifikation** (weil 1<2<3 eine echte Ordnung haben) — laut MDPI 2024 die State-of-the-Art

Vergleichstabelle der Strategien × Modelle ist ein zentraler Portfolio-Output.

### Stufe 3: Erklärbarkeit

- **SHAP** (Shapley Additive exPlanations) auf dem besten Modell
- Globale Feature Importance + lokale Erklärung einzelner Vorhersagen
- Interaction Plots (z.B. `USTUNDE × ULICHTVERH`)
- **Sanity-Check gegen Literatur:** stimmen die Top-Features mit den in der Forschung berichteten überein? Wenn ja → Validierung. Wenn nein → spannender Diskussionspunkt.

### Stufe 4 (optional, für Bestnote): Räumliche Erweiterung

- **Geo-aware Modell:** XGBoost mit H3-Hexagon-Bins als Feature (uber/h3 Python)
- **GeoML:** ein einfaches Graph Neural Network auf der Straßennetz-Topologie aus OSM
- **Hotspot-Clustering:** DBSCAN auf Geokoordinaten, dann Hotspot-Profile

---

## Visualisierungs-Strategie

Visualisierung wird in diesem Projekt nicht "Bonus", sondern Kernergebnis. Vier Visualisierungs-Ebenen:

### Ebene 1: EDA (Notebook)

- Verteilungs-Histogramme aller numerischen Features
- Korrelations-Heatmap (mit Anpassung für ordinale + kategoriale Daten: Cramér's V)
- Stündliches/wöchentliches Heatmap-Grid (Wochentag × Stunde, gefärbt nach mittlerer Schwere)
- Saisonale Zeitreihen pro Bundesland
- Verteilungsvergleich Hessen vs. NRW vs. Bayern

### Ebene 2: Geo-EDA (interaktiv)

- **Folium-Heatmap** über ganz Deutschland, getrennt nach Schweregrad
- **kepler.gl-Visualisierung** für 3D-Zeit-Animation (Unfälle pro Stunde im Tagesverlauf)
- **H3-Choroplethen** auf Hexagon-Ebene
- **Plotly Mapbox** mit Hover-Details pro Unfall

### Ebene 3: Modellierung

- Confusion Matrix (Plotly-interaktiv)
- ROC-Kurven pro Klasse (one-vs-rest)
- Precision-Recall-Kurven (wichtig wegen Imbalance)
- SHAP Summary Plot, Force Plots, Dependence Plots
- Lernkurven (Train vs. Validation Loss über Epochen/Iterationen)

### Ebene 4: Streamlit-App (Portfolio-Output)

Modulstruktur der App:

1. **Übersicht** — Bundesweite Heatmap mit Filter nach Jahr/Schwere/Verkehrsmittel
2. **Wiesbaden-Fokus** — gezoomte Karte mit Hotspot-Detection
3. **Risikoprofil-Predictor** — User wählt Stunde, Wochentag, Wetter, Verkehrsmittel, Standort → Modell liefert Wahrscheinlichkeitsverteilung für leichten/schweren/tödlichen Unfall
4. **SHAP-Explorer** — User lässt sich für eine eingegebene Konstellation die Top-Treiber visualisieren
5. **Modellvergleich** — interaktive Confusion-Matrix und Metrik-Tabelle für die trainierten Modelle

Streamlit + folium + streamlit-folium + plotly + shap reichen für alles.

---

## QUA³CK-Phasen-Mapping

### Q — Question (Notebook 1)

- Forschungsfrage formulieren und begründen
- Hypothesen aus Literatur ableiten (5–7 konkret formulierte Hypothesen mit Quellenangabe)
- Erfolgsmetriken definieren (macro-F1, F1-getötet, Recall-getötet)
- Baseline-Performance aus der Literatur als Vergleich nennen

### U — Understanding the Data (Notebooks 2 + 3)

Notebook 2: DIG-Description und -Introspection
- Datenladen, Schema, Range-Checks, Missing-Values, Verteilungen
- Geo-EDA mit Folium
- Beantwortbare vs. nicht beantwortbare Fragen explizit auflisten

Notebook 3: Datenanreicherung
- DWD-Wetter-Join
- OSM-Anreicherung
- Feature Engineering: zyklische Features für Stunde/Monat (sin/cos), Lag-Features, H3-Cell-IDs, Distanz zur nächsten Großstadt, Wochenende-Flag, Feiertag-Flag (über `holidays`-Paket)

### A³ — Algorithm / Adapt / Adjust (Notebooks 4 + 5)

Notebook 4: Baseline-Modelle
- Random/Majority/LogReg
- Random Forest, XGBoost, LightGBM, CatBoost
- Cross-Validation (StratifiedKFold) mit konsistentem Train/Val/Test-Split (chronologisch: 2016–2022 train, 2023 val, 2024 test)

Notebook 5: Imbalance-Behandlung und Tuning
- Class Weights, SMOTE, ADASYN, Threshold Moving, Ordinal Classification
- Hyperparameter-Tuning mit Optuna
- Ensemble (Voting/Stacking) wenn Zeit

### C — Conclude & Compare (Notebook 6)

- Vergleichstabelle aller Modelle × Imbalance-Strategien
- ROC, PR, Confusion Matrices
- SHAP-Analyse des Gewinner-Modells
- Diskussion: stimmen Top-Features mit Literatur überein?
- Limitationen ehrlich diskutieren (Selektionsbias, fehlende Demografie, Korrelation ≠ Kausalität)

### K — Knowledge Transfer (Streamlit-App + README)

- Streamlit-App deployen (lokal oder als Docker auf deinem Charon-Server)
- README mit Reproduzier-Anleitung
- Optional: kurzer Blogpost oder LinkedIn-Posting

---

## Repository-Struktur

```text
unfallatlas-qua3ck/
├── README.md
├── AGENTS.md                 # AI-Agent-Workflow im Stil deines Snake-AI-Projekts
├── pyproject.toml            # uv/poetry
├── .gitignore
├── .pre-commit-config.yaml   # ruff, black, nbstripout
├── data/
│   ├── accidents.parquet     # LFS: konsolidiertes Unfallregister
│   ├── raw/                  # gitignored, Skripte zum Download
│   ├── interim/              # gejointe Stages (LFS: accidents_with_weather.parquet)
│   └── processed/            # ML-ready Parquet
├── notebooks/
│   ├── 01_question.ipynb         # Q-Phase
│   ├── 02_dig_description.ipynb  # U-Phase Teil 1
│   ├── 03_dig_enrichment.ipynb   # U-Phase Teil 2 (Joins)
│   ├── 04_baselines.ipynb        # A³ Teil 1
│   ├── 05_tuning_imbalance.ipynb # A³ Teil 2
│   └── 06_compare_explain.ipynb  # C-Phase
├── src/unfallatlas/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py       # Unfallatlas Multi-Year Downloader
│   │   ├── dwd.py            # DWD CDC Stations + Hourly Loader
│   │   └── osm.py            # Overpass-Wrapper
│   ├── features/
│   │   ├── temporal.py       # zyklische Encodings, Feiertage
│   │   ├── spatial.py        # H3, Distanzen
│   │   └── enrich.py         # Joins
│   ├── models/
│   │   ├── baseline.py
│   │   ├── boosting.py
│   │   ├── ordinal.py
│   │   └── evaluate.py
│   └── viz/
│       ├── geo.py            # Folium / kepler.gl Helpers
│       ├── shap_plots.py
│       └── streamlit_app.py
├── app/
│   └── streamlit_app.py      # Entry-Point der Demo-App
├── tests/
│   └── test_features.py
└── reports/
    ├── figures/              # exportierte Plots fürs README
    └── final_report.md
```

Diese Struktur ist QUA³CK-konform (1 Notebook pro Phase + zusätzlich), modulartig, und passt zu deinem etablierten Stil aus Snake-AI/GitRay.

---

## Modell-Roadmap mit erwartbaren Metriken

Aus der Literatur (MDPI 2024, ScienceDirect 2025, ICML/NeurIPS-nahe Arbeiten 2022–2025) kann man realistische Zielwerte ableiten:

| Modell | Erwarteter macro-F1 | Erwarteter Recall (Getötete) |
|---|---|---|
| Majority Class | ~0.30 | 0.00 |
| Logistic Regression | 0.42–0.48 | 0.10–0.20 |
| Random Forest | 0.50–0.55 | 0.25–0.35 |
| XGBoost (default) | 0.55–0.60 | 0.30–0.40 |
| LightGBM + Class Weights | 0.60–0.65 | 0.45–0.55 |
| CatBoost + Threshold Moving | **0.65–0.72** | **0.55–0.70** |
| Ordinal CatBoost + SHAP | 0.65–0.72 | 0.55–0.70 (mit besserer Interpretierbarkeit) |

**Dein realistisches Zielband: macro-F1 0.60–0.70 mit Recall-Getötete > 0.50.**

---

## Risiken und Gegenmaßnahmen

| Risiko | Wahrscheinlichkeit | Gegenmaßnahme |
|---|---|---|
| OSM-Join für 2,5 Mio. Punkte zu langsam | hoch | auf Hessen oder Top-3-Bundesländer beschränken; oder H3-Cell-Aggregation statt Punkt-Lookup |
| DWD-Stationen liefern Lücken | mittel | räumliche Interpolation oder Open-Meteo als Fallback |
| Klassen-Imbalance nicht in den Griff zu kriegen | niedrig | Literatur ist klar: Threshold Moving + ordinale Klassifikation funktionieren |
| Modell macht zwar gute Vorhersagen, aber SHAP-Features sind nicht plausibel | mittel | Sanity-Check gegen Literatur explizit als Schritt einplanen |
| Streamlit-App zu langsam wegen großem Modell | niedrig | trainiertes Modell als Pickle/ONNX, App lädt nur Inferenz-Pipeline |
| Selektionsbias der 92 %-Geocoder-Quote wird übersehen | mittel | direkt in der DIG-Introspektion adressieren — wird im Bericht zum Pluspunkt statt Manko |

---

## Hinweis zur Note

Was die Bewertung typischerweise im Big-Data-/Data-Analytics-Kurs trennt:

- **2,0:** funktionierendes Notebook mit ein paar Modellen und einem Plot
- **1,3:** vollständige QUA³CK-Phasen, mehrere Modelle verglichen, ordentlich dokumentiert
- **1,0:** alles oben + nicht-trivialer ML-Beitrag (Imbalance-Strategien, ordinale Klassifikation), saubere SHAP-Erklärung mit Literatur-Validierung, Streamlit-App, klare Limitations-Diskussion in der DIG-Introspektion

Drei Dinge, die in Studienarbeiten typischerweise fehlen und dich heraushebem würden:

1. **Echte Held-Out-Test-Strategie**: chronologisch 2024 als Testjahr nicht zufällig — viele Studierende splitten zufällig und überschätzen ihre Modelle.
2. **Explizite Limitations-Diskussion** im DIG-Introspektion-Stil — sehr selten, aber genau das, was wissenschaftliches Arbeiten ausmacht.
3. **Sanity-Check der SHAP-Werte gegen peer-reviewed Literatur** — verbindet ML-Output mit dem Kanon der Verkehrssicherheitsforschung.

---

## Sofort-Schritte (für die nächste Session)

Wenn du loslegen willst, sind das die ersten konkreten Aktionen:

1. Repo `unfallatlas-qua3ck` anlegen (lokal + GitHub).
2. Unfallatlas 2024 CSV von GovData ziehen (~30 MB) — als Sample-Set für die D-Phase.
3. Notebook `01_question.ipynb` schreiben: Forschungsfrage, Hypothesen, Erfolgsmetriken, Literatur-Recherche-Block.
4. Notebook `02_dig_description.ipynb`: Schema validieren, Verteilungen plotten, Folium-Karte rendern.
5. Wenn das läuft: alle 9 Jahrgänge laden und harmonisieren.

Sag mir, ob du mit Schritt 1 (Repo + Notebook 01 + Notebook 02 als Skelett) starten willst — dann generiere ich dir AGENTS.md, README, pyproject.toml und die ersten beiden Notebooks als ausführbares Startgerüst.
