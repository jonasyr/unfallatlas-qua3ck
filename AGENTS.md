# AGENTS.md — AI Agent Workflow

Dieses Dokument beschreibt die Projektkonventionen für KI-Agenten, die an diesem Repository arbeiten.

## Projektübersicht

Multiclass-Klassifikation der Unfallschwere (`UKATGEORIE`: 1/2/3) auf dem deutschen Unfallatlas 2016–2024.
~2,09 Mio. Zeilen, 21 Spalten. Hauptdatei: `data/body.parquet`.

## Daten laden

```python
import duckdb
import pandas as pd
from pathlib import Path

DATA = Path("data/body.parquet")

# Empfohlen: DuckDB für große Abfragen
con = duckdb.connect()
df = con.execute(f"SELECT * FROM '{DATA}' WHERE UJAHR = 2024").df()

# Alternativ: pandas für kleinere Subsets
df = pd.read_parquet(DATA)
```

**Rohe CSV-Dateien** (`data/raw/unfalldaten/unfalldaten_XXXXX.csv`):
```python
df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
```

## Schlüsselspalten

| Spalte | Typ | Bedeutung |
|--------|-----|-----------|
| `UKATGEORIE` | TINYINT | **Zielvariable** — 1=Getötet, 2=Schwer, 3=Leicht |
| `UJAHR` | SMALLINT | Unfalljahr 2016–2024 |
| `UMONAT` | TINYINT | 1–12 |
| `USTUNDE` | TINYINT | 0–23 |
| `UWOCHENTAG` | TINYINT | 1=Sonntag, 2=Montag, … 7=Samstag |
| `UART` | TINYINT | Unfallart 0–9 |
| `UTYP1` | TINYINT | Unfalltyp 1–7 |
| `ULICHTVERH` | TINYINT | 0=Tageslicht, 1=Dämmerung, 2=Dunkelheit |
| `STRZUSTAND` | TINYINT | 0=trocken, 1=nass, 2=winterglatt |
| `IstRad` | BOOLEAN | Fahrradbeteiligung |
| `IstPKW` | BOOLEAN | PKW-Beteiligung |
| `IstFuss` | BOOLEAN | Fußgängerbeteiligung |
| `IstKrad` | BOOLEAN | Krad-Beteiligung |
| `IstGkfz` | BOOLEAN | Güterkraftfahrzeug |
| `IstSonstig` | BOOLEAN | Sonstiges Verkehrsmittel |
| `LON` | DOUBLE | Längengrad WGS84 |
| `LAT` | DOUBLE | Breitengrad WGS84 |
| `UREGBEZ` | VARCHAR | Regierungsbezirk-Code |
| `UKREIS` | VARCHAR | Kreis-Code |
| `UGEMEINDE` | VARCHAR | Gemeinde-Code |

## Bekannte Stolpersteine

- **Tippfehler im Spaltennamen:** `UKATGEORIE` (nicht `UKATEGORIE`) — so ist es im Parquet gespeichert.
- **Kein `ULAND`-Feld** im Parquet — Bundesland muss aus `UKREIS` (erste 2 Zeichen) abgeleitet werden: `df["ULAND"] = df["UKREIS"].str[:2].astype(int)`.
- **CSV-Dezimaltrennzeichen:** Rohe CSVs nutzen Komma als Dezimalzeichen → `decimal=","`.
- **Klassenimbalance:** 1% / 18% / 81% → Immer stratified splits und macro-F1 als primäre Metrik.
- **Chronologischer Split:** Train 2016–2022, Val 2023, Test 2024 — kein zufälliger Split.

## Test-Split-Strategie

```python
train = df[df.UJAHR <= 2022]
val   = df[df.UJAHR == 2023]
test  = df[df.UJAHR == 2024]
```

## Coding-Konventionen

- **Formatter:** `ruff` + `black` (line-length 100)
- **Kein `print()`** in Modulen — verwende `logging`
- **Pfade:** immer `pathlib.Path`, nie hartcodierte Strings
- **Modell-Artefakte:** in `data/processed/` speichern (gitignored)
- **Notebook-Outputs:** werden per `nbstripout` vor Commits entfernt

## Phasen-Übersicht

```
01_Q_Phase.ipynb     → Forschungsfrage + Hypothesen (FERTIG)
02_U_Phase.ipynb     → EDA + Feature Engineering (TODO)
03_A3_Phase.ipynb    → Modellierung + Tuning (TODO)
04_C_Phase.ipynb     → Vergleich + SHAP + Fazit (TODO)
app/streamlit_app.py → Streamlit-Demo (TODO)
```
