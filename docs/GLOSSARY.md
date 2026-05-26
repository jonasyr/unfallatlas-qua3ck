# Glossary and Term Explanations

This glossary explains key terms, methods, and metrics used in the Unfallatlas QUA³CK project in plain language.

---

## Dataset and Domain Terms

**Unfallatlas Deutschland**
A publicly available dataset published by the statistical offices of Germany's federal states. It records every police-reported road accident resulting in personal injury, one record per accident, since 2016.

**UKATGEORIE (Unfallkategorie)**
Source-data misspelling of *UKATEGORIE*; the project adopts the source spelling without correction.
The target variable. Classifies the worst injury outcome of an accident:

- `1` — at least one person killed (Getötet)
- `2` — at least one person severely injured, none killed (Schwer verletzt)
- `3` — only light injuries (Leicht verletzt)

**UJAHR / UMONAT / USTUNDE / UWOCHENTAG**
Temporal columns encoding the year, month, hour of day, and day of week of the accident. All are integer-coded; cyclic features (month, hour, weekday) receive sin/cos encoding before modelling.
`UWOCHENTAG` uses a Sunday-first encoding: 1 = Sonntag, 2 = Montag, …, 7 = Samstag.

**ULAND (Bundesland)**
Two-digit code identifying the federal state in which the accident occurred.
Together with `UREGBEZ`, `UKREIS`, and `UGEMEINDE`, it forms the official municipality key (*Amtlicher Gemeindeschlüssel*). Coverage start year varies by state (e.g. Nordrhein-Westfalen from 2019, Mecklenburg-Vorpommern from 2020).

**OBJECTID**
Unique integer identifier per accident row. No two rows share an OBJECTID; used in the §9.3 no-overlap check to confirm the chronological split contains no duplicate accidents across train / val / test sets. Dropped before model fitting.

**UART (Unfallart)**
The type of accident, recorded in the police report after the event. Subject to leakage audit in §9.1 of the U phase (conditional entropy reduction: 3.4 %, well below the 50 % trigger).

- `0` — Other accident type (Unfall anderer Art; ~15 % of records; 58 % bicycle involvement)
- `1` — Collision with a stationary/stopping/parked vehicle
- `2` — Rear-end collision with a preceding or waiting vehicle
- `3` — Sideswipe collision (same direction)
- `4` — Head-on collision
- `5` — Collision with a turning or crossing vehicle (most frequent)
- `6` — Collision between vehicle and pedestrian
- `7` — Impact with an obstacle on the roadway
- `8` — Run-off road to the right
- `9` — Run-off road to the left (codes 8/9 carry the highest fatality rate)

**UTYP1 (Unfalltyp)**
The structural accident type, recorded post-event. Subject to the same leakage probe as `UART` (§9.1; conditional entropy reduction: 2.2 %).

- `1` — Fahrunfall (driving accident)
- `2` — Abbiegeunfall (turning accident)
- `3` — Einbiegen/Kreuzen-Unfall (merging/crossing accident)
- `4` — Überschreiten-Unfall (pedestrian crossing accident)
- `5` — Unfall durch ruhenden Verkehr (accident involving parked/stationary traffic)
- `6` — Unfall im Längsverkehr (longitudinal-traffic accident)
- `7` — Sonstiger Unfall (other)

**ULICHTVERH (Lichtverhältnisse)**
Lighting conditions at the time of the accident:

- `0` — Tageslicht (daylight)
- `1` — Dämmerung (dusk/dawn)
- `2` — Dunkelheit (darkness)

**STRZUSTAND (Straßenzustand)**
Road surface condition at the time of the accident: dry (0), wet/moist/slippery (1), or winter conditions — ice/snow (2).

**IstRad / IstPKW / IstFuss / IstKrad / IstGkfz / IstSonstige**
Binary flags indicating which transport modes were involved: bicycle (`IstRad`), car (`IstPKW`), pedestrian (`IstFuss`), motorcycle or moped (`IstKrad`), heavy goods vehicle > 3.5 t (`IstGkfz`), and other (`IstSonstige`).
Note: `IstGkfz` is only available from 2018 onward; in 2016 and 2017 its accidents are subsumed under `IstSonstige`.

**UREGBEZ / UKREIS / UGEMEINDE**
Geographic identifiers at Regierungsbezirk (administrative region), Kreis (district), and Gemeinde (municipality) level. `UREGBEZ` and `UKREIS` are used as target-encoded features; `UGEMEINDE` is dropped due to very high cardinality.

**LAT / LON** *(source columns: `YGCSWGS84` / `XGCSWGS84`)*
WGS-84 geographic coordinates (decimal degrees) of the accident location, renamed from the DSB source columns during parquet consolidation. Used for spatial analysis and for the nearest-station lookup in the DWD enrichment.

**LINREFX / LINREFY**
UTM coordinates (ETRS89, Zone 32N) of the accident location projected onto the nearest road segment. Distinct from `XGCSWGS84 / YGCSWGS84` (WGS-84 decimal degrees). Not used in this project; the join and spatial analysis use the WGS-84 columns.

**PLST (Plausibilisierungsstufe)**
Geocoding quality indicator. `1` = accident location geocoded by the standard procedure; `2` = geocoded by the extended procedure for accidents involving bicycles. Only accidents that pass plausibility checks are included in the published dataset (~92 % of all recorded events); the remaining ~8 % are excluded, introducing a documented selection bias.

---

## DWD Weather Enrichment

**DWD (Deutscher Wetterdienst)**
Germany's national meteorological service. Publishes hourly station observations as open data under GeoNutzV.

**CDC (Climate Data Center)**
DWD's open-data portal for climate and weather observations. The hourly sub-tree provides per-station ZIP archives for each variable and period (recent / historical).

**dwd_temp_air_2m**
Air temperature in °C measured at 2 m above ground. Approximately normally distributed; imputed by median per (station, month).

**dwd_precip_mm**
Precipitation in mm for the hour. Strongly right-skewed (most hours have zero rain); receives a `log1p` transform before modelling.

**dwd_visibility_m**
Horizontal visibility in metres. Strongly right-skewed; receives a `log1p` transform. Low-visibility tail captures fog and night conditions relevant to severity.

**dwd_wind_speed_ms**
Mean wind speed in m/s. Approximately symmetric; no transform required beyond `StandardScaler`.

**dwd_station_dist_km**
Distance in km from the accident location to the nearest DWD station. Acts as a rural-character proxy: accidents far from stations are disproportionately on rural roads with higher fatal rates (§9.4).

**Join granularity**
Weather is joined on (station_id, year, month, hour-of-day). Because `accidents.parquet` lacks a day-of-month column (`UTAG`), values are averaged over all days in each month-hour bucket. This introduces day-level noise but no temporal leakage.

**cKDTree**
A k-d tree data structure from `scipy.spatial` used to vectorise the nearest-station lookup. Euclidean distance in radian-space is accurate to < 0.1 % for distances ≤ 30 km.

---

## Data Quality

**Missing Values**
Entries that are null or empty. Per-column strategies are specified in the §10 preprocessing decision table (U phase).

**Sentinel Values**
Values such as `-999` or `-9999` used by DWD to signal invalid or missing measurements. Replaced with `NaN` before joining.

**Duplicates**
Repeated rows or OBJECTID values. Exact row duplicates are negligible in this dataset; OBJECTID is unique.

**Class Imbalance**
The target distribution is approximately 1 % / 18 % / 81 % for classes 1 / 2 / 3. Naïve models will default to the majority class; mitigation options (class weights, SMOTE, threshold moving) are chosen in A³.

---

## Machine Learning Concepts

**Multiclass Classification**
A supervised learning task where the target variable has more than two discrete classes. Here: predicting `UKATGEORIE` ∈ {1, 2, 3}.

**Feature**
An input variable used by the model (e.g., `USTUNDE`, `dwd_precip_mm`, `ULICHTVERH`).

**Target Variable**
The variable the model predicts. Here: `UKATGEORIE` (accident severity class).

**Macro-F1**
The arithmetic mean of per-class F1 scores, treating all classes equally regardless of support. Primary evaluation metric for this project; protects against majority-class collapse.

**Recall (class 1)**
The fraction of fatal accidents (class 1) correctly identified. Secondary acceptance criterion: must exceed 0.50 on the held-out test set.

**Cramér's V**
A symmetric measure of association between two categorical variables, ranging from 0 (no association) to 1 (perfect association). Used in §6 and §8.7 of the U phase to quantify feature–target relationships.

**Conditional Entropy Reduction**
`1 − H(Y | X) / H(Y)`. A reduction near 100 % means feature X nearly determines target Y — a leakage flag. Used in the §9.1 leakage probe.

**Target Encoding**
Replaces a categorical value with the mean of the target across training rows that share that value, with additive smoothing to prevent overfitting on rare categories. Used for `UREGBEZ` and `UKREIS`.

**Cyclic Encoding**
Represents a periodic feature (month, hour, weekday) as a (sin, cos) pair so the model sees the circular distance between, e.g., hour 23 and hour 0.

**Overfitting**
When a model learns noise in the training data and generalises poorly to new data.

**Data Leakage**
When information unavailable at prediction time (future data, or features that definitionally encode the target) enters training, causing inflated performance estimates that do not hold at deployment.

**Chronological Split**
Train / validation / test division that preserves time order: train 2016–2022, val 2023, test 2024. Prevents temporal leakage between splits.

**TimeSeriesSplit**
A cross-validation strategy that always trains on past folds and validates on future folds, preserving the time-series property within the training window.

---

## Process Model

**QUA³CK**
A structured ML process model for data science projects: **Q**uestion, **U**nderstanding, **A**lgorithm selection / data **A**daptation / parameter **A**djustment, **C**onclude & Compare, **K**nowledge Transfer.

**Phase Q (Question)**
Defines the research question, stakeholders, success criteria, data sources, scope, and evaluation design before any data is touched.

**Phase U (Understanding)**
Exploratory data analysis: schema audit, missingness, distributions, bivariate associations, leakage probes, and the preprocessing decision table handed to A³.

**Phase A³ (Algorithm / Adapt / Adjust)**
Feature engineering, model selection, hyperparameter tuning, and cross-validation inside a sklearn `Pipeline` so preprocessing statistics never leak from training into validation or test sets.

**Phase C (Conclude & Compare)**
Final model evaluation on the held-out 2024 test set, comparison against baselines, and documentation of limitations.

**Phase K (Knowledge Transfer)**
Delivery artefacts: interactive Streamlit app, final report, and this documentation.

**DIG Framework**
A sub-framework for the U phase: **D**escription (inspect structure and samples), **I**ntrospection (formulate questions, identify limitations), **G**oal setting (decide whether data is suitable and define next steps).
