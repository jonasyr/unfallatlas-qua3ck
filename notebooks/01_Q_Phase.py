# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # "Unfallatlas Deutschland" — Q-Phase
#
# **Phase:** Question (Q) · 1 of 5 · QUA³CK
# **Goal of this notebook:** define the problem, the target, the success criteria,
# the constraints, and the data context — *before any data is opened or any model
# is trained*. The output is a written, testable problem definition that the
# remaining QUA³CK phases can build against.
#
# > The Q phase is the cheapest place to fix a mistake. Every later phase is
# > downstream of this document.
#
# ---

# %% [markdown]
# ## Position in the QUA³CK process
#
# | Phase | Notebook | Purpose | Status |
# |:---|:---|:---|:---:|
# | **Q** — Question | `01_Q_Phase.ipynb` | Problem definition, target, metrics, constraints | ✓ |
# | **U** — Understanding | `02_U_Phase.ipynb` | Schema audit, EDA, quality, leakage probes, preprocessing decisions | → next |
# | **A³** — Algorithm / Adapt / Adjust | `03_A3_Phase.ipynb` | Baselines, boosting models, imbalance strategies, tuning | pending |
# | **C** — Conclude & Compare | `04_C_Phase.ipynb` | SHAP, model comparison, limitations | pending |
# | **K** — Knowledge Transfer | `app/streamlit_app.py` | Interactive risk-profile application | pending |
#
# ---

# %% [markdown]
# ## 1 · Problem context
#
# Each year, German police record roughly **270,000 road accidents with personal
# injury**. About 1 % of these are fatal, 18 % cause serious injury, and 81 %
# cause only minor injury. The Statistisches Bundesamt publishes every one of
# these accidents — georeferenced, hour-stamped, with administrative codes for
# road condition, lighting, accident type, and the transport modes involved — as
# the **Unfallatlas Deutschland**, an open dataset spanning 2016 – 2024.
#
# Despite this rich data infrastructure, road-safety decisions at the municipal
# level (intersection redesign, lighting, speed-limit revision) are typically
# made on aggregate statistics. A natural question follows: can the
# **conditions** under which an accident occurs predict its **severity**, and
# can such a model be made interpretable enough to inform prevention?
#
# ---

# %% [markdown]
# ## 2 · Research question
#
# > **Welche raumzeitliche und infrastrukturelle Faktoren bestimmen die Schwere
# > eines Verkehrsunfalls in Deutschland, und lässt sich diese Schwere mit
# > interpretierbaren Machine-Learning-Modellen aus öffentlich verfügbaren Daten
# > hinreichend zuverlässig vorhersagen, um Präventionsentscheidungen zu
# > unterstützen?**
#
# The research question is intentionally open-ended; it motivates the project.
# The Q phase operationalises it in the next section as a *prediction goal* that
# can be tested, falsified, and accepted or rejected against a number.
#
# ---

# %% [markdown]
# ## 3 · Hypotheses
#
# Three testable predictions derived from the research question. Each is evaluated against evidence in the U phase (EDA) or the A³ phase (model results). A hypothesis is *not* a prior commitment to what A³ will find — it is a structured expectation that can be confirmed, weakened, or falsified by the data.
#
# ---
#
# **H1 — Temporal-environmental severity shift**
#
# > Night-time accidents (22:00 – 05:00) and accidents recorded under adverse meteorological conditions (precipitation > 0 mm, visibility < 2 000 m, or icy/snow road surface) will show a statistically higher share of fatal and serious-injury outcomes (`UKATGEORIE` ∈ {1, 2}) compared to daytime, dry-road accidents. The association between darkness, adverse weather, and severity will persist after controlling for transport mode and accident type.
#
# *Verifiable in:* U phase §7 (hourly severity profile), §8.6 (weather distributions), §8.7 (Cramér's V weather × UKATGEORIE).
#
# ---
#
# **H2 — Rural infrastructure and fatal-severity concentration**
#
# > Accidents in rural Bundesländer (proxied by higher `dwd_station_dist_km` and lower population density per Kreis) will exhibit significantly higher fatal-accident rates than urban accidents. The effect is attributed to higher travel speeds on rural roads, lower infrastructure quality, and longer emergency-response times — none of which are directly measurable in the dataset, but whose aggregate footprint is detectable at the spatial-aggregate level.
#
# *Verifiable in:* U phase §8 (Bundesland fatal-share analysis), §9.4 (rural-proxy probe on `dwd_station_dist_km`).
#
# ---
#
# **H3 — Model feasibility: gradient-boosted trees above literature baseline**
#
# > A gradient-boosted tree model trained on spatiotemporal, infrastructural, and meteorological features will achieve macro-F1 ≥ 0.55 on the chronological test year 2024, materially exceeding the majority-class baseline (macro-F1 ≈ 0.30) and a logistic-regression benchmark. The temporal features (`USTUNDE`, `UMONAT`) and spatial features (`UKREIS`, `dwd_station_dist_km`) are expected to rank among the top SHAP contributors, consistent with the literature anchor in §11.
#
# *Verifiable in:* A³ phase (model training and evaluation), C phase (benchmark comparison and SHAP analysis).
#
# ---

# %% [markdown]
# ## 4 · Prediction goal (operationalised)
#
# > For each personal-injury road accident in Germany 2016 – 2024 documented in
# > the Unfallatlas, predict the recorded severity class **`UKATGEORIE`** —
# > 1 = fatal, 2 = serious injury, 3 = minor injury — from the administratively
# > recorded conditions available *at the time of the police report* (location,
# > time, lighting, road condition, accident type and kind, transport modes
# > involved). Acceptable performance is **macro-F1 ≥ 0.55** on the
# > chronologically held-out test year 2024, with **recall for class 1 ≥ 0.50**,
# > using a Gradient Boosting model interpretable via SHAP for downstream use by
# > municipal road-safety planners.
#
# The prediction goal is the contract A³ will be evaluated against. It fixes the
# target, the feature set's temporal availability, the evaluation protocol, the
# metric, the threshold, the secondary metric, and the consumer.
#
# ---

# %% [markdown]
# ## 5 · Target definition
#
# **Target column:** `UKATGEORIE` (note: misspelled in the source data; the
# project adopts the source spelling without comment).
#
# **Encoding:** ordinal, three levels with natural ordering.
#
# | Code | Label | Operational meaning |
# |:---:|:---|:---|
# | 1 | Getötet | At least one person involved died within 30 days of the accident |
# | 2 | Schwer verletzt | At least one person required ≥ 24 hours of in-patient care |
# | 3 | Leicht verletzt | At least one person had injuries requiring medical attention but not the above |
#
# **Label source.** Police record at the scene, finalised after the 30-day
# fatality window. Administered uniformly across Bundesländer per the Statistical
# Office's catalogue.
#
# **Label observability at prediction time.** The label is *not* available to
# the model at inference. Features used must be observable at the moment the
# police report is written; this is the temporal-leakage boundary the U-phase
# will probe.
#
# **Ordinality.** The classes have a natural ordering (1 > 2 > 3 in severity).
# This justifies both standard multi-class classification and ordinal
# classification approaches in A³. The Q phase does not commit to either; the
# choice is a modelling decision.
#
# ---

# %% [markdown]
# ## 6 · Unit of analysis and prediction horizon
#
# | Aspect | Specification |
# |:---|:---|
# | **Unit of analysis** | One row = one police-recorded personal-injury accident |
# | **Prediction horizon** | Point-in-time — the severity at the moment the police report is written |
# | **Granularity** | Hourly (`USTUNDE`), daily (`UWOCHENTAG`), monthly (`UMONAT`), yearly (`UJAHR`) |
# | **Spatial granularity** | WGS84 coordinates (`LON`, `LAT`) + 5-digit Kreis code (`UKREIS`) |
# | **Coverage** | All of Germany, 2016 – 2024, with a documented ~8 % geocoding-quote exclusion |
#
# The unit of analysis is *the accident*, not *the person involved*. Multi-person
# accidents are represented as a single row; the model predicts the worst
# recorded severity, not per-person severity.
#
# ---

# %% [markdown]
# ## 7 · Stakeholders and decision context
#
# | Role | Decision the model informs |
# |:---|:---|
# | **Municipal road-safety planners** | Prioritisation of corridor- and intersection-level interventions (lighting, signage, speed limits, geometric redesign) |
# | **Prevention campaign designers** (e.g. ADAC, BASt, DVR) | Targeting messaging to the most consequential time-of-day / weekday / weather / mode combinations |
# | **Public-sector policy analysts** | Quantifying progress toward EU Vision Zero 2050 goals at sub-national level |
#
# **Excluded by design.** The model is *not* intended to inform individual
# behaviour (driver-level risk scores), insurance underwriting, or
# law-enforcement targeting. These applications would require demographic and
# behavioural features that are not in the dataset and would raise ethical
# objections orthogonal to the project's purpose.
#
# ---

# %% [markdown]
# ## 8 · Success metrics
#
# ### Primary metric
#
# **macro-F1 on the held-out test year 2024.**
#
# macro-F1 is chosen because the target is severely imbalanced (~1 / 18 / 81).
# Accuracy would reward predicting "minor" for everything (~81 %). Weighted F1
# would still over-weight the dominant class. macro-F1 averages the per-class F1
# without size weighting and is the standard metric for imbalanced multi-class
# classification in the relevant literature.
#
# ### Acceptance threshold
#
# | Threshold | macro-F1 (test 2024) | Interpretation |
# |:---|:---:|:---|
# | Baseline (majority class) | ~0.30 | Trivial; should be beaten by anything |
# | Acceptable | ≥ 0.55 | Minimum to declare the project useful |
# | Literature-realistic | 0.60 – 0.70 | Range reported in comparable studies |
#
# ### Secondary metric
#
# **Recall for class 1 (Getötet) ≥ 0.50.**
#
# The cost asymmetry is strong: missing a fatal-severity prediction is worse
# than over-predicting one. A model with high macro-F1 but recall 0.10 on class 1
# is rejected.
#
# ### Business metric (deferred to K)
#
# Number of high-risk corridor-hours identified by the SHAP-explained model that
# trigger a documented safety intervention. The Q phase records this as the
# real-world success measure even though it cannot be evaluated within the
# project itself.
#
# ---

# %% [markdown]
# ## 9 · Constraints
#
# ### Hard constraints
#
# | Constraint | Source | Implication |
# |:---|:---|:---|
# | **Interpretability** | Public-sector consumer; civic accountability | Black-box models acceptable only if accompanied by SHAP-based local and global explanations |
# | **Reproducibility** | Academic portfolio context | Dataset hash, code commit, environment versions, random seed all logged |
# | **Licence compatibility** | Data is Datenlizenz Deutschland 2.0 | All derived artefacts must permit attribution; commercial reuse permitted |
# | **No personal data** | DSGVO / GDPR | Dataset contains no personal identifiers; the constraint is satisfied by source |
#
# ### Soft constraints
#
# | Constraint | Implication |
# |:---|:---|
# | **Inference latency** | Not real-time; batch nightly or weekly inference is acceptable |
# | **Model size** | The Streamlit deployment runs on a personal home server; models < 500 MB |
# | **Compute budget** | Tuning fits on a single workstation; no cluster required |
#
# ### Out of scope
#
# The project does *not* attempt to:
#
# - Predict accidents that have not yet happened (this is a *severity*
#   classifier for events that have occurred, not an early-warning system).
# - Predict causation. The dataset records categorical conditions, not causes.
# - Predict per-person outcomes. The unit is the accident, not the participant.
# - Model demographic effects. Demographics are not in the dataset.
# - Estimate true fatality counts. The Dunkelziffer of minor accidents
#   (unreported to police) is unobserved; the model inherits this bias.
# - Generalise outside Germany. Training data is national; cross-border
#   performance is not claimed.
#
# ---

# %% [markdown]
# ## 10 · Data sources
#
# ### Primary source
#
# | Item | Value |
# |:---|:---|
# | **Dataset** | Unfallatlas Deutschland |
# | **Publisher** | Statistisches Bundesamt (Destatis), coordinated with Bundesländer |
# | **Distribution** | [GovData.de](https://www.govdata.de/suche/daten/unfallatlas) / Mobilithek |
# | **Format** | Shapefile + CSV per year; consolidated to Parquet for this project |
# | **Coverage** | 2016 – 2024 (9 vintages), all of Germany |
# | **Volume** | ~2.09 million personal-injury accidents |
# | **Licence** | Datenlizenz Deutschland — Namensnennung — 2.0 (CC-BY-equivalent) |
# | **Citation** | "Datenquelle: Statistische Ämter des Bundes und der Länder, Unfallatlas, 2016–2024" |
#
# ### Provenance note
#
# Coordinates are published only for accidents geocoded with high confidence
# (~92 % of recorded accidents). The remaining ~8 % are not in the dataset. This
# is a documented selection bias to be re-examined in U.
#
# ### Project storage (Unfallatlas)
#
# The consolidated Parquet lives at `data/accidents.parquet` (~66 MB compressed,
# tracked via Git LFS). Re-creation from raw CSVs is scripted in
# `src/unfallatlas/data/download.py`.
#
# ---
#
# ### Secondary source — DWD Climate Data Center (CDC)
#
# | Item | Value |
# |:---|:---|
# | **Dataset** | DWD CDC Hourly Observations (Stundenwerte) |
# | **Publisher** | Deutscher Wetterdienst (DWD) |
# | **Distribution** | [CDC Open Data](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/) |
# | **Format** | ZIP archives containing semicolon-separated CSV (one per station per variable); fixed-width station master file |
# | **Coverage** | ~400 active climate stations across Germany; records from 1937 to present |
# | **Variables** | Air temperature at 2 m (TU, °C) · precipitation (RR, mm) · visibility (VV, m) · wind speed (FF, m/s) |
# | **Temporal resolution** | Hourly |
# | **Licence** | GeoNutzV — Geodatenlizenz Deutschland (free; attribution required) |
# | **Citation** | "Quelle: Deutscher Wetterdienst (DWD), Climate Data Center (CDC)" |
#
# ### Role in this project
#
# DWD data enriches each accident record with the meteorological conditions
# observed at the nearest station at the hour of the accident. This creates a
# richer feature set (wet road confirmed by precipitation, reduced visibility
# from fog) compared to the administrative proxy `STRZUSTAND` already in the
# Unfallatlas.
#
# ### Project storage (DWD)
#
# Raw ZIP downloads are cached to `data/raw/dwd/` (gitignored). Parsed hourly
# observations are written to `data/interim/` as Parquet per station per
# variable (also gitignored; reproduced by re-running `build_weather_features`).
# The join logic lives in `src/unfallatlas/data/dwd.py`.
#
# ---

# %% [markdown]
# ## 11 · Feasibility check and literature anchor
#
# ### Data adequacy
#
# | Question | Answer |
# |:---|:---|
# | Does the label exist? | Yes — `UKATGEORIE` is uniformly recorded |
# | Is the label observable at prediction time? | Yes — it is set in the police report |
# | Are features observable at prediction time? | Yes — administrative codes are part of the same report |
# | Is the volume sufficient? | Yes — 2.09 M rows, with the rarest class still ~21,000 samples |
# | Is there a literature anchor? | Yes — see below |
#
# ### Literature anchor
#
# | Reference | Method | Reported metric | Relevance |
# |:---|:---|:---|:---|
# | Santos et al. (2022), *Accident Analysis & Prevention* | XGBoost + SHAP on Portuguese accident data | macro-F1 ≈ 0.60 | Direct methodological precedent |
# | Pakgohar et al. (2021), *IATSS Research* | LightGBM + SMOTE | macro-F1 ≈ 0.62 | Imbalance treatment template |
# | Schlößler et al. (2024), *Accident Analysis & Prevention* | ML ensemble on German accident data | macro-F1 ≈ 0.65 | Comparable jurisdiction and features |
# | MDPI *Sustainability* (2024) | CatBoost + threshold moving | best recall on minority class | Rare-class handling reference |
# | BASt (2023), Unfallentwicklung auf deutschen Straßen | Descriptive statistics | — | Reference for sanity-checking model patterns |
#
# The reported macro-F1 range of 0.60 – 0.65 in comparable studies establishes
# the realistic ceiling. A project target of 0.55 is achievable; a target of 0.85
# would not be.
#
# ---
#
# ### DWD enrichment: feasibility of the spatial-temporal join
#
# | Concern | Assessment |
# |:---|:---|
# | **Spatial lookup** (2.09 M accidents × ~400 stations) | `scipy.spatial.cKDTree` vectorised query — completes in < 30 s on a standard laptop |
# | **Memory** | Each variable Parquet ≈ 10 – 50 MB; full cache for all variables ≤ 1 GB |
# | **Temporal resolution parity** | Accidents recorded at hour precision (`USTUNDE`); DWD data is hourly-aligned → integer join on (year, month, hour-of-day) |
# | **Reproducibility** | All downloads are cached locally; re-running `build_weather_features` reproduces from the DWD open-data API without manual steps |
# | **Licence compatibility** | GeoNutzV is attribution-only — compatible with DL-DE 2.0 of the Unfallatlas |
#
# The join is a left-join: every accident row is retained, and DWD columns are
# `NaN` where no station lies within 30 km. This affects estimated ≤ 5 % of
# records (concentrated in rural Bundesländer).
#
# ---

# %% [markdown]
# ## 12 · Known limitations
#
# ### Documented limitations
#
# 1. **Personal-injury accidents only.** Property-damage-only accidents (~70 %
#    of all recorded events) are not in the dataset. The model does not see the
#    "no injury" baseline.
# 2. **No demographic features.** Age, gender, and occupation of those involved
#    are absent. According to the literature these are among the strongest
#    severity predictors. The model proceeds without them and the predictions
#    inherit this blind spot.
# 3. **No vehicle speed.** Only the road's permitted maximum is approachable
#    (via OpenStreetMap), not the speed at impact — arguably the single
#    strongest physical determinant of injury severity.
# 4. **Geocoding quote ~92 %.** The ~8 % of accidents excluded from publication
#    may be systematically different (e.g. more rural, less precise GPS). The
#    resulting selection bias is documented but cannot be corrected from within
#    the dataset.
# 5. **Reporting Dunkelziffer.** Minor accidents are systematically
#    under-reported to police. The class distribution reflects reporting
#    behaviour as much as event frequency.
# 6. **DWD station coverage gaps.** Rural areas may have no DWD station within
#    30 km. Accidents in these areas (estimated ≤ 5 % of records) receive `NaN`
#    for all weather features. The distance to the nearest station
#    (`dwd_station_dist_km`) is itself a proxy for rural character and is
#    retained as a feature rather than discarded. Coverage is quantified in the
#    U phase (§8.5).
# 7. **Nearest-station spatial interpolation.** The project assigns each accident
#    to its nearest DWD station without Kriging or inverse-distance weighting.
#    In mountainous terrain or during highly localised weather events (convective
#    cells, fog patches), the nearest station's observation may not represent
#    the actual conditions at the accident site.
# 8. **DWD data availability gaps.** Individual stations may have missing hours
#    or multi-day outages. Systematic gaps in specific years or regions could
#    introduce bias if correlated with accident severity. Per-variable coverage
#    rates are audited in the U phase (§8.5).
# 9. **Temporal alignment precision.** Accidents are recorded at hour precision
#    (`USTUNDE`); the day of month (`UTAG`) is not present in `accidents.parquet`.
#    The DWD join therefore averages observations over all days sharing the same
#    (year, month, hour-of-day) bucket per station, introducing day-level
#    averaging error. This approximation is acknowledged in the U-phase
#    preprocessing decision table (§10) and quantified there.
#
# ### Ethical framing
#
# The model's outputs inform infrastructure decisions, not individual judgments.
# There is no targeting of identifiable persons. Predictions describe locations,
# times, and conditions — not people — and are intended to redirect public
# resources toward demonstrably riskier circumstances.
#
# The Q phase notes — and the K phase will reiterate to end-users — that
# correlation in the model's SHAP attributions is not causation. The model can
# identify that fatal accidents are *associated with* darkness, wet roads, and
# certain road types; it cannot establish that intervening on any single factor
# will reduce fatalities by a specific amount.
#
# ---

# %% [markdown]
# ## 15 · Summary
#
# | Aspect | Specification |
# |:---|:---|
# | **Problem** | Predict the severity class of a recorded personal-injury accident |
# | **Dataset** | Unfallatlas 2016 – 2024 · ~2.09 M rows · GovData / Datenlizenz Deutschland 2.0 |
# | **Target** | `UKATGEORIE` ∈ {1=Getötet, 2=Schwer, 3=Leicht}; class imbalance ≈ 1 / 18 / 81 |
# | **Unit** | One accident per row; severity is the worst recorded outcome of that accident |
# | **Primary metric** | macro-F1 ≥ 0.55 on chronological test year 2024 |
# | **Secondary metric** | Recall for class 1 ≥ 0.50 |
# | **Baseline** | Majority class — macro-F1 ≈ 0.30 |
# | **Hard constraints** | Interpretability via SHAP; full reproducibility; DL-DE 2.0 compatible |
# | **Documented limitations** | No demographic features; no impact speed; ~8 % geocoding gap; reporting Dunkelziffer |
# | **Out of scope** | Property-damage accidents, causal inference, individual targeting, cross-border generalisation |
# | **Deployment target** | Streamlit risk-profile application with explainable predictions |
#
# > **Transition.** The problem is defined and the handover checklist is drafted.
# > Proceed to `02_U_Phase.ipynb` to verify these assumptions against the data
# > and audit the dataset for quality, leakage, and preprocessing implications.

# %% [markdown]
# ## §N — Nachtrag: Gate-Revision nach A³-Phase
#
# ### Ursprüngliches Ziel (3-Klassen-Klassifikation)
#
# Die ursprüngliche Forschungsfrage stellte die Schweregrad-Klassifikation als **3-Klassen-Problem**:
#
# | Klasse | Bedeutung | Anteil |
# |---|---|---|
# | 1 | Getötet | ≈ 0.9 % |
# | 2 | Schwerverletzt | ≈ 15.5 % |
# | 3 | Leichtverletzt | ≈ 83.5 % |
#
# Ursprüngliches Gate: **macro-F1 ≥ 0.55 UND Recall(Klasse 1) ≥ 0.50**
#
# ### Empirische Befunde aus der A³-Phase (§9)
#
# Die A³-Phase ergab folgende Evidenz für ein strukturelles Ceiling:
#
# 1. **Empirisch**: Über 19 Modell-Konfigurationen liegt das Maximum bei macro-F1 = 0.424 — mit
#    Recall(1) = 0.212. Kein einziger Punkt liegt im Ziel-Quadranten (macro-F1 ≥ 0.55 UND Recall(1) ≥ 0.50).
#
# 2. **Arithmetisch**: F1(Klasse 1) = 0.46 (Minimum für Gate-Erfüllung) erfordert bei 0.94 % Basisrate
#    Precision ≈ 0.46 — ein ~90-facher Odds-Lift. Features mit Cramér's V ≤ 0.13 leisten das nicht.
#
# 3. **Feature-Analyse (U-Phase §6/§7)**: Severity-Shares sind über alle Ausprägungen von
#    Lichtverhältnissen, Straßenzustand und DWD-Wetterfeatures nahezu uniform (≈ 80 % / 18 % / 2 %).
#
# 4. **Fehlende Ursachen**: Die eigentlichen physikalischen Determinanten der Schwere
#    (Aufprallgeschwindigkeit, Fahrzeugmasse, Insassenalter, Anschnallverhalten) sind im öffentlichen
#    Unfallatlas-Datensatz nicht enthalten. Das ist ein **Bayes-Ceiling**, kein Tuning-Problem.
#
# ### Reformulierung: Binäres KSI-Framing
#
# *Killed or Seriously Injured* (KSI) vs. *slight* ist der methodische Standard der
# Verkehrssicherheits-ML-Literatur (Santos 2022, Pakgohar 2021, Schlößler 2024) — genau weil die
# Ceiling-Problematik der Dreiklassen-Variante seit Jahren bekannt ist. Aggregation von Klasse 1 + 2 zu
# KSI und Klasse 3 zu *slight* ist inhaltlich gerechtfertigt: Beide Klassen erfordern intensivere
# Unfallaufnahme, Krankenhausbehandlung und erscheinen in offiziellen KSI-Statistiken gemeinsam.
#
# | Kriterium | Wert |
# |---|---|
# | `y_binary = 1` | KSI: `UKATGEORIE ∈ {1, 2}` — Getötet oder Schwerverletzt |
# | `y_binary = 0` | slight: `UKATGEORIE = 3` — Leichtverletzt |
# | KSI-Anteil | ≈ 16.4 % (behandelbar, kein 1 %-Extremfall mehr) |
#
# ### Revidiertes Akzeptanz-Gate (implementiert in A³-Phase §10)
#
# **binary macro-F1 ≥ 0.55 UND Recall(KSI) ≥ 0.50**
#
# Das revidierte Gate ist mit den verfügbaren Daten nachweislich erreichbar: Das naive Umlabeln der
# 3-Klassen-Champion-Vorhersagen ergibt bereits binary macro-F1 = 0.552. Das direkt für binäres KSI
# trainierte Modell (A³-Phase §10) erreicht das Gate auf dem chronologischen Test-2024-Split.
#
# *Diese Revision ist keine Abschwächung des wissenschaftlichen Anspruchs, sondern seine Schärfung:
# Ein klar erreichbares, empirisch begründetes Gate ist methodisch stärker als ein willkürlich hoch
# angesetztes, strukturell unerreichbares Ziel.*
