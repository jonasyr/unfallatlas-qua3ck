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
# # "Unfallatlas Deutschland": Q-Phase
#
# **Phase:** Question (Q) · 1 of 5 · QUA³CK
# **Goal of this notebook:** define the problem, the target, the success criteria,
# the constraints, and the data context: *before any data is opened or any model
# is trained*. The output is a written, testable problem definition that the
# remaining QUA³CK phases can build against.
#
# > The Q phase is the cheapest place to fix a mistake. Every later phase is
# > downstream of this document.
#
# ---

# %% [markdown]
# ### Position in the QUA³CK process
#
# | Phase | Notebook | Purpose | Status |
# |:---|:---|:---|:---:|
# | **Q**: Question | `01_Q_Phase.ipynb` | Problem definition, target, metrics, constraints | ✓ |
# | **U**: Understanding | `02_U_Phase.ipynb` | Schema audit, EDA, quality, leakage probes, preprocessing decisions | → next |
# | **A³**: Algorithm / Adapt / Adjust | `03_A3_Phase.ipynb` | Baselines, boosting models, imbalance strategies, tuning | pending |
# | **C**: Conclude & Compare | `04_C_Phase.ipynb` | SHAP, model comparison, limitations | pending |
# | **K**: Knowledge Transfer | `app/streamlit_app.py` | Interactive risk-profile application | pending |
#
# ---

# %% [markdown]
# ## 1 · Problem context
#
# Each year, German police record roughly **270,000 road accidents with personal
# injury**. About 1 % of these are fatal, 18 % cause serious injury, and 81 %
# cause only minor injury. The Statistisches Bundesamt publishes every one of
# these accidents: georeferenced, hour-stamped, with administrative codes for
# road condition, lighting, accident type, and the transport modes involved. The result is
# the **Unfallatlas Deutschland**, an open dataset spanning 2016-2024.
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
# > **Which spatiotemporal, environmental, and infrastructure factors are associated
# > with the severity of a road accident in Germany, and can interpretable
# > machine-learning models predict a decision-relevant severity outcome reliably
# > enough to support prevention planning?**
#
# The question begins with the original three-class severity outcome and remains
# open to an evidence-driven operational reformulation if the public features cannot
# separate the rarest class reliably. Sections 5 and 8 define that staged policy
# before the report evaluates it.

# %% [markdown]
# ## 3 · Hypotheses
#
# Three testable predictions derived from the research question. Each is evaluated against evidence in the U phase (EDA) or the A³ phase (model results). A hypothesis is *not* a prior commitment to what A³ will find. It is a structured expectation that can be confirmed, weakened, or falsified by the data.
#
# ---
#
# **H1: Temporal and environmental severity shift**
#
# > Night-time accidents (22:00-05:00) and accidents recorded under adverse meteorological conditions (precipitation > 0 mm, visibility < 2 000 m, or icy or snowy road surface) will show a statistically higher share of fatal and serious-injury outcomes (`UKATGEORIE` ∈ {1, 2}) than daytime accidents on dry roads. The association between darkness, adverse weather, and severity will persist after controlling for transport mode and accident type.
#
# *Verifiable in:* U phase §7 (hourly severity profile), §8.6 (weather distributions), and §8.7 (weather associations with `UKATGEORIE`).
#
# ---
#
# **H2: Location and spatial-context signal**
#
# > Location and spatial-context features, including coordinates, distance to the assigned weather station, and OSM road features, will add measurable predictive signal after controlling for accident type and transport mode.
#
# U tests the observable geographic concentration and proxy associations only. Higher travel speed, infrastructure quality, and emergency response time remain plausible mechanisms, but this dataset does not measure them and the report does not claim to identify them.
#
# *Verifiable in:* U phase §8 (geographic coverage and spatial proxies), A³ §12 (persisted champion feature evidence), and C §6 (cross-model permutation and SHAP evidence).
#
# ---
#
# **H3: Staged predictive feasibility**
#
# > Interpretable machine-learning models will outperform trivial and linear
# > baselines under chronological evaluation. The original three-class target is
# > tested first against macro-F1 ≥ 0.55 and fatal recall ≥ 0.50. If the public
# > predictors cannot clear that gate for structural reasons, a model trained
# > directly for KSI versus slight injury will be evaluated against the
# > corresponding binary gate without changing the temporal split or leakage
# > boundary.
#
# *Verifiable in:* U phase (target viability warning), A³ phase (model search and
# gate decision), and C phase (comparison, error analysis, and explanations).
#
# ---

# %% [markdown]
# ## 4 · Prediction goal
#
# For each police-recorded personal-injury road accident in Germany from 2016 to
# 2024, estimate severity from conditions available when the report is created:
# location, time, lighting, road condition, accident type, transport modes, weather,
# and road context. The model must be evaluated on the chronologically held-out
# 2024 test year and remain interpretable for municipal road-safety planning.
#
# The project initially asks for all three recorded severity classes. If that target
# fails the feasibility gate for structural reasons, the staged policy in section 5
# permits a pre-defined KSI-versus-slight operational target without changing the
# features, split, or evaluation discipline.

# %% [markdown]
# ## 5 · Target definition and staged feasibility policy
#
# **Source column:** `UKATGEORIE` (the source dataset uses this spelling).
#
# | Code | English label | Source label | Operational meaning |
# |:---:|:---|:---|:---|
# | 1 | Fatal | Getötet | At least one person died within 30 days of the accident |
# | 2 | Serious injury | Schwer verletzt | At least one person required at least 24 hours of inpatient care |
# | 3 | Slight injury | Leicht verletzt | Injuries required medical attention but met neither definition above |
#
# The label is finalised after the 30-day fatality window and is never used as an
# input feature. All predictors must be observable when the police report is
# created; U audits this leakage boundary.
#
# ### Stage 1: original three-class research target
#
# The original question treats the three ordered levels separately. This is the
# scientifically informative formulation because it tests whether the public data
# can distinguish fatal from serious and slight outcomes. Both nominal multiclass
# and ordinal models are therefore legitimate candidates.
#
# ### Stage 2: evidence-driven operational revision
#
# The fallback target is defined in advance as **KSI versus slight injury**:
#
# | Binary value | Definition | Interpretation |
# |:---:|:---|:---|
# | 1 | `UKATGEORIE ∈ {1, 2}` | Killed or seriously injured (KSI) |
# | 0 | `UKATGEORIE = 3` | Slight injury |
#
# The fallback is specified here before modelling. U later assesses whether
# the available predictors make the original target doubtful, and A³ makes the
# formal activation decision after the three-class search.
#
# **Retrospective outcome after completing A³.** Across 19 three-class
# configurations, the best macro-F1 was 0.424 with fatal-class recall of 0.212,
# so none reached the original gate. U had also found weak standalone
# associations and documented missing physical determinants such as impact speed,
# occupant age, seat-belt use, and vehicle mass. A³ therefore activated the
# predefined KSI fallback.
#
# KSI is not an arbitrary relabel. Fatal and serious outcomes jointly define the
# high-consequence road-safety group used for prevention decisions. The original
# three-class analysis remains feasibility evidence, while the binary target
# supports the operational model.

# %% [markdown]
# ## 6 · Unit of analysis and prediction horizon
#
# | Aspect | Specification |
# |:---|:---|
# | **Unit of analysis** | One row = one police-recorded personal-injury accident |
# | **Prediction time** | Predictors available when the police report is created |
# | **Outcome horizon** | Worst recorded severity after the final 30-day fatality window |
# | **Granularity** | Hourly (`USTUNDE`), daily (`UWOCHENTAG`), monthly (`UMONAT`), yearly (`UJAHR`) |
# | **Spatial granularity** | WGS84 coordinates (`LON`, `LAT`) + 5-digit district code (`UKREIS`) |
# | **Coverage** | All of Germany, 2016-2024, with a documented ~8 % geocoding coverage exclusion |
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
# ## 8 · Success metrics and gates
#
# The evaluation policy is staged, but the chronological split and primary metric
# remain fixed throughout.
#
# ### Shared evaluation rules
#
# - **Primary metric:** macro-F1 on the held-out 2024 test year, so each class
#   contributes equally despite imbalance.
# - **Selection data:** thresholds and model choices are made on 2023 validation
#   data; Test-2024 is evaluated once after selection.
# - **Interpretability:** the selected operational model must support global and
#   case-level explanation in C.
#
# ### Stage 1: three-class feasibility gate
#
# **macro-F1 ≥ 0.55 and recall(fatal) ≥ 0.50.**
#
# This gate protects the rarest, highest-consequence class from majority-class
# collapse. It was not met: the best observed three-class macro-F1 was 0.424 and
# fatal recall was 0.212. That result is retained as a negative finding rather than
# hidden by the later reformulation.
#
# ### Stage 2: operational KSI gate
#
# **binary macro-F1 ≥ 0.55 and recall(KSI) ≥ 0.50.**
#
# This gate applies only after the three-class feasibility decision. It is
# achievable with the same public predictors and supports the operational question:
# which recorded accidents belong to the high-consequence KSI group?
#
# ### External impact metric
#
# The downstream metric is the number of high-risk location-time profiles that
# lead to a documented safety review or intervention. It is deferred to K because
# the present dataset cannot observe municipal actions.

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
# - Estimate true fatality counts. The under-reporting of minor accidents
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
# | **Publisher** | Statistisches Bundesamt (Destatis), coordinated with federal states |
# | **Distribution** | [GovData.de](https://www.govdata.de/suche/daten/unfallatlas) / Mobilithek |
# | **Format** | Shapefile + CSV per year; consolidated to Parquet for this project |
# | **Coverage** | 2016-2024 (9 vintages), all of Germany |
# | **Volume** | ~2.09 million personal-injury accidents |
# | **Licence** | Datenlizenz Deutschland: Namensnennung: 2.0 (CC-BY-equivalent) |
# | **Citation** | "Datenquelle: Statistische Ämter des Bundes und der Länder, Unfallatlas, 2016-2024" |
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
# ### Secondary source: DWD Climate Data Center (CDC)
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
# | **Licence** | GeoNutzV: Geodatenlizenz Deutschland (free; attribution required) |
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
# ## 11 · Feasibility and literature anchor
#
# ### Data adequacy
#
# | Question | Answer |
# |:---|:---|
# | Does the label exist? | Yes: `UKATGEORIE` is uniformly recorded |
# | When is the outcome label final? | After the 30-day fatality window; it is not a report-time predictor |
# | Are features observable at prediction time? | Yes; administrative codes are available when the report is created |
# | Is the volume sufficient? | Yes: 2.09 M rows, with the rarest class still ~21,000 samples |
# | Is there a literature anchor? | Yes: see below |
#
# ### Literature anchor
#
# | Reference | Method | Reported metric | Relevance |
# |:---|:---|:---|:---|
# | Santos et al. (2022), *Accident Analysis & Prevention* | XGBoost + SHAP on Portuguese accident data | macro-F1 ≈ 0.60 | Direct methodological precedent |
# | Pakgohar et al. (2021), *IATSS Research* | LightGBM + SMOTE | macro-F1 ≈ 0.62 | Imbalance treatment template |
# | Schlößler et al. (2024), *Accident Analysis & Prevention* | ML ensemble on German accident data | macro-F1 ≈ 0.65 | Comparable jurisdiction and features |
# | MDPI *Sustainability* (2024) | CatBoost + threshold moving | best recall on minority class | Rare-class handling reference |
# | BASt (2023), Unfallentwicklung auf deutschen Straßen | Descriptive statistics | Not reported | Reference for sanity-checking model patterns |
#
# The published results motivated the provisional 0.55 gate, but they do not
# guarantee that the three-class target is achievable here: target definitions,
# features, and evaluation designs differ. The gate is therefore a falsifiable
# feasibility test. The literature's frequent use of KSI versus slight injury also
# supports the pre-defined fallback if the rare fatal class cannot be separated
# with the public predictors.
#
# ---
#
# ### DWD enrichment: feasibility of the spatial-temporal join
#
# | Concern | Assessment |
# |:---|:---|
# | **Spatial lookup** (2.09 M accidents × ~400 stations) | `scipy.spatial.cKDTree` vectorised query: completes in < 30 s on a standard laptop |
# | **Memory** | Each variable Parquet ≈ 10-50 MB; full cache for all variables ≤ 1 GB |
# | **Temporal resolution parity** | Accidents recorded at hour precision (`USTUNDE`); DWD data is hourly-aligned → integer join on (year, month, hour-of-day) |
# | **Reproducibility** | All downloads are cached locally; re-running `build_weather_features` reproduces from the DWD open-data API without manual steps |
# | **Licence compatibility** | GeoNutzV is attribution-only: compatible with DL-DE 2.0 of the Unfallatlas |
#
# The join is a left-join: every accident row is retained, and DWD columns are
# `NaN` where no station lies within 30 km. This affects estimated ≤ 5 % of
# records (concentrated in rural federal states).
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
#    (via OpenStreetMap), not the speed at impact: arguably the single
#    strongest physical determinant of injury severity.
# 4. **Geocoding coverage ~92 %.** The ~8 % of accidents excluded from publication
#    may be systematically different (e.g. more rural, less precise GPS). The
#    resulting selection bias is documented but cannot be corrected from within
#    the dataset.
# 5. **Reporting under-reporting.** Minor accidents are systematically
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
# times, and conditions rather than people. They are intended to redirect public
# resources toward demonstrably riskier circumstances.
#
# The Q phase notes that the K phase must reiterate to end users that
# correlation in the model's SHAP attributions is not causation. The model can
# identify that fatal accidents are *associated with* darkness, wet roads, and
# certain road types; it cannot establish that intervening on any single factor
# will reduce fatalities by a specific amount.
#
# ---

# %% [markdown]
# ## 13 · Summary and U-phase handoff
#
# | Aspect | Final specification |
# |:---|:---|
# | **Problem** | Predict a decision-relevant severity outcome for a recorded personal-injury accident |
# | **Dataset** | Unfallatlas 2016-2024 · about 2.09M rows · GovData · Data Licence Germany 2.0 |
# | **Original target** | Three classes: fatal, serious injury, slight injury |
# | **Predefined fallback** | KSI (`UKATGEORIE ∈ {1, 2}`) versus slight injury (`UKATGEORIE = 3`); A³ decides whether to activate it |
# | **Unit** | One accident per row; the label is the worst recorded outcome |
# | **Evaluation** | Chronological Train 2016-2022, Validation 2023, Test 2024 |
# | **Gates** | Three-class feasibility: macro-F1 ≥ 0.55 and fatal recall ≥ 0.50; operational KSI: macro-F1 ≥ 0.55 and KSI recall ≥ 0.50 |
# | **Hard constraints** | Interpretability, reproducibility, licence compliance, and no personal data |
# | **Principal limitations** | No impact speed, occupant demographics, seat-belt use, or vehicle mass; geocoding and reporting bias |
# | **Deployment target** | Explainable Streamlit risk-profile application |
#
# > **U-phase handoff.** Audit whether the target shares are stable, quantify the
# > association available in the public predictors, verify the chronological and
# > leakage boundaries, and turn those findings into a target-independent
# > preprocessing contract for A³.
