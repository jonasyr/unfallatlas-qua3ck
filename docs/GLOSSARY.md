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

**KSI (Killed or Seriously Injured)**
The binary target actually deployed by this project: `1` = KSI (`UKATGEORIE` ∈ {1, 2}), `0` = slight (`UKATGEORIE` = 3). KSI is the standard aggregation in road-safety work because it marks the boundary at which infrastructure measures are justified. The reframing from the original three-class target to binary KSI was a pre-registered fallback defined in Phase Q (hypothesis H3), not a post-hoc rescue: the three-class acceptance gate proved unreachable (see *Three-class ceiling*). The positive class rises from ~1 % (fatal only) to 18.9 % under this framing, which is what makes reliable estimation possible. The chronological split and every leakage boundary were left untouched; only the target definition changed.

**Three-class ceiling**
The finding that the original acceptance gate (macro-F1 ≥ 0.55 **and** recall(class 1) ≥ 0.50 simultaneously) is not reachable from the public Unfallatlas features. Two independent lines of evidence converge on this. Empirically, the best of 19 configurations reached macro-F1 0.424, and no configuration satisfied both criteria at once (high-macro-F1 candidates had recall ≈ 0.21 on the fatal class; high-recall candidates had macro-F1 ≈ 0.37). Arithmetically, clearing the gate would require the model to isolate a subgroup roughly 90 times more likely to be fatal than the base rate of 0.94 %, a concentration a strongest-association of Cramér's V ≈ 0.13 cannot produce. The cause is that the physical determinants of fatality (impact speed, occupant age, restraint use, vehicle mass, impact geometry) are absent from the dataset, which records only surrounding context. This was documented before the first model was trained.

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
A k-d tree data structure from `scipy.spatial` used to vectorise the nearest-station lookup. Naively, each of the 2.09 M accidents would need a distance computation against every station; a k-d tree partitions space recursively so entire subtrees can be excluded during search. Euclidean distance in radian-space is accurate to < 0.1 % for distances ≤ 30 km.

---

## OpenStreetMap Road-Context Enrichment

**OSM (OpenStreetMap)**
A free, openly licensed world map built and maintained by volunteer contributors. Published under the Open Database License (ODbL), which permits reuse with attribution and share-alike. This project's second enrichment source (after DWD): the German road network is downloaded via `osmnx` and used to describe the road environment around each accident. Unlike the accident data, OSM is not an official statistical product, so its completeness varies regionally.

**H3 (Hexagonal Hierarchical Spatial Index)**
Uber's hexagonal geospatial indexing system. It divides the earth's surface into hexagonal cells at multiple nested resolutions; resolution 8 cells (used here) cover ~0.7 km². Hexagons are preferred over a square grid because all six neighbours of a cell share the same centre-to-centre distance, whereas a square grid's diagonal neighbours sit further away than its orthogonal ones. The practical motivation is compute: a per-point nearest-road lookup for 2.09 M accidents against the full German road network is not feasible on a single workstation, so the network is aggregated once per federal state onto H3 cells, after which the per-accident join is a constant-time cell-ID lookup.

**OSM road-context features**
Five features aggregated per H3 cell (U phase §8.8) and joined to each accident by location:

- `osm_road_density` — road length per unit area in the cell; an indirect exposure proxy, since more road generally means more traffic
- `osm_way_count` — number of distinct OSM ways intersecting the cell
- `osm_maxspeed_max` / `osm_maxspeed_mean` — maximum and mean posted speed limit (km/h) across the cell's ways
- `osm_dominant_road_class` — the most prevalent OSM highway class in the cell (e.g. `residential`, `motorway`), treated as a categorical feature

**Present-day-network approximation**
OSM reflects the network as it exists today, while the accidents span 2016–2024, so a speed limit or road class may have changed since an accident occurred. This is a documented, accepted approximation (U phase §11, risk 5) rather than a leakage vector: OSM data does not depend on accident outcomes, so no information about the target flows backwards into the features.

---

## Data Quality

**Missing Values**
Entries that are null or empty. Per-column strategies are specified in the §10 preprocessing decision table (U phase).

**Sentinel Values**
Values such as `-999` or `-9999` used by DWD to signal invalid or missing measurements. Replaced with `NaN` before joining.

**Duplicates**
Repeated rows or OBJECTID values. Exact row duplicates are negligible in this dataset; OBJECTID is unique.

**Class Imbalance**
The three-class target distribution is approximately 1 % / 18 % / 81 % for classes 1 / 2 / 3, and stable to within one percentage point across all nine annual vintages — which is what makes a chronological split defensible in the first place. A naïve model trained without imbalance handling will exhibit *majority-class collapse*: it learns to always predict class 3 (minor injury), achieving ~81 % accuracy but macro-F1 ≈ 0.30 and recall 0.00 on classes 1 and 2. Mitigation options (class weights, SMOTE, ADASYN, ordinal decomposition, threshold moving) are evaluated in A³, where only class weighting proves effective. The binary KSI reframe eases but does not remove the problem: the positive class rises to 18.9 %.

**Selection Bias**
The observed sample is not drawn at random from the population of interest. Two instances are documented here. The dataset contains only accidents that could be geocoded and pass plausibility checks (~92 %), and geocodability plausibly correlates with location type, so the ~8 % excluded are not a random subset. Separately, the *Dunkelziffer* below means the sample is filtered by reporting behaviour before it is filtered by anything else.

**Dunkelziffer**
German term for the "dark figure" — the unknown quantity of accidents never reported to the police. Minor accidents are systematically under-reported, so the class distribution in the dataset reflects police reporting behaviour as much as true event frequency. This structural bias cannot be corrected from within the dataset.

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

**SHAP (SHapley Additive exPlanations)**
A game-theoretic framework that assigns each feature a contribution value for a specific prediction, based on the Shapley value from cooperative game theory. SHAP values satisfy additivity: the sum of all feature contributions equals the model's output minus the baseline. Computed in Phase C §6 for the **champion only**, using `shap.TreeExplainer` with the documented approximation setting (exact Tree SHAP is impractical on the champion's very deep trees) over a stratified Test-2024 sample. It deliberately does not stand in for cross-model evidence: SHAP operates on each model's own transformed feature space, so it cannot compare different models on equal terms. That role belongs to permutation importance. `src/unfallatlas/viz/shap_plots.py` is an empty stub; the SHAP analysis lives inline in the C-phase notebook, and the Streamlit app surfaces permutation importance rather than SHAP.

**Permutation Importance**
Shuffles one feature column across all rows, destroying its relationship with the target while leaving its marginal distribution intact, then measures how far model performance drops. A large drop means the model depended on that feature. Unlike a tree model's built-in impurity importance, it is **model-agnostic**: it works on the raw input columns through predict alone, so all four Phase C finalists can be compared on the same sample with the same metric. Persisted as `c_phase_permutation_importance.csv` and used by the Streamlit app's "Why This Prediction" page. The finalists' rankings agree only moderately (mean Spearman ≈ 0.37 against the champion, top-10 overlap ≈ 40 %), so the ordering below the top few features should not be over-interpreted.

**Gradient Boosting (XGBoost / LightGBM / CatBoost)**
An ensemble method that fits successive decision trees, each correcting the residual errors of the previous ones, optimising a differentiable loss function via gradient descent. LightGBM uses leaf-wise tree growth (faster on large datasets, more prone to overfitting); CatBoost handles categoricals natively without one-hot encoding. All three are candidates in A³ and all three reach the Phase C finalist round.

**Bagging vs. Boosting**
Two ways of combining many weak trees. *Bagging* (bootstrap aggregating, the basis of Random Forest) trains trees in parallel on random row and feature subsamples and averages them; individual trees overfit, but their errors are largely uncorrelated and average out. *Boosting* trains trees sequentially, each one fitting the residual error of the running sum. The distinction is visible in this project's results: the three boosting finalists disagree with each other on only ~5 % of cases but disagree with Random Forest on ~13 %.

**Random Forest**
A bagged ensemble of decision trees, each fitted on a bootstrap sample of rows and a random subset of features at each split. The deployed binary KSI champion: `n_estimators=180`, `max_depth=23`, `min_samples_leaf=8`, `class_weight="balanced"`.

**SVM (Support Vector Machine)**
A classifier that seeks the separating hyperplane with the largest margin to the nearest training points. The *kernel trick* allows non-linear separation by implicitly mapping into a higher-dimensional space without ever computing the coordinates there; RBF (Radial Basis Function) is the most common non-linear kernel. Added to the A³ candidate set to cover the course's kernel-method material. RBF-kernel training scales roughly cubically in row count, so it was fitted on an 8,000-row subsample rather than the full 1.55 M training rows; it also proved by far the least robust finalist to missing features.

**Class weights (`balanced`)**
Reweights the training loss inversely to class frequency, so errors on rare classes cost proportionally more. The only imbalance strategy in this project that actually worked: SMOTE, ADASYN, and the ordinal decomposition all collapsed recall on the fatal class to between 0.3 % and 3 %, while class weighting lifted it above 0.50.

**ADASYN (Adaptive Synthetic Sampling)**
A SMOTE variant that generates proportionally more synthetic examples in regions where the minority class is hardest to classify. Evaluated in A³ and rejected on the same evidence as SMOTE.

**Accuracy**
The fraction of all predictions that are correct. Deliberately *not* used as this project's headline metric: with a 1 / 18 / 81 class split, a model that always predicts the majority class scores ~81 % accuracy while being entirely useless.

**Precision, Recall, F1**
For a given class: *precision* is the share of predictions for that class that were correct; *recall* is the share of true members of that class that were found; *F1* is their harmonic mean. The harmonic mean is used rather than the arithmetic one because it collapses when either component is near zero (precision 1.0 with recall 0.0 gives F1 = 0, not 0.5), which is exactly the failure mode class imbalance produces.

**Recall gate**
The project's secondary acceptance criterion, applied as a **precondition rather than an objective**: candidates that miss it are eliminated, and the highest macro-F1 among the survivors wins. It exists because macro-F1 alone could in principle be reached through the two common classes while ignoring the critical one. Originally recall(class 1) ≥ 0.50 under the three-class target; recall(KSI) ≥ 0.50 after the binary reframe. The rule explains an otherwise surprising outcome: Random Forest has the *lowest* KSI recall of the tree and SVM candidates (0.540) yet still wins, because it clears the gate and then leads on macro-F1.

**Confusion Matrix**
A cross-tabulation of true against predicted class. A *false negative* here is a KSI accident classified as slight (the expensive error); a *false positive* is a slight accident flagged as KSI (the cheaper error, costing only attention). On Test 2024 the champion recovered 22,767 of ~44,200 true KSI accidents and missed 21,431, while flagging 51,887 slight accidents as KSI.

**Baseline**
A deliberately trivial comparison model. Two are used: random guessing (binary macro-F1 0.439) and always predicting the majority class (0.453). Anything above that line is genuine learned signal, which is what makes the champion's 0.608 interpretable as a real gain rather than an isolated number.

**Cramér's V**
A symmetric measure of association between two categorical variables derived from the chi-squared statistic, normalised to a 0 (no association) to 1 (one variable fully determines the other) range. Used in §6 and §8.7 of the U phase to quantify feature–target relationships. It is the appropriate counterpart to correlation here because the features are nominal and unordered, so Pearson correlation is undefined. Conventional reading at these degrees of freedom: < 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, > 0.5 strong. The strongest feature–target association in this dataset is ≈ 0.13, at the low end of "weak", and this is the central finding of the data phase.

**Chi-squared test**
Tests whether the observed joint distribution of two categorical variables departs from what independence would predict. The statistic underlying Cramér's V.

**Entropy**
An information-theoretic measure of uncertainty, in bits. `H(Y)` is the uncertainty about the target knowing nothing; `H(Y | X)` is what remains once feature X is known.

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
A cross-validation strategy that always trains on past folds and validates on future folds, preserving the time-series property within the training window. Considered during Q/U planning but not used in the final pipeline: because the accident year is the natural time unit here, `GroupKFold` grouped on `UJAHR` achieves the same protection more directly and is what A³ actually runs.

**Holdout set**
Data withheld from every training and selection decision until a single final evaluation. Here: Test 2024. Its entire value rests on being used exactly once — used repeatedly to choose between models, it becomes a second validation set and no longer yields an unbiased generalisation estimate.

**Model-selection leakage**
Leakage that occurs through the *selection procedure* rather than through the features. No data point crosses into training, yet information still flows from the test set into the decision of which model to ship. This is why the Phase C decision matrix ranking XGBoost ahead of Random Forest did **not** trigger a champion swap: reporting a test number for XGBoost would have meant evaluating Test 2024 a second time and reporting the better of two attempts, which would retroactively invalidate the champion's own reported test metrics as well. The operating rule is that validation decides and the test set only confirms. XGBoost is recorded instead as a candidate for a future, pre-registered comparison on a not-yet-published year.

**SMOTE (Synthetic Minority Over-sampling Technique)**
An oversampling strategy that synthesises new minority-class samples by interpolating between existing nearest neighbours in feature space, rather than simply duplicating rows. Used as one of the imbalance-mitigation candidates in A³ alongside class weights and threshold moving. Must be applied inside the training fold only (never to validation or test data) to avoid data leakage.

**StandardScaler**
A sklearn preprocessing step that transforms a numeric feature to zero mean and unit variance: `(x − μ) / σ`. Parameters (μ, σ) are fit on training data only and applied to val/test to prevent leakage. Used for continuous features fed to distance-based or linear baselines; not required for tree-based models. Implemented inside the sklearn `Pipeline`.

**log1p transform**
`log(1 + x)`: a variance-stabilising transform for right-skewed non-negative features. The `+1` shift makes the transform defined at zero (unlike `log(x)`). Applied to `dwd_precip_mm` and `dwd_visibility_m` to reduce the influence of extreme outliers before scaling.

**sklearn Pipeline**
A `sklearn.pipeline.Pipeline` chains preprocessing steps and a model estimator into a single object. All `fit`-based preprocessing steps (StandardScaler, target encoding, imputation statistics) are fitted only on the training fold and then applied to val/test — preventing train-test contamination. A³ wraps all preprocessing in a Pipeline so the U-phase preprocessing decisions are never accidentally applied globally.

**Decision threshold / Threshold Moving**
A classifier outputs a probability; the threshold turns it into a decision. The default 0.5 is a convention, not an optimisation. A³ swept 81 candidate values and selected the best subject to recall(KSI) ≥ 0.50, giving **0.49860**. Its closeness to 0.5 is a consequence of class weighting having already roughly balanced the probabilities. Because the threshold encodes how expensive a missed KSI accident is relative to a false alarm, it is a value judgement that properly belongs to the deploying organisation rather than to the developer.

**Ordinal Classification (Frank–Hall decomposition)**
Decomposes a K-class ordinal target into K−1 binary "is y greater than threshold i" classifiers, then recovers per-class probabilities by differencing consecutive cumulative probabilities. Exploits the natural ordering of `UKATGEORIE` (1 < 2 < 3) rather than treating the classes as unordered categories. Elegant in principle, but the worst-performing strategy tried here, at recall 0.003 on the fatal class.

**Multi-objective optimisation and the Pareto front**
Optimising several objectives at once yields not one best solution but a *Pareto front*: the set of solutions where improving one objective necessarily worsens another. A³ attempted a multi-objective retune of the champion and did not promote it, under a promotion rule fixed in advance — a candidate is adopted only if it is no worse on **both** axes. The retune improved recall (0.5187 vs. 0.5053) but lost macro-F1 (0.6057 vs. 0.6083), so it was rejected and retained as documented negative evidence rather than deleted.

**Odds and base rate**
*Odds* are `p / (1 − p)`; the *base rate* is a class's frequency in the population. Both appear in the arithmetic-ceiling argument: a base rate of 0.94 % corresponds to odds ≈ 0.0095, while the precision the three-class gate demands corresponds to odds ≈ 0.72, a lift of roughly 90×. The figure is an order-of-magnitude estimate (≈ 76–90 depending on rounding), and the argument rests on the magnitude, not the decimals.

**Optuna**
A hyperparameter-optimisation library using sequential model-based search (TPE — Tree-structured Parzen Estimator — by default). Used in A³ to tune the winning (model, imbalance-strategy) combination of each champion candidate, bounded to a fixed trial count per family to respect the project's single-workstation compute budget.

**GroupKFold**
A cross-validation splitter that guarantees all rows sharing a group value (here: `UJAHR`, the accident year) fall in the same fold. Used instead of a random `StratifiedKFold` to prevent a model from training and validating on the same year, which would leak temporal structure during model selection.

**Champion candidates**
The (model, configuration) combinations carried forward from Stage 0/1 into the imbalance-strategy comparison and hyperparameter tuning, selected by `select_best_candidate()`: highest validation macro-F1 among candidates clearing the recall gate, not by macro-F1 alone. Under the abandoned three-class target this ruled Random Forest out (highest raw macro-F1, recall far below the gate) in favour of CatBoost and LightGBM. Under the binary KSI target the same rule selects **Random Forest** (`binary_random_forest_balanced`) from the ten-candidate field, and that is the model actually deployed.

**Deployed champion (binary KSI)**
Random Forest with `class_weight="balanced"`, tuned over 20 Optuna trials to `n_estimators=180`, `max_depth=23`, `min_samples_leaf=8`, at threshold 0.49860. Validation 2023: macro-F1 0.6083, recall(KSI) 0.5053. Test 2024: macro-F1 0.6039, recall(KSI) 0.5151. Both acceptance gates pass, and the small validation-to-test gap indicates the validation set was not overfitted during selection.

---

## Evaluation and Model Comparison (Phase C)

**Weighted decision matrix**
The structured finalist comparison in Phase C. Each criterion is min-max normalised to 0–1 and combined under fixed weights; cost criteria such as latency are inverted first so that higher always means better. Two of the six nominal criteria (interpretability, training cost) were never measured, and rather than filling them with subjective estimates that would colour the outcome, they are dropped automatically and the remaining weights renormalised. The measured result puts XGBoost ahead (0.561) and Random Forest last (0.500); it is reported rather than suppressed, and deliberately does not change the deployed model (see *Model-selection leakage*).

**Min-max normalisation**
Rescaling to a 0–1 range via `(value − min) / (max − min)`, so criteria measured in different units can be combined in a weighted score.

**Latency (ms/1k)**
Milliseconds required for 1,000 predictions; the throughput criterion in the decision matrix, and a cost criterion (lower is better).

**Robustness probe**
Measures prediction stability when inputs degrade the way they realistically do in production. Five features that can plausibly fail (for example when the OSM enrichment is unavailable or a weather station reports nothing) are set to `NaN` one at a time, and the share of predictions that flip class is recorded. Random Forest is the most stable finalist at 3.77 %, the boosting models fall between 7 % and 9 %, and the RBF-kernel SVM exceeds 52 % on a single missing feature — a strong practical argument for tree ensembles in deployment.

**Pairwise disagreement**
The share of cases where two models assign different hard classes. Random Forest against the boosting models is ~13 %; the boosting models against each other only ~5 %. The three boosting models are effectively variants of one idea, while Random Forest is the methodological outlier — which also makes it the more useful ensemble partner, since uncorrelated errors cancel better.

**Spearman correlation**
Correlation computed on ranks rather than raw values; the appropriate tool for comparing two feature-importance orderings.

**Jaccard index**
Size of the intersection divided by size of the union of two sets. Used here to quantify the overlap between two models' top-10 important features.

**Error slice**
A subset of the test data defined by a feature value (for example a single lighting condition or accident type), evaluated separately to locate where the champion systematically underperforms rather than relying on a single aggregate score.

---

## Deployment and Reproducibility (Phase K)

**Inference contract** (`data/processed/c_phase_inference_contract.json`)
A machine-readable handover file that fixes everything a consuming application needs: the 30 required input columns with dtype, provenance and permitted value range, the decision threshold, and a SHA-256 checksum for each model artefact. It is generated from the real training data rather than written by hand, which prevents the usual failure of reconstructing these details incorrectly from scattered notebook cells.

**SHA-256**
A cryptographic hash producing a fixed-length fingerprint of arbitrary content; any change to the content changes the hash completely. Recomputed on load and compared against the contract, so metadata can never silently describe a model file that has since been replaced.

**Artefact**
A committed file that is the persisted output of an analysis step: the joblib model, the metric CSVs, the inference contract. The Streamlit app is built entirely from artefacts, so it runs without executing a single notebook.

**joblib**
The serialisation format used for fitted scikit-learn pipelines. The binary champion is ~408 MB, which is why it is tracked through Git LFS.

**Git LFS (Large File Storage)**
A Git extension that stores large binary files outside the repository history, leaving only a pointer in the commit. A fresh clone needs `git lfs pull` before the model and parquet files are actually present.

**Parquet**
A columnar file format: data is stored column by column rather than row by row, which compresses well and makes queries touching a few columns very fast.

**DuckDB**
An embedded analytical database, conceptually SQLite but column-oriented. It runs SQL directly against Parquet files with no import step, which is how the app reads from the 2.09 M-row `accidents.parquet` without ever loading it into memory.

**Streamlit**
A Python framework that turns a script into a web app. The whole script re-executes on every interaction, so caching decorators (`@st.cache_data`, `@st.cache_resource`) are what keep expensive work from repeating.

**Relative risk and shrinkage (Overview risk map)**
The risk map colours each grid cell by how its KSI rate compares to the national rate of **18.91 %**, not by whether KSI exceeds 50 % in absolute terms. The absolute rule was abandoned because it required 2.64× the national rate to fire and only 123 of 4,857 cells passed it, leaving 97.5 % of the map a single colour. Raw ratios are unusable on small cells, so each rate is *shrunk* toward the national baseline by adding `k` pseudo-observations at exactly the national rate: with many real accidents they barely matter, with few they pull the estimate strongly toward the average. This is a Bayesian estimate with the national rate as prior. `k = 20` was chosen by measurement, not taste (`k = 10` still admitted 6-accident cells into the top band, `k = 50` collapsed it to 52 cells). Sample size is additionally shown as fill opacity, so cells resting mostly on the prior look visibly less certain.

**AppTest** (`streamlit.testing.v1.AppTest`)
Streamlit's own test harness. It executes the app script and inspects the resulting widget tree, but runs **no browser**, so it cannot observe JavaScript errors — it reported zero exceptions while a map was rendering blank in real browsers. This is why a headless-Playwright test asserting zero `pageerror` events was added as a mandatory gate: a green test is only worth what it actually executes.

**Audit mode**
The notebooks re-verify every persisted metric against the value recorded in the artefacts and abort if any differs by more than 1e-9. Figures quoted in the documentation and presentation are therefore checked on each run rather than transcribed once.

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
A single evaluation of the frozen champion on the held-out 2024 test set, comparison against baselines and the three other finalists (ROC and operating-point analysis, weighted decision matrix, robustness probe, permutation importance, champion SHAP, error slices), documentation of limitations, and the inference contract handed to K.

**Phase K (Knowledge Transfer)**
Delivery artefacts: the four-page interactive Streamlit app, the exported notebook presentation site on GitHub Pages, the German presentation package (handout, slide deck, speaker notes, cue cards), and this documentation.

**Iterative process**
QUA³CK is not a single linear pass. A later phase is allowed to overturn an earlier decision, and doing so is the process working rather than failing. This project's central example is A³ invalidating the three-class target defined in Q, along the fallback path that Q itself had pre-registered.

**DIG Framework**
A sub-framework for the U phase: **D**escription (inspect structure and samples), **I**ntrospection (formulate questions, identify limitations), **G**oal setting (decide whether data is suitable and define next steps).

**Vision Zero**
A road-safety policy goal — originally Swedish, adopted as EU policy — that targets zero road fatalities and serious injuries by 2050. Referenced in the Q phase as the policy context in which this project's outputs (corridor-level risk scores) would be consumed by public-sector analysts.

**BASt (Bundesanstalt für Straßenwesen)**
Germany's Federal Highway Research Institute. Publishes the annual *Unfallentwicklung auf deutschen Straßen* report, which is used in this project to sanity-check model output patterns (e.g. the 2020 COVID-19 accident-count dip).
