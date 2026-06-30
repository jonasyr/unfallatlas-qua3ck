# AI Prompt for Phase U

## Build the U-Phase from the Q-Phase Contract

**Claude Code (Sonnet 4.6) (Effort: Medium) [AI_TOOL_DISCLOSURE.md](AI_TOOL_DISCLOSURE.md):**
```markdown
You are a senior Data Scientist, Machine Learning Engineer, curriculum designer, technical reviewer, and repository-quality documentation architect.

Your task is to implement the **U-Phase — Understanding the Data** of the QUA³CK process model for this repository.

This prompt will be executed in **Claude Code**, so you have direct access to the repository files, notebooks, source code, datasets, documentation, and the already completed **Q-Phase notebook**.

The **Q-Phase already exists** and must be treated as the authoritative project contract.

Your job is not to redesign the Q-Phase.
Your job is not to redesign the whole repository.
Your job is to create, revise, and complete the **U-Phase** so that it follows directly from the existing Q-Phase and prepares the project for A³ without drifting into modeling.

---

# PRIMARY OBJECTIVE

Create a complete, technically rigorous, visually polished, educational, and repository-ready implementation of the **U-Phase: Understanding the Data**.

The U-Phase must validate, inspect, and document the data foundations of the project defined in the Q-Phase.

The final result should be suitable for direct inclusion in a professional open-source educational repository.

The U-Phase should answer:

> “Given the project definition from Q, what does the data actually contain, how reliable is it, what patterns and risks are visible, and what preprocessing decisions are required before modeling?”

---

# REQUIRED REPOSITORY CONTEXT

Before making changes, read and use the repository context.

You MUST inspect:

* the existing Q-Phase notebook, especially `01_Q_Phase.ipynb` or its paired Jupytext `.py` file
* any existing U-Phase notebook, especially `02_U_Phase.ipynb` or its paired `.py` file
* repository README files
* project documentation
* data loading scripts
* preprocessing or feature scripts
* dataset files or metadata files
* existing visualization assets
* existing folder structure
* existing naming conventions
* any references to QUA³CK terminology

The Q-Phase defines the project contract. Use it to align the U-Phase with:

* the research question
* the prediction goal
* target variable `UKATGEORIE`
* unit of analysis
* stakeholders
* success metrics
* constraints
* data sources
* known limitations
* leakage boundary
* ethical framing
* Q-to-U transition

Do not contradict the Q-Phase unless the data proves a mismatch. If a mismatch exists, document it clearly as a U-Phase finding.

---

# PROJECT CONTEXT FROM Q-PHASE

The project is based on **Unfallatlas Deutschland** and focuses on predicting the severity class of personal-injury road accidents in Germany.

The Q-Phase defines:

* **Target:** `UKATGEORIE`
* **Target classes:**

  * `1` = fatal accident
  * `2` = serious injury
  * `3` = minor injury
* **Unit of analysis:** one police-recorded personal-injury accident
* **Coverage:** Germany, 2016–2024
* **Primary dataset:** Unfallatlas Deutschland
* **Secondary enrichment:** DWD hourly weather observations
* **Prediction goal:** predict accident severity from conditions available at report time
* **Primary metric for later A³:** macro-F1 on chronological test year 2024
* **Secondary metric for later A³:** recall for class 1
* **Known risks:** class imbalance, geocoding exclusion, reporting bias, DWD station gaps, temporal leakage, nearest-station approximation, day-level weather averaging
* **Excluded use cases:** individual targeting, insurance underwriting, law-enforcement targeting, causal claims, cross-border generalization

The U-Phase must verify and deepen these claims using the actual repository data.

---

# STRICT U-PHASE SCOPE

The U-Phase includes only data understanding, exploratory analysis, quality assessment, and preprocessing planning.

Include:

* dataset overview
* data source validation
* schema audit
* feature descriptions
* target variable inspection
* data type inspection
* dataset dimensions
* duplicate detection
* missing value analysis
* invalid value checks
* inconsistent value checks
* class balance analysis
* descriptive statistics
* categorical feature analysis
* numerical feature analysis
* temporal feature analysis
* spatial feature inspection
* weather-enrichment coverage inspection
* outlier analysis
* distribution analysis
* cardinality inspection
* skewness analysis
* correlation and association analysis
* relationship exploration
* leakage-risk analysis
* preprocessing requirements
* reproducible preprocessing plan
* Q-to-U validation findings
* U-to-A³ handoff checklist

You may discuss preprocessing decisions only at the planning level.

Allowed preprocessing-planning topics:

* missing-value handling strategy
* duplicate handling strategy
* invalid-value correction strategy
* datatype correction strategy
* scaling recommendations
* normalization recommendations
* categorical encoding considerations
* train/test contamination risks
* leakage-safe preprocessing orchestration

Do NOT include:

* model training
* model selection
* algorithm comparison
* baseline model implementation
* hyperparameter tuning
* cross-validation experiments
* SMOTE or imbalance treatment implementation
* SHAP analysis
* model evaluation
* confusion matrices
* ROC curves
* PR curves
* deployment
* Streamlit implementation
* MLflow
* CI/CD
* monitoring
* production infrastructure
* inference optimization
* advanced feature engineering
* final model pipeline construction

Those belong to later QUA³CK phases.

---

# IMPORTANT BOUNDARY RULES

Be extremely strict about QUA³CK phase boundaries.

The U-Phase may say:

* “This feature appears useful and should be considered in A³.”
* “This variable may require encoding before modeling.”
* “This missingness pattern creates a preprocessing requirement.”
* “This relationship should be tested in A³.”

The U-Phase must NOT say:

* “We train XGBoost here.”
* “This model achieves macro-F1.”
* “SHAP shows feature importance.”
* “This feature engineering improves performance.”
* “This is the final production pipeline.”
* “This dashboard will use these predictions.”
* “This hyperparameter setting is optimal.”

When uncertain, defer the topic to A³, C, or K instead of expanding U.

---

# TASK 1 — REPOSITORY REVIEW

Inspect the repository before editing.

Determine:

* what U-Phase files already exist
* whether `02_U_Phase.ipynb` or a Jupytext paired file exists
* how the Q-Phase is structured
* how QUA³CK terminology is used
* where datasets are stored
* how data is loaded
* what preprocessing already exists
* what visualizations already exist
* what documentation already exists
* what conventions should be preserved
* what content belongs elsewhere

Produce a concise internal understanding before making changes.

If there is already U-Phase content, improve it rather than blindly replacing it.

If no U-Phase exists, create it from scratch while matching the style, tone, and structure of the existing Q-Phase.

---

# TASK 2 — Q-TO-U ALIGNMENT

Use the existing Q-Phase as the starting contract.

Create a U-Phase section that explicitly validates the Q-Phase assumptions.

Check and document:

* whether `UKATGEORIE` exists
* whether class labels match the Q-Phase definition
* whether the unit of analysis appears to be one accident per row
* whether the year range matches 2016–2024
* whether the dataset size matches the Q-Phase claim
* whether the class imbalance matches the stated approximate 1 / 18 / 81 distribution
* whether required spatial columns exist
* whether required temporal columns exist
* whether DWD weather-enrichment columns exist
* whether DWD missingness behaves as expected
* whether `dwd_station_dist_km` exists and is interpretable
* whether the dataset contains columns that may leak the target
* whether any Q-Phase assumptions need correction

If the data contradicts the Q-Phase, document the discrepancy clearly and recommend whether Q or U should be updated.

---

# TASK 3 — DATASET OVERVIEW

Create a comprehensive dataset overview.

Include:

* dataset origin
* source files
* licensing notes
* data provenance
* row count
* column count
* year coverage
* geographic coverage
* target column
* feature groups
* primary keys or absence of a stable key
* storage format
* loading method
* reproducibility notes

Explain why each aspect matters for the ML workflow.

---

# TASK 4 — SCHEMA AUDIT

Perform a strict schema audit.

Include:

* column names
* inferred dtypes
* semantic types
* expected value ranges
* actual value ranges
* unique counts
* missing counts
* missing percentages
* examples of values
* feature group assignment

Group features into meaningful categories, for example:

* target
* temporal features
* spatial features
* administrative features
* accident-type features
* infrastructure or road-condition features
* transport-mode indicators
* DWD weather features
* derived support columns
* potential leakage-risk columns
* columns to exclude from modeling consideration

Document column meaning where known from the repository or official metadata.

Do not invent meanings. If uncertain, mark the field as requiring verification.

---

# TASK 5 — TARGET VARIABLE UNDERSTANDING

Analyze `UKATGEORIE`.

Include:

* class counts
* class percentages
* class labels
* imbalance ratio
* temporal stability of class distribution
* geographic variation of severity distribution
* missingness in target
* invalid target values
* whether the target matches the Q-Phase definition

Create appropriate visualizations such as:

* class distribution bar chart
* class distribution by year
* severity share by Bundesland if available
* severity share by month or hour if useful

Interpret every visualization.

Do not evaluate model performance.

---

# TASK 6 — DESCRIPTIVE STATISTICS AND FEATURE UNDERSTANDING

Perform descriptive analysis for all relevant feature groups.

Include:

* numerical feature summaries
* categorical feature summaries
* temporal feature distributions
* spatial coverage summaries
* transport-mode variable summaries
* weather feature summaries
* cardinality analysis
* skewness analysis
* rare-category inspection
* constant or near-constant feature detection

Explain:

* what each pattern suggests
* whether it is expected
* whether it creates a preprocessing requirement
* whether it creates a risk for modeling later

Do not perform feature selection as a modeling decision. Only flag candidates for later review.

---

# TASK 7 — MISSING VALUE ANALYSIS

Analyze missing values rigorously.

Include:

* missing count by column
* missing percentage by column
* missingness heatmap or matrix
* missingness by year
* missingness by region if applicable
* missingness by target class if applicable
* co-missingness patterns
* special focus on DWD weather columns
* special focus on spatial columns
* special focus on columns required for the prediction goal

Discuss possible missingness mechanisms:

* MCAR
* MAR
* MNAR

Only classify missingness when evidence supports it. Otherwise state that the mechanism is uncertain.

Recommend handling strategies, but do not implement final modeling-time imputation pipelines.

---

# TASK 8 — DATA QUALITY ASSESSMENT

Assess data quality across:

* completeness
* consistency
* validity
* uniqueness
* plausibility
* temporal coverage
* spatial coverage
* source consistency
* target integrity
* feature availability at prediction time

Check for:

* duplicate rows
* impossible dates or times
* impossible coordinates
* invalid category codes
* inconsistent data types
* impossible weather values
* implausible station distances
* columns with unexpected all-null or all-constant values
* values outside documented codebooks
* target leakage candidates

Document every issue found.

For each issue, include:

* issue description
* affected columns
* severity
* evidence
* recommended action
* phase where the action belongs

---

# TASK 9 — OUTLIER AND PLAUSIBILITY ANALYSIS

Identify outliers and suspicious values.

Include:

* statistical outliers
* domain outliers
* impossible values
* suspicious but valid values
* weather outliers
* spatial-distance outliers
* temporal anomalies
* highly rare categories

For each finding, recommend whether it should be:

* retained
* investigated
* corrected
* excluded
* deferred to A³ for robustness testing

Explain the reasoning.

Avoid aggressive removal recommendations unless values are clearly invalid.

---

# TASK 10 — RELATIONSHIPS, CORRELATIONS, AND ASSOCIATIONS

Explore relationships between features and between features and the target.

Include where appropriate:

* numeric correlation heatmaps
* categorical association analysis
* Cramér’s V for categorical variables
* target-wise distribution comparisons
* temporal severity patterns
* spatial severity patterns
* weather-severity relationships
* transport-mode-severity relationships
* road-condition-severity relationships

Important:

* Interpret associations as descriptive, not causal.
* Do not claim predictive performance.
* Do not perform model-based feature importance.
* Do not use SHAP.
* Do not select final features.
* Flag relationships that should be tested in A³.

---

# TASK 11 — VISUALIZATION SUITE

Create professional-quality U-Phase visualizations.

Use a consistent style across all plots.

Include appropriate examples from this list when relevant:

* class distribution bar chart
* yearly accident count chart
* monthly accident count chart
* weekday accident count chart
* hourly accident count chart
* severity-by-hour chart
* severity-by-month chart
* missingness matrix
* missingness heatmap
* numeric histograms
* KDE plots
* boxplots
* violin plots
* count plots
* categorical bar charts
* correlation heatmap
* Cramér’s V heatmap
* weather feature distributions
* DWD station-distance distribution
* spatial scatterplot or density map if coordinates exist
* Bundesland severity-share chart if regional code is available

For every visualization, provide:

* purpose
* what it shows
* interpretation
* key finding
* common interpretation pitfall
* whether it informs preprocessing or later A³

Do not create modeling visualizations.

All saved plots should use a clean, reproducible naming convention and should be stored in an appropriate U-Phase asset folder.

---

# TASK 12 — PREPROCESSING PLANNING

Create a preprocessing decision plan based on the U-Phase findings.

Include:

* datatype corrections
* target handling
* duplicate handling
* invalid-value handling
* missing-value strategy
* categorical encoding considerations
* scaling and normalization recommendations
* weather-column handling
* spatial-column handling
* high-cardinality-column handling
* rare-category handling
* leakage-risk exclusions
* train/test contamination precautions
* chronological split implications

For each decision, include:

* issue
* evidence from U
* recommended preprocessing action
* why it is appropriate
* phase where it should be implemented
* risk if ignored

Do not implement final model training pipelines in the U-Phase.

---

# TASK 13 — LEAKAGE PREVENTION

Identify and document possible leakage risks.

Include:

* target leakage
* temporal leakage
* preprocessing leakage
* train/test contamination
* duplicate leakage
* aggregation leakage
* weather-enrichment leakage
* post-accident-information leakage
* geographically encoded leakage risks
* leakage caused by fitting imputers, scalers, or encoders before the split

For each risk, include:

* description
* example in this project
* severity
* how to detect it
* how to prevent it
* which later phase must enforce it

Be especially careful with:

* `UKATGEORIE`
* post-outcome variables
* chronological test year 2024
* DWD joins
* preprocessing fitted on all years
* aggregate features computed using the full dataset

---

# TASK 14 — NOTEBOOK AND MARKDOWN QUALITY

The U-Phase notebook must be written as a polished educational chapter.

Use a structure similar to the Q-Phase:

* clear title
* phase position table
* concise goal statement
* numbered sections
* markdown explanations
* summary tables
* warnings
* tips
* callouts
* checklists
* interpretation after each major output
* transition to A³

Writing style:

* academically strong
* technically precise
* concise
* visually clean
* consistent with the Q-Phase tone
* restrained dry humor only if it improves readability
* no unnecessary jokes
* no filler text

The notebook should explain not only what is computed, but why it matters.

---

# TASK 15 — EXERCISES AND EDUCATIONAL TASKS

Add a short educational exercise section appropriate for a university-level repository.

Include exercises such as:

* identify a leakage-risk column
* interpret a missingness pattern
* explain why accuracy is misleading for the target distribution
* classify a variable as numerical, categorical, ordinal, temporal, or spatial
* decide whether an outlier should be removed
* propose a leakage-safe imputation strategy
* explain the difference between descriptive association and causal interpretation

For each exercise, include:

* learning goal
* task
* expected output
* common mistake

Keep exercises within U-Phase scope.

---

# TASK 16 — U-TO-A³ HANDOFF

End the U-Phase with a clear handoff to A³.

Include:

* validated dataset assumptions
* unresolved data issues
* preprocessing decisions to implement in A³
* leakage rules A³ must follow
* features requiring special treatment
* target imbalance implications
* visual findings worth testing in models
* issues deferred to C or K
* final readiness status

The handoff should make A³ easier without doing A³’s work.

---

# REQUIRED OUTPUTS

Produce or update the repository so that the U-Phase includes:

1. A complete `02_U_Phase.ipynb` notebook or matching existing naming convention
2. A paired Jupytext `.py` file if the repository uses paired notebooks
3. Clean Markdown sections
4. Reproducible data-loading code
5. Reproducible EDA code
6. Professional visualizations
7. Interpretation for every major output
8. Data quality report
9. Missing-value analysis
10. Target-variable analysis
11. Schema audit
12. Correlation and association analysis
13. Leakage-risk register
14. Preprocessing decision table
15. U-Phase checklist
16. Educational exercises
17. Final U-to-A³ handoff summary
18. Saved visualization assets, if appropriate
19. Any small helper functions needed for clean notebook execution

Do not create unnecessary complexity.
Do not add heavy infrastructure.
Do not add modeling code.

---

# REPOSITORY ORGANIZATION GUIDANCE

Preserve existing repository conventions wherever possible.

If creating new folders, prefer clear U-Phase-specific paths such as:

* `notebooks/02_U_Phase.ipynb`
* `notebooks/02_U_Phase.py`
* `reports/figures/u_phase/`
* `reports/tables/u_phase/`
* `docs/quaack/u_phase.md`

Only create paths that fit the existing repository structure.

Do not reorganize the whole repository unless absolutely necessary.

---

# QUALITY REQUIREMENTS

The final U-Phase must be:

* technically accurate
* reproducible
* visually polished
* clearly structured
* educational
* portfolio-grade
* compatible with the existing Q-Phase
* strict about phase boundaries
* useful for a real ML workflow
* ready to support the later A³ phase

Every important claim must be supported by repository evidence, code output, documentation, or a clearly stated assumption.

Avoid unsupported claims.

---

# FINAL RESPONSE FORMAT

After completing the repository work, respond with:

## Summary of Changes

Briefly list what was created or updated.

## Key U-Phase Findings

Summarize the most important data-understanding findings.

## Files Changed

List the files created or modified.

## Boundary Notes

Mention anything intentionally deferred to A³, C, or K.

## Remaining Risks

List unresolved issues or assumptions that should be revisited.

Do not paste the entire notebook into the final response unless explicitly requested.

```

## Final U-Phase Polish & Alignment Pass

**Claude Code (Sonnet 4.6) (Effort: Medium) [AI_TOOL_DISCLOSURE.md](AI_TOOL_DISCLOSURE.md):**

````md
You are working in Claude Code with full repository access.

Your task is to perform a final polish, consistency, alignment, and visualization-cleanup pass over the completed **U-Phase** implementation of the QUA³CK Unfallatlas project.

This is a follow-up task after the main U-Phase implementation has already been completed.

Do not redesign the whole project.
Do not rewrite the entire U-Phase from scratch.
Do not add modeling work.
Do not expand the scope beyond U-Phase documentation, EDA clarity, visualization quality, glossary consistency, README consistency, and Q/U alignment.

Your job is to make the existing work visually clean, internally consistent, human-readable, and repository-ready.

---

# Current Repository State

The repository already contains a completed or near-completed U-Phase implementation.

The previous U-Phase work included:

* Full U-Phase data analysis and EDA, especially sections §1–§9
* Notebook writing for all major U-Phase sections
* Creation of `GLOSSARY.md`
* Updates to `README.md`
* DWD CDC weather enrichment implementation in `src/unfallatlas/data/dwd.py`
* DWD station-list parsing
* nearest-station spatial lookup using `scipy.spatial.cKDTree`
* per-station DWD download logic with directory-listing URL discovery
* left-join weather enrichment pipeline
* weather features for:

  * temperature
  * precipitation
  * visibility
  * wind speed
  * DWD station distance
* U-Phase extension with:

  * §8.5 weather coverage
  * §8.6 weather distributions
  * §8.7 weather bivariate analysis
  * §9.4 temporal leakage probe
* §10 preprocessing decision table updated with DWD-related decisions
* §11 summary updated with DWD-related findings
* Q-Phase back-filled with:

  * data-source table
  * feasibility section
  * limitations section

Your task is to inspect all of this and fix remaining presentation, alignment, naming, labeling, documentation, and consistency issues.

---

# Primary Objective

Make the completed U-Phase package feel like a polished professional educational repository.

Focus especially on:

* visualization alignment
* plot layout
* human-readable labels
* figure consistency
* notebook readability
* section numbering consistency
* README consistency
* glossary consistency
* Q-Phase and U-Phase alignment
* DWD terminology consistency
* file naming consistency
* German/English terminology consistency
* avoiding U-Phase scope creep into A³, C, or K

The final state should be suitable for direct inclusion in a portfolio-grade open-source repository.

---

# Files and Areas to Review

Inspect and update as needed:

* `notebooks/02_U_Phase.ipynb`
* paired Jupytext file for the U-Phase, if present
* `notebooks/01_Q_Phase.ipynb`
* paired Jupytext file for the Q-Phase, if present
* `README.md`
* `GLOSSARY.md`
* `docs/DSB_Unfallatlas.md`
* `src/unfallatlas/data/dwd.py`
* any U-Phase helper scripts
* any generated U-Phase figures
* any generated U-Phase tables
* any reports or docs folders used by the U-Phase

Preserve the existing repository structure and naming conventions unless a small correction is clearly needed.

---

# Critical Scope Boundary

This is a cleanup and consistency pass.

Do NOT add:

* model training
* baseline models
* algorithm comparison
* SMOTE implementation
* target encoding implementation
* hyperparameter tuning
* SHAP analysis
* confusion matrices
* ROC curves
* PR curves
* test-set evaluation
* Streamlit application logic
* MLflow
* CI/CD
* production monitoring
* final deployment material

The U-Phase may discuss preprocessing decisions and leakage risks, but implementation of modeling pipelines belongs to A³.

---

# Task 1 — Fix All Visualization Layout and Alignment Issues

Review every visualization in the U-Phase notebook and saved figure outputs.

Fix:

* overlapping labels
* cramped titles
* unreadable legends
* truncated tick labels
* inconsistent plot sizes
* inconsistent colors
* too-small figures
* dense charts that should be horizontal
* raw coded category values shown without explanation
* raw feature names shown where human-readable labels should be used
* legends with unclear ordering
* severity classes ordered inconsistently
* charts whose exported version looks worse than the notebook version

Use `constrained_layout=True`, `tight_layout()`, larger figure sizes, horizontal bar charts, wrapped labels, or shortened labels where appropriate.

Do not remove useful plots unless they are genuinely redundant or misleading.

---

# Task 2 — Use Human-Readable Unfallatlas Labels Everywhere

Use the official Unfallatlas data description in:

`docs/DSB_Unfallatlas.md`

as the source of truth for coded categorical labels.

Add or centralize the following label dictionaries in the U-Phase notebook or an appropriate helper location if one already exists.

```python
# ── Human-readable labels for all coded categoricals ────────────────────────
# Source: Datensatzbeschreibung Unfallatlas (Stand 10.06.2025)

ULICHTVERH_LABELS = {
    0: "Tageslicht",
    1: "Dämmerung",
    2: "Dunkelheit",
}

STRZUSTAND_LABELS = {
    0: "Trocken",
    1: "Nass/feucht/schlüpfrig",
    2: "Winterglatt",
}

UART_LABELS = {
    0: "Anderer Art",
    1: "Zzs. ruhendes Fz.",
    2: "Zzs. vorausfahrendes Fz.",
    3: "Zzs. seitlich gleichfahrendes Fz.",
    4: "Zzs. Gegenverkehr",
    5: "Zzs. einbiegendes/kreuzendes Fz.",
    6: "Zzs. Fußgänger",
    7: "Aufprall Fahrbahnhindernis",
    8: "Abkommen nach rechts",
    9: "Abkommen nach links",
}

UTYP1_LABELS = {
    1: "Fahrunfall",
    2: "Abbiegeunfall",
    3: "Einbiegen/Kreuzen",
    4: "Überschreiten",
    5: "Ruhender Verkehr",
    6: "Längsverkehr",
    7: "Sonstiger Unfall",
}

MODE_LABELS = {
    "IstRad": "Fahrrad",
    "IstPKW": "PKW",
    "IstFuss": "Fußgänger",
    "IstKrad": "Kraftrad",
    "IstGkfz": "Güterkraftfahrzeug",
    "IstSonstig": "Sonstiges Fahrzeug",
}

DWD_COL_LABELS = {
    "dwd_temp_air_2m": "Lufttemperatur (°C)",
    "dwd_precip_mm": "Niederschlag (mm)",
    "dwd_visibility_m": "Sichtweite (m)",
    "dwd_wind_speed_ms": "Windgeschwindigkeit (m/s)",
    "dwd_station_dist_km": "DWD-Stationsabstand (km)",
    "_precip_bucket": "Niederschlagsmenge",
}

FEATURE_LABELS = {
    "UKATGEORIE": "Unfallkategorie (Ziel)",
    "UART": "Unfallart",
    "UTYP1": "Unfalltyp",
    "ULICHTVERH": "Lichtverhältnisse",
    "STRZUSTAND": "Straßenzustand",
    "UWOCHENTAG": "Wochentag",
    "UMONAT": "Monat",
    "USTUNDE": "Stunde",
    "UJAHR": "Jahr",
    "ULAND": "Bundesland",
    "UREGBEZ": "Regierungsbezirk",
    "UKREIS": "Kreis",
    "UGEMEINDE": "Gemeinde",
    "LAT": "Breitengrad",
    "LON": "Längengrad",
    "IstRad": "Fahrrad",
    "IstPKW": "PKW",
    "IstFuss": "Fußgänger",
    "IstKrad": "Kraftrad",
    "IstGkfz": "Güterkraftfahrzeug",
    "IstSonstig": "Sonstiges Fahrzeug",
    **DWD_COL_LABELS,
}

COL_CODE_LABELS = {
    "ULICHTVERH": ULICHTVERH_LABELS,
    "STRZUSTAND": STRZUSTAND_LABELS,
    "UART": UART_LABELS,
    "UTYP1": UTYP1_LABELS,
}
```

Also add or verify:

```python
UKATGEORIE_LABELS = {
    1: "Getötet",
    2: "Schwer verletzt",
    3: "Leicht verletzt",
}

UKATGEORIE_ORDER = [
    "Getötet",
    "Schwer verletzt",
    "Leicht verletzt",
]
```

Use these labels consistently for:

* plot titles
* axis labels
* legends
* heatmap labels
* table labels
* markdown interpretation text where helpful
* saved figure filenames only where appropriate

Do not show raw codes like `ULICHTVERH = 2` in final visual outputs unless the code itself is being explained.

---

# Task 3 — Add Label Helper Functions

Create small helper functions if not already present.

Use them consistently instead of manually relabeling every plot.

Recommended helpers:

```python
def feature_label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


def apply_code_labels(series: pd.Series, column: str) -> pd.Series:
    mapping = COL_CODE_LABELS.get(column)
    if mapping is None:
        return series
    return series.map(mapping).fillna(series.astype(str))


def severity_label(series: pd.Series) -> pd.Series:
    return series.map(UKATGEORIE_LABELS).fillna(series.astype(str))


def ordered_severity_dtype(series: pd.Series) -> pd.Series:
    return pd.Categorical(
        severity_label(series),
        categories=UKATGEORIE_ORDER,
        ordered=True,
    )
```

If a better helper system already exists, improve that instead of duplicating it.

---

# Task 4 — Apply Consistent Plot Styling

Use a consistent visual theme unless the repository already has one.

Recommended default:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="colorblind",
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})
```

For long category labels, prefer horizontal layouts:

```python
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
sns.barplot(data=plot_df, y="label", x="value", ax=ax)
```

For heatmaps:

```python
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    square=False,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
```

Make sure exported figures use the same visual quality as notebook-rendered figures.

---

# Task 5 — Review Specific U-Phase Visualizations

Check and fix all relevant plots, including:

* target class distribution
* target class distribution by year
* severity by hour
* severity by month
* severity by weekday
* severity by Bundesland or region
* Unfallart (`UART`) plots
* Unfalltyp (`UTYP1`) plots
* Lichtverhältnisse (`ULICHTVERH`) plots
* Straßenzustand (`STRZUSTAND`) plots
* transport-mode plots:

  * `IstRad`
  * `IstPKW`
  * `IstFuss`
  * `IstKrad`
  * `IstGkfz`
  * `IstSonstig`
* DWD weather coverage plots
* DWD weather distribution plots
* DWD weather bivariate plots
* precipitation bucket plots
* DWD station-distance plots
* weather missingness plots
* missingness matrices
* missingness heatmaps
* numeric histograms
* KDE plots
* boxplots
* violin plots
* categorical count plots
* correlation heatmaps
* Cramér’s V heatmaps
* leakage-probe visualizations
* temporal leakage probe charts
* preprocessing-decision supporting visuals

For every plot, verify:

* labels are human-readable
* title is concise
* axes are understandable
* legend order is meaningful
* target severity order is consistent
* colors are readable
* layout is not cramped
* interpretation text still matches the visual
* saved plot file is updated if plots are exported

---

# Task 6 — Check README Consistency

Review `README.md` and fix inconsistencies caused by the new U-Phase and DWD enrichment.

In particular, check this current QUA³CK table entry:

`U — Understanding | DIG-Description, EDA, Geo-Visualisierung, Feature Engineering`

This likely overstates the U-Phase because advanced feature engineering belongs to A³.

Update the README so the U-Phase description is strict and accurate.

Suggested wording:

`Schema-Audit, EDA, Datenqualität, Wetter-Enrichment, Leakage-Prüfung, Preprocessing-Plan`

Also check:

* dataset description
* DWD weather enrichment mention
* setup instructions
* data file descriptions
* notebook list
* phase descriptions
* metric descriptions
* chronological split statement
* license and source attribution

Add a concise DWD data-source note if it is missing.

Make sure the README does not imply that U performs model training or feature engineering beyond preprocessing planning.

---

# Task 7 — Check Glossary Consistency

Review `GLOSSARY.md`.

Fix:

* spelling inconsistencies
* column-name inconsistencies
* raw codes that should have labels
* terminology that conflicts with Q or U
* duplicated definitions
* unclear DWD explanations
* outdated statements after the U-Phase changes
* phase-boundary issues

Pay special attention to these terms:

* `IstSonstig` vs `IstSonstige`
* `UKATGEORIE` source spelling
* `UART`
* `UTYP1`
* `ULICHTVERH`
* `STRZUSTAND`
* `UWOCHENTAG`
* `OBJECTID`
* `PLST`
* `LAT` / `LON`
* `LINREFX` / `LINREFY`
* `dwd_temp_air_2m`
* `dwd_precip_mm`
* `dwd_visibility_m`
* `dwd_wind_speed_ms`
* `dwd_station_dist_km`
* DWD join granularity
* cKDTree
* chronological split
* conditional entropy reduction
* target leakage
* preprocessing leakage
* QUA³CK phase definitions

Important: Keep the glossary plain-language and useful for readers.

Do not turn the glossary into a modeling chapter.

If terms like target encoding, SMOTE, SHAP, CatBoost, and TimeSeriesSplit are included, ensure they are clearly marked as later-phase concepts and do not imply they are implemented in U.

---

# Task 8 — Check Q-Phase / U-Phase Alignment

Review the Q-Phase and U-Phase together.

Ensure they agree on:

* research question
* target variable
* target labels
* unit of analysis
* years covered
* row count
* source data
* DWD enrichment
* weather join assumptions
* known limitations
* leakage boundary
* chronological split
* success metrics
* out-of-scope uses
* transition from Q to U
* transition from U to A³

If the U-Phase found evidence that corrects the Q-Phase, update the Q-Phase carefully and minimally.

Do not backfill the Q-Phase with EDA findings unless they belong there as feasibility or limitation notes.

---

# Task 9 — Check DWD Enrichment Documentation and Code Clarity

Review `src/unfallatlas/data/dwd.py`.

Do not rewrite the full DWD module unless necessary.

Check for:

* clear function names
* clear docstrings
* robust handling of DWD directory listings
* clear station-list parsing
* clear nearest-station lookup logic
* correct cKDTree usage
* clear left-join behavior
* clear sentinel-value handling
* clear output column names
* consistent names with U-Phase and Glossary
* no hard-coded fragile assumptions that contradict the notebook
* no silent data loss
* no accidental inner join where a left join is required
* no weather feature naming mismatch

If code comments are missing around complex parts, add concise comments.

Do not add heavy infrastructure.

---

# Task 10 — Verify DWD Terminology Everywhere

Ensure DWD weather enrichment is described consistently across:

* Q-Phase
* U-Phase
* README
* GLOSSARY
* `src/unfallatlas/data/dwd.py`

Use consistent language for:

* “DWD CDC hourly observations”
* “nearest station”
* “left join”
* “station distance”
* “temperature”
* “precipitation”
* “visibility”
* “wind speed”
* “day-level averaging noise”
* “no temporal leakage”
* “missing station coverage”
* “rural proxy”

Do not overclaim accuracy of weather attribution.

State clearly that the join is an approximation caused by the absence of day-of-month in `accidents.parquet`.

---

# Task 11 — Verify Section Numbering and Notebook Layout

Review the U-Phase notebook section numbering.

The previous implementation includes:

* §1–§9 main analysis
* §8.5 weather coverage
* §8.6 weather distributions
* §8.7 weather bivariate analysis
* §9.4 temporal leakage probe
* §10 preprocessing decision table
* §11 summary

Ensure:

* section numbers are sequential
* subsection numbers are not duplicated
* cross-references point to real sections
* markdown headings match references
* tables are not too wide
* callouts are visually consistent
* summary tables are readable
* code cells are not mixed with unrelated markdown
* long code helper sections are hidden, collapsed, or cleanly organized where possible

Do not over-format at the expense of notebook usability.

---

# Task 12 — Fix German/English Terminology Drift

The project mixes German source terminology with English technical explanations.

That is acceptable, but make it consistent.

Use German labels for source categories, for example:

* `Getötet`
* `Schwer verletzt`
* `Leicht verletzt`
* `Tageslicht`
* `Dämmerung`
* `Dunkelheit`
* `Trocken`
* `Nass/feucht/schlüpfrig`
* `Winterglatt`

Use English for technical explanations where the notebook already does so.

Avoid switching between multiple translations for the same concept.

For example, do not alternate between:

* “fatal”
* “killed”
* “Getötet”

unless the mapping is explicit.

Preferred style:

* In plots: German category labels
* In technical prose: English explanation with German label in parentheses where helpful

---

# Task 13 — Update Interpretation Text After Visual Cleanup

Because labels and ordering may change, review the interpretation text below each major plot.

Fix any text that no longer matches the plot.

For each major visualization, ensure the text answers:

* What does the plot show?
* What is the key observation?
* Why does it matter for U?
* Does it create a preprocessing requirement?
* Does it create a later A³ consideration?
* What is a common interpretation pitfall?

Keep the interpretations concise.

Do not add long speculative causal claims.

---

# Task 14 — Review Saved Figures and Tables

If the notebook exports figures or tables, verify that:

* output directories exist
* filenames are clear
* old duplicate figures are not left behind
* regenerated plots overwrite or clearly version old ones
* saved figures use readable DPI
* saved figures do not clip labels
* saved tables are consistent with notebook tables
* figure paths in markdown still work

Suggested U-Phase asset paths, only if consistent with the repo:

* `reports/figures/u_phase/`
* `reports/tables/u_phase/`

Do not reorganize the entire repository.

---

# Task 15 — Keep U-Phase Boundary Clean

During cleanup, remove or reword U-Phase content that sounds like A³, C, or K.

Examples:

* Replace “we will use this feature because it improves performance” with “this feature should be reviewed in A³ because it has a descriptive association with the target.”
* Replace “target encoding is used” with “target encoding is a candidate preprocessing strategy for A³ and must be fit inside the training fold only.”
* Replace “SHAP will show” with “interpretability is deferred to C.”
* Replace “deployment will use” with “Knowledge Transfer is deferred to K.”

The U-Phase should end with a clean U-to-A³ handoff, not with model conclusions.

---

# Task 16 — Final Validation Pass

Run or validate the notebook execution as far as practical.

Check for:

* broken imports
* missing helper functions
* stale variable names
* undefined label dictionaries
* plots depending on old raw-code columns
* broken paths
* inconsistent figure exports
* markdown references to missing plots
* outdated README statements
* outdated glossary definitions
* inconsistent `IstSonstig` / `IstSonstige` spelling
* DWD column-name mismatches

Do not make the notebook slower unless absolutely necessary.

If full execution is too expensive, run the changed cells or perform a targeted validation and document what was checked.

---

# Expected Final State

After this task, the repository should have:

* U-Phase visualizations with readable labels and clean layout
* consistent severity ordering
* official Unfallatlas labels in all relevant plots
* consistent DWD terminology across notebook, README, glossary, and code
* README accurately describing the U-Phase
* glossary aligned with actual repository implementation
* Q-Phase and U-Phase aligned
* no accidental U-to-A³ scope creep
* cleaner section numbering and cross-references
* updated saved figures where applicable

---

# Final Response Format

After completing the cleanup, respond with:

## Summary of Fixes

Briefly list the main improvements.

## Visualization Improvements

List which visualization categories were fixed.

## Documentation Updates

List README, glossary, Q-Phase, or U-Phase documentation changes.

## Code Updates

List any helper functions or `dwd.py` changes.

## Files Changed

List modified files.

## Boundary Notes

Mention anything intentionally deferred to A³, C, or K.

## Remaining Issues

List unresolved issues, skipped validations, or plots that still need manual review.


````
