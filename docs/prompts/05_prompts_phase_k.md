# AI Prompt Records: Phase K

Each record preserves the available prompt or planning context. Detailed
implementation instructions live only in the linked plan files.

> **Filename note:** this record follows the established `NN_prompts_phase_X.md`
> numbering. `docs/superpowers/plans/2026-07-27-phase-k-streamlit-app.md` and
> `docs/superpowers/specs/2026-07-27-phase-k-streamlit-app-design.md` both
> instructed `docs/prompts/04_prompts_phase_k.md`, but `04_prompts_phase_c.md`
> already existed by the time those Phase K documents were written, so this
> record uses the next free number (`05`) instead of colliding with the
> existing Phase C record.

## "Risk Explainer & Model Console" concept selection, design, and implementation plan

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)<br>
**Design spec:** [2026-07-27-phase-k-streamlit-app-design.md](../superpowers/specs/2026-07-27-phase-k-streamlit-app-design.md)<br>
**Implementation plan:** [2026-07-27-phase-k-streamlit-app.md](../superpowers/plans/2026-07-27-phase-k-streamlit-app.md)

### Recorded prompt: brainstorming and concept selection

The session opened with `superpowers:brainstorming` to scope the Phase K
("Knowledge Transfer") deliverable: an interactive Streamlit app, built
entirely from artifacts already committed by the C-phase, with no
retraining and no notebook execution required to run it. Five concepts were
proposed for the user to choose from, spanning different framings of the
same underlying committed artifact set (the binary KSI champion model, its
inference contract, the 10-candidate C-phase comparison table, and the
permutation-importance CSV). The user selected **Concept A: Risk Explainer
& Model Console** — a four-page native-multipage app (Overview, Risk
Predictor, Why This Prediction, Model Comparison) pairing a live prediction
form against the real committed champion model with the project's own
model-selection evidence, rather than a narrower single-purpose demo or a
geospatial-map-first concept (deferred to a later, unscheduled iteration).

Four follow-up clarifying decisions were made explicit before design work
began:

1. **English UI** — the app text is English throughout, matching the
   project's existing English-first documentation direction rather than
   German or bilingual labels.
2. **Native multipage navigation** — `st.navigation()` / `app/pages/`
   structure, not a single-page app with manual tabs or radio-button
   routing.
3. **Ceiling story included on Overview** — the three-class-ceiling vs.
   binary-reframe narrative (the project's strongest evidence-based
   finding, per the A³/C-phase Cramér's V and Pareto-front analysis) is
   surfaced on the first page a user sees, not buried in a later page or
   omitted as "already covered in the notebooks."
4. **Local-only deployment** — `uv run streamlit run app/streamlit_app.py`
   only; no Docker packaging, no cloud hosting, matching AGENTS.md's
   already-documented launch command.

### Recorded prompt: design-doc creation

Following concept selection, a written design spec was produced for the
user's review before any implementation plan or code was written. The
design spec (linked above) locked in:

- The **architecture boundary**: `app/streamlit_app.py` and the four
  `app/pages/*.py` files stay thin (page config, navigation wiring, and
  `st.*` widget calls only); all data loading, model inference, and
  DataFrame assembly lives in `src/unfallatlas/viz/streamlit_app.py`,
  which contains no Streamlit widget calls itself so it stays unit-testable
  without a Streamlit runtime — matching the repo's existing
  notebook-to-library boundary convention.
- The **complete data flow**: cached loaders for the inference contract,
  the champion joblib pipeline, the model card, the binary/3-class
  comparison CSVs, the candidate-metrics CSV, and the permutation-importance
  CSV, plus `build_input_row`/`predict_ksi` as pure, independently testable
  functions.
- A **verified required-columns reference table** (30 columns) mapping
  each inference-contract column to its widget type, sourced from a direct
  read of the committed `c_phase_inference_contract.json` rather than
  reconstructed from memory — including the decision to source `UKREIS`'s
  87 option values live via a `DuckDB` lookup against the already-committed
  `data/accidents.parquet` (a single-column columnar scan, not a full
  dataset load) instead of hardcoding a list liable to drift, while
  `dwd_station_id` and `h3_cell` (technical join keys with no meaningful
  user interpretation) stay non-user-facing with one fixed representative
  default value each.
- Explicit **non-goals**: no geospatial map, no live SHAP computation, no
  live ROC/PR curve recomputation from the 2.69M-row scores parquet, no
  retraining, no Docker/cloud deployment, and no changes to
  `src/unfallatlas/models/`, `features/`, or the notebooks themselves.
- The **SHAP-to-permutation-importance substitution** (detailed below) as
  a design-level constraint on the "Why This Prediction" page, not an
  implementation-time afterthought.

### Recorded prompt: implementation-plan creation

`superpowers:writing-plans` then produced the linked 11-task implementation
plan from the approved design spec:

1. Core artifact loaders (`load_inference_contract`, `load_model_card`,
   the comparison-CSV and permutation-importance loaders) in
   `src/unfallatlas/viz/streamlit_app.py`.
2. Categorical widget options, `get_column_spec`, and shared UI constants
   (severity colors, limitations text, default widget values).
3. Input-row assembly and threshold-aware prediction
   (`build_input_row`, `load_champion_model`, `predict_ksi`), with a real
   end-to-end integration test against the committed 407 MB joblib —
   this is where the `IstGkfz` bug (below) was caught.
4. `app/streamlit_app.py` entry point and native multipage navigation
   wiring.
5. Overview page (champion headline metrics + the two-Pareto-front ceiling
   narrative).
6. Risk Predictor page (the full 30-field input form, wired to
   `build_input_row`/`predict_ksi`).
7. Why This Prediction page (global permutation importance + the user's
   own input values for the top-15 globally important features, with the
   SHAP-disclosure notice).
8. Model Comparison page (10-candidate table, Pareto front, champion
   confusion matrix, 4-model finalist comparison quoting the C-phase's own
   deployment-champion rationale).
9. End-to-end fresh-environment verification: confirmed every artifact the
   app reads is actually committed (not just present locally,
   `git ls-files`), ran the full test suite, linted the whole app, and
   manually walked all four pages in order, submitting at least three
   predictions covering different `IstGkfz`, `osm_dominant_road_class`,
   `UWOCHENTAG`, and non-default `UKREIS` combinations with no exception
   raised — confirming `git lfs pull` + `uv sync` +
   `uv run streamlit run app/streamlit_app.py` is sufficient with no
   notebook execution.
10. Fixed stale `AGENTS.md` auto-managed sections that still described
    `notebooks/04_C_Phase.ipynb` as an empty TODO stub and the Streamlit
    entry points as 0-byte stubs.
11. This documentation task: update `docs/AI TOOL DISCLOSURE.md` and add
    this prompt record.

Execution followed `superpowers:subagent-driven-development`: a fresh
implementer subagent per task, followed by an independent reviewer
subagent per task, with fix-and-re-review loops where issues were found.

### A real bug caught during implementation (Task 3)

The C-phase inference contract's `required_columns` metadata for
`IstGkfz` declares `dtype: "object"` with string categories
`["False", "True"]`, unlike its five `Ist*` sibling columns
(`IstRad`, `IstPKW`, `IstFuss`, `IstKrad`, `IstSonstig`), which all declare
`dtype: "bool"`. Taken at face value, this metadata implies `IstGkfz` needs
a string cast (`str(bool_value)`) that its siblings do not. This was
verified empirically against the actual committed
`data/processed/a3_binary_best_model.joblib` rather than trusted as
written: passing the string `"False"` raises
`ValueError: could not convert string to float: 'False'` inside the fitted
pipeline, because the `ColumnTransformer` routes all six `Ist*` columns
through the same passthrough group into the `RandomForestClassifier`,
which casts the whole array to `float32` — a real Python `bool` converts
cleanly (`True`/`False` → `1.0`/`0.0`), but the string does not. `IstGkfz`
is therefore built identically to its five siblings: the raw `bool`, no
cast. The contract's recorded `deployment_model_sha256` also does not
match the actual committed joblib's SHA256, suggesting this metadata
inconsistency is a pre-existing artifact of the C-phase pipeline (not
introduced by this plan) — likely worth a dedicated C-phase touch-up in a
future iteration to regenerate the contract's checksum and dtype metadata
against the model actually deployed.

### No SHAP: permutation importance instead

No SHAP values were ever computed for this project — this was a decision
made in the C-phase (`src/unfallatlas/viz/shap_plots.py` is an intentional
stub) and is reaffirmed rather than revisited here. The C-phase's global
permutation-importance analysis (`c_phase_permutation_importance.csv`) is
the only feature-importance evidence that exists, and it is model-level
and global, not per-instance and causal. The Streamlit app's "Why This
Prediction" page uses this permutation importance and carries a prominent,
unmissable notice directly in the app UI: this is global, model-level
permutation importance from the C-phase analysis, not a per-instance SHAP
explanation — none was computed for this project. The user's own input
values are shown for context alongside the top-15 globally important
features, explicitly captioned as such rather than framed as an
explanation of that specific prediction.

## Post-launch enhancements: readable labels, interactive maps, and deployment prep

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)<br>
**Implementation plan:** [2026-07-27-phase-k-streamlit-enhancements.md](../superpowers/plans/2026-07-27-phase-k-streamlit-enhancements.md)

### Recorded prompt

After the initial four-page app above shipped, the user requested a set of
follow-up improvements in the same session, given directly rather than via
`/brainstorming` (the concepts were concrete UI fixes, not an open design
choice):

> for UREGBEZ and UKREIS not the numeric values but the actual meaning like
> eg for UWOCHENTAG Monday because for these two only numbers are
> selectable. and also it would be waaaay better to not select a LON and
> LAT but rather have a map and be able to select a point on the map thatll
> be converted to the representative Coordinates. [...] also for the Why
> this prediction page the same thing instead of the actual labels use the
> representative meaning eg for dwd_wind_speed_ms Wind Speed (m/s)

> maybe for Overview an interactive OSM Map with severyti of accidents

> and we need to deploy it on streamlit itself and referencer it in the
> readme also maybe a reference from the streamlit app to the github pages
> notebook htmls and from the notebook htmls to the streamlit app

> use /writing-plans and sdd

`superpowers:writing-plans` produced the linked 8-task implementation plan
directly (no separate brainstorming pass, since the user's asks were already
concrete), followed immediately by `superpowers:subagent-driven-development`
execution: fresh implementer + independent reviewer per task, then a
whole-branch review on the most capable available model, then a consolidated
fix pass for the review's findings, then a second reviewer confirming the
fixes.

### A real, honest limitation surfaced during design (not silently worked around)

The user's request for `UREGBEZ`/`UKREIS` to show "the actual meaning like
... Monday" cannot be fully satisfied: unlike `UWOCHENTAG`, which decodes via
a small fixed lookup, `UREGBEZ`/`UKREIS` require the Bundesland key (`ULAND`)
to resolve to an official Gemeindeschlüssel/region name, and `ULAND` is not
part of the model's feature set or the inference contract (`UREGBEZ` has
only 10 categories, `UKREIS` is documented as "high-cardinality, 87 unique
values, no fixed category list" in `data/processed/c_phase_inference_contract.json`).
`docs/dataset/DSB_Unfallatlas.md` confirms the official code requires all
three fields together. There is no offline name-lookup table in this repo
and no permitted new dependency to build one within this session's "no new
dependencies, no network calls" constraint. Rather than inventing plausible-
looking but fake region names, the fix is transparency: the widget labels
and a caption now explicitly say these are internal dataset codes, not
official region names, and explain why.

### What shipped

- Honest `UREGBEZ`/`UKREIS` code labeling (with the limitation above stated
  in-app), an interactive `folium`/`streamlit-folium` map picker replacing
  the raw LON/LAT number inputs on Risk Predictor (with bounding-box
  validation against the contract's real lat/lon ranges), an interactive
  KSI-vs-slight severity map on Overview (DuckDB-aggregated into a ~0.1°
  grid rather than rendering all 2.09M rows), human-readable feature names
  and decoded category values on Why This Prediction, and a bilingual
  (German + English) explanation of why Random Forest remains the
  deployment champion even though a later robustness/latency comparison
  ranks XGBoost ahead on the combined criteria (model selection was already
  locked in on validation data; the held-out test set cannot be reused to
  re-pick a champion without invalidating its own reported metrics).
- `requirements.txt` and documented manual deployment steps for Streamlit
  Community Cloud (which does not understand this repo's `uv.lock` or its
  PEP 621/hatchling `pyproject.toml`, and would otherwise misparse it as
  Poetry format), plus reciprocal links between the Streamlit app and the
  GitHub Pages notebook-presentation site.
- `folium`/`streamlit-folium` were already base dependencies before this
  work (not the `geo` extra) — no new dependency was introduced.

### Findings from the whole-branch review, and what was fixed

The final review (dispatched on the most capable available model) found no
Critical issues and three Important ones, all fixed in one consolidated
follow-up commit: (1) the deployment doc only told the user to update the
Streamlit-Cloud placeholder URL in one of four places it actually appears,
now all four are listed; (2) the README's new "Live Deployment" section
initially overclaimed that the GitHub Pages site already links back to the
Streamlit app — it does only in the template, since the actually-committed,
already-exported HTML was deliberately not regenerated this session (an
explicit user instruction), so the README wording was corrected to state
this accurately instead of overclaiming; (3) the Overview page's severity
map (~4,857 markers) was rebuilt uncached on every rerun, measured at ~3.4s
and ~4.9 MB of HTML — now built once behind `@st.cache_resource`. Five
further Minor findings (a potentially misleading red/teal color reading on
the severity map with no legend, a map-click warning that stays visible
after the click that caused it, a stale page subheading left over from the
LON/LAT-to-map-picker change, an unpinned `requirements.txt` versus the
project's locked dependency versions, and an unverified claim about
Streamlit Community Cloud's Git LFS handling) were logged rather than fixed,
as explicitly lower-priority follow-ups for the project owner.
