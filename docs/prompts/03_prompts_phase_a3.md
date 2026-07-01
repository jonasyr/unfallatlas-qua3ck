# AI Prompt for Phase A³

## Build the A³-Phase from the U-Phase Contract

**Claude Code (Sonnet 5), effort: medium (Anthropic, 2026):**

### Initial prompt

The A³-phase plan was produced with the `superpowers:writing-plans` skill
(Claude Code / superpowers plugin), starting from the following two
verbatim user messages:

> Now please do me a favour now the A Phase writing creation and
> implementation is the next step and i want you to first read all the
> @docs/ and especiall the relevant ones additionally the already existing
> Q and U Phase as it builds upon these 2 then craft a veeery detailed plan
> using /writing-plans where you make AN EXACT SCOPE what the exact things
> are that HAVE to be done in A Phase and what explicitly DONT HAS to go in
> the A PHase as i had issues previously about exactly this in Q and U
> phase with you so make this extra good. Then utilize all the information
> to make a extremly strong A Phase tailored perfectly to my project and
> its context. Then display the plan to me so i can approve it

> IMPORTANT write the finished approved plan as first task inside
> @docs/prompts/ folder as a new prompt file this one for A phase and
> update @"docs/AI TOOL DISCLOSURE.md"

The resulting task-by-task plan lives at
`docs/superpowers/plans/2026-07-01-a3-phase-modelling.md`; this prompt file
is that plan's Task 1 deliverable.

```markdown
You are a senior Machine Learning Engineer, Data Scientist, and repository-quality
documentation architect.

Your task is to implement the **A³-Phase — Algorithm / Adapt / Adjust** of the
QUA³CK process model for this repository.

The **Q-Phase** and **U-Phase already exist** and are the authoritative contracts.
Your job is not to redesign them. Your job is to implement the U-Phase §10
preprocessing decision table, train and compare the models the U-Phase
handed off to A³, tune the winner, and evaluate it once against the
Q-Phase acceptance criteria.

---

# PRIMARY OBJECTIVE

Implement `notebooks/03_A3_Phase.ipynb` (+ paired `.py` mirror) and the
supporting `src/unfallatlas/{features,models}/` library code so that the
project has a working, tested, reproducible modelling pipeline answering:

> "Given the U-Phase's preprocessing contract, which model and imbalance
> strategy best predicts UKATGEORIE, and does the tuned winner clear the
> Q-Phase bar of macro-F1 >= 0.55 and recall(class 1) >= 0.50 on the 2024
> held-out test year?"

---

# REQUIRED REPOSITORY CONTEXT

You MUST read before writing anything:

* `notebooks/01_Q_Phase.ipynb` / `.py` — target, metrics, split, constraints
* `notebooks/02_U_Phase.ipynb` / `.py`, especially §9 (leakage audit) and
  §10 (preprocessing decisions) and §11 (top-4 risks for A³)
* `AGENTS.md` — architecture, conventions, notebook policy
* `docs/GLOSSARY.md` — existing terminology
* `pyproject.toml` — already-declared dependencies (no new ones needed)
* the empty stub files at `src/unfallatlas/features/temporal.py` and
  `src/unfallatlas/models/{baseline,boosting,ordinal,evaluate}.py`

Do not contradict the Q-Phase or U-Phase. If you find a mismatch between
what U-Phase §10 specifies and what the data actually contains, document it
as an A³-Phase finding — do not silently change the U-Phase.

---

# STRICT A³-PHASE SCOPE

Include:

* the U-Phase §10 preprocessing pipeline, implemented exactly as decided
  (cyclic encoding, one-hot, target-mean encoding with smoothing, log/log1p
  + scaling, drops) — no new engineered features beyond that table
* baseline models: random guess, majority class, logistic regression
* tree ensembles: Random Forest, XGBoost, LightGBM, CatBoost, each with a
  default and a class-weighted configuration
* selection of a single champion model by validation macro-F1
* an imbalance-strategy comparison (SMOTE, ADASYN, threshold moving,
  ordinal/Frank-Hall classification) run only on the champion
* bounded Optuna tuning (<= 40 trials) of only the single winning
  (model, strategy) combination
* exactly one evaluation on the 2024 test year
* a saved model artefact + model card for the C-Phase to consume
* a model-comparison table as the central portfolio output
* pytest coverage for every new library function

Do NOT include:

* SHAP or any explainability analysis (Phase C)
* literature-comparison narrative or limitations discussion (Phase C)
* Streamlit app or any deployment work (Phase K)
* OSM/Overpass, H3 hex-bins, or any spatial enrichment beyond what the
  U-Phase cache already contains
* ensembling/stacking of multiple models
* full hyperparameter tuning across every model x every strategy
  combination — only the single winner is tuned
* new dependencies in `pyproject.toml`
* edits to `04_C_Phase.ipynb`, `app/streamlit_app.py`, or
  `reports/final_report.md`
* rebuilding the DWD weather cache

Those belong to later QUA³CK phases or are out of scope entirely.

---

# IMPORTANT BOUNDARY RULES

The A³-Phase may say:

* "The champion model is X; the tuned winner is X + strategy Y."
* "macro-F1 on test-2024 is Z; this [passes/fails] the >= 0.55 threshold."
* "This preprocessing decision follows U-Phase §10 verbatim."

The A³-Phase must NOT say:

* "SHAP shows feature Z is most important." (Phase C)
* "This model is ready to deploy in Streamlit." (Phase K)
* "We should add H3 features here." (out of scope — note it as a future
  idea in the handoff, do not implement it)

When uncertain, defer the topic to C or K instead of expanding A³.

---

# TASKS

Follow `docs/superpowers/plans/2026-07-01-a3-phase-modelling.md` task-by-task.
Each task specifies exact files, function signatures, and test code.

---

# REQUIRED OUTPUTS

1. `src/unfallatlas/features/temporal.py` + `preprocessing.py`, tested
2. `src/unfallatlas/models/{evaluate,baseline,boosting,ordinal,imbalance}.py`, tested
3. `notebooks/03_A3_Phase.ipynb` + paired `.py`, executing end-to-end
4. `data/processed/a3_best_model.joblib` + `a3_model_card.json`
5. Updated `AGENTS.md` notebook status table and architecture section
6. Updated `docs/GLOSSARY.md` with new A³ terms
7. This prompt file and the corresponding `AI TOOL DISCLOSURE.md` row

---

# QUALITY REQUIREMENTS

The A³-Phase must be reproducible, tested, strict about phase boundaries,
and honest about its single test-set evaluation (no re-running the test
set to chase a better number). Every reported metric must come from an
executed cell, not a fabricated value.

---

# FINAL RESPONSE FORMAT

## Summary of Changes

## Key A³-Phase Findings (champion, winning strategy, final test metrics, PASS/FAIL)
```
```
