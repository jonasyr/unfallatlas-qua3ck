# GLOSSARY.md Update Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring GLOSSARY.md in sync with DSB_Unfallatlas.md and the Q/U-phase notebooks — fixing inaccuracies, expanding thin entries with DSB code tables, and adding missing ML and domain terms.

**Architecture:** Single-file edit to `docs/GLOSSARY.md`. Three groups of changes: (1) corrections to existing entries, (2) new entries for terms present in the DSB but absent from the glossary, (3) new entries for ML/modelling terms introduced in the notebooks.

**Tech Stack:** Markdown only — no code changes.

---

## Source of truth

| Source | What it contributes |
|:---|:---|
| `docs/DSB_Unfallatlas.md` | Authoritative column names, code tables, and field definitions |
| `notebooks/01_Q_Phase.ipynb` | ML concepts: SHAP, Vision Zero, BASt, Dunkelziffer, model families, Pipeline |
| `notebooks/02_U_Phase.ipynb` | Preprocessing concepts: log1p, StandardScaler, SMOTE, majority-class collapse, OBJECTID |

---

## Task 1: Correct existing entries

**Files:**
- Modify: `docs/GLOSSARY.md`

- [ ] **Step 1: Fix `UKATGEORIE` entry — add source-spelling note**

  Replace the existing `UKATGEORIE` heading line:

  ```
  **UKATGEORIE (Unfallkategorie)**
  ```

  with:

  ```
  **UKATGEORIE (Unfallkategorie)**
  Source-data misspelling of *UKATEGORIE*; the project adopts the source
  spelling without correction.
  ```

- [ ] **Step 2: Fix `ULICHTVERH` entry — remove inaccurate "with/without street lighting"**

  The DSB defines exactly three codes with no sub-distinction for lighting:

  Replace:
  ```
  Lighting conditions at the time of the accident: daylight, dusk/dawn, or darkness with/without street lighting.
  ```

  with:
  ```
  Lighting conditions at the time of the accident:
  - `0` — Tageslicht (daylight)
  - `1` — Dämmerung (dusk/dawn)
  - `2` — Dunkelheit (darkness)
  ```

- [ ] **Step 3: Fix `IstSonstig` → `IstSonstige` and add 2016/2017 note**

  The DSB spells the column `IstSonstige`. The current entry also omits the dataset note about 2016/2017.

  Replace in the `IstRad / IstPKW / IstFuss / IstKrad / IstGkfz / IstSonstig` entry:

  ```
  **IstRad / IstPKW / IstFuss / IstKrad / IstGkfz / IstSonstig**
  Binary flags indicating which transport modes were involved: bicycle, car, pedestrian, motorcycle, heavy goods vehicle, and other.
  ```

  with:

  ```
  **IstRad / IstPKW / IstFuss / IstKrad / IstGkfz / IstSonstige**
  Binary flags indicating which transport modes were involved: bicycle (`IstRad`), car (`IstPKW`), pedestrian (`IstFuss`), motorcycle or moped (`IstKrad`), heavy goods vehicle > 3.5 t (`IstGkfz`), and other (`IstSonstige`).
  Note: `IstGkfz` is only available from 2018 onward; in 2016 and 2017 its accidents are subsumed under `IstSonstige`.
  ```

- [ ] **Step 4: Update `LAT / LON` entry with actual DSB column names**

  The DSB column names are `XGCSWGS84` (longitude) and `YGCSWGS84` (latitude). The project renames these to `LON` / `LAT` during parquet consolidation.

  Replace:
  ```
  **LAT / LON**
  WGS-84 geographic coordinates of the accident location. Used for spatial analysis and for the nearest-station lookup in the DWD enrichment.
  ```

  with:

  ```
  **LAT / LON** *(source columns: `YGCSWGS84` / `XGCSWGS84`)*
  WGS-84 geographic coordinates (decimal degrees) of the accident location, renamed from the DSB source columns during parquet consolidation. Used for spatial analysis and for the nearest-station lookup in the DWD enrichment.
  ```

- [ ] **Step 5: Expand `UART` entry with DSB code table**

  Replace:
  ```
  **UART (Unfallart)**
  The type of accident (e.g., collision with oncoming traffic, rear-end collision, pedestrian crossing). Recorded after the event; subject to leakage audit in §9.1 of the U phase.
  ```

  with:

  ```
  **UART (Unfallart)**
  The type of accident, recorded in the police report after the event. Subject to leakage audit in §9.1 of the U phase.
  - `0` — Other accident type (`Unfall anderer Art`; ~15 % of records; 58 % bicycle involvement)
  - `1` — Collision with a stationary/stopping/parked vehicle
  - `2` — Rear-end collision with a preceding or waiting vehicle
  - `3` — Sideswipe collision (same direction)
  - `4` — Head-on collision
  - `5` — Collision with a turning or crossing vehicle (most frequent)
  - `6` — Collision between vehicle and pedestrian
  - `7` — Impact with an obstacle on the roadway
  - `8` — Run-off road to the right
  - `9` — Run-off road to the left (codes 8/9 carry the highest fatality rate)
  ```

- [ ] **Step 6: Expand `UTYP1` entry with DSB code table**

  Replace:
  ```
  **UTYP1 (Unfalltyp)**
  The structural accident type (e.g., driving accident, turning accident, intersection accident). Also recorded post-event and subject to the same leakage probe as `UART`.
  ```

  with:

  ```
  **UTYP1 (Unfalltyp)**
  The structural accident type, recorded post-event. Subject to the same leakage probe as `UART` (§9.1).
  - `1` — Fahrunfall (driving accident)
  - `2` — Abbiegeunfall (turning accident)
  - `3` — Einbiegen/Kreuzen-Unfall (merging/crossing accident)
  - `4` — Überschreiten-Unfall (pedestrian crossing accident)
  - `5` — Unfall durch ruhenden Verkehr (accident involving parked/stationary traffic)
  - `6` — Unfall im Längsverkehr (longitudinal-traffic accident)
  - `7` — Sonstiger Unfall (other)
  ```

- [ ] **Step 7: Commit corrections**

  ```bash
  git add docs/GLOSSARY.md
  git commit -m "docs(glossary): fix inaccurate/incomplete existing entries"
  ```

---

## Task 2: Add missing DSB-sourced entries

**Files:**
- Modify: `docs/GLOSSARY.md` — "Dataset and Domain Terms" section

- [ ] **Step 1: Add `ULAND` entry**

  Insert after the `UJAHR / UMONAT / USTUNDE / UWOCHENTAG` block:

  ```markdown
  **ULAND (Bundesland)**
  Two-digit code identifying the federal state in which the accident occurred.
  Together with `UREGBEZ`, `UKREIS`, and `UGEMEINDE`, it forms the official
  municipality key (*Amtlicher Gemeindeschlüssel*). Coverage start year varies
  by state (e.g. Nordrhein-Westfalen from 2019, Mecklenburg-Vorpommern from 2020).
  ```

- [ ] **Step 2: Add `UWOCHENTAG` code table**

  The existing entry mentions encoding but omits the fact that `1 = Sonntag`
  (Sunday-first encoding, not Monday-first). Append to the existing
  `UJAHR / UMONAT / USTUNDE / UWOCHENTAG` entry:

  ```
  `UWOCHENTAG` uses a Sunday-first encoding: 1 = Sonntag, 2 = Montag, …, 7 = Samstag.
  ```

- [ ] **Step 3: Add `OBJECTID` entry**

  Insert in "Dataset and Domain Terms" section (e.g. after `ULAND`):

  ```markdown
  **OBJECTID**
  Unique integer identifier per accident row. No two rows share an OBJECTID;
  used in the §9.3 no-overlap check to confirm the chronological split
  contains no duplicate accidents across train / val / test sets.
  Dropped before model fitting.
  ```

- [ ] **Step 4: Add `LINREFX / LINREFY` entry**

  Insert after the `LAT / LON` entry:

  ```markdown
  **LINREFX / LINREFY**
  UTM coordinates (ETRS89, Zone 32N) of the accident location projected onto
  the nearest road segment. Distinct from `XGCSWGS84 / YGCSWGS84` (WGS-84
  decimal degrees). Not used in this project; the join and spatial analysis
  use the WGS-84 columns.
  ```

- [ ] **Step 5: Add `PLST` entry**

  Insert after `LINREFX / LINREFY`:

  ```markdown
  **PLST (Plausibilisierungsstufe)**
  Geocoding quality indicator. `1` = accident location geocoded by the
  standard procedure; `2` = geocoded by the extended procedure for accidents
  involving bicycles. Only accidents that pass plausibility checks are
  included in the published dataset (~92 % of all recorded events); the
  remaining ~8 % are excluded, introducing a documented selection bias.
  ```

- [ ] **Step 6: Commit new DSB entries**

  ```bash
  git add docs/GLOSSARY.md
  git commit -m "docs(glossary): add ULAND, OBJECTID, LINREFX/Y, PLST; expand UWOCHENTAG"
  ```

---

## Task 3: Add missing ML/modelling entries

**Files:**
- Modify: `docs/GLOSSARY.md` — "Machine Learning Concepts" section

- [ ] **Step 1: Add `SHAP` entry**

  Insert after `Macro-F1`:

  ```markdown
  **SHAP (SHapley Additive exPlanations)**
  A game-theoretic framework that assigns each feature a contribution value
  for a specific prediction. SHAP values satisfy additivity: the sum of all
  feature contributions equals the model's output minus the baseline. Used
  in Phase C for both global (feature importance across the dataset) and
  local (per-accident) explanations. Required by the interpretability hard
  constraint (§9 of the Q phase).
  ```

- [ ] **Step 2: Add gradient-boosting model entry**

  Insert after `SHAP`:

  ```markdown
  **Gradient Boosting (XGBoost / LightGBM / CatBoost)**
  An ensemble method that fits successive decision trees, each correcting
  the residual errors of the previous ones, optimising a differentiable loss
  function via gradient descent. The three implementations differ in speed
  and native categorical handling: XGBoost is the methodological baseline
  from the literature anchor; LightGBM uses leaf-wise growth (faster on large
  datasets); CatBoost handles ordinal categoricals natively without
  one-hot encoding. All three are candidates in A³.
  ```

- [ ] **Step 3: Add `SMOTE` entry**

  Expand the existing `Class Imbalance` entry or add a dedicated entry after it:

  ```markdown
  **SMOTE (Synthetic Minority Over-sampling Technique)**
  An oversampling strategy that synthesises new minority-class samples by
  interpolating between existing nearest neighbours in feature space, rather
  than simply duplicating rows. Used as one of the imbalance-mitigation
  candidates in A³ alongside class weights and threshold moving.
  Must be applied inside the training fold only (never to validation or test
  data) to avoid data leakage.
  ```

- [ ] **Step 4: Add `StandardScaler` entry**

  Insert after `SMOTE`:

  ```markdown
  **StandardScaler**
  A sklearn preprocessing step that transforms a numeric feature to zero mean
  and unit variance: `(x − μ) / σ`. Parameters (μ, σ) are fit on training
  data only and applied to val/test to prevent leakage. Used for continuous
  features fed to distance-based or linear baselines; not required for
  tree-based models. Implemented inside the sklearn `Pipeline`.
  ```

- [ ] **Step 5: Add `log1p transform` entry**

  Insert after `StandardScaler`:

  ```markdown
  **log1p transform**
  `log(1 + x)`: a variance-stabilising transform for right-skewed non-negative
  features. The `+1` shift makes the transform defined at zero (unlike `log(x)`).
  Applied to `dwd_precip_mm` and `dwd_visibility_m` to reduce the influence of
  extreme outliers before scaling. `log1p` is the inverse of `expm1`, so the
  transform is fully reversible.
  ```

- [ ] **Step 6: Add `sklearn Pipeline` entry**

  Insert after `log1p transform`:

  ```markdown
  **sklearn Pipeline**
  A `sklearn.pipeline.Pipeline` chains preprocessing steps and a model
  estimator into a single object. Crucially, all `fit`-based preprocessing
  steps (StandardScaler, target encoding, imputation statistics) are fitted
  only on the training fold and then applied to val/test — the same guarantee
  that prevents train-test contamination. A³ wraps all preprocessing in a
  Pipeline so the U-phase preprocessing decisions are never accidentally
  applied globally.
  ```

- [ ] **Step 7: Add `Dunkelziffer` entry**

  Insert in "Dataset and Domain Terms" after `Class Imbalance`:

  ```markdown
  **Dunkelziffer**
  German term for the "dark figure" — the unknown quantity of accidents that
  were never reported to the police. Minor accidents are systematically
  under-reported, so the class distribution in the dataset reflects police
  reporting behaviour as much as true event frequency. This structural bias
  cannot be corrected from within the dataset.
  ```

- [ ] **Step 8: Add `Vision Zero` and `BASt` to Process Model section**

  Insert at the end of the "Process Model" section:

  ```markdown
  **Vision Zero**
  A road-safety policy goal — originally Swedish, adopted as EU policy — that
  targets zero road fatalities and serious injuries by 2050. Referenced in the
  Q phase as the policy context in which this project's outputs (corridor-level
  risk scores) would be consumed by public-sector analysts.

  **BASt (Bundesanstalt für Straßenwesen)**
  Germany's Federal Highway Research Institute. Publishes the annual
  *Unfallentwicklung auf deutschen Straßen* report, which is used in this
  project to sanity-check model output patterns (e.g. the 2020 COVID-19
  accident-count dip).
  ```

- [ ] **Step 9: Add `Majority-class collapse` clarification to Class Imbalance entry**

  Append to the existing `Class Imbalance` entry:

  ```
  A naïve model trained without imbalance handling will exhibit
  *majority-class collapse*: it learns to always predict class 3 (minor
  injury), achieving ~81 % accuracy but macro-F1 ≈ 0.30 and recall 0.00
  on classes 1 and 2.
  ```

- [ ] **Step 10: Commit new ML entries**

  ```bash
  git add docs/GLOSSARY.md
  git commit -m "docs(glossary): add SHAP, gradient boosting, SMOTE, Pipeline, Dunkelziffer, Vision Zero, BASt"
  ```

---

## Self-Review Checklist

After all edits, re-read GLOSSARY.md top to bottom and verify:

- [ ] Every DSB column in `docs/DSB_Unfallatlas.md` has at least a mention in the glossary.
- [ ] `IstSonstige` (not `IstSonstig`) everywhere.
- [ ] `ULICHTVERH` no longer claims "with/without street lighting".
- [ ] `UART` code `0` is present (it's the anomalous "other" bucket).
- [ ] `LAT/LON` entry references `XGCSWGS84 / YGCSWGS84`.
- [ ] `UKATGEORIE` misspelling note is present.
- [ ] `SHAP`, `SMOTE`, `StandardScaler`, `log1p`, `Pipeline` are all present.
- [ ] `Dunkelziffer`, `Vision Zero`, `BASt` are present.
- [ ] No entry contradicts information in `DSB_Unfallatlas.md`.
