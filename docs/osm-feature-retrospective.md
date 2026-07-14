# OSM Road-Context Features: Retrospective (U-phase build → A³ retrain)

**Purpose of this note:** a standalone record of what was built, what data flowed through it, and what measurable effect it had — for future brainstorming on whether/how to push macro-F1 further. Not part of the formal QUA³CK notebook chain; a working reference.

---

## 1. What was built (U-phase)

**Goal:** add road-context signal (speed limit, road class, density) to every accident, since the existing feature set had no notion of *where* an accident happened beyond administrative region codes.

**Pipeline:** `src/unfallatlas/data/osm.py` + `src/unfallatlas/features/spatial.py`

1. **Fetch.** For each of Germany's 16 Bundesländer, `download_road_network()` fetches OSM road ways via `osmnx`. A whole-state `graph_from_place()` call reliably **OOM-killed the process** (confirmed via `journalctl`: ~26.4GB RSS, SIGKILL) — osmnx builds the full raw, unsimplified graph before ever simplifying, so `simplify=True` alone didn't help. Fixed by **tiling**: each state is split into a 0.2° lat/lon grid (`_grid_tiles`), and each tile is fetched independently via `graph_from_bbox(..., truncate_by_edge=True)`, keeping peak memory in the hundreds-of-MB range regardless of state size (validated against Baden-Württemberg and Bayern, the two states that had previously OOM'd).
2. **Cache.** Every tile is cached individually (`data/raw/osm/<state>_tiles/tile_NNNN.parquet`), so an interrupted multi-day run resumes instead of restarting. `_TransientFetchError` distinguishes "Overpass backend hiccupped, retry later" from "confirmed no roads in this tile" so flaky failures are never cached as empty.
3. **Reliability hardening.** A state-level retry loop (max 5 attempts, only re-fetching tiles still missing) resolves most states within one call; the existing cross-state retry pass (max 3 passes) is the outer safety net. An adaptive rolling-average ETA (per-state and whole-run, both cache-aware) makes an unattended overnight run monitorable.
4. **Dedup.** `truncate_by_edge=True` means boundary-crossing edges appear in both neighboring tiles (deliberate, to avoid grid-aligned gaps) — deduped after concatenation on `(highway, maxspeed, geometry-as-WKB)`.
5. **Aggregate.** `aggregate_roads_to_h3()` rolls every road-vertex point up to its H3 resolution-8 cell (~0.7 km²), vectorized via `shapely.get_coordinates()` + numpy indexing (the original nested-`iterrows()` version would have been tens of millions of Python-level dict allocations at Baden-Württemberg's real scale — never actually exercised until the OOM fix landed).
6. **Join.** `build_spatial_features()` joins the per-cell aggregates onto every accident by its own H3 cell, producing:
   - `osm_dominant_road_class` — highest-ranked road class touching the cell (categorical, 15 possible values)
   - `osm_maxspeed_mean` / `osm_maxspeed_max` — parsed speed limit stats, km/h
   - `osm_road_density` — road-vertex-point count in the cell (proxy for road presence/length)
   - `osm_way_count` — distinct-way count in the cell (junction/complexity proxy)

**First full successful 16-state run** (2026-07-09, after the tiling fix): 2,546 tiles fetched across all 16 states, joined onto 2,092,401 accident rows. Coverage: 100% for road class/density/way-count, 96.2% for maxspeed (not every OSM way carries a speed-limit tag).

**Known, accepted limitation (documented, not solved):** OSM reflects the *present-day* road network; accidents span 2016–2024, and some roads' classification/speed limit will have changed since. Same category of approximation as the existing DWD weather join's day-of-month averaging.

---

## 2. What was wired into A³ (this plan)

Prior to this plan, the 5 OSM columns existed in the cached training frame (`data/interim/accidents_with_weather_spatial.parquet`) but `build_preprocessor()` didn't reference them — `ColumnTransformer(remainder="drop")` silently dropped them, same as it still does for `h3_cell` (the join key) and `dwd_station_id`.

This plan (`docs/superpowers/plans/2026-07-09-a3-osm-feature-integration.md`) did four things:

1. **Wired the 5 columns into `build_preprocessor()`**, exactly per the U-phase §10 decision table (which was written when the features were designed, months before this integration — "U decides, A³ implements"):
   - `osm_dominant_road_class` → new pipeline: impute missing as `"unknown"` (0.0065% of rows) → one-hot encode.
   - `osm_maxspeed_mean`/`osm_maxspeed_max` → added to the existing `PLAIN_NUMERIC_COLUMNS` list (median impute, `StandardScaler`) — same treatment as `dwd_temp_air_2m`.
   - `osm_road_density`/`osm_way_count` → added to the existing `LOG1P_COLUMNS` list (zero-fill, `log1p`, `StandardScaler`) — same treatment as `dwd_precip_mm`.
2. **Fixed 3 Minor findings** deferred from the A³ champion-pivot plan's final review: a latent `{family}_default`-row selection bug in §6 (tightened to an explicit allow-list), an undocumented dead CV-strategy cell in §2 (now explicitly marked as an illustrative sanity check, variables renamed to make that obvious), and a missing committed CSV of the full model-comparison table (now saved to `data/processed/a3_model_comparison.csv`, since `nbstripout` strips all notebook outputs on commit and `reports/a3_progress.log` is git-ignored — this CSV is now the only durable, git-tracked record of the full ~19-row comparison table).
3. **Retrained everything.** Every single cached model (baselines, tree ensembles, imbalance strategies, tuned configs) was invalid the moment the feature matrix's shape changed — no checkpoint was carried forward, unlike every prior A³ plan. Full re-run: Stufe 0/1 (11 models) → §6 imbalance-strategy comparison (4 strategies × 2 families) → §7 Optuna tuning (9 trials × 2 families) → §8 final refit + single test-2024 evaluation.
4. **Documented the build** in `docs/AI TOOL DISCLOSURE.md`.

---

## 3. Measured impact

| | Pre-OSM (champion-pivot plan) | Post-OSM (this plan) | Δ |
|---|---|---|---|
| Champion family | lightgbm (tuned) | lightgbm (tuned) | same family won both times |
| Test-2024 macro-F1 | 0.358 | 0.362 | **+0.004** |
| Test-2024 recall(class 1) | 0.615 | 0.649 | **+0.034** |
| Q-phase gate (macro-F1≥0.55 **and** recall(1)≥0.50) | FAIL | FAIL | unchanged |

Both metrics moved in the right direction. Neither moved enough to flip the gate outcome — macro-F1 is still ~0.19 short of the 0.55 threshold.

### Why the improvement was small, and what the Optuna trial spread suggests

During the live retrain, every Optuna trial's (macro-F1, recall(1)) pair sat on what looks like a single precision/recall tradeoff frontier rather than showing any trial pushing both metrics up together:

| Trial | macro-F1 | recall(1) |
|---|---|---|
| catboost trial 1 | 0.415 | 0.248 |
| catboost trial 2 | 0.348 | 0.694 |
| catboost trial 9 | 0.408 | 0.252 |
| lightgbm trial 1 | 0.423 | 0.133 |
| lightgbm trial 3 | 0.373 | 0.655 |

High-macro-F1 trials collapse recall(1) to 0.13–0.25; high-recall(1) trials cap macro-F1 around 0.35–0.37. This is the classic imbalanced-3-class shape (class shares ≈1%/18%/81%): pushing the rare class's recall trades directly against overall macro-F1, because macro-F1 also rewards not hurting the two majority classes. **Tuned CatBoost's best selected trial (0.374/0.626) barely beat the untuned §6 comparison's `lightgbm_balanced` (0.372/0.621)** — a ~0.002 macro-F1 gain after 9 trials. That's consistent with hitting a genuine plateau for this hyperparameter space, not an under-searched one.

**Implication for future brainstorming:** widening the Optuna search (more trials, larger ranges) is unlikely to help much on its own — the evidence points to a tradeoff-frontier plateau, not a search-coverage gap. Higher-leverage directions, not yet explored and out of scope for this plan:
- **Per-fold-safe resampling inside Optuna's CV** — `_build_pipeline_for` currently raises `NotImplementedError` if SMOTE/ADASYN/ordinal ever wins a family's §6 comparison, because implementing a fold-safe `imblearn.pipeline.Pipeline` resampling step inside `cross_validate` was explicitly deferred as real scope expansion in the champion-pivot plan. This has never actually been tried under tuning.
- **A deliberately-tuned threshold-moving step** specifically optimized for a target point on the tradeoff curve (rather than compared as one fixed strategy among four).
- **Additional feature sources** beyond OSM — the literature anchor (`docs/project/PROJEKTPLAN_SETUP.md`) predicted CatBoost/threshold-moving reaching 0.65–0.72 macro-F1, which hasn't materialized yet even with OSM added; this suggests either more/better features are still missing, or the ceiling for this feature set + model family combination genuinely sits below the literature anchor's optimistic case.
- **Revisiting the ordinal (Frank-Hall) decomposition** with the enlarged feature set specifically, since it wasn't the strategy that won for either family this run but hasn't been tuned on its own.

---

## 4. Artifacts from this work

- `data/raw/osm/<state>.parquet` (16 files, git-ignored, local cache)
- `data/interim/accidents_with_weather_spatial.parquet` (committed, LFS)
- `data/processed/a3_best_model.joblib` (7.6MB, committed, LFS)
- `data/processed/a3_model_card.json` (committed)
- `data/processed/a3_model_comparison.csv` (19 rows, committed — new in this plan)
- `docs/superpowers/plans/2026-07-06-u-phase-osm-spatial-features.md` (U-phase build plan, complete)
- `docs/superpowers/plans/2026-07-09-a3-osm-feature-integration.md` (this integration plan, complete)
