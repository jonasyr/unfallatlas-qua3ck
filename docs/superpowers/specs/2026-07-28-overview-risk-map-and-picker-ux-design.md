# Overview Relative-Risk Map and Location-Picker UX: Design

**Date:** 2026-07-28
**Scope:** Phase K Streamlit app - Overview severity map correctness/legibility, Risk
Predictor location-picker interaction bug, supporting tests and documentation.
**Status:** Approved for implementation planning.

---

## Problem statement

Three defects in the shipped Phase K app, all user-reported, all confirmed against the
real data in this repository rather than assumed.

### P1: The severity map's coloring is analytically wrong, not just plain-looking

`build_severity_map` colors a grid cell red when `ksi_share >= 0.5`, i.e. when
killed-or-seriously-injured accidents are the local *majority*. KSI is a minority
outcome nationally, so this threshold is far above any meaningful notion of "worse than
normal":

| Measured quantity | Value |
| --- | --- |
| National KSI rate (`UKATGEORIE IN (1,2)` over 2,092,401 rows) | **0.18914** |
| Grid cells at `precision=0.1` | 4,857 |
| Cells the current rule colors red | **123 (2.5%)** |

A cell must reach 2.64x the national rate before it changes color at all. The
consequence is that 97.5% of the map renders in a single color, so the visualization
distinguishes only the most extreme outliers and presents a cell at 1.9x the national
rate as identical to the safest cell in Germany. This is a loss of real signal, not a
cosmetic shortcoming.

### P2: The layer legend/toggle is missing, and the reason it is missing is a real bug

An earlier iteration built two `folium.FeatureGroup` layers plus a `folium.LayerControl`
and attached all of them to the `@st.cache_resource`-cached `folium.Map` with
`.add_to()`. At runtime this raised `ReferenceError: feature_group_<hash> is not
defined` in the browser and blanked the entire map. The cause is that
`streamlit-folium` re-injects the rendered map into its own `map_div` execution context
on every rerun, and only rewrites layer variable references correctly for objects passed
through its own `st_folium(feature_group_to_add=..., layer_control=...)` parameters -
layers baked into the `Map` object beforehand do not resolve in that context.

The layers were removed wholesale to restore a working map, which fixed the blank map
but left the user with no legend and no way to isolate cells of interest.

### P3: Grid markers are drawn at bin coordinates, not where the accidents are

`load_severity_grid` groups by `ROUND(LAT, 1)` / `ROUND(LON, 1)` and the map plots those
rounded bin values as the circle center. Accidents inside a cell are not uniformly
distributed within it, so every marker sits at a systematic offset from its own
accidents:

| Offset of bin coordinate from true accident centroid | Distance |
| --- | --- |
| Mean | 1.71 km |
| 95th percentile | 3.80 km |
| Maximum | 6.48 km |

### P4: The Risk Predictor map requires two clicks to place a point

Reported as "I need to click twice for the point to add after some short freeze".
Confirmed as a script-ordering bug, not a rendering delay:

- `app/pages/risk_predictor.py:151` builds `picker_map` and its `folium.Marker` from
  `st.session_state["picked_lat"]` / `["picked_lon"]`.
- `app/pages/risk_predictor.py:165` is where the click handler *updates* those same
  session-state keys.

The handler therefore runs after the map for that rerun has already been constructed.
Click #1 updates state but renders the marker at the previous position; only the rerun
triggered by click #2 reflects click #1. The current page copy documents the symptom as
expected behaviour ("The map reloads after every click, which can take a moment - that's
expected, not a freeze"), which is a caption working around a bug.

The accompanying pause is partly genuine work, but the work is small: the
`nearest_location_features` DuckDB bbox lookup against the 105 MB
`accidents_with_weather_spatial.parquet` measures **0.26 s**. The rest is the
Streamlit rerun round-trip plus re-serialization of the whole folium map, and it is
presented to the user as unexplained dead air.

---

## Design

### D1: Relative-risk banding with sample-size shrinkage

Replace the absolute 50% majority test with a cell's KSI rate **relative to the national
baseline**, computed on a shrunk rate so that thinly-sampled cells cannot reach extreme
values on noise alone.

For each cell:

```
shrunk_rate    = (ksi_count + k * baseline) / (total + k)
relative_risk  = shrunk_rate / baseline
```

where `baseline` is the national KSI rate computed from the same parquet in the same
query, and `k = 20` is a pseudo-count.

**`k = 20` is chosen from measured band populations, not picked arbitrarily.** Candidate
values were evaluated against the real 4,857-cell grid:

| k | `<0.75x` | `0.75-1.1x` | `1.1-1.5x` | `1.5-2x` | `>2x` | Smallest cell in top band |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 258 | 1,079 | 1,833 | 1,352 | 335 | 6 accidents |
| **20** | **239** | **1,137** | **2,014** | **1,263** | **204** | **10 accidents** |
| 50 | 220 | 1,278 | 2,380 | 927 | 52 | 64 accidents |

`k = 10` still admits 6-accident cells into the ">2x" band, which is the noise the
shrinkage exists to suppress. `k = 50` over-shrinks, collapsing the top band to 52 cells
and pulling the map back toward uniformity. `k = 20` populates all five bands, with the
largest band holding 41.5% of cells.

**Bands** (fixed thresholds on `relative_risk`):

| Band | Range | Meaning |
| --- | --- | --- |
| 1 | `< 0.75` | Well below average |
| 2 | `0.75 - 1.1` | Around average |
| 3 | `1.1 - 1.5` | Elevated |
| 4 | `1.5 - 2.0` | High |
| 5 | `>= 2.0` | Very high |

Band boundaries are half-open on the upper edge (`[lo, hi)`), with band 5 unbounded
above, so every cell falls in exactly one band.

**Colors** interpolate the app's existing accent pair
(`SEVERITY_COLORS["slight"] = #2A9D8F` teal to `SEVERITY_COLORS["KSI"] = #E63946` red),
keeping one palette across the app rather than introducing a second scheme.

**Confidence encoding.** Fill opacity is set from the cell's accident count in three
discrete tiers rather than a continuous fade, so the legend can state exactly what a pale
cell means instead of implying unearned precision:

| Accidents in cell | Fill opacity | Reading |
| --- | --- | --- |
| `< 20` | 0.25 | Thin sample, treat as uncertain |
| `20 - 99` | 0.45 | Moderate sample |
| `>= 100` | 0.65 | Well sampled |

**Circle radius** continues to scale with the cell's total accident count, unchanged in
spirit from today. Radius therefore answers "how much traffic activity", color answers
"how severe relative to normal", and opacity answers "how much should I trust this cell" -
three independent questions on three independent visual channels.

### D2: Legend and per-band toggles via the correct streamlit-folium API

Build one `folium.FeatureGroup` per band, named for the band, and pass them to
`st_folium` through its own parameters:

```python
st_folium(
    base_map,
    feature_group_to_add=[fg_band_1, ..., fg_band_5],
    layer_control=True,
    ...
)
```

This is the API path that `streamlit-folium` rewrites correctly for its `map_div`
context, and it is verified present in the installed version (`feature_group_to_add`,
`layer_control`, `center`, `zoom`, `key`, `returned_objects` all confirmed in the
`st_folium` signature).

**Caching boundary - this is the part that previously broke.** The FeatureGroups must be
constructed fresh on every rerun and must never be attached to a cached `Map`:

- `@st.cache_data` keeps the aggregated grid DataFrame (the expensive DuckDB work).
- `@st.cache_resource` keeps only the empty base `folium.Map` (tiles, center, zoom).
- FeatureGroup construction happens per rerun, outside any cache, and the objects reach
  the browser only via `feature_group_to_add`.

The `LayerControl` produced by `layer_control=True` doubles as the legend: each entry is
a named band, so the control simultaneously explains the colors and lets the user isolate
a band (for example, showing only ">2x average" cells to see where they cluster).
Because the control names alone cannot convey color or the opacity convention, a static
markdown legend under the map states the band colors, their numeric ranges, and the
confidence tiers.

### D3: Centroid placement

`load_severity_grid` gains `AVG(LAT)` and `AVG(LON)` per cell in the same single DuckDB
aggregation. `ROUND(LAT, 1)` / `ROUND(LON, 1)` remain as the `GROUP BY` key only; the
centroid becomes the plotted position. No extra query and no extra data crossing into
pandas beyond two float columns on an already-small grouped result.

### D4: Location picker - first-click correctness and honest loading state

Four coordinated changes to `app/pages/risk_predictor.py`:

1. **Marker moves to a FeatureGroup passed via `feature_group_to_add`.** The marker stops
   being part of the map object, so its position is no longer frozen at map-construction
   time.
2. **Base picker map becomes `@st.cache_resource`.** With the marker extracted, the base
   map is static, so it no longer needs re-serializing on every interaction.
3. **`st.rerun()` after a click that changed state.** This guarantees the marker,
   coordinates, and all auto-filled context fields reflect the click that just happened
   rather than the previous one. The existing `last_processed_click` guard already
   prevents the same click from being reprocessed, so this cannot loop.
4. **`st.spinner("Reading road context for this location...")` around the
   `nearest_location_features` lookup.** The 0.26 s query becomes a labeled loading state
   instead of unexplained dead air.

`st_folium`'s `center=` parameter follows the picked point so the view tracks the
selection without rebuilding the map.

The caption that currently describes the reload pause as expected behaviour is deleted
rather than reworded: it documented a bug that this design removes.

### D5: Module boundary

All new logic that is not a widget call lives in `src/unfallatlas/viz/streamlit_app.py`,
consistent with the established boundary (that module contains cached loaders and pure
helpers only; `app/pages/*.py` contains the `st.*` calls). Specifically, band assignment,
color interpolation, and opacity tiering become **pure functions with no Streamlit
dependency**, so they are unit-testable without a Streamlit runtime and without touching
the 105 MB parquet.

---

## Testing strategy

### Unit tests (pure, fast, no Streamlit runtime)

Band and styling logic is pure arithmetic and gets exercised directly:

- Exact band boundaries, including that a cell landing precisely on 0.75, 1.1, 1.5, and
  2.0 falls in the upper band (half-open intervals).
- Shrinkage arithmetic: a cell with zero accidents' worth of evidence returns exactly the
  baseline; a very large cell converges to its raw rate.
- Opacity tier boundaries at 20 and 100 accidents.
- Every band maps to a distinct color, and the extreme bands map to the app's existing
  teal and red anchors.

### Data-layer tests

- `load_severity_grid` returns the new centroid and relative-risk columns.
- Centroid values fall inside their own cell's bounds (a centroid cannot be more than
  half a cell width from its bin coordinate).
- The national baseline the loader computes matches an independently computed value.

### Page tests (`AppTest`)

Existing `tests/test_overview_page.py` and `tests/test_risk_predictor_page.py` patterns
extend to assert no exception is raised and that the new legend content is present.

### Browser test (mandatory gate)

`AppTest` executes the Python script but never runs the frontend, and it provably did
**not** catch the previous blank-map failure - it reported zero exceptions while the map
was completely broken in the browser. Because this design deliberately reintroduces
`feature_group_to_add` and `layer_control`, the exact API involved in that failure, a
headless-browser check is a required gate rather than an optional extra:

- Launch the app, visit Overview and Risk Predictor.
- Register a `pageerror` listener and assert **zero** JS exceptions.
- Assert the Leaflet map container and the layer-control element are actually present in
  the DOM, and that the expected number of circle paths rendered.
- Click the picker map once and assert the coordinate readout updates from that single
  click (the P4 regression test).

Environment note carried forward from earlier work in this repository: the Playwright MCP
tool's default `chrome` channel is not installed in this sandbox and cannot be installed
without root. The working approach is the project venv's own `playwright` package,
launching the already-downloaded chromium at
`/home/jonas/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome` with
`headless=True, args=["--no-sandbox"]`.

---

## Non-goals

- **The animated "Risk Scan" simulator page.** Discussed in the same session and
  explicitly deferred by the user to its own spec, plan, and branch.
- Switching the grid from the 0.1-degree lat/lon binning to H3 cells. The existing
  binning is adequate for this fix and the change would be unrelated churn.
- Any change to model artifacts, training code, notebooks, or the inference contract.
- Any new runtime dependency. `folium`, `streamlit-folium`, `branca` (a folium
  dependency), and `duckdb` are all already present.
- Re-exporting the notebook presentation HTML.

---

## Honest limitations to state in the app

- **Relative risk is not causal.** A cell at 2x the national KSI rate is not thereby
  shown to be dangerous *because of* its road layout; severity mix correlates with rural
  road type, speed, and time-to-treatment, none of which this map controls for.
- **Shrinkage is a deliberate bias.** It pulls small cells toward the national average on
  purpose, which means a genuinely dangerous small cell is understated. The opacity
  channel is what keeps this visible instead of hidden.
- **Reporting-driven denominator.** The dataset records police-reported personal-injury
  accidents only, so a cell's total is a function of both traffic volume and reporting
  behaviour.
- **Grid cells are not administrative units.** A ~11 km cell can straddle a city boundary
  and a rural road, mixing two very different severity regimes into one number.

---

## Expected finding worth surfacing in the page copy

The new banding exposes a real inversion that the current map hides entirely. Median
accident count per cell, by band:

| Band | Median accidents per cell |
| --- | --- |
| `< 0.75x` (well below average) | 1,339 |
| `>= 2x` (very high) | 91 |

Cells with the *most* accidents are the *least* severe on average - dense urban areas
generate many slight-injury collisions, while rural cells generate far fewer accidents
that are far more often KSI. This is a legitimate, defensible reading of the project's
own data and directly reinforces the app's existing framing of why severity is hard to
predict from volume-like features.
