# Overview Relative-Risk Map and Picker UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Overview severity map's absolute 50%-majority coloring with
shrinkage-corrected relative-risk bands that carry a working legend and per-band toggles,
plot cells at their true accident centroids, and make the Risk Predictor's map picker
respond to the first click instead of the second.

**Architecture:** All non-widget logic goes into `src/unfallatlas/viz/streamlit_app.py`
as pure functions (band assignment, opacity tiering, legend text) plus cached loaders;
`app/pages/*.py` keeps only `st.*` calls. Toggleable map layers reach the browser
exclusively through `st_folium(feature_group_to_add=..., layer_control=True)` and are
never attached to a cached `folium.Map` with `.add_to()`, which is what previously
produced a JS `ReferenceError` and a blank map.

**Tech Stack:** Python 3.12, Streamlit, folium + streamlit-folium, DuckDB, pandas,
pytest, `streamlit.testing.v1.AppTest`, Playwright (headless chromium), uv.

## Global Constraints

- Work on branch `feat/overview-risk-map-and-simulator`. Do not merge to `main`.
- **No `Co-Authored-By` trailers in any commit.** Conventional Commit subjects only.
- Use `uv` exclusively: `uv run pytest`, `uv run ruff check .`, `uv run ruff format`,
  `uv run streamlit run app/streamlit_app.py`.
- **No new runtime dependencies.** `folium`, `streamlit-folium`, `branca`, `duckdb`,
  `playwright` are all already available.
- Query `data/accidents.parquet` through DuckDB only. Never load it into pandas
  row-by-row; only grouped results (a few thousand rows) may cross into pandas.
- `src/unfallatlas/viz/streamlit_app.py` must contain **no `st.*` widget calls** - only
  `st.cache_data` / `st.cache_resource` decorated loaders and plain functions, so it stays
  importable without a Streamlit runtime.
- Toggleable layers go through `st_folium`'s `feature_group_to_add=` / `layer_control=`
  parameters. **Never** `.add_to()` a `FeatureGroup` or `LayerControl` onto a cached map.
- All existing tests must keep passing. Current baseline: **399 passed, 9 deselected**
  (`-m "not browser"` is in `addopts`).
- New Playwright tests carry `@pytest.mark.browser` so they stay opt-in, matching
  `tests/presentation/test_browser.py`.
- Exact measured constants, to be used verbatim, not re-derived or rounded differently:
  - National KSI rate: `395766 / 2092401 = 0.1891444326398238`
  - Shrinkage pseudo-count: `k = 20`
  - Grid cells at `precision=0.1`: `4857`
  - Band populations at `k=20`: `239 / 1137 / 2014 / 1263 / 204`

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `src/unfallatlas/viz/streamlit_app.py` | Cached loaders + pure helpers | Modify: add band/opacity/legend helpers, centroid columns, baseline loader, replace `build_severity_map` with a base-map builder plus a FeatureGroup builder |
| `app/pages/overview.py` | Overview widgets | Modify: new `st_folium` call with layers, new legend + caption |
| `app/pages/risk_predictor.py` | Risk Predictor widgets | Modify: marker as FeatureGroup, cached base map, `st.rerun()` after click, spinner, caption cleanup |
| `tests/test_streamlit_app.py` | Pure-helper + loader tests | Modify: add band/opacity/centroid/baseline tests |
| `tests/test_overview_page.py` | Overview `AppTest` | Modify: assert legend renders |
| `tests/test_risk_predictor_page.py` | Risk Predictor `AppTest` | Modify: assert no exception with new map wiring |
| `tests/conftest.py` | Shared `chromium_browser` fixture | Create (moved from `tests/presentation/test_browser.py`) |
| `tests/presentation/test_browser.py` | Presentation browser tests | Modify: drop the now-shared local fixture |
| `tests/test_streamlit_browser.py` | Live-app browser gate | Create |
| `docs/AI TOOL DISCLOSURE.md` | Provenance | Modify: one disclosure row, one plan-index row |
| `docs/prompts/05_prompts_phase_k.md` | Prompt record | Modify: new section |

### Two deliberate deviations from the spec, with rationale

Both are decided here so no implementer has to guess:

1. **Band colors are a hand-picked 5-stop ramp, not computed RGB interpolation.** The
   spec says colors "interpolate" teal `#2A9D8F` to red `#E63946`. Linear RGB
   interpolation between those two passes through muddy brown-grey
   (`#886B6A` at the midpoint), which is unreadable as a risk ramp. The plan instead
   uses a fixed diverging ramp that keeps both brand anchors as its endpoints and
   passes through a warm sand midpoint. This is also less code and needs no `branca`
   colormap.
2. **`st_folium`'s `center=` parameter is NOT used on the picker.** The spec mentioned
   it. Passing `center` on every rerun risks snapping the user's view (and possibly
   zoom) back after each click, which would be worse UX than the bug being fixed.
   Re-centering is unnecessary anyway: the user clicked a point that is by definition
   already visible. `feature_group_to_add` updates the marker without touching the view,
   which is precisely its purpose.

---

## Task 1: Pure risk-band, opacity, and legend helpers

**Files:**
- Modify: `src/unfallatlas/viz/streamlit_app.py` (add after the `SEVERITY_COLORS`
  constant at line 32)
- Test: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `SEVERITY_COLORS` (existing constant, `{"KSI": "#E63946", "slight": "#2A9D8F"}`)
- Produces, all used by Tasks 2 and 3:
  - `NATIONAL_KSI_RATE_FALLBACK: float` = `0.1891444326398238`
  - `SHRINKAGE_K: int` = `20`
  - `RiskBand` - `NamedTuple` with fields `label: str`, `lower: float`, `upper: float`, `color: str`
  - `RISK_BANDS: tuple[RiskBand, ...]` - exactly 5 entries, ascending
  - `shrunk_relative_risk(ksi_count: int, total: int, baseline: float, k: int = SHRINKAGE_K) -> float`
  - `risk_band_index(relative_risk: float) -> int` - returns `0..4`
  - `confidence_opacity(total: int) -> float`
  - `severity_legend_markdown() -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streamlit_app.py`. Add the new names to the existing
`from unfallatlas.viz.streamlit_app import (...)` block at the top of the file
(keep it alphabetically sorted - ruff's isort rules are enforced).

```python
def test_shrunk_relative_risk_returns_baseline_ratio_for_empty_cell():
    # A cell with no accidents is pure prior: shrunk rate == baseline, ratio == 1.0
    assert shrunk_relative_risk(0, 0, 0.1891444326398238) == pytest.approx(1.0)


def test_shrunk_relative_risk_converges_to_raw_rate_for_large_cell():
    # 100_000 accidents at a 40% KSI rate: shrinkage of k=20 is negligible
    result = shrunk_relative_risk(40_000, 100_000, 0.1891444326398238)
    raw_ratio = 0.4 / 0.1891444326398238
    assert result == pytest.approx(raw_ratio, rel=1e-3)


def test_shrunk_relative_risk_pulls_small_noisy_cell_toward_baseline():
    # 1 accident, and it was KSI. Raw rate is 100% (5.29x baseline) - pure noise.
    # Shrinkage must pull it far down, below the "very high" 2.0x band threshold.
    baseline = 0.1891444326398238
    raw_ratio = 1.0 / baseline
    shrunk = shrunk_relative_risk(1, 1, baseline)
    assert raw_ratio > 5.0
    assert shrunk < 2.0


def test_shrunk_relative_risk_uses_explicit_k():
    baseline = 0.2
    # (ksi + k*baseline) / (total + k) / baseline
    # k=0 disables shrinkage entirely -> raw ratio
    assert shrunk_relative_risk(5, 10, baseline, k=0) == pytest.approx(0.5 / 0.2)


def test_shrunk_relative_risk_rejects_non_positive_baseline():
    with pytest.raises(ValueError, match="baseline"):
        shrunk_relative_risk(1, 10, 0.0)


def test_risk_bands_are_five_contiguous_ascending_ranges():
    assert len(RISK_BANDS) == 5
    assert RISK_BANDS[0].lower == 0.0
    assert RISK_BANDS[-1].upper == float("inf")
    for lower_band, upper_band in zip(RISK_BANDS, RISK_BANDS[1:], strict=True):
        # contiguous: no gaps, no overlaps
        assert lower_band.upper == upper_band.lower


def test_risk_bands_endpoints_reuse_the_app_severity_palette():
    assert RISK_BANDS[0].color == SEVERITY_COLORS["slight"]
    assert RISK_BANDS[-1].color == SEVERITY_COLORS["KSI"]


def test_risk_bands_all_have_distinct_colors_and_labels():
    assert len({band.color for band in RISK_BANDS}) == 5
    assert len({band.label for band in RISK_BANDS}) == 5


@pytest.mark.parametrize(
    ("relative_risk", "expected_index"),
    [
        (0.0, 0),
        (0.74, 0),
        (0.75, 1),  # boundaries are half-open [lower, upper): 0.75 lands in band 1
        (1.0, 1),
        (1.09, 1),
        (1.1, 2),
        (1.49, 2),
        (1.5, 3),
        (1.99, 3),
        (2.0, 4),
        (100.0, 4),
    ],
)
def test_risk_band_index_uses_half_open_intervals(relative_risk, expected_index):
    assert risk_band_index(relative_risk) == expected_index


def test_risk_band_index_rejects_negative_risk():
    with pytest.raises(ValueError, match="relative_risk"):
        risk_band_index(-0.1)


@pytest.mark.parametrize(
    ("total", "expected_opacity"),
    [(0, 0.25), (19, 0.25), (20, 0.45), (99, 0.45), (100, 0.65), (25_916, 0.65)],
)
def test_confidence_opacity_tiers(total, expected_opacity):
    assert confidence_opacity(total) == expected_opacity


def test_severity_legend_markdown_states_every_band_and_the_baseline():
    legend = severity_legend_markdown()
    for band in RISK_BANDS:
        assert band.label in legend
        assert band.color in legend
    # The national baseline must be stated as a percentage, so "2x average" is readable
    assert "18.9" in legend
    # The opacity convention must be explained, not left implicit
    assert "20" in legend and "100" in legend
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "risk_band or shrunk or confidence_opacity or legend_markdown" -v`

Expected: collection error, `ImportError: cannot import name 'shrunk_relative_risk'`.

- [ ] **Step 3: Implement the helpers**

Insert into `src/unfallatlas/viz/streamlit_app.py` immediately after the
`SEVERITY_COLORS` line. Add `from typing import NamedTuple` to the existing import
block (ruff enforces sorted imports; `typing` sorts after `pathlib`).

```python
# Measured over the full committed data/accidents.parquet:
# 395766 KSI rows (UKATGEORIE IN (1, 2)) out of 2092401 total.
# Used only as a fallback if the DuckDB baseline query is unavailable; the live
# value comes from load_national_ksi_rate().
NATIONAL_KSI_RATE_FALLBACK = 0.1891444326398238

# Pseudo-count for shrinking a cell's KSI rate toward the national baseline, so a
# cell with 1 accident cannot register as "100% KSI". Chosen by measuring band
# populations over the real 4857-cell grid: k=10 still admits 6-accident cells into
# the top ">=2x" band (the noise shrinkage exists to suppress), while k=50
# over-shrinks it to 52 cells and flattens the map. k=20 fills all five bands
# (239 / 1137 / 2014 / 1263 / 204), with the largest band holding 41.5% of cells.
SHRINKAGE_K = 20


class RiskBand(NamedTuple):
    """One relative-risk band: a half-open [lower, upper) ratio range and its color."""

    label: str
    lower: float
    upper: float
    color: str


# Diverging ramp anchored on the app's two existing brand colors
# (SEVERITY_COLORS["slight"] teal -> SEVERITY_COLORS["KSI"] red) with a warm sand
# midpoint. Deliberately hand-picked rather than RGB-interpolated: linear
# interpolation between teal and red passes through muddy brown-grey (#886B6A at
# the midpoint), which does not read as an ordered risk scale.
RISK_BANDS = (
    RiskBand("Well below average (<0.75x)", 0.0, 0.75, SEVERITY_COLORS["slight"]),
    RiskBand("Around average (0.75-1.1x)", 0.75, 1.1, "#8FC7BE"),
    RiskBand("Elevated (1.1-1.5x)", 1.1, 1.5, "#F2C879"),
    RiskBand("High (1.5-2x)", 1.5, 2.0, "#EE8062"),
    RiskBand("Very high (>=2x)", 2.0, float("inf"), SEVERITY_COLORS["KSI"]),
)

# Fill-opacity tiers by a cell's accident count, so a thinly-sampled cell reads as
# uncertain instead of looking as confident as a well-sampled one. Three discrete
# steps rather than a continuous fade, so the legend can state exactly what a pale
# cell means instead of implying unearned precision.
CONFIDENCE_OPACITY_TIERS = ((20, 0.25), (100, 0.45))
CONFIDENCE_OPACITY_MAX = 0.65


def shrunk_relative_risk(
    ksi_count: int, total: int, baseline: float, k: int = SHRINKAGE_K
) -> float:
    """Return a cell's KSI rate relative to the national baseline, shrunk toward it.

    An absolute "is KSI the local majority?" test is the wrong question: KSI is a
    ~18.9% minority outcome nationally, so a cell must reach 2.64x the national rate
    before crossing 50%. Only 123 of 4857 cells do, leaving 97.5% of the map a single
    color. Relative risk asks the answerable question - "is this cell worse than
    normal, and by how much?" - and the k-weighted shrinkage stops small cells from
    reaching extremes on noise.

    Returns 1.0 (exactly baseline) for an empty cell, since with no evidence the best
    estimate is the prior.
    """
    if baseline <= 0:
        raise ValueError(f"baseline must be positive, got {baseline}")
    shrunk_rate = (ksi_count + k * baseline) / (total + k)
    return shrunk_rate / baseline


def risk_band_index(relative_risk: float) -> int:
    """Return the index into RISK_BANDS for a relative-risk ratio.

    Bands are half-open [lower, upper), so a ratio landing exactly on a boundary
    falls into the higher band and every ratio maps to exactly one band.
    """
    if relative_risk < 0:
        raise ValueError(f"relative_risk must be non-negative, got {relative_risk}")
    for index, band in enumerate(RISK_BANDS):
        if relative_risk < band.upper:
            return index
    return len(RISK_BANDS) - 1


def confidence_opacity(total: int) -> float:
    """Return the fill opacity for a cell, lower when its sample is thin."""
    for threshold, opacity in CONFIDENCE_OPACITY_TIERS:
        if total < threshold:
            return opacity
    return CONFIDENCE_OPACITY_MAX


def severity_legend_markdown() -> str:
    """Render the static map legend as markdown with inline color swatches.

    folium's LayerControl names the bands but cannot show their colors or explain the
    opacity convention, so this legend carries both.
    """
    swatches = "\n".join(
        f'- <span style="display:inline-block;width:0.85rem;height:0.85rem;'
        f'background:{band.color};border:1px solid rgba(0,0,0,0.25);'
        f'vertical-align:middle;margin-right:0.5rem"></span>{band.label}'
        for band in RISK_BANDS
    )
    return (
        "**Relative KSI risk** - each cell's share of killed/seriously-injured "
        "accidents, compared against the national average of **18.9%**. "
        f"A cell at 2x is twice as likely to be severe as Germany overall.\n\n"
        f"{swatches}\n\n"
        "Rates are shrunk toward the national average in proportion to how few "
        "accidents a cell has, so a single severe accident cannot mark a cell as "
        "high-risk. Fill opacity shows that confidence directly: cells with fewer "
        "than 20 accidents are rendered faintest, under 100 mid-way, and 100 or "
        "more at full strength. Circle size still scales with the cell's total "
        "accident count."
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -k "risk_band or shrunk or confidence_opacity or legend_markdown" -v`

Expected: all PASS (25 tests, counting parametrized cases).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
uv run ruff check src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git add src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat(viz): add shrinkage-corrected relative-risk bands for the severity map"
```

---

## Task 2: Centroid positions and the national baseline in the data layer

**Files:**
- Modify: `src/unfallatlas/viz/streamlit_app.py:336-361` (`load_severity_grid`)
- Test: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `ACCIDENTS_PARQUET`, `precision_decimals(precision: float) -> int` (both existing)
- Produces:
  - `load_national_ksi_rate() -> float` - `@st.cache_data`
  - `load_severity_grid(precision: float = 0.1) -> pd.DataFrame` - same name, now with
    columns `lat_bin`, `lon_bin`, `center_lat`, `center_lon`, `ksi_count`,
    `slight_count`, `total`

**Context:** the current query groups by `ROUND(LAT, 1)` / `ROUND(LON, 1)` and the map
plots those rounded bin values. Accidents are not uniformly spread inside a cell, so
markers sit a measured mean of 1.71 km (p95 3.80 km, max 6.48 km) away from their own
accidents. Adding `AVG(LAT)`/`AVG(LON)` to the same aggregation fixes this with no extra
query.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streamlit_app.py` and add `load_national_ksi_rate` to the import
block.

```python
def test_load_national_ksi_rate_matches_the_measured_value():
    # 395766 KSI rows out of 2092401 in the committed parquet
    assert load_national_ksi_rate() == pytest.approx(0.1891444326398238)


def test_load_severity_grid_has_expected_shape_and_columns():
    grid = load_severity_grid()
    assert len(grid) == 4857
    assert {
        "lat_bin",
        "lon_bin",
        "center_lat",
        "center_lon",
        "ksi_count",
        "slight_count",
        "total",
    }.issubset(grid.columns)


def test_load_severity_grid_counts_sum_to_the_full_dataset():
    grid = load_severity_grid()
    assert grid["total"].sum() == 2092401
    assert (grid["ksi_count"] + grid["slight_count"] == grid["total"]).all()


def test_load_severity_grid_centroids_lie_inside_their_own_cell():
    # A centroid of points that all round to the same 0.1-degree bin cannot be more
    # than half a bin width away from that bin's coordinate. Verified: 0 violations.
    grid = load_severity_grid()
    assert ((grid["center_lat"] - grid["lat_bin"]).abs() <= 0.05).all()
    assert ((grid["center_lon"] - grid["lon_bin"]).abs() <= 0.05).all()


def test_load_severity_grid_centroids_actually_differ_from_bin_coordinates():
    # Guards against a regression that silently aliases center_lat back to lat_bin.
    grid = load_severity_grid()
    assert (grid["center_lat"] != grid["lat_bin"]).sum() > 4000


def test_severity_grid_bands_populate_every_band():
    # The whole point of the redesign: all five bands carry cells, instead of the
    # old 50%-majority rule that colored only 123 of 4857 cells red.
    grid = load_severity_grid()
    baseline = load_national_ksi_rate()
    indices = [
        risk_band_index(shrunk_relative_risk(row.ksi_count, row.total, baseline))
        for row in grid.itertuples()
    ]
    counts = {index: indices.count(index) for index in range(5)}
    assert counts == {0: 239, 1: 1137, 2: 2014, 3: 1263, 4: 204}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -k "national_ksi_rate or severity_grid" -v`

Expected: `ImportError: cannot import name 'load_national_ksi_rate'`.

- [ ] **Step 3: Implement the loader changes**

Insert `load_national_ksi_rate` immediately before `load_severity_grid` in
`src/unfallatlas/viz/streamlit_app.py`:

```python
@st.cache_data
def load_national_ksi_rate() -> float:
    """Return the share of all accidents that are KSI, across the whole dataset.

    This is the baseline the Overview map's relative-risk bands are measured
    against. Computed live from the committed parquet rather than hardcoded, so it
    stays correct if the dataset is ever revised.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT AVG(CASE WHEN UKATGEORIE IN (1, 2) THEN 1.0 ELSE 0.0 END)
            FROM '{ACCIDENTS_PARQUET}'
        """  # noqa: S608
        return float(con.execute(query).fetchone()[0])
    except Exception:
        logger.exception("Failed to load national KSI rate")
        raise
```

Then replace the `query` string inside `load_severity_grid` (keep the decorator,
signature, `try`/`except`, and docstring, extending the docstring as shown):

```python
@st.cache_data
def load_severity_grid(precision: float = 0.1) -> pd.DataFrame:
    """Aggregate accidents.parquet into a lat/lon grid with per-cell KSI/slight counts.

    UKATGEORIE in (1, 2) is KSI, UKATGEORIE == 3 is slight (verified against
    notebooks/01_Q_Phase.py lines 156-157; no invalid UKATGEORIE values exist
    in this file). Aggregation happens entirely in DuckDB so the ~2.09M-row
    parquet is never loaded into pandas row-by-row - only the grouped result
    (a few thousand rows at precision=0.1) crosses into pandas.

    `lat_bin`/`lon_bin` are the grouping key only. `center_lat`/`center_lon` are the
    mean coordinates of the accidents actually inside the cell, and are what callers
    should plot: the rounded bin coordinate sits a mean of 1.71 km (p95 3.80 km, max
    6.48 km) away from the cell's own accidents, because accidents are not uniformly
    distributed inside a ~11 km cell.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT
                ROUND(LAT, {precision_decimals(precision)}) AS lat_bin,
                ROUND(LON, {precision_decimals(precision)}) AS lon_bin,
                AVG(LAT) AS center_lat,
                AVG(LON) AS center_lon,
                SUM(CASE WHEN UKATGEORIE IN (1, 2) THEN 1 ELSE 0 END) AS ksi_count,
                SUM(CASE WHEN UKATGEORIE = 3 THEN 1 ELSE 0 END) AS slight_count,
                COUNT(*) AS total
            FROM '{ACCIDENTS_PARQUET}'
            GROUP BY 1, 2
        """  # noqa: S608
        return con.execute(query).df()
    except Exception:
        logger.exception("Failed to load severity grid")
        raise
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -k "national_ksi_rate or severity_grid" -v`

Expected: all PASS. The band-population test confirms the exact
`239 / 1137 / 2014 / 1263 / 204` split.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
uv run ruff check src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git add src/unfallatlas/viz/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat(viz): plot severity cells at accident centroids and expose the KSI baseline"
```

---

## Task 3: Banded, toggleable Overview severity map

**Files:**
- Modify: `src/unfallatlas/viz/streamlit_app.py:364-409` (replace `build_severity_map`)
- Modify: `app/pages/overview.py:41-52`
- Test: `tests/test_overview_page.py`, `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `load_severity_grid`, `load_national_ksi_rate`, `shrunk_relative_risk`,
  `risk_band_index`, `confidence_opacity`, `RISK_BANDS`, `severity_legend_markdown`
- Produces:
  - `build_severity_base_map() -> folium.Map` - `@st.cache_resource`, an **empty** map
  - `build_severity_feature_groups(precision: float = 0.1) -> list[folium.FeatureGroup]`
    - `@st.cache_resource`, exactly 5 groups, one per band, ascending
  - `build_severity_map` is **removed**. Its only caller is `app/pages/overview.py`.

**Critical context - the bug this task must not reintroduce.** A previous version built
FeatureGroups and a `LayerControl` and attached them to the cached `folium.Map` with
`.add_to()`. In the browser that raised `ReferenceError: feature_group_<hash> is not
defined` and rendered a completely blank map, while `AppTest` reported zero exceptions.
`streamlit-folium` re-injects the rendered map into its own `map_div` execution context
and only rewrites layer variable references for objects passed through its own
`st_folium(feature_group_to_add=..., layer_control=...)` parameters. Therefore:

- `build_severity_base_map()` returns a map with **no layers attached at all**.
- FeatureGroups are returned as a plain list and handed to `st_folium` via
  `feature_group_to_add=`.
- **Never** call `.add_to(base_map)` on a FeatureGroup or a `LayerControl`.
- `layer_control=True` is passed to `st_folium`; no `folium.LayerControl` is constructed
  by hand.

**Caching decision.** Building 4,857 `folium.Circle` objects was previously measured at
~3.4 s and ~4.9 MB of HTML per rerun, which is why the old map was cached. The
FeatureGroups are therefore also cached with `@st.cache_resource`, and Task 5's browser
test explicitly navigates away from Overview and back to prove the cached objects still
render correctly on a second pass. If that test fails, the fallback is to drop the
`@st.cache_resource` decorator from `build_severity_feature_groups` only (correctness
beats latency) and re-run the gate.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_streamlit_app.py` (import `build_severity_base_map` and
`build_severity_feature_groups`):

```python
def test_build_severity_base_map_has_no_layers_attached():
    # Layers must reach the browser only via st_folium(feature_group_to_add=...).
    # Anything attached here raises a JS ReferenceError and blanks the whole map.
    import folium

    base_map = build_severity_base_map()
    children = base_map._children.values()
    assert not any(isinstance(child, folium.FeatureGroup) for child in children)
    assert not any(isinstance(child, folium.LayerControl) for child in children)


def test_build_severity_feature_groups_returns_one_named_group_per_band():
    groups = build_severity_feature_groups()
    assert len(groups) == len(RISK_BANDS)
    assert [group.layer_name for group in groups] == [band.label for band in RISK_BANDS]


def test_build_severity_feature_groups_covers_every_grid_cell_exactly_once():
    groups = build_severity_feature_groups()
    circles_per_group = [len(group._children) for group in groups]
    assert sum(circles_per_group) == 4857
    assert circles_per_group == [239, 1137, 2014, 1263, 204]


def test_build_severity_map_is_gone():
    # Replaced by build_severity_base_map + build_severity_feature_groups.
    import unfallatlas.viz.streamlit_app as module

    assert not hasattr(module, "build_severity_map")
```

Replace the body of `tests/test_overview_page.py` with:

```python
from streamlit.testing.v1 import AppTest


def test_overview_page_loads_and_renders_severity_map():
    at = AppTest.from_file("app/pages/overview.py", default_timeout=120)
    at.run()
    assert not at.exception


def test_overview_page_renders_the_relative_risk_legend():
    at = AppTest.from_file("app/pages/overview.py", default_timeout=120)
    at.run()
    assert not at.exception
    rendered = " ".join(element.value for element in at.markdown)
    # The legend must name every band and state the national baseline, so a reader
    # can tell what "2x" is relative to.
    assert "Well below average (<0.75x)" in rendered
    assert "Very high (>=2x)" in rendered
    assert "18.9%" in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_overview_page.py tests/test_streamlit_app.py -k "severity_base_map or feature_groups or build_severity_map_is_gone or overview_page" -v`

Expected: `ImportError: cannot import name 'build_severity_base_map'`.

- [ ] **Step 3: Replace `build_severity_map` in the viz module**

Delete the whole existing `build_severity_map` function (lines 364-409) and put these
two functions in its place:

```python
@st.cache_resource
def build_severity_base_map():
    """Return an empty, cached folium map centred on Germany.

    Deliberately carries NO layers. Every marker layer reaches the browser through
    `st_folium(feature_group_to_add=...)` instead. An earlier version attached
    FeatureGroups and a LayerControl here with `.add_to()`, which raised
    `ReferenceError: feature_group_<hash> is not defined` in the browser and blanked
    the entire map (AppTest saw no exception at all - only a real browser catches
    this). streamlit-folium re-injects the rendered map into its own `map_div`
    execution context and only rewrites layer references for objects passed through
    its own parameters, so layers baked in here do not resolve.

    folium.Map objects aren't meaningfully comparable by value, so this uses
    cache_resource (identity-cached singleton), not cache_data.
    """
    import folium

    return folium.Map(location=[51.1657, 10.4515], zoom_start=6)


@st.cache_resource
def build_severity_feature_groups(precision: float = 0.1):
    """Build one toggleable FeatureGroup per relative-risk band.

    Returned as a plain list for `st_folium(feature_group_to_add=...)`; the caller
    must not attach these to a map (see build_severity_base_map). folium's
    LayerControl over these groups doubles as the map's legend and lets a reader
    isolate a single band - for example showing only ">=2x average" cells to see
    where they cluster.

    Uses `folium.Circle` (radius in metres) rather than `CircleMarker` (radius in
    screen pixels) so a cell's marker keeps a constant size relative to the real
    geography at every zoom level. Cells are drawn at their accident centroid, not
    at the rounded grid-bin coordinate.

    Cached because building ~4857 Circle objects costs roughly 3.4 s and 4.9 MB of
    HTML, which is too slow to repeat on every rerun.
    """
    import folium

    grid_df = load_severity_grid(precision)
    baseline = load_national_ksi_rate()
    groups = [folium.FeatureGroup(name=band.label, show=True) for band in RISK_BANDS]

    for cell in grid_df.itertuples():
        relative_risk = shrunk_relative_risk(cell.ksi_count, cell.total, baseline)
        band_index = risk_band_index(relative_risk)
        band = RISK_BANDS[band_index]
        folium.Circle(
            location=[cell.center_lat, cell.center_lon],
            radius=min(5000, 300 + cell.total * 4),
            color=band.color,
            weight=1,
            fill=True,
            fill_color=band.color,
            fill_opacity=confidence_opacity(cell.total),
            popup=(
                f"{relative_risk:.2f}x national KSI rate ({band.label})<br>"
                f"KSI: {int(cell.ksi_count)}, slight: {int(cell.slight_count)}, "
                f"total: {int(cell.total)}"
            ),
        ).add_to(groups[band_index])

    return groups
```

- [ ] **Step 4: Rewire the Overview page**

In `app/pages/overview.py`, change the import block to pull the new names:

```python
from unfallatlas.viz.streamlit_app import (
    LIMITATIONS_TEXT,
    build_severity_base_map,
    build_severity_feature_groups,
    load_3class_comparison,
    load_binary_comparison,
    load_model_card,
    severity_legend_markdown,
)
```

Then replace lines 41-52 (the `st.markdown("---")` through the `st_folium(...)` call)
with:

```python
st.markdown("---")
st.subheader("Where severe accidents concentrate")
st.caption(
    "Each circle aggregates the accidents inside a ~0.1 degree (~11 km) cell, drawn "
    "at the mean position of those accidents. Color shows how the cell's share of "
    "KSI (killed/seriously injured) accidents compares against the national average, "
    "not how many accidents it has - use the layer control to isolate a single risk "
    "band. Note the inversion this reveals: the lowest-risk cells carry a median of "
    "1,339 accidents each while the highest-risk cells carry 91. Dense urban areas "
    "produce many mostly-slight collisions; rural roads produce far fewer that are "
    "far more often severe."
)
st_folium(
    build_severity_base_map(),
    feature_group_to_add=build_severity_feature_groups(),
    layer_control=True,
    height=720,
    width=None,
    key="overview_severity_map",
    returned_objects=[],
)
st.markdown(severity_legend_markdown(), unsafe_allow_html=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_overview_page.py tests/test_streamlit_app.py -v`

Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src/unfallatlas/viz/streamlit_app.py app/pages/overview.py tests/
uv run ruff check src/unfallatlas/viz/streamlit_app.py app/pages/overview.py tests/
git add src/unfallatlas/viz/streamlit_app.py app/pages/overview.py tests/test_overview_page.py tests/test_streamlit_app.py
git commit -m "feat(overview): band the severity map by relative KSI risk with a layer legend"
```

---

## Task 4: Risk Predictor picker responds to the first click

**Files:**
- Modify: `app/pages/risk_predictor.py:143-218`
- Modify: `src/unfallatlas/viz/streamlit_app.py` (add `build_picker_base_map`)
- Test: `tests/test_risk_predictor_page.py`

**Interfaces:**
- Consumes: `nearest_location_features(lat: float, lon: float) -> dict` (existing),
  `_clamp(value, lo, hi)` (existing, local to the page)
- Produces: `build_picker_base_map() -> folium.Map` - `@st.cache_resource`, empty map

**Root cause to fix.** `app/pages/risk_predictor.py:151` builds `picker_map` and its
`folium.Marker` from `st.session_state["picked_lat"]`/`["picked_lon"]`, but line 165 is
where the click handler *updates* those keys. The handler runs after the map for that
rerun is already built, so click #1 updates state while rendering the marker at the old
position, and only the rerun caused by click #2 shows click #1. This is a script-ordering
bug, not a rendering delay.

The `nearest_location_features` DuckDB lookup measures 0.26 s, so it is not the cause of
the perceived freeze - the rerun round-trip is - but it is unlabeled dead air and gets a
spinner.

- [ ] **Step 1: Write the failing test**

Replace `tests/test_risk_predictor_page.py` with:

```python
from streamlit.testing.v1 import AppTest


def test_risk_predictor_page_loads():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=180)
    at.run()
    assert not at.exception


def test_risk_predictor_page_does_not_excuse_the_reload_pause():
    # The old copy told users the click lag was expected behaviour. It was a
    # script-ordering bug, now fixed, so the apology must be gone.
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=180)
    at.run()
    assert not at.exception
    captions = " ".join(element.value for element in at.caption)
    assert "not a freeze" not in captions


def test_risk_predictor_form_submits_a_prediction():
    at = AppTest.from_file("app/pages/risk_predictor.py", default_timeout=180)
    at.run()
    at.button[-1].click().run()
    assert not at.exception
```

Note on `at.button[-1]`: the page renders two "load example" buttons before the form, so
the submit button is last. If `at.button` ordering proves unreliable, select it by label
with `next(b for b in at.button if "Predict" in b.label)`.

- [ ] **Step 2: Run the tests to verify the caption test fails**

Run: `uv run pytest tests/test_risk_predictor_page.py -v`

Expected: `test_risk_predictor_page_does_not_excuse_the_reload_pause` FAILS
(the "not a freeze" caption is still present); the other two PASS.

- [ ] **Step 3: Add the cached picker base map to the viz module**

Insert after `build_severity_feature_groups` in
`src/unfallatlas/viz/streamlit_app.py`:

```python
@st.cache_resource
def build_picker_base_map():
    """Return an empty, cached folium map for the Risk Predictor location picker.

    Carries no marker: the selected-point marker is passed per rerun through
    `st_folium(feature_group_to_add=...)`. Keeping the marker out of the map object
    is what makes the picker respond to the first click - a marker baked into the
    map is fixed at map-construction time, which happens before the click handler
    runs, so it always lagged one rerun behind.
    """
    import folium

    return folium.Map(location=[51.1657, 10.4515], zoom_start=6)
```

- [ ] **Step 4: Rewire the picker on the Risk Predictor page**

In `app/pages/risk_predictor.py`, add `build_picker_base_map` to the
`from unfallatlas.viz.streamlit_app import (...)` block (alphabetically, before
`build_input_row`).

Replace lines 143-218 (from `st.subheader("Pick a location")` through the
`st.caption(f"Selected: ...")` call) with:

```python
st.subheader("Pick a location")
st.caption(
    "Click a point on the map to set the accident's longitude/latitude and auto-fill "
    "the administrative codes and road context below from the nearest recorded "
    "accident. Weather stays under your control - it depends on when an accident "
    "happens, not where. Clicks outside Germany's covered area are ignored with a "
    "warning."
)

selected_marker_group = folium.FeatureGroup(name="Selected location")
folium.Marker(
    [st.session_state["picked_lat"], st.session_state["picked_lon"]],
    tooltip="Selected location",
).add_to(selected_marker_group)

map_state = st_folium(
    build_picker_base_map(),
    feature_group_to_add=selected_marker_group,
    height=500,
    width=None,
    key="location_picker",
    returned_objects=["last_clicked"],
)

if map_state and map_state.get("last_clicked"):
    clicked = (map_state["last_clicked"]["lat"], map_state["last_clicked"]["lng"])
    # Only act the first time a given click is seen - st_folium keeps replaying the
    # same last_clicked value on every later rerun (e.g. form submission), which
    # would otherwise re-show a stale out-of-bounds warning forever, and would make
    # the st.rerun() below loop.
    if clicked != st.session_state["last_processed_click"]:
        st.session_state["last_processed_click"] = clicked
        clicked_lat, clicked_lon = clicked
        if (
            lat_spec["min"] <= clicked_lat <= lat_spec["max"]
            and lon_spec["min"] <= clicked_lon <= lon_spec["max"]
        ):
            st.session_state["picked_lat"] = clicked_lat
            st.session_state["picked_lon"] = clicked_lon
            with st.spinner("Reading road context for this location..."):
                features = nearest_location_features(clicked_lat, clicked_lon)
            if features["UREGBEZ"] in uregbez_categories:
                st.session_state["picked_uregbez"] = features["UREGBEZ"]
            if features["UKREIS"] in ukreis_options:
                st.session_state["picked_ukreis"] = features["UKREIS"]
            if features["osm_dominant_road_class"] in road_class_categories:
                st.session_state["risk_osm_road_class"] = features["osm_dominant_road_class"]
            st.session_state["risk_dwd_station_id"] = features["dwd_station_id"]
            st.session_state["risk_h3_cell"] = features["h3_cell"]
            # Only autofill fields that are genuinely properties of the location
            # itself (road geometry, nearest weather station). The weather readings
            # (temp/precip/visibility/wind) are properties of *when* an accident
            # happens, not *where* - a given spot can see any weather depending on
            # the day, so pulling them from whichever historical accident happens to
            # be nearest would be misleading rather than helpful. Those sliders stay
            # untouched. The nearest record can have NaN in some optional columns
            # (not every OSM way carries a maxspeed tag) - skip those fields rather
            # than crashing _clamp() on a None, leaving the previous value in place.
            for key, spec_name, spec in (
                ("risk_dwd_station_dist_km", "dwd_station_dist_km", station_dist_spec),
                ("risk_osm_maxspeed_mean", "osm_maxspeed_mean", maxspeed_mean_spec),
                ("risk_osm_maxspeed_max", "osm_maxspeed_max", maxspeed_max_spec),
                ("risk_osm_road_density", "osm_road_density", density_spec),
                ("risk_osm_way_count", "osm_way_count", way_count_spec),
            ):
                raw_value = features[spec_name]
                if raw_value is not None:
                    st.session_state[key] = _clamp(raw_value, spec["min"], spec["max"])
            # Rerun so the marker, the coordinate readout, and every auto-filled
            # widget reflect THIS click. Without it the page renders one click
            # behind, which is what made the picker seem to need two clicks. The
            # last_processed_click guard above makes this terminate: on the rerun,
            # `clicked` equals the stored value and this branch is skipped.
            st.rerun()
        else:
            st.warning(
                f"Clicked point ({clicked_lat:.4f}, {clicked_lon:.4f}) is outside the "
                f"covered range (lat {lat_spec['min']:.2f}-{lat_spec['max']:.2f}, "
                f"lon {lon_spec['min']:.2f}-{lon_spec['max']:.2f}) and was ignored."
            )

st.caption(
    f"Selected: lat {st.session_state['picked_lat']:.4f}, lon {st.session_state['picked_lon']:.4f}"
)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_risk_predictor_page.py -v`

Expected: all 3 PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format app/pages/risk_predictor.py src/unfallatlas/viz/streamlit_app.py tests/test_risk_predictor_page.py
uv run ruff check app/pages/risk_predictor.py src/unfallatlas/viz/streamlit_app.py tests/test_risk_predictor_page.py
git add app/pages/risk_predictor.py src/unfallatlas/viz/streamlit_app.py tests/test_risk_predictor_page.py
git commit -m "fix(risk-predictor): place the picked map point on the first click"
```

---

## Task 5: Headless-browser gate for both maps

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/presentation/test_browser.py:44-73` (remove the now-shared fixture)
- Create: `tests/test_streamlit_browser.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at the Python level; drives the running app
- Produces: `chromium_browser` pytest fixture, session-scoped, shared across
  `tests/` and `tests/presentation/`

**Why this task is mandatory, not optional.** `AppTest` runs the Streamlit script but
never the frontend. When the previous FeatureGroup/LayerControl attempt blanked the
Overview map with a JS `ReferenceError`, `AppTest` reported zero exceptions. This plan
deliberately reintroduces `feature_group_to_add` and `layer_control` - the exact API
involved - so a real browser check is the only test that can prove the map works.

- [ ] **Step 1: Move the chromium fixture into a shared conftest**

Create `tests/conftest.py`:

```python
from collections.abc import Iterator
from pathlib import Path
from shutil import which
from typing import Any

import pytest


@pytest.fixture(scope="session")
def chromium_browser() -> Iterator[Any]:
    playwright = pytest.importorskip(
        "playwright.sync_api",
        reason="install the presentation-test extra to run opt-in browser checks",
    )
    with playwright.sync_playwright() as runtime:
        launch_options: dict[str, object] = {"headless": True, "args": ["--no-sandbox"]}
        if not Path(runtime.chromium.executable_path).is_file():
            system_chromium = next(
                (
                    executable
                    for name in ("chromium", "chromium-browser", "google-chrome", "chrome")
                    if (executable := which(name)) is not None
                ),
                None,
            )
            if system_chromium is None:
                pytest.fail(
                    "No Playwright-managed or system Chromium executable is available. "
                    "Run `uv run playwright install chromium`."
                )
            launch_options["executable_path"] = system_chromium
        browser = runtime.chromium.launch(**launch_options)
        yield browser
        browser.close()
```

Then delete the `chromium_browser` fixture from `tests/presentation/test_browser.py`
(the `@pytest.fixture(scope="session")` decorator plus the `def chromium_browser()`
body, lines 44-73 inclusive - check the exact decorator line above `def chromium_browser`
before cutting). Remove any imports that become unused as a result (`which`, `Iterator`,
possibly `Path`) - run `uv run ruff check tests/presentation/test_browser.py` to find
out exactly which, since `Path` is used elsewhere in that file.

- [ ] **Step 2: Verify the existing browser tests still pass with the shared fixture**

Run: `uv run pytest tests/presentation/test_browser.py -m browser -q`

Expected: the same 9 tests pass as before the move. If they error on fixture lookup, the
fixture was not found - confirm `tests/conftest.py` exists and that no stale
`chromium_browser` definition remains in `test_browser.py`.

- [ ] **Step 3: Write the browser gate test**

Create `tests/test_streamlit_browser.py`:

```python
"""Headless-browser checks for the live Streamlit app.

AppTest executes the Streamlit script but never the frontend, and it provably did
NOT catch the previous blank-map failure: it reported zero exceptions while a JS
`ReferenceError: feature_group_<hash> is not defined` had blanked the Overview map
completely. These tests are the only thing that can catch that class of bug, so they
gate any change to how folium layers reach the browser.
"""

import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import pytest

APP_ENTRY = "app/streamlit_app.py"
STARTUP_TIMEOUT_S = 180


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def streamlit_server() -> Iterator[str]:
    """Run the real app on a random port and yield its base URL."""
    port = _free_port()
    process = subprocess.Popen(  # noqa: S603
        [
            "uv",
            "run",
            "streamlit",
            "run",
            APP_ENTRY,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read().decode() if process.stdout else ""
                pytest.fail(f"Streamlit exited before serving:\n{output}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail(f"Streamlit did not start within {STARTUP_TIMEOUT_S}s")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def _open_app(browser: Any, url: str) -> tuple[Any, Any, list[str]]:
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(url, wait_until="load")
    return context, page, page_errors


def _wait_for_leaflet_paths(page: Any, minimum: int) -> int:
    """Wait until at least `minimum` Leaflet vector paths have rendered."""
    page.wait_for_function(
        "min => {"
        "  const frames = Array.from(document.querySelectorAll('iframe'));"
        "  return frames.some(frame => {"
        "    const doc = frame.contentDocument;"
        "    return doc && doc.querySelectorAll('path.leaflet-interactive').length >= min;"
        "  });"
        "}",
        arg=minimum,
        timeout=120_000,
    )
    return int(
        page.evaluate(
            "() => Math.max(0, ...Array.from(document.querySelectorAll('iframe'))"
            ".map(frame => frame.contentDocument"
            " ? frame.contentDocument.querySelectorAll('path.leaflet-interactive').length"
            " : 0))"
        )
    )


def _open_page_by_name(page: Any, name: str) -> None:
    page.get_by_role("link", name=name).click()
    page.wait_for_timeout(2_000)


@pytest.mark.browser
def test_overview_severity_map_renders_without_js_errors(
    chromium_browser: Any, streamlit_server: str
) -> None:
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        # 4857 cells are drawn; require a large fraction to have rendered so a
        # partially-broken layer set cannot pass.
        rendered = _wait_for_leaflet_paths(page, 4_000)
        assert rendered >= 4_000
        assert page_errors == [], f"JS errors on Overview: {page_errors}"
    finally:
        context.close()


@pytest.mark.browser
def test_overview_layer_control_lists_every_risk_band(
    chromium_browser: Any, streamlit_server: str
) -> None:
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        _wait_for_leaflet_paths(page, 4_000)
        labels = page.evaluate(
            "() => Array.from(document.querySelectorAll('iframe'))"
            ".flatMap(frame => frame.contentDocument"
            " ? Array.from(frame.contentDocument.querySelectorAll("
            "'.leaflet-control-layers-overlays label')).map(node => node.textContent.trim())"
            " : [])"
        )
        joined = " | ".join(labels)
        for band_label in (
            "Well below average (<0.75x)",
            "Around average (0.75-1.1x)",
            "Elevated (1.1-1.5x)",
            "High (1.5-2x)",
            "Very high (>=2x)",
        ):
            assert band_label in joined, f"missing layer toggle {band_label!r} in {joined!r}"
        assert page_errors == []
    finally:
        context.close()


@pytest.mark.browser
def test_overview_map_survives_navigating_away_and_back(
    chromium_browser: Any, streamlit_server: str
) -> None:
    # The FeatureGroups are cached with st.cache_resource. If streamlit-folium
    # mutates them while rendering, the second render would break - this is the test
    # that catches it. On failure, drop @st.cache_resource from
    # build_severity_feature_groups and re-run.
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        _wait_for_leaflet_paths(page, 4_000)
        _open_page_by_name(page, "Model Comparison")
        _open_page_by_name(page, "Overview")
        rendered = _wait_for_leaflet_paths(page, 4_000)
        assert rendered >= 4_000
        assert page_errors == [], f"JS errors after re-navigation: {page_errors}"
    finally:
        context.close()


@pytest.mark.browser
def test_risk_predictor_places_the_marker_on_a_single_click(
    chromium_browser: Any, streamlit_server: str
) -> None:
    context, page, page_errors = _open_app(chromium_browser, streamlit_server)
    try:
        _open_page_by_name(page, "Risk Predictor")
        map_frame = page.frame_locator("iframe").first
        container = map_frame.locator(".leaflet-container")
        container.wait_for(timeout=120_000)

        before = page.get_by_text("Selected: lat").inner_text()
        box = container.bounding_box()
        # Click left-of-centre and above centre: still inside Germany's bbox at
        # zoom 6 centred on [51.1657, 10.4515], but clearly away from the default
        # marker so the readout must change.
        container.click(position={"x": box["width"] * 0.45, "y": box["height"] * 0.4})
        page.wait_for_timeout(8_000)
        after = page.get_by_text("Selected: lat").inner_text()

        # ONE click must move the point. Before the fix this required two.
        assert after != before, f"coordinate readout unchanged after one click: {before!r}"
        assert page_errors == [], f"JS errors on Risk Predictor: {page_errors}"
    finally:
        context.close()
```

- [ ] **Step 4: Run the browser gate**

Run: `uv run pytest tests/test_streamlit_browser.py -m browser -v`

Expected: all 4 PASS. Notes for triage:
- If `test_overview_map_survives_navigating_away_and_back` is the only failure, remove
  `@st.cache_resource` from `build_severity_feature_groups` in
  `src/unfallatlas/viz/streamlit_app.py`, then re-run the whole file.
- If the single-click test fails with the readout unchanged, the `st.rerun()` from Task 4
  is missing or unreachable.
- If a click lands outside Germany's bbox an out-of-bounds warning appears instead;
  adjust the click position fractions and re-run.

- [ ] **Step 5: Confirm the browser tests stay out of the default run**

Run: `uv run pytest --collect-only -q | tail -3`

Expected: the summary reports the new browser tests as deselected (`addopts` carries
`-m "not browser"`), so the default suite is unchanged for CI.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format tests/conftest.py tests/test_streamlit_browser.py tests/presentation/test_browser.py
uv run ruff check tests/conftest.py tests/test_streamlit_browser.py tests/presentation/test_browser.py
git add tests/conftest.py tests/test_streamlit_browser.py tests/presentation/test_browser.py
git commit -m "test: gate the Streamlit maps behind headless-browser JS-error checks"
```

---

## Task 6: Full-suite verification and provenance documentation

**Files:**
- Modify: `docs/AI TOOL DISCLOSURE.md:41-58` (phase table) and `:67-83` (plan index)
- Modify: `docs/prompts/05_prompts_phase_k.md` (append a section)

**Interfaces:**
- Consumes: the completed Tasks 1-5 and their commit history
- Produces: no code interfaces

- [ ] **Step 1: Run the whole default suite**

Run: `uv run pytest`

Expected: at least the pre-existing 399 tests pass, plus the tests added by Tasks 1-4;
the browser tests are deselected. Any failure here must be fixed before continuing -
do not document work that does not pass.

- [ ] **Step 2: Lint the whole repository**

```bash
uv run ruff format --check .
uv run ruff check .
```

Expected: both clean.

- [ ] **Step 3: Add the disclosure row**

In `docs/AI TOOL DISCLOSURE.md`, append one row to the phase table, immediately after
the existing final Phase K row (the one ending
`plans/2026-07-27-phase-k-streamlit-enhancements.md)`):

```markdown
| **Phase K** | Claude Opus 5, effort: medium; used July 2026 | Phase K map-correctness follow-up: replaced the Overview severity map's absolute 50%-majority coloring (which left 97.5% of cells one color) with shrinkage-corrected relative-risk bands against the measured 18.9% national KSI rate, restored the per-band layer legend through `st_folium`'s own `feature_group_to_add`/`layer_control` parameters after diagnosing why the earlier `.add_to()` approach blanked the map, moved cell markers onto their accident centroids, and fixed the Risk Predictor picker's script-ordering bug that made a point need two clicks; added a headless-browser JS-error gate because `AppTest` cannot see frontend failures | [Prompt record](prompts/05_prompts_phase_k.md); [Design spec](superpowers/specs/2026-07-28-overview-risk-map-and-picker-ux-design.md); [Implementation plan](superpowers/plans/2026-07-28-overview-risk-map-and-picker-ux.md) |
```

- [ ] **Step 4: Add the plan-index row**

Append to the plan-index table in the same file, after the
`2026-07-27-phase-k-streamlit-enhancements.md` row:

```markdown
| 2026-07-28 | Overview relative-risk map and picker UX fixes | Opus 5, medium | [2026-07-28-overview-risk-map-and-picker-ux.md](superpowers/plans/2026-07-28-overview-risk-map-and-picker-ux.md) |
```

- [ ] **Step 5: Add the prompt record**

Append to `docs/prompts/05_prompts_phase_k.md`, following the existing section
structure (`##` heading, then the `**Tool:**`/`**Model release:**`/`**Used:**`/
`**Effort:**`/`**Disclosure:**` metadata block with `<br>` line breaks, then
`### Recorded prompt`, then outcome sections):

```markdown
## Map-correctness follow-up: relative-risk banding and first-click picker

**Tool:** Claude Code (Opus 5)<br>
**Model release:** 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)<br>
**Design spec:** [2026-07-28-overview-risk-map-and-picker-ux-design.md](../superpowers/specs/2026-07-28-overview-risk-map-and-picker-ux-design.md)<br>
**Implementation plan:** [2026-07-28-overview-risk-map-and-picker-ux.md](../superpowers/plans/2026-07-28-overview-risk-map-and-picker-ux.md)

### Recorded prompt

The user reviewed the shipped Overview map and questioned its statistical basis
directly, rather than reporting it as a visual complaint:

> the thing is for one the legend is gone again where i can enable/disable
> sligh/ksi cells. additionally im not sure the current coloring is sufficient
> because it colors it red if KSI >= slight if im correct and thats it and the
> circles are drawn systematically and increase size for count but are placed not
> really at the location of these accidents correct? Why not make the cells red if
> KSI > as the expected percentage eg expected percentage 10% KSI if in that cell
> 15% KSI red or redder do you know what i mean or is this not an ideal approach?

And on the Risk Predictor:

> eg the map in risk predictor is shit i need to click twice for the point to add
> after some short freeze that is bad ui/ux for example

`superpowers:brainstorming` was used to validate each claim against the real data
before designing anything, then `superpowers:writing-plans` produced the linked
6-task plan, executed via `superpowers:subagent-driven-development`.

### All three user observations were correct, and measurement confirmed each

The user's proposed fix - compare a cell against the *expected* KSI percentage
rather than against 50% - is the statistically sound one, and the measurements show
why the original was worse than it looked:

- The national KSI rate is **0.18914** (395,766 of 2,092,401 accidents). The
  `ksi_share >= 0.5` rule therefore only fired at **2.64x** the national rate.
- Just **123 of 4,857** grid cells (2.5%) passed it, so 97.5% of the map rendered in
  a single color and a cell at 1.9x the national rate was drawn identically to the
  safest cell in Germany. This was a loss of real signal, not a cosmetic issue.
- Marker placement used the rounded `ROUND(LAT, 1)` bin coordinate rather than the
  accidents' own centroid, a measured mean offset of **1.71 km** (p95 3.80 km, max
  6.48 km).

### Shrinkage constant chosen by measurement, not taste

Relative risk alone would let a cell with one KSI accident render as 5.29x the
national rate. Each cell's rate is therefore shrunk toward the national baseline
with a pseudo-count `k`, and `k` was selected by measuring band populations over the
real 4,857-cell grid: `k=10` still admitted 6-accident cells into the top band,
`k=50` collapsed that band to 52 cells and re-flattened the map, and `k=20` filled
all five bands (239 / 1,137 / 2,014 / 1,263 / 204), with the largest band holding
41.5% of cells. Sample size is additionally encoded as fill opacity in three discrete tiers,
so shrinkage biasing small cells toward the average stays visible rather than hidden.

### Why the legend had disappeared, and the correct way to restore it

The missing legend was the visible remnant of a real bug, not an oversight. An
earlier iteration attached `folium.FeatureGroup` layers and a `folium.LayerControl`
to the `@st.cache_resource`-cached `folium.Map` with `.add_to()`, which raised
`ReferenceError: feature_group_<hash> is not defined` in the browser and blanked the
entire map; the layers were removed wholesale to restore a working map.
`streamlit-folium` re-injects the rendered map into its own `map_div` execution
context and only rewrites layer references for objects passed through its own
`st_folium(feature_group_to_add=..., layer_control=...)` parameters. The toggles are
therefore restored through those parameters, with the base map kept completely
layer-free.

### A finding the new map surfaces that the old one hid

Median accidents per cell is **1,339** in the lowest-risk band and **91** in the
highest. The cells with the most accidents are the least severe: dense urban areas
produce many mostly-slight collisions while rural roads produce far fewer that are
far more often KSI. The 50%-threshold map could not show this at all.

### The picker bug was script ordering, not latency

`app/pages/risk_predictor.py` built the map and its marker from session state at
line 151, but the click handler that *updates* that state ran at line 165 - after.
Click #1 therefore rendered the marker at the previous position, and only the rerun
caused by click #2 reflected click #1. The fix moves the marker out of the map object
into a per-rerun `FeatureGroup` and adds an explicit `st.rerun()` after a
state-changing click. The accompanying pause was measured rather than assumed: the
`nearest_location_features` DuckDB lookup takes 0.26 s and is not the cause (the
rerun round-trip is), but it was unlabeled dead air and now shows a spinner. The
page caption that had described the lag as expected behaviour was deleted, since it
documented a bug rather than a constraint.

### Verification

`streamlit.testing.v1.AppTest` provably cannot catch this class of defect: it
reported zero exceptions while the earlier blank-map JS error was live. Because this
work deliberately reintroduces the same `feature_group_to_add`/`layer_control` API,
a headless-Playwright gate was added (`tests/test_streamlit_browser.py`, opt-in via
`-m browser`) that asserts zero `pageerror` events, that all five band toggles appear
in the layer control, that the map survives navigating away and back (proving the
cached FeatureGroups re-render safely), and that one click on the picker moves the
selected point - a direct regression test for the two-click bug.
```

- [ ] **Step 6: Verify the documentation tests still pass and commit**

```bash
uv run pytest tests/presentation/test_documentation.py -v
uv run ruff format --check .
git add "docs/AI TOOL DISCLOSURE.md" docs/prompts/05_prompts_phase_k.md
git commit -m "docs: record the relative-risk map and picker UX work"
```

Note: `tests/presentation/test_documentation.py::test_updated_ai_provenance_markdown_avoids_sentence_dash_punctuation`
asserts that `docs/AI TOOL DISCLOSURE.md` (and `docs/prompts/04_prompts_phase_c.md`, not
this record) avoids spaced-dash sentence punctuation (` - `, en dash, em dash) in prose.
The disclosure row added in Steps 3-4 must therefore use hyphenated compounds,
parentheses, or semicolons instead of a spaced dash. If the test flags a line, rewrite
that sentence rather than disabling the check.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| P1 relative-risk banding, `k=20`, 5 bands | 1 |
| Opacity confidence tiers | 1 |
| Static legend with colors and baseline | 1, 3 |
| P2 legend/toggle via `feature_group_to_add`/`layer_control` | 3 |
| Caching boundary (base map cached, no `.add_to`) | 3 |
| P3 centroid placement | 2 |
| National baseline computed from data | 2 |
| P4 first-click marker, cached base map, `st.rerun()`, spinner, caption deletion | 4 |
| D5 module boundary (pure functions in viz module) | 1, 2, 3, 4 |
| Unit tests for bands/shrinkage/opacity | 1 |
| Data-layer tests incl. centroid-in-cell invariant | 2 |
| `AppTest` page tests | 3, 4 |
| Mandatory browser gate incl. single-click regression | 5 |
| Disclosure + prompt record | 6 |

Two spec items are deliberately not implemented, with the rationale recorded in the
"deliberate deviations" section above: computed RGB color interpolation (replaced by a
hand-picked ramp, because interpolating teal to red passes through muddy brown) and
`st_folium`'s `center=` parameter on the picker (omitted to avoid snapping the user's
view on every click). Both are documented in the plan rather than silently dropped.

**Placeholder scan:** no TBD/TODO markers; every code step carries complete code; every
test step carries real assertions with concrete expected values.

**Type consistency:** `RiskBand` fields (`label`, `lower`, `upper`, `color`) are used
consistently in Tasks 1 and 3. `load_severity_grid`'s new columns
(`center_lat`, `center_lon`) are produced in Task 2 and consumed by the same names in
Task 3. `build_severity_base_map` / `build_severity_feature_groups` /
`build_picker_base_map` are named identically at definition (Tasks 3, 4) and at every
call site (`app/pages/overview.py`, `app/pages/risk_predictor.py`, tests). The band
labels asserted in Task 5's browser test match the `RISK_BANDS` labels defined in
Task 1 exactly.
