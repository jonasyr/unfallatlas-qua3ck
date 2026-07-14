# U-Phase OSM/H3 Road-Context Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenStreetMap-derived road-context features (dominant road class, speed-limit statistics, road density, way count) to the U-phase preprocessing contract, aggregated per H3-8 hexagonal cell and joined onto every accident by location — filling the `src/unfallatlas/features/spatial.py` and `src/unfallatlas/data/osm.py` stub files that were scaffolded in the original project setup but never implemented, and giving A³ a genuinely new predictive signal (*where* an accident happened, not just *when* and under what named administrative region) that the current feature set entirely lacks.

**Why this is a U-phase task, not an A3 task:** this project has kept phase boundaries strict throughout — DWD weather enrichment was added as a U-phase addendum (`unfallatlas.data.dwd`, §8.5–§9.4 of `02_U_Phase.ipynb`), then A³ consumed it without re-deciding anything. Adding a new engineered feature source is the same shape of decision (U decides *what* features exist and how they're encoded/imputed; A³ implements it inside a `Pipeline`). This plan does **not** touch `notebooks/03_A3_Phase.ipynb` or `src/unfallatlas/models/`; consuming the new columns into A3's `build_preprocessor` and re-running the model comparison is a separate, following plan (which will also fold in three Minor findings from the A3 champion-pivot's final review — see "Deferred to the next plan" below).

**Architecture:** Two new library modules, mirroring `unfallatlas.data.dwd`'s existing structure exactly:
- `src/unfallatlas/features/spatial.py` — pure, network-free geometry/aggregation functions (H3 cell assignment, `maxspeed` tag parsing, road-class ranking, per-cell aggregation of a road GeoDataFrame). Fully unit-testable with synthetic data, no live calls.
- `src/unfallatlas/data/osm.py` — network I/O: fetches Germany's road network from OpenStreetMap via `osmnx` in 16 per-Bundesland queries (avoids one Overpass query timing out on the whole country), each cached to `data/raw/osm/<state>.parquet`; a top-level `build_spatial_features()` orchestrator (mirrors `unfallatlas.data.dwd.build_weather_features` signature and caching pattern exactly) joins the H3 aggregates onto the accident frame and caches the combined result to `data/interim/accidents_with_weather_spatial.parquet`.

`notebooks/02_U_Phase.ipynb` gets new cells: run the enrichment, visualize the new features, run a leakage/temporal-consistency probe (mirroring the existing §9.4 conditional-entropy probe), and extend the §10 preprocessing decision table with the new columns — exactly the same pattern already used for the DWD weather columns.

**Tech Stack:** `osmnx>=1.9` (2.1.0 installed) for the OSM/Overpass fetch, `h3>=4` (4.4.2 installed) for cell indexing, `geopandas` (1.1.3, an `osmnx` transitive dependency, already installed) for the road geometry. All three verified importable and working in this environment before this plan was written — no new dependencies needed in `pyproject.toml`.

## Global Constraints

- **U decides, A³ implements** (`AGENTS.md`, established throughout this project): this plan only adds a *feature source* and documents its preprocessing decision (missing strategy, encoding, scaling) in `02_U_Phase.ipynb`'s §10 table — it does not wire the new columns into `build_preprocessor()` or re-run any model. That is explicitly the next plan's job.
- **No relitigating the existing §10 table.** All existing rows (cyclic time encodings, one-hot categoricals, target-mean-encoded region codes, DWD weather) stay exactly as they are — this plan only *appends* new rows and new notebook sections, mirroring the DWD weather addendum's own structure (`## 10 — Preprocessing decisions`, `### DWD weather features` subsection).
- **Compute budget (Q-phase §9, single workstation):** the OSM fetch must not attempt one Overpass query for all of Germany (their own risk log flags this as high-probability-to-fail) — chunk by the 16 Bundesländer, cache each state's result to disk immediately (mirrors `unfallatlas.data.dwd.download_station_data`'s per-station-per-variable caching exactly), so an interrupted run resumes instead of restarting.
- **H3 resolution 8** (~0.7 km² hexagons) for cell aggregation — chosen over resolution 9 specifically to avoid sparse, noisy per-cell aggregates in rural areas that would risk teaching downstream models on noise rather than signal (a stated goal of the follow-up work this plan supports is avoiding both over- and under-fitting).
- **Known limitation to document, not solve:** OSM reflects the *present-day* road network, while accidents span 2016–2024; road speed limits and classifications can change over that window. This is the same category of accepted approximation the DWD weather join already uses (day-of-month averaging, documented as "a known approximation" in `dwd.py`'s docstring) — document it in the U-phase notebook's §10 table and §11 risk list, do not attempt to source historical OSM snapshots (out of scope, no free historical Overpass API exists for this).
- **Caching convention:** `data/raw/` is git-ignored (local-only, matches the existing `data/raw/` `.gitignore` pattern) — per-state raw OSM caches go there. `data/interim/*.parquet` is committed via Git LFS (`*.parquet` pattern already in `.gitattributes`, matching the existing `accidents_with_weather.parquet`) — the new combined `accidents_with_weather_spatial.parquet` follows the same convention, no `.gitattributes` change needed.
- **Testing convention:** every function in `src/unfallatlas/` gets a focused pytest test. Network-bound functions (the actual Overpass/osmnx fetch) are not directly unit-tested, matching the existing precedent (`dwd.py`'s `download_station_list`/`download_station_data` are not directly tested either) — but every function's *pure logic* (parsing, cleaning, aggregation, cell assignment) must be split out so it CAN be tested with synthetic data, no network required.
- **Notebook policy (`AGENTS.md`):** `notebooks/*.ipynb` is source of truth; `.py` is a Jupytext percent-format mirror regenerated via `jupytext --sync`, never hand-edited. Use `NotebookEdit` on the `.ipynb`.
- **Code conventions:** ruff + black, line length 100, Python ≥3.11, no `print()` in `src/unfallatlas/` (use `logging`), all paths via `pathlib.Path`.

## Deferred to the next plan (do not do these here — listed so they aren't lost)

Three Minor findings from the A3 champion-pivot plan's final whole-branch review, explicitly out of scope for this U-phase-only plan (they're all inside `notebooks/03_A3_Phase.ipynb`, which this plan does not touch) — the next plan (updating A3 to consume these new spatial features) must address all three:
1. `notebooks/03_A3_Phase.py`'s §6 `strategy_rows = comparison_df[comparison_df["model"].str.startswith(family)]` filter incidentally also matches `{family}_default` rows, which have no refit path in `_build_pipeline_for` (latent, currently unreachable given the data, but should be tightened to an explicit allow-list of the 4 strategy names + the family's `candidate_names` entry).
2. Dead `cv_groups`/a throwaway `GroupKFold(n_splits=7)` computed and printed in notebook §2 but never consumed (§7 builds its own separately) — wire it through or comment that it's illustrative only.
3. The full Stufe-0/1/§6 model-comparison table currently only survives as summary fields in `a3_model_card.json` (notebook cell outputs are stripped by `nbstripout`, `reports/a3_progress.log` is git-ignored) — persist the full `comparison_df` to a committed CSV artifact.

---

## Task 1: `spatial.py` — H3 cell assignment and `maxspeed` parsing

**Files:**
- Modify: `src/unfallatlas/features/spatial.py` (currently an empty stub)
- Test: `tests/test_spatial.py` (new file)

**Interfaces:**
- Produces: `assign_h3_cell(lat: float, lon: float, resolution: int = 8) -> str`, `parse_maxspeed(value) -> float | None`, `ROAD_CLASS_RANK: dict[str, int]`, `dominant_road_class(classes: list[str]) -> str` — all consumed by Task 3's aggregation function and Task 2's cleaning function.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spatial.py`:

```python
import math

from unfallatlas.features.spatial import (
    ROAD_CLASS_RANK,
    assign_h3_cell,
    dominant_road_class,
    parse_maxspeed,
)


def test_assign_h3_cell_returns_stable_string_id():
    # Berlin coordinates - same point must always map to the same cell.
    cell_a = assign_h3_cell(52.5200, 13.4050)
    cell_b = assign_h3_cell(52.5200, 13.4050)
    assert cell_a == cell_b
    assert isinstance(cell_a, str)
    assert len(cell_a) > 0


def test_assign_h3_cell_different_points_different_cells():
    berlin = assign_h3_cell(52.5200, 13.4050)
    munich = assign_h3_cell(48.1351, 11.5820)
    assert berlin != munich


def test_assign_h3_cell_resolution_changes_cell_id():
    res8 = assign_h3_cell(52.5200, 13.4050, resolution=8)
    res9 = assign_h3_cell(52.5200, 13.4050, resolution=9)
    assert res8 != res9


def test_parse_maxspeed_numeric_string():
    assert parse_maxspeed("50") == 50.0


def test_parse_maxspeed_numeric_with_mph_suffix_converts_to_kmh():
    result = parse_maxspeed("30 mph")
    assert math.isclose(result, 48.28, rel_tol=0.01)


def test_parse_maxspeed_de_urban_zone_code():
    assert parse_maxspeed("DE:urban") == 50.0


def test_parse_maxspeed_de_rural_zone_code():
    assert parse_maxspeed("DE:rural") == 100.0


def test_parse_maxspeed_unparseable_returns_none():
    assert parse_maxspeed("signals") is None
    assert parse_maxspeed("DE:motorway") is None  # no fixed limit, not a number
    assert parse_maxspeed(None) is None
    assert parse_maxspeed(float("nan")) is None


def test_parse_maxspeed_handles_semicolon_separated_list_by_taking_first():
    # OSM occasionally has "50;30" for conditional limits - take the first value.
    assert parse_maxspeed("50;30") == 50.0


def test_road_class_rank_orders_motorway_above_residential():
    assert ROAD_CLASS_RANK["motorway"] > ROAD_CLASS_RANK["residential"]


def test_dominant_road_class_picks_highest_ranked():
    assert dominant_road_class(["residential", "primary", "track"]) == "primary"


def test_dominant_road_class_empty_list_returns_none():
    assert dominant_road_class([]) is None


def test_dominant_road_class_ignores_unknown_values():
    # A highway value not in ROAD_CLASS_RANK (e.g. "footway", already filtered
    # out upstream, but defensive here) must not crash - just be ignored.
    assert dominant_road_class(["not_a_real_highway_value", "secondary"]) == "secondary"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_spatial.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (the functions don't exist yet).

- [ ] **Step 3: Implement `spatial.py`**

Write `src/unfallatlas/features/spatial.py`:

```python
"""Pure H3 cell-assignment and OSM road-attribute parsing helpers.

No network I/O here - see unfallatlas.data.osm for the OSM/Overpass fetch.
Kept separate so every function here is unit-testable with synthetic data.
"""

from __future__ import annotations

import re

import h3

# Ranked by road importance/typical speed - higher rank wins when multiple
# road classes pass through the same H3 cell (see dominant_road_class).
# *_link variants rank one below their parent class.
ROAD_CLASS_RANK: dict[str, int] = {
    "motorway": 14,
    "motorway_link": 13,
    "trunk": 12,
    "trunk_link": 11,
    "primary": 10,
    "primary_link": 9,
    "secondary": 8,
    "secondary_link": 7,
    "tertiary": 6,
    "tertiary_link": 5,
    "unclassified": 4,
    "residential": 3,
    "living_street": 2,
    "service": 1,
    "track": 0,
}

# OSM "DE:<zone>" implicit speed-limit codes that resolve to a fixed number.
# DE:motorway has no fixed limit (recommended 130 km/h, not enforced) and is
# deliberately excluded here - parse_maxspeed returns None for it, since
# including a non-binding "recommendation" as if it were a real limit would
# misrepresent the data.
_DE_ZONE_SPEEDS: dict[str, float] = {
    "DE:urban": 50.0,
    "DE:rural": 100.0,
    "DE:living_street": 7.0,
    "DE:zone20": 20.0,
    "DE:zone30": 30.0,
}

_NUMERIC_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mph)?\s*$", re.IGNORECASE)


def assign_h3_cell(lat: float, lon: float, resolution: int = 8) -> str:
    """H3 cell index (as a string) containing (lat, lon) at the given resolution."""
    return h3.latlng_to_cell(lat, lon, resolution)


def parse_maxspeed(value) -> float | None:
    """Parse an OSM `maxspeed` tag value into km/h, or None if unparseable.

    Handles: plain numeric strings ("50"), mph-suffixed strings ("30 mph"),
    known DE: zone codes (DE:urban, DE:rural, ...), and semicolon-separated
    conditional lists (takes the first value). Returns None for anything
    else (e.g. "signals", "none", "DE:motorway" which has no fixed limit,
    missing/NaN values) - callers must handle None as a missing value, not
    coerce it to 0.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN check without numpy import
        return None
    text = str(value).strip()
    if not text:
        return None

    if ";" in text:
        text = text.split(";")[0].strip()

    if text in _DE_ZONE_SPEEDS:
        return _DE_ZONE_SPEEDS[text]

    match = _NUMERIC_RE.match(text)
    if not match:
        return None
    speed = float(match.group(1))
    if match.group(2):  # "mph" suffix
        speed *= 1.60934
    return speed


def dominant_road_class(classes: list[str]) -> str | None:
    """Highest-ranked road class among `classes` (per ROAD_CLASS_RANK).

    Unknown values (not in ROAD_CLASS_RANK) are ignored rather than raising,
    since upstream filtering may not be perfectly exhaustive. Returns None
    for an empty or all-unknown input.
    """
    known = [c for c in classes if c in ROAD_CLASS_RANK]
    if not known:
        return None
    return max(known, key=lambda c: ROAD_CLASS_RANK[c])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_spatial.py -v`
Expected: all tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/unfallatlas/features/spatial.py tests/test_spatial.py
git add src/unfallatlas/features/spatial.py tests/test_spatial.py
git commit -m "feat: add H3 cell assignment and OSM maxspeed parsing helpers"
```

---

## Task 2: `osm.py` — fetch and clean Germany's road network per Bundesland

**Files:**
- Modify: `src/unfallatlas/data/osm.py` (currently an empty stub)
- Test: `tests/test_osm.py` (new file)

**Interfaces:**
- Consumes: nothing new (this is the first thing built in `osm.py`).
- Produces: `GERMAN_STATES: list[str]`, `_clean_road_gdf(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame` (pure, testable), `download_road_network(state: str, cache_dir: Path, force_refresh: bool = False) -> gpd.GeoDataFrame` (network I/O, cached) — both consumed by Task 4's `build_spatial_features`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_osm.py`:

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from unfallatlas.data.osm import GERMAN_STATES, _clean_road_gdf


def _toy_raw_gdf():
    """Mimics osmnx.features.features_from_place(tags={"highway": True}) output:
    a GeoDataFrame with a "highway" column (sometimes a list, per OSM's own
    multi-value tag convention) and a "maxspeed" column, plus irrelevant
    pedestrian-only ways that must be filtered out."""
    return gpd.GeoDataFrame(
        {
            "highway": ["primary", "residential", "footway", ["secondary", "primary"], "cycleway"],
            "maxspeed": ["100", "30", None, "50", None],
            "geometry": [
                LineString([(13.0, 52.0), (13.01, 52.01)]),
                LineString([(13.1, 52.1), (13.11, 52.11)]),
                LineString([(13.2, 52.2), (13.21, 52.21)]),
                LineString([(13.3, 52.3), (13.31, 52.31)]),
                LineString([(13.4, 52.4), (13.41, 52.41)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_german_states_has_sixteen_bundeslaender():
    assert len(GERMAN_STATES) == 16
    assert "Hessen" in GERMAN_STATES
    assert "Bayern" in GERMAN_STATES


def test_clean_road_gdf_drops_pedestrian_only_ways():
    cleaned = _clean_road_gdf(_toy_raw_gdf())
    assert "footway" not in cleaned["highway"].tolist()
    assert "cycleway" not in cleaned["highway"].tolist()
    assert len(cleaned) == 3  # primary, residential, and the list-valued row


def test_clean_road_gdf_takes_first_value_from_list_valued_highway_tags():
    cleaned = _clean_road_gdf(_toy_raw_gdf())
    # The list ["secondary", "primary"] row must resolve to a single string.
    list_row = cleaned[cleaned["maxspeed"] == "50"]
    assert len(list_row) == 1
    assert list_row.iloc[0]["highway"] == "secondary"


def test_clean_road_gdf_preserves_geometry_column():
    cleaned = _clean_road_gdf(_toy_raw_gdf())
    assert isinstance(cleaned, gpd.GeoDataFrame)
    assert "geometry" in cleaned.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_osm.py -v`
Expected: FAIL with `ImportError` (module has no `GERMAN_STATES`/`_clean_road_gdf` yet).

- [ ] **Step 3: Implement `osm.py`**

Write `src/unfallatlas/data/osm.py`:

```python
"""OSM road-network fetch, cached per Bundesland (German federal state).

Mirrors unfallatlas.data.dwd's per-item caching pattern: each state is
fetched once via osmnx (Overpass under the hood) and cached to disk, so an
interrupted run resumes instead of restarting - fetching all of Germany in
one Overpass query is prone to timing out (matches the risk already
flagged in docs/project/PROJEKTPLAN_SETUP.md's risk log).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import osmnx as ox

log = logging.getLogger(__name__)

GERMAN_STATES: list[str] = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]

# Pedestrian/non-vehicle infrastructure - excluded because this project models
# vehicle-involved accident severity; a footway's speed/class carries no
# signal for that. Kept as an explicit allow-list (not a deny-list) so a new,
# unanticipated OSM highway value defaults to being dropped, not silently
# included.
_VEHICLE_HIGHWAY_VALUES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "track",
}


def _clean_road_gdf(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to vehicle-relevant highway types and normalise list-valued tags.

    OSM allows a way to carry multiple values for one tag (rendered by
    osmnx as a Python list, e.g. highway=["secondary", "primary"] for a
    reclassified road segment) - this takes the first value, a standard,
    documented OSM-processing simplification.
    """
    gdf = raw_gdf.copy()
    gdf["highway"] = gdf["highway"].apply(lambda v: v[0] if isinstance(v, list) else v)
    return gdf[gdf["highway"].isin(_VEHICLE_HIGHWAY_VALUES)].reset_index(drop=True)


def download_road_network(
    state: str, cache_dir: Path, force_refresh: bool = False
) -> gpd.GeoDataFrame:
    """Fetch and cache OSM road ways (as a GeoDataFrame) for one German state.

    Returns a GeoDataFrame with columns [highway, maxspeed, geometry],
    filtered to vehicle-relevant road types via _clean_road_gdf. Uses
    osmnx.features.features_from_place rather than osmnx.graph_from_place -
    this only needs road attributes for aggregation, not a routable graph,
    which avoids osmnx's (expensive) graph-consolidation step entirely.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_slug = state.lower().replace(" ", "_").replace("ü", "ue").replace("ä", "ae")
    cache_path = cache_dir / f"{state_slug}.parquet"

    if cache_path.exists() and not force_refresh:
        log.info("Using cached OSM road network for %s at %s", state, cache_path)
        return gpd.read_parquet(cache_path)

    log.info("Fetching OSM road network for %s (this can take a few minutes)...", state)
    raw = ox.features.features_from_place(f"{state}, Germany", tags={"highway": True})
    raw = raw[raw.geom_type.isin(["LineString", "MultiLineString"])]
    cleaned = _clean_road_gdf(raw[["highway", "maxspeed", "geometry"]])

    cleaned.to_parquet(cache_path)
    log.info("Cached %s road network → %s (%d ways)", state, cache_path, len(cleaned))
    return cleaned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_osm.py -v`
Expected: all tests pass (these tests only exercise `_clean_road_gdf`, which has no network dependency; `download_road_network` itself is not directly unit-tested, matching the existing `dwd.py` precedent for network-bound functions).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/unfallatlas/data/osm.py tests/test_osm.py
git add src/unfallatlas/data/osm.py tests/test_osm.py
git commit -m "feat: fetch and clean OSM road network per Bundesland, cached"
```

---

## Task 3: `spatial.py` — aggregate road geometry to H3 cells

**Files:**
- Modify: `src/unfallatlas/features/spatial.py`
- Test: `tests/test_spatial.py`

**Interfaces:**
- Consumes: `assign_h3_cell`, `parse_maxspeed`, `dominant_road_class` (Task 1); a `gpd.GeoDataFrame` with `[highway, maxspeed, geometry]` columns (Task 2's `download_road_network` output shape).
- Produces: `aggregate_roads_to_h3(roads_gdf: gpd.GeoDataFrame, resolution: int = 8) -> pd.DataFrame` with columns `[h3_cell, osm_dominant_road_class, osm_maxspeed_mean, osm_maxspeed_max, osm_road_density, osm_way_count]` — consumed by Task 4's `build_spatial_features`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_spatial.py`:

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from unfallatlas.features.spatial import aggregate_roads_to_h3


def _toy_roads_gdf():
    # Two ways ~20-30m apart (verified empirically to land in the same
    # H3-8 cell, which is ~0.7 km² wide) - deliberately close together so
    # both ways aggregate into one cell, not split across a boundary.
    return gpd.GeoDataFrame(
        {
            "highway": ["primary", "residential"],
            "maxspeed": ["100", "30"],
            "geometry": [
                LineString([(13.4050, 52.5200), (13.4051, 52.5201)]),
                LineString([(13.4050, 52.5200), (13.4052, 52.5202)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_aggregate_roads_to_h3_returns_expected_columns():
    result = aggregate_roads_to_h3(_toy_roads_gdf(), resolution=8)
    assert set(result.columns) == {
        "h3_cell",
        "osm_dominant_road_class",
        "osm_maxspeed_mean",
        "osm_maxspeed_max",
        "osm_road_density",
        "osm_way_count",
    }


def test_aggregate_roads_to_h3_dominant_class_is_highest_ranked_in_cell():
    result = aggregate_roads_to_h3(_toy_roads_gdf(), resolution=8)
    # Both ways are in the same tiny area -> same cell; primary outranks residential.
    row = result[result["osm_way_count"] >= 1].iloc[0]
    assert row["osm_dominant_road_class"] == "primary"


def test_aggregate_roads_to_h3_maxspeed_stats_correct():
    result = aggregate_roads_to_h3(_toy_roads_gdf(), resolution=8)
    row = result.iloc[0]
    assert row["osm_maxspeed_max"] == 100.0
    assert row["osm_maxspeed_mean"] == 65.0  # (100 + 30) / 2


def test_aggregate_roads_to_h3_way_count_counts_distinct_ways_not_vertices():
    result = aggregate_roads_to_h3(_toy_roads_gdf(), resolution=8)
    # 2 ways, each a 2-point LineString (4 vertices total) - way_count must be 2, not 4.
    total_way_count = result["osm_way_count"].sum()
    assert total_way_count == 2


def test_aggregate_roads_to_h3_empty_input_returns_empty_frame_with_correct_columns():
    empty = gpd.GeoDataFrame({"highway": [], "maxspeed": [], "geometry": []}, crs="EPSG:4326")
    result = aggregate_roads_to_h3(empty, resolution=8)
    assert len(result) == 0
    assert set(result.columns) == {
        "h3_cell",
        "osm_dominant_road_class",
        "osm_maxspeed_mean",
        "osm_maxspeed_max",
        "osm_road_density",
        "osm_way_count",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_spatial.py -v -k aggregate_roads`
Expected: FAIL with `ImportError` (`aggregate_roads_to_h3` doesn't exist yet).

- [ ] **Step 3: Implement `aggregate_roads_to_h3`**

Append to `src/unfallatlas/features/spatial.py`:

```python
import geopandas as gpd
import pandas as pd


def aggregate_roads_to_h3(roads_gdf: gpd.GeoDataFrame, resolution: int = 8) -> pd.DataFrame:
    """Roll up a road GeoDataFrame (columns: highway, maxspeed, geometry) to
    one row per H3 cell at the given resolution.

    Method: every vertex of every way's LineString is assigned to its H3
    cell (not just endpoints or midpoints - long ways span multiple cells).
    Per cell:
        osm_dominant_road_class  highest-ranked highway value present (str)
        osm_maxspeed_mean/_max   parsed maxspeed stats in km/h, NaN if none
                                  parseable in the cell (see parse_maxspeed)
        osm_road_density         count of road-vertex points in the cell -
                                  a proxy for road presence/length, not a
                                  precise km/km² figure
        osm_way_count            count of DISTINCT ways touching the cell -
                                  a junction/complexity proxy (cells with
                                  more distinct roads passing through are
                                  more likely to be intersections), chosen
                                  over true topological intersection
                                  detection to avoid the compute cost of
                                  building a full routable graph

    Cells with no roads at all are simply absent from the output - callers
    must treat a missing h3_cell as "no OSM road data available", not zero.
    """
    columns = [
        "h3_cell",
        "osm_dominant_road_class",
        "osm_maxspeed_mean",
        "osm_maxspeed_max",
        "osm_road_density",
        "osm_way_count",
    ]
    if len(roads_gdf) == 0:
        return pd.DataFrame(columns=columns)

    records = []
    for way_id, row in roads_gdf.reset_index(drop=True).iterrows():
        speed = parse_maxspeed(row["maxspeed"])
        coords = list(row["geometry"].coords)
        for lon, lat in coords:
            records.append(
                {
                    "way_id": way_id,
                    "h3_cell": assign_h3_cell(lat, lon, resolution),
                    "highway": row["highway"],
                    "maxspeed": speed,
                }
            )
    points = pd.DataFrame.from_records(records)

    grouped = points.groupby("h3_cell")
    result = grouped.agg(
        osm_dominant_road_class=("highway", lambda s: dominant_road_class(list(s))),
        osm_maxspeed_mean=("maxspeed", "mean"),
        osm_maxspeed_max=("maxspeed", "max"),
        osm_road_density=("way_id", "count"),
        osm_way_count=("way_id", "nunique"),
    ).reset_index()
    return result[columns]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_spatial.py -v`
Expected: all tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/unfallatlas/features/spatial.py tests/test_spatial.py
git add src/unfallatlas/features/spatial.py tests/test_spatial.py
git commit -m "feat: aggregate OSM road network to per-H3-cell features"
```

---

## Task 4: `build_spatial_features` orchestrator + `load_training_frame` update

**Files:**
- Modify: `src/unfallatlas/data/osm.py`
- Modify: `src/unfallatlas/features/preprocessing.py:169-179` (the `load_training_frame` function)
- Test: `tests/test_osm.py`

**Interfaces:**
- Consumes: `GERMAN_STATES`, `download_road_network` (Task 2); `aggregate_roads_to_h3`, `assign_h3_cell` (Task 3).
- Produces: `build_spatial_features(accidents_df: pd.DataFrame, raw_cache_dir: Path, interim_cache_dir: Path, resolution: int = 8, force_refresh: bool = False) -> pd.DataFrame` — consumed by Task 5's U-phase notebook cells and by `load_training_frame` (this task).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_osm.py`:

```python
import pandas as pd

from unfallatlas.data.osm import build_spatial_features


def test_build_spatial_features_requires_lat_lon_columns(tmp_path):
    bad_df = pd.DataFrame({"UJAHR": [2020]})
    try:
        build_spatial_features(bad_df, tmp_path / "raw", tmp_path / "interim")
        raise AssertionError("expected RuntimeError for missing LAT/LON")
    except RuntimeError as exc:
        assert "LAT" in str(exc) or "LON" in str(exc)


def test_build_spatial_features_uses_cache_when_present(tmp_path, monkeypatch):
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    cached = pd.DataFrame({"LAT": [52.5], "LON": [13.4], "osm_way_count": [1]})
    cached.to_parquet(interim_dir / "accidents_with_weather_spatial.parquet")

    def _boom(*args, **kwargs):
        raise AssertionError("should not fetch OSM data when a cache exists")

    monkeypatch.setattr("unfallatlas.data.osm.download_road_network", _boom)

    accidents = pd.DataFrame({"LAT": [52.5], "LON": [13.4]})
    result = build_spatial_features(accidents, tmp_path / "raw", interim_dir)
    assert "osm_way_count" in result.columns
    assert result.iloc[0]["osm_way_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_osm.py -v -k build_spatial_features`
Expected: FAIL with `ImportError` (`build_spatial_features` doesn't exist yet).

- [ ] **Step 3: Implement `build_spatial_features`**

Append to `src/unfallatlas/data/osm.py` (add `from unfallatlas.features.spatial import aggregate_roads_to_h3, assign_h3_cell` and `import pandas as pd` to the imports):

```python
import pandas as pd

from unfallatlas.features.spatial import aggregate_roads_to_h3, assign_h3_cell


def build_spatial_features(
    accidents_df: pd.DataFrame,
    raw_cache_dir: Path,
    interim_cache_dir: Path,
    resolution: int = 8,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Join OSM road-context features to the accident DataFrame.

    Adds columns to accidents_df:
        h3_cell                  the accident's H3 cell (str)
        osm_dominant_road_class  highest-ranked road class in the cell (str, NaN if no OSM road data)
        osm_maxspeed_mean        mean parsed speed limit in the cell, km/h (float, NaN if none parseable)
        osm_maxspeed_max         max parsed speed limit in the cell, km/h (float, NaN if none parseable)
        osm_road_density         road-vertex-point count in the cell (float, NaN if no roads)
        osm_way_count            distinct OSM ways touching the cell (float, NaN if no roads)

    Known limitation: OSM reflects the present-day road network; accidents
    span 2016-2024 and road classifications/speed limits can change over
    that window. Documented, not solved - same category of accepted
    approximation as the DWD weather join's day-of-month averaging.

    The enriched DataFrame is cached to
    interim_cache_dir/accidents_with_weather_spatial.parquet.
    """
    required = {"LAT", "LON"}
    missing_cols = required - set(accidents_df.columns)
    if missing_cols:
        raise RuntimeError(
            f"accidents_df is missing required columns: {missing_cols}\n"
            "Ensure you are passing a DataFrame loaded from data/accidents.parquet "
            "(optionally already weather-enriched)."
        )

    raw_cache_dir = Path(raw_cache_dir)
    interim_cache_dir = Path(interim_cache_dir)
    interim_cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = interim_cache_dir / "accidents_with_weather_spatial.parquet"
    if out_path.exists() and not force_refresh:
        log.info("Using cached spatially-enriched DataFrame at %s", out_path)
        return pd.read_parquet(out_path)

    # --- Fetch + aggregate every state's road network ---
    all_cell_aggregates = []
    for state in GERMAN_STATES:
        roads = download_road_network(state, raw_cache_dir / "osm", force_refresh=force_refresh)
        all_cell_aggregates.append(aggregate_roads_to_h3(roads, resolution=resolution))
    cell_features = pd.concat(all_cell_aggregates, ignore_index=True)
    # A cell straddling a state boundary query could appear in two states'
    # extracts - keep the higher-way-count (more complete) version.
    cell_features = (
        cell_features.sort_values("osm_way_count", ascending=False)
        .drop_duplicates(subset="h3_cell", keep="first")
        .reset_index(drop=True)
    )

    # --- Join by H3 cell ---
    df = accidents_df.copy()
    df["h3_cell"] = [
        assign_h3_cell(lat, lon, resolution) for lat, lon in zip(df["LAT"], df["LON"])
    ]
    df = df.merge(cell_features, on="h3_cell", how="left")

    df.to_parquet(out_path, index=False)
    log.info("Saved spatially-enriched DataFrame → %s (%d rows)", out_path, len(df))
    return df
```

- [ ] **Step 4: Update `load_training_frame` in `preprocessing.py`**

Using Serena's `find_symbol` to read the current body of `load_training_frame` (`src/unfallatlas/features/preprocessing.py`), then `replace_symbol_body` to change it to:

```python
def load_training_frame(base_dir: Path) -> pd.DataFrame:
    """Load the DWD-and-OSM-enriched accidents frame built by the U-phase.

    Reuses the cache from unfallatlas.data.dwd.build_weather_features and
    unfallatlas.data.osm.build_spatial_features (A³ does not rebuild this
    cache — raises if it is missing).
    """
    cache = base_dir / "data" / "interim" / "accidents_with_weather_spatial.parquet"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found. Run notebooks/02_U_Phase.ipynb §8.5 (weather) "
            "and §8.x (OSM spatial features) first."
        )
    df = pd.read_parquet(cache)
    return df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_osm.py tests/test_preprocessing.py -v`
Expected: all tests pass. (`test_preprocessing.py`'s existing tests construct their own toy DataFrames directly, not via `load_training_frame`'s file path, so they are unaffected by this change — confirm this by reading `tests/test_preprocessing.py` before running, and if any test DOES depend on the old cache filename, update it to match.)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/unfallatlas/data/osm.py src/unfallatlas/features/preprocessing.py tests/test_osm.py
git add src/unfallatlas/data/osm.py src/unfallatlas/features/preprocessing.py tests/test_osm.py
git commit -m "feat: join OSM spatial features onto the accident frame, update cache loader"
```

---

## Task 5: U-phase notebook — run the enrichment, visualize, leakage probe

**Files:**
- Modify: `notebooks/02_U_Phase.ipynb`

**Interfaces:**
- Consumes: `build_spatial_features`, `GERMAN_STATES` (`unfallatlas.data.osm`, Task 4); the notebook's existing `df_weather` variable (the weather-enriched frame from §8.5, already loaded earlier in this notebook) and its existing `entropy`/`conditional_entropy` helper functions (already used by the §9.4 DWD leakage probe — reuse them, do not redefine).

This task only ADDS new cells after the existing §8.7 weather-analysis section and before §9 (or after §9.4's leakage probe, whichever the current notebook structure makes cleaner — inspect the live cell boundaries with `get_symbols_overview`-equivalent notebook inspection before choosing exactly where).

- [ ] **Step 1: Add a new markdown + code cell running the enrichment**

Using `NotebookEdit` (insert after the last §8.7-equivalent cell), add a markdown cell:

```markdown
## 8.8 — OSM road-context enrichment

`src/unfallatlas/features/spatial.py` and `src/unfallatlas/data/osm.py` add
road-context features aggregated per H3-8 cell (~0.7 km² hexagons):
dominant road class, mean/max speed limit, road density, and a
junction/complexity proxy (distinct-way count). Fetched once from
OpenStreetMap per Bundesland (16 states), cached to `data/raw/osm/`, then
joined onto every accident by its H3 cell.

**Known limitation:** OSM reflects the present-day road network; accidents
span 2016–2024 and some roads' classification/speed limits will have
changed since. This is an accepted approximation (same category as the DWD
weather join's day-of-month averaging, §8.5) — not solvable without a paid
historical-OSM-snapshot service, out of scope here.
```

Then a code cell:

```python
from unfallatlas.data.osm import GERMAN_STATES, build_spatial_features

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"

print(f"Fetching OSM road networks for {len(GERMAN_STATES)} states (uses per-state cache if present)...")
df_spatial = build_spatial_features(df_weather, RAW_DIR, INTERIM_DIR, resolution=8)
print(f"Spatially-enriched frame: {len(df_spatial):,} rows, {df_spatial.shape[1]} columns")

osm_cols = ["osm_dominant_road_class", "osm_maxspeed_mean", "osm_maxspeed_max", "osm_road_density", "osm_way_count"]
coverage = df_spatial[osm_cols].notna().mean() * 100
print("\nOSM feature coverage (% of accidents with a matched H3 cell):")
print(coverage.round(1))
```

- [ ] **Step 2: Add visualization cells**

```python
fig = px.histogram(
    df_spatial, x="osm_dominant_road_class",
    category_orders={"osm_dominant_road_class": list(ROAD_CLASS_RANK.keys())},
    title="Dominant OSM road class by accident (H3-8 cell)",
)
save_fig(fig, "08_8_osm_road_class_distribution")
fig.show()

# %%
fig = px.histogram(
    df_spatial, x="osm_maxspeed_mean", nbins=40,
    title="Mean OSM speed limit (km/h) in the accident's H3 cell",
)
save_fig(fig, "08_8_osm_maxspeed_distribution")
fig.show()
```

(Add `from unfallatlas.features.spatial import ROAD_CLASS_RANK` to the notebook's imports cell if not already present — check with a quick read of the current import cell first.)

- [ ] **Step 3: Add a leakage/consistency probe mirroring §9.4**

```python
# %% [markdown]
# ### §9.5 — OSM feature consistency probe
#
# Mirrors §9.4's conditional-entropy method: does knowing the OSM road
# class trivially determine the target (which would suggest a data
# artefact, not a genuine relationship)? A large entropy reduction here
# would be suspicious - OSM data is independent of accident outcomes by
# construction (it describes the road, not the crash), so a strong result
# should read as a real severity signal, not a leak.

# %%
if "osm_dominant_road_class" in df_spatial.columns:
    baseline_entropy = entropy(df_spatial["UKATGEORIE"])
    cond_entropy = conditional_entropy(df_spatial["UKATGEORIE"], df_spatial["osm_dominant_road_class"])
    reduction_pct = 100 * (baseline_entropy - cond_entropy) / baseline_entropy
    print(f"Baseline entropy: {baseline_entropy:.4f}")
    print(f"Conditional entropy given osm_dominant_road_class: {cond_entropy:.4f}")
    print(f"Reduction: {reduction_pct:.1f}%")
    if reduction_pct > 50:
        print("\n  → WARNING: reduction exceeds the 50% trigger - investigate before including.")
    else:
        print("\n  → Below the 50% trigger - retain as a feature.")
else:
    print("OSM data not loaded — skipping §9.5 probe.")
```

- [ ] **Step 4: Sync jupytext and verify**

```bash
uv run jupytext --sync notebooks/02_U_Phase.ipynb
uv run ruff check notebooks/02_U_Phase.py
uv run python -c "import ast; ast.parse(open('notebooks/02_U_Phase.py').read())"
```
Expected: ruff passes (or only the pre-existing `black` formatting note already known from the A3 branch — not introduced by this task), no syntax errors. Do not execute the full notebook yet — that's Task 8, after Task 6's §10 table update and Task 7's docs are also in place, so the run produces final, fully-documented cell outputs in one pass.

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_U_Phase.ipynb notebooks/02_U_Phase.py
git commit -m "feat: run OSM spatial enrichment in U-phase, add visualizations + consistency probe"
```

---

## Task 6: U-phase §10 decision table + §11 risk-list update

**Files:**
- Modify: `notebooks/02_U_Phase.ipynb` (the `## 10 — Preprocessing decisions` markdown cell, currently `notebooks/02_U_Phase.py:1813-1863`, and the `## 11 — Summary` cell's "Top-4 risks for A³" list, currently around `:1895-1917`)

**Interfaces:**
- Consumes: nothing new — this is pure documentation, extending the existing §10/§11 markdown cells.

- [ ] **Step 1: Extend the §10 table with an OSM road-context subsection**

Using `NotebookEdit`, insert a new subsection immediately after the existing `### DWD weather features` table (before the `> **Note.**` paragraph that follows it, or after that paragraph and before `### Imbalance handling` — inspect the live cell to place it consistently with the DWD section's own internal structure):

```markdown
### OSM road-context features

| Feature | Missing strategy | Recommended transform | Recommended scaling | EDA finding that drives the decision |
|:---|:---|:---|:---|:---|
| `osm_dominant_road_class` | mode (or a dedicated "unknown" category) | one-hot | n/a | 15 nominal road classes ranked by literature-established severity/speed association; §9.5 entropy-reduction check must clear the 50% trigger before inclusion |
| `osm_maxspeed_mean` | median imputation | none (already a natural km/h scale) | `StandardScaler` | speed limit is one of the strongest literature-documented predictors of crash severity specifically (not just occurrence) |
| `osm_maxspeed_max` | median imputation | none | `StandardScaler` | captures the fastest road touching a mixed-road-class cell, complementing the mean |
| `osm_road_density` | zero-fill (absence of OSM data in a cell most often reflects genuinely low road density, e.g. remote areas, not a data gap) | `log1p` (right-skewed - most cells have few road-vertex points) | `StandardScaler` | proxy for local traffic exposure |
| `osm_way_count` | zero-fill (same rationale as `osm_road_density`) | `log1p` | `StandardScaler` | junction/complexity proxy — cells with multiple distinct roads are more likely to be intersections |

> **Note.** OSM road-context reflects the *present-day* network; some roads'
> classification or posted speed limit will have changed since the earliest
> accidents in this dataset (2016). This is an accepted approximation — see
> §8.8 — not a defect to fix here; A³ should note it as a limitation when
> interpreting SHAP importances for these features in Phase C.
```

- [ ] **Step 2: Add a 5th risk to the "Top-4 risks for A³" list (rename to "Top-5")**

Using `NotebookEdit`, change the heading `### Top-4 risks for A³` to `### Top-5 risks for A³` and append a new risk item after risk 4:

```markdown
5. **OSM road-context is a present-day snapshot, not historical.** Road
   classifications and speed limits reflect today's OpenStreetMap data,
   applied uniformly across all accident years (2016–2024). A road that was
   reclassified or had its speed limit changed during that window is
   silently treated as if its current state always applied. This mainly
   affects `osm_maxspeed_mean`/`osm_maxspeed_max`, less so `osm_dominant_road_class`
   (road hierarchy changes far less often than posted speed limits).
```

- [ ] **Step 3: Sync jupytext and verify**

```bash
uv run jupytext --sync notebooks/02_U_Phase.ipynb
uv run ruff check notebooks/02_U_Phase.py
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_U_Phase.ipynb notebooks/02_U_Phase.py
git commit -m "docs: extend U-phase §10 decision table and risk list for OSM features"
```

---

## Task 7: Documentation updates

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/GLOSSARY.md`
- Modify: `docs/AI TOOL DISCLOSURE.md`

**Interfaces:**
- Consumes: nothing — pure documentation, run after Task 6 so the notebook content being documented already exists.

- [ ] **Step 1: Update `AGENTS.md`'s architecture block**

In the `<!-- AUTO-MANAGED: architecture -->` block, the `data/` and `features/` listings currently read (after the A3 champion-pivot plan's own edits):
```
│   ├── data/               # download.py, dwd.py (weather), osm.py
│   ├── features/           # enrich.py, spatial.py, temporal.py, preprocessing.py
```
Change to:
```
│   ├── data/               # download.py, dwd.py (weather), osm.py (road network)
│   ├── features/           # enrich.py, spatial.py (H3/OSM aggregation), temporal.py, preprocessing.py
```

- [ ] **Step 2: Add new terms to `docs/GLOSSARY.md`**

Append to the "## Machine Learning Concepts" section, after the "Champion candidates" entry added by the A3 champion-pivot plan, before the `---` that precedes "## Process Model":

```markdown
**H3 (Hexagonal Hierarchical Spatial Index)**
Uber's hexagonal geospatial indexing system. Divides the earth's surface into hexagonal cells at multiple resolutions; resolution 8 cells (used in this project) are ~0.7 km² wide. Used to aggregate OSM road-network data to a manageable, uniform spatial grid before joining to each accident by location — an alternative to a per-point nearest-road lookup, chosen for compute-bounded joins across all of Germany.

**OSM Road-Context Features**
Road class, speed limit, density, and way-count statistics aggregated per H3 cell from OpenStreetMap, joined to each accident by location (U-phase §8.8). Reflects the present-day road network, applied uniformly across all accident years (2016–2024) — a documented, accepted approximation (see §11 risk 5), not a temporal-leakage vector (OSM data does not depend on accident outcomes).
```

- [ ] **Step 3: Add a new `AI TOOL DISCLOSURE.md` row**

Insert a new table row after the last existing Phase A³ row (matching the existing table format), before the closing `---`:

```markdown
| **Phase U** | Claude Code (Sonnet 5), effort: medium (Anthropic, 2026) | U-phase addendum built with `superpowers:brainstorming` and `superpowers:writing-plans`: OSM road-context feature engineering (`src/unfallatlas/data/osm.py`, `src/unfallatlas/features/spatial.py`) aggregated per H3-8 cell and joined to every accident, filling stub files scaffolded in the original project setup but never implemented; extends the §10 preprocessing decision table and §11 risk list following the same pattern as the existing DWD weather addendum | view [docs/prompts/02_prompts_phase_u.md](docs/prompts/02_prompts_phase_u.md) |
```

- [ ] **Step 4: Run the full test suite and lint**

```bash
uv run pytest -v
uv run ruff check .
```
Expected: all tests pass (including the new `test_spatial.py`/`test_osm.py`), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md docs/GLOSSARY.md "docs/AI TOOL DISCLOSURE.md"
git commit -m "docs: document the OSM/H3 road-context feature addendum"
```

---

## Task 8: Execute the U-phase notebook end-to-end and verify

**Files:**
- Execute: `notebooks/02_U_Phase.ipynb` (no code changes — verification only)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `data/raw/osm/<state>.parquet` (16 files, git-ignored), `data/interim/accidents_with_weather_spatial.parquet` (committed via Git LFS).

- [ ] **Step 1: Run the notebook as a detached, separate process**

This run fetches 16 states' road networks from OpenStreetMap (network-bound, Overpass rate limits apply) plus re-runs every earlier U-phase cell — expect this to take a while and to be the first live execution of the new OSM cells specifically. Launch it detached so it survives the terminal/IDE closing (same convention established in the A3 champion-pivot plan):

```bash
setsid uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_U_Phase.ipynb \
  > /tmp/u_phase_nbconvert.log 2>&1 < /dev/null &
disown
```

Confirm it's detached:
```bash
ps -ef | grep nbconvert | grep -v grep
```

Monitor via `tail -f /tmp/u_phase_nbconvert.log` (this notebook does not yet have the `_log_progress`/`reports/*.log` convention the A3 notebook uses — the nbconvert log itself is the only progress signal here; if this run proves slow/flaky enough to need resumable progress logging, that is a signal to add the same `_log_progress`/checkpoint pattern from `03_A3_Phase.ipynb` to this cell in a follow-up, not something to build speculatively now).

Expected: exits 0, no exception cells.

- [ ] **Step 2: Verify OSM coverage and the consistency probe**

Check the executed notebook's own printed coverage stats (from Task 5 Step 1's cell) — expect high coverage (>95%) for `osm_dominant_road_class`/`osm_way_count`/`osm_road_density` (Germany's road network is dense; the main coverage gap would be from any Overpass query that failed for a whole state, which the per-state cache/log will surface clearly), and expect the §9.5 consistency-probe entropy reduction to print without the "> 50%, investigate" warning (if it DOES print that warning, stop — do not proceed to commit — and treat it as a real finding to bring back for a decision, not something to suppress).

- [ ] **Step 3: Full test suite and lint**

```bash
uv run pytest -v
uv run ruff check .
uv run black --check .
```
Expected: all tests pass; ruff clean. (If `black --check .` fails only on `notebooks/02_U_Phase.py`'s own reformatting — not a new file — run `uv run black notebooks/02_U_Phase.py` and re-verify; this file IS part of this plan's diff, unlike the pre-existing, out-of-scope failure noted in the A3 champion-pivot plan's final review.)

- [ ] **Step 4: Sync jupytext and commit the executed notebook + data cache**

```bash
uv run jupytext --sync notebooks/02_U_Phase.ipynb
git add notebooks/02_U_Phase.ipynb notebooks/02_U_Phase.py data/interim/accidents_with_weather_spatial.parquet
git commit -m "feat: execute U-phase OSM enrichment end-to-end, save enriched cache"
```

If `accidents_with_weather_spatial.parquet` exceeds the pre-commit hook's large-file limit, it is already covered by the existing `*.parquet` Git LFS pattern in `.gitattributes` (no new `.gitattributes` entry needed — verify with `git check-attr filter data/interim/accidents_with_weather_spatial.parquet` before committing, expected output: `filter: lfs`).

---

## Self-Review Notes

- **Spec coverage:** brainstormed goal (spatial/OSM road-context features, H3-8 aggregation, U-phase-first structure) → Tasks 1-6; compute-budget/risk-log mitigation (chunk by Bundesland, not one national query) → Task 2; avoiding overfitting via coarser H3 resolution → Global Constraints + Task 3's design; documenting (not solving) the present-day-OSM-vs-historical-accidents limitation → Task 6; the 3 deferred Minor findings from the A3 review → explicitly listed in "Deferred to the next plan" so they are not lost, not silently dropped.
- **Placeholder scan:** no "TBD"/"handle edge cases" left; every code block is complete, runnable content matching the existing `dwd.py` patterns exactly (function signatures, caching behavior, docstring style).
- **Type consistency:** `download_road_network(state: str, cache_dir: Path, force_refresh: bool) -> gpd.GeoDataFrame` (Task 2) is called identically in `build_spatial_features` (Task 4). `aggregate_roads_to_h3(roads_gdf: gpd.GeoDataFrame, resolution: int) -> pd.DataFrame` (Task 3) is called identically in Task 4. `assign_h3_cell(lat, lon, resolution)` (Task 1) is used consistently in both Task 3 (per-vertex aggregation) and Task 4 (per-accident join) with the same 3-argument signature.
- **Scope check:** this plan is appropriately sized for a single implementation pass — 8 tasks, all within `src/unfallatlas/{data,features}/` + `notebooks/02_U_Phase.ipynb` + docs, no A3-phase or modeling work included (explicitly deferred, tracked above).
