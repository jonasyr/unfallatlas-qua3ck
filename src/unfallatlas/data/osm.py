"""OSM road-network fetch, cached per Bundesland (German federal state).

Mirrors unfallatlas.data.dwd's per-item caching pattern: each state is
fetched once via osmnx (Overpass under the hood) and cached to disk, so an
interrupted run resumes instead of restarting - fetching all of Germany in
one Overpass query is prone to timing out (matches the risk already
flagged in docs/project/PROJEKTPLAN_SETUP.md's risk log).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

from unfallatlas.features.spatial import aggregate_roads_to_h3, assign_h3_cell

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

    Progress: prints (not just logs) a "[i/16] fetching <state>..." line
    before each state and a "-> done in Xs (N ways)" line after, via
    print(..., flush=True) - this shows up in a live Jupyter cell's output
    immediately regardless of whether Python's logging module has a
    configured handler, unlike the log.info calls in download_road_network
    itself. Each state's per-state cache (see download_road_network) means
    a re-run after an interruption resumes instead of restarting.
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
        print(f"Using cached spatially-enriched DataFrame at {out_path}", flush=True)
        return pd.read_parquet(out_path)

    # --- Fetch + aggregate every state's road network ---
    all_cell_aggregates = []
    total = len(GERMAN_STATES)
    for i, state in enumerate(GERMAN_STATES, start=1):
        state_cache_path = (
            raw_cache_dir
            / "osm"
            / f"{state.lower().replace(' ', '_').replace('ü', 'ue').replace('ä', 'ae')}.parquet"
        )
        already_cached = state_cache_path.exists() and not force_refresh
        status = (
            "cached" if already_cached else "fetching from OSM (can take a while for large states)"
        )
        print(f"[{i}/{total}] {state}: {status}...", flush=True)
        start = time.time()
        roads = download_road_network(state, raw_cache_dir / "osm", force_refresh=force_refresh)
        elapsed = time.time() - start
        print(f"  -> {state} done in {elapsed:.1f}s ({len(roads):,} ways)", flush=True)
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
    df["h3_cell"] = [assign_h3_cell(lat, lon, resolution) for lat, lon in zip(df["LAT"], df["LON"])]
    df = df.merge(cell_features, on="h3_cell", how="left")

    df.to_parquet(out_path, index=False)
    log.info("Saved spatially-enriched DataFrame → %s (%d rows)", out_path, len(df))
    print(f"Saved spatially-enriched DataFrame -> {out_path} ({len(df):,} rows)", flush=True)
    return df
