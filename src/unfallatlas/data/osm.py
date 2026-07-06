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
