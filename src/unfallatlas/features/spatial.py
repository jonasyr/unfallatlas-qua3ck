"""Pure H3 cell-assignment and OSM road-attribute parsing helpers.

No network I/O here - see unfallatlas.data.osm for the OSM/Overpass fetch.
Kept separate so every function here is unit-testable with synthetic data.
"""

from __future__ import annotations

import re

import geopandas as gpd
import h3
import pandas as pd

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

    Note the aggregation is vertex-weighted, not way-length-weighted - a way
    with more vertices contributes more to osm_maxspeed_mean/osm_road_density
    proportionally. This is an accepted characteristic of the point-based
    aggregation method, not a bug.

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
