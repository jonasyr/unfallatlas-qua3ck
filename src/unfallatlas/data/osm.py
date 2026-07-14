"""OSM road-network fetch, cached per Bundesland (German federal state).

Mirrors unfallatlas.data.dwd's per-item caching pattern: each state is
fetched once via osmnx (Overpass under the hood) and cached to disk, so an
interrupted run resumes instead of restarting - fetching all of Germany in
one Overpass query is prone to timing out (matches the risk already
flagged in docs/project/PROJEKTPLAN_SETUP.md's risk log).
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests

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


class _TransientFetchError(Exception):
    """Raised when a tile's fetch exhausts all retries on a network error.

    Deliberately distinct from returning None (which means "confirmed no
    roads in this tile, safe to cache as empty") - a transient failure must
    NOT be cached, so a later run retries it instead of treating a
    temporary outage as permanent, real-world absence of roads.
    """


# Adaptive ETA: a 0.2° tile is roughly constant-cost regardless of which
# state it belongs to, so seconds-per-tile is tracked as ONE rolling window
# across the whole run (not per-state, not a single static average over
# the entire history) - a fresh state immediately inherits a realistic
# estimate instead of starting from "no data yet", and the estimate adapts
# within minutes if the Overpass mirror gets slower/faster mid-run, rather
# than reporting one generic number for the whole multi-hour run.
_tile_seconds_history: deque[float] = deque(maxlen=50)


def _record_tile_seconds(seconds: float) -> None:
    _tile_seconds_history.append(seconds)


def _avg_tile_seconds() -> float | None:
    return statistics.mean(_tile_seconds_history) if _tile_seconds_history else None


def _format_duration(seconds: float) -> str:
    """Human-readable Hh Mm / Mm Ss / Ss, dropping zero leading units."""
    seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _eta_str(remaining_units: int) -> str:
    """ETA for `remaining_units` tiles at the current adaptive rate.

    Returns an explicit "warming up" placeholder rather than a number until
    at least one real tile fetch has completed this run - a fabricated ETA
    from zero data would be actively misleading, not just imprecise.
    """
    if remaining_units <= 0:
        return "done"
    avg = _avg_tile_seconds()
    if avg is None:
        return "ETA warming up..."
    return f"ETA {_format_duration(avg * remaining_units)} ({avg:.1f}s/tile avg)"


# Whole-run progress: set once per build_spatial_features call
# (_init_run_progress) and updated per-tile (_note_tile_done) so the
# per-tile log line - which fires constantly, unlike the once-per-state
# summary line - can show a live overall ETA too. download_road_network
# has no reason to know about other states' tile budgets, so this lives
# at module level rather than being threaded through as a parameter.
_run_progress: dict[str, float | int | None] = {"total_tiles": None, "tiles_done": 0, "start": None}


def _init_run_progress(total_tiles: int, tiles_already_done: int) -> None:
    _run_progress["total_tiles"] = total_tiles
    _run_progress["tiles_done"] = tiles_already_done
    _run_progress["start"] = time.time()


def _note_tile_done() -> None:
    if _run_progress["total_tiles"] is not None:
        _run_progress["tiles_done"] += 1


def _overall_progress_str() -> str:
    """Whole-run progress/ETA, or "" if _init_run_progress was never called
    (e.g. download_road_network exercised directly, outside
    build_spatial_features - has no run-wide tile budget to report)."""
    total = _run_progress["total_tiles"]
    if total is None:
        return ""
    done = _run_progress["tiles_done"]
    remaining = max(total - done, 0)
    elapsed = time.time() - _run_progress["start"]
    return (
        f"overall {done}/{total} tiles, elapsed {_format_duration(elapsed)}, {_eta_str(remaining)}"
    )


def _state_total_tiles(state: str) -> int:
    """Tile count for one state's bbox at the fixed 0.2° grid.

    A geocode-only call (no Overpass fetch) - cheap enough to run for all
    16 states upfront so build_spatial_features can size an overall-run
    ETA before a single tile has actually been fetched.
    """
    west, south, east, north = ox.geocode_to_gdf(f"{state}, Germany").total_bounds
    return len(_grid_tiles(west, south, east, north, tile_size_deg=0.2))


def _state_cached_tile_count(cache_dir: Path, state: str) -> int:
    """How many of a state's tiles already have a cache file on disk."""
    state_slug = state.lower().replace(" ", "_").replace("ü", "ue").replace("ä", "ae")
    tile_dir = Path(cache_dir) / f"{state_slug}_tiles"
    if not tile_dir.exists():
        return 0
    return sum(1 for _ in tile_dir.glob("tile_*.parquet"))


def _grid_tiles(
    west: float, south: float, east: float, north: float, tile_size_deg: float = 0.2
) -> list[tuple[float, float, float, float]]:
    """Split a (west, south, east, north) bbox into a grid of tiles, each at
    most tile_size_deg x tile_size_deg, row-major order. Trailing tiles in
    each row/column are clipped to the original bbox (never overshoot) -
    the union of all returned tiles reconstructs the input bbox exactly,
    with no gaps or overlaps.

    Uses index counts (not an accumulating `while lon < east: lon +=
    tile_size_deg` loop) to avoid float-drift producing a spurious sliver
    tile or infinite loop when the span isn't an exact multiple of
    tile_size_deg.
    """
    if east <= west or north <= south:
        raise ValueError(
            f"Invalid bbox: west={west}, south={south}, east={east}, north={north} "
            "(east must be > west, north must be > south)"
        )
    n_lon = math.ceil((east - west) / tile_size_deg)
    n_lat = math.ceil((north - south) / tile_size_deg)

    tiles = []
    for i in range(n_lat):
        tile_south = south + i * tile_size_deg
        tile_north = min(tile_south + tile_size_deg, north)
        for j in range(n_lon):
            tile_west = west + j * tile_size_deg
            tile_east = min(tile_west + tile_size_deg, east)
            tiles.append((tile_west, tile_south, tile_east, tile_north))
    return tiles


def _fetch_tile_edges(
    bbox: tuple[float, float, float, float], custom_filter: str, max_retries: int = 2
) -> gpd.GeoDataFrame | None:
    """Fetch one tile's road network as a raw [highway, maxspeed, geometry]
    GeoDataFrame, or None if the tile is CONFIRMED to have no matching roads
    (a tile over water/forest, or one that only grazes a state's actual
    boundary since the state bbox is a rectangle superset of its real,
    irregular polygon).

    truncate_by_edge=True is required, not optional: with the default
    truncate_by_edge=False, graph_from_bbox removes any node outside the
    bbox and every edge incident to it - including the portion of that
    edge that legitimately lies inside this tile. Because the far
    endpoint is a real, stable OSM junction, the neighbouring tile's
    independent fetch drops the same edge for the same reason, producing
    silent, grid-aligned gaps in the combined data. truncate_by_edge=True
    avoids the gap by letting both adjacent tiles retain the boundary
    node - at the cost of duplicating the boundary-crossing edge's
    geometry across both tiles, which download_road_network deduplicates
    afterward on (highway, maxspeed, geometry).

    Retries max_retries times on a transient network error
    (requests.exceptions.RequestException - confirmed empirically: osmnx's
    own internal rate-limit status check, not just the main Overpass
    query, can raise requests.exceptions.ReadTimeout/ConnectionError on an
    overloaded public mirror or a single dead backend in its DNS
    round-robin pool - overpass-api.de resolves to multiple IPs and one of
    them being down causes "Connection refused" on whichever retry happens
    to land on it). If every retry is exhausted, raises
    _TransientFetchError rather than returning None - a real live run
    (Bayern/Brandenburg/Hamburg) proved this distinction matters: an
    earlier version returned None uniformly for both "confirmed empty"
    and "gave up after retries", which download_road_network then cached
    identically as a permanent 0-row tile - silently and permanently
    erasing real road data for 70 tiles (~10% of Bayern and Brandenburg)
    for an outage that was actually transient (one dead backend in the
    Overpass round-robin; other tiles retried against a healthy backend
    and succeeded). A tile with no matching roads at all raises ValueError
    (osmnx's InsufficientResponseError, or a plain ValueError from
    truncate_graph_polygon) - NOT retried, since retrying can't produce
    data that doesn't exist, and legitimately safe to cache as empty.
    """
    for attempt in range(max_retries + 1):
        try:
            graph = ox.graph_from_bbox(
                bbox=bbox,
                custom_filter=custom_filter,
                simplify=True,
                retain_all=True,
                truncate_by_edge=True,
            )
            break
        except ValueError as exc:
            log.debug("Empty tile %s: %s", bbox, exc)
            return None
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                log.warning(
                    "Network error fetching tile %s (attempt %d/%d): %s - retrying",
                    bbox,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                continue
            log.warning("Giving up on tile %s after %d attempts: %s", bbox, max_retries + 1, exc)
            raise _TransientFetchError(str(exc)) from exc

    if len(graph.edges) == 0:
        return None

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index(drop=True)
    del graph
    if "maxspeed" not in edges.columns:
        edges["maxspeed"] = None
    return edges[["highway", "maxspeed", "geometry"]]


def _clean_road_gdf(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to vehicle-relevant highway types and normalise list-valued tags.

    OSM allows a way to carry multiple values for one tag (rendered by
    osmnx as a Python list) - both "highway" (e.g. ["secondary", "primary"]
    for a reclassified road segment) and "maxspeed" (e.g. ["50", "100"] for
    lanes with different limits) can do this; this takes the first value
    for each, a standard, documented OSM-processing simplification. Without
    normalising maxspeed too, a list value survives all the way to
    GeoDataFrame.to_parquet(), which pyarrow rejects outright (confirmed via
    a real, non-mocked OSM fetch - Bremen: "Expected bytes, got a 'list'
    object" on the maxspeed column).
    """
    gdf = raw_gdf.copy()
    gdf["highway"] = gdf["highway"].apply(lambda v: v[0] if isinstance(v, list) else v)
    gdf["maxspeed"] = gdf["maxspeed"].apply(lambda v: v[0] if isinstance(v, list) else v)
    return gdf[gdf["highway"].isin(_VEHICLE_HIGHWAY_VALUES)].reset_index(drop=True)


def _state_cache_path(cache_dir: Path, state: str) -> Path:
    """The whole-state cache parquet path download_road_network reads/writes.

    Shared with build_spatial_features so its retry-incomplete-states pass
    checks the exact same path download_road_network itself uses to decide
    whether a state is fully resolved.
    """
    state_slug = state.lower().replace(" ", "_").replace("ü", "ue").replace("ä", "ae")
    return Path(cache_dir) / f"{state_slug}.parquet"


def download_road_network(
    state: str, cache_dir: Path, force_refresh: bool = False
) -> gpd.GeoDataFrame:
    """Fetch and cache OSM road ways (as a GeoDataFrame) for one German state.

    Returns a GeoDataFrame with columns [highway, maxspeed, geometry],
    filtered to vehicle-relevant road types via _clean_road_gdf.

    Fetches in a grid of small tiles (_grid_tiles/_fetch_tile_edges) rather
    than one osmnx.graph_from_place call for the whole state - root-caused
    via journalctl evidence (a live run was reproducibly SIGKILLed by the
    Linux OOM killer ~10 minutes in, every time, regardless of how the
    notebook was run: "Out of memory: Killed process ... anon-rss:
    27643136kB" - a SIGKILL Python can never catch or log, which is why no
    traceback ever appeared). osmnx.graph_from_place/graph_from_bbox ALWAYS
    builds the full, raw, unsimplified MultiDiGraph first (every OSM node
    becomes a graph node - for Baden-Württemberg, osmnx's own logging
    showed "12,101,740 OSM nodes and 25,417,314 edges") and only applies
    simplification as a subsequent step - so simplify=True/False changes
    only the FINAL output size, not the peak memory during construction,
    which is the actual OOM trigger (confirmed: switching to simplify=True
    alone was tried and still hit 26.5GB RSS / 98% system memory on a live
    retry). osmnx also already internally subdivides huge query polygons
    into multiple smaller Overpass HTTP requests, but still accumulates
    every sub-request's nodes/ways into one dict before ever building the
    graph - that only bounds request size, never peak memory.

    Tiling (0.2° grid, ~same order of magnitude as Frankfurt am Main's own
    bounding box, empirically verified safe at 654MB even fully
    unsimplified) bounds peak memory per tile instead, at the cost of many
    more, smaller Overpass requests per state. Uses osmnx.graph_from_bbox
    with an explicit custom_filter matching _VEHICLE_HIGHWAY_VALUES exactly,
    NOT osmnx.features.features_from_place (the original implementation) -
    features_from_place returns every raw OSM tag as its own DataFrame
    column with no way to restrict this (confirmed empirically to raise
    MemoryError trying to allocate ~79 GiB for one state), independent of
    tile size. graph_from_bbox, by contrast, respects
    ox.settings.useful_tags_way to limit columns to just what we need.

    ox.settings.log_file = True (NOT log_console) surfaces osmnx's own
    internal progress messages (request/pause/download timing, sub-query
    counts, node/edge counts) - log_console was tried first and confirmed
    NOT to work here: reading osmnx's own source (utils.py's log()
    function), log_console deliberately writes to sys.__stdout__ ("print
    explicitly to terminal in case Jupyter has captured stdout") -
    designed to bypass Jupyter's stdout capture for real terminal use,
    which means it never reaches a Jupyter cell's rendered output at all.
    log_file routes through a standard logging.Logger("OSMnx") instead,
    which propagates to the root logger's handler (configured via
    logging.basicConfig in the notebook) the normal way.

    Per-tile caching (tile_cache_dir): a live Baden-Württemberg run took
    2h15m for 180 tiles (~44% of tiles took 30-180s each - public Overpass
    server load/near-timeouts, not something tile size controls), and the
    whole-state cache_path is only written after ALL tiles succeed - so an
    interrupted run (killed, crashed, network drop) previously lost every
    tile fetched so far and restarted tile 1 from scratch. Each tile's
    result is now cached individually to
    cache_dir/{state_slug}_tiles/tile_{i:04d}.parquet immediately after a
    successful fetch, and re-fetching skips any tile whose cache file
    already exists (unless force_refresh) - so a retry resumes instead of
    restarting.

    State-level retry: if any tiles remain unresolved (transient network
    errors) after a full pass over the tile list, the pass is repeated in
    place - up to 5 attempts total - before returning a partial result to
    the caller. Each retry attempt only re-fetches tiles still missing;
    everything already cached (from this call's earlier attempts or a
    prior call) is skipped. This means a state resolves itself within a
    single call whenever possible; build_spatial_features's own
    cross-state retry-pass is only reached if a mirror outage outlasts
    even these 5 in-place attempts.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_slug = state.lower().replace(" ", "_").replace("ü", "ue").replace("ä", "ae")
    cache_path = _state_cache_path(cache_dir, state)

    if cache_path.exists() and not force_refresh:
        log.info("Using cached OSM road network for %s at %s", state, cache_path)
        return gpd.read_parquet(cache_path)

    log.info("Fetching OSM road network for %s (this can take a few minutes)...", state)
    ox.settings.useful_tags_way = ["highway", "maxspeed"]
    ox.settings.log_file = True
    # Live-verified (Baden-Württemberg): peak RSS during tiled fetching stayed
    # under ~1.1GB after 10 tiles at tile_size_deg=0.2, ~30x below the 26GB
    # OOM threshold. tile_size_deg=0.4 (4x the area) was tried to cut request
    # count/wall-clock time, but caused a WORSE failure mode: several tiles
    # took 3+ minutes per attempt before hitting a 180s server-side read
    # timeout, apparently because the heavier per-tile query exceeded what
    # the Overpass mirror could finish in time - a slow, expensive failure,
    # not the fast, cheap 504-and-retry seen at 0.2 deg. Reverted to 0.2 -
    # proven to complete tiles reliably, just with more (individually
    # faster/cheaper) requests. Use the default public overpass-api.de mirror
    # for these tile-sized requests; the previous 504s were observed with the
    # older, much larger state-level requests, which are a very different load
    # profile from 0.2 degree bboxes.
    ox.settings.overpass_url = "https://overpass-api.de/api"
    highway_filter = "|".join(sorted(_VEHICLE_HIGHWAY_VALUES))
    custom_filter = f'["highway"~"^({highway_filter})$"]'

    west, south, east, north = ox.geocode_to_gdf(f"{state}, Germany").total_bounds
    tiles = _grid_tiles(west, south, east, north, tile_size_deg=0.2)
    log.info("%s: split into %d tiles for fetching", state, len(tiles))

    tile_cache_dir = cache_dir / f"{state_slug}_tiles"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    # State-level retry: a handful of tiles exhausting their own internal
    # retries (_fetch_tile_edges, transient network errors) no longer
    # requires a whole separate external call to pick them back up - retry
    # just those tiles, in place, up to max_state_retries times, before
    # falling through to build_spatial_features's cross-state retry-pass
    # (kept as the final safety net for a mirror outage that outlasts all
    # of this too). Each attempt re-walks the full tile list, but only
    # force_refresh on attempt 1 - already-cached tiles are still applied
    # (skip-and-load) so later attempts touch only what's actually missing.
    max_state_retries = 5
    tile_frames = []
    incomplete_tiles: list[int] = []
    for state_attempt in range(1, max_state_retries + 1):
        effective_force_refresh = force_refresh if state_attempt == 1 else False
        # Tiles that failed the previous pass must be re-fetched even when
        # effective_force_refresh=False: their old stale cache from a prior run
        # is still on disk (failed fetches never write a new cache), so without
        # this the cache-hit check below would load the stale file instead of
        # retrying the actual fetch.
        retry_these = set(incomplete_tiles)
        tile_frames = []
        incomplete_tiles = []
        for i, tile in enumerate(tiles, start=1):
            tile_cache_path = tile_cache_dir / f"tile_{i:04d}.parquet"
            if tile_cache_path.exists() and not effective_force_refresh and i not in retry_these:
                log.info(
                    "  [tile %d/%d] cached at %s, skipping fetch", i, len(tiles), tile_cache_path
                )
                tile_edges = gpd.read_parquet(tile_cache_path)
                if len(tile_edges) > 0:
                    tile_frames.append(tile_edges)
                continue
            remaining_tiles_this_state = len(tiles) - i + 1
            log.info("  [tile %d/%d] %s", i, len(tiles), tile)
            overall_progress = _overall_progress_str()
            if overall_progress:
                log.info(
                    "      state %s | %s",
                    _eta_str(remaining_tiles_this_state),
                    overall_progress,
                )
            else:
                log.info("      state %s", _eta_str(remaining_tiles_this_state))
            tile_fetch_start = time.time()
            try:
                tile_edges = _fetch_tile_edges(tile, custom_filter)
            except _TransientFetchError as exc:
                # Do NOT write a tile cache file here - caching this as empty
                # would permanently record a transient outage as "no roads in
                # this tile", indistinguishable from a genuinely empty tile on
                # every future run. Leaving no cache file means a later
                # attempt (this state-retry loop, or a later external call)
                # retries exactly this tile instead of skipping it forever.
                log.warning(
                    "  [tile %d/%d] leaving uncached after repeated failures: %s",
                    i,
                    len(tiles),
                    exc,
                )
                incomplete_tiles.append(i)
                continue
            # Only real, successfully-completed fetches feed the adaptive ETA -
            # a tile that exhausted retries above already spent minutes retrying
            # a dead backend, which would skew the average toward "everything is
            # slow" even though the healthy backend handles most tiles quickly.
            _record_tile_seconds(time.time() - tile_fetch_start)
            _note_tile_done()
            if tile_edges is not None:
                # Clean before caching (not just at the end, on the combined
                # frame): raw tile_edges can have list-valued "highway" tags
                # (OSM's own multi-value tag convention, per
                # _clean_road_gdf's docstring), which pyarrow's parquet
                # writer rejects outright. _clean_road_gdf is idempotent, so
                # running it again on the combined frame below is harmless.
                tile_edges = _clean_road_gdf(tile_edges)
                tile_edges.to_parquet(tile_cache_path)
                if len(tile_edges) > 0:
                    tile_frames.append(tile_edges)
            else:
                gpd.GeoDataFrame(columns=["highway", "maxspeed", "geometry"]).to_parquet(
                    tile_cache_path
                )

        if not incomplete_tiles:
            break
        if state_attempt < max_state_retries:
            log.warning(
                "%s: attempt %d/%d left %d/%d tiles unresolved, retrying state now "
                "(max %d attempts)...",
                state,
                state_attempt,
                max_state_retries,
                len(incomplete_tiles),
                len(tiles),
                max_state_retries,
            )
        else:
            log.warning(
                "%s: %d/%d tiles still unresolved after %d attempts - giving up for now, "
                "a later run or the cross-state retry-pass will retry them",
                state,
                len(incomplete_tiles),
                len(tiles),
                max_state_retries,
            )

    if not tile_frames:
        raise RuntimeError(f"No road data found for {state} across {len(tiles)} tiles")

    log.info(
        "%s: all %d tiles fetched (%d with road data), combining and deduping...",
        state,
        len(tiles),
        len(tile_frames),
    )
    combined = gpd.GeoDataFrame(pd.concat(tile_frames, ignore_index=True), geometry="geometry")
    cleaned = _clean_road_gdf(combined)

    # truncate_by_edge=True (required in _fetch_tile_edges to avoid
    # grid-aligned data gaps at tile seams) duplicates boundary-crossing
    # edges across both adjacent tiles - dedup on (highway, maxspeed,
    # geometry) removes them. Two genuinely distinct real-world segments
    # coinciding exactly in both tags and geometry is not realistic.
    before_dedup = len(cleaned)
    cleaned = cleaned.assign(_geom_wkb=cleaned.geometry.to_wkb())
    cleaned = (
        cleaned.drop_duplicates(subset=["highway", "maxspeed", "_geom_wkb"])
        .drop(columns=["_geom_wkb"])
        .reset_index(drop=True)
    )
    log.info("%s: %d tile rows -> %d after tile-boundary dedup", state, before_dedup, len(cleaned))

    if incomplete_tiles:
        # Do NOT write the whole-state cache_path: doing so would mark this
        # state as permanently "done" (download_road_network's cache_path
        # check short-circuits all tile logic on the next call), baking the
        # missing tiles' absence into the final cached network forever -
        # exactly what happened to Bayern and Brandenburg before this fix
        # (70 tiles silently zeroed out across two already-finalized state
        # caches). Returning the partial result lets the caller proceed for
        # now, but a later call with the same cache_dir will retry exactly
        # these tiles and only then write cache_path.
        log.warning(
            "%s: %d/%d tiles left uncached after retries - NOT writing %s yet, "
            "re-run to retry them",
            state,
            len(incomplete_tiles),
            len(tiles),
            cache_path,
        )
        return cleaned

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

    Progress: logs (via log.info, NOT print) a "[i/16] fetching <state>..."
    line before each state and a "-> done in Xs (N ways)" line after.
    Deliberately NOT print() - confirmed empirically that neither print()
    nor any logging handler writing to sys.stdout/sys.__stdout__ (including
    osmnx's own log_console) is visible when this runs under
    `jupyter nbconvert --execute`: nbconvert captures each cell's stdout
    into the notebook's own cell-output JSON, not into nbconvert's own
    process-level stdout stream - so redirecting nbconvert's stdout to a
    file (`nbconvert ... > file.log`) never receives it, confirmed with a
    minimal reproduction. This only ever worked when watching a live
    Jupyter kernel (e.g. VSCode's interactive window), which renders cell
    output directly. log.info() itself is unaffected by this - it's real
    behavior depends entirely on which handlers are attached to the
    logger, so callers (the notebook) must attach a logging.FileHandler
    (direct OS-level file write, immune to nbconvert's stdout capture) if
    they need progress visibility under nbconvert specifically.
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
    osm_cache_dir = raw_cache_dir / "osm"

    # Size the overall-run ETA upfront: a geocode-only bounds lookup per
    # state (no Overpass tile fetching) is cheap and gives a real tile
    # budget across all 16 states before any tile is actually fetched -
    # without this, an early "overall ETA" would have to guess the size of
    # states not yet reached, which varies wildly (Bremen: 9 tiles vs
    # Bayern: 425) and would make the "overall" number worse than useless.
    log.info("Sizing overall ETA: checking tile counts for all %d states...", len(GERMAN_STATES))
    state_total_tiles = {state: _state_total_tiles(state) for state in GERMAN_STATES}
    total_tiles = sum(state_total_tiles.values())
    # Tiles already on disk (from this or a prior run) count toward "done"
    # from the very first log line - _init_run_progress is what the per-tile
    # log line inside download_road_network reads via _overall_progress_str,
    # so it must be set before the first state's fetch starts, not after.
    initial_tiles_done = sum(_state_cached_tile_count(osm_cache_dir, s) for s in GERMAN_STATES)
    _init_run_progress(total_tiles, initial_tiles_done)

    all_cell_aggregates = []
    total = len(GERMAN_STATES)
    for i, state in enumerate(GERMAN_STATES, start=1):
        state_cache_path = _state_cache_path(osm_cache_dir, state)
        already_cached = state_cache_path.exists() and not force_refresh
        status = (
            "cached" if already_cached else "fetching from OSM (can take a while for large states)"
        )
        log.info("[%d/%d] %s: %s...", i, total, state, status)
        start = time.time()
        roads = download_road_network(state, osm_cache_dir, force_refresh=force_refresh)
        elapsed = time.time() - start
        log.info(
            "  -> %s done in %.1fs (%d ways) | %s",
            state,
            elapsed,
            len(roads),
            _overall_progress_str(),
        )
        all_cell_aggregates.append(aggregate_roads_to_h3(roads, resolution=resolution))

    # A state whose fetch left tiles uncached (transient network errors
    # exhausting retries - see _TransientFetchError) returns its best-effort
    # partial GeoDataFrame above without writing its whole-state cache_path,
    # so cell_features already reflects whatever was fetched. But leaving it
    # there means the state's cache never finalizes and a future run repeats
    # the same partial fetch. Retry those states here, within this same call,
    # bounded to avoid looping forever against a genuinely-down mirror.
    max_retry_passes = 3
    for attempt in range(1, max_retry_passes + 1):
        incomplete = [
            (i, state)
            for i, state in enumerate(GERMAN_STATES)
            if not _state_cache_path(osm_cache_dir, state).exists()
        ]
        if not incomplete:
            break
        log.info(
            "Retry pass %d/%d: %d state(s) still have unresolved tiles: %s",
            attempt,
            max_retry_passes,
            len(incomplete),
            [state for _, state in incomplete],
        )
        for i, state in incomplete:
            start = time.time()
            roads = download_road_network(state, osm_cache_dir, force_refresh=False)
            elapsed = time.time() - start
            log.info(
                "  -> %s retry done in %.1fs (%d ways) | %s",
                state,
                elapsed,
                len(roads),
                _overall_progress_str(),
            )
            all_cell_aggregates[i] = aggregate_roads_to_h3(roads, resolution=resolution)

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

    still_incomplete = [
        state for state in GERMAN_STATES if not _state_cache_path(osm_cache_dir, state).exists()
    ]
    if still_incomplete:
        log.warning(
            "%d state(s) still have unresolved tiles after all retry passes: %s — "
            "NOT writing %s to avoid caching a partial result; re-run to complete them.",
            len(still_incomplete),
            still_incomplete,
            out_path,
        )
        return df

    df.to_parquet(out_path, index=False)
    log.info("Saved spatially-enriched DataFrame → %s (%d rows)", out_path, len(df))
    return df
