import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from unfallatlas.data.osm import (
    GERMAN_STATES,
    _clean_road_gdf,
    _eta_str,
    _format_duration,
    _grid_tiles,
    _init_run_progress,
    _note_tile_done,
    _overall_progress_str,
    _record_tile_seconds,
    _tile_seconds_history,
    build_spatial_features,
    download_road_network,
)


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


def test_clean_road_gdf_takes_first_value_from_list_valued_maxspeed_tags():
    """OSM allows list-valued maxspeed (e.g. different limits per lane), same
    as highway - a list surviving un-normalised crashes to_parquet() later
    (pyarrow rejects list-in-object-column), confirmed via a real OSM fetch."""
    raw = gpd.GeoDataFrame(
        {
            "highway": ["primary"],
            "maxspeed": [["50", "100"]],
            "geometry": [LineString([(13.0, 52.0), (13.01, 52.01)])],
        },
        crs="EPSG:4326",
    )
    cleaned = _clean_road_gdf(raw)
    assert cleaned.iloc[0]["maxspeed"] == "50"


def test_build_spatial_features_requires_lat_lon_columns(tmp_path):
    bad_df = pd.DataFrame({"UJAHR": [2020]})
    try:
        build_spatial_features(bad_df, tmp_path / "raw", tmp_path / "interim")
        raise AssertionError("expected RuntimeError for missing LAT/LON")
    except RuntimeError as exc:
        assert "LAT" in str(exc) or "LON" in str(exc)


def _assert_tiles_reconstruct_bbox(tiles, west, south, east, north, tile_size_deg):
    """Every tile must be within tile_size_deg x tile_size_deg, and the
    union of all tiles must reconstruct the original bbox exactly - no
    gaps, no overlaps, no overshoot past the original bounds."""
    for t_west, t_south, t_east, t_north in tiles:
        assert t_east - t_west <= tile_size_deg + 1e-9
        assert t_north - t_south <= tile_size_deg + 1e-9
        assert t_west >= west - 1e-9
        assert t_south >= south - 1e-9
        assert t_east <= east + 1e-9
        assert t_north <= north + 1e-9
    assert min(t[0] for t in tiles) == pytest.approx(west)
    assert min(t[1] for t in tiles) == pytest.approx(south)
    assert max(t[2] for t in tiles) == pytest.approx(east)
    assert max(t[3] for t in tiles) == pytest.approx(north)


def test_grid_tiles_bbox_smaller_than_one_tile_returns_single_tile():
    tiles = _grid_tiles(west=10.0, south=50.0, east=10.05, north=50.05, tile_size_deg=0.2)
    assert tiles == [(10.0, 50.0, 10.05, 50.05)]


def test_grid_tiles_evenly_divisible_bbox_returns_four_equal_tiles():
    tiles = _grid_tiles(west=0.0, south=0.0, east=0.4, north=0.4, tile_size_deg=0.2)
    assert len(tiles) == 4
    for t_west, t_south, t_east, t_north in tiles:
        assert t_east - t_west == pytest.approx(0.2)
        assert t_north - t_south == pytest.approx(0.2)
    _assert_tiles_reconstruct_bbox(tiles, 0.0, 0.0, 0.4, 0.4, tile_size_deg=0.2)


def test_grid_tiles_uneven_bbox_has_correct_trailing_tile_and_reconstructs_bbox():
    tiles = _grid_tiles(west=0.0, south=0.0, east=0.6, north=0.3, tile_size_deg=0.25)
    # width 0.6 / 0.25 -> tiles at 0.25, 0.25, 0.10 (trailing remainder)
    # height 0.3 / 0.25 -> tiles at 0.25, 0.05 (trailing remainder)
    _assert_tiles_reconstruct_bbox(tiles, 0.0, 0.0, 0.6, 0.3, tile_size_deg=0.25)
    trailing_col_widths = {round(t_east - t_west, 10) for t_west, _, t_east, _ in tiles}
    assert 0.1 in trailing_col_widths  # the clipped trailing column exists
    trailing_row_heights = {round(t_north - t_south, 10) for _, t_south, _, t_north in tiles}
    assert 0.05 in trailing_row_heights  # the clipped trailing row exists


def test_grid_tiles_rejects_degenerate_bbox():
    with pytest.raises(ValueError):
        _grid_tiles(west=10.0, south=50.0, east=10.0, north=50.05)  # east == west
    with pytest.raises(ValueError):
        _grid_tiles(west=10.0, south=50.05, east=10.05, north=50.05)  # north == south


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


def test_download_road_network_uses_default_overpass_for_small_tiles(tmp_path, monkeypatch):
    import unfallatlas.data.osm as osm_module

    captured_tiles = []
    captured_urls = []

    class DummyBounds:
        total_bounds = (10.0, 50.0, 10.4, 50.2)

    def _fake_fetch(tile, custom_filter):
        captured_tiles.append(tile)
        captured_urls.append(osm_module.ox.settings.overpass_url)
        return _toy_raw_gdf()

    monkeypatch.setattr(osm_module.ox, "geocode_to_gdf", lambda _: DummyBounds())
    monkeypatch.setattr(osm_module, "_fetch_tile_edges", _fake_fetch)

    download_road_network("Hessen", tmp_path, force_refresh=True)

    assert captured_tiles
    assert set(captured_urls) == {"https://overpass-api.de/api"}
    for west, south, east, north in captured_tiles:
        assert east - west <= 0.2 + 1e-9
        assert north - south <= 0.2 + 1e-9


def test_download_road_network_resumes_from_per_tile_cache_after_interruption(
    tmp_path, monkeypatch
):
    """Simulates a run that fetched every tile but was killed before the final
    whole-state cache_path got written (e.g. during combine/dedup, or the
    process was interrupted right after the last tile) - a second call must
    not re-fetch any tile over the network, since each tile's result was
    already cached individually as soon as it was fetched."""
    import unfallatlas.data.osm as osm_module

    fetch_calls = []

    class DummyBounds:
        total_bounds = (10.0, 50.0, 10.4, 50.2)

    def _fake_fetch(tile, custom_filter):
        fetch_calls.append(tile)
        return _toy_raw_gdf()

    monkeypatch.setattr(osm_module.ox, "geocode_to_gdf", lambda _: DummyBounds())
    monkeypatch.setattr(osm_module, "_fetch_tile_edges", _fake_fetch)

    download_road_network("Hessen", tmp_path, force_refresh=True)
    n_tiles_first_run = len(fetch_calls)
    assert n_tiles_first_run > 0

    # Simulate the interrupted run: the whole-state cache was never written,
    # but the per-tile cache directory survives (it's written incrementally).
    cache_path = tmp_path / "hessen.parquet"
    assert cache_path.exists()
    cache_path.unlink()

    download_road_network("Hessen", tmp_path, force_refresh=False)

    assert len(fetch_calls) == n_tiles_first_run  # no additional network fetches
    assert cache_path.exists()  # the whole-state cache is now written


def test_format_duration_picks_the_coarsest_nonzero_unit():
    assert _format_duration(45) == "45s"
    assert _format_duration(125) == "2m05s"
    assert _format_duration(3725) == "1h02m"


def test_eta_str_is_zero_before_any_tile_has_been_timed(monkeypatch):
    monkeypatch.setattr("unfallatlas.data.osm._tile_seconds_history", type(_tile_seconds_history)())
    assert _eta_str(10) == "ETA warming up..."


def test_eta_str_reports_done_for_zero_remaining():
    assert _eta_str(0) == "done"


def test_eta_str_uses_rolling_average_of_recorded_tile_durations(monkeypatch):
    history = type(_tile_seconds_history)()
    monkeypatch.setattr("unfallatlas.data.osm._tile_seconds_history", history)
    _record_tile_seconds(2.0)
    _record_tile_seconds(4.0)
    # avg = 3.0s/tile, 10 remaining -> 30s
    assert _eta_str(10) == "ETA 30s (3.0s/tile avg)"


def test_overall_progress_str_is_empty_when_run_progress_was_never_initialised(monkeypatch):
    monkeypatch.setattr(
        "unfallatlas.data.osm._run_progress",
        {"total_tiles": None, "tiles_done": 0, "start": None},
    )
    assert _overall_progress_str() == ""


def test_overall_progress_str_reflects_tiles_already_done_before_any_fetch(monkeypatch):
    monkeypatch.setattr(
        "unfallatlas.data.osm._run_progress",
        {"total_tiles": None, "tiles_done": 0, "start": None},
    )
    # 500 of 1000 tiles already cached from a prior run - _init_run_progress
    # must reflect that from the very first line, not start from 0.
    _init_run_progress(total_tiles=1000, tiles_already_done=500)
    assert "500/1000 tiles" in _overall_progress_str()


def test_note_tile_done_advances_overall_progress(monkeypatch):
    monkeypatch.setattr(
        "unfallatlas.data.osm._run_progress",
        {"total_tiles": None, "tiles_done": 0, "start": None},
    )
    _init_run_progress(total_tiles=10, tiles_already_done=0)
    _note_tile_done()
    _note_tile_done()
    assert "2/10 tiles" in _overall_progress_str()


def test_download_road_network_retries_failed_tiles_within_the_same_call(tmp_path, monkeypatch):
    """A tile that fails transiently on the first pass but would succeed on
    a later attempt must resolve within this single download_road_network
    call (state-level retry, up to 5 attempts) rather than requiring the
    caller to invoke download_road_network again."""
    import unfallatlas.data.osm as osm_module
    from unfallatlas.data.osm import _TransientFetchError

    class DummyBounds:
        # Smaller than one 0.2deg tile on both axes -> exactly one tile,
        # so call counting below isn't sensitive to _grid_tiles' tile count.
        total_bounds = (10.0, 50.0, 10.1, 50.1)

    calls = []

    def _flaky_then_healthy_fetch(tile, custom_filter):
        calls.append(tile)
        if len(calls) < 3:
            raise _TransientFetchError("simulated transient failure, recovers later")
        return _toy_raw_gdf()

    monkeypatch.setattr(osm_module.ox, "geocode_to_gdf", lambda _: DummyBounds())
    monkeypatch.setattr(osm_module, "_fetch_tile_edges", _flaky_then_healthy_fetch)

    download_road_network("Hessen", tmp_path, force_refresh=True)

    # Resolved by the 3rd attempt, well within the 5-attempt budget - the
    # whole-state cache must be written without any extra external call.
    assert len(calls) == 3
    assert (tmp_path / "hessen.parquet").exists()
    assert (tmp_path / "hessen_tiles" / "tile_0001.parquet").exists()


def test_download_road_network_gives_up_after_max_state_retries(tmp_path, monkeypatch):
    """A tile that never recovers must not retry forever within one call -
    bounded at 5 attempts, then returned to the caller as still-incomplete
    (leaving the state's cache-level retry, and eventually
    build_spatial_features's cross-state retry-pass, to pick it up later)."""
    import unfallatlas.data.osm as osm_module
    from unfallatlas.data.osm import _TransientFetchError

    class DummyBounds:
        total_bounds = (10.0, 50.0, 10.1, 50.1)  # exactly one tile

    calls = []

    def _always_flaky_fetch(tile, custom_filter):
        calls.append(tile)
        raise _TransientFetchError("simulated permanent failure")

    monkeypatch.setattr(osm_module.ox, "geocode_to_gdf", lambda _: DummyBounds())
    monkeypatch.setattr(osm_module, "_fetch_tile_edges", _always_flaky_fetch)

    # The only tile never resolves, so tile_frames stays empty and
    # download_road_network raises (matching its existing "no road data
    # found" contract) - the retry budget is still what's under test here.
    with pytest.raises(RuntimeError, match="No road data found"):
        download_road_network("Hessen", tmp_path, force_refresh=True)

    assert len(calls) == 5  # exactly the 5-attempt budget, not unbounded
    assert not (tmp_path / "hessen.parquet").exists()


def test_download_road_network_does_not_cache_tiles_that_exhaust_retries(tmp_path, monkeypatch):
    """A tile that gives up after repeated network errors must NOT be cached
    as empty - that would permanently record a transient outage as "no
    roads here", indistinguishable from a genuinely empty tile on every
    future run. A real live run (Bayern/Brandenburg) proved this: 70 tiles
    were silently zeroed out this way, ~10% of two already-finalized state
    caches, because of one dead backend in overpass-api.de's DNS
    round-robin pool - not because those areas have no roads."""
    import unfallatlas.data.osm as osm_module
    from unfallatlas.data.osm import _TransientFetchError

    class DummyBounds:
        total_bounds = (10.0, 50.0, 10.4, 50.2)

    def _flaky_fetch(tile, custom_filter):
        # First tile fails every retry; the rest succeed normally.
        if tile[0] == 10.0:
            raise _TransientFetchError("simulated repeated network failure")
        return _toy_raw_gdf()

    monkeypatch.setattr(osm_module.ox, "geocode_to_gdf", lambda _: DummyBounds())
    monkeypatch.setattr(osm_module, "_fetch_tile_edges", _flaky_fetch)

    result = download_road_network("Hessen", tmp_path, force_refresh=True)

    # The whole-state cache must NOT be written while a tile remains unresolved.
    assert not (tmp_path / "hessen.parquet").exists()
    # No cache file exists for the failed tile either - it must be retried later.
    assert not (tmp_path / "hessen_tiles" / "tile_0001.parquet").exists()
    # Other tiles' data is still returned to the caller.
    assert len(result) > 0
