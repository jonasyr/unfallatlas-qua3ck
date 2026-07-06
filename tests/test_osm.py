import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from unfallatlas.data.osm import (
    GERMAN_STATES,
    _clean_road_gdf,
    _grid_tiles,
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
