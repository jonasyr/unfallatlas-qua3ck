import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from unfallatlas.data.osm import GERMAN_STATES, _clean_road_gdf, build_spatial_features


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
