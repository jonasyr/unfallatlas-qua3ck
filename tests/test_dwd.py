import pandas as pd
from unfallatlas.data.dwd import find_nearest_station

STATIONS = pd.DataFrame(
    {
        "station_id": ["00001", "00002", "00003"],
        "lat": [52.5200, 48.1351, 50.1109],  # Berlin, Munich, Frankfurt
        "lon": [13.4050, 11.5820, 8.6821],
    }
)


def test_find_nearest_station_returns_closest_within_radius():
    # A point near Berlin should resolve to the Berlin station.
    station_id, distance_km = find_nearest_station(52.52, 13.40, STATIONS)

    assert station_id == "00001"
    assert distance_km < 1.0


def test_find_nearest_station_returns_none_outside_max_km():
    # Mid-Atlantic coordinates are far from every station in STATIONS.
    station_id, distance_km = find_nearest_station(40.0, -30.0, STATIONS, max_km=30.0)

    assert station_id is None
    assert distance_km > 30.0
