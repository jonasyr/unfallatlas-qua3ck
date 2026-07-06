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
