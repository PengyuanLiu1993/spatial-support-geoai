from gravity_gnn.geo_osm import is_walkable_highway, parse_numeric_tag, poi_category


def test_parse_numeric_tag() -> None:
    assert parse_numeric_tag("12.5 m") == 12.5
    assert parse_numeric_tag("3;4") == 3.0
    assert parse_numeric_tag(None) is None


def test_poi_category_mapping() -> None:
    assert poi_category({"amenity": "hospital"}) == "hospital"
    assert poi_category({"shop": "supermarket"}) == "supermarket"
    assert poi_category({"leisure": "park"}) == "greenspace"
    assert poi_category({"amenity": "bench"}) is None


def test_walkability_filter() -> None:
    assert is_walkable_highway({"highway": "residential"})
    assert not is_walkable_highway({"highway": "motorway"})
    assert is_walkable_highway({"highway": "trunk", "foot": "yes"})
    assert not is_walkable_highway({"highway": "path", "foot": "private"})
