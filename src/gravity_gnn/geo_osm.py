from __future__ import annotations

import re
from collections.abc import Mapping


_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def parse_numeric_tag(value: str | None) -> float | None:
    """Parse the first numeric value from a simple OSM height/level tag."""

    if not value:
        return None
    match = _NUMBER.search(str(value).replace(",", "."))
    if match is None:
        return None
    parsed = float(match.group(0))
    return parsed if parsed >= 0 else None


def poi_category(tags: Mapping[str, str]) -> str | None:
    """Map open OSM tags to stable analysis categories."""

    amenity = tags.get("amenity", "")
    healthcare = tags.get("healthcare", "")
    shop = tags.get("shop", "")
    leisure = tags.get("leisure", "")
    landuse = tags.get("landuse", "")
    natural = tags.get("natural", "")

    if amenity == "hospital" or healthcare == "hospital":
        return "hospital"
    if amenity in {"clinic", "doctors"} or healthcare in {"clinic", "doctor"}:
        return "clinic_doctor"
    if amenity == "pharmacy" or healthcare == "pharmacy":
        return "pharmacy"
    if amenity == "dentist" or healthcare == "dentist":
        return "dentist"
    if amenity in {"school", "kindergarten", "college"}:
        return "school"
    if amenity == "university":
        return "university"
    if shop == "supermarket":
        return "supermarket"
    if shop in {"convenience", "greengrocer"}:
        return "convenience"
    if amenity in {"restaurant", "cafe", "food_court"}:
        return "restaurant_cafe"
    if amenity == "fast_food":
        return "fast_food"
    if amenity in {"pub", "bar", "nightclub"}:
        return "pub_bar"
    if shop in {"tobacco", "e-cigarette"}:
        return "tobacco"
    if amenity == "gambling" or shop in {"bookmaker", "lottery"}:
        return "gambling"
    if leisure in {"fitness_centre", "sports_centre", "swimming_pool", "stadium"}:
        return "sport_fitness"
    if leisure in {"park", "garden", "playground", "nature_reserve"}:
        return "greenspace"
    if landuse in {"recreation_ground", "village_green", "forest"}:
        return "greenspace"
    if natural in {"wood", "water", "wetland", "beach"}:
        return "green_blue_space"
    return None


def is_walkable_highway(tags: Mapping[str, str]) -> bool:
    highway = tags.get("highway", "")
    if not highway:
        return False
    foot = tags.get("foot", "")
    access = tags.get("access", "")
    if foot in {"no", "private"}:
        return False
    if access == "private" and foot not in {"yes", "designated", "permissive"}:
        return False
    if highway in {"motorway", "motorway_link", "trunk", "trunk_link"} and foot not in {
        "yes",
        "designated",
    }:
        return False
    return True
