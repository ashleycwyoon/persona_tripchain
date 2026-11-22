"""Distance band of a trip, for the distance-conditioned trip-mode distribution reward.

A trip runs from activity i to activity i+1. Its band is decided from the two
administrative dongs:
- same_dong: both ends in the same dong (85% walk in the v2 train GT),
- <3km / 3-10km / 10km+: straight-line distance between the dongs' POI-catalog
  centroids (Seoul dongs),
- outside_seoul: an end outside Seoul or not resolvable to a dong code.

Ground-truth locations carry dong codes; Student2 locations carry dong name + gu,
which are mapped to codes with the same lookup the environment uses
(tripchain.poi.build_location_lookup on paths.location_lookup_source).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .common import resolve
from .poi import build_location_lookup, resolve_location_code

DISTANCE_BANDS = ("same_dong", "<3km", "3-10km", "10km+", "outside_seoul")


def band_of_km(km: float) -> str:
    return "<3km" if km < 3 else "3-10km" if km < 10 else "10km+"


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


class TripDistance:
    def __init__(self, centroids: dict[str, tuple[float, float]], lookup: dict[tuple[str, str], str] | None = None):
        self.centroids = centroids
        self.lookup = lookup

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TripDistance":
        paths = config["paths"]
        catalog = pd.read_parquet(
            resolve(paths["indexed_poi_catalog"]), columns=["administrative_dong_code", "latitude", "longitude"]
        )
        means = catalog.groupby(catalog["administrative_dong_code"].astype(str))[["latitude", "longitude"]].mean()
        centroids = {code: (float(row.latitude), float(row.longitude)) for code, row in means.iterrows()}
        return cls(centroids, build_location_lookup(paths["location_lookup_source"]))

    def code(self, location: dict[str, Any] | None) -> str | None:
        if not location:
            return None
        try:
            return resolve_location_code(location, self.lookup) or None
        except ValueError:  # outside coverage or unknown name pair
            return None

    def band(self, origin: dict[str, Any] | None, destination: dict[str, Any] | None) -> str:
        a, b = self.code(origin), self.code(destination)
        if a and a == b:
            return "same_dong"
        if a in self.centroids and b in self.centroids:
            return band_of_km(haversine_km(self.centroids[a], self.centroids[b]))
        return "outside_seoul"

    def day_bands(self, activities: list[dict[str, Any]]) -> list[str]:
        """One band per trip of a day (activities with a `location` each)."""
        locations = [item.get("location") for item in activities]
        return [self.band(locations[i], locations[i + 1]) for i in range(len(locations) - 1)]
