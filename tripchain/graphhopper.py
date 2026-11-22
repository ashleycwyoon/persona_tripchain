"""GraphHopper routing client: walk, car, bus and subway routes, plus a synthetic bicycle."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen


# Below this distance /route-pt never yields a bus/subway leg (walking is
# faster), and for POIs that snap to the same road node it answers HTTP 500.
PT_MIN_DISTANCE_M = 100.0


def _distance_m(origin: dict[str, Any], destination: dict[str, Any]) -> float:
    lat1, lon1 = math.radians(float(origin["latitude"])), math.radians(float(origin["longitude"]))
    lat2, lon2 = math.radians(float(destination["latitude"])), math.radians(float(destination["longitude"]))
    x = (lon2 - lon1) * math.cos((lat1 + lat2) / 2)
    return 6_371_000.0 * math.hypot(x, lat2 - lat1)


class RoutingUnavailable(RuntimeError):
    """GraphHopper did not answer (timeout, connection or server error).

    Distinct from a genuine no-route answer: it must never be cached or scored
    as an infeasible trip, so it propagates and the rollout is marked failed.
    """


class GraphHopperClient:
    """GraphHopper adapter with a deliberately synthetic bicycle approximation.

    The initial framework does not require a bike graph. Bicycle routes reuse the
    walking route geometry/distance and set TT_bike = TT_walk / bike_speedup.
    The provenance fields ensure that this approximation cannot be mistaken for
    a routed bicycle profile in later analysis.
    """

    def __init__(self, config: dict[str, Any]):
        self.base_url = str(config["base_url"]).rstrip("/")
        self.service_date = str(config["service_date"])
        self.timezone_offset = str(config["timezone_offset"])
        self.timeout = float(config["timeout_sec"])
        self.bike_speedup = float(config.get("bike_walk_speedup", 3.0))
        if self.bike_speedup <= 0:
            raise ValueError("bike_walk_speedup must be positive")
        self.max_cache_entries = max(0, int(config.get("max_cache_entries", 20_000)))
        self.request_attempts = max(1, int(config.get("request_attempts", 4)))
        self.cache: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()

    def _cache_get(self, key: tuple[Any, ...]) -> list[dict[str, Any]] | None:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return [dict(item) for item in self.cache[key]]

    def _cache_put(self, key: tuple[Any, ...], value: list[dict[str, Any]]) -> None:
        if self.max_cache_entries == 0:
            return
        self.cache[key] = [dict(item) for item in value]
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_cache_entries:
            self.cache.popitem(last=False)

    def healthcheck(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/health", timeout=self.timeout) as response:  # noqa: S310
                return response.read().decode().strip() == "OK"
        except Exception:
            return False

    def _fetch(self, url: str) -> list[dict[str, Any]]:
        """Return GraphHopper paths; [] only when GraphHopper answers "no route"."""
        last_error: Exception | None = None
        for attempt in range(self.request_attempts):
            try:
                with urlopen(url, timeout=self.timeout) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8")).get("paths") or []
            except HTTPError as exc:
                if 400 <= exc.code < 500:
                    # e.g. "Cannot find point", out of bounds, no connection found.
                    return []
                last_error = exc
            except Exception as exc:  # timeout, connection reset, malformed body
                last_error = exc
            time.sleep(min(30.0, 2.0**attempt))
        raise RoutingUnavailable(f"GraphHopper unavailable after {self.request_attempts} attempts: {last_error}")

    def _departure(self, minute: float) -> str:
        """ISO departure time of a minute counted from the 03:00 day origin."""
        local = datetime.combine(date.fromisoformat(self.service_date), datetime.min.time())
        local += timedelta(hours=3, minutes=max(0.0, min(1439.0, float(minute))))
        return local.isoformat(timespec="seconds") + self.timezone_offset

    def routes(
        self,
        origin: dict[str, Any],
        destination: dict[str, Any],
        mode: str,
        departure_min: float,
    ) -> list[dict[str, Any]]:
        """Route options between two POIs in one mode; [] when there is no route."""
        # Only timetable-based public transport depends on departure time.
        # Keeping time in walk/car/bike keys causes identical OD routes to be
        # recomputed for every DP state without changing the result.
        departure_bucket = int(float(departure_min)) if mode in {"bus", "subway"} else 0
        key = (origin["poi_id"], destination["poi_id"], mode, departure_bucket)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        if mode == "bike":
            walking = self.routes(origin, destination, "walk", departure_min)
            result = []
            for route in walking:
                value = dict(route)
                value.update(
                    {
                        "route_id": "SYNTH_BIKE_" + str(route["route_id"]),
                        "mode": "bike",
                        "routing_profile": "bike_proxy_walk",
                        "travel_time_min": float(route["travel_time_min"]) / self.bike_speedup,
                        "travel_time_source": f"walk_divided_by_{self.bike_speedup:g}",
                    }
                )
                result.append(value)
            self._cache_put(key, result)
            return [dict(item) for item in result]

        if origin["poi_id"] == destination["poi_id"]:
            result = [
                {
                    "route_id": "SAME_POI",
                    "mode": mode,
                    "routing_profile": mode,
                    "travel_time_min": 0.0,
                    "distance_km": 0.0,
                    "transfer_count": 0,
                    "travel_time_source": "same_poi",
                }
            ]
            self._cache_put(key, result)
            return [dict(item) for item in result]

        points = [
            f"{origin['latitude']},{origin['longitude']}",
            f"{destination['latitude']},{destination['longitude']}",
        ]
        if mode in {"bus", "subway"}:
            # Bus and subway send the identical /route-pt query and only filter
            # the answer differently, so the raw paths are fetched once and shared.
            pt_key = (origin["poi_id"], destination["poi_id"], "pt_paths", departure_bucket)
            paths = self._cache_get(pt_key)
            if _distance_m(origin, destination) < PT_MIN_DISTANCE_M:
                # No transit trip at this distance (see PT_MIN_DISTANCE_M).
                paths = []
            params = [
                ("point", points[0]),
                ("point", points[1]),
                ("pt.earliest_departure_time", self._departure(departure_min)),
                ("profile", "foot"),
                ("pt.profile", "true"),
                ("limit_solutions", 4),
                ("locale", "en"),
                ("points_encoded", "false"),
            ]
            endpoint = "/route-pt"
        else:
            profile = {"walk": "foot", "car": "car"}.get(mode)
            if profile is None:
                return []
            params = [
                ("point", points[0]),
                ("point", points[1]),
                ("profile", profile),
                ("locale", "en"),
                ("instructions", "false"),
                ("points_encoded", "false"),
            ]
            if profile != "car":
                params.append(("ch.disable", "true"))
            endpoint = "/route"

        url = f"{self.base_url}{endpoint}?{urlencode(params)}"
        if mode in {"bus", "subway"}:
            if paths is None:
                paths = self._fetch(url)
                self._cache_put(pt_key, paths)
        else:
            paths = self._fetch(url)

        result: list[dict[str, Any]] = []
        for path_index, path in enumerate(paths):
            legs = path.get("legs") or []
            route_ids = [str(leg.get("route_id", "")) for leg in legs if leg.get("type") == "pt"]
            if mode == "bus" and not any(value.startswith("BR_") for value in route_ids):
                continue
            if mode == "subway" and not any(value.startswith(("SR_", "RR_")) for value in route_ids):
                continue
            signature = json.dumps([key, path_index, route_ids], sort_keys=True)
            result.append(
                {
                    "route_id": "GH_" + hashlib.sha1(signature.encode()).hexdigest()[:16],
                    "mode": mode,
                    "routing_profile": mode,
                    "travel_time_min": float(path.get("time", 0.0) or 0.0) / 60000.0,
                    "distance_km": float(path.get("distance", 0.0) or 0.0) / 1000.0,
                    "transfer_count": max(0, int(path.get("transfers", 0) or 0)),
                    "transit_route_ids": route_ids,
                    "travel_time_source": "graphhopper",
                }
            )
        self._cache_put(key, result)
        return [dict(item) for item in result]
