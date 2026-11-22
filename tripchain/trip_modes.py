"""Per-trip main modes: KTDB derivation, routing, and person-level summaries.

A KTDB trip records up to ten stages (TP5_1..TP5_10), each with a mode and a
duration (TP5_i_t1); 15% of Seoul trips have several, mostly walking to and
from a bus or subway. The main mode of a trip is the vehicle ridden longest;
walking stages only count when the whole trip is walked. GraphHopper adds the
walk to and from transit itself, so a main mode is what the router needs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

KTDB_MODE = {
    "1": "walk",
    "3": "bus",
    "4": "intercity_bus",
    "5": "other_bus",
    "7": "subway",
    "8": "rail",
    "9": "high_speed_rail",
    "10": "taxi",
    "11": "truck",
    "12": "bicycle",
    "13": "motorcycle",
    "14": "personal_mobility",
    "15": "mobility_aid",
    "16": "air",
    "17": "ferry",
    "97": "other",
}
KTDB_CAR_ROLE = {"1": "car_driver", "2": "car_passenger"}
# Tie-break only, for equally long vehicle stages (the usual Korean survey order).
TIE_BREAK = (
    "high_speed_rail",
    "rail",
    "subway",
    "intercity_bus",
    "other_bus",
    "bus",
    "car_driver",
    "car_passenger",
    "taxi",
    "truck",
    "motorcycle",
    "bicycle",
    "personal_mobility",
    "mobility_aid",
    "air",
    "ferry",
    "other",
    "walk",
)
_TIE_RANK = {mode: rank for rank, mode in enumerate(TIE_BREAK)}

# Main mode -> (GraphHopper routing mode, traveler mode recorded on the leg).
ROUTING = {
    "walk": ("walk", "walk"),
    "mobility_aid": ("walk", "walk"),
    "bicycle": ("bike", "bicycle"),
    "personal_mobility": ("bike", "bicycle"),
    "bus": ("bus", "bus"),
    "other_bus": ("bus", "bus"),
    "intercity_bus": ("bus", "bus"),
    "subway": ("subway", "subway"),
    "rail": ("subway", "subway"),
    "high_speed_rail": ("subway", "subway"),
    "car_driver": ("car", "car_driver"),
    "car_passenger": ("car", "car_passenger"),
    "taxi": ("car", "taxi"),
    "truck": ("car", "truck"),
    "motorcycle": ("car", "motorcycle"),
}


def _clean(value: Any) -> str:
    text = str(value).strip()
    return "" if text in {"", "nan", "None"} else text.split(".")[0]


def ktdb_stages(row: Any) -> list[tuple[str, float]]:
    """(mode, minutes) for each recorded stage of one KTDB trip row."""
    stages = []
    for i in range(1, 11):
        code = _clean(row.get(f"TP5_{i}"))
        if not code:
            continue
        if code == "2":
            role = _clean(row.get(f"TP5_{i}_2_1"))
            if role not in KTDB_CAR_ROLE:
                raise ValueError(f"car/van stage TP5_{i}=2 has invalid TP5_{i}_2_1={role!r}")
            mode = KTDB_CAR_ROLE[role]
        else:
            mode = KTDB_MODE.get(code, "other")
        try:
            minutes = float(row.get(f"TP5_{i}_t1"))
        except (TypeError, ValueError):
            minutes = 0.0
        stages.append((mode, minutes if minutes == minutes else 0.0))
    return stages


def main_mode(stages: Iterable[tuple[str, float]]) -> str:
    stages = list(stages)
    if not stages:
        return "other"
    vehicles = [stage for stage in stages if stage[0] != "walk"]
    if not vehicles:
        return "walk"
    return min(vehicles, key=lambda stage: (-stage[1], _TIE_RANK.get(stage[0], len(TIE_BREAK))))[0]


def mode_preference_from_trips(trip_modes: list[str]) -> list[dict[str, Any]]:
    """Person-level ranking (count desc, then first use) of per-trip main modes."""
    counts = Counter(trip_modes)
    first = {}
    for index, mode in enumerate(trip_modes):
        first.setdefault(mode, index)
    ordered = sorted(counts, key=lambda mode: (-counts[mode], first[mode]))
    return [{"rank": i + 1, "mode": mode, "count": counts[mode]} for i, mode in enumerate(ordered)]
