"""POI grounding: candidate POIs for each activity inside its predicted administrative dong."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .common import read_jsonl, resolve


# Activity type -> categories of configs/poi_category_map.yaml that can host it.
ACTIVITY_MAP = {
    "home": ["home"],
    "work": ["work"],
    "school": ["school"],
    "shopping": ["shop_grocery", "shop_other"],
    "medical": ["medical"],
    "eat_out": ["eat_out"],
    "leisure": ["leisure_outdoor", "leisure_indoor", "social", "religious"],
    "escort_care": ["escort"],
    "other": ["other"],
}


def _location_key(location: dict[str, Any]) -> tuple[str, str]:
    return (
        " ".join(str(location.get("administrative_dong_name") or "").split()).casefold(),
        " ".join(str(location.get("gu") or "").split()).casefold(),
    )


def build_location_lookup(path: str | Path) -> dict[tuple[str, str], str]:
    """(dong name, gu) -> administrative dong code, from observed KTDB locations.

    Name pairs that map to more than one code are dropped as ambiguous.
    """
    lookup: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        for activity in row.get("activities") or []:
            location = activity.get("location") or {}
            code = str(location.get("administrative_dong_code") or "")
            key = _location_key(location)
            if not code or not key[0]:
                continue
            if key in lookup and lookup[key] != code:
                ambiguous.add(key)
            else:
                lookup[key] = code
    for key in ambiguous:
        lookup.pop(key, None)
    return lookup


def resolve_location_code(location: dict[str, Any], lookup: dict[tuple[str, str], str] | None) -> str:
    """Administrative dong code of a location; ValueError outside coverage or for an unknown name."""
    code = str(location.get("administrative_dong_code") or "")
    if code:
        return code
    if location.get("scope") == "external":
        raise ValueError("OUTSIDE_POI_COVERAGE")
    key = _location_key(location)
    if lookup is None or key not in lookup:
        raise ValueError(f"UNKNOWN_LOCATION_PAIR:{location.get('administrative_dong_name')}|{location.get('gu')}")
    return lookup[key]


def _stable_rng(seed: int, *values: Any) -> random.Random:
    material = "|".join([str(seed), *(str(value) for value in values)]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _not_empty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).ne("")


def semantic_mask(
    frame: pd.DataFrame,
    activity_type: str,
    mapping: dict[str, Any],
    min_confidence: float,
) -> pd.Series:
    """Rows of a dong's POI frame that can host `activity_type` (OSM tags or official building use)."""
    aliases = ACTIVITY_MAP.get(activity_type, ["other"])
    if aliases == ["other"]:
        return _not_empty(frame.get("name", pd.Series(None, index=frame.index))) | _not_empty(
            frame.get("official_building_name", pd.Series(None, index=frame.index))
        )
    result = pd.Series(False, index=frame.index)
    confidence = pd.to_numeric(
        frame.get("match_confidence", pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0.0)
    official = frame.get("official_main_use", pd.Series("", index=frame.index)).fillna("").astype(str)
    for alias in aliases:
        spec = mapping.get(alias) or {}
        for key, allowed in (spec.get("osm") or {}).items():
            values = frame.get(key, pd.Series("", index=frame.index)).fillna("").astype(str)
            result |= values.ne("") if "*" in allowed else values.isin([str(v) for v in allowed])
        for token in spec.get("official_use_contains") or []:
            result |= official.str.contains(str(token), regex=False) & confidence.ge(min_confidence)
    return result


def _clean(value: Any) -> Any:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def poi_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "poi_id": str(row["poi_id"]),
        "name": _clean(row.get("name")) or _clean(row.get("official_building_name")),
        "longitude": float(row["longitude"]),
        "latitude": float(row["latitude"]),
        "administrative_dong_code": str(row["administrative_dong_code"]),
        "source_kind": _clean(row.get("source_kind")),
        "osm_category": next(
            (
                f"{key}={row[key]}"
                for key in ("amenity", "shop", "office", "leisure", "tourism", "healthcare", "building")
                if _clean(row.get(key))
            ),
            None,
        ),
        "official_main_use": _clean(row.get("official_main_use")),
    }


class POIGrounder:
    """Non-learning administrative-dong POI grounding module."""

    def __init__(self, config: dict[str, Any]):
        paths = config["paths"]
        catalog = pd.read_parquet(Path(paths["indexed_poi_catalog"]))
        self.by_dong = {
            str(code): group.reset_index(drop=True)
            for code, group in catalog.groupby(catalog["administrative_dong_code"].astype(str))
        }
        self.mapping = yaml.safe_load(resolve(paths["poi_category_map"]).read_text(encoding="utf-8"))
        self.lookup = build_location_lookup(paths["location_lookup_source"])
        self.config = config["poi_sampling"]

    def sample_layers(self, record: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """Candidate POI pool per activity and an audit trail of how each was drawn.

        Every home activity reuses one synthetic home anchor; other activities get
        a seeded sample of the activity-compatible POIs of their predicted dong.
        """
        master_size = int(self.config["master_pool_size"])
        active_size = int(self.config["active_pool_size"])
        seed = int(self.config["seed"])
        min_confidence = float(self.config["min_official_match_confidence"])
        sequence = record["activity_sequence"]
        spatial = record["location_dwell_by_activity"]
        if len(sequence) != len(spatial):
            raise ValueError("ACTIVITY_LOCATION_LENGTH_MISMATCH")

        first_home = next((i for i, value in enumerate(sequence) if value == "home"), None)
        if first_home is None:
            raise ValueError("NO_HOME_ACTIVITY")
        home_code = resolve_location_code(spatial[first_home]["location"], self.lookup)
        if not home_code.startswith("11"):
            raise ValueError("HOME_OUTSIDE_SEOUL")
        home_frame = self.by_dong.get(home_code)
        if home_frame is None:
            raise ValueError(f"NO_POI_DONG:{home_code}")
        eligible_home = home_frame[semantic_mask(home_frame, "home", self.mapping, min_confidence)]
        if eligible_home.empty:
            raise ValueError(f"NO_HOME_POI:{home_code}")
        home_rng = _stable_rng(seed, record["person_id"], "synthetic_home", home_code)
        fixed_home = poi_record(eligible_home.iloc[home_rng.randrange(len(eligible_home))].to_dict())
        fixed_home["synthetic_home_anchor"] = True

        layers: list[list[dict[str, Any]]] = []
        audit: list[dict[str, Any]] = []
        for index, activity_type in enumerate(sequence):
            location = spatial[index]["location"]
            if location.get("scope") == "external":
                raise ValueError(f"OUTSIDE_POI_COVERAGE:activity={index}")
            code = resolve_location_code(location, self.lookup)
            if not code.startswith("11"):
                raise ValueError(f"OUTSIDE_POI_COVERAGE:activity={index}")
            if activity_type == "home":
                layers.append([fixed_home])
                audit.append(
                    {
                        "activity_index": index,
                        "activity_type": activity_type,
                        "predicted_dong_code": code,
                        "effective_dong_code": home_code,
                        "synthetic_home_anchor": True,
                        "home_anchor_reused": True,
                        "eligible_count": len(eligible_home),
                        "active_pool": [fixed_home],
                    }
                )
                continue
            frame = self.by_dong.get(code)
            if frame is None:
                raise ValueError(f"NO_POI_DONG:{code}:activity={index}")
            eligible = frame[semantic_mask(frame, activity_type, self.mapping, min_confidence)]
            if eligible.empty:
                raise ValueError(f"NO_ACTIVITY_POI:{activity_type}:{code}:activity={index}")
            rng = _stable_rng(seed, record["person_id"], index, activity_type, code)
            order = list(range(len(eligible)))
            rng.shuffle(order)
            master = [poi_record(row) for row in eligible.iloc[order[:master_size]].to_dict("records")]
            active_order = list(range(len(master)))
            rng.shuffle(active_order)
            active = [master[position] for position in active_order[:active_size]]
            layers.append(active)
            audit.append(
                {
                    "activity_index": index,
                    "activity_type": activity_type,
                    "predicted_dong_code": code,
                    "effective_dong_code": code,
                    "synthetic_home_anchor": False,
                    "home_anchor_reused": False,
                    "eligible_count": len(eligible),
                    "master_pool_shortage": len(master) < master_size,
                    "active_pool_shortage": len(active) < active_size,
                    "master_pool": master,
                    "active_pool": active,
                }
            )
        return layers, audit
