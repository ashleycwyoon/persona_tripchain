"""Build KTDB Student 2 SFT examples from teacher personas and spatial targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from tripchain.ktdb.ktdb_student_common import (
    DEFAULT_TEACHER_SHARDS,
    compact_json,
    deterministic_split,
    load_teacher_rows,
    persona_bundle,
    read_jsonl_by_id,
    require_completed_teacher_shards,
    require_identical_ids,
)
from tripchain.paths import TRIPCHAIN_1015_DATA as PARENT_DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCATION_WHITELIST = PARENT_DATA_ROOT / "processed" / "ktdb_location_whitelist.json"

__all__ = [
    "DEFAULT_TEACHER_SHARDS",
    "build_student2_examples",
    "deterministic_split",
    "require_completed_teacher_shards",
    "DEFAULT_LOCATION_WHITELIST",
]


def load_location_whitelist(path: Path = DEFAULT_LOCATION_WHITELIST) -> dict[str, Any]:
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not {"source", "as_of", "seoul", "external_sigungu"} <= set(value):
        raise ValueError(f"invalid KTDB location whitelist: {path}")
    return {"seoul": value["seoul"], "external_sigungu": value["external_sigungu"]}


def _validate_model_location(person_id: str, location: dict[str, Any], whitelist: dict[str, Any]) -> None:
    if location["scope"] == "seoul":
        allowed = whitelist["seoul"].get(location["gu"], [])
        if location["administrative_dong_name"] not in allowed:
            raise ValueError(f"person_id={person_id}: Seoul location absent from codebook whitelist: {location}")
        return
    allowed_external = {(item["sido"], item["sigungu"]) for item in whitelist["external_sigungu"]}
    if (location["sido"], location["sigungu"]) not in allowed_external:
        raise ValueError(f"person_id={person_id}: external location absent from codebook whitelist: {location}")


def _location_dwell_targets(
    person_id: str,
    spatial: dict[str, Any],
    labels_by_code: dict[str, tuple[str, str | None, str | None]],
    location_whitelist: dict[str, Any],
) -> list[dict[str, Any]]:
    source_activities = spatial.get("activities")
    if not isinstance(source_activities, list) or not source_activities:
        raise ValueError(f"person_id={person_id}: empty spatial activities")
    targets: list[dict[str, Any]] = []
    for expected_index, activity in enumerate(source_activities):
        if activity.get("activity_index") != expected_index:
            raise ValueError(f"person_id={person_id}: non-contiguous activity_index")
        location = activity.get("location")
        dwell_time = activity.get("dwell_time_min")
        if not isinstance(location, dict):
            raise ValueError(f"person_id={person_id}: missing administrative-dong location")
        name, gu, sido = (location.get("administrative_dong_name"), location.get("gu"), location.get("sido"))
        if not name or not gu or not sido:
            name, gu, sido = labels_by_code.get(str(location.get("administrative_dong_code") or ""), (name, gu, sido))
        if not gu or not sido or (sido == "Seoul" and not name):
            raise ValueError(f"person_id={person_id}: incomplete natural-language location")
        if not isinstance(dwell_time, int) or dwell_time < 0:
            raise ValueError(f"person_id={person_id}: invalid dwell_time_min")
        model_location = (
            {"scope": "seoul", "administrative_dong_name": name, "gu": gu}
            if sido == "Seoul"
            else {"scope": "external", "sido": sido, "sigungu": gu}
        )
        if not all(value for key, value in model_location.items() if key != "scope"):
            raise ValueError(f"person_id={person_id}: incomplete natural-language location")
        _validate_model_location(person_id, model_location, location_whitelist)
        targets.append(
            {
                "activity_index": expected_index,
                # Codes remain in the source data for evaluation/joining only. The
                # language model learns human-readable location labels, not numeric IDs.
                "location": model_location,
                "dwell_time_min": dwell_time,
            }
        )
    return targets


def build_student2_examples(
    demographics_path: Path,
    activity_path: Path,
    spatial_path: Path,
    teacher_paths: Iterable[Path],
    allow_teacher_subset: bool = False,
    location_whitelist_path: Path = DEFAULT_LOCATION_WHITELIST,
) -> list[dict[str, Any]]:
    demographics = read_jsonl_by_id(Path(demographics_path))
    activities = read_jsonl_by_id(Path(activity_path))
    spatial = read_jsonl_by_id(Path(spatial_path))
    teachers = load_teacher_rows(teacher_paths)
    if allow_teacher_subset:
        teacher_ids = set(teachers)
        for name, rows in {
            "demographics": demographics,
            "activities": activities,
            "spatial": spatial,
        }.items():
            missing = sorted(teacher_ids - set(rows))[:5]
            if missing:
                raise ValueError(f"teacher person_ids missing from {name}: {missing}")
        person_ids = sorted(teacher_ids, key=lambda value: (len(value), value))
    else:
        person_ids = require_identical_ids(
            {
                "demographics": demographics,
                "activities": activities,
                "spatial": spatial,
                "teachers": teachers,
            }
        )

    location_whitelist = load_location_whitelist(location_whitelist_path)
    labels_by_code: dict[str, tuple[str, str | None, str | None]] = {}
    for spatial_row in spatial.values():
        for item in spatial_row.get("activities") or []:
            location = item.get("location") or {}
            code = str(location.get("administrative_dong_code") or "")
            name = location.get("administrative_dong_name")
            if code and name:
                labels_by_code[code] = (name, location.get("gu"), location.get("sido"))

    examples: list[dict[str, Any]] = []
    for person_id in person_ids:
        sequence = activities[person_id].get("activity_sequence")
        spatial_sequence = [item.get("activity_type") for item in spatial[person_id]["activities"]]
        if not isinstance(sequence, list) or not sequence or sequence != spatial_sequence:
            raise ValueError(f"person_id={person_id}: activity sequence mismatch")
        teacher = teachers[person_id]
        user = {
            "demographics": demographics[person_id]["demographics"],
            "activity_sequence": sequence,
            "persona_bundle": persona_bundle(teacher),
            "reasoning1": teacher["reasoning1"],
        }
        target = {
            "location_dwell_by_activity": _location_dwell_targets(
                person_id, spatial[person_id], labels_by_code, location_whitelist
            ),
            "mode_preference": spatial[person_id].get("mode_preference", []),
            "reasoning2": teacher["reasoning2"],
        }
        examples.append(
            {
                "person_id": person_id,
                "messages": [
                    {"role": "user", "content": compact_json(user)},
                    {"role": "assistant", "content": compact_json(target)},
                ],
            }
        )
    return examples
