"""KTDB two-stage teacher data loading, XML prompting, and validation."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tripchain.paths import TRIPCHAIN_1015_DATA as PARENT_DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
KTDB_PROCESSED = PARENT_DATA_ROOT / "processed" / "train"
DEFAULT_DEMOGRAPHICS = KTDB_PROCESSED / "demographics.jsonl"
DEFAULT_ACTIVITY = KTDB_PROCESSED / "activity_sequence.jsonl"
DEFAULT_SPATIAL = KTDB_PROCESSED / "location_dwell_mode_preference.jsonl"
PROMPT_SOURCE = "prompts/rationalizer_prompt.md"
PROMPT_SCHEMA_VERSION = "ktdb-two-stage-xml-v10-vehicle-roles"


def _prompt_block(heading: str) -> str:
    text = (REPO_ROOT / PROMPT_SOURCE).read_text(encoding="utf-8")
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n\s*```text\n(.*?)\n```",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"Could not find {heading!r} text block in {PROMPT_SOURCE}")
    return match.group(1).strip()


SYSTEM_PROMPT = _prompt_block("System prompt")


class TeacherInference(BaseModel):
    """Parsed model response. Extra output fields are never accepted."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    professional_persona: str = Field(min_length=10)
    family_persona: str = Field(min_length=10)
    persona: str = Field(min_length=10)
    hobbies_and_interests_list: list[str] = Field(min_length=1, max_length=5)
    reasoning1: str = Field(min_length=10)
    reasoning2: str = Field(min_length=10)


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            person_id = str(row["person_id"])
            if person_id in rows:
                raise ValueError(f"duplicate person_id={person_id} in {path}:{line_number}")
            rows[person_id] = row
    return rows


def load_teacher_records(
    demographics_path: Path = DEFAULT_DEMOGRAPHICS,
    activity_path: Path = DEFAULT_ACTIVITY,
    spatial_path: Path = DEFAULT_SPATIAL,
) -> list[dict[str, Any]]:
    """Join the three clean KTDB training inputs on their shared person IDs."""

    demographics = _read_jsonl_by_id(Path(demographics_path))
    activity = _read_jsonl_by_id(Path(activity_path))
    spatial = _read_jsonl_by_id(Path(spatial_path))
    shared = sorted(set(demographics) & set(activity) & set(spatial), key=lambda x: (len(x), x))
    if not shared:
        raise ValueError("No person_id is shared by all three KTDB teacher inputs")

    records: list[dict[str, Any]] = []
    for person_id in shared:
        sequence = activity[person_id]["activity_sequence"]
        source_activities = spatial[person_id]["activities"]
        activities = []
        for source_item in source_activities:
            # Per-trip mode legs are intentionally not part of the two-stage
            # framework. The teacher receives only spatial/duration evidence
            # plus the separate person-level mode_preference ranking.
            item = {
                key: source_item[key]
                for key in (
                    "activity_index",
                    "activity_type",
                    "location",
                    "start_time_min",
                    "dwell_time_min",
                )
                if key in source_item
            }
            source_location = dict(item.get("location") or {})
            if source_location.get("sido") == "Seoul":
                model_location = {
                    "scope": "seoul",
                    "gu": source_location.get("gu"),
                    "administrative_dong_name": source_location.get("administrative_dong_name"),
                }
            else:
                model_location = {
                    "scope": "external",
                    "sido": source_location.get("sido"),
                    "sigungu": source_location.get("gu"),
                }
            item["location"] = model_location
            activities.append(item)
        for item in activities:
            location = item.get("location") or {}
            region_text = " ".join(
                str(location.get(key) or "") for key in ("administrative_dong_name", "gu", "sido", "sigungu")
            )
            if re.search(r"[가-힣\u3400-\u9fff]", region_text):
                raise ValueError(f"non-English location name for person_id={person_id}: {region_text}")
        spatial_sequence = [item["activity_type"] for item in activities]
        if sequence != spatial_sequence:
            raise ValueError(f"activity sequence mismatch for person_id={person_id}")
        records.append(
            {
                "person_id": person_id,
                "demographics": demographics[person_id]["demographics"],
                "activity_sequence": sequence,
                "activities": activities,
                "mode_preference": spatial[person_id]["mode_preference"],
            }
        )
    return records


def build_user_prompt(record: dict[str, Any]) -> str:
    demographics = record["demographics"]
    occupation = demographics.get("occupation", "unknown")
    demo_lines = [
        f"- Gender: {demographics.get('gender', 'unknown')}",
        f"- Age: {demographics.get('age', 'unknown')}",
        f"- Occupation (survey): {occupation}",
        f"- Can drive: {demographics.get('can_drive', 'unknown')}",
    ]
    compact = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    model_activities = []
    for source_item in record["activities"]:
        item = dict(source_item)
        location = dict(item.get("location") or {})
        location.pop("administrative_dong_code", None)
        item["location"] = location
        model_activities.append(item)
    return (
        "## Respondent\n"
        + "\n".join(demo_lines)
        + "\n\n## Observed activity types and sequence (evidence to explain)\n"
        + compact(record["activity_sequence"])
        + "\n\n## Observed locations and dwell times (evidence to explain)\n"
        + compact(model_activities)
        + "\n\n## Observed person-level mode preference (most-used mode first)\n"
        + compact(record["mode_preference"])
        + "\n\nAbduce the latent persona bundle that explains this day, then write the six XML blocks."
    )


_XML_RESPONSE = re.compile(
    r"\s*<professional_persona>\s*(?P<professional_persona>.*?)\s*</professional_persona>\s*"
    r"<family_persona>\s*(?P<family_persona>.*?)\s*</family_persona>\s*"
    r"<persona>\s*(?P<persona>.*?)\s*</persona>\s*"
    r"<hobbies_and_interests_list>\s*(?P<hobbies_and_interests_list>.*?)\s*</hobbies_and_interests_list>\s*"
    r"<reasoning1>\s*(?P<reasoning1>.*?)\s*</reasoning1>\s*"
    r"<reasoning2>\s*(?P<reasoning2>.*?)\s*</reasoning2>\s*",
    flags=re.DOTALL,
)
_CJK = re.compile(r"[가-힣\u3040-\u30ff\u3400-\u9fff]")
_LIST_ITEM = re.compile(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+")


def parse_teacher_output(raw: str) -> TeacherInference:
    """Parse the six XML blocks used by the expanded teacher schema."""

    match = _XML_RESPONSE.fullmatch(raw)
    if not match:
        raise ValueError("response must contain exactly the six required XML blocks")
    values = {key: html.unescape(value.strip()) for key, value in match.groupdict().items()}
    try:
        hobbies = json.loads(values.pop("hobbies_and_interests_list"))
    except json.JSONDecodeError as exc:
        raise ValueError("hobbies_and_interests_list must contain a JSON array") from exc
    if not isinstance(hobbies, list) or not all(isinstance(item, str) for item in hobbies):
        raise ValueError("hobbies_and_interests_list must be a JSON string array")
    return TeacherInference(hobbies_and_interests_list=hobbies, **values)


def validate_teacher_content(record: dict[str, Any], output: TeacherInference) -> None:
    """Reject malformed, non-English, or excessively long teacher annotations."""

    values = (
        output.professional_persona,
        output.family_persona,
        output.persona,
        *output.hobbies_and_interests_list,
        output.reasoning1,
        output.reasoning2,
    )
    if _CJK.search(" ".join(values)):
        raise ValueError("teacher response contains Hangul, Kana, or Hanzi")
    limits = (
        ("professional_persona", output.professional_persona, 70),
        ("family_persona", output.family_persona, 60),
        ("persona", output.persona, 80),
        ("reasoning1", output.reasoning1, 120),
        ("reasoning2", output.reasoning2, 130),
    )
    for name, value, limit in limits:
        if len(value.split()) > limit:
            raise ValueError(f"{name} exceeds {limit} words")
    for hobby in output.hobbies_and_interests_list:
        if not hobby.strip():
            raise ValueError("hobbies_and_interests_list contains an empty item")
        if len(hobby.split()) > 8:
            raise ValueError("a hobbies_and_interests_list item exceeds 8 words")
    reasoning = f"{output.reasoning1} {output.reasoning2}"
    if len(reasoning.split()) > 250:
        raise ValueError("reasoning1 and reasoning2 exceed 250 words in total")
    if _LIST_ITEM.search(output.reasoning1) or _LIST_ITEM.search(output.reasoning2):
        raise ValueError("reasoning1 and reasoning2 must be flowing prose, not lists")
    if not record.get("activity_sequence"):
        raise ValueError("source record has an empty activity sequence")
