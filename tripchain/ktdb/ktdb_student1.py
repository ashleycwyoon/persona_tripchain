"""Build KTDB Student 1 SFT examples from completed teacher annotations."""

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

__all__ = [
    "DEFAULT_TEACHER_SHARDS",
    "build_student1_examples",
    "deterministic_split",
    "require_completed_teacher_shards",
]


def build_student1_examples(
    demographics_path: Path,
    activity_path: Path,
    teacher_paths: Iterable[Path],
    allow_teacher_subset: bool = False,
) -> list[dict[str, Any]]:
    demographics = read_jsonl_by_id(Path(demographics_path))
    activities = read_jsonl_by_id(Path(activity_path))
    teachers = load_teacher_rows(teacher_paths)
    if allow_teacher_subset:
        teacher_ids = set(teachers)
        for name, rows in {"demographics": demographics, "activities": activities}.items():
            missing = sorted(teacher_ids - set(rows))[:5]
            if missing:
                raise ValueError(f"teacher person_ids missing from {name}: {missing}")
        person_ids = sorted(teacher_ids, key=lambda value: (len(value), value))
    else:
        person_ids = require_identical_ids(
            {"demographics": demographics, "activities": activities, "teachers": teachers}
        )

    examples: list[dict[str, Any]] = []
    for person_id in person_ids:
        sequence = activities[person_id].get("activity_sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"person_id={person_id}: empty activity_sequence")
        teacher = teachers[person_id]
        user = {"demographics": demographics[person_id]["demographics"]}
        target = {
            "activity_sequence": sequence,
            "persona_bundle": persona_bundle(teacher),
            "reasoning1": teacher["reasoning1"],
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
