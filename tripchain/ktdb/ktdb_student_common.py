"""Shared schema and provenance checks for the two KTDB student datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tripchain.ktdb.ktdb_two_stage_teacher import (
    PROMPT_SCHEMA_VERSION,
    PROMPT_SOURCE,
    TeacherInference,
)
from tripchain.paths import TEACHER_OUTPUT_ROOT as OUTPUT_ROOT

PERSONA_BUNDLE_KEYS = {
    "professional_persona",
    "family_persona",
    "persona",
    "hobbies_and_interests_list",
}
TEACHER_OUTPUT_KEYS = {"person_id", *PERSONA_BUNDLE_KEYS, "reasoning1", "reasoning2"}
DEFAULT_TEACHER_SHARDS = (
    OUTPUT_ROOT / "reports" / "ktdb_teacher_qwen32b_english_demographics_shard0.jsonl",
    OUTPUT_ROOT / "reports" / "ktdb_teacher_qwen32b_english_demographics_shard1.jsonl",
)


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                person_id = str(row["person_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if person_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate person_id={person_id}")
            rows[person_id] = row
    return rows


def load_teacher_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for path in paths:
        for person_id, row in read_jsonl_by_id(Path(path)).items():
            if person_id in combined:
                raise ValueError(f"duplicate teacher person_id={person_id} across shards")
            if set(row) != TEACHER_OUTPUT_KEYS:
                raise ValueError(
                    f"teacher person_id={person_id}: expected fields {sorted(TEACHER_OUTPUT_KEYS)}, got {sorted(row)}"
                )
            TeacherInference(**{key: value for key, value in row.items() if key != "person_id"})
            combined[person_id] = row
    if not combined:
        raise ValueError("teacher shards contain no successful records")
    return combined


def require_completed_teacher_shards(paths: Iterable[Path]) -> None:
    """Require complete v6 shards and verify each manifest against its data file."""

    for path in map(Path, paths):
        manifest_path = path.with_name(path.name + ".complete.json")
        if not path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"incomplete teacher shard: {path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid teacher completion manifest: {manifest_path}") from exc
        if manifest.get("prompt_source") != PROMPT_SOURCE:
            raise RuntimeError(f"teacher shard uses a different prompt source: {path}")
        if manifest.get("prompt_schema_version") != PROMPT_SCHEMA_VERSION:
            raise RuntimeError(f"teacher shard uses an incompatible prompt schema: {path}")
        expected = manifest.get("expected_records")
        successful = manifest.get("successful_records")
        unresolved = manifest.get("unresolved_records")
        if not isinstance(expected, int) or not isinstance(successful, int):
            raise RuntimeError(f"teacher manifest lacks record counts: {manifest_path}")
        if successful != expected or unresolved != 0:
            raise RuntimeError(
                f"incomplete teacher shard: {path} "
                f"(successful={successful}, expected={expected}, unresolved={unresolved})"
            )
        with path.open("rb") as handle:
            actual = sum(1 for line in handle if line.strip())
        if actual != successful:
            raise RuntimeError(f"teacher shard line count mismatch: {path} (lines={actual}, manifest={successful})")


def persona_bundle(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "professional_persona",
            "family_persona",
            "persona",
            "hobbies_and_interests_list",
        )
    }


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def deterministic_split(
    rows: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("at least two examples are required for a train/validation split")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['person_id']}".encode("utf-8")).digest(),
    )
    validation_size = max(1, min(len(rows) - 1, round(len(rows) * validation_fraction)))
    validation_ids = {row["person_id"] for row in ordered[:validation_size]}
    train = [row for row in rows if row["person_id"] not in validation_ids]
    validation = [row for row in rows if row["person_id"] in validation_ids]
    return train, validation


def require_identical_ids(named_rows: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    names = list(named_rows)
    reference_name = names[0]
    reference = set(named_rows[reference_name])
    for name in names[1:]:
        current = set(named_rows[name])
        if current != reference:
            missing = sorted(reference - current)[:5]
            extra = sorted(current - reference)[:5]
            raise ValueError(f"person_id mismatch: {name} vs {reference_name}; missing={missing}, extra={extra}")
    return sorted(reference, key=lambda value: (len(value), value))
