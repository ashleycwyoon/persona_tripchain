"""Shared helpers: config loading, JSONL I/O and the Students' JSON output orders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import expand_config, locate


# Repository root; repo-relative config and data paths are resolved against it.
ROOT = Path(__file__).resolve().parents[1]

# Generation order of each Student's JSON answer. Reasons come before the
# decisions they condition: Student1 writes persona and reasoning1 before the
# activity sequence, Student2 writes reasoning2 before locations/dwell and mode
# preference. SFT targets and GRPO traces both serialize in this order.
STUDENT1_OUTPUT_ORDER = ("persona_bundle", "reasoning1", "activity_sequence")
STUDENT2_OUTPUT_ORDER = ("reasoning2", "location_dwell_by_activity", "mode_preference")
# Selected per run by `inference.output_order`; it must match how the adapters
# were trained. decision_first is the order of the original upstream SFT.
OUTPUT_ORDERS = {
    "reason_first": (STUDENT1_OUTPUT_ORDER, STUDENT2_OUTPUT_ORDER),
    "decision_first": (
        ("activity_sequence", "persona_bundle", "reasoning1"),
        ("location_dwell_by_activity", "mode_preference", "reasoning2"),
    ),
    # Student2 names the main mode of every trip (trip i: activity i -> i+1)
    # instead of a person-level mode ranking; see tripchain/trip_modes.py.
    "reason_first_tripmode": (
        STUDENT1_OUTPUT_ORDER,
        ("reasoning2", "location_dwell_by_activity", "trip_modes"),
    ),
}


def output_orders(config: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(Student1 order, Student2 order) selected by `inference.output_order`."""
    name = (config.get("inference") or {}).get("output_order", "reason_first")
    if name not in OUTPUT_ORDERS:
        raise ValueError(f"unknown inference.output_order: {name}")
    return OUTPUT_ORDERS[name]


def ordered_output(value: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    """`value` with exactly the keys of `order`, in that order."""
    if set(value) != set(order):
        raise ValueError(f"output keys {sorted(value)} do not match {list(order)}")
    return {key: value[key] for key in order}


def resolve(path: str | Path) -> Path:
    """Absolute location of a repo-relative path (see paths.locate)."""
    return locate(path, ROOT)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand its ${...} path variables."""
    return expand_config(yaml.safe_load(resolve(path).read_text(encoding="utf-8")))


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with resolve(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def rows_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    """JSONL rows keyed by person_id, in file order; a repeated person is an error."""
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        person_id = str(row["person_id"])
        if person_id in rows:
            raise ValueError(f"duplicate person_id={person_id}: {path}")
        rows[person_id] = row
    return rows


def completed_ids(path: Path, terminal_statuses: set[str] | None = None) -> set[str]:
    """People already written to `path` with a terminal status (for --resume)."""
    if not path.exists():
        return set()
    terminal_statuses = terminal_statuses or {"ok"}
    return {
        str(row["person_id"])
        for row in read_jsonl(path)
        if row.get("status", "ok") in terminal_statuses and row.get("person_id") is not None
    }


def clean_json(text: str) -> dict[str, Any]:
    """Parse a model response as one JSON object, tolerating a ``` code fence."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value
