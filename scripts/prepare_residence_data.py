#!/usr/bin/env python3
"""Add the observed home location to the demographics both Students are conditioned on.

The residence (sido, sigungu, administrative dong) comes from the KTDB
personal-characteristics survey fields SQ1_5..SQ1_8, not from an inferred home
activity. Reads the v1 train/test demographics written by the KTDB
preprocessing (scripts/prepare_ktdb_training_data.py)
and writes <output-root>/{train,test}/demographics_with_residence.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import DATA_ROOT, SHARED_DATA_ROOT, TRIPCHAIN_1015_DATA  # noqa: E402
from tripchain.residence import augment_demographics, validate_residence_demographics  # noqa: E402

SOURCE = TRIPCHAIN_1015_DATA
RAW_PEOPLE = SHARED_DATA_ROOT / "KTDB_2021/①개인특성.csv"
# Shared administrative-code workbook (also ktdb_admin_registry.DEFAULT_WORKBOOK).
ADMIN_WORKBOOK = SHARED_DATA_ROOT / "행정동코드_20210726(말소코드포함).xlsx"

# KTDB helpers of the preprocessing (scripts/, also importable when this file is imported).
sys.path.insert(0, str(ROOT / "scripts"))
from ktdb_admin_registry import load_admin_registry  # noqa: E402
from prepare_ktdb_training_data import clean, home_dong  # noqa: E402


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_residences() -> dict[str, dict[str, str]]:
    """person_id -> home location, for every person of the raw KTDB survey."""
    import pandas as pd

    people = pd.read_csv(
        RAW_PEOPLE,
        encoding="cp949",
        dtype=str,
        usecols=["idx", "SQ1_5", "SQ1_6", "SQ1_7", "SQ1_8"],
        low_memory=False,
    )
    registry = load_admin_registry(ADMIN_WORKBOOK)
    residences: dict[str, dict[str, str]] = {}
    for _, person in people.iterrows():
        person_id = clean(person["idx"])
        if person_id:
            residences[person_id] = home_dong(person, registry)
    return residences


def augment_demographic_file(demographics_path: Path, residences: dict[str, dict[str, str]], output: Path) -> int:
    """Write `demographics_path` with a residence added to every person; returns the row count."""
    demographics = read_rows(demographics_path)
    result = []
    for row in demographics:
        person_id = str(row["person_id"])
        if person_id not in residences:
            raise ValueError(f"person_id={person_id}: missing survey residence")
        value = dict(row)
        value["demographics"] = augment_demographics(row["demographics"], residences[person_id])
        validate_residence_demographics(value["demographics"])
        result.append(value)
    write_rows(output, result)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "processed")
    args = parser.parse_args()
    output = args.output_root.resolve()
    residences = load_residences()
    summary = {
        "train_demographics": augment_demographic_file(
            SOURCE / "processed/train/demographics.jsonl",
            residences,
            output / "train/demographics_with_residence.jsonl",
        ),
        "test_demographics": augment_demographic_file(
            SOURCE / "processed/test/demographics.jsonl",
            residences,
            output / "test/demographics_with_residence.jsonl",
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
