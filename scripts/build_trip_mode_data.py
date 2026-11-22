#!/usr/bin/env python3
"""Add per-trip main modes to the v1 ground truth.

The upstream targets rank a person's modes by counting every KTDB stage, so
the walk to and from each bus or subway makes walking the top mode of most
transit riders. Here every trip keeps its own main mode (tripchain/trip_modes.py):
`trip_modes[i]` is the mode of the trip from activity i to activity i+1.
Ground truth `mode_preference` is re-derived from these trip modes.

Trips are read exactly as the upstream preprocessing builds activities (rows
with a purpose, ordered by th_seq), so trip i arrives at activity i+1; a count
mismatch is an error. Writes <output-root>/{train,test}/ground_truth_tripmode.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import DATA_ROOT, SHARED_DATA_ROOT, TRIPCHAIN_1015_DATA  # noqa: E402
from tripchain.trip_modes import ktdb_stages, main_mode, mode_preference_from_trips  # noqa: E402

RAW_TRIPS = SHARED_DATA_ROOT / "KTDB_2021/②이동특성.csv"
TEST_TRUTH = TRIPCHAIN_1015_DATA / "processed/test/ground_truth_reference.jsonl"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_trip_modes(person_ids: set[str]) -> dict[str, list[str]]:
    """person_id -> main mode of each of that person's KTDB trips, in trip order."""
    import pandas as pd

    columns = [
        "idx",
        "th_seq",
        "TP2",
        *[f"TP5_{i}" for i in range(1, 11)],
        *[f"TP5_{i}_2_1" for i in range(1, 11)],
        *[f"TP5_{i}_t1" for i in range(1, 11)],
    ]
    trips = pd.read_csv(RAW_TRIPS, encoding="cp949", dtype=str, usecols=columns, low_memory=False)
    trips["idx"] = trips["idx"].map(lambda value: str(value).strip())
    trips = trips[trips["idx"].isin(person_ids)]
    trips = trips[trips["TP2"].map(lambda value: str(value).strip() not in {"", "nan"})].copy()
    trips["_seq"] = pd.to_numeric(trips["th_seq"], errors="coerce")
    trips = trips.sort_values(["idx", "_seq"], kind="stable")
    return {
        person_id: [main_mode(ktdb_stages(row)) for _, row in group.iterrows()]
        for person_id, group in trips.groupby("idx", sort=False)
    }


def with_trip_modes(row: dict, modes: dict[str, list[str]]) -> dict:
    person_id = str(row["person_id"])
    trip_modes = modes.get(person_id, [])
    if len(trip_modes) != len(row["activities"]) - 1:
        raise ValueError(f"person_id={person_id}: {len(trip_modes)} trips for {len(row['activities'])} activities")
    return {**row, "trip_modes": trip_modes, "mode_preference": mode_preference_from_trips(trip_modes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-truth", type=Path, default=DATA_ROOT / "processed/train/ground_truth_reference.jsonl")
    parser.add_argument("--test-truth", type=Path, default=TEST_TRUTH)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT / "processed")
    args = parser.parse_args()

    truths = {"train": read_rows(args.train_truth), "test": read_rows(args.test_truth)}
    ids = {str(row["person_id"]) for rows in truths.values() for row in rows}
    modes = load_trip_modes(ids)

    summary: dict[str, int] = {}
    for split, rows in truths.items():
        output = args.output_root / split / "ground_truth_tripmode.jsonl"
        write_rows(output, [with_trip_modes(row, modes) for row in rows])
        summary[f"{split}_truth"] = len(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
