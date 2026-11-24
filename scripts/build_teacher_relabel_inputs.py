#!/usr/bin/env python3
"""Teacher inputs (v1 format) for the people whose v2 ground truth changed.

The KTDB Teacher (scripts/run_teacher_2gpu.py)
reads three files joined on person_id: demographics (English, without
residence), activity_sequence and location_dwell_mode_preference. Its
mode_preference is the v1 stage-level ranking (upstream ranked_mode_preference),
so it is recomputed here from the KTDB trips that survive the v2 day clipping
and round-trip merging.

Reads <v2>/teacher_relabel_ids.txt from scripts/rebuild_ground_truth_v2.py and
writes <v2>/teacher_relabel/{demographics,activity_sequence,
location_dwell_mode_preference}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

# rebuild_ground_truth_v2 also puts the upstream scripts directory on sys.path.
import rebuild_ground_truth_v2 as v2  # noqa: E402
from prepare_ktdb_training_data import ranked_mode_preference  # noqa: E402
from tripchain.paths import DATA_ROOT, TRIPCHAIN_1015_DATA  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-dir", type=Path, default=DATA_ROOT / "processed_v2")
    parser.add_argument("--merge-max-dwell", type=float, default=10.0)
    args = parser.parse_args()

    ids = [line.strip() for line in (args.v2_dir / "teacher_relabel_ids.txt").read_text().splitlines() if line.strip()]
    wanted = set(ids)
    demographics = {
        str(r["person_id"]): r for r in v2.read_rows(TRIPCHAIN_1015_DATA / "processed/train/demographics.jsonl")
    }
    missing = wanted - set(demographics)
    if missing:
        raise SystemExit(f"{len(missing)} relabel people lack v1 Teacher demographics, e.g. {sorted(missing)[:5]}")
    truth = {}
    for split in ("train", "validation"):
        for row in v2.read_rows(args.v2_dir / split / "ground_truth_tripmode.jsonl"):
            if row["person_id"] in wanted:
                truth[row["person_id"]] = row

    registry = v2.load_admin_registry(v2.ADMIN_WORKBOOK)
    people = pd.read_csv(v2.RAW / "①개인특성.csv", encoding="cp949", dtype=str, low_memory=False)
    people["idx"] = people["idx"].map(v2.clean)
    people = people[people["idx"].isin(wanted)]
    trips = pd.read_csv(v2.RAW / "②이동특성.csv", encoding="cp949", dtype=str, low_memory=False)
    trips["idx"] = trips["idx"].map(v2.clean)
    trips = trips[trips["idx"].isin(wanted) & trips["TP2"].map(v2.clean).ne("")]
    trips["_seq"] = pd.to_numeric(trips["th_seq"], errors="coerce")
    trips = trips.sort_values(["idx", "_seq"], kind="stable")
    grouped = {pid: [row for _, row in g.iterrows()] for pid, g in trips.groupby("idx", sort=False)}

    out = args.v2_dir / "teacher_relabel"
    demo_rows, sequence_rows, spatial_rows = [], [], []
    for _, person in people.iterrows():
        pid = person["idx"]
        day = v2.build_day(person, grouped.get(pid, []), registry)
        v2.merge_round_trips(day, args.merge_max_dwell)
        row = v2.finish(day)
        # Same day as the written v2 ground truth (guards against drift).
        if row["activities"] != truth[pid]["activities"]:
            raise SystemExit(f"person_id={pid}: rebuilt day differs from processed_v2")
        demo_rows.append(demographics[pid])
        sequence_rows.append({"person_id": pid, "activity_sequence": row["activity_sequence"]})
        spatial_rows.append(
            {
                "person_id": pid,
                "activities": row["activities"],
                "mode_preference": ranked_mode_preference(day["trip_rows"]),
                "timeline_valid": True,
            }
        )
    counts = {
        name: v2.write_rows(out / f"{name}.jsonl", rows)
        for name, rows in (
            ("demographics", demo_rows),
            ("activity_sequence", sequence_rows),
            ("location_dwell_mode_preference", spatial_rows),
        )
    }
    print(json.dumps({"relabel_ids": len(ids), **counts}, indent=2))


if __name__ == "__main__":
    main()
