#!/usr/bin/env python3
"""Rebuild the KTDB ground truth (v2) and re-split it train / validation / test.

Changes from the v1 ground truth (upstream prepare_ktdb_training_data.py +
scripts/build_trip_mode_data.py):

1. Day boundary. KTDB clocks run past 24 h (e.g. 26:35). v1 took clock
   minutes modulo 1440 from the 03:00 day origin, so a time after 03:00 the
   next day wrapped to the early morning (negative dwell; such train people
   were dropped, test people kept). v2 unwraps the clock in trip order and
   clips the day to [0, 1440]: activities that would start at or after 03:00
   the next day are removed together with the trip into them, and the stay
   before them ends at 1440. Times before 03:00 on the survey day clip to 0.
   Only timelines whose times run backwards are invalid (and excluded).
2. Short round trips. A stop X with dwell <= --merge-max-dwell minutes whose
   previous and next activities have the same type and administrative dong
   (A -> X -> A) is folded into one A stay, for every activity type. The A
   stay runs from the first A start to the second A end, so the two trips and
   the stop are absorbed into its dwell. Applied repeatedly.
3. Split. train / validation / test = 70 / 10 / 20. Teacher pseudo-labels
   exist only for the v1 train people, so train and validation are drawn from
   them; test = v1 test + people newly valid under (1) + a random draw from
   v1 train to reach 20%.

Writes <out>/{train,validation,test}/ground_truth_tripmode.jsonl and
demographics_with_residence.jsonl, <out>/split_manifest.csv, <out>/manifest.json
(counts) and <out>/teacher_relabel_ids.txt (the train/validation people whose
ground truth differs from v1 and who therefore need new Teacher labels).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import DATA_ROOT, SHARED_DATA_ROOT, TRIPCHAIN_1015_CODE  # noqa: E402
from tripchain.residence import augment_demographics  # noqa: E402
from tripchain.trip_modes import ktdb_stages, main_mode, mode_preference_from_trips  # noqa: E402

# KTDB helpers of the upstream preprocessing.
sys.path.insert(0, str(TRIPCHAIN_1015_CODE / "scripts"))
from ktdb_admin_registry import load_admin_registry  # noqa: E402
from prepare_ktdb_training_data import (  # noqa: E402
    admin_dong,
    classify_activity,
    clean,
    demographics,
    home_dong,
    integer,
)

RAW = SHARED_DATA_ROOT / "KTDB_2021"
ADMIN_WORKBOOK = SHARED_DATA_ROOT / "행정동코드_20210726(말소코드포함).xlsx"
V1 = DATA_ROOT / "processed"
DAY = 1440


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def raw_minute(hour: object, minute: object) -> int | None:
    """Minutes after the 03:00 day origin, without wrapping (26:35 -> 1415, 27:20 -> 1460)."""
    h, m = integer(hour), integer(minute)
    if h is None or m is None or not 0 <= m < 60:
        return None
    return (h - 3) * 60 + m


def _location_key(location: dict) -> tuple:
    return (
        location.get("sido"),
        location.get("gu"),
        location.get("administrative_dong_code") or location.get("administrative_dong_name"),
    )


def build_day(person: pd.Series, rows: list[pd.Series], registry: dict) -> dict:
    """Activities (start, dwell) and per-trip main modes, clipped to [0, DAY]."""
    pid = clean(person["idx"])
    home = home_dong(person, registry)
    if not rows:
        return {
            "person_id": pid,
            "timeline_valid": True,
            "clipped": False,
            "trip_modes": [],
            "trip_rows": [],
            "activities": [{"activity_type": "home", "location": home, "start_time_min": 0, "dwell_time_min": DAY}],
        }
    departures = [raw_minute(r.get("TP3_1"), r.get("TP3_2")) for r in rows]
    arrivals = [raw_minute(r.get("TP4_1"), r.get("TP4_2")) for r in rows]
    clock = [value for pair in zip(departures, arrivals) for value in pair]
    valid = None not in clock and all(b >= a for a, b in zip(clock, clock[1:]))
    if not valid:
        return {"person_id": pid, "timeline_valid": False}
    clipped = clock[0] < 0 or clock[-1] >= DAY

    first = rows[0]
    origin_type = classify_activity("", "", first.get("sTP1"))
    activities = [
        {
            "activity_type": origin_type,
            "location": home if origin_type == "home" else admin_dong(first, "sTP1", registry),
            "start_time_min": 0,
        }
    ]
    modes, kept_rows = [], []
    for row, arrival in zip(rows, arrivals):
        start = min(max(arrival, 0), DAY)
        if start >= DAY:
            break  # this activity (and its trip) starts after the day ends
        kind = classify_activity(row.get("TP2"), row.get("TP2_a"), row.get("TP1"))
        activities.append(
            {
                "activity_type": kind,
                "location": home if kind == "home" else admin_dong(row, "TP1", registry),
                "start_time_min": start,
            }
        )
        modes.append(main_mode(ktdb_stages(row)))
        kept_rows.append(row)
    # Dwell runs to the next trip's departure; the last stay runs to the day end.
    for i, activity in enumerate(activities):
        end = min(max(departures[i], 0), DAY) if i < len(activities) - 1 else DAY
        activity["dwell_time_min"] = end - activity["start_time_min"]
    return {
        "person_id": pid,
        "timeline_valid": True,
        "clipped": clipped,
        "activities": activities,
        "trip_modes": modes,
        "trip_rows": kept_rows,
    }


def merge_round_trips(day: dict, max_dwell: float) -> int:
    """Fold A -> X (dwell <= max_dwell) -> A into one A stay; returns merges done."""
    acts, modes, merges, i = day["activities"], day["trip_modes"], 0, 1
    while i < len(acts) - 1:
        before, stop, after = acts[i - 1], acts[i], acts[i + 1]
        if (
            stop["dwell_time_min"] <= max_dwell
            and before["activity_type"] == after["activity_type"]
            and _location_key(before["location"]) == _location_key(after["location"])
        ):
            merged = dict(before)
            merged["dwell_time_min"] = after["start_time_min"] + after["dwell_time_min"] - before["start_time_min"]
            acts[i - 1 : i + 2] = [merged]
            del modes[i - 1 : i + 1]
            del day["trip_rows"][i - 1 : i + 1]
            merges += 1
            i = max(1, i - 1)
            continue
        i += 1
    return merges


def finish(day: dict) -> dict:
    """The ground-truth record of a built (and merged) day."""
    activities = [{"activity_index": i, **activity} for i, activity in enumerate(day["activities"])]
    return {
        "person_id": day["person_id"],
        "activity_sequence": [a["activity_type"] for a in activities],
        "activities": activities,
        "mode_preference": mode_preference_from_trips(day["trip_modes"]),
        "timeline_valid": True,
        "trip_modes": day["trip_modes"],
    }


def same_truth(new: dict, old: dict | None) -> bool:
    """True when a person's v2 day equals their v1 day (activities, places, times, trip modes)."""
    if old is None:
        return False
    strip = lambda row: [
        (a["activity_type"], _location_key(a.get("location") or {}), a["start_time_min"], a["dwell_time_min"])
        for a in row["activities"]
    ]
    return strip(new) == strip(old) and new["trip_modes"] == old.get("trip_modes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "processed_v2")
    parser.add_argument("--merge-max-dwell", type=float, default=10.0)
    parser.add_argument("--fractions", default="0.7,0.1,0.2")
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    f_train, f_val, f_test = (float(x) for x in args.fractions.split(","))
    assert abs(f_train + f_val + f_test - 1) < 1e-9

    registry = load_admin_registry(ADMIN_WORKBOOK)
    people = pd.read_csv(RAW / "①개인특성.csv", encoding="cp949", dtype=str, low_memory=False)
    people = people[people["SQ1_6"].map(clean).eq("서울특별시")].copy()
    people["idx"] = people["idx"].map(clean)
    people = people.sort_values("idx").reset_index(drop=True)
    trips = pd.read_csv(RAW / "②이동특성.csv", encoding="cp949", dtype=str, low_memory=False)
    trips["idx"] = trips["idx"].map(clean)
    trips = trips[trips["idx"].isin(set(people["idx"]))].copy()
    trips = trips[trips["TP2"].map(clean).ne("")]
    trips["_seq"] = pd.to_numeric(trips["th_seq"], errors="coerce")
    trips = trips.sort_values(["idx", "_seq"], kind="stable")
    grouped = {pid: [row for _, row in g.iterrows()] for pid, g in trips.groupby("idx", sort=False)}

    # Household car propagation, as in the upstream prepare_ktdb_training_data.main.
    pid_to_fid = trips.groupby("idx", sort=False)["fid"].first().map(clean).to_dict()
    fid_car_values: dict[str, list[bool]] = {}
    for _, person in people.iterrows():
        fid = pid_to_fid.get(person["idx"], "")
        car, van = integer(person.get("Q4_1")), integer(person.get("Q4_2"))
        if fid and (car is not None or van is not None):
            fid_car_values.setdefault(fid, []).append((car or 0) + (van or 0) > 0)
    fid_car = {fid: any(values) for fid, values in fid_car_values.items()}

    v1_truth = {
        str(r["person_id"]): r
        for split in ("train", "test")
        for r in read_rows(V1 / split / "ground_truth_tripmode.jsonl")
    }
    v1_demo = {
        str(r["person_id"]): r
        for split in ("train", "test")
        for r in read_rows(V1 / split / "demographics_with_residence.jsonl")
    }
    v1_train = [str(r["person_id"]) for r in read_rows(V1 / "train/demographics_with_residence.jsonl")]
    v1_test = [str(r["person_id"]) for r in read_rows(V1 / "test/demographics_with_residence.jsonl")]

    truths, demos, stats = {}, {}, Counter()
    for _, person in people.iterrows():
        pid = person["idx"]
        day = build_day(person, grouped.get(pid, []), registry)
        if not day["timeline_valid"]:
            stats["invalid_unordered"] += 1
            continue
        stats["clipped_people"] += day["clipped"]
        merges = merge_round_trips(day, args.merge_max_dwell)
        stats["merges"] += merges
        stats["merged_people"] += merges > 0
        stats["home_only_after_merge"] += merges > 0 and len(day["activities"]) == 1
        truths[pid] = finish(day)
        if pid in v1_demo:
            demos[pid] = v1_demo[pid]
        else:
            record = {
                "person_id": pid,
                "demographics": demographics(person, household_car_override=fid_car.get(pid_to_fid.get(pid, ""))),
            }
            record["demographics"] = augment_demographics(record["demographics"], home_dong(person, registry))
            demos[pid] = record

    # Split: train/validation only from v1 train (Teacher labels exist there).
    total = len(truths)
    n_test, n_val = int(round(total * f_test)), int(round(total * f_val))
    rng = np.random.default_rng(args.seed)
    labeled = [pid for pid in v1_train if pid in truths]
    rng.shuffle(labeled)
    fixed_test = [pid for pid in v1_test if pid in truths] + sorted(set(truths) - set(v1_train) - set(v1_test))
    extra = n_test - len(fixed_test)
    val_ids = set(labeled[extra : extra + n_val])
    train_ids = set(labeled[extra + n_val :])
    split = {pid: ("train" if pid in train_ids else "validation" if pid in val_ids else "test") for pid in truths}

    order = list(truths)  # KTDB idx order
    counts = {}
    for name in ("train", "validation", "test"):
        ids = [pid for pid in order if split[pid] == name]
        counts[name] = write_rows(args.out_dir / name / "ground_truth_tripmode.jsonl", (truths[p] for p in ids))
        write_rows(args.out_dir / name / "demographics_with_residence.jsonl", (demos[p] for p in ids))
    pd.DataFrame({"person_id": order, "split": [split[p] for p in order]}).to_csv(
        args.out_dir / "split_manifest.csv", index=False
    )

    changed = sorted(pid for pid in truths if not same_truth(truths[pid], v1_truth.get(pid)))
    relabel = [pid for pid in changed if split[pid] in ("train", "validation")]
    (args.out_dir / "teacher_relabel_ids.txt").write_text("\n".join(relabel) + "\n")
    trips_total = sum(len(t["trip_modes"]) for t in truths.values())
    manifest = {
        "rules": {
            "day": "unwrapped KTDB clock clipped to [0, 1440] from 03:00",
            "merge": f"A -> X(dwell <= {args.merge_max_dwell:g}) -> A, same type and dong, all types",
            "split": f"{args.fractions}; train/validation from v1 train; test = v1 test + new + draw",
            "seed": args.seed,
        },
        "people": total,
        "split_counts": counts,
        "trips": trips_total,
        "walk_share": sum(m == "walk" for t in truths.values() for m in t["trip_modes"]) / trips_total,
        "newly_included_people": len(set(truths) - set(v1_train) - set(v1_test)),
        "changed_vs_v1": len(changed),
        "teacher_relabel_train_validation": len(relabel),
        **stats,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
