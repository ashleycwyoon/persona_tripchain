#!/usr/bin/env python3
"""Prepare the KTDB 2021 Seoul population for the two-stage training flow.

The split is made at person level before any training target is written.  All
output JSONL files are UTF-8 and contain one person per line.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripchain.paths import SHARED_DATA_ROOT, TRIPCHAIN_1015_DATA as DATA_ROOT  # noqa: E402

from korean_admin_english import english_location
from ktdb_admin_registry import (
    DEFAULT_WORKBOOK,
    build_model_whitelist,
    load_admin_registry,
    normalize_code,
)


OCCUPATION = {
    "1": "self_employed",
    "2": "sales_service",
    "3": "skilled_trade",
    "4": "general_labor",
    "5": "technician",
    "6": "office_worker",
    "7": "manager",
    "8": "professional",
    "9": "freelancer",
    "10": "simple_labor",
    "11": "agriculture_fishery",
}

MODE = {
    "1": "walk",
    "3": "bus",
    "4": "intercity_bus",
    "5": "other_bus",
    "7": "subway",
    "8": "rail",
    "9": "high_speed_rail",
    "10": "taxi",
    "11": "truck",
    "12": "bicycle",
    "13": "motorcycle",
    "14": "personal_mobility",
    "15": "mobility_aid",
    "16": "air",
    "17": "ferry",
    "97": "other",
}

CAR_ROLE = {
    # KTDB TP5_i_2_1: role in a passenger car/van (TP5_i == 2).
    "1": "car_driver",
    "2": "car_passenger",
}

PURPOSE = {
    "1": "home",
    "2": "work",
    "3": "work",
    "4": "school",
    "5": "school",
    "6": "work",
    "7": "shopping",
    "8": "leisure",
    "9": "eat_out",
    "10": "leisure",  # social visit is folded into the requested leisure class
    "11": "medical",
    "12": "escort_care",
}

PLACE = {"1": "home", "2": "work", "3": "school"}
ACTIVITY_TYPES = {
    "home",
    "work",
    "school",
    "leisure",
    "shopping",
    "medical",
    "escort_care",
    "eat_out",
    "other",
}


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def integer(value: object) -> int | None:
    value = clean(value)
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def clock_minute(hour: object, minute: object) -> int | None:
    h, m = integer(hour), integer(minute)
    if h is None or m is None or not 0 <= m < 60:
        return None
    return ((h - 3) * 60 + m) % 1440


def occupation(row: pd.Series) -> str | None:
    code = clean(row.get("R3_2"))
    if code in OCCUPATION:
        return OCCUPATION[code]
    if code == "97":
        # R3_2_a is an uncontrolled Korean free-text field with hundreds of
        # spellings.  It is not a model-safe categorical feature.  Preserve
        # the codebook meaning in English rather than leaking raw Korean into
        # Teacher/Student prompts.
        return "other"
    # Children/students legitimately skip the occupation question.
    student_code = clean(row.get("R2"))
    if student_code in {"0", "1", "2", "3", "4"}:
        return "student"
    # R3=3 means no regular workplace because the respondent does not work.
    if clean(row.get("R3")) == "3":
        return "not_working"
    return None


def english_yes_no(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def demographics(row: pd.Series, household_car_override: bool | None = None) -> dict:
    car = integer(row.get("Q4_1"))
    van = integer(row.get("Q4_2"))
    license_code = clean(row.get("R1"))
    household_known = car is not None or van is not None
    household_car = ((car or 0) + (van or 0) > 0) if household_known else household_car_override
    age = integer(row.get("SQ3_1"))
    has_license = True if license_code == "1" else False if license_code == "2" else None
    # R1 is not administered below driving age; this is a structural skip, not missingness.
    if has_license is None and age is not None and age < 18:
        has_license = False
    # Three-valued AND: either explicit "no" is sufficient to rule out car
    # driving; only two explicit "yes" values permit it. Otherwise unknown.
    if household_car is False or has_license is False:
        can_drive = False
    elif household_car is True and has_license is True:
        can_drive = True
    else:
        can_drive = None
    return {
        "gender": {"1": "male", "2": "female"}.get(clean(row.get("SQ2")), "unknown"),
        # Age is a measured quantity, not a coded category. Keep the actual
        # integer so the models can learn precise age effects.
        "age": age if age is not None else "unknown",
        "occupation": occupation(row) or "unknown",
        # Household vehicle ownership and licence status are used only to
        # derive this compact model feature; do not spend prompt tokens on the
        # two source variables.
        "can_drive": english_yes_no(can_drive),
    }


def classify_activity(purpose: object, purpose_text: object, place: object) -> str:
    purpose_code = clean(purpose)
    place_code = clean(place)
    text = clean(purpose_text).lower()

    # Explicit home has priority. Other explicit destination place types are
    # used together with purpose; a specific purpose wins over a generic place.
    if purpose_code == "1" or place_code == "1":
        return "home"
    if purpose_code in PURPOSE:
        return PURPOSE[purpose_code]
    if place_code in PLACE:
        return PLACE[place_code]

    keyword_rules = (
        (r"귀가|집으로|귀택", "home"),
        (r"출근|직장|업무|회사|근무", "work"),
        (r"등교|학교|학원|수업|공부|독서실", "school"),
        (r"쇼핑|구매|물건|장보기|마트|시장", "shopping"),
        (r"병원|의원|진료|치료|약국", "medical"),
        (r"태우|내려주|픽업|배웅|등원|하원|돌봄|간병", "escort_care"),
        (r"식사|회식|밥|점심|저녁|아침|카페|커피", "eat_out"),
        (r"여가|운동|관광|레저|산책|친지|친구|방문|종교|교회|성당|절", "leisure"),
    )
    for pattern, label in keyword_rules:
        if re.search(pattern, text):
            return label
    return "other"


def _canonical_location(code: object, fallback: dict, registry: dict[str, dict[str, str]]) -> dict:
    code = normalize_code(code) or None
    canonical = registry.get(code or "")
    if canonical is None and code:
        # A few survey records use newly split dong codes that are not present
        # in the 2021-07-26 snapshot.  Their first five digits still identify a
        # unique si/gun/gu in the supplied workbook.  Recover only that parent
        # region from the workbook and use the raw dong text as a label; never
        # expose the unmatched numeric code to the model.
        parent_candidates = {
            (item["sido_ko"], item["sigungu_ko"], item["sido"], item["sigungu"])
            for registry_code, item in registry.items()
            if registry_code[:5] == code[:5]
        }
        if len(parent_candidates) == 1:
            sido_ko, sigungu_ko, sido, sigungu = parent_candidates.pop()
            canonical = {
                "sido_ko": sido_ko,
                "sigungu_ko": sigungu_ko,
                "dong_ko": fallback.get("administrative_dong_name"),
                "sido": sido,
                "sigungu": sigungu,
                "administrative_dong_name": english_location(fallback).get("administrative_dong_name"),
            }
    if canonical:
        return {
            "administrative_dong_code": code,
            "administrative_dong_name": canonical["administrative_dong_name"],
            "gu": canonical["sigungu"],
            "sido": canonical["sido"],
        }
    # Missing codes remain representable, but no numeric value is ever exposed
    # as a model-facing location label.
    return english_location({"administrative_dong_code": code, **fallback})


def admin_dong(row: pd.Series, prefix: str, registry: dict[str, dict[str, str]]) -> dict:
    return _canonical_location(
        row.get(f"{prefix}_1_5"),
        {
            "administrative_dong_name": clean(row.get(f"{prefix}_1_8")) or None,
            "gu": clean(row.get(f"{prefix}_1_7")) or None,
            "sido": clean(row.get(f"{prefix}_1_6")) or None,
        },
        registry,
    )


def home_dong(person: pd.Series, registry: dict[str, dict[str, str]]) -> dict:
    return _canonical_location(
        person.get("SQ1_5"),
        {
            "administrative_dong_name": clean(person.get("SQ1_8")) or None,
            "gu": clean(person.get("SQ1_7")) or None,
            "sido": clean(person.get("SQ1_6")) or None,
        },
        registry,
    )


def trip_modes(row: pd.Series) -> list[str]:
    result = []
    for i in range(1, 11):
        code = clean(row.get(f"TP5_{i}"))
        if code:
            if code == "2":
                role = clean(row.get(f"TP5_{i}_2_1"))
                # The KTDB codebook requires this follow-up for car/van stages.
                # Fail loudly instead of collapsing driver and passenger into a
                # misleading generic `car` label.
                if role not in CAR_ROLE:
                    raise ValueError(f"car/van stage TP5_{i}=2 has invalid TP5_{i}_2_1={role!r}")
                result.append(CAR_ROLE[role])
            else:
                result.append(MODE.get(code, "other"))
    return result


def ranked_mode_preference(rows: list[pd.Series]) -> list[dict]:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    serial = 0
    for row in rows:
        for mode in trip_modes(row):
            counts[mode] += 1
            first_seen.setdefault(mode, serial)
            serial += 1
    ordered = sorted(counts, key=lambda mode: (-counts[mode], first_seen[mode]))
    return [{"rank": i + 1, "mode": mode, "count": counts[mode]} for i, mode in enumerate(ordered)]


def build_targets(
    person: pd.Series, trip_group: pd.DataFrame | None, registry: dict[str, dict[str, str]]
) -> tuple[dict, dict, dict]:
    pid = clean(person["idx"])
    rows: list[pd.Series] = []
    if trip_group is not None:
        rows = [row for _, row in trip_group.iterrows() if clean(row.get("TP2"))]

    if not rows:
        episode = {
            "activity_index": 0,
            "activity_type": "home",
            "location": home_dong(person, registry),
            "start_time_min": 0,
            "dwell_time_min": 1440,
        }
        sequence = {"person_id": pid, "activity_sequence": ["home"]}
        spatial = {
            "person_id": pid,
            "activities": [episode],
            "mode_preference": [],
            "timeline_valid": True,
        }
        return sequence, spatial, {"person_id": pid, "activity_sequence": sequence["activity_sequence"], **spatial}

    first = rows[0]
    first_departure = clock_minute(first.get("TP3_1"), first.get("TP3_2"))
    origin_type = classify_activity("", "", first.get("sTP1"))
    origin_location = admin_dong(first, "sTP1", registry)
    if origin_type == "home":
        origin_location = home_dong(person, registry)

    episodes = [
        {
            "activity_index": 0,
            "activity_type": origin_type,
            "location": origin_location,
            "start_time_min": 0,
            "dwell_time_min": first_departure,
        }
    ]
    timeline_valid = first_departure is not None and first_departure >= 0

    for i, row in enumerate(rows):
        activity_type = classify_activity(row.get("TP2"), row.get("TP2_a"), row.get("TP1"))
        location = home_dong(person, registry) if activity_type == "home" else admin_dong(row, "TP1", registry)
        arrival = clock_minute(row.get("TP4_1"), row.get("TP4_2"))
        if i + 1 < len(rows):
            next_departure = clock_minute(rows[i + 1].get("TP3_1"), rows[i + 1].get("TP3_2"))
            dwell = None if arrival is None or next_departure is None else next_departure - arrival
        else:
            dwell = None if arrival is None else 1440 - arrival
        if arrival is None or dwell is None or dwell < 0:
            timeline_valid = False
        episodes.append(
            {
                "activity_index": i + 1,
                "activity_type": activity_type,
                "location": location,
                "start_time_min": arrival,
                "dwell_time_min": dwell,
            }
        )

    sequence_values = [episode["activity_type"] for episode in episodes]
    assert set(sequence_values) <= ACTIVITY_TYPES
    sequence = {"person_id": pid, "activity_sequence": sequence_values}
    spatial = {
        "person_id": pid,
        "activities": episodes,
        "mode_preference": ranked_mode_preference(rows),
        "timeline_valid": timeline_valid,
    }
    reference = {"person_id": pid, "activity_sequence": sequence_values, **spatial}
    return sequence, spatial, reference


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=SHARED_DATA_ROOT / "KTDB_2021")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "processed")
    parser.add_argument("--admin-codebook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    args = parser.parse_args()
    admin_registry = load_admin_registry(args.admin_codebook)

    people_path = args.raw_dir / "①개인특성.csv"
    trips_path = args.raw_dir / "②이동특성.csv"
    people = pd.read_csv(people_path, encoding="cp949", dtype=str, low_memory=False)
    people = people[people["SQ1_6"].map(clean).eq("서울특별시")].copy()
    people["idx"] = people["idx"].map(clean)
    people = people.sort_values("idx").reset_index(drop=True)

    trip_columns = [
        "idx",
        "fid",
        "th_seq",
        "sTP1",
        "sTP1_1_5",
        "sTP1_1_6",
        "sTP1_1_7",
        "sTP1_1_8",
        "TP1",
        "TP1_1_5",
        "TP1_1_6",
        "TP1_1_7",
        "TP1_1_8",
        "TP2",
        "TP2_a",
        "TP3_1",
        "TP3_2",
        "TP4_1",
        "TP4_2",
        *[f"TP5_{i}" for i in range(1, 11)],
        *[f"TP5_{i}_2_1" for i in range(1, 11)],
    ]
    trips = pd.read_csv(trips_path, encoding="cp949", dtype=str, usecols=trip_columns, low_memory=False)
    trips["idx"] = trips["idx"].map(clean)
    trips = trips[trips["idx"].isin(set(people["idx"]))].copy()
    trips["_seq"] = pd.to_numeric(trips["th_seq"], errors="coerce")
    trips = trips.sort_values(["idx", "_seq"], kind="stable")
    grouped = {pid: group for pid, group in trips.groupby("idx", sort=False)}

    # Q4 is a household-level question and is blank for non-representative
    # household members.  Use trip-table fid to propagate a reported household
    # car/van value without treating structural blanks as "no car".
    pid_to_fid = trips.groupby("idx", sort=False)["fid"].first().map(clean).to_dict()
    fid_car_values: dict[str, list[bool]] = {}
    for _, person in people.iterrows():
        pid = person["idx"]
        fid = pid_to_fid.get(pid, "")
        car, van = integer(person.get("Q4_1")), integer(person.get("Q4_2"))
        if fid and (car is not None or van is not None):
            fid_car_values.setdefault(fid, []).append((car or 0) + (van or 0) > 0)
    fid_car = {fid: any(values) for fid, values in fid_car_values.items()}

    ids = people["idx"].to_numpy(copy=True)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * args.test_fraction))
    test_ids = set(ids[:n_test])
    split = {pid: ("test" if pid in test_ids else "train") for pid in people["idx"]}

    demographic_records: dict[str, dict] = {}
    sequences: dict[str, dict] = {}
    spatials: dict[str, dict] = {}
    references: dict[str, dict] = {}
    for _, person in people.iterrows():
        pid = person["idx"]
        household_car = fid_car.get(pid_to_fid.get(pid, ""))
        demographic_records[pid] = {
            "person_id": pid,
            "demographics": demographics(person, household_car_override=household_car),
        }
        seq, spatial, reference = build_targets(person, grouped.get(pid), admin_registry)
        sequences[pid], spatials[pid], references[pid] = seq, spatial, reference

    train_ids = [pid for pid in people["idx"] if split[pid] == "train"]
    test_ids_ordered = [pid for pid in people["idx"] if split[pid] == "test"]
    # Invalid train timelines cannot supply the paired activity/spatial targets
    # required by either student. Remove those people from every processed
    # population-level artifact, rather than leaving unmatched demographics.
    rejected_train_ids = [pid for pid in train_ids if not spatials[pid]["timeline_valid"]]
    rejected_train_set = set(rejected_train_ids)
    train_ids = [pid for pid in train_ids if pid not in rejected_train_set]
    processed_people_ids = [pid for pid in people["idx"] if pid not in rejected_train_set]
    out = args.out_dir
    write_jsonl(out / "seoul_demographics_all.jsonl", (demographic_records[pid] for pid in processed_people_ids))
    write_jsonl(out / "train" / "demographics.jsonl", (demographic_records[pid] for pid in train_ids))
    write_jsonl(out / "test" / "demographics.jsonl", (demographic_records[pid] for pid in test_ids_ordered))
    clean_train_ids = train_ids
    write_jsonl(out / "train" / "activity_sequence_all.jsonl", (sequences[pid] for pid in train_ids))
    write_jsonl(out / "train" / "location_dwell_mode_preference_all.jsonl", (spatials[pid] for pid in train_ids))
    write_jsonl(out / "train" / "activity_sequence.jsonl", (sequences[pid] for pid in clean_train_ids))
    write_jsonl(out / "train" / "location_dwell_mode_preference.jsonl", (spatials[pid] for pid in clean_train_ids))
    # Keep the artifact path stable for downstream tooling, but do not retain
    # records for people removed from the processed population.
    write_jsonl(out / "train" / "rejected_invalid_timeline.jsonl", ())
    write_jsonl(out / "test" / "ground_truth_reference.jsonl", (references[pid] for pid in test_ids_ordered))
    whitelist = build_model_whitelist(admin_registry)
    whitelist["source"] = str(args.admin_codebook)
    (out / "ktdb_location_whitelist.json").write_text(
        json.dumps(whitelist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_people = people[people["idx"].isin(processed_people_ids)]
    manifest = pd.DataFrame(
        {
            "person_id": manifest_people["idx"],
            "split": manifest_people["idx"].map(split),
        }
    )
    manifest.to_csv(out / "split_manifest.csv", index=False, encoding="utf-8")

    invalid_train = 0
    invalid_test = sum(not spatials[pid]["timeline_valid"] for pid in test_ids_ordered)
    no_trip = sum(len(sequences[pid]["activity_sequence"]) == 1 for pid in processed_people_ids)
    activity_counts = Counter(a for pid in train_ids for a in sequences[pid]["activity_sequence"])
    missing = {
        key: sum(demographic_records[pid]["demographics"][key] == "unknown" for pid in processed_people_ids)
        for key in ("gender", "age", "occupation")
    }
    missing["can_drive"] = sum(
        demographic_records[pid]["demographics"]["can_drive"] == "unknown" for pid in processed_people_ids
    )
    summary = {
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "seoul_residents": len(processed_people_ids),
        "train_people": len(train_ids),
        "clean_train_people": len(clean_train_ids),
        "test_people": len(test_ids_ordered),
        "no_trip_people": no_trip,
        "invalid_timeline_train": invalid_train,
        "invalid_timeline_test": invalid_test,
        "missing_demographics": missing,
        "train_activity_counts": dict(sorted(activity_counts.items())),
        "mode_preference_method": (
            "count all non-empty TP5_1..TP5_10 stage-mode occurrences; split "
            "passenger car/van (code 2) into car_driver vs car_passenger using "
            "TP5_i_2_1; keep taxi (code 10) separate; sort by count desc, then first occurrence"
        ),
        "administrative_location_source": str(args.admin_codebook),
        "model_location_fields": "English natural-language names; numeric code retained only for internal joins",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
