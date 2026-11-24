#!/usr/bin/env python3
"""v2 data with household / student / work-status context in the Student inputs (processed_v2ctx).

The v2 Students see gender, age, occupation, can_drive and residence. The KTDB 2021
person file also records household and status attributes of the kind a
synthetic population carries. This adds them to `demographics` everywhere the
Students read it:

    demographics.household  members, children_under_5, children_5_to_13, teens_14_to_18,
                            adults_70_plus, housing, monthly_income, cars
    demographics.student    school level, when the person is a student
    demographics.work       fixed_workplace: yes / no fixed place / not working

Left out on purpose: workplace and school locations (a synthetic
population has none; the Students would copy the survey sample's places instead
of generating them), work schedule (workdays, telework days, transport job),
motorcycles / bicycles, regular driving (15% answered), survey-day answers (R4_1,
R4_2, D1) and habitual travel (R4_3-R4_6), which restate the outcome. Missing
answers are omitted, not filled.

Reads processed_v2; writes <out>/{train,validation,test}/demographics_with_residence.jsonl,
ground_truth_tripmode.jsonl (symlinks), sft/{student1_residence,student2_tripmode}/*.jsonl
(the v2 SFT examples with the user demographics replaced) and manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import DATA_ROOT, SHARED_DATA_ROOT  # noqa: E402

SPLITS = ("train", "validation", "test")
HOUSING = {
    "1": "apartment",
    "2": "row house (villa)",
    "3": "multi-family house",
    "4": "detached house",
    "5": "officetel",
    "97": "other",
}
INCOME = {
    "1": "under 1M KRW",
    "2": "1-3M KRW",
    "3": "3-5M KRW",
    "4": "5-10M KRW",
    "5": "10-15M KRW",
    "6": "15M KRW or more",
}
STUDENT = {
    "0": "preschool",
    "1": "elementary school",
    "2": "middle school",
    "3": "high school",
    "4": "university or graduate school",
}
FIXED = {"1": "yes", "2": "no fixed place", "3": "not working"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def count(value: str) -> int | None:
    value = value.strip()
    return int(float(value)) if value else None


def context(r: dict) -> dict:
    out: dict = {}
    household = {
        "members": count(r["Q1"]),
        "children_under_5": count(r["Q1_1"]),
        "children_5_to_13": count(r["Q1_2"]),
        "teens_14_to_18": count(r["Q1_5"]),
        "adults_70_plus": count(r["Q1_4"]),
        "housing": HOUSING.get(r["Q2"].strip()),
        "monthly_income": INCOME.get(r["Q3"].strip()),
    }
    if household["members"] is not None:  # the vehicle questions are blank exactly when A1 is
        household["cars"] = (count(r["Q4_1"]) or 0) + (count(r["Q4_2"]) or 0)
    household = {k: v for k, v in household.items() if v is not None}
    if household:
        out["household"] = household
    if r["R2"].strip() in STUDENT:
        out["student"] = STUDENT[r["R2"].strip()]
    if r["R3"].strip() in FIXED:
        out["work"] = {"fixed_workplace": FIXED[r["R3"].strip()]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-dir", type=Path, default=DATA_ROOT / "processed_v2")
    parser.add_argument("--raw", type=Path, default=SHARED_DATA_ROOT / "KTDB_2021/①개인특성.csv")
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "processed_v2ctx")
    args = parser.parse_args()

    with args.raw.open(encoding="cp949") as handle:
        raw = {r["idx"]: r for r in csv.DictReader(handle)}

    ctx, counts = {}, {}
    for split in SPLITS:
        rows = read_rows(args.v2_dir / split / "demographics_with_residence.jsonl")
        for row in rows:
            pid = str(row["person_id"])
            ctx[pid] = context(raw[pid])
            row["demographics"] = {**row["demographics"], **ctx[pid]}
        counts[f"{split}/demographics"] = write_rows(args.out / split / "demographics_with_residence.jsonl", rows)
        link = args.out / split / "ground_truth_tripmode.jsonl"
        if not link.exists():
            link.symlink_to(args.v2_dir / split / "ground_truth_tripmode.jsonl")

    for name in ("student1_residence", "student2_tripmode"):
        for part in ("train", "validation"):
            rows = read_rows(args.v2_dir / "sft" / name / f"{part}.jsonl")
            for row in rows:
                user = json.loads(row["messages"][0]["content"])
                user["demographics"] = {**user["demographics"], **ctx[str(row["person_id"])]}
                row["messages"][0]["content"] = json.dumps(user, ensure_ascii=False, separators=(",", ":"))
            counts[f"sft/{name}/{part}"] = write_rows(args.out / "sft" / name / f"{part}.jsonl", rows)

    people = list(ctx.values())
    coverage = {key: round(sum(key in c for c in people) / len(people), 4) for key in ("household", "student", "work")}
    manifest = {"source": str(args.raw), "base": str(args.v2_dir), "counts": counts, "coverage": coverage}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
