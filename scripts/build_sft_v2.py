#!/usr/bin/env python3
"""Student1 / Student2 SFT data on the v2 ground truth and split.

Steps (the Teacher and the example builders are the upstream ones, vendored
under third_party/tripchain_1015/persona-rationale-multi-country-qwen35):
1. Teacher-format inputs for v2 train + validation people: v1 files for people
   whose ground truth did not change, scripts/build_teacher_relabel_inputs.py
   output for the rest.
2. Teacher labels: the v1 Teacher shards with the changed people replaced by
   the relabel run (same Teacher, prompt and schema; checked through its
   completion manifests).
3. Chat examples via the upstream build_student1_examples /
   build_student2_examples, assigned to train / validation by the v2 split
   (not the builders' own 5% hash split).
4. Residence added to the user demographics, target keys written in the
   reason-first order, and Student2 targets given the per-trip main modes of the
   v2 ground truth in place of mode_preference.

Writes <v2>/sft/student1_residence and <v2>/sft/student2_tripmode (and the
intermediate joined inputs under <v2>/sft_build).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tripchain.common import OUTPUT_ORDERS, ordered_output  # noqa: E402
from tripchain.paths import DATA_ROOT, OUTPUT_ROOT, TRIPCHAIN_1015_CODE, TRIPCHAIN_1015_DATA  # noqa: E402
from tripchain.residence import augment_demographics, validate_residence_demographics  # noqa: E402
import prepare_residence_data as residence  # noqa: E402

TEACHER_REPO = TRIPCHAIN_1015_CODE / "persona-rationale-multi-country-qwen35"
sys.path.insert(0, str(TEACHER_REPO))
from src.data.ktdb_student1 import build_student1_examples  # noqa: E402
from src.data.ktdb_student2 import build_student2_examples  # noqa: E402
from src.data.ktdb_student_common import (  # noqa: E402
    DEFAULT_TEACHER_SHARDS,
    require_completed_teacher_shards,
)

V1_INPUTS = TRIPCHAIN_1015_DATA / "processed/train"
INPUT_FILES = ("demographics", "activity_sequence", "location_dwell_mode_preference")


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-dir", type=Path, default=DATA_ROOT / "processed_v2")
    parser.add_argument("--relabel-prefix", type=Path, default=OUTPUT_ROOT / "teacher_relabel_v2" / "teacher_v2")
    parser.add_argument("--relabel-shards", type=int, default=2)
    args = parser.parse_args()
    v2_dir = args.v2_dir
    work = v2_dir / "sft_build"

    split = {}
    truth = {}
    for name in ("train", "validation"):
        for row in read_rows(v2_dir / name / "ground_truth_tripmode.jsonl"):
            split[row["person_id"]] = name
            truth[row["person_id"]] = row
    relabel_ids = set((v2_dir / "teacher_relabel_ids.txt").read_text().split())

    # 1. Teacher-format inputs.
    for name in INPUT_FILES:
        old = {str(r["person_id"]): r for r in read_rows(V1_INPUTS / f"{name}.jsonl")}
        new = {str(r["person_id"]): r for r in read_rows(v2_dir / "teacher_relabel" / f"{name}.jsonl")}
        rows = [new[pid] if pid in relabel_ids else old[pid] for pid in split]
        write_rows(work / f"{name}.jsonl", rows)

    # 2. Teacher labels.
    relabel_shards = [Path(f"{args.relabel_prefix}_shard{i}.jsonl") for i in range(args.relabel_shards)]
    require_completed_teacher_shards(list(DEFAULT_TEACHER_SHARDS) + relabel_shards)
    teacher = {}
    for path in DEFAULT_TEACHER_SHARDS:
        for row in read_rows(path):
            teacher[str(row["person_id"])] = row
    replaced = 0
    for path in relabel_shards:
        for row in read_rows(path):
            teacher[str(row["person_id"])] = row
            replaced += 1
    if replaced != len(relabel_ids):
        raise SystemExit(f"relabel run has {replaced} people, expected {len(relabel_ids)}")
    teacher_path = work / "teacher.jsonl"
    write_rows(teacher_path, (teacher[pid] for pid in split))

    # 3. Chat examples, split by v2.
    paths = {name: work / f"{name}.jsonl" for name in INPUT_FILES}
    student1 = build_student1_examples(paths["demographics"], paths["activity_sequence"], [teacher_path])
    student2 = build_student2_examples(
        paths["demographics"], paths["activity_sequence"], paths["location_dwell_mode_preference"], [teacher_path]
    )

    # 4. Residence, key order, per-trip modes.
    residences = residence.load_residences()
    s1_order, _ = OUTPUT_ORDERS["reason_first"]
    _, s2_order = OUTPUT_ORDERS["reason_first_tripmode"]

    def convert(row: dict, order: tuple, trip_modes: bool) -> dict:
        pid = str(row["person_id"])
        messages = [dict(m) for m in row["messages"]]
        user = json.loads(messages[0]["content"])
        user["demographics"] = augment_demographics(user["demographics"], residences[pid])
        validate_residence_demographics(user["demographics"])
        messages[0]["content"] = json.dumps(user, ensure_ascii=False, separators=(",", ":"))
        target = json.loads(messages[1]["content"])
        if trip_modes:
            modes = truth[pid]["trip_modes"]
            if len(modes) != len(target["location_dwell_by_activity"]) - 1:
                raise ValueError(f"person_id={pid}: trip count does not match activities")
            target.pop("mode_preference")
            target["trip_modes"] = modes
        messages[1]["content"] = json.dumps(ordered_output(target, order), ensure_ascii=False, separators=(",", ":"))
        return {**row, "messages": messages}

    counts = {}
    for name, examples, order, modes in (
        ("student1_residence", student1, s1_order, False),
        ("student2_tripmode", student2, s2_order, True),
    ):
        out = v2_dir / "sft" / name
        for part in ("train", "validation"):
            rows = [convert(r, order, modes) for r in examples if split[str(r["person_id"])] == part]
            counts[f"{name}/{part}"] = write_rows(out / f"{part}.jsonl", rows)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "v2 ground truth (scripts/rebuild_ground_truth_v2.py)",
                    "target_key_order": list(order),
                    "split": "processed_v2 train / validation",
                    "teacher": "v1 shards + relabel " + str(args.relabel_prefix),
                    "counts": counts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    print(json.dumps({"people": len(split), "relabeled": replaced, **counts}, indent=2))


if __name__ == "__main__":
    main()
