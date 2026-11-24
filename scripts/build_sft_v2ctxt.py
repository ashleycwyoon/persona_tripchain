#!/usr/bin/env python3
"""v2ctx SFT data with Teacher labels that saw the context (processed_v2ctxt).

The v2ctx SFT targets (persona bundle, reasoning) came from a Teacher that saw only
gender / age / occupation / can_drive, so they often contradict the household context
in the Student inputs. This rebuilds the same examples with the labels of
scripts/teacher_ctx.py (all v2 train + validation people):

  inputs   processed_v2/sft_build (the v1-format Teacher inputs build_sft_v2.py wrote)
  labels   <teacher-prefix>_shard{0,1}.jsonl
  examples upstream build_student1_examples / build_student2_examples, then the
           build_sft_v2.py conversion (residence, key order, per-trip modes) and the
           build_context_v2.py context merge.

User messages must equal processed_v2ctx/sft except for the Teacher text Student2 reads
(persona_bundle, reasoning1); this is checked.
Writes <out>/sft/{student1_residence,student2_tripmode}/{train,validation}.jsonl and manifest.json.
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
from tripchain.paths import DATA_ROOT, OUTPUT_ROOT, TRIPCHAIN_1015_CODE  # noqa: E402
from tripchain.residence import augment_demographics, validate_residence_demographics  # noqa: E402
import prepare_residence_data as residence  # noqa: E402

sys.path.insert(0, str(TRIPCHAIN_1015_CODE / "persona-rationale-multi-country-qwen35"))
from src.data.ktdb_student1 import build_student1_examples  # noqa: E402
from src.data.ktdb_student2 import build_student2_examples  # noqa: E402
from src.data.ktdb_student_common import require_completed_teacher_shards  # noqa: E402

from build_sft_v2 import read_rows, write_rows  # noqa: E402

TEACHER_INPUTS = ("persona_bundle", "reasoning1")  # Student2 reads these Teacher fields, so they change


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-dir", type=Path, default=DATA_ROOT / "processed_v2")
    parser.add_argument("--ctx-dir", type=Path, default=DATA_ROOT / "processed_v2ctx")
    parser.add_argument("--teacher-prefix", type=Path, default=OUTPUT_ROOT / "teacher_ctx" / "teacher_ctx")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "processed_v2ctxt")
    args = parser.parse_args()
    build = args.v2_dir / "sft_build"

    split, truth, ctx = {}, {}, {}
    for name in ("train", "validation"):
        for row in read_rows(args.v2_dir / name / "ground_truth_tripmode.jsonl"):
            split[row["person_id"]] = name
            truth[row["person_id"]] = row
        for row in read_rows(args.ctx_dir / name / "demographics_with_residence.jsonl"):
            ctx[str(row["person_id"])] = {
                k: row["demographics"][k] for k in ("household", "student", "work") if k in row["demographics"]
            }

    shards = [Path(f"{args.teacher_prefix}_shard{i}.jsonl") for i in range(args.shards)]
    require_completed_teacher_shards(shards)
    teacher = {str(r["person_id"]): r for path in shards for r in read_rows(path)}
    if set(teacher) != set(split):
        raise SystemExit(f"teacher has {len(teacher)} people, split has {len(split)}")
    work = args.out / "sft_build"
    teacher_path = work / "teacher.jsonl"
    write_rows(teacher_path, (teacher[pid] for pid in split))

    student1 = build_student1_examples(build / "demographics.jsonl", build / "activity_sequence.jsonl", [teacher_path])
    student2 = build_student2_examples(
        build / "demographics.jsonl",
        build / "activity_sequence.jsonl",
        build / "location_dwell_mode_preference.jsonl",
        [teacher_path],
    )

    residences = residence.load_residences()
    s1_order, _ = OUTPUT_ORDERS["reason_first"]
    _, s2_order = OUTPUT_ORDERS["reason_first_tripmode"]

    def convert(row: dict, order: tuple, trip_modes: bool) -> dict:  # build_sft_v2.py + build_context_v2.py
        pid = str(row["person_id"])
        messages = [dict(m) for m in row["messages"]]
        user = json.loads(messages[0]["content"])
        user["demographics"] = augment_demographics(user["demographics"], residences[pid])
        validate_residence_demographics(user["demographics"])
        user["demographics"] = {**user["demographics"], **ctx[pid]}
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

    def inputs(row: dict) -> str:  # the user message without the Teacher text Student2 is given
        user = json.loads(row["messages"][0]["content"])
        for key in TEACHER_INPUTS:
            user.pop(key, None)
        return json.dumps(user, ensure_ascii=False, separators=(",", ":"))

    counts = {}
    for name, examples, order, modes in (
        ("student1_residence", student1, s1_order, False),
        ("student2_tripmode", student2, s2_order, True),
    ):
        for part in ("train", "validation"):
            rows = [convert(r, order, modes) for r in examples if split[str(r["person_id"])] == part]
            reference = {
                str(r["person_id"]): inputs(r) for r in read_rows(args.ctx_dir / "sft" / name / f"{part}.jsonl")
            }
            mismatched = [r["person_id"] for r in rows if reference.get(str(r["person_id"])) != inputs(r)]
            if mismatched or len(rows) != len(reference):
                raise SystemExit(
                    f"{name}/{part}: {len(mismatched)} user messages differ from v2ctx "
                    f"({len(rows)} vs {len(reference)} rows), e.g. {mismatched[:3]}"
                )
            counts[f"{name}/{part}"] = write_rows(args.out / "sft" / name / f"{part}.jsonl", rows)

    manifest = {
        "teacher": [str(p) for p in shards],
        "inputs": str(build),
        "context": str(args.ctx_dir),
        "counts": counts,
        "user_messages": "identical to processed_v2ctx/sft apart from persona_bundle / reasoning1",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
