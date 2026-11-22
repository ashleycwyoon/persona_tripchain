#!/usr/bin/env python3
"""Regenerate Student1 then Student2 for people whose Student2 output never validated.

Student2 inference already resamples the unchanged prompt. People still missing
from the Student2 output get a fresh Student1 sample (same demographics prompt,
normal temperature) followed by Student2 sampling on that new Student1 output,
mirroring refinement Lv.3. Both rows are replaced together so the Student1 and
Student2 activity sequences stay identical, and every accepted output keeps the
exact prompt that produced it for GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.common import append_jsonl, load_config, read_jsonl, resolve  # noqa: E402
from tripchain.inference import (  # noqa: E402
    build_student2_user,
    load_location_whitelist,
    load_model,
    resample_valid,
    student2_output_row,
    validate_student1,
    validate_student2,
)


def write_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".repair.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--student1", required=True, help="merged Student1 output; rewritten in place")
    parser.add_argument("--student2", required=True, help="merged Student2 output; rewritten in place")
    parser.add_argument("--student1-adapter", default=None)
    parser.add_argument("--student2-adapter", default=None)
    parser.add_argument("--seed", type=int, default=None, help="the seed used for this Student2 run")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, inf = cfg["paths"], cfg["inference"]
    seed = int(args.seed if args.seed is not None else inf["seed"])
    student1_path, student2_path = resolve(args.student1), resolve(args.student2)
    student1_rows = list(read_jsonl(student1_path))
    student2_rows = {str(row["person_id"]): row for row in read_jsonl(student2_path)}
    missing = [row for row in student1_rows if str(row["person_id"]) not in student2_rows]
    if not missing:
        print(json.dumps({"missing": 0, "repaired": 0, "unrepaired": 0}))
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    # Offset from the inference seed so repair draws differ from the failed ones.
    random.seed(seed + 7919)
    torch.manual_seed(seed + 7919)
    cache = Path(paths["model_cache"])
    model1, tokenizer1 = load_model(
        resolve(args.student1_adapter or paths["student1_adapter"]), inf["base_model"], cache
    )
    model2, tokenizer2 = load_model(
        resolve(args.student2_adapter or paths["student2_adapter"]), inf["base_model"], cache
    )
    whitelist = load_location_whitelist(paths["location_whitelist"])
    attempts = int(inf.get("student2_resample_attempts", 4))
    rounds = int(inf.get("student1_repair_rounds", 2))
    temperature, top_p = float(inf["temperature"]), float(inf["top_p"])

    replaced: dict[str, dict] = {}
    unrepaired = []
    for row in tqdm(missing, desc="Student2 repair (Student1 -> Student2)", unit="person"):
        person_id = str(row["person_id"])
        last_error = None
        for round_index in range(1, rounds + 1):
            try:
                user1 = {"demographics": row["demographics"]}
                value1 = resample_valid(
                    model1,
                    tokenizer1,
                    user1,
                    inf["student1_max_new_tokens"],
                    attempts,
                    temperature,
                    top_p,
                    validate_student1,
                )
                repair = {"stage": "student1_then_student2", "round": round_index}
                new1 = {
                    "person_id": person_id,
                    "demographics": row["demographics"],
                    **value1,
                    "student1_input": user1,
                    "status": "ok",
                    "repair": repair,
                }
                user2 = build_student2_user(new1)
                value2 = resample_valid(
                    model2,
                    tokenizer2,
                    user2,
                    inf["student2_max_new_tokens"],
                    attempts,
                    temperature,
                    top_p,
                    lambda value: validate_student2(value, new1["activity_sequence"], whitelist),
                )
            except Exception as exc:
                last_error = exc
                continue
            replaced[person_id] = new1
            student2_rows[person_id] = {**student2_output_row(new1, value2, user2), "repair": repair}
            break
        else:
            unrepaired.append(
                {
                    "person_id": person_id,
                    "status": "error",
                    "error": f"repair failed after {rounds} rounds: {last_error}",
                }
            )

    write_atomic(student1_path, [replaced.get(str(row["person_id"]), row) for row in student1_rows])
    write_atomic(
        student2_path,
        [student2_rows[str(row["person_id"])] for row in student1_rows if str(row["person_id"]) in student2_rows],
    )
    error_path = student2_path.with_name(student2_path.name + ".repair_errors.jsonl")
    with error_path.open("w", encoding="utf-8") as handle:
        for item in unrepaired:
            append_jsonl(handle, item)
    print(
        json.dumps(
            {
                "missing": len(missing),
                "repaired": len(replaced),
                "unrepaired": len(unrepaired),
                "unrepaired_log": str(error_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
