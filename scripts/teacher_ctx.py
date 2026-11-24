#!/usr/bin/env python3
"""Run the KTDB Teacher with household / student / work context.

The Teacher (scripts/run_teacher_2gpu.py, scripts/teacher_worker.py) shows only gender, age, occupation and can_drive, so its personas contradict the
context the v2ctx Students read (55% of multi-member households were described
as living alone). This wrapper leaves the Teacher code as it is: it replaces
build_user_prompt and adds one consistency rule to SYSTEM_PROMPT in the loaded
module, then runs the original worker script.

    inputs   python scripts/teacher_ctx.py --build-inputs
             -> processed_v2ctx/teacher_input/{demographics,activity_sequence,
                location_dwell_mode_preference}.jsonl (v2 train + validation people,
                demographics with the build_context_v2.py context)
    teacher  run_teacher_2gpu.py --python scripts/teacher_ctx_python ...
             (it starts `<python> teacher_worker.py ...`; teacher_ctx_python
             runs this file with those arguments)
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import DATA_ROOT  # noqa: E402

SFT_BUILD = DATA_ROOT / "processed_v2/sft_build"
CTX = DATA_ROOT / "processed_v2ctx"
INPUT_DIR = CTX / "teacher_input"

RULE = (
    "3. The respondent section may list household, school and work facts from the\n"
    "   survey. They are true. The personas must agree with them: never describe a\n"
    "   respondent in a multi-member household as living alone, and keep children,\n"
    "   older adults, cars, income, housing, school level and workplace status\n"
    "   consistent with what is listed.\n\n"
)
ANCHOR = "Output EXACTLY these six XML blocks"


def context_lines(d: dict) -> list[str]:
    lines = []
    h = d.get("household")
    if h:
        parts = []
        if "members" in h:
            parts.append(f"{h['members']} member{'s' if h['members'] != 1 else ''}")
        for key, label in (
            ("children_under_5", "children under 5"),
            ("children_5_to_13", "children aged 5-13"),
            ("teens_14_to_18", "teens aged 14-18"),
            ("adults_70_plus", "adults aged 70+"),
        ):
            if key in h:
                parts.append(f"{label}: {h[key]}")
        lines.append("- Household: " + ", ".join(parts))
        for key, label in (
            ("housing", "Housing"),
            ("monthly_income", "Monthly household income"),
            ("cars", "Household cars"),
        ):
            if key in h:
                lines.append(f"- {label}: {h[key]}")
    if "student" in d:
        lines.append(f"- Student: {d['student']}")
    if "work" in d:
        lines.append(f"- Fixed workplace: {d['work']['fixed_workplace']}")
    return lines


def patch() -> None:
    import tripchain.ktdb.ktdb_two_stage_teacher as teacher

    original = teacher.build_user_prompt
    if ANCHOR not in teacher.SYSTEM_PROMPT:
        raise SystemExit("SYSTEM_PROMPT anchor not found; the Teacher prompt changed")
    teacher.SYSTEM_PROMPT = teacher.SYSTEM_PROMPT.replace(ANCHOR, RULE + ANCHOR, 1)

    def build_user_prompt(record: dict) -> str:
        prompt = original(record)
        extra = context_lines(record["demographics"])
        if not extra:
            return prompt
        head, sep, rest = prompt.partition("\n\n## Observed activity types")
        if not sep:
            raise ValueError("unexpected Teacher prompt layout")
        return head + "\n" + "\n".join(extra) + sep + rest

    teacher.build_user_prompt = build_user_prompt


def build_inputs() -> None:
    ctx = {}
    for split in ("train", "validation"):
        for line in (CTX / split / "demographics_with_residence.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            ctx[str(row["person_id"])] = {
                k: row["demographics"][k] for k in ("household", "student", "work") if k in row["demographics"]
            }
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with (INPUT_DIR / "demographics.jsonl").open("w", encoding="utf-8") as out:
        for line in (SFT_BUILD / "demographics.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            row["demographics"] = {**row["demographics"], **ctx[str(row["person_id"])]}
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    for name in ("activity_sequence", "location_dwell_mode_preference"):
        link = INPUT_DIR / f"{name}.jsonl"
        if not link.exists():
            link.symlink_to(SFT_BUILD / f"{name}.jsonl")
    print(f"{n} people -> {INPUT_DIR}")


def main() -> None:
    if sys.argv[1:] == ["--build-inputs"]:
        build_inputs()
        return
    if sys.argv[1:2] == ["--show-prompt"]:  # --show-prompt <person_id>
        patch()
        import tripchain.ktdb.ktdb_two_stage_teacher as teacher

        records = teacher.load_teacher_records(
            INPUT_DIR / "demographics.jsonl",
            INPUT_DIR / "activity_sequence.jsonl",
            INPUT_DIR / "location_dwell_mode_preference.jsonl",
        )
        record = next(r for r in records if r["person_id"] == sys.argv[2])
        print(teacher.SYSTEM_PROMPT[-900:], "\n-----\n", teacher.build_user_prompt(record))
        return
    script = sys.argv[1]
    patch()
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
