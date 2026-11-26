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

Answers the survey left blank are not in the inputs (scripts/build_context_v2.py omits
them; nothing is filled in). A label that says a respondent lives alone although the
household has several members fails validation and is generated again
(context_violation). Labels written before this check are replaced with

    select   python scripts/teacher_ctx.py --select-fix <teacher-prefix>
             -> processed_v2ctx/teacher_input_fix/ (people whose label fails the check)
    teacher  the same Teacher command on teacher_input_fix, --output-prefix <fix-prefix>
    merge    python scripts/teacher_ctx.py --merge-fix <teacher-prefix> <fix-prefix>
             (the replaced shards are kept as <shard>.before_fix)
"""

from __future__ import annotations

import json
import re
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import DATA_ROOT  # noqa: E402

SFT_BUILD = DATA_ROOT / "processed_v2/sft_build"
CTX = DATA_ROOT / "processed_v2ctx"
INPUT_DIR = CTX / "teacher_input"
FIX_DIR = CTX / "teacher_input_fix"
INPUT_NAMES = ("demographics", "activity_sequence", "location_dwell_mode_preference")
LABEL_TEXTS = ("professional_persona", "family_persona", "persona", "reasoning1", "reasoning2")

RULE = (
    "3. The respondent section may list household, school and work facts from the\n"
    "   survey. They are true. The personas must agree with them: never describe a\n"
    "   respondent in a multi-member household as living alone, and keep children,\n"
    "   older adults, cars, income, housing, school level and workplace status\n"
    "   consistent with what is listed.\n\n"
)
ANCHOR = "Output EXACTLY these six XML blocks"

_LIVE = r"(?:lives?|living|lived|resides?|residing)"
ALONE = re.compile(
    rf"\b{_LIVE} (?:alone|by (?:him|her|them)sel(?:f|ves)|on (?:his|her|their) own)\b"
    r"|\b(?:single|one)[- ](?:person|member|adult) household\b"
    r"|\bhousehold (?:consists|is made up) of (?:just|only) (?:him|her|one)\b",
    re.IGNORECASE,
)

def context_violation(demographics: dict, texts: list[str]) -> str | None:
    """What a Teacher label states against the household answers; None if nothing."""
    members = (demographics.get("household") or {}).get("members")
    if members and members >= 2 and ALONE.search(" ".join(texts)):
        return f"the household has {members} members, so the respondent does not live alone"
    return None


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

    validate = teacher.validate_teacher_content

    def validate_teacher_content(record: dict, output) -> None:
        validate(record, output)
        violation = context_violation(record["demographics"], [getattr(output, name) for name in LABEL_TEXTS])
        if violation:
            raise ValueError(violation)

    teacher.validate_teacher_content = validate_teacher_content


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


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def shard_paths(prefix: str) -> list[Path]:
    paths = sorted(Path(prefix).parent.glob(Path(prefix).name + "_shard*.jsonl"))
    if not paths:
        raise SystemExit(f"no Teacher shards at {prefix}_shard*.jsonl")
    return paths


def label_violations(prefix: str) -> dict[str, str | None]:
    """person_id -> violation of the current label (None when the label passes)."""
    demographics = {str(r["person_id"]): r["demographics"] for r in read_rows(INPUT_DIR / "demographics.jsonl")}
    return {
        str(row["person_id"]): context_violation(
            demographics[str(row["person_id"])], [row[name] for name in LABEL_TEXTS]
        )
        for path in shard_paths(prefix)
        for row in read_rows(path)
    }


def select_fix(prefix: str) -> None:
    chosen = {pid for pid, violation in label_violations(prefix).items() if violation}
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    for name in INPUT_NAMES:
        rows = [r for r in read_rows(INPUT_DIR / f"{name}.jsonl") if str(r["person_id"]) in chosen]
        write_rows(FIX_DIR / f"{name}.jsonl", rows)
    print(f"{len(chosen)} people whose label fails the check -> {FIX_DIR}")


def merge_fix(prefix: str, fix_prefix: str) -> None:
    fixed = {str(row["person_id"]): row for path in shard_paths(fix_prefix) for row in read_rows(path)}
    expected = {str(r["person_id"]) for r in read_rows(FIX_DIR / "demographics.jsonl")}
    if set(fixed) != expected:
        raise SystemExit(f"fix shards cover {len(fixed)} of {len(expected)} selected people; finish the Teacher run")
    for path in shard_paths(prefix):
        backup = path.with_name(path.name + ".before_fix")
        if not backup.exists():
            shutil.copy2(path, backup)
        rows = read_rows(path)
        write_rows(path, [fixed.get(str(row["person_id"]), row) for row in rows])
    left = sum(1 for violation in label_violations(prefix).values() if violation)
    print(f"replaced {len(fixed)} labels; {left} labels still fail the check")


def main() -> None:
    if sys.argv[1:] == ["--build-inputs"]:
        build_inputs()
        return
    if sys.argv[1:2] == ["--select-fix"] and len(sys.argv) == 3:
        select_fix(sys.argv[2])
        return
    if sys.argv[1:2] == ["--merge-fix"] and len(sys.argv) == 4:
        merge_fix(sys.argv[2], sys.argv[3])
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
