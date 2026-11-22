#!/usr/bin/env python3
"""Student2 SFT: P(reasoning2, location_dwell_by_activity, trip_modes | demographics, Student1 output).

Runs the upstream QLoRA trainer (third_party/tripchain_1015/
persona-rationale-multi-country-qwen35/scripts/20b_train_ktdb_student2.py)
unchanged except for its row check, which expects a `mode_preference` target:
`trip_modes` (one main mode per trip) is validated here and the rest of the row
goes through the original check. The trainer's options pass through: --config,
--gpu, --dry-run, --max-train-rows and --resume-from-checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.inference import MODE_LABELS  # noqa: E402
from tripchain.paths import TRIPCHAIN_1015_CODE  # noqa: E402

SOURCE = TRIPCHAIN_1015_CODE / "persona-rationale-multi-country-qwen35/scripts/20b_train_ktdb_student2.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("tripchain_1015_student2_trainer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Student2 trainer: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.validate_row

    def validate_row(row: dict, source: Path, line_number: int) -> None:
        user = json.loads(row["messages"][0]["content"])
        target = json.loads(row["messages"][1]["content"])
        trip_modes = target.pop("trip_modes", None)
        if not isinstance(trip_modes, list) or len(trip_modes) != len(user["activity_sequence"]) - 1:
            raise ValueError(f"{source}:{line_number}: trip_modes must name one mode per trip")
        if any(mode not in MODE_LABELS for mode in trip_modes):
            raise ValueError(f"{source}:{line_number}: unknown trip mode in {trip_modes}")
        target["mode_preference"] = []
        shadow = {**row, "messages": [row["messages"][0], {**row["messages"][1], "content": json.dumps(target)}]}
        original(shadow, source, line_number)

    module.validate_row = validate_row
    module.main()


if __name__ == "__main__":
    main()
