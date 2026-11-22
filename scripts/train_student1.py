#!/usr/bin/env python3
"""Student1 SFT: P(persona_bundle, reasoning1, activity_sequence | demographics).

Runs the upstream QLoRA trainer unchanged (third_party/tripchain_1015/
persona-rationale-multi-country-qwen35/scripts/20a_train_ktdb_student1.py).
Its options pass through: --config, --gpu, --dry-run (validate the data and the
assistant-only loss mask without training), --max-train-rows and
--resume-from-checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.paths import TRIPCHAIN_1015_CODE  # noqa: E402

SOURCE = TRIPCHAIN_1015_CODE / "persona-rationale-multi-country-qwen35/scripts/20a_train_ktdb_student1.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("tripchain_1015_student1_trainer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Student1 trainer: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
