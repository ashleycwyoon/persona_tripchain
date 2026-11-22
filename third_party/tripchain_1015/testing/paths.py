"""Locations of data and outputs, kept outside the code checkout.

The repository holds code only. Datasets live under DATA_ROOT and experiment
outputs (runs, reports, logs) under OUTPUT_ROOT; datasets shared by several
projects (e.g. KTDB 2021 raw microdata) sit directly under SHARED_DATA_ROOT.

    SHARED_DATA_ROOT = $DATA_HOME                      (default ~/data)
    DATA_ROOT        = $DATA_HOME/tripchain_1015
    OUTPUT_ROOT      = $OUTPUT_HOME/tripchain_1015          (default ~/outputs)

Set DATA_HOME / OUTPUT_HOME to move the whole storage tree elsewhere.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT = "tripchain_1015"

SHARED_DATA_ROOT = Path(os.environ.get("DATA_HOME", Path.home() / "data")).expanduser()
OUTPUT_HOME = Path(os.environ.get("OUTPUT_HOME", Path.home() / "outputs")).expanduser()
DATA_ROOT = SHARED_DATA_ROOT / PROJECT
OUTPUT_ROOT = OUTPUT_HOME / PROJECT

_ROOTS = {
    "DATA_ROOT": DATA_ROOT,
    "OUTPUT_ROOT": OUTPUT_ROOT,
    "SHARED_DATA_ROOT": SHARED_DATA_ROOT,
    "DATA_HOME": SHARED_DATA_ROOT,
    "OUTPUT_HOME": OUTPUT_HOME,
}
_VAR = re.compile(r"\$\{(" + "|".join(_ROOTS) + r")\}")


def expand(value: str) -> str:
    """Replace ${DATA_ROOT}, ${OUTPUT_ROOT}, ${SHARED_DATA_ROOT}, ${DATA_HOME}, ${OUTPUT_HOME} in a config value."""
    return _VAR.sub(lambda m: str(_ROOTS[m.group(1)]), value)


def expand_config(obj):
    """Apply expand() to every string in a loaded YAML/JSON config."""
    if isinstance(obj, str):
        return expand(obj)
    if isinstance(obj, dict):
        return {k: expand_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_config(v) for v in obj]
    return obj


def locate(path: str | Path, root: Path) -> Path:
    """Resolve a repo-relative path, sending data/output locations outside the repo.

    "outputs/X" -> OUTPUT_ROOT/X, "runs/X" / "logs/X" -> OUTPUT_ROOT/runs/X / logs/X,
    "data/processed/X" / "data/raw/X" -> DATA_ROOT/processed/X / raw/X; anything else
    (configs/, prompts/, data/time_windows.json, ...) stays under the repo root.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "outputs":
        return OUTPUT_ROOT.joinpath(*parts[1:])
    if parts and parts[0] in ("runs", "logs"):
        return OUTPUT_ROOT / path
    if len(parts) > 1 and parts[0] == "data" and parts[1] in ("processed", "raw"):
        return DATA_ROOT.joinpath(*parts[1:])
    return root / path
