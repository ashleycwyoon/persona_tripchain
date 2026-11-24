"""Locations of data, outputs and model weights, all kept outside the code checkout.

The repository holds code only. Datasets live under DATA_ROOT and experiment
outputs (runs, reports, logs) under OUTPUT_ROOT; inputs shared with the upstream
stage (KTDB 2021 raw microdata, the administrative-code workbook, the OSM
extract) sit directly under SHARED_DATA_ROOT.

    SHARED_DATA_ROOT = $DATA_HOME                      (default ~/data)
    DATA_ROOT        = $DATA_HOME/tripchain_1030
    OUTPUT_ROOT      = $OUTPUT_HOME/tripchain_1030     (default ~/outputs)
    HF_HUB_CACHE     = $HF_HUB_CACHE                   (default: the Hugging Face hub cache)

Set DATA_HOME / OUTPUT_HOME to move the whole storage tree elsewhere, and
HF_HUB_CACHE (or HF_HOME) to choose where model weights are downloaded.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT = "tripchain_1030"

SHARED_DATA_ROOT = Path(os.environ.get("DATA_HOME", Path.home() / "data")).expanduser()
OUTPUT_HOME = Path(os.environ.get("OUTPUT_HOME", Path.home() / "outputs")).expanduser()
DATA_ROOT = SHARED_DATA_ROOT / PROJECT
OUTPUT_ROOT = OUTPUT_HOME / PROJECT

# The first stage (KTDB preprocessing and the Teacher labels of the v1 split) keeps
# its historical folders: data under DATA_HOME/tripchain_1015, Teacher labels under
# OUTPUT_HOME/tripchain_1015/persona-rationale-multi-country-qwen35.
TRIPCHAIN_1015_DATA = SHARED_DATA_ROOT / "tripchain_1015"
TEACHER_OUTPUT_ROOT = OUTPUT_HOME / "tripchain_1015" / "persona-rationale-multi-country-qwen35"


def _hf_hub_cache() -> Path:
    """The Hugging Face hub cache, resolved the way huggingface_hub resolves it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    hf_home = os.environ.get("HF_HOME") or Path(cache_home) / "huggingface"
    return Path(hf_home).expanduser() / "hub"


HF_HUB_CACHE = _hf_hub_cache()

_ROOTS = {
    "DATA_ROOT": DATA_ROOT,
    "OUTPUT_ROOT": OUTPUT_ROOT,
    "SHARED_DATA_ROOT": SHARED_DATA_ROOT,
    "DATA_HOME": SHARED_DATA_ROOT,
    "OUTPUT_HOME": OUTPUT_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
}
_VAR = re.compile(r"\$\{(" + "|".join(_ROOTS) + r")\}")


def expand(value: str) -> str:
    """Replace ${DATA_ROOT}, ${OUTPUT_ROOT}, ${SHARED_DATA_ROOT}, ${DATA_HOME}, ${OUTPUT_HOME}
    and ${HF_HUB_CACHE} in a config value."""
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
    (configs/, ...) stays under the repo root.
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
