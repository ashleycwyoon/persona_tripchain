#!/usr/bin/env python3
"""Run Student1 or Student2 inference on one GPU (tripchain/inference.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tripchain.inference import main  # noqa: E402

if __name__ == "__main__":
    main()
