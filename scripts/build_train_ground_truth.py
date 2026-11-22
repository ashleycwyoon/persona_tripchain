#!/usr/bin/env python3
"""Build the v1 train-split ground truth reference.

The upstream preprocessing (third_party/tripchain_1015/scripts/prepare_ktdb_training_data.py)
writes a ground truth reference for the test split only. This joins its train activity
sequences and spatial targets into the same format,
{"person_id", "activity_sequence", **location_dwell_mode_preference}, ordered like the
residence-augmented train demographics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tripchain.paths import DATA_ROOT, TRIPCHAIN_1015_DATA  # noqa: E402

SOURCE = TRIPCHAIN_1015_DATA / "processed/train"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", default=str(SOURCE / "activity_sequence.jsonl"))
    parser.add_argument("--spatial", default=str(SOURCE / "location_dwell_mode_preference.jsonl"))
    parser.add_argument("--order", default=str(DATA_ROOT / "processed/train/demographics_with_residence.jsonl"))
    parser.add_argument("--output", default=str(DATA_ROOT / "processed/train/ground_truth_reference.jsonl"))
    args = parser.parse_args()

    sequences = {str(row["person_id"]): row["activity_sequence"] for row in read_jsonl(Path(args.sequences))}
    spatial = {str(row["person_id"]): row for row in read_jsonl(Path(args.spatial))}
    order = [str(row["person_id"]) for row in read_jsonl(Path(args.order))]

    missing = [pid for pid in order if pid not in sequences or pid not in spatial]
    if missing:
        raise ValueError(f"{len(missing)} people lack targets, e.g. {missing[:5]}")

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for pid in order:
            rest = {k: v for k, v in spatial[pid].items() if k != "person_id"}
            if [a["activity_type"] for a in rest["activities"]] != sequences[pid]:
                raise ValueError(f"activity sequence mismatch for {pid}")
            record = {"person_id": pid, "activity_sequence": sequences[pid], **rest}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"people": len(order), "first_id": order[0], "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
