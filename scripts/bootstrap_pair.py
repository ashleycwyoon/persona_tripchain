#!/usr/bin/env python3
"""Paired person-level bootstrap between two rollout populations on the same people.

Metrics: exact activity-sequence match, LCS F1, location (administrative dong)
and dwell within 30 minutes on LCS-matched activities, trip-mode accuracy, and
the subgroup-mean JSDs of tripchain/population_metrics.py. Decisions are read
from each rollout's final decision_trace, as in scripts/evaluate_population.py.
Every resample draws the same people for A and B, so the interval of B - A
reflects only which people were tested.

Usage: bootstrap_pair.py --truth GT.jsonl --a ROLLOUTS_A --b ROLLOUTS_B --output OUT.json
       [--labels Pre-RL,Round1] [--resamples 500]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_population import location_key, realized_timing, truth_timing  # noqa: E402
from tripchain.population_metrics import population_metrics  # noqa: E402
from tripchain.rewards import lcs_pairs  # noqa: E402

JSD_KEYS = [
    "sequence_pattern",
    "activity_count",
    "activity_type",
    "trip_mode",
    "dwell_time",
    "first_departure_hour",
    "travel_time",
]

# Per-person records of both populations and their shared people. Set in main()
# before the worker pool starts, so the (forked) workers inherit them.
A: dict = {}
B: dict = {}
SHARED: list[str] = []


def read(path: Path) -> dict[str, dict]:
    """JSONL rows keyed by person_id."""
    rows = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    return {str(row["person_id"]): row for row in rows}


def people_for(rollouts: Path, truth: dict[str, dict]) -> dict[str, dict]:
    """Per-person match records of one rollout file against the ground truth."""
    out = {}
    for pid, row in read(rollouts).items():
        trace = row.get("decision_trace") or {}
        o1 = (trace.get("student1") or {}).get("output") or {}
        o2 = (trace.get("student2") or {}).get("output") or {}
        if pid not in truth or "activity_sequence" not in o1 or "location_dwell_by_activity" not in o2:
            continue
        target = truth[pid]
        pseq, tseq = o1["activity_sequence"], target["activity_sequence"]
        pairs = lcs_pairs(pseq, tseq)
        loc, dwell, mode = [], [], []
        for pi, ti in pairs:
            if pi < len(o2["location_dwell_by_activity"]) and ti < len(target["activities"]):
                p_item, t_item = o2["location_dwell_by_activity"][pi], target["activities"][ti]
                loc.append(location_key(p_item["location"]) == location_key(t_item["location"]))
                dwell.append(abs(float(p_item["dwell_time_min"]) - float(t_item["dwell_time_min"])) <= 30)
            if 0 < pi <= len(o2.get("trip_modes") or []) and 0 < ti <= len(target.get("trip_modes") or []):
                mode.append(o2["trip_modes"][pi - 1] == target["trip_modes"][ti - 1])
        truth_extra, pred_extra = {}, {}
        if "trip_modes" in o2 and "trip_modes" in target:
            truth_extra["trip_mode"], pred_extra["trip_mode"] = target["trip_modes"], o2["trip_modes"]
        realized = realized_timing(row)
        if realized:
            truth_times = truth_timing(target)
            for name in set(truth_times) | set(realized):
                truth_extra[name], pred_extra[name] = truth_times.get(name, []), realized.get(name, [])
        demo = ((trace.get("student1") or {}).get("input") or {}).get("demographics") or {}
        out[pid] = {
            "exact": float(pseq == tseq),
            "lcs": 2 * len(pairs) / max(1, len(pseq) + len(tseq)),
            "loc": loc,
            "dwell": dwell,
            "mode": mode,
            "dist": {
                "demographics": demo,
                "truth": tseq,
                "pred": pseq,
                "truth_extra": truth_extra,
                "pred_extra": pred_extra,
            },
        }
    return out


def summarize(people: list[dict]) -> dict[str, float]:
    """Metric values of one (resampled) population."""
    flat = lambda key: [v for p in people for v in p[key]]
    dist = population_metrics([p["dist"] for p in people])["subgroup_mean"]
    return {
        "exact": float(np.mean([p["exact"] for p in people])),
        "lcs": float(np.mean([p["lcs"] for p in people])),
        "location": float(np.mean(flat("loc"))),
        "dwell30": float(np.mean(flat("dwell"))),
        "trip_mode": float(np.mean(flat("mode"))),
        **{f"jsd_{k}": float(dist[k]) for k in JSD_KEYS if k in dist},
    }


def draw(seed: int) -> dict[str, float]:
    """B - A on one resample of the shared people."""
    rng = np.random.default_rng(seed)
    sample = [SHARED[i] for i in rng.integers(0, len(SHARED), len(SHARED))]
    a, b = summarize([A[p] for p in sample]), summarize([B[p] for p in sample])
    return {m: b[m] - a[m] for m in a}


def main() -> None:
    global A, B, SHARED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", default="A,B")
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    truth = read(Path(args.truth))
    A, B = people_for(Path(args.a), truth), people_for(Path(args.b), truth)
    SHARED = sorted(set(A) & set(B))
    point_a, point_b = summarize([A[p] for p in SHARED]), summarize([B[p] for p in SHARED])
    with ProcessPoolExecutor(args.workers) as pool:
        draws = list(pool.map(draw, range(args.resamples)))
    pct = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    label_a, label_b = args.labels.split(",")
    result = {
        "people": len(SHARED),
        "resamples": args.resamples,
        "a": label_a,
        "b": label_b,
        "point": {label_a: point_a, label_b: point_b},
        "diff": {m: {"point": point_b[m] - point_a[m], "ci": pct([d[m] for d in draws])} for m in point_a},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{label_b} - {label_a}, {len(SHARED)} people, {args.resamples} resamples")
    for m, d in result["diff"].items():
        sig = "*" if d["ci"][0] > 0 or d["ci"][1] < 0 else ""
        print(
            f"{m:26s} {point_a[m]:.3f} -> {point_b[m]:.3f}  {d['point']:+.4f} "
            f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] {sig}"
        )


if __name__ == "__main__":
    main()
