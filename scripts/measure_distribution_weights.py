#!/usr/bin/env python3
"""Distribution reward weights for individual : distribution = 50 : 50 within-group spread.

Same grouping as scripts/prepare_policy_updates.py: Student2 groups are a branch's
Student2 samples (one prompt), Student1 groups are a person's branches with the
branch value = mean Student1 return over its samples. Every term is centered within
its group, and the RMS over all groups is its spread.

- Student1: sequence = RMS(individual S1) / RMS(distribution_sequence).
- Student2: trip_mode and first_departure share the distribution part evenly in
  variance, so the combined distribution spread equals the individual spread
  (their within-group correlation is printed; ~0 makes the split exact):
  trip_mode = RMS(individual S2) / (sqrt(2) RMS(mode)), first_departure likewise.

With distribution.trip_distance in the config, its spread and correlations are
printed as well, and `trip_distance` is set to give it the same spread as the
first-departure term (weights of the other terms are unchanged by it).

Rollouts are rescored with the --config rewards first (as prepare_policy_updates.py
--rescore). Prints JSON with the spreads and the weights; copy the weights into
`rewards.distribution` of the GRPO train config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_policy_updates as ppu  # noqa: E402
from tripchain.common import load_config, resolve, rows_by_id  # noqa: E402


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def corr(a: list[float], b: list[float]) -> float:
    sa, sb = rms(a), rms(b)
    return sum(x * y for x, y in zip(a, b)) / len(a) / (sa * sb) if sa and sb else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", nargs="+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--min-group-size", type=int, default=2)
    args = parser.parse_args()

    people: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for source in args.rollouts:
        for line in Path(source).open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("status") in {"ok", "infeasible", ppu.UNJUDGED} and row.get("decision_trace"):
                    people[str(row["person_id"])][ppu.branch_key(row)].append(row)
    rows = [r for branches in people.values() for rs in branches.values() for r in rs]

    config = load_config(args.config)
    has_distance = bool(config["rewards"]["distribution"].get("trip_distance"))
    config["rewards"]["distribution"].update(sequence=1.0, trip_mode=1.0, first_departure=1.0)
    if has_distance:
        config["rewards"]["distribution"]["trip_distance"] = 1.0
    targets = rows_by_id(resolve(config["paths"]["ground_truth_reference"]))
    ppu.rescore(rows, config, targets)
    # Individual returns (neutralized as in prepare_policy_updates.py) before the
    # distribution terms are added; raw distribution components after.
    rewards = config["rewards"]
    individual = {}
    for branches in people.values():
        flat = [r for rs in branches.values() for r in rs]
        for r, value in zip(flat, ppu.neutral_returns(flat, rewards)):
            individual[id(r)] = value
    s2_individual = {}
    for branches in people.values():
        for rs in branches.values():
            for r, (_, s2) in zip(rs, ppu.neutral_returns(rs, rewards)):
                s2_individual[id(r)] = s2
    summary = ppu.add_distribution_rewards(rows, config, targets)

    c = lambda r, k: float(r["rewards"]["components"][k])
    s1_ind, s1_seq, s2_ind, s2_mode, s2_dep, s2_dist = [], [], [], [], [], []
    for branches in people.values():
        for rs in branches.values():
            if len(rs) < args.min_group_size:
                continue
            s2_ind += ppu.centered([s2_individual[id(r)] for r in rs])
            s2_mode += ppu.centered([c(r, "distribution_trip_mode") for r in rs])
            s2_dep += ppu.centered([c(r, "distribution_first_departure") for r in rs])
            if has_distance:
                s2_dist += ppu.centered([c(r, "distribution_trip_distance") for r in rs])
        if len(branches) < args.min_group_size:
            continue
        brs = list(branches.values())
        s1_ind += ppu.centered([sum(individual[id(r)][0] for r in rs) / len(rs) for rs in brs])
        s1_seq += ppu.centered([c(rs[0], "distribution_sequence") for rs in brs])

    spread = {
        "s1_individual": rms(s1_ind),
        "s1_sequence": rms(s1_seq),
        "s2_individual": rms(s2_ind),
        "s2_trip_mode": rms(s2_mode),
        "s2_first_departure": rms(s2_dep),
        "s2_corr_mode_departure": corr(s2_mode, s2_dep),
        "s2_corr_individual_mode": corr(s2_ind, s2_mode),
        "s2_corr_individual_departure": corr(s2_ind, s2_dep),
        "s1_corr_individual_sequence": corr(s1_ind, s1_seq),
    }
    weights = {
        "sequence": spread["s1_individual"] / spread["s1_sequence"],
        "trip_mode": spread["s2_individual"] / (math.sqrt(2) * spread["s2_trip_mode"]),
        "first_departure": spread["s2_individual"] / (math.sqrt(2) * spread["s2_first_departure"]),
    }
    if has_distance:
        spread.update(
            s2_trip_distance=rms(s2_dist),
            s2_corr_individual_distance=corr(s2_ind, s2_dist),
            s2_corr_mode_distance=corr(s2_mode, s2_dist),
            s2_corr_departure_distance=corr(s2_dep, s2_dist),
        )
        weights["trip_distance"] = (
            weights["first_departure"] * spread["s2_first_departure"] / spread["s2_trip_distance"]
        )
    print(
        json.dumps(
            {
                "people": len(people),
                "rollouts": len(rows),
                "s1_groups_values": len(s1_ind),
                "s2_groups_values": len(s2_ind),
                "spread": spread,
                "weights": weights,
                "distribution_summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
