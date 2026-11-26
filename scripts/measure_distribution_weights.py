#!/usr/bin/env python3
"""Distribution reward weights for individual : distribution = 50 : 50 spread in the advantage.

Same grouping and centering as scripts/prepare_policy_updates.py: Student2 groups are a
branch's Student2 samples (one prompt), Student1 groups are a person's branches with the
branch value = mean Student1 return over its samples. The individual return is centered
within its group; a distribution term is centered within the person's demographic cell
(occupation x age x gender) and not within the group. The RMS over all group members is
the spread of a term, i.e. its size in the advantage before the shared scaling.

`spread_within_group` gives the distribution spreads with the earlier centering (within
the group), which the weights of the runs before v2ctxt_s1cell were measured with. With
--weights-from-config the output also has `individual_share`: the share of the individual
return in the variance of the advantage under the weights written in --config.

- Student1: sequence = RMS(individual S1) / RMS(distribution_sequence). With
  distribution.activity_count in the config, the two Student1 terms get the same spread
  and their sum the individual spread, using their measured correlation:
  each = RMS(individual S1) / (sqrt(2 + 2 corr) RMS(term)).
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
    parser.add_argument(
        "--weights-from-config",
        action="store_true",
        help="also report the individual share of the advantage variance under the --config weights",
    )
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
    has_count = bool(config["rewards"]["distribution"].get("activity_count"))
    configured = {
        name: float(config["rewards"]["distribution"].get(name) or 0.0)
        for name in ("sequence", "activity_count", "trip_mode", "first_departure", "trip_distance")
    }
    config["rewards"]["distribution"].update(sequence=1.0, trip_mode=1.0, first_departure=1.0)
    if has_distance:
        config["rewards"]["distribution"]["trip_distance"] = 1.0
    if has_count:
        config["rewards"]["distribution"]["activity_count"] = 1.0
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

    raw = lambda r, k: float(r["rewards"]["components"][k])
    names = ["distribution_sequence", "distribution_trip_mode", "distribution_first_departure"]
    if has_distance:
        names.append("distribution_trip_distance")
    if has_count:
        names.append("distribution_activity_count")
    # Cell means over all rollouts, as add_distribution_rewards() centers the terms.
    cell_of = {id(r): ppu.cell_values(r["decision_trace"]["student1"]["input"]["demographics"]) for r in rows}
    cell_of = {key: (v["occupation"], v["age"], v["gender"]) for key, v in cell_of.items()}
    totals: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        for name in names:
            total = totals[(name, cell_of[id(r)])]
            total[0] += raw(r, name)
            total[1] += 1.0
    c = lambda r, k: raw(r, k) - totals[(k, cell_of[id(r)])][0] / totals[(k, cell_of[id(r)])][1]

    s1_ind, s1_seq, s2_ind, s2_mode, s2_dep, s2_dist = [], [], [], [], [], []
    old_seq, old_mode, old_dep, old_dist = [], [], [], []
    s1_count: list[float] = []
    for branches in people.values():
        for rs in branches.values():
            if len(rs) < args.min_group_size:
                continue
            s2_ind += ppu.centered([s2_individual[id(r)] for r in rs])
            s2_mode += [c(r, "distribution_trip_mode") for r in rs]
            s2_dep += [c(r, "distribution_first_departure") for r in rs]
            old_mode += ppu.centered([raw(r, "distribution_trip_mode") for r in rs])
            old_dep += ppu.centered([raw(r, "distribution_first_departure") for r in rs])
            if has_distance:
                s2_dist += [c(r, "distribution_trip_distance") for r in rs]
                old_dist += ppu.centered([raw(r, "distribution_trip_distance") for r in rs])
        if len(branches) < args.min_group_size:
            continue
        brs = list(branches.values())
        s1_ind += ppu.centered([sum(individual[id(r)][0] for r in rs) / len(rs) for rs in brs])
        s1_seq += [c(rs[0], "distribution_sequence") for rs in brs]
        if has_count:
            s1_count += [c(rs[0], "distribution_activity_count") for rs in brs]
        old_seq += ppu.centered([raw(rs[0], "distribution_sequence") for rs in brs])

    spread_within_group = {
        "s1_sequence": rms(old_seq),
        "s2_trip_mode": rms(old_mode),
        "s2_first_departure": rms(old_dep),
    }
    if has_distance:
        spread_within_group["s2_trip_distance"] = rms(old_dist)
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
    if has_count:
        both = math.sqrt(2 + 2 * corr(s1_seq, s1_count))
        spread.update(
            s1_activity_count=rms(s1_count),
            s1_corr_sequence_count=corr(s1_seq, s1_count),
            s1_corr_individual_count=corr(s1_ind, s1_count),
        )
        weights["sequence"] = spread["s1_individual"] / (both * spread["s1_sequence"])
        weights["activity_count"] = spread["s1_individual"] / (both * spread["s1_activity_count"])
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
    result = {
        "people": len(people),
        "rollouts": len(rows),
        "s1_groups_values": len(s1_ind),
        "s2_groups_values": len(s2_ind),
        "spread": spread,
        "spread_within_group": spread_within_group,
        "weights": weights,
        "distribution_summary": summary,
    }
    if args.weights_from_config:
        s1_total = [a + configured["sequence"] * b for a, b in zip(s1_ind, s1_seq)]
        if has_count:
            s1_total = [a + configured["activity_count"] * b for a, b in zip(s1_total, s1_count)]
        s2_parts = [(configured["trip_mode"], s2_mode), (configured["first_departure"], s2_dep)]
        if has_distance:
            s2_parts.append((configured["trip_distance"], s2_dist))
        s2_total = [
            value + sum(weight * part[index] for weight, part in s2_parts) for index, value in enumerate(s2_ind)
        ]
        result["configured_weights"] = configured
        result["individual_share"] = {
            "student1": (rms(s1_ind) / rms(s1_total)) ** 2,
            "student2": (rms(s2_ind) / rms(s2_total)) ** 2,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
