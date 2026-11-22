#!/usr/bin/env python3
"""Convert two-level rollout groups into Student1/Student2 credit-assignment records.

Rollouts are sampled hierarchically: each person has several Student1 outputs
("branches"), and each branch has several Student2 samples drawn from the same
Student1 output, i.e. the identical Student2 prompt.

- Student2 returns are centered within one branch, so every compared
  Student2 response answers the same prompt.
- Student1 returns are centered across a person's branches. A branch's
  return is the mean Student1 return over its Student2 samples, so one lucky or
  unlucky Student2 draw does not decide the Student1 credit.

Centered values are then divided by ONE pooled std per Student, taken over all
centered values of that Student's groups, instead of each group's own std.
Groups whose samples nearly tie therefore get small advantages rather than
being blown up to unit scale, while the overall advantage scale stays ~1. This
works for any group size; a zero pooled std falls back to 1.0.

A rollout whose destinations fall outside the POI catalog (status
outside_poi_coverage, e.g. outside Seoul) has no realized schedule, so its
feasibility and start-time components are not judged rather than wrong. Their
stored value is 0, which would rank such a sample below its judged siblings and
teach the Students to avoid outside destinations. In any group containing such
a rollout (the branch for Student2, the person for Student1), the realized
term is removed from every sample's return before centering. Outside-Seoul
choices are then compared only through the Location reward (correct region:
reward, wrong: none), and cannot be used to escape an infeasible schedule
either. Groups without such a rollout keep the full return.

--rescore recomputes every rollout's rewards with the current
tripchain/rewards.py and the --config ground truth, so a reward change can be
applied to existing rollouts without rerunning the environment.

When the --config has rewards.distribution, a population-level term
(tripchain/distribution_reward.py) is added to the returns after rescoring:
Student1 gets `sequence` x log P/Q of its activity sequence, Student2 gets
`trip_mode` x the mean log P/Q of its trip modes, both within the person's
demographic cell. P is the train ground truth (paths.ground_truth_reference,
demographics from paths.test_demographics, which is the train file in GRPO
configs); Q is the policy that generated these rollouts. The mode term is not
passed on to Student1: it is decided by Student2's sampling, not by the
sequence. With `first_departure` set, Student2 also gets that weight x log P/Q
of its first departure hour (= its first home dwell), in cells that add the
first out-of-home activity. With `trip_mode_cells: distance` the trip-mode cells
also carry each trip's distance band (prior `trip_mode_prior_strength`). With
`trip_distance` set, Student2 also gets that weight x the mean log P/Q of its trips'
distance bands in trip-purpose x demographic cells (prior
`trip_distance_prior_strength`).

This does not silently treat a single sample as GRPO: groups smaller than
--min-group-size are skipped and counted.

Writes <output-dir>/student1_updates.jsonl and student2_updates.jsonl, the
inputs of scripts/train_group_relative.py.
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

from tripchain.common import load_config, resolve, rows_by_id  # noqa: E402
from tripchain.distribution_reward import (  # noqa: E402
    build_departure_rewards,
    build_distribution_rewards,
    departure_hour,
    distance_band_rewards,
)
from tripchain.rewards import score_rollout  # noqa: E402

# Rollout status without a realized schedule (a destination outside the POI catalog).
UNJUDGED = "outside_poi_coverage"


def branch_key(row: dict) -> str:
    """Identifies a Student1 branch: rollouts sharing one Student1 output."""
    return json.dumps(row["decision_trace"]["student1"]["output"], sort_keys=True, ensure_ascii=False)


def centered(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def pooled_advantages(groups: list[list[float]]) -> list[list[float]]:
    """Center each group, then scale all groups by one shared std."""
    centered_groups = [centered(values) for values in groups]
    flat = [value for values in centered_groups for value in values]
    scale = (math.sqrt(sum(value * value for value in flat) / len(flat)) if flat else 0.0) or 1.0
    return [[value / scale for value in values] for values in centered_groups]


def realized_term(row: dict, weights: dict) -> float:
    """Weighted feasibility (+ start time) of one rollout, as in rewards.score_rollout."""
    components = row["rewards"]["components"]
    value = float(weights["feasibility"]) * float(components["feasibility"])
    if "start_time" in components and "start_time" in weights:
        value += float(weights["start_time"]) * float(components["start_time"])
    return value


def neutral_returns(rows: list[dict], rewards_config: dict | None) -> list[tuple[float, float]]:
    """(student1, student2) returns; without realized terms if any row is unjudged."""
    returns = [
        (float(row["rewards"]["returns"]["student1"]), float(row["rewards"]["returns"]["student2"])) for row in rows
    ]
    if not any(row.get("status") == UNJUDGED for row in rows):
        return returns
    weights, gamma = rewards_config["weights"], float(rewards_config["future_discount"])
    # Realized outcomes reach Student2 once and Student1 twice discounted.
    return [
        (student1 - gamma**2 * realized_term(row, weights), student2 - gamma * realized_term(row, weights))
        for row, (student1, student2) in zip(rows, returns)
    ]


def rescore(rows: list[dict], config: dict, targets: dict[str, dict]) -> None:
    """Recompute stored rewards in place from each rollout's own trace."""
    for row in rows:
        environment = row["decision_trace"]["environment"]
        policy = environment.get("policy_output") or {}
        row["rewards"] = score_rollout(policy, targets[str(row["person_id"])], environment, config)


def add_distribution_rewards(rows: list[dict], config: dict, targets: dict[str, dict]) -> dict:
    """Add the population-level term to each row's returns in place."""
    settings = config["rewards"]["distribution"]
    people = rows_by_id(resolve(config["paths"]["test_demographics"]))
    truth = [
        (
            people[person_id].get("demographics", people[person_id]),
            target.get("activity_sequence") or [],
            target.get("trip_modes") or [],
        )
        for person_id, target in targets.items()
        if person_id in people
    ]
    # trip_mode_cells: distance -> trip-mode cells add the trip's distance band
    # (tripchain/trip_distance.py), from GT dong codes and Student2's own locations.
    distance = settings.get("trip_mode_cells", "demographic") == "distance"
    distance_weight = float(settings.get("trip_distance", 0.0))
    if distance or distance_weight:
        # Imported here because it loads pandas and the POI catalog, which only these options need.
        from tripchain.trip_distance import TripDistance

        bands = TripDistance.from_config(config)
    if distance:
        truth = [
            (*item, bands.day_bands(targets[person_id].get("activities") or []))
            for item, person_id in zip(truth, [pid for pid in targets if pid in people])
        ]
    branch_index: dict[tuple[str, str], int] = {}
    branches = []
    for row in rows:
        key = (str(row["person_id"]), branch_key(row))
        if key not in branch_index:
            student1 = row["decision_trace"]["student1"]
            branch_index[key] = len(branches)
            branches.append((student1["input"]["demographics"], student1["output"]["activity_sequence"]))
    generated = []
    for row in rows:
        policy = row["decision_trace"]["environment"].get("policy_output") or {}
        item = (row["decision_trace"]["student1"]["input"]["demographics"], policy.get("trip_modes") or [])
        if distance:
            item = (*item, bands.day_bands(policy.get("location_dwell_by_activity") or []))
        generated.append(item)
    mode_prior = settings.get("trip_mode_prior_strength")
    sequence_rewards, mode_rewards = build_distribution_rewards(
        branches,
        generated,
        truth,
        float(settings.get("prior_strength", 20.0)),
        float(settings.get("clip", 3.0)),
        None if mode_prior is None else float(mode_prior),
    )
    sequence_weight, mode_weight = float(settings["sequence"]), float(settings["trip_mode"])
    departure_weight = float(settings.get("first_departure", 0.0))
    departure_rewards = [0.0] * len(rows)
    if departure_weight:
        departure_truth = []
        for person_id, target in targets.items():
            activities = target.get("activities") or []
            if person_id in people and len(activities) > 1:
                departure_truth.append(
                    (
                        people[person_id].get("demographics", people[person_id]),
                        str(activities[1]["activity_type"]),
                        departure_hour(
                            float(activities[0].get("start_time_min", 0)) + float(activities[0]["dwell_time_min"])
                        ),
                    )
                )
        departure_rewards = build_departure_rewards(
            [generated_departure(row) for row in rows],
            departure_truth,
            float(settings.get("prior_strength", 20.0)),
            float(settings.get("clip", 3.0)),
        )
    distance_rewards = [0.0] * len(rows)
    if distance_weight:
        # A trip's purpose is the activity it arrives at.
        distance_truth = []
        for person_id, target in targets.items():
            activities = target.get("activities") or []
            if person_id in people and len(activities) > 1:
                distance_truth.append(
                    (
                        people[person_id].get("demographics", people[person_id]),
                        [str(item["activity_type"]) for item in activities[1:]],
                        bands.day_bands(activities),
                    )
                )
        generated_distance = []
        for row in rows:
            student1 = row["decision_trace"]["student1"]
            policy = row["decision_trace"]["environment"].get("policy_output") or {}
            sequence = policy.get("activity_sequence") or student1["output"]["activity_sequence"]
            spatial = policy.get("location_dwell_by_activity") or []
            generated_distance.append(
                (
                    student1["input"]["demographics"],
                    [str(activity) for activity in sequence[1 : len(spatial)]],
                    bands.day_bands(spatial),
                )
            )
        distance_prior = settings.get("trip_distance_prior_strength", settings.get("prior_strength", 20.0))
        distance_rewards = distance_band_rewards(
            generated_distance, distance_truth, float(distance_prior), float(settings.get("clip", 3.0))
        )
    for row, mode_reward, departure_reward, distance_reward in zip(
        rows, mode_rewards, departure_rewards, distance_rewards
    ):
        sequence_reward = sequence_rewards[branch_index[(str(row["person_id"]), branch_key(row))]]
        row["rewards"]["components"]["distribution_sequence"] = sequence_reward
        row["rewards"]["components"]["distribution_trip_mode"] = mode_reward
        returns = row["rewards"]["returns"]
        returns["student1"] = float(returns["student1"]) + sequence_weight * sequence_reward
        returns["student2"] = float(returns["student2"]) + mode_weight * mode_reward
        if departure_weight:
            row["rewards"]["components"]["distribution_first_departure"] = departure_reward
            returns["student2"] = float(returns["student2"]) + departure_weight * departure_reward
        if distance_weight:
            row["rewards"]["components"]["distribution_trip_distance"] = distance_reward
            returns["student2"] = float(returns["student2"]) + distance_weight * distance_reward
    return {
        "distribution_truth_people": len(truth),
        "distribution_branches": len(branches),
        "distribution_sequence_mean": sum(sequence_rewards) / max(1, len(sequence_rewards)),
        "distribution_trip_mode_mean": sum(mode_rewards) / max(1, len(mode_rewards)),
        "distribution_first_departure_mean": sum(departure_rewards) / max(1, len(departure_rewards)),
        "distribution_trip_distance_mean": sum(distance_rewards) / max(1, len(distance_rewards)),
    }


def generated_departure(row: dict) -> tuple[dict, str, str] | None:
    """(demographics, first out-of-home activity, departure hour) of one rollout.

    The day starts at home at the day origin, so Student2's first home dwell is
    the departure time (the realized schedule keeps it unchanged)."""
    student1 = row["decision_trace"]["student1"]
    policy = row["decision_trace"]["environment"].get("policy_output") or {}
    sequence = policy.get("activity_sequence") or student1["output"]["activity_sequence"]
    spatial = policy.get("location_dwell_by_activity") or []
    if len(sequence) < 2 or not spatial:
        return None
    return (
        student1["input"]["demographics"],
        str(sequence[1]),
        departure_hour(float(spatial[0].get("dwell_time_min", 0))),
    )


def update_record(person_id: str, row: dict, student: str, value: float, advantage: float, **extra) -> dict:
    """One training record: the Student's stored prompt and answer with its return and advantage."""
    trace = row["decision_trace"][student]
    if not isinstance(trace.get("input"), dict):
        raise ValueError(
            f"person_id={person_id} rollout_id={row.get('rollout_id')}: "
            f"{student} trace has no stored prompt input; regenerate Student "
            "outputs and rollouts with the current inference code"
        )
    return {
        "person_id": person_id,
        "rollout_id": row.get("rollout_id"),
        "messages": [
            {"role": "user", "content": json.dumps(trace["input"], ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(trace["output"], ensure_ascii=False, separators=(",", ":"))},
        ],
        "return": value,
        "group_advantage": advantage,
        "reward_components": row["rewards"]["components"],
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument(
        "--config",
        required=True,
        help="GRPO train config: reward weights, discount, distribution term and ground truth",
    )
    parser.add_argument(
        "--rescore", action="store_true", help="recompute rewards with the current reward code before grouping"
    )
    args = parser.parse_args()
    people: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for source in args.rollouts:
        with Path(source).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") in {"ok", "infeasible", "outside_poi_coverage"} and row.get("decision_trace"):
                        people[str(row["person_id"])][branch_key(row)].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        name: (output_dir / f"{name}_updates.jsonl").open("w", encoding="utf-8") for name in ("student1", "student2")
    }
    counts = {"student1": 0, "student2": 0, "skipped_student1_people": 0, "skipped_student2_branches": 0}
    # Groups are collected first because the pooled std needs every group.
    all_rows = [row for branches in people.values() for rows in branches.values() for row in rows]
    rewards_config = None
    config = load_config(args.config)
    distribution = "distribution" in config["rewards"]
    if args.rescore or distribution or any(row.get("status") == UNJUDGED for row in all_rows):
        rewards_config = config["rewards"]
        targets = rows_by_id(resolve(config["paths"]["ground_truth_reference"]))
        if args.rescore:
            rescore(all_rows, config, targets)
        if distribution:
            # After rescoring, which would overwrite the added terms.
            counts.update(add_distribution_rewards(all_rows, config, targets))
    counts["unjudged_rollouts"] = sum(
        row.get("status") == UNJUDGED for branches in people.values() for rows in branches.values() for row in rows
    )
    student2_groups: list[tuple[str, list[dict], list[float]]] = []
    student1_groups: list[tuple[str, list[list[dict]], list[float]]] = []
    for person_id, branches in people.items():
        for rows in branches.values():
            if len(rows) < args.min_group_size:
                counts["skipped_student2_branches"] += 1
                continue
            returns = [student2 for _, student2 in neutral_returns(rows, rewards_config)]
            student2_groups.append((person_id, rows, returns))

        if len(branches) < args.min_group_size:
            counts["skipped_student1_people"] += 1
            continue
        branch_rows = list(branches.values())
        # Neutralize over all of the person's rollouts, then average per branch.
        flat = [row for rows in branch_rows for row in rows]
        student1_by_row = {id(row): student1 for row, (student1, _) in zip(flat, neutral_returns(flat, rewards_config))}
        means = [sum(student1_by_row[id(row)] for row in rows) / len(rows) for rows in branch_rows]
        student1_groups.append((person_id, branch_rows, means))

    try:
        student2_advantages = pooled_advantages([returns for _, _, returns in student2_groups])
        for (person_id, rows, returns), group_advantages in zip(student2_groups, student2_advantages):
            for row, value, advantage in zip(rows, returns, group_advantages):
                record = update_record(person_id, row, "student2", value, advantage)
                handles["student2"].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["student2"] += 1

        student1_advantages = pooled_advantages([means for _, _, means in student1_groups])
        for (person_id, branch_rows, means), group_advantages in zip(student1_groups, student1_advantages):
            for rows, value, advantage in zip(branch_rows, means, group_advantages):
                # Every row in a branch shares the same Student1 prompt and output.
                record = update_record(person_id, rows[0], "student1", value, advantage, branch_size=len(rows))
                handles["student1"].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["student1"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
