"""Dense individual rewards of one rollout and the discounted return of each Student.

Components: Activity (sequence overlap), Location (administrative dong),
Dwell, Trip mode (or the person-level mode ranking), Feasibility and Start
time. Location, dwell, trip mode and start time compare LCS-matched activities;
unmatched activities score 0.
"""

from __future__ import annotations

import math
from typing import Any


def lcs_pairs(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    """Index pairs of one longest common activity subsequence.

    Shared by the dense rewards and scripts/evaluate_population.py so training
    and reporting align predicted and observed activities the same way.
    """
    n, m = len(left), len(right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = 1 + dp[i + 1][j + 1] if left[i] == right[j] else max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def activity_reward(predicted: list[str], target: list[str]) -> float:
    """LCS F1 of the predicted and observed activity sequences."""
    if not predicted or not target:
        return 0.0
    return 2.0 * len(lcs_pairs(predicted, target)) / (len(predicted) + len(target))


def _location_parts(location: dict[str, Any]) -> tuple[str, str]:
    dong = str(location.get("administrative_dong_name") or location.get("administrative_dong_code") or "").casefold()
    gu = str(location.get("gu") or location.get("sigungu") or "").casefold()
    return dong, gu


def _aligned(
    predicted_sequence: list[str],
    target_sequence: list[str],
    predicted: list[dict[str, Any]],
    target: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(predicted item, target item) for every LCS-matched activity pair."""
    return [
        (predicted[pi], target[ti])
        for pi, ti in lcs_pairs(predicted_sequence, target_sequence)
        if pi < len(predicted) and ti < len(target)
    ]


def is_external(location: dict[str, Any]) -> bool:
    """Outside Seoul: a Student2 'external' choice, or an observed non-Seoul sido."""
    if location.get("scope"):
        return location["scope"] == "external"
    return str(location.get("sido") or "Seoul").casefold() != "seoul"


def location_match(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    """1 at the finest level Student2 can name, 0.5 one level up, 0 otherwise.

    Inside Seoul the finest level is the administrative dong (then gu). Outside
    Seoul Student2 can only name sido/sigungu, so sigungu is the finest level
    (then sido). Choosing Seoul for an outside destination, or the reverse,
    scores 0, so "outside Seoul" is scored like any other choice.
    """
    if is_external(predicted) != is_external(target):
        return 0.0
    pdong, pgu = _location_parts(predicted)
    tdong, tgu = _location_parts(target)
    if is_external(predicted):
        psido = str(predicted.get("sido") or "").casefold()
        tsido = str(target.get("sido") or "").casefold()
        if pgu and pgu == tgu:
            return 1.0
        return 0.5 if psido and psido == tsido else 0.0
    if pdong and pdong == tdong and pgu == tgu:
        return 1.0
    return 0.5 if pgu and pgu == tgu else 0.0


def location_reward(
    predicted_sequence: list[str],
    target_sequence: list[str],
    predicted: list[dict[str, Any]],
    target: list[dict[str, Any]],
) -> float:
    """Score locations on LCS-matched activities; unmatched activities score 0."""
    if not predicted or not target:
        return 0.0
    scores = [
        location_match(pitem.get("location") or {}, titem.get("location") or {})
        for pitem, titem in _aligned(predicted_sequence, target_sequence, predicted, target)
    ]
    return sum(scores) / max(len(predicted), len(target))


def dwell_reward(
    predicted_sequence: list[str],
    target_sequence: list[str],
    predicted: list[dict[str, Any]],
    target: list[dict[str, Any]],
    tolerance_min: float,
    final_home_residual: bool = False,
) -> float:
    """Mean exp(-|dwell error| / tolerance) over LCS-matched activities."""
    if final_home_residual:
        # The environment sets the last stay (home or, on a day that ends away
        # from home, any activity) to the rest of the day, so neither side's
        # final dwell is a prediction to score.
        if len(predicted_sequence) > 1:
            predicted_sequence, predicted = predicted_sequence[:-1], predicted[:-1]
        if len(target_sequence) > 1:
            target_sequence, target = target_sequence[:-1], target[:-1]
    if not predicted or not target:
        return 0.0
    errors = [
        abs(float(p.get("dwell_time_min", 0)) - float(t.get("dwell_time_min", 0)))
        for p, t in _aligned(predicted_sequence, target_sequence, predicted, target)
    ]
    # Bounded smooth reward; unmatched activities receive zero through the
    # denominator rather than letting a short prediction look artificially good.
    similarities = [math.exp(-error / max(1.0, tolerance_min)) for error in errors]
    return sum(similarities) / max(len(predicted), len(target))


def trip_mode_reward(
    predicted_sequence: list[str],
    target_sequence: list[str],
    predicted_modes: list[str],
    target_modes: list[str],
) -> float:
    """Share of trips whose main mode matches, aligned like the activities.

    Trip i arrives at activity i+1, so a trip is matched through the LCS pair
    of the activities it arrives at; unmatched trips count through the
    denominator, as in the location and dwell rewards.
    """
    if not predicted_modes and not target_modes:
        return 1.0
    if not predicted_modes or not target_modes:
        return 0.0
    hits = sum(
        predicted_modes[pi - 1] == target_modes[ti - 1]
        for pi, ti in lcs_pairs(predicted_sequence, target_sequence)
        if 0 < pi <= len(predicted_modes) and 0 < ti <= len(target_modes)
    )
    return hits / max(len(predicted_modes), len(target_modes))


def mode_preference_reward(
    predicted: list[dict[str, Any]], target: list[dict[str, Any]], persistence: float = 0.8
) -> float:
    """Rank-biased overlap for observed person-level mode rankings."""
    p = [str(item.get("mode")) for item in predicted]
    t = [str(item.get("mode")) for item in target]
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    depth = max(len(p), len(t))
    score = 0.0
    pset: set[str] = set()
    tset: set[str] = set()
    for d in range(1, depth + 1):
        if d <= len(p):
            pset.add(p[d - 1])
        if d <= len(t):
            tset.add(t[d - 1])
        overlap = len(pset & tset) / d
        score += (1.0 - persistence) * (persistence ** (d - 1)) * overlap
    score += persistence**depth * (len(pset & tset) / depth)
    return max(0.0, min(1.0, score))


def _leg_count(environment: dict[str, Any]) -> int:
    """Trips in the evaluated day: the realized trips, else activities minus one."""
    chain = environment.get("final_trip_chain") or {}
    if chain.get("trips"):
        return len(chain["trips"])
    sequence = (environment.get("policy_output") or {}).get("activity_sequence") or []
    return max(0, len(sequence) - 1)


def feasibility_reward(environment: dict[str, Any], config: dict[str, Any]) -> float:
    """Bonus for a hard-feasible day minus penalties for its constraint violations."""
    cfg = config["rewards"]["feasibility"]
    if environment.get("status") == "outside_poi_coverage":
        # Not judged: no schedule exists to check. prepare_policy_updates.py
        # leaves realized outcomes out of any group containing such a rollout,
        # so outside-Seoul choices are scored only by the Location reward.
        return 0.0
    violations = environment.get("violations") or {}
    value = float(cfg["bonus"]) if environment.get("feasible") else 0.0
    value -= float(cfg["time_per_minute"]) * float(violations.get("time_min", 0.0))
    value -= float(cfg["route_per_leg"]) * int(violations.get("route_count", 0))
    mode_count = int(violations.get("mode_count", 0))
    if "mode_share_penalty" in cfg:
        # Unroutable Student2 modes as a share of the day's legs: a per-leg count
        # made every extra trip a further penalty, so longer days scored lower
        # even when the person's real day was long. 1.0 = every leg fell back
        # (one of two legs on home-work-home keeps the former 0.5).
        value -= float(cfg["mode_share_penalty"]) * mode_count / max(1, _leg_count(environment), mode_count)
    else:
        value -= float(cfg["mode_per_violation"]) * mode_count
    value -= float(cfg["activity_per_violation"]) * int(violations.get("activity_count", 0))
    return value


def start_time_reward(
    target_sequence: list[str],
    environment: dict[str, Any],
    target: list[dict[str, Any]],
    tolerance_min: float,
) -> float:
    """Score each realized activity start against this person's own GT start.

    Activities are matched by LCS on the activity sequence, as in the dwell
    reward. The first activity (home at the day origin) starts at 0 on both
    sides, so it is left out. Each matched pair scores
    exp(-|start - GT start| / tolerance); unmatched activities score 0 through
    the denominator. Without a realized schedule (infeasible or outside POI
    coverage) there are no start times to judge, so the score is 0.
    """
    activities = (environment.get("final_trip_chain") or {}).get("activities") or []
    predicted_sequence = [str(item.get("activity_type")) for item in activities][1:]
    predicted, target = activities[1:], target[1:]
    target_sequence = target_sequence[1:]
    if not predicted or not target:
        return 0.0
    similarities = [
        math.exp(-abs(float(p.get("start_min", 0)) - float(t.get("start_time_min", 0))) / max(1.0, tolerance_min))
        for p, t in _aligned(predicted_sequence, target_sequence, predicted, target)
    ]
    return sum(similarities) / max(len(predicted), len(target))


def score_rollout(
    policy: dict[str, Any],
    target: dict[str, Any],
    environment: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reward components, their weighted total and the discounted return of each Student.

    Student2's return is its own local reward (location, dwell, mode) plus the
    discounted realized outcome (feasibility, start time). Student1's return is
    the activity reward plus Student2's local reward one step discounted and the
    realized outcome two steps discounted.
    """
    predicted_spatial = policy.get("location_dwell_by_activity") or []
    target_spatial = target.get("activities") or target.get("location_dwell_by_activity") or []
    predicted_sequence = policy.get("activity_sequence") or []
    target_sequence = target.get("activity_sequence") or []
    components = {
        "activity": activity_reward(predicted_sequence, target_sequence),
        "location": location_reward(predicted_sequence, target_sequence, predicted_spatial, target_spatial),
        "dwell": dwell_reward(
            predicted_sequence,
            target_sequence,
            predicted_spatial,
            target_spatial,
            float(config["rewards"]["dwell_tolerance_min"]),
            bool(config.get("scheduling", {}).get("final_home_residual", False)),
        ),
        "feasibility": feasibility_reward(environment, config),
    }
    weights = config["rewards"]["weights"]
    # Per-trip main modes (weights.trip_mode) replace the person-level ranking.
    mode_component = "trip_mode" if "trip_mode" in weights else "mode_preference"
    if mode_component == "trip_mode":
        components["trip_mode"] = trip_mode_reward(
            predicted_sequence,
            target_sequence,
            policy.get("trip_modes") or [],
            target.get("trip_modes") or [],
        )
    else:
        components["mode_preference"] = mode_preference_reward(
            policy.get("mode_preference") or [], target.get("mode_preference") or []
        )
    # Per-person start times on the realized schedule (weights.start_time).
    if "start_time" in weights:
        components["start_time"] = start_time_reward(
            target_sequence,
            environment,
            target_spatial,
            float(config["rewards"].get("start_time_tolerance_min", 60.0)),
        )
    total = sum(float(weights[name]) * value for name, value in components.items())
    gamma = float(config["rewards"]["future_discount"])
    student2_local = (
        float(weights["location"]) * components["location"]
        + float(weights["dwell"]) * components["dwell"]
        + float(weights[mode_component]) * components[mode_component]
    )
    # Realized-schedule outcomes: produced jointly by both Students and the
    # environment, so both reach the Students one and two steps discounted.
    realized = float(weights["feasibility"]) * components["feasibility"]
    if "start_time" in components:
        realized += float(weights["start_time"]) * components["start_time"]
    returns = {
        "student2": student2_local + gamma * realized,
        "student1": float(weights["activity"]) * components["activity"] + gamma * student2_local + gamma**2 * realized,
    }
    return {"components": components, "weighted_total": total, "returns": returns}
