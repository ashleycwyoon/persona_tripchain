"""One rollout: both Students' outputs -> environment -> dense rewards -> decision trace."""

from __future__ import annotations

from typing import Any

from .common import ordered_output, output_orders
from .environment import TripChainEnvironment
from .rewards import score_rollout
from .trip_modes import mode_preference_from_trips


def merge_policy_outputs(student1: dict[str, Any], student2: dict[str, Any]) -> dict[str, Any]:
    """The policy record the environment evaluates, from one person's two Student rows."""
    if str(student1["person_id"]) != str(student2["person_id"]):
        raise ValueError("student person_id mismatch")
    sequence = student1["activity_sequence"]
    if student2.get("activity_sequence") not in (None, sequence):
        raise ValueError("Student1/Student2 activity sequence mismatch")
    return {
        "person_id": str(student1["person_id"]),
        "demographics": student2.get("demographics") or student1.get("demographics") or {},
        "activity_sequence": sequence,
        "persona_bundle": student1.get("persona_bundle") or student2.get("persona_bundle"),
        "reasoning1": student1.get("reasoning1") or student2.get("reasoning1"),
        "location_dwell_by_activity": student2["location_dwell_by_activity"],
        # With per-trip modes the ranking is derived from them (environment
        # car/bicycle availability, evaluation top-1 mode).
        **(
            {
                "trip_modes": student2["trip_modes"],
                "mode_preference": mode_preference_from_trips(student2["trip_modes"]),
            }
            if "trip_modes" in student2
            else {"mode_preference": student2.get("mode_preference") or []}
        ),
        "reasoning2": student2.get("reasoning2"),
    }


def student2_trace_output(policy: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    """Student2's answer as generated, in its output order (prompt/target for GRPO)."""
    return ordered_output({key: policy.get(key, []) for key in order}, order)


def run_rollout(
    student1: dict[str, Any],
    student2: dict[str, Any],
    target: dict[str, Any],
    environment: TripChainEnvironment,
    config: dict[str, Any],
    rollout_id: str = "0",
) -> dict[str, Any]:
    """Evaluate one person's Student outputs and return the rollout row (trace, rewards, status)."""
    policy = merge_policy_outputs(student1, student2)
    evaluated = environment.evaluate(policy)
    rewards = score_rollout(policy, target, evaluated, config)
    student1_order, student2_order = output_orders(config)
    return {
        "person_id": policy["person_id"],
        "rollout_id": str(rollout_id),
        "decision_trace": {
            "student1": {
                # Student1's prompt is demographics only, so older outputs
                # without a stored input can be reconstructed exactly.
                "input": student1.get("student1_input") or {"demographics": policy["demographics"]},
                "output": ordered_output(
                    {
                        "activity_sequence": policy["activity_sequence"],
                        "persona_bundle": policy["persona_bundle"],
                        "reasoning1": policy["reasoning1"],
                    },
                    student1_order,
                ),
            },
            "student2": {
                # Keep the exact prompt stored at generation time. Older outputs
                # without it keep None and are rejected by prepare_policy_updates.py.
                "input": student2.get("student2_input"),
                "output": student2_trace_output(policy, student2_order),
            },
            "environment": evaluated,
        },
        "rewards": rewards,
        "status": (
            "outside_poi_coverage"
            if evaluated.get("status") == "outside_poi_coverage"
            else "ok"
            if evaluated.get("feasible")
            else "infeasible"
        ),
    }
