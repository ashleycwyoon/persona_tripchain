"""Inference-only iterative refinement of infeasible days (Lv.1 POI, Lv.2 Student2, Lv.3 Student1)."""

from __future__ import annotations

import copy
from typing import Any, Callable

from .environment import TripChainEnvironment


# (record, environment result) -> record with that Student's output regenerated.
Student1Regenerator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
Student2Regenerator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def requires_student1_repair(environment: dict[str, Any]) -> bool:
    """True for structural failures that only a new activity sequence can fix."""
    reason = str(environment.get("failure_reason") or "")
    return any(
        token in reason
        for token in (
            "NO_ACTIVITY",
            "START_ACTIVITY_NOT_HOME",
            "ACTIVITY_LOCATION_LENGTH_MISMATCH",
        )
    )


def refine_inference(
    initial_record: dict[str, Any],
    config: dict[str, Any],
    regenerate_student2: Student2Regenerator | None = None,
    regenerate_student1: Student1Regenerator | None = None,
    level1_attempts: int = 1,
) -> dict[str, Any]:
    """Bounded inference-only escalation; repair actions are not policy-trained.

    Lv.1 changes only the deterministic POI sampling seed. Lv.2 asks Student2
    to regenerate its spatial/dwell/preference decision. Lv.3 asks Student1 to
    regenerate the activity/persona decision and then requires Student2 again.
    The callbacks make policy inference replaceable without coupling the
    non-learning environment to a particular LLM runtime.
    """
    history: list[dict[str, Any]] = []

    def evaluate(record: dict[str, Any], level: str, attempt: int, seed_offset: int = 0):
        cfg = copy.deepcopy(config)
        cfg["poi_sampling"]["seed"] = int(cfg["poi_sampling"]["seed"]) + seed_offset
        result = TripChainEnvironment(cfg).evaluate(record)
        history.append(
            {
                "level": level,
                "attempt": attempt,
                "poi_seed": cfg["poi_sampling"]["seed"],
                "feasible": result.get("feasible", False),
                "failure_reason": result.get("failure_reason"),
            }
        )
        return result

    def done(record, result):
        if result.get("feasible"):
            return {"status": "ok", "record": record, "environment": result, "repair_history": history}
        if result.get("status") == "outside_poi_coverage":
            # Not a constraint violation; repair would only push it into Seoul.
            return {
                "status": "outside_poi_coverage",
                "record": record,
                "environment": result,
                "repair_history": history,
            }
        return None

    record = copy.deepcopy(initial_record)
    result = evaluate(record, "initial", 0)
    if finished := done(record, result):
        return finished

    structural_student1_failure = requires_student1_repair(result)
    if not structural_student1_failure:
        for attempt in range(1, level1_attempts + 1):
            result = evaluate(record, "lv1_poi", attempt, seed_offset=attempt)
            if finished := done(record, result):
                return finished

    if not structural_student1_failure and regenerate_student2 is not None:
        record = regenerate_student2(record, result)
        result = evaluate(record, "lv2_student2", 1, seed_offset=level1_attempts + 1)
        if finished := done(record, result):
            return finished

    if regenerate_student1 is not None and regenerate_student2 is not None:
        record = regenerate_student1(record, result)
        record = regenerate_student2(record, result)
        result = evaluate(record, "lv3_student1", 1, seed_offset=level1_attempts + 2)
        if finished := done(record, result):
            return finished

    return {"status": "infeasible", "record": record, "environment": result, "repair_history": history}
