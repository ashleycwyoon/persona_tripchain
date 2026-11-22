"""The non-learning environment: POI grounding, network routing and scheduling of one day."""

from __future__ import annotations

from typing import Any

from .graphhopper import GraphHopperClient, RoutingUnavailable
from .optimizer import optimize_chain
from .poi import POIGrounder


# Failure reasons that count as an activity violation / an unresolvable location.
_ACTIVITY_FAILURES = (
    "NO_HOME",
    "START_ACTIVITY_NOT_HOME",
    "END_ACTIVITY_NOT_HOME",
    "NO_POI",
    "NO_ACTIVITY_POI",
)
_LOCATION_FAILURES = ("OUTSIDE_POI_COVERAGE", "UNKNOWN_LOCATION")


def _violations(
    selected: dict[str, Any] | None,
    stats: dict[str, Any],
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Hard-constraint violation counts of one day, read by the feasibility reward."""
    if selected is not None:
        return {
            "time_min": 0.0,
            "route_count": 0,
            # Legs Student2 gave a mode the environment could not realize.
            "mode_count": sum(bool(edge.get("mode_fallback")) for edge in selected.get("edges", [])),
            "activity_count": 0,
        }
    reason = failure_reason or ""
    return {
        "time_min": float(stats.get("minimum_time_violation_min") or 0.0),
        # A day without a feasible chain counts one route violation, and one more
        # when a location could not be resolved inside the POI catalog.
        "route_count": 1 + int(any(token in reason for token in _LOCATION_FAILURES)),
        "mode_count": int(stats.get("mode_constraint_rejections", 0) > 0),
        "activity_count": int(any(token in reason for token in _ACTIVITY_FAILURES)),
    }


class TripChainEnvironment:
    """Non-learning POI grounding, network routing, and scheduling environment."""

    def __init__(
        self,
        config: dict[str, Any],
        grounder: POIGrounder | None = None,
        client: GraphHopperClient | None = None,
    ):
        self.config = config
        self.grounder = grounder or POIGrounder(config)
        self.client = client or GraphHopperClient(config["graphhopper"])

    def evaluate(self, record: dict[str, Any]) -> dict[str, Any]:
        """Ground, route and schedule one policy record; never raises for an infeasible day."""
        result: dict[str, Any] = {
            "person_id": str(record["person_id"]),
            "status": "failed",
            "feasible": False,
            "policy_output": {
                "activity_sequence": record["activity_sequence"],
                "location_dwell_by_activity": record["location_dwell_by_activity"],
                "mode_preference": record.get("mode_preference", []),
                **({"trip_modes": record["trip_modes"]} if "trip_modes" in record else {}),
            },
        }
        try:
            sequence = record.get("activity_sequence") or []
            constraints = self.config.get("activity_constraints", {})
            if not sequence:
                raise ValueError("NO_ACTIVITY")
            if constraints.get("require_start_home") and sequence[0] != "home":
                raise ValueError("START_ACTIVITY_NOT_HOME")
            if constraints.get("require_end_home") and sequence[-1] != "home":
                raise ValueError("END_ACTIVITY_NOT_HOME")
            # Coverage rules: a home outside Seoul is infeasible for a Seoul
            # resident; any other activity outside the Seoul POI catalogue cannot
            # be grounded as a complete chain, so the person is set aside as
            # outside_poi_coverage without POI sampling, routing, or a
            # feasibility judgement.
            spatial = record.get("location_dwell_by_activity") or []
            external = [
                {"activity_index": index, "activity_type": activity_type, "original_location": item.get("location")}
                for index, (activity_type, item) in enumerate(zip(sequence, spatial))
                if (item.get("location") or {}).get("scope") == "external"
            ]
            if any(item["activity_type"] == "home" for item in external):
                raise ValueError("HOME_OUTSIDE_SEOUL")
            if external:
                result.update(
                    {
                        "status": "outside_poi_coverage",
                        "feasible": None,
                        "failure_reason": "OUTSIDE_POI_COVERAGE",
                        "outside_poi_coverage": external,
                        "violations": {"time_min": 0.0, "route_count": 0, "mode_count": 0, "activity_count": 0},
                    }
                )
                return result
            layers, audit = self.grounder.sample_layers(record)
            result["poi_grounding"] = audit

            # The first sampled POI in every layer measures how feasible the
            # unrescued initial realization is. The complete active pools then
            # measure what global POI/mode optimization can recover.
            initial_layers = [[layer[0]] for layer in layers]
            initial, initial_stats = optimize_chain(record, initial_layers, self.client, self.config)
            selected, stats = optimize_chain(record, layers, self.client, self.config)
            result["initial_plan_feasible"] = initial is not None
            result["initial_plan"] = initial
            result["initial_optimizer_audit"] = initial_stats
            result["optimizer_audit"] = stats
            result["post_poi_optimization_feasible"] = selected is not None
            denominator = max(1, int(stats.get("terminal_attempts", 0)))
            result["feasible_candidate_ratio"] = float(stats.get("terminal_candidates", 0)) / denominator
            if selected is None:
                result["failure_reason"] = "NO_FEASIBLE_GLOBAL_COMBINATION"
                result["violations"] = _violations(None, stats, result["failure_reason"])
                return result

            result["selected_candidate"] = selected
            result["final_trip_chain"] = {
                "activities": selected["activities"],
                "trips": selected["edges"],
                "day_end_min": selected["day_end_min"],
                "unallocated_day_slack_min": selected["unallocated_day_slack_min"],
                "synthetic_home_anchor": True,
            }
            result["violations"] = _violations(selected, stats)
            result["feasible"] = True
            result["status"] = "ok"
            return result
        except RoutingUnavailable:
            # An unanswered request says nothing about feasibility; fail the rollout.
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            result["failure_reason"] = reason
            result["violations"] = _violations(None, {}, reason)
            return result
