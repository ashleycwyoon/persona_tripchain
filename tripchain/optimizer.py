"""Joint POI / mode selection for one day: a layered search over hard-feasible chains.

Each activity has a pool of candidate POIs (a layer). States are expanded leg by
leg with every routable mode, pruned to a diverse set of time / vehicle / mode
signatures, and one hard-feasible terminal state is sampled reproducibly. The
selected POI chain is then re-routed with its exact departure times.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .graphhopper import GraphHopperClient
from .scheduling import schedule_activity
from .trip_modes import ROUTING, mode_preference_from_trips


def can_drive(demographics: dict[str, Any]) -> bool:
    value = demographics.get("can_drive")
    return isinstance(value, str) and value.strip().casefold() == "yes"


def _mode_ranks(record: dict[str, Any]) -> dict[str, int]:
    """Person-level mode ranking (1 = most used), from the trip modes when Student2 names them."""
    ranks: dict[str, int] = {}
    preference = (
        mode_preference_from_trips(record["trip_modes"])
        if "trip_modes" in record
        else record.get("mode_preference") or []
    )
    for item in preference:
        if isinstance(item, dict) and isinstance(item.get("mode"), str):
            ranks[item["mode"]] = int(item.get("rank", len(ranks) + 1))
    return ranks


def _mode_penalty(mode: str, ranks: dict[str, int]) -> float:
    if not ranks:
        return 0.5
    return (ranks.get(mode, len(ranks) + 2) - 1) / max(1, len(ranks))


def _road_roles(record: dict[str, Any], state: "State", origin_id: str) -> list[str]:
    """Traveler modes a car route can be realized as: driving only if the car is at the origin."""
    ranks = _mode_ranks(record)
    roles: list[str] = []
    if can_drive(record.get("demographics") or {}) and state.car_poi_id == origin_id:
        roles.append("car_driver")
    non_driver = [mode for mode in ("car_passenger", "taxi") if mode in ranks]
    roles.append(min(non_driver, key=lambda mode: ranks[mode]) if non_driver else "taxi")
    return list(dict.fromkeys(roles))


@dataclass
class State:
    """A partial day: visited POIs, scheduled activities, legs and where the vehicles are."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]] = field(default_factory=list)
    activities: list[dict[str, Any]] = field(default_factory=list)
    arrival_min: float = 0.0
    travel_time_min: float = 0.0
    distance_km: float = 0.0
    transfer_count: int = 0
    waiting_min: float = 0.0
    preference_penalty: float = 0.0
    car_poi_id: str | None = None
    bicycle_poi_id: str | None = None

    @property
    def selection_cost(self) -> float:
        return self.preference_penalty


def _expand_route(
    state: State,
    route: dict[str, Any],
    traveler_mode: str,
    destination: dict[str, Any],
    departure_min: float,
    ranks: dict[str, int],
    weights: dict[str, float],
    mode_cost: float | None = None,
    leg_tags: dict[str, Any] | None = None,
) -> State:
    """Copy of `state` extended by one leg to `destination`."""
    value = copy.deepcopy(state)
    edge = dict(route)
    edge["routing_profile"] = route.get("routing_profile", route["mode"])
    edge["traveler_mode"] = traveler_mode
    edge["mode"] = traveler_mode
    edge["origin_poi_id"] = state.nodes[-1]["poi_id"]
    edge["destination_poi_id"] = destination["poi_id"]
    edge["departure_min"] = departure_min
    edge["arrival_min"] = departure_min + float(route["travel_time_min"])
    edge.update(leg_tags or {})
    value.nodes.append(destination)
    value.edges.append(edge)
    value.arrival_min = edge["arrival_min"]
    value.travel_time_min += float(route["travel_time_min"])
    value.distance_km += float(route.get("distance_km", 0.0))
    value.transfer_count += int(route.get("transfer_count", 0))
    value.preference_penalty += (
        float(weights["preference_weight"]) * (_mode_penalty(traveler_mode, ranks) if mode_cost is None else mode_cost)
        + float(weights["distance_weight"]) * float(route.get("distance_km", 0.0))
        + float(weights["transfer_weight"]) * int(route.get("transfer_count", 0))
    )
    if traveler_mode == "car_driver":
        value.car_poi_id = destination["poi_id"]
    if traveler_mode == "bicycle":
        value.bicycle_poi_id = destination["poi_id"]
    return value


def _signature(state: State, bucket_min: int) -> tuple[Any, ...]:
    return (
        state.nodes[-1]["poi_id"],
        state.car_poi_id,
        state.bicycle_poi_id,
        int(state.arrival_min // max(1, bucket_min)),
        state.edges[-1]["mode"] if state.edges else None,
    )


def _prune(states: list[State], config: dict[str, Any]) -> list[State]:
    """Keep diverse time/vehicle/mode signatures instead of shortest paths only."""
    per_signature = int(config["max_states_per_signature"])
    max_total = int(config["max_total_states"])
    bucket = int(config["diversity_time_bucket_min"])
    groups: dict[tuple[Any, ...], list[State]] = defaultdict(list)
    for state in states:
        groups[_signature(state, bucket)].append(state)
    kept: list[State] = []
    for values in groups.values():
        values.sort(key=lambda item: (item.selection_cost, item.travel_time_min, item.distance_km))
        if len(values) <= per_signature:
            kept.extend(values)
            continue
        # Preserve an even spread through the ranked list so that a single
        # minimum-time/mode family cannot erase longer feasible alternatives.
        last = len(values) - 1
        indices = sorted({round(i * last / (per_signature - 1)) for i in range(per_signature)})
        kept.extend(values[i] for i in indices)
    if len(kept) > max_total:
        kept.sort(key=lambda item: (item.selection_cost, item.arrival_min))
        last = len(kept) - 1
        indices = sorted({round(i * last / (max_total - 1)) for i in range(max_total)})
        kept = [kept[i] for i in indices]
    return kept


def _weighted_select(states: list[State], person_id: str, config: dict[str, Any]) -> State:
    """Sample one state with probability decreasing in cost, seeded by the person."""
    temperature = max(1e-6, float(config["selection_temperature"]))
    minimum = min(state.selection_cost for state in states)
    weights = [math.exp(-(state.selection_cost - minimum) / temperature) for state in states]
    material = f"{config['selection_seed']}|{person_id}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
    return rng.choices(states, weights=weights, k=1)[0]


def optimize_chain(
    record: dict[str, Any],
    layers: list[list[dict[str, Any]]],
    client: GraphHopperClient,
    config: dict[str, Any],
    use_leg_route_cache: bool = True,
    exact_refinement: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select one hard-feasible realization of the day, or None, plus search statistics.

    layers[i] is the candidate POI pool of activity i. A chain is feasible when
    every leg is routable under the vehicle-continuity rules and the day ends
    within `scheduling.day_minutes`.
    """
    scheduling = config["scheduling"]
    optimization = config["optimization"]
    graphhopper = config["graphhopper"]
    day_minutes = float(scheduling["day_minutes"])
    # The last stay runs to the end of the survey day; it is the remainder of
    # the day, not a choice. With this on, its predicted dwell is ignored so a
    # chain cannot overrun the day by its final stay. This holds for any final
    # activity type, as in the ground truth (a day may end away from home, e.g.
    # at work, when activity_constraints.require_end_home is off).
    final_home_residual = bool(scheduling.get("final_home_residual", False))
    ranks = _mode_ranks(record)
    bicycle_available = "bicycle" in ranks or "personal_mobility" in ranks
    # Student2's main mode per trip (trip i: activity i -> i+1). A leg is routed
    # in that mode; only when it cannot be (no route, or the car/bicycle is not
    # at the origin) does the leg fall back to the preference-ranked search, at
    # a penalty and reported as a mode violation.
    trip_modes = record.get("trip_modes")
    fallback_penalty = float(optimization.get("mode_fallback_penalty", 2.0))
    modes = list(graphhopper["modes"])
    stats: dict[str, Any] = {
        "route_queries": 0,
        "route_options": 0,
        "missing_route_transitions": 0,
        "mode_constraint_rejections": 0,
        "day_overrun_rejections": 0,
        "states_before_pruning": [],
        "states_after_pruning": [],
        "terminal_candidates": 0,
        "terminal_attempts": 0,
        "minimum_time_violation_min": None,
        "leg_route_cache_hits": 0,
        "mode_fallback_transitions": 0,
    }
    if not layers or any(not layer for layer in layers):
        return None, stats

    states = [
        State(
            nodes=[node],
            car_poi_id=node["poi_id"] if can_drive(record.get("demographics") or {}) else None,
            bicycle_poi_id=node["poi_id"] if bicycle_available else None,
        )
        for node in layers[0]
    ]
    sequence = record["activity_sequence"]
    spatial = record["location_dwell_by_activity"]
    leg_route_cache: dict[tuple[int, str, str, str], list[dict[str, Any]]] = {}

    for index, activity_type in enumerate(sequence):
        next_states: list[State] = []
        dwell = float(spatial[index]["dwell_time_min"])
        for state in states:
            if index == len(sequence) - 1:
                stats["terminal_attempts"] += 1
            planned = dwell
            if final_home_residual and index == len(sequence) - 1:
                planned = max(0.0, day_minutes - state.arrival_min)
            timing = schedule_activity(activity_type, state.arrival_min, planned)
            scheduled = copy.deepcopy(state)
            scheduled.waiting_min += timing["waiting_min"]
            scheduled.preference_penalty += float(optimization["waiting_weight"]) * timing["waiting_min"]
            scheduled.activities.append(
                {
                    "activity_index": index,
                    "activity_type": activity_type,
                    "poi": scheduled.nodes[-1],
                    "predicted_dwell_time_min": dwell,
                    **timing,
                }
            )
            if timing["end_min"] > day_minutes:
                stats["day_overrun_rejections"] += 1
                violation = float(timing["end_min"] - day_minutes)
                incumbent = stats["minimum_time_violation_min"]
                stats["minimum_time_violation_min"] = violation if incumbent is None else min(incumbent, violation)
                continue
            if index == len(sequence) - 1:
                next_states.append(scheduled)
                continue

            origin = scheduled.nodes[-1]
            destination_layer = layers[index + 1]

            def expand(destination, routing_modes, roles_for, extra_cost, tags, mode_cost=None):
                """Append realizations of one leg; returns (states added, any route found)."""
                added, found = 0, False
                for routing_mode in routing_modes:
                    if routing_mode == "bike" and scheduled.bicycle_poi_id != origin["poi_id"]:
                        stats["mode_constraint_rejections"] += 1
                        continue
                    cache_key = (index, origin["poi_id"], destination["poi_id"], routing_mode)
                    routes = leg_route_cache.get(cache_key) if use_leg_route_cache else None
                    if routes is None:
                        stats["route_queries"] += 1
                        routes = client.routes(origin, destination, routing_mode, timing["end_min"])
                        if use_leg_route_cache:
                            leg_route_cache[cache_key] = routes
                    else:
                        stats["leg_route_cache_hits"] += 1
                    if not routes:
                        continue
                    found = True
                    for route in routes:
                        stats["route_options"] += 1
                        for role in roles_for(routing_mode):
                            if role == "car_driver" and scheduled.car_poi_id != origin["poi_id"]:
                                stats["mode_constraint_rejections"] += 1
                                continue
                            state = _expand_route(
                                scheduled,
                                route,
                                role,
                                destination,
                                timing["end_min"],
                                ranks,
                                optimization,
                                mode_cost,
                                tags,
                            )
                            state.preference_penalty += extra_cost
                            next_states.append(state)
                            added += 1
                return added, found

            def ranked_roles(routing_mode):
                if routing_mode == "car":
                    return _road_roles(record, scheduled, origin["poi_id"])
                return ["bicycle"] if routing_mode == "bike" else [routing_mode]

            specified = trip_modes[index] if trip_modes and index < len(trip_modes) else None
            for destination in destination_layer:
                if specified in ROUTING:
                    routing_mode, traveler = ROUTING[specified]
                    added, found = expand(
                        destination,
                        [routing_mode],
                        lambda _mode: [traveler],
                        0.0,
                        {"specified_mode": specified, "mode_fallback": False},
                        mode_cost=0.0,
                    )
                    if not added:
                        stats["mode_fallback_transitions"] += 1
                        _, fallback_found = expand(
                            destination,
                            modes,
                            ranked_roles,
                            fallback_penalty,
                            {"specified_mode": specified, "mode_fallback": True},
                        )
                        found = found or fallback_found
                else:
                    _, found = expand(destination, modes, ranked_roles, 0.0, None)
                if not found:
                    stats["missing_route_transitions"] += 1
        stats["states_before_pruning"].append(len(next_states))
        if index < len(sequence) - 1:
            states = _prune(next_states, optimization)
            stats["states_after_pruning"].append(len(states))
            if not states:
                return None, stats
        else:
            states = next_states

    stats["terminal_candidates"] = len(states)
    if not states:
        return None, stats
    selected = _weighted_select(states, str(record["person_id"]), optimization)
    if exact_refinement and len(sequence) > 1:
        # Candidate search may reuse one timetable query per leg/OD/mode. Re-run
        # the selected POI chains with their actual state-dependent departures;
        # only this exact pass is returned as the final feasible chain.
        candidates = [selected]
        seen = {tuple(node["poi_id"] for node in selected.nodes)}
        for candidate in sorted(states, key=lambda item: (item.selection_cost, item.arrival_min)):
            signature = tuple(node["poi_id"] for node in candidate.nodes)
            if signature not in seen:
                seen.add(signature)
                candidates.append(candidate)
            if len(candidates) >= int(optimization.get("exact_candidate_attempts", 8)):
                break
        exact_audits = []
        for candidate in candidates:
            exact, audit = optimize_chain(
                record,
                [[node] for node in candidate.nodes],
                client,
                config,
                use_leg_route_cache=False,
                exact_refinement=False,
            )
            exact_audits.append(audit)
            if exact is not None:
                stats["exact_candidate_attempts"] = len(exact_audits)
                stats["exact_refinement_audit"] = audit
                exact["approximate_candidate_selection_cost"] = candidate.selection_cost
                return exact, stats
        stats["exact_candidate_attempts"] = len(exact_audits)
        stats["exact_refinement_audits"] = exact_audits
        return None, stats
    result = {
        "nodes": selected.nodes,
        "edges": selected.edges,
        "activities": selected.activities,
        "travel_time_min": selected.travel_time_min,
        "distance_km": selected.distance_km,
        "transfer_count": selected.transfer_count,
        "waiting_min": selected.waiting_min,
        "selection_cost": selected.selection_cost,
        "day_end_min": selected.activities[-1]["end_min"],
        "unallocated_day_slack_min": day_minutes - selected.activities[-1]["end_min"],
        "feasible": True,
    }
    return result, stats
