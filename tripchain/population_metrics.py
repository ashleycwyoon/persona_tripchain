"""Distribution-level evaluation of generated activity sequences.

Individual metrics ask whether a person's own day was reproduced; these ask
whether the generated days have the right distribution, for the whole
population and within each persona subgroup (occupation, age, gender). The
subgroup level is what tests persona conditioning: a model that ignored the
demographics and sampled the overall distribution would score well on the
population level but badly here.

All values are Jensen-Shannon divergences (base 2; 0 = identical, 1 = disjoint)
against the ground truth of the same people. Small groups inflate JSD through
sampling noise, so compare models on the same people, never across populations.

Besides the activity sequence, a person may carry per-day categorical values
under "truth_extra"/"pred_extra" ({name: [values]}): trip modes, binned dwell
times, first departure and travel times (see scripts/evaluate_population.py).
Each name becomes its own distribution, scored only where both sides have values.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Iterable

TOP_PATTERNS = 20
MIN_GROUP_PEOPLE = 30
MAX_COUNT = 9


def jsd(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    p_total, q_total = sum(p.values()), sum(q.values())
    if not keys or not p_total or not q_total:
        return float("nan")
    result = 0.0
    for key in keys:
        a, b = p.get(key, 0) / p_total, q.get(key, 0) / q_total
        m = (a + b) / 2
        if a:
            result += a / 2 * math.log2(a / m)
        if b:
            result += b / 2 * math.log2(b / m)
    return result


def age_bucket(age) -> str:
    age = int(age)
    if age < 18:
        return "under18"
    if age < 35:
        return "18-34"
    if age < 55:
        return "35-54"
    return "55plus"


SUBGROUPS: dict[str, Callable[[dict], str]] = {
    "occupation": lambda d: str(d.get("occupation", "unknown")),
    "age": lambda d: age_bucket(d.get("age", 0)),
    "gender": lambda d: str(d.get("gender", "unknown")),
}


def _distributions(people: Iterable[dict], top: set[str], side: str) -> dict[str, Counter]:
    patterns, counts, types, modes = Counter(), Counter(), Counter(), Counter()
    trips = Counter()
    extra: dict[str, Counter] = defaultdict(Counter)
    for person in people:
        seq = person[side]
        pattern = "-".join(seq)
        patterns[pattern if pattern in top else "other"] += 1
        counts[min(len(seq), MAX_COUNT)] += 1
        trips[min(max(len(seq) - 1, 0), MAX_COUNT - 1)] += 1
        types.update(a for a in seq if a != "home")
        if person.get(f"{side}_mode"):
            modes[person[f"{side}_mode"]] += 1
        for name, values in (person.get(f"{side}_extra") or {}).items():
            extra[name].update(values)
    return {
        "sequence_pattern": patterns,
        "activity_count": counts,
        "trip_count": trips,
        "activity_type": types,
        "mode_top1": modes,
        **extra,
    }


def _divergences(people: list[dict], top: set[str]) -> dict[str, float]:
    truth, pred = _distributions(people, top, "truth"), _distributions(people, top, "pred")
    return {name: jsd(pred[name], truth[name]) for name in truth if truth[name] and pred.get(name)}


def population_metrics(people: list[dict], min_group: int = MIN_GROUP_PEOPLE) -> dict:
    """people: dicts with demographics, truth/pred sequences and optional truth_mode/pred_mode."""
    top = {p for p, _ in Counter("-".join(person["truth"]) for person in people).most_common(TOP_PATTERNS)}
    result = {
        "people": len(people),
        "top_patterns": TOP_PATTERNS,
        "min_group_people": min_group,
        "population": _divergences(people, top),
        "subgroup": {},
    }
    per_dimension: dict[str, list[float]] = defaultdict(list)
    for dimension, key in SUBGROUPS.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for person in people:
            groups[key(person["demographics"])].append(person)
        kept = {name: members for name, members in groups.items() if len(members) >= min_group}
        per_group = {
            name: {"people": len(members), **_divergences(members, top)} for name, members in sorted(kept.items())
        }
        metrics = sorted({m for values in per_group.values() for m in values if m != "people"})
        mean = {
            m: sum(v[m] * v["people"] for v in per_group.values() if m in v)
            / sum(v["people"] for v in per_group.values() if m in v)
            for m in metrics
        }
        result["subgroup"][dimension] = {
            "groups": per_group,
            "people_weighted_mean": mean,
            "people_in_kept_groups": sum(len(m) for m in kept.values()),
        }
        for m, value in mean.items():
            per_dimension[m].append(value)
    # Headline subgroup score: mean over the three dimensions of their people-weighted means.
    result["subgroup_mean"] = {m: sum(values) / len(values) for m, values in per_dimension.items()}
    return result
