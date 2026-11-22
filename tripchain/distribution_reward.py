"""Population-level reward: does a group of similar people get the right spread?

Individual rewards compare each person with their own ground truth. Under
aleatoric uncertainty (the same persona can go home-work-home or not) every
person is then pushed to the single most likely answer, and the generated
population collapses onto a few patterns. This reward compares, within a cell of
similar people (occupation x age bucket x gender), the distribution the current
policy generates (Q) with the train ground-truth distribution (P):

    r(x) = clip(log P_cell(x) - log Q_cell(x), -clip, clip)

Over-generated values (Q > P) are penalized and under-generated ones rewarded;
E_Q[r] = -KL(Q || P), so maximizing it moves the policy towards P per cell.

P comes from the train split only (no test data). Small cells are smoothed
towards their parent (occupation x age -> occupation -> everyone), and the root
gets an add-epsilon floor so unseen values have finite log-probabilities.

Q is estimated from the rollouts being trained on, i.e. from the policy that
generated them (offline, one round), with the same smoothing.

Q for each sample is always counted without that sample itself (leave-one-out).
With self-inclusion, a rare (e.g. long, many-activity) pattern seen once in a
cell gets Q >= 1/n and an almost always negative reward even when the policy
under-generates that whole class of days.

First departure (build_departure_rewards) uses the same reward on the hour the
person first leaves home, within cells that also carry the first out-of-home
activity (work leaves at 8, shopping at 13): (purpose x occupation x age x
gender) -> ... -> (purpose) -> everyone. It goes to Student2, whose first home
dwell is the departure time (the day starts at home at the day origin).

Trip mode can instead use distance-conditioned cells (build_distribution_rewards
with trip bands, tripchain/trip_distance.py): (distance band x occupation x age x
gender) -> (band x age x gender) -> (band x age) -> (band) -> everyone. The
demographic-only cells cannot tell which trip's mode to change (a cell with too
much subway also loses subway on 15 km trips); the band carries most of the mode
information (v2 train GT: 0.52 bits from 5 bands vs 0.29 from 92 demographic
cells). Occupation is dropped first because it explains mode least; the validation
log-loss of the smoothed P was best with this order and prior strength 100
(1.887 bits vs 2.333 for the demographic cells). A rollout's own trips are left
out of Q per cell, so a trip in another band does not change this trip's Q.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from .population_metrics import age_bucket

# Finest cell first; each level is backed off to the next one.
LEVELS = (
    ("occupation", "age", "gender"),
    ("occupation", "age"),
    ("occupation",),
    (),
)
# Trip mode with distance bands; "band" is read from demographics["distance_band"].
MODE_DISTANCE_LEVELS = (
    ("band", "occupation", "age", "gender"),
    ("band", "age", "gender"),
    ("band", "age"),
    ("band",),
    (),
)


def cell_values(demographics: dict[str, Any]) -> dict[str, str]:
    """The cell coordinates of one person (and of one trip, when it carries a distance band)."""
    return {
        "occupation": str(demographics.get("occupation", "unknown")),
        "age": age_bucket(demographics.get("age", 0)),
        "gender": str(demographics.get("gender", "unknown")),
        "band": str(demographics.get("distance_band", "")),
    }


def cell_keys(
    demographics: dict[str, Any], context: str | None = None, levels: tuple = LEVELS
) -> list[tuple[str, ...]]:
    """Finest cell first. With a context (e.g. the first activity), every level is
    prefixed by it and the context-free root () comes last. Levels of one
    distribution have distinct lengths, so their keys never collide."""
    values = cell_values(demographics)
    keys = [tuple(values[name] for name in level) for level in levels]
    if context is None:
        return keys
    return [(str(context), *key) for key in keys] + [()]


class CellDistribution:
    """Hierarchically smoothed categorical distribution per demographic cell."""

    def __init__(
        self,
        items: Iterable[tuple],
        vocabulary: set[str],
        prior_strength: float,
        root_epsilon: float = 0.5,
        levels: tuple = LEVELS,
    ):
        """items: (demographics, value) or (demographics, value, context); a
        distribution uses context for all of its items or for none."""
        self.levels = levels
        # Keys of different levels differ in length, so one dict holds all levels.
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for demographics, value, *context in items:
            for key in cell_keys(demographics, context[0] if context else None, levels):
                self.counts[key][value] += 1
        self.vocabulary_size = max(1, len(vocabulary))
        self.prior_strength = float(prior_strength)
        self.root_epsilon = float(root_epsilon)

    def prob(
        self,
        demographics: dict[str, Any],
        value: str,
        exclude: Counter | None = None,
        context: str | None = None,
        exclude_by_key: dict[tuple[str, ...], Counter] | None = None,
    ) -> float:
        """exclude: value counts of one sample (with these demographics) to leave out
        at every level. exclude_by_key: per-cell counts to leave out instead, for a
        sample whose items fall into different cells (own_counts_by_key)."""
        keys = cell_keys(demographics, context, self.levels)
        if exclude_by_key is None:
            exclude = exclude or Counter()
            exclude_by_key = {key: exclude for key in keys}
        empty = Counter()

        def counts(key):
            cell = self.counts.get(key) or Counter()
            own = exclude_by_key.get(key, empty)
            return cell[value] - own[value], sum(cell.values()) - sum(own.values())

        hit, total = counts(keys[-1])
        probability = (hit + self.root_epsilon) / (total + self.root_epsilon * self.vocabulary_size)
        for key in reversed(keys[:-1]):
            hit, total = counts(key)
            if total <= 0:
                continue
            probability = (hit + self.prior_strength * probability) / (total + self.prior_strength)
        return probability

    def own_counts_by_key(self, items: Iterable[tuple[dict[str, Any], str]]) -> dict[tuple[str, ...], Counter]:
        """Per-cell value counts of one sample's items, for leave-one-out."""
        own: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for demographics, value in items:
            for key in cell_keys(demographics, None, self.levels):
                own[key][value] += 1
        return own


def log_ratio(
    target: CellDistribution,
    policy: CellDistribution,
    demographics: dict[str, Any],
    value: str,
    clip: float,
    exclude: Counter | None = None,
) -> float:
    """clip(log P(value) - log Q(value)) in the person's cell; `exclude` is left out of Q."""
    ratio = math.log(target.prob(demographics, value)) - math.log(policy.prob(demographics, value, exclude))
    return max(-clip, min(clip, ratio))


def sequence_key(sequence: list[str]) -> str:
    return "-".join(sequence)


def build_distribution_rewards(
    branches: list[tuple[dict[str, Any], list[str]]],
    rollouts: list[tuple],
    truth: list[tuple],
    prior_strength: float,
    clip: float,
    mode_prior_strength: float | None = None,
) -> tuple[list[float], list[float]]:
    """Sequence reward per Student1 branch and mean trip-mode reward per rollout.

    branches: (demographics, activity_sequence) of each generated Student1 output.
    rollouts: (demographics, trip_modes) of each generated Student2 output, or
              (demographics, trip_modes, trip_bands) for distance-conditioned cells.
    truth:    (demographics, activity_sequence, trip_modes[, trip_bands]) of the
              train ground truth; bands are used only when every item has them.
    Q for each sample is counted without that sample (the branch's sequence, or
    all trip modes of the rollout).
    """
    distance = bool(truth) and all(len(item) > 3 for item in truth) and all(len(item) > 2 for item in rollouts)
    true_sequences = [(demo, sequence_key(seq)) for demo, seq, *_ in truth]
    policy_sequences = [(demo, sequence_key(seq)) for demo, seq in branches]
    vocabulary = {value for _, value in true_sequences} | {value for _, value in policy_sequences}
    p_sequence = CellDistribution(true_sequences, vocabulary, prior_strength)
    q_sequence = CellDistribution(policy_sequences, vocabulary, prior_strength)
    sequence_rewards = [
        log_ratio(p_sequence, q_sequence, demo, value, clip, Counter([value])) for demo, value in policy_sequences
    ]

    if distance:
        return sequence_rewards, distance_mode_rewards(
            rollouts, truth, prior_strength if mode_prior_strength is None else mode_prior_strength, clip
        )
    true_modes = [(demo, mode) for demo, _, modes, *_ in truth for mode in modes]
    policy_modes = [(demo, mode) for demo, modes, *_ in rollouts for mode in modes]
    vocabulary = {mode for _, mode in true_modes} | {mode for _, mode in policy_modes}
    p_mode = CellDistribution(true_modes, vocabulary, prior_strength)
    q_mode = CellDistribution(policy_modes, vocabulary, prior_strength)
    mode_rewards = []
    for demo, modes, *_ in rollouts:
        own = Counter(modes)
        mode_rewards.append(
            sum(log_ratio(p_mode, q_mode, demo, mode, clip, own) for mode in modes) / len(modes) if modes else 0.0
        )
    return sequence_rewards, mode_rewards


def distance_mode_rewards(rollouts: list[tuple], truth: list[tuple], prior_strength: float, clip: float) -> list[float]:
    """Mean per-trip mode reward in (distance band x demographic) cells.

    A trip without a band (more modes than locations) is skipped."""

    def items(demo, modes, bands):
        return [({**demo, "distance_band": band}, mode) for mode, band in zip(modes, bands)]

    true_items = [item for demo, _, modes, bands in truth for item in items(demo, modes, bands)]
    policy = [items(demo, modes, bands) for demo, modes, bands in rollouts]
    vocabulary = {mode for _, mode in true_items} | {mode for own in policy for _, mode in own}
    p = CellDistribution(true_items, vocabulary, prior_strength, levels=MODE_DISTANCE_LEVELS)
    q = CellDistribution(
        [item for own in policy for item in own], vocabulary, prior_strength, levels=MODE_DISTANCE_LEVELS
    )
    rewards = []
    for own in policy:
        if not own:
            rewards.append(0.0)
            continue
        exclude = q.own_counts_by_key(own)
        values = [
            max(-clip, min(clip, math.log(p.prob(demo, mode)) - math.log(q.prob(demo, mode, exclude_by_key=exclude))))
            for demo, mode in own
        ]
        rewards.append(sum(values) / len(values))
    return rewards


DEPARTURE_HOURS = 24


def departure_hour(minutes_after_origin: float) -> str:
    """Hour bin of the first departure, counted from the day origin (as in
    scripts/evaluate_population.py)."""
    return str(min(max(int(float(minutes_after_origin) // 60), 0), DEPARTURE_HOURS - 1))


def build_departure_rewards(
    rollouts: list[tuple[dict[str, Any], str, str] | None],
    truth: list[tuple[dict[str, Any], str, str]],
    prior_strength: float,
    clip: float,
) -> list[float]:
    """First-departure reward per rollout (0 for a rollout without one).

    rollouts: (demographics, first out-of-home activity, departure hour) of each
              generated rollout, or None for a home-only day.
    truth:    the same for each train ground-truth day that leaves home.
    Q for each rollout is counted without that rollout.
    """
    vocabulary = {str(hour) for hour in range(DEPARTURE_HOURS)}
    p = CellDistribution([(demo, hour, purpose) for demo, purpose, hour in truth], vocabulary, prior_strength)
    q = CellDistribution(
        [(demo, hour, purpose) for item in rollouts if item for demo, purpose, hour in [item]],
        vocabulary,
        prior_strength,
    )
    rewards = []
    for item in rollouts:
        if item is None:
            rewards.append(0.0)
            continue
        demo, purpose, hour = item
        value = math.log(p.prob(demo, hour, context=purpose)) - math.log(
            q.prob(demo, hour, Counter([hour]), context=purpose)
        )
        rewards.append(max(-clip, min(clip, value)))
    return rewards
