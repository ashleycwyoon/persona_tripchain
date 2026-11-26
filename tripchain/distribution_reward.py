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

Q usually has more samples than P (two Student1 outputs and four rollouts per
train person against one observed day). With the same prior strength and root
floor, Q is then smoothed less than P: for a value that neither side has in a
cell, P/Q is about (n_Q + prior) / (n_P + prior) at every level, up to 2 per
level for the sequences. On the v2 train rollouts a sequence that never occurs
in the train ground truth got +1.42 on average, -0.50 with one sample per
person. `balanced=True` (rewards.distribution.balanced_smoothing) multiplies
Q's prior strength and root floor by n_Q / n_P, so both sides are smoothed by
the same number of people.

Activity count (activity_count_rewards, Student1) uses the same reward on the number
of activities of the day (1 ... 9+, as scripts/evaluate_population.py counts them), in
the same demographic cells as the sequence. The sequence reward compares exact
sequences, and the long tail of rare sequences is too sparse for that: on the v2 train
rollouts the sequences outside the 12 most generated ones are 36% of the ground truth
and 21% of the generated days (log ratio +0.55), but their mean sequence reward is
+0.07, and below the cell mean once Q is smoothed like P. The count has few values, so
its cells are well estimated: 3 activities 54.5% generated vs 41.0%, 6 or more 3.5% vs
10.3%. A long day then gets the signal whether or not its exact sequence was observed.

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

Trip distance (distance_band_rewards, Student2) uses the same reward on each trip's
distance band, in cells of the trip purpose (the activity the trip arrives at):
(purpose x occupation x age x gender) -> (purpose x age x gender) -> (purpose x age)
-> (purpose) -> everyone. The distance-conditioned mode reward cannot fix the
overall mode shares when the policy picks too many far locations (v2 Pre-RL test:
10 km+ trips 19% vs GT 11%, <3 km 22% vs 31%), since subway is then right per band.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from .population_metrics import MAX_COUNT, age_bucket

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
# Trip distance band; "purpose" is read from demographics["trip_purpose"].
DISTANCE_LEVELS = (
    ("purpose", "occupation", "age", "gender"),
    ("purpose", "age", "gender"),
    ("purpose", "age"),
    ("purpose",),
    (),
)


def cell_values(demographics: dict[str, Any]) -> dict[str, str]:
    """The cell coordinates of one person (and of one trip, when it carries a distance band or a purpose)."""
    return {
        "occupation": str(demographics.get("occupation", "unknown")),
        "age": age_bucket(demographics.get("age", 0)),
        "gender": str(demographics.get("gender", "unknown")),
        "band": str(demographics.get("distance_band", "")),
        "purpose": str(demographics.get("trip_purpose", "")),
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


def policy_scale(policy_items: int, truth_items: int, balanced: bool) -> float:
    """Factor on Q's prior strength and root floor: n_Q / n_P when balanced, else 1."""
    return policy_items / truth_items if balanced and policy_items and truth_items else 1.0


def sequence_key(sequence: list[str]) -> str:
    return "-".join(sequence)


def activity_count_bucket(sequence: list[str]) -> str:
    """Number of activities of a day, capped as in the activity-count metric."""
    return str(min(len(sequence), MAX_COUNT))


def activity_count_rewards(
    branches: list[tuple[dict[str, Any], list[str]]],
    truth: list[tuple],
    prior_strength: float,
    clip: float,
    balanced: bool = False,
) -> list[float]:
    """Activity-count reward per Student1 branch, in the demographic cells of the sequence reward.

    branches: (demographics, activity_sequence) of each generated Student1 output.
    truth:    (demographics, activity_sequence, ...) of the train ground truth.
    Q for each branch is counted without that branch.
    """
    true_counts = [(demo, activity_count_bucket(seq)) for demo, seq, *_ in truth]
    policy_counts = [(demo, activity_count_bucket(seq)) for demo, seq in branches]
    vocabulary = {str(count) for count in range(1, MAX_COUNT + 1)}
    scale = policy_scale(len(policy_counts), len(true_counts), balanced)
    p = CellDistribution(true_counts, vocabulary, prior_strength)
    q = CellDistribution(policy_counts, vocabulary, prior_strength * scale, 0.5 * scale)
    return [log_ratio(p, q, demo, value, clip, Counter([value])) for demo, value in policy_counts]


def build_distribution_rewards(
    branches: list[tuple[dict[str, Any], list[str]]],
    rollouts: list[tuple],
    truth: list[tuple],
    prior_strength: float,
    clip: float,
    mode_prior_strength: float | None = None,
    balanced: bool = False,
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
    scale = policy_scale(len(policy_sequences), len(true_sequences), balanced)
    q_sequence = CellDistribution(policy_sequences, vocabulary, prior_strength * scale, 0.5 * scale)
    sequence_rewards = [
        log_ratio(p_sequence, q_sequence, demo, value, clip, Counter([value])) for demo, value in policy_sequences
    ]

    if distance:
        return sequence_rewards, distance_mode_rewards(
            rollouts, truth, prior_strength if mode_prior_strength is None else mode_prior_strength, clip, balanced
        )
    true_modes = [(demo, mode) for demo, _, modes, *_ in truth for mode in modes]
    policy_modes = [(demo, mode) for demo, modes, *_ in rollouts for mode in modes]
    vocabulary = {mode for _, mode in true_modes} | {mode for _, mode in policy_modes}
    p_mode = CellDistribution(true_modes, vocabulary, prior_strength)
    scale = policy_scale(len(policy_modes), len(true_modes), balanced)
    q_mode = CellDistribution(policy_modes, vocabulary, prior_strength * scale, 0.5 * scale)
    mode_rewards = []
    for demo, modes, *_ in rollouts:
        own = Counter(modes)
        mode_rewards.append(
            sum(log_ratio(p_mode, q_mode, demo, mode, clip, own) for mode in modes) / len(modes) if modes else 0.0
        )
    return sequence_rewards, mode_rewards


def distance_mode_rewards(
    rollouts: list[tuple], truth: list[tuple], prior_strength: float, clip: float, balanced: bool = False
) -> list[float]:
    """Mean per-trip mode reward in (distance band x demographic) cells.

    A trip without a band (more modes than locations) is skipped."""

    def items(demo, modes, bands):
        return [({**demo, "distance_band": band}, mode) for mode, band in zip(modes, bands)]

    true_items = [item for demo, _, modes, bands in truth for item in items(demo, modes, bands)]
    policy = [items(demo, modes, bands) for demo, modes, bands in rollouts]
    vocabulary = {mode for _, mode in true_items} | {mode for own in policy for _, mode in own}
    p = CellDistribution(true_items, vocabulary, prior_strength, levels=MODE_DISTANCE_LEVELS)
    policy_items = [item for own in policy for item in own]
    scale = policy_scale(len(policy_items), len(true_items), balanced)
    q = CellDistribution(policy_items, vocabulary, prior_strength * scale, 0.5 * scale, levels=MODE_DISTANCE_LEVELS)
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


def distance_band_rewards(
    rollouts: list[tuple], truth: list[tuple], prior_strength: float, clip: float, balanced: bool = False
) -> list[float]:
    """Mean per-trip distance-band reward in (trip purpose x demographic) cells.

    rollouts / truth: (demographics, trip purposes, trip bands); a trip's purpose is
    the activity it arrives at. Q for each rollout leaves out its own trips per cell.
    """

    def items(demo, purposes, bands):
        return [({**demo, "trip_purpose": purpose}, band) for purpose, band in zip(purposes, bands)]

    true_items = [item for demo, purposes, bands in truth for item in items(demo, purposes, bands)]
    policy = [items(demo, purposes, bands) for demo, purposes, bands in rollouts]
    vocabulary = {band for _, band in true_items} | {band for own in policy for _, band in own}
    p = CellDistribution(true_items, vocabulary, prior_strength, levels=DISTANCE_LEVELS)
    policy_items = [item for own in policy for item in own]
    scale = policy_scale(len(policy_items), len(true_items), balanced)
    q = CellDistribution(policy_items, vocabulary, prior_strength * scale, 0.5 * scale, levels=DISTANCE_LEVELS)
    rewards = []
    for own in policy:
        if not own:
            rewards.append(0.0)
            continue
        exclude = q.own_counts_by_key(own)
        values = [
            max(-clip, min(clip, math.log(p.prob(demo, band)) - math.log(q.prob(demo, band, exclude_by_key=exclude))))
            for demo, band in own
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
    balanced: bool = False,
) -> list[float]:
    """First-departure reward per rollout (0 for a rollout without one).

    rollouts: (demographics, first out-of-home activity, departure hour) of each
              generated rollout, or None for a home-only day.
    truth:    the same for each train ground-truth day that leaves home.
    Q for each rollout is counted without that rollout.
    """
    vocabulary = {str(hour) for hour in range(DEPARTURE_HOURS)}
    p = CellDistribution([(demo, hour, purpose) for demo, purpose, hour in truth], vocabulary, prior_strength)
    policy_items = [(demo, hour, purpose) for item in rollouts if item for demo, purpose, hour in [item]]
    scale = policy_scale(len(policy_items), len(truth), balanced)
    q = CellDistribution(policy_items, vocabulary, prior_strength * scale, 0.5 * scale)
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
