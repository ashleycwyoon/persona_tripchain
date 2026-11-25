#!/usr/bin/env python3
"""Score Student1 activity sequences as a distribution, from several samples per person.

A single generated day cannot show whether a model spreads its probability the way
real people do. Give every model K Student1 output files of the same people, generated
with different seeds (the validation run plus scripts/run_v2_s1_multiseed.sh), and this
script reports, per model:

  exact            share of samples equal to the person's ground-truth sequence
  same_pattern     probability that two samples of one person are the same sequence
  brier            1 - 2 q(GT) + sum_s q(s)^2 per person, lower is better; sum q^2 is
                   estimated without bias from the pairs of samples
  log_score        mean log Q_cell(GT), higher is better; Q_cell pools the model's
                   samples over the person's demographic cell (occupation x age x
                   gender), smoothed toward the model's overall distribution
  patterns         number of distinct sequences in the first sample of every person
  entropy_bits     entropy of those sequences
  home_work_home   share of home-work-home in the first sample
  unseen_in_train  share of samples whose sequence no train person has

The "Ground truth" row gives the diversity numbers of the evaluated people themselves.
Two reference rows are built from the train ground truth of the person's cell: "Mode"
always answers the most frequent sequence, "Calibrated" answers with the cell's
distribution. With --reference, paired differences to that model are printed with a
95% interval (mean +- 1.96 standard errors over people).

Example:
  python scripts/score_sequences.py --config configs/framework_v2ctxt_validation.yaml \
    --model "SeqBoost=a/student1.jsonl,b/seqboost_seed43.jsonl,b/seqboost_seed44.jsonl" \
    --model "New=c/student1.jsonl,b/new_seed43.jsonl,b/new_seed44.jsonl" \
    --reference SeqBoost --output scores.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.common import load_config, resolve, rows_by_id  # noqa: E402
from tripchain.distribution_reward import cell_values  # noqa: E402
from tripchain.paths import DATA_ROOT  # noqa: E402

# Smoothing of the cell distributions used by the log score.
GLOBAL_PSEUDOCOUNT, CELL_PRIOR = 0.5, 5.0


def key(sequence: list[str]) -> str:
    return "-".join(str(activity) for activity in sequence)


def cell(demographics: dict) -> tuple[str, str, str]:
    values = cell_values(demographics)
    return values["occupation"], values["age"], values["gender"]


def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    return -sum(count / total * math.log2(count / total) for count in counts.values())


def smoothed(cell_counts: Counter, global_counts: Counter, vocabulary: int):
    """Cell distribution backed off to the global one, positive for every sequence."""
    global_total, cell_total = sum(global_counts.values()), sum(cell_counts.values())

    def probability(sequence: str) -> float:
        prior = (global_counts.get(sequence, 0) + GLOBAL_PSEUDOCOUNT) / (global_total + GLOBAL_PSEUDOCOUNT * vocabulary)
        return (cell_counts.get(sequence, 0) + CELL_PRIOR * prior) / (cell_total + CELL_PRIOR)

    return probability


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def diversity(first_samples: list[str]) -> dict:
    counts = Counter(first_samples)
    return {
        "patterns": len(counts),
        "entropy_bits": entropy_bits(counts),
        "home_work_home": counts.get("home-work-home", 0) / len(first_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="evaluation config: ground truth and demographics")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=FILE,FILE[,FILE...]",
        help="Student1 outputs of one model for the same people, one file per seed (at least two)",
    )
    parser.add_argument(
        "--train-ground-truth",
        default=str(DATA_ROOT / "processed_v2" / "train" / "ground_truth_tripmode.jsonl"),
    )
    parser.add_argument(
        "--train-demographics",
        default=str(DATA_ROOT / "processed_v2" / "train" / "demographics_with_residence.jsonl"),
    )
    parser.add_argument("--reference", help="label of the model the others are compared with")
    parser.add_argument("--output", help="write the table as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    truth = {
        person_id: key(row["activity_sequence"])
        for person_id, row in rows_by_id(resolve(config["paths"]["ground_truth_reference"])).items()
    }
    demographics = {
        person_id: row.get("demographics", row)
        for person_id, row in rows_by_id(resolve(config["paths"]["test_demographics"])).items()
    }
    train_demographics = rows_by_id(args.train_demographics)
    train_cells: dict[tuple, Counter] = defaultdict(Counter)
    train_all: Counter = Counter()
    for person_id, row in rows_by_id(args.train_ground_truth).items():
        if person_id in train_demographics:
            person = train_demographics[person_id]
            sequence = key(row["activity_sequence"])
            train_cells[cell(person.get("demographics", person))][sequence] += 1
            train_all[sequence] += 1

    files = {}
    for item in args.model:
        label, _, paths = item.partition("=")
        files[label] = [path for path in paths.split(",") if path]
        if len(files[label]) < 2:
            parser.error(f"{label}: at least two sample files are needed")
    if args.reference and args.reference not in files:
        parser.error(f"--reference {args.reference} is not one of the models")
    samples: dict[str, dict[str, list[str]]] = {}
    for label, paths in files.items():
        per_person: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            for person_id, row in rows_by_id(path).items():
                if row.get("activity_sequence"):
                    per_person[person_id].append(key(row["activity_sequence"]))
        samples[label] = per_person
    people = [
        person_id
        for person_id in truth
        if person_id in demographics
        and all(len(samples[label].get(person_id, [])) == len(paths) for label, paths in files.items())
    ]
    if not people:
        raise SystemExit("no person has every sample of every model")
    print(f"people with every sample: {len(people)} / {len(truth)}", file=sys.stderr)
    vocabulary = len(
        set(train_all)
        | {truth[person_id] for person_id in people}
        | {sequence for per_person in samples.values() for person_id in people for sequence in per_person[person_id]}
    )

    table: dict[str, dict] = {"Ground truth": diversity([truth[person_id] for person_id in people])}
    table["Ground truth"]["unseen_in_train"] = mean([float(truth[p] not in train_all) for p in people])
    for name in ("Mode (train ground truth)", "Calibrated (train ground truth)"):
        exact, brier, log_score = [], [], []
        for person_id in people:
            counts = train_cells.get(cell(demographics[person_id])) or train_all
            total, target = sum(counts.values()), truth[person_id]
            if name.startswith("Mode"):
                top = counts.most_common(1)[0][0]
                hit, collision = float(top == target), 1.0
                probability = smoothed(Counter({top: 1000}), Counter({top: 1}), vocabulary)(target)
            else:
                hit = counts[target] / total
                collision = sum((count / total) ** 2 for count in counts.values())
                probability = smoothed(counts, train_all, vocabulary)(target)
            exact.append(hit)
            brier.append(1 - 2 * hit + collision)
            log_score.append(math.log(probability))
        table[name] = {"exact": mean(exact), "brier": mean(brier), "log_score": mean(log_score)}

    per_person_scores: dict[str, tuple[list[float], list[float]]] = {}
    for label in files:
        cells: dict[tuple, Counter] = defaultdict(Counter)
        overall: Counter = Counter()
        for person_id in people:
            for sequence in samples[label][person_id]:
                cells[cell(demographics[person_id])][sequence] += 1
                overall[sequence] += 1
        exact, same, brier, log_score = [], [], [], []
        for person_id in people:
            drawn, target = samples[label][person_id], truth[person_id]
            hit = sum(sequence == target for sequence in drawn) / len(drawn)
            collision = mean([float(a == b) for a, b in combinations(drawn, 2)])
            exact.append(hit)
            same.append(collision)
            brier.append(1 - 2 * hit + collision)
            log_score.append(math.log(smoothed(cells[cell(demographics[person_id])], overall, vocabulary)(target)))
        per_person_scores[label] = (brier, log_score)
        table[label] = {
            "samples_per_person": len(files[label]),
            "exact": mean(exact),
            "same_pattern": mean(same),
            "brier": mean(brier),
            "log_score": mean(log_score),
            **diversity([samples[label][person_id][0] for person_id in people]),
            "unseen_in_train": sum(count for sequence, count in overall.items() if sequence not in train_all)
            / sum(overall.values()),
        }
    if args.reference:
        for label in files:
            if label == args.reference:
                continue
            for position, name in enumerate(("brier", "log_score")):
                differences = [
                    value - base
                    for value, base in zip(
                        per_person_scores[label][position], per_person_scores[args.reference][position]
                    )
                ]
                center = mean(differences)
                deviation = math.sqrt(sum((d - center) ** 2 for d in differences) / (len(differences) - 1))
                half = 1.96 * deviation / math.sqrt(len(differences))
                table[label][f"{name}_minus_{args.reference}"] = [center, center - half, center + half]
    table = {
        label: {
            name: ([round(v, 4) for v in value] if isinstance(value, list) else round(value, 4))
            for name, value in row.items()
        }
        for label, row in table.items()
    }
    result = {"people": len(people), "models": table}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
