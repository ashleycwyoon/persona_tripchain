#!/usr/bin/env python3
"""Environment rollouts: ground, route and schedule each person's Student outputs, then score them.

For every person with a Student1 output, a Student2 output and a ground truth,
writes one rollout row: the decision trace (both prompts and answers, POI
candidates, selected routes, schedule, violations), the dense reward components
and the discounted returns of both Students. Needs the GraphHopper server
(scripts/start_graphhopper.sh). This is the single-process worker;
scripts/run_rollouts_parallel.py runs several of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.common import append_jsonl, load_config, resolve, rows_by_id  # noqa: E402
from tripchain.environment import TripChainEnvironment  # noqa: E402
from tripchain.rollout import run_rollout  # noqa: E402


def reusable_prior_rows(args, person_ids: list[str], done: set[str], student1: dict, student2: dict) -> list[dict]:
    """Rows of --prior for this shard's people, if made from the same Student outputs."""
    if not (args.resume and args.prior and Path(args.prior).exists()):
        return []
    wanted = set(person_ids)
    rows = []
    for line in Path(args.prior).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        person_id = str(row["person_id"])
        if (
            person_id not in wanted
            or f"{person_id}|{args.rollout_id}" in done
            or row.get("rollout_id") != args.rollout_id
            or row.get("status") not in {"ok", "infeasible", "outside_poi_coverage"}
        ):
            continue
        trace = row.get("decision_trace") or {}
        policy = (trace.get("environment") or {}).get("policy_output") or {}
        if (
            (trace.get("student1") or {}).get("output", {}).get("activity_sequence")
            != student1[person_id].get("activity_sequence")
            or policy.get("trip_modes") != student2[person_id].get("trip_modes")
            or policy.get("location_dwell_by_activity") != student2[person_id].get("location_dwell_by_activity")
        ):
            continue
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--student1")
    parser.add_argument("--student2")
    parser.add_argument("--ground-truth")
    parser.add_argument("--output")
    parser.add_argument("--rollout-id", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prior",
        help="with --resume: a merged rollouts file whose rows for these "
        "people and Student outputs are reused instead of rerun",
    )
    parser.add_argument("--skip-healthcheck", action="store_true")
    parser.add_argument("--active-pool-size", type=int, default=None)
    parser.add_argument("--graphhopper-timeout", type=float, default=None)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.active_pool_size is not None:
        cfg["poi_sampling"]["active_pool_size"] = args.active_pool_size
    if args.graphhopper_timeout is not None:
        cfg["graphhopper"]["timeout_sec"] = args.graphhopper_timeout
    paths = cfg["paths"]
    student1 = rows_by_id(args.student1 or paths["student1_predictions"])
    student2 = rows_by_id(args.student2 or paths["student2_predictions"])
    targets = rows_by_id(args.ground_truth or paths["ground_truth_reference"])
    person_ids = [pid for pid in student1 if pid in student2 and pid in targets]
    if args.limit:
        person_ids = person_ids[: args.limit]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        person_ids = [
            person_id for index, person_id in enumerate(person_ids) if index % args.num_shards == args.shard_index
        ]
    output = resolve(args.output or paths["rollout_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                # Failed rows (e.g. GraphHopper unavailable) are retried; the
                # parallel merge keeps the last row written for a person.
                if row.get("status") in {"ok", "infeasible", "outside_poi_coverage"}:
                    done.add(f"{row['person_id']}|{row.get('rollout_id', '0')}")

    prior = reusable_prior_rows(args, person_ids, done, student1, student2)

    environment = TripChainEnvironment(cfg)
    if not args.skip_healthcheck and not environment.client.healthcheck():
        raise RuntimeError(f"GraphHopper healthcheck failed: {cfg['graphhopper']['base_url']}")
    mode = "a" if args.resume else "w"
    summary = {"ok": 0, "infeasible": 0, "outside_poi_coverage": 0, "failed": 0, "skipped": 0}
    with output.open(mode, encoding="utf-8") as handle:
        for row in prior:
            append_jsonl(handle, row)
            done.add(f"{row['person_id']}|{args.rollout_id}")
        for person_id in tqdm(person_ids, desc="dense-reward rollouts", unit="person"):
            key = f"{person_id}|{args.rollout_id}"
            if key in done:
                summary["skipped"] += 1
                continue
            try:
                row = run_rollout(
                    student1[person_id],
                    student2[person_id],
                    targets[person_id],
                    environment,
                    cfg,
                    args.rollout_id,
                )
                append_jsonl(handle, row)
                summary[row["status"]] += 1
            except Exception as exc:
                append_jsonl(
                    handle,
                    {
                        "person_id": person_id,
                        "rollout_id": args.rollout_id,
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}:{exc}",
                    },
                )
                summary["failed"] += 1
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
