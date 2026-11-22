#!/usr/bin/env python3
"""Refine infeasible rollouts with the bounded iterative refinement of tripchain/refinement.py.

Lv.1 redraws the POIs, Lv.2 regenerates Student2, Lv.3 regenerates Student1 and
then Student2. Refinement is inference-only: it is not part of the rewards or
of GRPO training. Rows that are already feasible, or outside the POI catalog,
are copied unchanged.

People are independent, so `--workers N` splits the infeasible rows across N
worker processes (spread over `--gpus`); each worker loads its own Student pair
and appends every refined row immediately, so an interrupted run resumes where
it stopped. The parent merges the rows back in input order.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.common import (  # noqa: E402
    append_jsonl,
    load_config,
    ordered_output,
    output_orders,
    rows_by_id,
)


def refine_row(row, policies, targets, cfg, level1_attempts):
    """Refine one rollout row and return it in the rollout format, with its repair history."""
    from tripchain.refinement import refine_inference
    from tripchain.rewards import score_rollout
    from tripchain.rollout import student2_trace_output

    student1_order, student2_order = output_orders(cfg)
    person_id = str(row["person_id"])
    target = targets[person_id]
    dt = row.get("decision_trace", {})
    s1 = dt.get("student1", {}).get("output", {})
    s2 = dt.get("student2", {}).get("output", {})
    student1_input = dt.get("student1", {}).get("input") or {}
    demographics = student1_input.get("demographics", {})

    # Carry the exact prompts forward; regeneration replaces them.
    record = {
        "person_id": person_id,
        "demographics": demographics,
        **s1,
        **s2,
        "student1_input": student1_input,
        "student2_input": dt.get("student2", {}).get("input"),
    }
    res = refine_inference(
        record,
        cfg,
        regenerate_student2=policies.regenerate_student2,
        regenerate_student1=policies.regenerate_student1,
        level1_attempts=level1_attempts,
    )
    reward = score_rollout(res["record"], target, res["environment"], cfg)
    return {
        "person_id": person_id,
        "rollout_id": row.get("rollout_id", "refined"),
        "status": res["status"],
        "decision_trace": {
            "student1": {
                "input": res["record"].get("student1_input") or {"demographics": demographics},
                "output": ordered_output(
                    {
                        "activity_sequence": res["record"]["activity_sequence"],
                        "persona_bundle": res["record"].get("persona_bundle"),
                        "reasoning1": res["record"].get("reasoning1"),
                    },
                    student1_order,
                ),
            },
            "student2": {
                "input": res["record"].get("student2_input"),
                "output": student2_trace_output(res["record"], student2_order),
            },
            "environment": res["environment"],
            "repair_history": res.get("repair_history", []),
        },
        "rewards": reward,
    }


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shard_path(output: Path, index: int, count: int) -> Path:
    return output.with_name(f"{output.stem}.refine-shard{index}-of-{count}{output.suffix}")


def run_worker(args, rows: list[dict]) -> None:
    """Refine this worker's shard of the infeasible rows, appending each result to its shard file."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from tripchain.policy_runtime import StudentPolicyPair

    cfg = load_config(args.config)
    if args.student1_adapter:
        cfg["paths"]["student1_adapter"] = args.student1_adapter
    if args.student2_adapter:
        cfg["paths"]["student2_adapter"] = args.student2_adapter
    targets = rows_by_id(cfg["paths"]["ground_truth_reference"])
    pending = [row for row in rows if row.get("status") == "infeasible"]
    pending = [row for index, row in enumerate(pending) if index % args.num_shards == args.shard_index]
    output = Path(args.worker_output)
    done = {str(row["person_id"]) for row in read_rows(output)}
    pending = [row for row in pending if str(row["person_id"]) not in done]
    if not pending:
        return
    policies = StudentPolicyPair(cfg)
    with output.open("a", encoding="utf-8") as handle:
        for row in tqdm(pending, desc=f"Refining shard {args.shard_index}", disable=args.no_progress):
            try:
                refined = refine_row(row, policies, targets, cfg, args.level1_attempts)
            except Exception as exc:
                refined = {**row, "refine_error": f"{type(exc).__name__}: {exc}"}
            append_jsonl(handle, refined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Path to initial rollouts.jsonl")
    parser.add_argument("--output", required=True, help="Path to final refined rollouts.jsonl")
    parser.add_argument("--student1-adapter", default=None)
    parser.add_argument("--student2-adapter", default=None)
    parser.add_argument("--level1-attempts", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="GPU for --workers 1")
    parser.add_argument("--gpus", default=None, help="comma-separated GPUs for --workers > 1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.level1_attempts < 0:
        parser.error("--level1-attempts must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path)

    if args.shard_index is not None:
        run_worker(args, rows)
        return

    infeasible_count = sum(1 for row in rows if row.get("status") == "infeasible")
    print(f"Total rows: {len(rows)}, Infeasible to refine: {infeasible_count}")
    workers = max(1, min(args.workers, infeasible_count))
    gpus = [value.strip() for value in (args.gpus or str(args.gpu)).split(",") if value.strip()]
    shards = [shard_path(output_path, index, workers) for index in range(workers)]
    processes = []
    logs = []
    if infeasible_count:
        for index in range(workers):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                args.config,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--level1-attempts",
                str(args.level1_attempts),
                "--gpu",
                gpus[index % len(gpus)],
                "--num-shards",
                str(workers),
                "--shard-index",
                str(index),
                "--worker-output",
                str(shards[index]),
                "--no-progress",
            ]
            for flag, value in (
                ("--student1-adapter", args.student1_adapter),
                ("--student2-adapter", args.student2_adapter),
            ):
                if value:
                    command.extend([flag, value])
            log = shards[index].with_name(shards[index].name + ".log").open("a", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        try:

            def finished() -> int:
                return sum(len(read_rows(path)) for path in shards)

            with tqdm(
                total=infeasible_count,
                initial=finished(),
                desc=f"Refining infeasible ({workers} workers)",
                unit="person",
            ) as progress:
                while any(process.poll() is None for process in processes):
                    progress.update(finished() - progress.n)
                    time.sleep(10)
                progress.update(finished() - progress.n)
        finally:
            for log in logs:
                log.close()
        failures = [process.returncode for process in processes if process.returncode]
        if failures:
            raise SystemExit(f"refinement workers failed: {failures}; see {shards[0].name}.log")

    refined = {str(row["person_id"]): row for path in shards for row in read_rows(path)}
    recovered = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("status") == "infeasible":
                row = refined.get(str(row["person_id"]), {**row, "refine_error": "not refined"})
                recovered += row.get("status") == "ok"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Refinement done: {recovered}/{infeasible_count} recovered to OK. Output written to {output_path}")


if __name__ == "__main__":
    main()
