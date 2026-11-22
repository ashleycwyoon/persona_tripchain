"""Data-parallel Student inference: one independent worker (shard) per GPU, then a merge.

Each worker runs scripts/run_student_inference.py on every N-th person and
writes its own shard file, so there are no concurrent writes. When all workers
finish, the successful rows are merged in source-person order. Re-running the
same command with --resume continues where the previous run stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .common import ROOT, completed_ids, load_config, read_jsonl, resolve
from .inference import shard_output_path


def _stage_paths(stage: str, cfg: dict[str, Any]) -> tuple[Path, Path]:
    """(input file, merged output file) of a stage."""
    paths = cfg["paths"]
    if stage == "student1":
        return resolve(paths["test_demographics"]), resolve(paths["student1_predictions"])
    return resolve(paths["student1_predictions"]), resolve(paths["student2_predictions"])


def _source_rows(stage: str, source_path: Path, limit: int) -> list[dict[str, Any]]:
    rows = list(read_jsonl(source_path))
    if limit:
        rows = rows[:limit]
    if stage == "student2":
        rows = [row for row in rows if row.get("status", "ok") == "ok"]
    return rows


def _successful_ids(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        done.update(completed_ids(path))
    return done


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def merge_successful_outputs(
    source: list[dict[str, Any]], base_output: Path, shard_outputs: list[Path], include_base: bool
) -> int:
    """Atomically merge successful records, deduplicated in source-person order."""
    records: dict[str, dict[str, Any]] = {}
    # The merged base is authoritative: repair_student2_failures.py rewrites
    # rows there, and stale shard copies must not overwrite them on --resume.
    inputs = shard_outputs + ([base_output] if include_base and base_output.exists() else [])
    for path in inputs:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("status", "ok") == "ok" and row.get("person_id") is not None:
                records[str(row["person_id"])] = row
    ordered_ids = [str(row["person_id"]) for row in source]
    temporary = base_output.with_name(base_output.name + ".merge.tmp")
    base_output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for person_id in ordered_ids:
            if person_id in records:
                handle.write(json.dumps(records[person_id], ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(base_output)
    return sum(person_id in records for person_id in ordered_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Student inference as independent data-parallel shards on multiple GPUs."
    )
    parser.add_argument("stage", choices=["student1", "student2"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--student1-input", default=None)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU index")
    cfg = load_config(args.config)
    if args.output:
        cfg["paths"][f"{args.stage}_predictions"] = args.output
    if args.student1_input:
        cfg["paths"]["student1_predictions"] = args.student1_input
    if args.adapter:
        cfg["paths"][f"{args.stage}_adapter"] = args.adapter
    source_path, base_output = _stage_paths(args.stage, cfg)
    base_output.parent.mkdir(parents=True, exist_ok=True)
    source = _source_rows(args.stage, source_path, args.limit)
    shards = len(gpus)
    # run_student_inference.py writes a lone shard straight to the base output
    # (shard paths only when --num-shards > 1); reading a never-written
    # shard0-of-1 file here made the merge overwrite that output with nothing.
    shard_outputs = (
        [base_output] if shards == 1 else [shard_output_path(base_output, index, shards) for index in range(shards)]
    )
    logs = [path.with_name(path.name + ".log") for path in shard_outputs]
    error_paths = [path.with_name(path.name + ".errors.jsonl") for path in shard_outputs]

    if not args.resume:
        for path in shard_outputs + error_paths + logs:
            if path.exists():
                path.unlink()

    initial_done = _successful_ids(([base_output] if args.resume else []) + shard_outputs)
    source_ids = {str(row["person_id"]) for row in source}
    initial = len(initial_done & source_ids)
    initial_errors = [_line_count(path) for path in error_paths]
    processes: list[subprocess.Popen[str]] = []
    log_handles = []
    env = os.environ.copy()
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or len(gpus))
    threads = max(1, available_cpus // len(gpus))
    # Cap CPU threads when inference overlaps CPU-bound rollouts/GraphHopper.
    threads = int(os.environ.get("INFERENCE_CPU_THREADS", threads))
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            "TOKENIZERS_PARALLELISM": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )

    try:
        for shard_index, gpu in enumerate(gpus):
            command = [
                args.python,
                str(ROOT / "scripts" / "run_student_inference.py"),
                args.stage,
                "--config",
                args.config,
                "--gpu",
                gpu,
                "--batch-size",
                str(args.batch_size),
                "--num-shards",
                str(shards),
                "--shard-index",
                str(shard_index),
                "--no-progress",
                "--output",
                str(base_output),
            ]
            if args.stage == "student2":
                command.extend(["--student1-input", str(source_path)])
            if args.adapter:
                command.extend(["--adapter", args.adapter])
            if args.seed is not None:
                command.extend(["--seed", str(args.seed)])
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            if args.resume:
                command.append("--resume")
            log_handle = logs[shard_index].open("a" if args.resume else "w", encoding="utf-8")
            log_handles.append(log_handle)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        previous = initial
        with tqdm(
            total=len(source),
            initial=initial,
            desc=f"{args.stage} inference ({shards} GPUs)",
            unit="person",
            dynamic_ncols=True,
        ) as progress:
            progress.set_postfix_str("loading models")
            while any(process.poll() is None for process in processes):
                successes = len(_successful_ids(([base_output] if args.resume else []) + shard_outputs) & source_ids)
                errors = sum(max(0, _line_count(path) - start) for path, start in zip(error_paths, initial_errors))
                current = min(len(source), successes + errors)
                if current > previous:
                    progress.update(current - previous)
                    previous = current
                progress.set_postfix_str(f"ok={successes} errors={errors}")
                time.sleep(2)

            successes = len(_successful_ids(([base_output] if args.resume else []) + shard_outputs) & source_ids)
            errors = sum(max(0, _line_count(path) - start) for path, start in zip(error_paths, initial_errors))
            current = min(len(source), successes + errors)
            if current > previous:
                progress.update(current - previous)
            progress.set_postfix_str(f"ok={successes} errors={errors}")

        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            failures = [
                f"GPU {gpu}: exit={code}, log={log}" for gpu, code, log in zip(gpus, return_codes, logs) if code
            ]
            raise SystemExit("Parallel inference worker failed:\n" + "\n".join(failures))
        merged = merge_successful_outputs(source, base_output, shard_outputs, include_base=args.resume)
        print(f"{args.stage} complete: merged {merged:,}/{len(source):,} successful records -> {base_output}")
        if merged != len(source):
            print("Some records failed validation. Re-run this command with --resume to retry them.")
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
        raise SystemExit("Interrupted. Re-run the same command with --resume to continue.")
    finally:
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    main()
