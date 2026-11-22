#!/usr/bin/env python3
"""Launch one or more KTDB teacher workers and show aggregate progress with ETA."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ktdb_two_stage_teacher import (  # noqa: E402
    DEFAULT_ACTIVITY,
    DEFAULT_DEMOGRAPHICS,
    DEFAULT_SPATIAL,
    load_teacher_records,
)
from src.paths import OUTPUT_ROOT  # noqa: E402


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpus", default="0",
        help="comma-separated data-parallel GPU IDs; use 0 for one-GPU teacher inference",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--download-dir", type=Path, default=Path("/data-fast/chaewon/huggingface"))
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument(
        "--output-prefix", type=Path,
        default=OUTPUT_ROOT / "reports" / "ktdb_teacher_qwen32b_english_demographics",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-temperature", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU index")
    shard_count = len(gpus)

    records = load_teacher_records(args.demographics, args.activity, args.spatial)
    total = len(records)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [Path(f"{args.output_prefix}_shard{i}.jsonl") for i in range(shard_count)]
    errors = [path.with_name(path.name + ".errors.jsonl") for path in outputs]
    logs = [path.with_name(path.name + ".log") for path in outputs]
    initial_error_counts = [line_count(path) for path in errors]

    if not args.resume:
        existing = [path for path in outputs + errors if path.exists()]
        if existing:
            names = "\n".join(f"  {path}" for path in existing)
            raise SystemExit(f"Output already exists; use --resume or a new --output-prefix:\n{names}")

    processes: list[subprocess.Popen[str]] = []
    log_handles = []
    env = os.environ.copy()
    available_cpus = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else (os.cpu_count() or shard_count))
    threads = max(1, available_cpus // shard_count)
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
        for shard, gpu in enumerate(gpus):
            command = [
                args.python,
                str(ROOT / "scripts" / "09c_abduce_ktdb_two_stage.py"),
                "--model", args.model,
                "--download-dir", str(args.download_dir),
                "--demographics", str(args.demographics),
                "--activity", str(args.activity),
                "--spatial", str(args.spatial),
                "--out", str(outputs[shard]),
                "--shard", f"{shard}/{shard_count}",
                "--gpu", gpu,
                "--batch-size", str(args.batch_size),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-new-tokens", str(args.max_new_tokens),
                "--max-retries", str(args.max_retries),
                "--retry-temperature", str(args.retry_temperature),
            ]
            if args.resume:
                command.append("--resume")
            log_handle = logs[shard].open("a" if args.resume else "w", encoding="utf-8")
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

        initial = min(total, sum(line_count(path) for path in outputs))
        with tqdm(
            total=total,
            initial=initial,
            desc=f"KTDB teacher ({shard_count} GPU{'s' if shard_count != 1 else ''})",
            unit="person",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        ) as progress:
            progress.set_postfix_str("loading models")
            previous = initial
            while any(process.poll() is None for process in processes):
                new_errors = sum(
                    max(0, line_count(path) - starting)
                    for path, starting in zip(errors, initial_error_counts)
                )
                current = min(total, sum(line_count(path) for path in outputs) + new_errors)
                if current > previous:
                    progress.update(current - previous)
                    previous = current
                    progress.set_postfix_str(
                        f"ok={sum(line_count(p) for p in outputs)} retry_errors={new_errors}"
                    )
                time.sleep(2)
            new_errors = sum(
                max(0, line_count(path) - starting)
                for path, starting in zip(errors, initial_error_counts)
            )
            current = min(total, sum(line_count(path) for path in outputs) + new_errors)
            if current > previous:
                progress.update(current - previous)
            progress.set_postfix_str(
                f"ok={sum(line_count(p) for p in outputs)} retry_errors={new_errors}"
            )

        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            failed = [f"GPU{gpu}: exit={code}, log={log}"
                      for gpu, code, log in zip(gpus, return_codes, logs) if code]
            raise SystemExit("Teacher shard failed:\n" + "\n".join(failed))
        print("Teacher inference complete.")
        for path in outputs:
            print(path)
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
