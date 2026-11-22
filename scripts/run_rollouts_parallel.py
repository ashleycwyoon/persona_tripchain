#!/usr/bin/env python3
"""Parallel environment rollouts: several scripts/run_rollouts.py workers, then a merge.

Each worker handles every N-th person and writes its own shard file next to
--output; when all workers finish, the shards are merged in the order of the
Student1 file. With --resume, workers skip people they already finished and
reuse matching rows of an existing --output file.
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

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]


def count(path: Path) -> int:
    """Number of non-empty lines of a file (0 when it does not exist yet)."""
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--student1", required=True)
    parser.add_argument("--student2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N people of --student1 (0: everyone)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rollout-id", default="primary")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.student1).open(encoding="utf-8") as handle:
        source_ids = [str(json.loads(line)["person_id"]) for line in handle if line.strip()]
    if args.limit:
        source_ids = source_ids[: args.limit]
    requested = args.limit or len(source_ids)
    shards = [output.with_name(f"{output.stem}.shard{i}-of-{args.workers}{output.suffix}") for i in range(args.workers)]
    logs = [path.with_name(path.name + ".log") for path in shards]
    processes = []
    handles = []
    try:
        for index in range(args.workers):
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_rollouts.py"),
                "--config",
                args.config,
                "--student1",
                args.student1,
                "--student2",
                args.student2,
                "--output",
                str(shards[index]),
                "--limit",
                str(args.limit),
                "--rollout-id",
                args.rollout_id,
                "--num-shards",
                str(args.workers),
                "--shard-index",
                str(index),
            ]
            if args.resume:
                command.append("--resume")
                # Rows already merged into the output (e.g. copied from an
                # earlier run) are reused by the worker that owns each person.
                if output.exists():
                    command.extend(["--prior", str(output)])
            handle = logs[index].open("a" if args.resume else "w", encoding="utf-8")
            handles.append(handle)
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        previous = 0
        with tqdm(total=requested, desc=f"environment ({args.workers} workers)", unit="person") as progress:
            while any(process.poll() is None for process in processes):
                current = min(requested, sum(count(path) for path in shards))
                if current > previous:
                    progress.update(current - previous)
                    previous = current
                progress.set_postfix(done=current)
                time.sleep(5)
            current = min(requested, sum(count(path) for path in shards))
            if current > previous:
                progress.update(current - previous)

        failures = [process.returncode for process in processes if process.returncode]
        if failures:
            raise SystemExit(f"rollout workers failed: {failures}; inspect {logs}")

        # A retried person has several rows in a shard; the last one written wins.
        rows = {}
        for shard in shards:
            if not shard.exists():
                continue
            for line in shard.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["person_id"])] = row
        temporary = output.with_name(output.name + ".merge.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for person_id in source_ids:
                if person_id in rows:
                    handle.write(json.dumps(rows[person_id], ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "requested": requested,
                    "merged": sum(person_id in rows for person_id in source_ids),
                    "output": str(output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
