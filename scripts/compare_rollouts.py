#!/usr/bin/env python3
"""Per-person comparison of two rollout files (e.g. pre-RL and post-RL) on the people they share.

Writes one JSON report: feasible counts, mean weighted reward before and after,
and each person's status, reward components, returns and activity sequence in
both files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    return {str(row["person_id"]): row for row in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pre", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pre, post = load(args.pre), load(args.post)
    shared = sorted(set(pre) & set(post))
    people = []
    for person_id in shared:
        before, after = pre[person_id], post[person_id]
        b_reward, a_reward = before["rewards"], after["rewards"]
        people.append(
            {
                "person_id": person_id,
                "pre_status": before["status"],
                "post_status": after["status"],
                "pre_feasible": before["decision_trace"]["environment"].get("feasible"),
                "post_feasible": after["decision_trace"]["environment"].get("feasible"),
                "pre_weighted_reward": b_reward["weighted_total"],
                "post_weighted_reward": a_reward["weighted_total"],
                "delta_weighted_reward": a_reward["weighted_total"] - b_reward["weighted_total"],
                "pre_components": b_reward["components"],
                "post_components": a_reward["components"],
                "pre_returns": b_reward["returns"],
                "post_returns": a_reward["returns"],
                "pre_activity_sequence": before["decision_trace"]["student1"]["output"]["activity_sequence"],
                "post_activity_sequence": after["decision_trace"]["student1"]["output"]["activity_sequence"],
            }
        )
    mean_pre = sum(item["pre_weighted_reward"] for item in people) / max(1, len(people))
    mean_post = sum(item["post_weighted_reward"] for item in people) / max(1, len(people))
    report = {
        "people": len(people),
        "pre_feasible_count": sum(item["pre_feasible"] is True for item in people),
        "post_feasible_count": sum(item["post_feasible"] is True for item in people),
        "mean_pre_weighted_reward": mean_pre,
        "mean_post_weighted_reward": mean_post,
        "delta_mean_weighted_reward": mean_post - mean_pre,
        "per_person": people,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
