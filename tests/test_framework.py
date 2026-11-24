from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tripchain.graphhopper import GraphHopperClient
from tripchain.optimizer import optimize_chain
from tripchain.rewards import score_rollout
from tripchain.scheduling import schedule_activity
from tripchain.rollout import merge_policy_outputs
from tripchain.environment import TripChainEnvironment
from tripchain.refinement import requires_student1_repair
from tripchain.residence import (
    augment_demographics,
    residence_from_activities,
    validate_residence_demographics,
)
from tripchain.inference import build_student2_user


class FakeClient:
    def routes(self, origin, destination, mode, departure_min):
        times = {"walk": 30.0, "bike": 10.0, "car": 8.0, "bus": 20.0, "subway": 15.0}
        return [
            {
                "route_id": f"{mode}:{origin['poi_id']}:{destination['poi_id']}",
                "mode": mode,
                "routing_profile": mode,
                "travel_time_min": times[mode],
                "distance_km": 3.0,
                "transfer_count": 0,
                "travel_time_source": "fake",
            }
        ]


def poi(name):
    return {"poi_id": name, "latitude": 37.5, "longitude": 127.0}


def write_rewards_config(tmp_path):
    """A minimal --config for scripts/prepare_policy_updates.py: reward settings, no distribution term."""
    path = tmp_path / "rewards.yaml"
    path.write_text(
        json.dumps(
            {
                "rewards": {  # JSON is valid YAML
                    "weights": {
                        "activity": 1.0,
                        "location": 1.0,
                        "dwell": 1.0,
                        "mode_preference": 0.5,
                        "feasibility": 1.5,
                        "start_time": 1.0,
                    },
                    "dwell_tolerance_min": 120.0,
                    "start_time_tolerance_min": 60.0,
                    "future_discount": 0.95,
                    "feasibility": {
                        "bonus": 1.0,
                        "time_per_minute": 0.01,
                        "route_per_leg": 0.5,
                        "mode_per_violation": 0.5,
                        "activity_per_violation": 0.5,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def optimizer_config(modes):
    return {
        "scheduling": {
            "day_minutes": 1440,
        },
        "graphhopper": {"modes": modes},
        "optimization": {
            "max_states_per_signature": 16,
            "max_total_states": 256,
            "selection_temperature": 0.01,
            "selection_seed": 1030,
            "diversity_time_bucket_min": 30,
            "preference_weight": 1.0,
            "distance_weight": 0.0,
            "transfer_weight": 0.0,
            "waiting_weight": 0.0,
        },
    }


def test_bike_time_is_walk_time_divided_by_three(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"paths": [{"time": 900000, "distance": 3000}]}).encode()

    monkeypatch.setattr("tripchain.graphhopper.urlopen", lambda *args, **kwargs: Response())
    client = GraphHopperClient(
        {
            "base_url": "http://localhost:8989",
            "service_date": "2026-08-06",
            "timezone_offset": "+09:00",
            "timeout_sec": 1,
            "bike_walk_speedup": 3,
        }
    )
    origin, destination = poi("a"), poi("b")
    walk = client.routes(origin, destination, "walk", 100)[0]
    bike = client.routes(origin, destination, "bike", 100)[0]
    assert walk["travel_time_min"] == 15
    assert bike["travel_time_min"] == 5
    assert bike["travel_time_source"] == "walk_divided_by_3"
    assert bike["routing_profile"] == "bike_proxy_walk"


def test_scheduler_does_not_force_day_to_1440_or_stretch_home():
    record = {
        "person_id": "p1",
        "demographics": {"can_drive": "no"},
        "activity_sequence": ["home", "work", "home"],
        "location_dwell_by_activity": [
            {"dwell_time_min": 100},
            {"dwell_time_min": 200},
            {"dwell_time_min": 100},
        ],
        "mode_preference": [{"rank": 1, "mode": "walk", "count": 2}],
    }
    selected, _ = optimize_chain(
        record,
        [[poi("h")], [poi("w")], [poi("h")]],
        FakeClient(),
        optimizer_config(["walk"]),
    )
    assert selected is not None
    assert selected["day_end_min"] == 460
    assert selected["unallocated_day_slack_min"] == 980
    assert selected["activities"][-1]["predicted_dwell_time_min"] == 100
    assert selected["activities"][-1]["end_min"] - selected["activities"][-1]["start_min"] == 100


def test_final_home_residual_fills_the_day_instead_of_overrunning():
    record = {
        "person_id": "p1",
        "demographics": {"can_drive": "no"},
        "activity_sequence": ["home", "work", "home"],
        "location_dwell_by_activity": [
            {"dwell_time_min": 400},
            {"dwell_time_min": 600},
            {"dwell_time_min": 900},  # would end past 1440 if taken literally
        ],
        "mode_preference": [{"rank": 1, "mode": "walk", "count": 2}],
    }
    layers = [[poi("h")], [poi("w")], [poi("h")]]
    literal, _ = optimize_chain(record, layers, FakeClient(), optimizer_config(["walk"]))
    assert literal is None
    config = optimizer_config(["walk"])
    config["scheduling"]["final_home_residual"] = True
    selected, _ = optimize_chain(record, layers, FakeClient(), config)
    assert selected is not None
    assert selected["day_end_min"] == 1440
    assert selected["activities"][1]["end_min"] - selected["activities"][1]["start_min"] == 600


def test_final_residual_also_ends_a_day_away_from_home():
    record = {
        "person_id": "p1",
        "demographics": {"can_drive": "no"},
        "activity_sequence": ["home", "work"],
        "location_dwell_by_activity": [{"dwell_time_min": 1000}, {"dwell_time_min": 900}],
        "mode_preference": [{"rank": 1, "mode": "walk", "count": 1}],
    }
    layers = [[poi("h")], [poi("w")]]
    config = optimizer_config(["walk"])
    config["scheduling"]["final_home_residual"] = True
    selected, _ = optimize_chain(record, layers, FakeClient(), config)
    assert selected is not None
    assert selected["day_end_min"] == 1440
    assert selected["activities"][-1]["activity_type"] == "work"


def test_dwell_reward_skips_final_home_when_it_is_residual():
    from tripchain.rewards import dwell_reward

    sequence = ["home", "work", "home"]
    predicted = [{"dwell_time_min": 400}, {"dwell_time_min": 600}, {"dwell_time_min": 100}]
    target = [{"dwell_time_min": 400}, {"dwell_time_min": 600}, {"dwell_time_min": 400}]
    assert dwell_reward(sequence, sequence, predicted, target, 30.0) < 1.0
    assert dwell_reward(sequence, sequence, predicted, target, 30.0, final_home_residual=True) == 1.0


def test_private_car_continuity_never_teleports_vehicle():
    record = {
        "person_id": "driver",
        "demographics": {"can_drive": "yes"},
        "activity_sequence": ["home", "work", "shopping", "home"],
        "location_dwell_by_activity": [{"dwell_time_min": 30}] * 4,
        "mode_preference": [
            {"rank": 1, "mode": "car_driver", "count": 2},
            {"rank": 2, "mode": "walk", "count": 1},
        ],
    }
    selected, _ = optimize_chain(
        record,
        [[poi("h")], [poi("w")], [poi("s")], [poi("h")]],
        FakeClient(),
        optimizer_config(["car", "walk"]),
    )
    assert selected is not None
    car_location = "h"
    for edge in selected["edges"]:
        if edge["mode"] == "car_driver":
            assert edge["origin_poi_id"] == car_location
            car_location = edge["destination_poi_id"]


def test_dense_reward_passes_future_feasibility_to_both_students():
    policy = {
        "activity_sequence": ["home", "work", "home"],
        "location_dwell_by_activity": [
            {"location": {"gu": "A", "administrative_dong_name": "x"}, "dwell_time_min": 100},
            {"location": {"gu": "A", "administrative_dong_name": "y"}, "dwell_time_min": 200},
            {"location": {"gu": "A", "administrative_dong_name": "x"}, "dwell_time_min": 100},
        ],
        "mode_preference": [{"rank": 1, "mode": "walk", "count": 2}],
    }
    target = {
        "activity_sequence": policy["activity_sequence"],
        "activities": policy["location_dwell_by_activity"],
        "mode_preference": policy["mode_preference"],
    }
    config = {
        "rewards": {
            "weights": {
                "activity": 1,
                "location": 1,
                "dwell": 1,
                "mode_preference": 1,
                "feasibility": 2,
            },
            "dwell_tolerance_min": 120,
            "future_discount": 0.9,
            "feasibility": {
                "bonus": 1,
                "time_per_minute": 0.01,
                "route_per_leg": 0.5,
                "mode_per_violation": 0.5,
                "activity_per_violation": 0.5,
            },
        }
    }
    result = score_rollout(policy, target, {"feasible": True, "violations": {}}, config)
    assert result["components"]["feasibility"] == 1
    assert result["returns"]["student2"] > 3
    assert result["returns"]["student1"] > 4


def test_activity_starts_on_arrival_without_time_windows():
    # Time windows were removed: a late arrival is kept and starts when it arrives.
    record = {
        "person_id": "late",
        "demographics": {"can_drive": "no"},
        "activity_sequence": ["home", "work"],
        "location_dwell_by_activity": [{"dwell_time_min": 500}, {"dwell_time_min": 60}],
        "mode_preference": [{"rank": 1, "mode": "walk", "count": 1}],
    }
    selected, _ = optimize_chain(record, [[poi("h")], [poi("w")]], FakeClient(), optimizer_config(["walk"]))
    assert selected is not None
    work = selected["activities"][1]
    assert work["start_min"] == work["arrival_min"] and work["waiting_min"] == 0
    timing = schedule_activity("work", 50, 60)
    assert timing["start_min"] == 50 and timing["end_min"] == 110 and timing["waiting_min"] == 0


def test_realized_return_is_feasibility_only():
    policy = {
        "activity_sequence": ["home"],
        "location_dwell_by_activity": [{"location": {}, "dwell_time_min": 1440}],
        "mode_preference": [],
    }
    target = {"activity_sequence": ["home"], "activities": policy["location_dwell_by_activity"], "mode_preference": []}
    config = {
        "rewards": {
            "weights": {"activity": 0, "location": 0, "dwell": 0, "mode_preference": 0, "feasibility": 2},
            "dwell_tolerance_min": 120,
            "future_discount": 0.5,
            "feasibility": {
                "bonus": 1,
                "time_per_minute": 0,
                "route_per_leg": 0,
                "mode_per_violation": 0,
                "activity_per_violation": 0,
            },
        }
    }
    environment = {"feasible": True, "violations": {}, "final_trip_chain": {"activities": [{"activity_type": "home"}]}}
    result = score_rollout(policy, target, environment, config)
    assert "time_window" not in result["components"]
    assert result["returns"]["student2"] == pytest.approx(0.5 * 2)
    assert result["returns"]["student1"] == pytest.approx(0.25 * 2)


def test_start_time_reward_compares_realized_starts_with_person_gt():
    from tripchain.rewards import start_time_reward

    target = [
        {"activity_type": "home", "start_time_min": 0},
        {"activity_type": "work", "start_time_min": 300},
        {"activity_type": "home", "start_time_min": 900},
    ]
    chain = {
        "activities": [
            {"activity_type": "home", "start_min": 0},
            {"activity_type": "work", "start_min": 360},
            {"activity_type": "home", "start_min": 900},
        ]
    }
    sequence = ["home", "work", "home"]
    # First home is excluded; work is 60 min late (exp(-1)), final home exact.
    assert start_time_reward(sequence, {"final_trip_chain": chain}, target, 60) == pytest.approx((math.exp(-1) + 1) / 2)
    # An extra unmatched activity scores 0 through the denominator.
    longer = {
        "activities": chain["activities"][:2] + [{"activity_type": "meal", "start_min": 500}] + chain["activities"][2:]
    }
    assert start_time_reward(sequence, {"final_trip_chain": longer}, target, 60) == pytest.approx(
        (math.exp(-1) + 1) / 3
    )
    # No realized schedule: nothing to judge.
    assert start_time_reward(sequence, {}, target, 60) == 0.0


def test_start_time_reward_enters_both_returns_as_realized_outcome():
    policy = {"activity_sequence": ["home", "work", "home"], "location_dwell_by_activity": [], "mode_preference": []}
    target = {
        "activity_sequence": ["home", "work", "home"],
        "mode_preference": [],
        "activities": [
            {"activity_type": "home", "start_time_min": 0},
            {"activity_type": "work", "start_time_min": 300},
            {"activity_type": "home", "start_time_min": 900},
        ],
    }
    config = {
        "rewards": {
            "weights": {
                "activity": 0,
                "location": 0,
                "dwell": 0,
                "mode_preference": 0,
                "feasibility": 0,
                "start_time": 1,
            },
            "dwell_tolerance_min": 120,
            "start_time_tolerance_min": 60,
            "future_discount": 0.5,
            "feasibility": {
                "bonus": 0,
                "time_per_minute": 0,
                "route_per_leg": 0,
                "mode_per_violation": 0,
                "activity_per_violation": 0,
            },
        }
    }
    environment = {
        "feasible": True,
        "violations": {},
        "final_trip_chain": {
            "activities": [
                {"activity_type": "home", "start_min": 0},
                {"activity_type": "work", "start_min": 300},
                {"activity_type": "home", "start_min": 900},
            ]
        },
    }
    result = score_rollout(policy, target, environment, config)
    assert result["components"]["start_time"] == pytest.approx(1.0)
    assert result["returns"]["student2"] == pytest.approx(0.5)
    assert result["returns"]["student1"] == pytest.approx(0.25)


def test_student_prompt_interfaces_match_original_two_stage_schema():
    student1 = {
        "person_id": "p",
        "demographics": {"age": 30},
        "activity_sequence": ["home"],
        "persona_bundle": {
            "professional_persona": "p",
            "family_persona": "f",
            "persona": "g",
            "hobbies_and_interests_list": ["h"],
        },
        "reasoning1": "r1",
    }
    student2 = {
        **student1,
        "location_dwell_by_activity": [
            {
                "activity_index": 0,
                "location": {"scope": "seoul", "administrative_dong_name": "Sajik-dong", "gu": "Jongno-gu"},
                "dwell_time_min": 100,
            }
        ],
        "mode_preference": [{"rank": 1, "mode": "walk", "count": 1}],
        "reasoning2": "r2",
    }
    merged = merge_policy_outputs(student1, student2)
    assert merged["activity_sequence"] == ["home"]
    assert set(merged) == {
        "person_id",
        "demographics",
        "activity_sequence",
        "persona_bundle",
        "reasoning1",
        "location_dwell_by_activity",
        "mode_preference",
        "reasoning2",
    }


def test_home_only_day_is_physically_feasible():
    class HomeGrounder:
        def sample_layers(self, record):
            home = {"poi_id": "home", "latitude": 37.5, "longitude": 127.0}
            return [[home]], [{"activity_index": 0, "synthetic_home_anchor": True}]

    environment = TripChainEnvironment.__new__(TripChainEnvironment)
    environment.config = {
        "activity_constraints": {
            "require_start_home": True,
        },
        "scheduling": {
            "day_minutes": 1440,
        },
        "graphhopper": {"modes": ["walk"]},
        "optimization": {
            "max_states_per_signature": 16,
            "max_total_states": 256,
            "selection_temperature": 0.1,
            "selection_seed": 1030,
            "diversity_time_bucket_min": 30,
            "preference_weight": 1.0,
            "distance_weight": 0.0,
            "transfer_weight": 0.0,
            "waiting_weight": 0.0,
        },
    }
    environment.grounder = HomeGrounder()
    environment.client = object()
    result = environment.evaluate(
        {
            "person_id": "home-day",
            "activity_sequence": ["home"],
            "location_dwell_by_activity": [
                {
                    "activity_index": 0,
                    "location": {"scope": "seoul", "administrative_dong_name": "Sajik-dong", "gu": "Jongno-gu"},
                    "dwell_time_min": 1440,
                }
            ],
            "mode_preference": [],
        }
    )
    assert result["feasible"] is True
    assert result["final_trip_chain"]["day_end_min"] == 1440
    assert result["violations"]["activity_count"] == 0
    assert not requires_student1_repair({"failure_reason": "NO_FEASIBLE_GLOBAL_COMBINATION"})


def test_residence_is_added_to_the_demographics_seen_by_both_students():
    activities = [
        {
            "activity_type": "home",
            "location": {
                "sido": "Seoul",
                "gu": "Gangnam-gu",
                "administrative_dong_name": "Samseong 2-dong",
            },
        }
    ]
    residence = residence_from_activities(activities)
    assert residence == {
        "sido": "Seoul",
        "sigungu": "Gangnam-gu",
        "administrative_dong_name": "Samseong 2-dong",
    }
    demographics = augment_demographics({"age": 30}, residence)
    validate_residence_demographics(demographics)
    assert demographics["residence"] == residence


def test_residence_validation_rejects_incomplete_demographics():
    try:
        validate_residence_demographics({"residence": {"sido": "Seoul"}})
    except ValueError as error:
        assert "demographics.residence" in str(error)
    else:
        raise AssertionError("incomplete residence must be rejected")


def test_student2_prompt_is_the_original_four_key_schema_without_location_list():
    record = {
        "person_id": "p",
        "demographics": {
            "age": 30,
            "residence": {
                "sido": "Seoul",
                "sigungu": "Gangnam-gu",
                "administrative_dong_name": "Samseong 2-dong",
            },
        },
        "activity_sequence": ["home", "leisure", "home"],
        "persona_bundle": {},
        "reasoning1": "test",
        "status": "ok",
    }
    user = build_student2_user(record)
    assert set(user) == {"demographics", "activity_sequence", "persona_bundle", "reasoning1"}


def test_rollout_trace_keeps_exact_student2_prompt_for_policy_update(tmp_path):
    import subprocess
    import sys

    from tripchain.rollout import run_rollout

    class FixedEnvironment:
        def evaluate(self, record):
            return {"feasible": True, "violations": {}}

    demographics = {"age": 30}
    student2_input = {
        "demographics": demographics,
        "activity_sequence": ["home"],
        "persona_bundle": {},
        "reasoning1": "r1",
    }
    student1 = {
        "person_id": "p",
        "demographics": demographics,
        "activity_sequence": ["home"],
        "persona_bundle": {},
        "reasoning1": "r1",
    }
    student2 = {
        "person_id": "p",
        "demographics": demographics,
        "activity_sequence": ["home"],
        "location_dwell_by_activity": [],
        "mode_preference": [],
        "reasoning2": "r2",
        "student2_input": student2_input,
    }
    config = {
        "rewards": {
            "weights": {"activity": 1, "location": 1, "dwell": 1, "mode_preference": 1, "feasibility": 1},
            "dwell_tolerance_min": 120,
            "future_discount": 0.9,
            "feasibility": {
                "bonus": 1,
                "time_per_minute": 0,
                "route_per_leg": 0,
                "mode_per_violation": 0,
                "activity_per_violation": 0,
            },
        }
    }
    rows = []
    for rollout_id, reasoning in (("a", "r2"), ("b", "other")):
        row = run_rollout(
            student1,
            {**student2, "reasoning2": reasoning},
            {"activity_sequence": ["home"]},
            FixedEnvironment(),
            config,
            rollout_id,
        )
        rows.append(row)
    assert rows[0]["decision_trace"]["student2"]["input"] == student2_input
    assert rows[0]["decision_trace"]["student1"]["input"] == {"demographics": demographics}

    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_policy_updates.py"
    config_path = write_rewards_config(tmp_path)
    subprocess.run(
        [sys.executable, str(script), str(rollouts), "--output-dir", str(tmp_path / "u"), "--config", config_path],
        check=True,
        capture_output=True,
    )
    update = json.loads((tmp_path / "u/student2_updates.jsonl").read_text().splitlines()[0])
    assert json.loads(update["messages"][0]["content"]) == student2_input

    legacy = dict(rows[0])
    legacy["decision_trace"] = {
        **rows[0]["decision_trace"],
        "student2": {**rows[0]["decision_trace"]["student2"], "input": None},
    }
    rollouts.write_text(json.dumps(legacy) + "\n" + json.dumps(rows[1]) + "\n", encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, str(script), str(rollouts), "--output-dir", str(tmp_path / "v"), "--config", config_path],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0 and "no stored prompt input" in failed.stderr


def test_student2_failure_resamples_the_unchanged_prompt(monkeypatch):
    from tripchain import inference

    calls = []
    outputs = iter(['{"bad": 1}', '{"ok": 1}'])

    def fake_generate(model, tokenizer, users, max_tokens, temperature, top_p):
        calls.append((users[0], temperature))
        return [next(outputs)]

    def validator(value):
        if "ok" not in value:
            raise ValueError("invalid")

    monkeypatch.setattr(inference, "generate_batch", fake_generate)
    user = {"demographics": {}, "activity_sequence": ["home"]}
    value = inference.resample_valid(None, None, user, 10, 4, 0.7, 0.9, validator)
    assert value == {"ok": 1}
    assert calls == [(user, 0.7), (user, 0.7)]


def test_resume_merge_keeps_repaired_base_rows_over_stale_shards(tmp_path):
    from tripchain.parallel_inference import merge_successful_outputs

    base = tmp_path / "student1.jsonl"
    shard = tmp_path / "student1.shard0-of-1.jsonl"
    base.write_text(json.dumps({"person_id": "p", "activity_sequence": ["repaired"]}) + "\n")
    shard.write_text(
        json.dumps({"person_id": "p", "activity_sequence": ["stale"]})
        + "\n"
        + json.dumps({"person_id": "q", "activity_sequence": ["new"]})
        + "\n"
    )
    merged = merge_successful_outputs([{"person_id": "p"}, {"person_id": "q"}], base, [shard], include_base=True)
    rows = [json.loads(line) for line in base.read_text().splitlines()]
    assert merged == 2
    assert [row["activity_sequence"] for row in rows] == [["repaired"], ["new"]]


def test_location_and_dwell_rewards_align_activities_like_evaluation():
    from tripchain.rewards import dwell_reward, location_reward

    def item(gu, dong, dwell):
        return {"location": {"gu": gu, "administrative_dong_name": dong}, "dwell_time_min": dwell}

    target_sequence = ["home", "work", "home"]
    target = [item("Mapo-gu", "a", 600), item("Gangnam-gu", "b", 480), item("Mapo-gu", "a", 300)]
    predicted_sequence = ["home", "shopping", "work", "home"]
    predicted = [
        item("Mapo-gu", "a", 600),
        item("Jung-gu", "c", 60),
        item("Gangnam-gu", "b", 480),
        item("Mapo-gu", "a", 300),
    ]
    # All three matched activities are correct; the extra activity scores 0.
    assert location_reward(predicted_sequence, target_sequence, predicted, target) == 3 / 4
    assert dwell_reward(predicted_sequence, target_sequence, predicted, target, 120) == 3 / 4


def test_two_level_groups_compare_student2_within_one_student1_output(tmp_path):
    import subprocess
    import sys

    def rollout(branch, sample, s1_return, s2_return):
        return {
            "person_id": "p",
            "rollout_id": f"{branch}-{sample}",
            "status": "ok",
            "decision_trace": {
                "student1": {"input": {"demographics": {}}, "output": {"activity_sequence": [branch]}},
                "student2": {"input": {"activity_sequence": [branch]}, "output": {"reasoning2": sample}},
            },
            "rewards": {"components": {}, "returns": {"student1": s1_return, "student2": s2_return}},
        }

    rows = [
        rollout("A", "1", 1.0, 1.0),
        rollout("A", "2", 3.0, 3.0),
        rollout("B", "1", 0.0, 9.0),
        rollout("B", "2", 0.0, 9.5),
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_policy_updates.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(path),
            "--output-dir",
            str(tmp_path / "u"),
            "--config",
            write_rewards_config(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    read = lambda name: [json.loads(line) for line in (tmp_path / "u" / name).read_text().splitlines()]

    student2 = {row["rollout_id"]: row["group_advantage"] for row in read("student2_updates.jsonl")}
    # B's high Student2 returns are not compared against A's samples: each branch is
    # centered on its own mean, then both share one pooled std, so B's near-tie
    # (9.0 vs 9.5) gets a smaller advantage than A's clear gap (1.0 vs 3.0).
    scale = math.sqrt((1.0 + 1.0 + 0.0625 + 0.0625) / 4)
    assert student2 == pytest.approx(
        {"A-1": -1.0 / scale, "A-2": 1.0 / scale, "B-1": -0.25 / scale, "B-2": 0.25 / scale}
    )

    student1 = read("student1_updates.jsonl")
    assert [(row["return"], row["group_advantage"], row["branch_size"]) for row in student1] == [
        (2.0, 1.0, 2),
        (0.0, -1.0, 2),
    ]


def test_pooled_advantages_keep_between_group_magnitudes():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts/prepare_policy_updates.py"
    spec = importlib.util.spec_from_file_location("prepare_policy_updates", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    clear, tie, three = module.pooled_advantages([[0.0, 4.0], [1.0, 1.1], [0.0, 1.0, 2.0]])
    # Centered within each group, so every group sums to zero.
    assert sum(clear) == pytest.approx(0.0) and sum(three) == pytest.approx(0.0)
    # A per-group z-score would give the near-tie group +/-1 like the clear one.
    assert abs(tie[0]) < 0.05 * abs(clear[0])
    flat = clear + tie + three
    assert math.sqrt(sum(value * value for value in flat) / len(flat)) == pytest.approx(1.0)
    # All-equal returns: zero std falls back to 1.0 instead of dividing by zero.
    assert module.pooled_advantages([[2.0, 2.0], [5.0, 5.0]]) == [[0.0, 0.0], [0.0, 0.0]]


def test_trace_output_order_follows_the_configured_profile():
    from tripchain.common import OUTPUT_ORDERS
    from tripchain.rollout import run_rollout

    assert OUTPUT_ORDERS["reason_first"] == (
        ("persona_bundle", "reasoning1", "activity_sequence"),
        ("reasoning2", "location_dwell_by_activity", "mode_preference"),
    )

    class FixedEnvironment:
        def evaluate(self, record):
            return {"feasible": True, "violations": {}}

    student1 = {
        "person_id": "p",
        "demographics": {},
        "activity_sequence": ["home"],
        "persona_bundle": {},
        "reasoning1": "r1",
    }
    student2 = {
        "person_id": "p",
        "activity_sequence": ["home"],
        "location_dwell_by_activity": [],
        "mode_preference": [],
        "reasoning2": "r2",
        "student2_input": {},
    }
    for name, (order1, order2) in OUTPUT_ORDERS.items():
        config = {
            "inference": {"output_order": name},
            "rewards": {
                "weights": {"activity": 1, "location": 1, "dwell": 1, "mode_preference": 1, "feasibility": 1},
                "dwell_tolerance_min": 120,
                "future_discount": 0.9,
                "feasibility": {
                    "bonus": 1,
                    "time_per_minute": 0,
                    "route_per_leg": 0,
                    "mode_per_violation": 0,
                    "activity_per_violation": 0,
                },
            },
        }
        trace = run_rollout(student1, student2, {"activity_sequence": ["home"]}, FixedEnvironment(), config)[
            "decision_trace"
        ]
        assert tuple(trace["student1"]["output"]) == order1
        assert tuple(trace["student2"]["output"]) == order2


def test_outside_seoul_activity_is_set_aside_like_tripchain_1015():
    from tripchain.rewards import feasibility_reward

    environment = TripChainEnvironment.__new__(TripChainEnvironment)
    environment.config = {"activity_constraints": {"require_start_home": True}}

    def record(home_scope, work_scope):
        place = lambda scope: {"location": {"scope": scope}, "dwell_time_min": 60}
        return {
            "person_id": "p",
            "activity_sequence": ["home", "work", "home"],
            "location_dwell_by_activity": [place(home_scope), place(work_scope), place(home_scope)],
        }

    outside = environment.evaluate(record("seoul", "external"))
    assert outside["status"] == "outside_poi_coverage" and outside["feasible"] is None
    config = {
        "rewards": {
            "feasibility": {
                "bonus": 1,
                "time_per_minute": 0.01,
                "route_per_leg": 0.5,
                "mode_per_violation": 0.5,
                "activity_per_violation": 0.5,
            }
        }
    }
    assert feasibility_reward(outside, config) == 0.0
    home_outside = environment.evaluate(record("external", "seoul"))
    assert home_outside["feasible"] is False and "HOME_OUTSIDE_SEOUL" in home_outside["failure_reason"]


def test_outside_poi_coverage_rollouts_are_not_penalized_in_groups():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts/prepare_policy_updates.py"
    spec = importlib.util.spec_from_file_location("prepare_policy_updates", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rewards = {"weights": {"feasibility": 1.5}, "future_discount": 0.5}

    def row(status, feasibility, local):
        realized = 1.5 * feasibility
        return {
            "status": status,
            "rewards": {
                "components": {"feasibility": feasibility},
                "returns": {"student2": local + 0.5 * realized, "student1": local + 0.25 * realized},
            },
        }

    # A group with an unjudged sample is compared on its local terms only: the outside
    # sample with the better location wins, and it cannot escape an infeasible sibling.
    rows = [row("ok", 1.0, 1.0), row("infeasible", 0.0, 1.0), row("outside_poi_coverage", 0.0, 2.0)]
    assert module.neutral_returns(rows, rewards) == [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]
    # Without an unjudged sample, the full returns are kept.
    judged = rows[:2]
    assert module.neutral_returns(judged, None) == [(1.0 + 0.25 * 1.5, 1.0 + 0.5 * 1.5), (1.0, 1.0)]


def test_outside_seoul_is_scored_as_a_location_choice():
    from tripchain.rewards import location_match

    def seoul(dong, gu):
        return {"scope": "seoul", "administrative_dong_name": dong, "gu": gu}

    def observed(dong, gu, sido="Seoul"):
        return {"administrative_dong_name": dong, "gu": gu, "sido": sido}

    def external(sigungu, sido):
        return {"scope": "external", "sido": sido, "sigungu": sigungu}

    assert location_match(seoul("a", "Mapo-gu"), observed("a", "Mapo-gu")) == 1.0
    assert location_match(seoul("b", "Mapo-gu"), observed("a", "Mapo-gu")) == 0.5
    # Sigungu is the finest level Student2 can name outside Seoul: full credit.
    truth = observed("Gochon-eup", "Gimpo-si", "Gyeonggi-do")
    assert location_match(external("Gimpo-si", "Gyeonggi-do"), truth) == 1.0
    assert location_match(external("Suwon-si", "Gyeonggi-do"), truth) == 0.5
    # Wrong side of the Seoul boundary scores 0 either way.
    assert location_match(external("Gimpo-si", "Gyeonggi-do"), observed("a", "Mapo-gu")) == 0.0
    assert location_match(seoul("a", "Mapo-gu"), truth) == 0.0


def trip_mode_record(trip_modes, can_drive="yes"):
    return {
        "person_id": "tm",
        "demographics": {"can_drive": can_drive},
        "activity_sequence": ["home", "work", "home"],
        "location_dwell_by_activity": [{"dwell_time_min": 400}, {"dwell_time_min": 500}, {"dwell_time_min": 400}],
        "trip_modes": trip_modes,
    }


def test_each_leg_is_routed_in_the_student2_trip_mode():
    layers = [[poi("h")], [poi("w")], [poi("h")]]
    selected, stats = optimize_chain(
        trip_mode_record(["subway", "bus"]),
        layers,
        FakeClient(),
        optimizer_config(["walk", "bike", "bus", "subway", "car"]),
    )
    assert [edge["mode"] for edge in selected["edges"]] == ["subway", "bus"]
    assert not any(edge["mode_fallback"] for edge in selected["edges"])
    assert stats["mode_fallback_transitions"] == 0


def test_unrealizable_trip_mode_falls_back_and_counts_as_mode_violation():
    from tripchain.environment import _violations

    # Took the subway to work, so the car is still at home for the drive back.
    layers = [[poi("h")], [poi("w")], [poi("h")]]
    selected, stats = optimize_chain(
        trip_mode_record(["subway", "car_driver"]),
        layers,
        FakeClient(),
        optimizer_config(["walk", "bus", "subway", "car"]),
    )
    assert selected is not None
    assert selected["edges"][1]["mode_fallback"] is True
    assert selected["edges"][1]["mode"] != "car_driver"
    assert _violations(selected, stats)["mode_count"] == 1


def test_main_mode_is_the_longest_vehicle_stage():
    from tripchain.trip_modes import main_mode, mode_preference_from_trips

    assert main_mode([("walk", 8), ("bus", 5), ("subway", 30), ("walk", 6)]) == "subway"
    assert main_mode([("walk", 20), ("subway", 5)]) == "subway"  # walking to transit is access
    assert main_mode([("walk", 15)]) == "walk"
    assert main_mode([("bus", 10), ("subway", 10)]) == "subway"  # tie: survey order
    assert mode_preference_from_trips(["subway", "walk", "subway"])[0] == {"rank": 1, "mode": "subway", "count": 2}


def test_trip_mode_reward_aligns_trips_through_their_arrival_activities():
    from tripchain.rewards import trip_mode_reward

    sequence = ["home", "work", "home"]
    assert trip_mode_reward(sequence, sequence, ["subway", "bus"], ["subway", "walk"]) == 0.5
    assert trip_mode_reward(["home"], ["home"], [], []) == 1.0
    # A missing activity drops its trip; the denominator keeps the longer chain.
    assert (
        trip_mode_reward(
            ["home", "work", "home"], ["home", "shopping", "work", "home"], ["bus", "bus"], ["walk", "bus", "bus"]
        )
        == 2 / 3
    )


def test_student2_trip_mode_schema_is_validated():
    from tripchain.inference import validate_student2

    location = {"scope": "seoul", "administrative_dong_name": "a", "gu": "b"}
    value = {
        "reasoning2": "r",
        "location_dwell_by_activity": [
            {"activity_index": i, "location": location, "dwell_time_min": 10} for i in range(3)
        ],
        "trip_modes": ["subway", "walk"],
    }
    validate_student2(value, ["home", "work", "home"])
    with pytest.raises(ValueError):
        validate_student2({**value, "trip_modes": ["subway"]}, ["home", "work", "home"])
    with pytest.raises(ValueError):
        validate_student2({**value, "trip_modes": ["subway", "rocket"]}, ["home", "work", "home"])


def test_distribution_reward_penalizes_overgenerated_values():
    from tripchain.distribution_reward import build_distribution_rewards

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    hwh, hwoh = ["home", "work", "home"], ["home", "work", "other", "home"]
    truth = [(demo, hwh, ["subway", "subway"])] * 5 + [(demo, hwoh, ["bus", "walk", "bus"])] * 5
    branches = [(demo, hwh)] * 9 + [(demo, hwoh)]
    rollouts = [(demo, ["subway", "subway"])] * 9 + [(demo, ["bus", "walk", "bus"])]
    sequence, modes = build_distribution_rewards(branches, rollouts, truth, prior_strength=1.0, clip=3.0)
    assert sequence[0] < 0 < sequence[-1]
    assert modes[0] < 0 < modes[-1]


def test_distribution_reward_leaves_the_sample_out_of_q():
    # A rare pattern under-generated as a class (1 of 10 vs 3 of 10 in the truth)
    # but unique in its cell: with itself counted in Q it was penalized.
    from tripchain.distribution_reward import build_distribution_rewards

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    hwh, rare = ["home", "work", "home"], ["home", "work", "other", "shopping", "home"]
    truth = [(demo, hwh, [])] * 7 + [(demo, rare, [])] * 3
    branches = [(demo, hwh)] * 9 + [(demo, rare)]
    sequence, _ = build_distribution_rewards(branches, [], truth, prior_strength=1.0, clip=3.0)
    assert sequence[-1] > 0 > sequence[0]


def test_distance_mode_reward_changes_the_mode_of_the_right_trips():
    # Truth: short trips walk, long trips subway. The policy uses subway for half of
    # the short trips. Demographic-only cells see too much subway and also punish
    # subway on long trips; distance cells punish only the short-trip subway.
    from tripchain.distribution_reward import build_distribution_rewards

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    truth = [(demo, [], ["walk", "subway"], ["same_dong", "10km+"])] * 10
    rollouts = [(demo, ["walk", "subway"], ["same_dong", "10km+"])] * 5 + [
        (demo, ["subway", "subway"], ["same_dong", "10km+"])
    ] * 5
    _, by_distance = build_distribution_rewards([], rollouts, truth, prior_strength=1.0, clip=3.0)
    _, by_demo = build_distribution_rewards(
        [], [r[:2] for r in rollouts], [t[:3] for t in truth], prior_strength=1.0, clip=3.0
    )
    assert by_distance[0] > 0 > by_distance[-1]
    # long-trip subway is right in both rollouts: its distance-cell reward is the same
    # for good and bad rollouts, so the gap comes from the short trip only
    assert by_distance[0] - by_distance[-1] > by_demo[0] - by_demo[-1] > 0


def test_distance_mode_reward_leaves_own_trips_out_per_cell():
    from collections import Counter
    from tripchain.distribution_reward import MODE_DISTANCE_LEVELS, CellDistribution

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    item = lambda band, mode: ({**demo, "distance_band": band}, mode)
    q = CellDistribution(
        [item("same_dong", "walk")] * 3 + [item("10km+", "subway")] * 3,
        {"walk", "subway"},
        prior_strength=1.0,
        levels=MODE_DISTANCE_LEVELS,
    )
    own = [item("same_dong", "walk"), item("10km+", "subway")]
    exclude = q.own_counts_by_key(own)
    # the short-trip cell loses one walk, the long-trip cell one subway, the root both
    keys_short = [k for k in exclude if k and k[0] == "same_dong"]
    assert all(exclude[k] == Counter({"walk": 1}) for k in keys_short)
    assert exclude[()] == Counter({"walk": 1, "subway": 1})
    # leaving one walk out of three still makes walk the dominant short-trip mode
    assert q.prob(*own[0], exclude_by_key=exclude) > 0.5


def test_trip_distance_bands():
    from tripchain.trip_distance import TripDistance

    centroids = {"1": (37.50, 127.00), "2": (37.51, 127.00), "3": (37.60, 127.00)}
    d = TripDistance(centroids, {("a-dong", "b-gu"): "2"})
    loc = lambda code: {"administrative_dong_code": code}
    assert d.band(loc("1"), loc("1")) == "same_dong"
    assert d.band(loc("1"), {"administrative_dong_name": "A-dong", "gu": "B-gu"}) == "<3km"
    assert d.band(loc("1"), loc("3")) == "10km+"
    assert d.band(loc("1"), loc("4150000000")) == "outside_seoul"
    assert d.band(loc("1"), {"scope": "external"}) == "outside_seoul"
    assert d.day_bands([{"location": loc("1")}, {"location": loc("1")}, {"location": loc("3")}]) == [
        "same_dong",
        "10km+",
    ]


def test_departure_reward_spreads_hours_within_the_same_purpose():
    # Truth: workers leave at 5, 7 and 8; the policy sends 9 of 10 at 8 o'clock.
    from tripchain.distribution_reward import build_departure_rewards

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    truth = [(demo, "work", "5")] * 3 + [(demo, "work", "7")] * 3 + [(demo, "work", "8")] * 4
    rollouts = [(demo, "work", "8")] * 9 + [(demo, "work", "5"), None]
    rewards = build_departure_rewards(rollouts, truth, prior_strength=1.0, clip=3.0)
    assert rewards[0] < 0 < rewards[9]
    assert rewards[10] == 0.0


def test_departure_reward_compares_within_the_first_purpose():
    # Shopping leaves at 13 in the truth; a 13 o'clock shopper is not pushed
    # towards the workers' 8 o'clock although workers dominate the people.
    from tripchain.distribution_reward import build_departure_rewards

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    truth = [(demo, "work", "8")] * 20 + [(demo, "shopping", "13")] * 5
    rollouts = [(demo, "work", "8")] * 20 + [(demo, "shopping", "13")] * 3 + [(demo, "shopping", "8")] * 2
    rewards = build_departure_rewards(rollouts, truth, prior_strength=1.0, clip=3.0)
    assert rewards[20] > 0 > rewards[-1]


def test_student1_rejects_unknown_activity_types():
    from tripchain.inference import validate_student1

    value = {
        "activity_sequence": ["home", "farm", "home"],
        "persona_bundle": {
            "professional_persona": "",
            "family_persona": "",
            "persona": "",
            "hobbies_and_interests_list": [],
        },
        "reasoning1": "x",
    }
    with pytest.raises(ValueError, match="unknown activity"):
        validate_student1(value)
    validate_student1({**value, "activity_sequence": ["home", "work", "home"]})


def test_mode_fallback_penalty_is_a_share_of_the_day_legs():
    from tripchain.rewards import feasibility_reward

    cfg = {
        "rewards": {
            "feasibility": {
                "bonus": 1,
                "time_per_minute": 0.01,
                "route_per_leg": 0.5,
                "mode_share_penalty": 1.0,
                "activity_per_violation": 0.5,
            }
        }
    }

    def day(legs, fallbacks):
        trips = [{"mode_fallback": i < fallbacks} for i in range(legs)]
        return {
            "status": "ok",
            "feasible": True,
            "final_trip_chain": {"trips": trips},
            "violations": {"time_min": 0, "route_count": 0, "mode_count": fallbacks, "activity_count": 0},
        }

    # home-work-home with one unroutable leg keeps the former per-violation 0.5
    assert feasibility_reward(day(2, 1), cfg) == 0.5
    # the same fallback rate on a longer day is no longer penalized more
    assert feasibility_reward(day(4, 2), cfg) == feasibility_reward(day(2, 1), cfg)
    assert feasibility_reward(day(6, 1), cfg) > feasibility_reward(day(2, 1), cfg)
    # configs without the key keep the per-violation count
    old = {"rewards": {"feasibility": {**cfg["rewards"]["feasibility"], "mode_per_violation": 0.5}}}
    del old["rewards"]["feasibility"]["mode_share_penalty"]
    assert feasibility_reward(day(4, 2), old) == 0.0


def test_distance_band_reward_moves_far_work_trips_closer():
    from tripchain.distribution_reward import distance_band_rewards

    demo = {"occupation": "office_worker", "age": 40, "gender": "female"}
    # GT: work trips mostly short; policy: half of the work trips far.
    truth = [(demo, ["work", "home"], ["<3km", "<3km"])] * 8 + [(demo, ["work", "home"], ["10km+", "10km+"])] * 2
    near = (demo, ["work", "home"], ["<3km", "<3km"])
    far = (demo, ["work", "home"], ["10km+", "10km+"])
    rewards = distance_band_rewards([near, far] * 5, truth, prior_strength=1.0, clip=3.0)
    assert rewards[0] > 0 > rewards[1]
    # purpose matters: far shopping trips are judged in the shopping cell, not by work trips
    shop_truth = truth + [(demo, ["shopping", "home"], ["10km+", "10km+"])] * 10
    shop = (demo, ["shopping", "home"], ["10km+", "10km+"])
    r = distance_band_rewards([near, far] * 5 + [shop] * 10, shop_truth, prior_strength=1.0, clip=3.0)
    assert r[-1] > r[1]
