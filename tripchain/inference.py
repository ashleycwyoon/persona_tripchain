"""Student1 / Student2 inference on one GPU: prompts, generation and output validation.

Student1 maps demographics to {persona_bundle, reasoning1, activity_sequence}.
Student2 maps demographics plus the Student1 output to {reasoning2,
location_dwell_by_activity, trip_modes}. Outputs are validated after generation
(there is no constrained decoding); an invalid output is regenerated, and every
accepted row stores the exact prompt that produced it for the GRPO update.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .common import append_jsonl, clean_json, completed_ids, load_config, read_jsonl, resolve
from .residence import validate_residence_demographics


PERSONA_FIELDS = {"professional_persona", "family_persona", "persona", "hobbies_and_interests_list"}
SEOUL_LOCATION_KEYS = {"scope", "administrative_dong_name", "gu"}
EXTERNAL_LOCATION_KEYS = {"scope", "sido", "sigungu"}
# KTDB 2021 activity purposes as named in the SFT targets; anything else
# (e.g. "farm", "study") is an invalid Student1 output and is resampled.
ACTIVITY_LABELS = {
    "home",
    "work",
    "school",
    "shopping",
    "eat_out",
    "leisure",
    "other",
    "escort_care",
    "medical",
}
MODE_LABELS = {
    "walk",
    "car_driver",
    "car_passenger",
    "bus",
    "intercity_bus",
    "other_bus",
    "subway",
    "rail",
    "high_speed_rail",
    "taxi",
    "truck",
    "bicycle",
    "motorcycle",
    "personal_mobility",
    "mobility_aid",
    "air",
    "ferry",
    "other",
}


def load_location_whitelist(path: str | Path) -> dict[str, Any]:
    """The KTDB location codebook Student2 locations are validated against."""
    value = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not {"seoul", "external_sigungu"} <= set(value):
        raise ValueError(f"invalid location whitelist: {path}")
    return {"seoul": value["seoul"], "external_sigungu": value["external_sigungu"]}


def build_student2_user(record: dict[str, Any]) -> dict[str, Any]:
    """Student2 prompt: the original four-key schema, with no location list.

    Locations are validated against the full codebook after generation, so the
    prompt depends only on the Student1 output it conditions on.
    """
    validate_residence_demographics(record.get("demographics") or {})
    return {
        "demographics": record["demographics"],
        "activity_sequence": record["activity_sequence"],
        "persona_bundle": record["persona_bundle"],
        "reasoning1": record["reasoning1"],
    }


def shard_output_path(path: Path, shard_index: int, num_shards: int) -> Path:
    """Return a collision-free output path for one data-parallel worker."""
    return path.with_name(f"{path.stem}.shard{shard_index}-of-{num_shards}{path.suffix}")


def _source_for_worker(source: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    """The first --limit rows, then this worker's shard of them."""
    if args.limit:
        source = source[: args.limit]
    if args.num_shards > 1:
        source = [row for index, row in enumerate(source) if index % args.num_shards == args.shard_index]
    return source


def _completed_for_resume(base_output: Path, args) -> set[str]:
    if not args.resume:
        return set()
    paths = [base_output]
    if args.num_shards > 1:
        paths.extend(shard_output_path(base_output, index, args.num_shards) for index in range(args.num_shards))
    done: set[str] = set()
    for path in paths:
        done.update(completed_ids(path))
    return done


def validate_student1(value: dict[str, Any]) -> None:
    """Raise ValueError unless `value` is a well-formed Student1 answer."""
    if set(value) != {"activity_sequence", "persona_bundle", "reasoning1"}:
        raise ValueError(f"invalid Student1 keys: {sorted(value)}")
    if not isinstance(value["activity_sequence"], list) or not value["activity_sequence"]:
        raise ValueError("empty activity_sequence")
    unknown = [activity for activity in value["activity_sequence"] if activity not in ACTIVITY_LABELS]
    if unknown:
        raise ValueError(f"unknown activity types: {unknown}")
    if not isinstance(value["persona_bundle"], dict) or set(value["persona_bundle"]) != PERSONA_FIELDS:
        raise ValueError("invalid persona_bundle")
    if not isinstance(value["reasoning1"], str) or not value["reasoning1"].strip():
        raise ValueError("empty reasoning1")


def validate_student2(
    value: dict[str, Any], activity_sequence: list[str], whitelist: dict[str, Any] | None = None
) -> None:
    """Raise ValueError unless `value` is a well-formed Student2 answer for this sequence."""
    # Two schemas: a person-level mode ranking, or one main mode per trip
    # (output_order reason_first_tripmode; trip i goes from activity i to i+1).
    mode_key = "trip_modes" if "trip_modes" in value else "mode_preference"
    expected = {"location_dwell_by_activity", mode_key, "reasoning2"}
    if set(value) != expected:
        raise ValueError(f"invalid Student2 keys: {sorted(value)}")
    activities = value["location_dwell_by_activity"]
    if not isinstance(activities, list) or len(activities) != len(activity_sequence):
        raise ValueError("activity/location length mismatch")
    for index, activity in enumerate(activities):
        if set(activity) != {"activity_index", "location", "dwell_time_min"}:
            raise ValueError(f"invalid spatial keys at {index}")
        if activity["activity_index"] != index:
            raise ValueError(f"non-contiguous activity_index at {index}")
        location = activity["location"]
        if not isinstance(location, dict):
            raise ValueError(f"invalid location at {index}")
        keys = set(location)
        if keys == SEOUL_LOCATION_KEYS and location.get("scope") == "seoul":
            gu = location.get("gu")
            dong = location.get("administrative_dong_name")
            if not all(isinstance(item, str) and item.strip() for item in (gu, dong)):
                raise ValueError(f"invalid Seoul location at {index}")
            if whitelist is not None and dong not in whitelist["seoul"].get(gu, []):
                raise ValueError(f"Seoul location is outside codebook whitelist at {index}: {gu}|{dong}")
        elif keys == EXTERNAL_LOCATION_KEYS and location.get("scope") == "external":
            sido, sigungu = location.get("sido"), location.get("sigungu")
            if not all(isinstance(item, str) and item.strip() for item in (sido, sigungu)):
                raise ValueError(f"invalid external location at {index}")
            allowed = {(item["sido"], item["sigungu"]) for item in (whitelist or {}).get("external_sigungu", [])}
            if whitelist is not None and (sido, sigungu) not in allowed:
                raise ValueError(f"external location is outside codebook whitelist at {index}: {sido}|{sigungu}")
        else:
            raise ValueError(f"invalid location schema at {index}: {sorted(keys)}")
        if not isinstance(activity["dwell_time_min"], int) or activity["dwell_time_min"] < 0:
            raise ValueError(f"invalid dwell_time_min at {index}")
    if mode_key == "trip_modes":
        trip_modes = value["trip_modes"]
        if not isinstance(trip_modes, list) or len(trip_modes) != len(activity_sequence) - 1:
            raise ValueError("trip_modes must name one mode per trip")
        for mode in trip_modes:
            if mode not in MODE_LABELS:
                raise ValueError(f"invalid trip mode: {mode}")
        return
    if not isinstance(value["mode_preference"], list):
        raise ValueError("invalid mode_preference")
    for expected_rank, item in enumerate(value["mode_preference"], 1):
        if not isinstance(item, dict) or set(item) != {"rank", "mode", "count"}:
            raise ValueError("invalid mode_preference item")
        if (
            item["rank"] != expected_rank
            or not isinstance(item["mode"], str)
            or not isinstance(item["count"], int)
            or item["count"] < 0
        ):
            raise ValueError("invalid mode_preference rank/mode/count")
        if item["mode"] not in MODE_LABELS:
            raise ValueError(f"invalid mode_preference mode: {item['mode']}")
    if "incoming_modes" in json.dumps(value, ensure_ascii=False):
        raise ValueError("Student2 emitted forbidden incoming_modes")


def load_model(adapter: Path, base_model: str, cache_dir: Path):
    """4-bit base model with the LoRA adapter applied, and the adapter's tokenizer."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    user_values: list[dict[str, Any]],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> list[str]:
    """Generate one answer per compact-JSON user message (greedy when temperature is 0)."""
    import torch

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for value in user_values
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
        add_special_tokens=False,
    ).to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        generated = model.generate(**encoded, **kwargs)
    new_tokens = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _retry_one(
    model, tokenizer, user: dict[str, Any], max_tokens: int, retries: int, temperature: float, top_p: float, validator
) -> dict[str, Any]:
    """Regenerate one answer: greedy first, then `retries` samples at `temperature`."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = generate_batch(
                model,
                tokenizer,
                [user],
                max_tokens,
                0.0 if attempt == 0 else temperature,
                top_p,
            )[0]
            value = clean_json(raw)
            validator(value)
            return value
        except Exception as exc:  # generation errors are retained in the final error file
            last_error = exc
    raise RuntimeError(str(last_error))


def resample_valid(
    model, tokenizer, user: dict[str, Any], max_tokens: int, attempts: int, temperature: float, top_p: float, validator
) -> dict[str, Any]:
    """Draw fresh samples from the unchanged prompt until one validates.

    Unlike a correction prompt, every accepted sample comes from the same
    policy distribution and prompt that GRPO later scores.
    """
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            raw = generate_batch(model, tokenizer, [user], max_tokens, temperature, top_p)[0]
            value = clean_json(raw)
            validator(value)
            return value
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def student2_output_row(row: dict[str, Any], value: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Student2 output row: the Student1 fields it conditions on, its answer and its prompt."""
    return {
        "person_id": str(row["person_id"]),
        "demographics": row["demographics"],
        "activity_sequence": row["activity_sequence"],
        "persona_bundle": row["persona_bundle"],
        "reasoning1": row["reasoning1"],
        **value,
        "student2_input": user,
        "status": "ok",
    }


def run_student1(args, cfg: dict[str, Any]) -> None:
    paths, inf = cfg["paths"], cfg["inference"]
    source = _source_for_worker(list(read_jsonl(paths["test_demographics"])), args)
    for row in source:
        validate_residence_demographics(row.get("demographics") or {})
    base_output = resolve(paths["student1_predictions"])
    output = shard_output_path(base_output, args.shard_index, args.num_shards) if args.num_shards > 1 else base_output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_for_resume(base_output, args)
    mode = "a" if args.resume else "w"
    pending = [row for row in source if str(row["person_id"]) not in done]
    if not pending:
        print(f"Student1: all {len(source):,} records already completed; nothing to resume.")
        return
    model, tokenizer = load_model(resolve(paths["student1_adapter"]), inf["base_model"], Path(paths["model_cache"]))
    error_output = output.with_name(output.name + ".errors.jsonl")
    initial = len(source) - len(pending)
    counts = {"ok": 0, "error": 0}
    with output.open(mode, encoding="utf-8") as handle, error_output.open(mode, encoding="utf-8") as error_handle:
        with tqdm(
            total=len(source),
            initial=initial,
            desc="Student1 inference",
            unit="person",
            dynamic_ncols=True,
            position=args.progress_position,
            disable=args.no_progress,
        ) as progress:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                # This wrapper matches the residence-augmented Student1 SFT user schema.
                users = [{"demographics": row["demographics"]} for row in batch]
                raw_values = generate_batch(
                    model,
                    tokenizer,
                    users,
                    inf["student1_max_new_tokens"],
                    float(inf["temperature"]),
                    float(inf["top_p"]),
                )
                for row, user, raw in zip(batch, users, raw_values):
                    try:
                        value = clean_json(raw)
                        validate_student1(value)
                    except Exception as first_error:
                        try:
                            value = _retry_one(
                                model,
                                tokenizer,
                                user,
                                inf["student1_max_new_tokens"],
                                inf["max_retries"],
                                inf["retry_temperature"],
                                float(inf["top_p"]),
                                validate_student1,
                            )
                        except Exception as final_error:
                            append_jsonl(
                                error_handle,
                                {
                                    "person_id": str(row["person_id"]),
                                    "status": "error",
                                    "error": f"{type(first_error).__name__}:{first_error}; retry:{final_error}",
                                    "raw_output": raw[:4000],
                                },
                            )
                            counts["error"] += 1
                            progress.update(1)
                            progress.set_postfix(ok=counts["ok"], error=counts["error"])
                            continue
                    append_jsonl(
                        handle,
                        {
                            "person_id": str(row["person_id"]),
                            "demographics": row["demographics"],
                            **value,
                            "student1_input": user,
                            "status": "ok",
                        },
                    )
                    counts["ok"] += 1
                    progress.update(1)
                    progress.set_postfix(ok=counts["ok"], error=counts["error"])


def run_student2(args, cfg: dict[str, Any]) -> None:
    paths, inf = cfg["paths"], cfg["inference"]
    source = _source_for_worker(list(read_jsonl(paths["student1_predictions"])), args)
    for row in source:
        validate_residence_demographics(row.get("demographics") or {})
    base_output = resolve(paths["student2_predictions"])
    output = shard_output_path(base_output, args.shard_index, args.num_shards) if args.num_shards > 1 else base_output
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_for_resume(base_output, args)
    mode = "a" if args.resume else "w"
    pending = [row for row in source if str(row["person_id"]) not in done]
    if not pending:
        print(f"Student2: all {len(source):,} records already completed; nothing to resume.")
        return
    model, tokenizer = load_model(resolve(paths["student2_adapter"]), inf["base_model"], Path(paths["model_cache"]))
    location_whitelist = load_location_whitelist(paths["location_whitelist"])
    error_output = output.with_name(output.name + ".errors.jsonl")
    initial = len(source) - len(pending)
    counts = {"ok": 0, "error": 0}
    with output.open(mode, encoding="utf-8") as handle, error_output.open(mode, encoding="utf-8") as error_handle:
        with tqdm(
            total=len(source),
            initial=initial,
            desc="Student2 inference",
            unit="person",
            dynamic_ncols=True,
            position=args.progress_position,
            disable=args.no_progress,
        ) as progress:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                users = [build_student2_user(row) for row in batch]
                raw_values = generate_batch(
                    model,
                    tokenizer,
                    users,
                    inf["student2_max_new_tokens"],
                    float(inf["temperature"]),
                    float(inf["top_p"]),
                )
                for row, user, raw in zip(batch, users, raw_values):
                    validator = lambda value, seq=row["activity_sequence"]: validate_student2(
                        value, seq, location_whitelist
                    )
                    try:
                        value = clean_json(raw)
                        validator(value)
                    except Exception as first_error:
                        # Resample the same prompt; persistent failures are left to
                        # scripts/repair_student2_failures.py (Student1 then Student2).
                        try:
                            value = resample_valid(
                                model,
                                tokenizer,
                                user,
                                inf["student2_max_new_tokens"],
                                int(inf.get("student2_resample_attempts", 4)),
                                float(inf["temperature"]),
                                float(inf["top_p"]),
                                validator,
                            )
                        except Exception as final_error:
                            append_jsonl(
                                error_handle,
                                {
                                    "person_id": str(row["person_id"]),
                                    "status": "error",
                                    "error": f"{type(first_error).__name__}:{first_error}; resample:{final_error}",
                                    "raw_output": raw[:4000],
                                },
                            )
                            counts["error"] += 1
                            progress.update(1)
                            progress.set_postfix(ok=counts["ok"], error=counts["error"])
                            continue
                    append_jsonl(handle, student2_output_row(row, value, user))
                    counts["ok"] += 1
                    progress.update(1)
                    progress.set_postfix(ok=counts["ok"], error=counts["error"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["student1", "student2"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--student1-input", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--progress-position", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must satisfy 0 <= shard-index < num-shards")
    cfg = load_config(args.config)
    if args.output:
        cfg["paths"][f"{args.stage}_predictions"] = args.output
    if args.student1_input:
        cfg["paths"]["student1_predictions"] = args.student1_input
    if args.adapter:
        cfg["paths"][f"{args.stage}_adapter"] = args.adapter
    args.batch_size = args.batch_size or int(cfg["inference"]["batch_size"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed = int(args.seed if args.seed is not None else cfg["inference"]["seed"])
    random.seed(seed)
    # Sampling uses PyTorch's RNG. Seed it explicitly so temperature/top-p
    # decoding is reproducible for a fixed command, shard layout, and batch size.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.stage == "student1":
        run_student1(args, cfg)
    else:
        run_student2(args, cfg)


if __name__ == "__main__":
    main()
