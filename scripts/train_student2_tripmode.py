#!/usr/bin/env python3
"""Student2 SFT (QLoRA): P(reasoning2, location_dwell_by_activity, trip_modes | demographics, Student1 output).

The target carries one main mode per trip (`trip_modes`). Options: --config (default
configs/student2_v2_sft.yaml), --gpu, --dry-run (validate the data and the assistant-only
loss mask without training), --max-train-rows and --resume-from-checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tripchain.inference import MODE_LABELS  # noqa: E402
from tripchain.paths import expand_config  # noqa: E402

USER_KEYS = {
    "demographics",
    "activity_sequence",
    "persona_bundle",
    "reasoning1",
}
TARGET_KEYS = {"location_dwell_by_activity", "trip_modes", "reasoning2"}
PERSONA_BUNDLE_KEYS = {
    "professional_persona",
    "family_persona",
    "persona",
    "hobbies_and_interests_list",
}
SPATIAL_ACTIVITY_KEYS = {"activity_index", "location", "dwell_time_min"}
SEOUL_LOCATION_KEYS = {"scope", "administrative_dong_name", "gu"}
EXTERNAL_LOCATION_KEYS = {"scope", "sido", "sigungu"}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_row(row: dict, source: Path, line_number: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "user",
        "assistant",
    ]:
        raise ValueError(f"{source}:{line_number}: expected user/assistant messages")
    user = json.loads(messages[0]["content"])
    if set(user) != USER_KEYS:
        raise ValueError(f"{source}:{line_number}: invalid Student 2 input keys")
    if not isinstance(user["activity_sequence"], list) or not user["activity_sequence"]:
        raise ValueError(f"{source}:{line_number}: empty activity_sequence")
    bundle = user["persona_bundle"]
    if not isinstance(bundle, dict) or set(bundle) != PERSONA_BUNDLE_KEYS:
        raise ValueError(f"{source}:{line_number}: invalid persona_bundle")
    if not isinstance(bundle["hobbies_and_interests_list"], list) or not bundle["hobbies_and_interests_list"]:
        raise ValueError(f"{source}:{line_number}: invalid hobbies_and_interests_list")
    for key in PERSONA_BUNDLE_KEYS - {"hobbies_and_interests_list"}:
        if not isinstance(bundle[key], str) or not bundle[key].strip():
            raise ValueError(f"{source}:{line_number}: empty persona_bundle.{key}")
    if not isinstance(user["reasoning1"], str) or not user["reasoning1"].strip():
        raise ValueError(f"{source}:{line_number}: empty reasoning1")
    target = json.loads(messages[1]["content"])
    if set(target) != TARGET_KEYS:
        raise ValueError(f"{source}:{line_number}: invalid Student 2 target keys")
    spatial = target["location_dwell_by_activity"]
    if not isinstance(spatial, list) or len(spatial) != len(user["activity_sequence"]):
        raise ValueError(f"{source}:{line_number}: spatial/activity sequence length mismatch")
    for index, activity in enumerate(spatial):
        if not isinstance(activity, dict) or set(activity) != SPATIAL_ACTIVITY_KEYS:
            raise ValueError(f"{source}:{line_number}: invalid spatial target at index {index}")
        if activity["activity_index"] != index:
            raise ValueError(f"{source}:{line_number}: non-contiguous activity_index")
        location = activity["location"]
        if not isinstance(location, dict) or frozenset(location) not in {
            frozenset(SEOUL_LOCATION_KEYS),
            frozenset(EXTERNAL_LOCATION_KEYS),
        }:
            raise ValueError(f"{source}:{line_number}: invalid location at index {index}")
        if location["scope"] == "seoul":
            if not all(
                isinstance(location[key], str) and location[key].strip() for key in ("administrative_dong_name", "gu")
            ):
                raise ValueError(f"{source}:{line_number}: invalid Seoul location at index {index}")
        elif location["scope"] == "external":
            if not all(isinstance(location[key], str) and location[key].strip() for key in ("sido", "sigungu")):
                raise ValueError(f"{source}:{line_number}: invalid external location at index {index}")
        else:
            raise ValueError(f"{source}:{line_number}: invalid location scope at index {index}")
        if not isinstance(activity["dwell_time_min"], int) or activity["dwell_time_min"] < 0:
            raise ValueError(f"{source}:{line_number}: invalid dwell_time_min at index {index}")
    trip_modes = target["trip_modes"]
    if not isinstance(trip_modes, list) or len(trip_modes) != len(user["activity_sequence"]) - 1:
        raise ValueError(f"{source}:{line_number}: trip_modes must name one mode per trip")
    if any(mode not in MODE_LABELS for mode in trip_modes):
        raise ValueError(f"{source}:{line_number}: unknown trip mode in {trip_modes}")
    if not isinstance(target["reasoning2"], str) or not target["reasoning2"].strip():
        raise ValueError(f"{source}:{line_number}: empty reasoning2")


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            validate_row(json.loads(line), path, line_number)
            count += 1
    if count == 0:
        raise ValueError(f"{path}: empty dataset")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "student2_v2_sft.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="validate data and tokenizer masking only")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    cfg = expand_config(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    train_path = _resolve(cfg["data"]["train"])
    validation_path = _resolve(cfg["data"]["validation"])
    train_count = validate_jsonl(train_path)
    validation_count = validate_jsonl(validation_path)
    print(f"[data] train={train_count:,} validation={validation_count:,}")

    if not distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"].get("cache_dir")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.chat_template = (ROOT / "templates" / "qwen35_text_assistant_only.jinja").read_text(encoding="utf-8")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": str(train_path), "validation": str(validation_path)}
    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_rows:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_rows, len(dataset["train"]))))

    mask_probe = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    assistant_mask = mask_probe.get("assistant_masks") or mask_probe.get("assistant_tokens_mask")
    if assistant_mask is None or not any(assistant_mask):
        raise RuntimeError("chat template did not produce an assistant loss mask")
    eos_positions = [
        index for index, token_id in enumerate(mask_probe["input_ids"]) if token_id == tokenizer.eos_token_id
    ]
    if not eos_positions or not assistant_mask[eos_positions[-1]]:
        raise RuntimeError("assistant end-of-turn token is missing from the loss mask")
    if args.dry_run:
        print(f"[dry-run] assistant tokens={sum(assistant_mask)} total tokens={len(mask_probe['input_ids'])}")
        return

    import torch
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quantization_config = None
    if cfg["model"].get("qlora_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if distributed:
        torch.cuda.set_device(local_rank)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": local_rank} if distributed else "auto",
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
        )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    training = cfg["training"]
    output_dir = _resolve(training["output_dir"])
    configured_accumulation = int(training["gradient_accumulation_steps"])
    if configured_accumulation % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by distributed world size")
    gradient_accumulation = configured_accumulation // world_size
    if local_rank == 0:
        effective_batch = int(training["per_device_train_batch_size"]) * gradient_accumulation * world_size
        print(
            f"[parallel] world_size={world_size} gradient_accumulation="
            f"{gradient_accumulation} effective_batch={effective_batch}"
        )
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(training["epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training["scheduler"],
        warmup_ratio=float(training["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        bf16=bool(training["bf16"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        packing=bool(training["packing"]),
        max_length=int(cfg["model"]["max_length"]),
        assistant_only_loss=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=int(training["seed"]),
        ddp_find_unused_parameters=False if distributed else None,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] Student 2 adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
