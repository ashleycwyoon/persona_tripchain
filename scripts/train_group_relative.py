#!/usr/bin/env python3
"""Offline clipped group-relative policy update (GRPO) from environment rollouts.

Run `prepare_policy_updates.py` first. The frozen reference policy is --adapter,
the adapter that generated the rollouts (the SFT adapter in round 1); training
starts from the same adapter. Advantages are normalized within each person's
rollout group, and the clipped ratio plus a small reference penalty prevents an
unbounded negative-advantage objective.

Ratios, clipping and the reference penalty are per token (as in PPO/GRPO), then
averaged over each response. A response is ~240 tokens of mostly persona and
reasoning text, while the decisions (activities, dwell numbers) are a few tokens;
a ratio of the length-averaged log-probability hid large changes on those tokens,
so the clip never engaged. `--ratio-level sequence` keeps the former objective.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--updates", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="model download directory (default: the Hugging Face hub cache, $HF_HUB_CACHE)",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--reference-penalty", type=float, default=0.02)
    parser.add_argument(
        "--ratio-level",
        choices=("token", "sequence"),
        default="token",
        help="clip and penalize per-token ratios (default) or the length-averaged ratio",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=0, help="intermediate checkpoints every N steps (0: none)")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="continue an interrupted run from its checkpoint (optimizer, scheduler, data position)",
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    rows = [json.loads(line) for line in Path(args.updates).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("no policy-update records")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    def load(trainable: bool):
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            quantization_config=quantization,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        return PeftModel.from_pretrained(base, args.adapter, is_trainable=trainable)

    policy = load(True)
    reference = load(False)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    class UpdateDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            user, assistant = row["messages"]
            prompt = tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            input_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=args.max_length)[
                "input_ids"
            ]
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids) :]
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "advantage": torch.tensor(float(row["group_advantage"]), dtype=torch.float32),
            }

    def collate(items):
        return {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in items], batch_first=True, padding_value=tokenizer.pad_token_id
            ),
            "labels": pad_sequence([item["labels"] for item in items], batch_first=True, padding_value=-100),
            "advantage": torch.stack([item["advantage"] for item in items]),
        }

    def token_logp(model, input_ids, labels):
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        labels = labels.to(model_device)
        attention = input_ids.ne(tokenizer.pad_token_id)
        # Only assistant tokens are scored; skip the prompt's logits. A full
        # [batch, seq, 248k-vocab] fp32 log_softmax ran GPU 0 out of memory.
        first_label = int(labels.ne(-100).float().argmax(-1).min())
        start = max(first_label - 1, 0)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention,
            logits_to_keep=input_ids.shape[1] - start,
        ).logits[:, :-1]
        targets = labels[:, start + 1 :]
        mask = targets.ne(-100)
        safe_targets = targets.masked_fill(~mask, 0)
        picked = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1).float()
        return picked - torch.logsumexp(logits.float(), dim=-1), mask

    def response_mean(values, mask):
        return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    class GroupRelativeTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            advantages = inputs.pop("advantage").to(model.device)
            current, mask = token_logp(model, inputs["input_ids"], inputs["labels"])
            with torch.no_grad():
                old, _ = token_logp(reference, inputs["input_ids"], inputs["labels"])
            old = old.to(current.device)
            mask = mask.to(current.device)
            if args.ratio_level == "sequence":
                current, old = response_mean(current, mask), response_mean(old, mask)
                mask = torch.ones_like(current, dtype=torch.bool)
                current, old, mask = current[:, None], old[:, None], mask[:, None]
            log_ratio = (current - old).clamp(-10, 10)
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1 - args.clip_range, 1 + args.clip_range)
            surrogate = torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
            per_token = -surrogate + args.reference_penalty * log_ratio.square()
            loss = response_mean(per_token, mask).mean()
            self._ratio_log.append(log_ratio.detach()[mask].float().cpu())
            return (loss, {"log_ratio": log_ratio}) if return_outputs else loss

        def log(self, logs, *log_args, **log_kwargs):
            # How far the policy has moved from the SFT reference since the last log
            # (per token, or per response with --ratio-level sequence).
            if self._ratio_log and "loss" in logs:
                values = torch.cat(self._ratio_log)
                logs["log_ratio_abs_mean"] = round(values.abs().mean().item(), 5)
                outside = (values > math.log(1 + args.clip_range)) | (values < math.log(1 - args.clip_range))
                logs["clip_fraction"] = round(outside.float().mean().item(), 4)
                self._ratio_log.clear()
            super().log(logs, *log_args, **log_kwargs)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = GroupRelativeTrainer(
        model=policy,
        args=training_args,
        train_dataset=UpdateDataset(),
        data_collator=collate,
    )
    trainer._ratio_log = []
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final = Path(args.output) / "final"
    policy.save_pretrained(final)
    tokenizer.save_pretrained(final)


if __name__ == "__main__":
    main()
