#!/usr/bin/env python3
"""Run KTDB two-stage persona/reasoning teacher inference on a local GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tripchain.paths import HF_HUB_CACHE  # noqa: E402
from tripchain.ktdb.ktdb_two_stage_teacher import (  # noqa: E402
    DEFAULT_ACTIVITY,
    DEFAULT_DEMOGRAPHICS,
    DEFAULT_SPATIAL,
    PROMPT_SCHEMA_VERSION,
    PROMPT_SOURCE,
    SYSTEM_PROMPT,
    build_user_prompt,
    load_teacher_records,
    parse_teacher_output,
    validate_teacher_content,
)


def load_model(model_name: str, download_dir: Path, max_model_len: int, gpu_memory_utilization: float):
    """Load the 32B AWQ teacher through vLLM's Marlin kernel."""

    # vLLM 0.7.x still reads this tokenizer property, which was removed in
    # transformers 5.x.  Keep the compatibility shim local to this entrypoint
    # so the shared Python environment does not need to be downgraded.
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda tokenizer: list(tokenizer.all_special_tokens)
        )

    from vllm import LLM

    model = LLM(
        model=model_name,
        quantization="awq_marlin",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir=str(download_dir),
        trust_remote_code=False,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    return model, model.get_tokenizer()


def ascii_allowed_token_ids(tokenizer) -> list[int]:
    """Return tokens whose standalone decoded form contains ASCII only.

    Correction attempts use this hard decoding constraint. Prompt instructions
    alone occasionally fail for Qwen on very short/home-only activity chains.
    """
    allowed = []
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded and all(ord(character) < 128 for character in decoded):
            allowed.append(token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in allowed:
        allowed.append(tokenizer.eos_token_id)
    return allowed


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    allowed_token_ids: list[int] | None = None,
) -> list[str]:
    from vllm import SamplingParams

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.9 if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
        allowed_token_ids=allowed_token_ids,
    )
    outputs = model.generate(rendered, sampling, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


def generate_batch_safe(tokenizer, model, prompts, max_new_tokens, temperature, allowed_token_ids=None):
    try:
        return generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens,
            temperature,
            allowed_token_ids=allowed_token_ids,
        )
    except RuntimeError as exc:
        if len(prompts) == 1 or "out of memory" not in str(exc).lower():
            raise
        middle = len(prompts) // 2
        return generate_batch_safe(
            tokenizer,
            model,
            prompts[:middle],
            max_new_tokens,
            temperature,
            allowed_token_ids,
        ) + generate_batch_safe(
            tokenizer,
            model,
            prompts[middle:],
            max_new_tokens,
            temperature,
            allowed_token_ids,
        )


def correction_prompt(original_prompt: str, violation: str) -> str:
    return (
        original_prompt
        + "\n\nIMPORTANT: the previous attempt violated the required format or content rules. "
        + f"The validator reported: {violation}. "
        + "Rewrite the answer as exactly six XML blocks named professional_persona, "
        + "family_persona, persona, hobbies_and_interests_list, reasoning1, and "
        + "reasoning2, following every system instruction. Both reasoning blocks must "
        + "be flowing prose that connects the persona bundle to the observed day. "
        + "Use English ASCII text only: do not emit Hangul, Kana, Hanzi, or any other "
        + "non-English translation. Close every XML tag and emit each required block "
        + "exactly once in the required order. "
        + "Do not add any other text."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--download-dir", type=Path, default=HF_HUB_CACHE)
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=0, help="deterministic random sample; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", default="0/1", help="positional shard i/N")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-temperature", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.temperature:
        raise SystemExit("--temperature must be >= 0")
    shard_index, shard_count = (int(value) for value in args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit("--shard must satisfy 0 <= i < N")

    completion_path = args.out.with_name(args.out.name + ".complete.json")
    completion_path.unlink(missing_ok=True)

    records = load_teacher_records(args.demographics, args.activity, args.spatial)
    if args.n and args.n < len(records):
        import numpy as np

        rng = np.random.default_rng(args.seed)
        selected = sorted(rng.choice(len(records), size=args.n, replace=False).tolist())
        records = [records[index] for index in selected]
    records = records[shard_index::shard_count]
    shard_total = len(records)

    done_ids: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            try:
                record = json.loads(line)
                if not record.get("error"):
                    done_ids.add(str(record["person_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    records = [record for record in records if record["person_id"] not in done_ids]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(
        f"[teacher] model={args.model} shard={args.shard} gpu={args.gpu} "
        f"records={len(records)} skipped={len(done_ids)}",
        flush=True,
    )
    args.download_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(
        args.model,
        args.download_dir,
        args.max_model_len,
        args.gpu_memory_utilization,
    )
    # Built once per worker. Retries then cannot emit Hangul/Kana/Hanzi even if
    # the multilingual model ignores the natural-language instruction.
    retry_allowed_token_ids = ascii_allowed_token_ids(tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    error_path = args.out.with_name(args.out.name + ".errors.jsonl")

    written = 0
    errors = 0
    successful_ids = set(done_ids)
    mode = "a" if args.resume else "w"
    with (
        args.out.open(mode, encoding="utf-8") as handle,
        error_path.open(mode, encoding="utf-8") as error_handle,
    ):
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = [build_user_prompt(record) for record in batch]
            try:
                raw_outputs = generate_batch_safe(tokenizer, model, prompts, args.max_new_tokens, args.temperature)
            except Exception as exc:
                raw_outputs = [""] * len(batch)
                batch_error = f"generation_error: {type(exc).__name__}: {exc}"
            else:
                batch_error = ""

            for record, prompt, raw in zip(batch, prompts, raw_outputs):
                parsed = None
                error = batch_error
                if not error:
                    try:
                        parsed = parse_teacher_output(raw)
                        validate_teacher_content(record, parsed)
                    except Exception as first_exc:
                        # Never retain the first parsed object after semantic
                        # validation failed; only the corrected object may pass.
                        parsed = None
                        problems = [f"{type(first_exc).__name__}: {first_exc}"]
                        for retry_index in range(args.max_retries):
                            try:
                                # Increase diversity after repeated validation
                                # failures instead of repeating the same low-
                                # temperature completion ten times.
                                retry_temperature = min(
                                    1.0,
                                    max(args.retry_temperature, 0.2) + 0.1 * retry_index,
                                )
                                retry_raw = generate_batch_safe(
                                    tokenizer,
                                    model,
                                    [correction_prompt(prompt, problems[-1])],
                                    args.max_new_tokens,
                                    retry_temperature,
                                    retry_allowed_token_ids,
                                )[0]
                                corrected = parse_teacher_output(retry_raw)
                                validate_teacher_content(record, corrected)
                                parsed = corrected
                                raw = retry_raw
                                break
                            except Exception as retry_exc:
                                problems.append(f"{type(retry_exc).__name__}: {retry_exc}")
                        if parsed is None:
                            error = "validation_error: " + "; retry: ".join(problems)

                if parsed is not None:
                    output = {
                        "person_id": record["person_id"],
                        **parsed.model_dump(),
                    }
                    written += 1
                    successful_ids.add(str(record["person_id"]))
                    handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                else:
                    output = {
                        "person_id": record["person_id"],
                        "error": error,
                        "raw_output": raw[:4000],
                    }
                    errors += 1
                    error_handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
                    error_handle.flush()
            print(
                f"  [{min(start + len(batch), len(records))}/{len(records)}] ok={written} errors={errors}", flush=True
            )

    # On resume, old failures may now be successful. Keep only the latest error
    # for IDs that remain unresolved so the error file reflects final state.
    unresolved_errors: dict[str, dict[str, Any]] = {}
    if error_path.exists():
        for line in error_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                person_id = str(row["person_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if person_id not in successful_ids:
                unresolved_errors[person_id] = row
    error_tmp = error_path.with_name(error_path.name + ".tmp")
    with error_tmp.open("w", encoding="utf-8") as handle:
        for row in unresolved_errors.values():
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    error_tmp.replace(error_path)

    final_errors = len(unresolved_errors)
    print(
        f"[done] new_ok={written} unresolved={final_errors} output={args.out} error_output={error_path}",
        flush=True,
    )
    completion = {
        "shard": args.shard,
        "prompt_source": PROMPT_SOURCE,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "expected_records": shard_total,
        "successful_records": len(successful_ids),
        "unresolved_records": shard_total - len(successful_ids),
        "output": str(args.out),
        "error_output": str(error_path),
    }
    completion_tmp = completion_path.with_name(completion_path.name + ".tmp")
    completion_tmp.write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completion_tmp.replace(completion_path)


if __name__ == "__main__":
    main()
