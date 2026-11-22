"""Both Student models in one process, for the refinement loop (scripts/refine_infeasible.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import clean_json, resolve
from .inference import (
    build_student2_user,
    generate_batch,
    load_location_whitelist,
    load_model,
    validate_student1,
    validate_student2,
)
from .residence import validate_residence_demographics


class StudentPolicyPair:
    """Residence-conditioned Student inference with bounded repair prompts."""

    def __init__(self, config: dict[str, Any]):
        paths, inference = config["paths"], config["inference"]
        self.config = inference
        self.student1, self.tokenizer1 = load_model(
            resolve(paths["student1_adapter"]),
            inference["base_model"],
            Path(paths["model_cache"]),
        )
        self.student2, self.tokenizer2 = load_model(
            resolve(paths["student2_adapter"]),
            inference["base_model"],
            Path(paths["model_cache"]),
        )
        self.whitelist = load_location_whitelist(paths["location_whitelist"])

    def _generate(self, model, tokenizer, user, max_tokens, validator):
        """One validated answer: `temperature` first, then up to `max_retries` at `retry_temperature`."""
        last_error: Exception | None = None
        for attempt in range(int(self.config["max_retries"]) + 1):
            try:
                raw = generate_batch(
                    model,
                    tokenizer,
                    [user],
                    max_tokens,
                    float(self.config["temperature"] if attempt == 0 else self.config["retry_temperature"]),
                    float(self.config["top_p"]),
                )[0]
                value = clean_json(raw)
                validator(value)
                return value
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error))

    @staticmethod
    def _feedback(environment: dict[str, Any], level: str) -> dict[str, Any]:
        """What went wrong in the environment, appended to the prompt of a regeneration."""
        return {
            "level": level,
            "failure_reason": environment.get("failure_reason"),
            "violations": environment.get("violations"),
            "instruction": (
                "Regenerate the required JSON using the original output schema. "
                "Reduce the listed constraint violations while preserving plausible essential activities."
            ),
        }

    def generate_student1(
        self, demographics: dict[str, Any], environment: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        validate_residence_demographics(demographics)
        # The outer schema is unchanged; demographics now includes residence.
        user: dict[str, Any] = {"demographics": demographics}
        if environment is not None:
            user["refinement_feedback"] = self._feedback(environment, "lv3_student1")
        value = self._generate(
            self.student1,
            self.tokenizer1,
            user,
            int(self.config["student1_max_new_tokens"]),
            validate_student1,
        )
        return {**value, "student1_input": user}

    def generate_student2(self, record: dict[str, Any], environment: dict[str, Any] | None = None) -> dict[str, Any]:
        validate_residence_demographics(record.get("demographics") or {})
        # The outer schema is unchanged; demographics now includes residence.
        user: dict[str, Any] = build_student2_user(record)
        if environment is not None:
            user["refinement_feedback"] = self._feedback(environment, "lv2_student2")
        validator = lambda value: validate_student2(value, record["activity_sequence"], self.whitelist)
        value = self._generate(
            self.student2,
            self.tokenizer2,
            user,
            int(self.config["student2_max_new_tokens"]),
            validator,
        )
        return {**value, "student2_input": user}

    def regenerate_student2(self, record: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
        """Refinement Lv.2: a new Student2 answer for the same Student1 output."""
        value = self.generate_student2(record, environment)
        return {**record, **value}

    def regenerate_student1(self, record: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
        """Refinement Lv.3: a new Student1 answer (Student2 is regenerated right after)."""
        value = self.generate_student1(record["demographics"], environment)
        return {**record, **value}
