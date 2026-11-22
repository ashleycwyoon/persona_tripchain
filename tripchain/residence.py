"""Observed residence (sido, sigungu, administrative dong) in the Student demographics."""

from __future__ import annotations

from typing import Any


RESIDENCE_KEYS = {"sido", "sigungu", "administrative_dong_name"}


def residence_from_location(location: dict[str, Any]) -> dict[str, str]:
    """Model-facing residence of a KTDB location record."""
    if not isinstance(location, dict):
        raise ValueError("residence location must be an object")
    residence = {
        "sido": str(location.get("sido") or "").strip(),
        "sigungu": str(location.get("gu") or location.get("sigungu") or "").strip(),
        "administrative_dong_name": str(location.get("administrative_dong_name") or "").strip(),
    }
    missing = [key for key, value in residence.items() if not value]
    if missing:
        raise ValueError(f"residence location is missing fields: {missing}")
    return residence


def residence_from_activities(activities: list[dict[str, Any]]) -> dict[str, str]:
    """Return the first observed home location in a model-facing schema."""
    home = next(
        (item for item in activities if str(item.get("activity_type", "")).casefold() == "home"),
        None,
    )
    if home is None:
        raise ValueError("residence requires at least one home activity")
    return residence_from_location(home.get("location") or {})


def augment_demographics(
    demographics: dict[str, Any], residence_or_activities: dict[str, Any] | list[dict[str, Any]]
) -> dict[str, Any]:
    """Copy of `demographics` with a `residence` taken from a location or an activity list."""
    if not isinstance(demographics, dict):
        raise ValueError("demographics must be an object")
    value = dict(demographics)
    value["residence"] = (
        residence_from_activities(residence_or_activities)
        if isinstance(residence_or_activities, list)
        else residence_from_location(residence_or_activities)
    )
    return value


def validate_residence_demographics(demographics: dict[str, Any]) -> None:
    """Both Students are residence-conditioned; reject demographics without a full residence."""
    residence = demographics.get("residence")
    if not isinstance(residence, dict) or set(residence) != RESIDENCE_KEYS:
        raise ValueError("demographics.residence must contain sido, sigungu, and administrative_dong_name")
    if not all(isinstance(value, str) and value.strip() for value in residence.values()):
        raise ValueError("demographics.residence values must be non-empty strings")
