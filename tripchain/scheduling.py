"""Deterministic activity scheduling: an activity starts when the traveler arrives."""

from __future__ import annotations


def schedule_activity(activity_type: str, arrival_min: float, dwell_min: float) -> dict[str, float]:
    """Place one activity without stretching its predicted dwell.

    The activity starts on arrival and lasts its predicted dwell. No residual
    time is injected into home or travel, and start times are not constrained
    to time windows; only the day length bounds the schedule.
    """
    start = float(arrival_min)
    end = start + max(0.0, float(dwell_min))
    return {
        "arrival_min": start,
        "start_min": start,
        "end_min": end,
        "waiting_min": 0.0,
        "day_overrun_min": max(0.0, end - 1440.0),
    }
