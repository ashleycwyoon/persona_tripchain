#!/usr/bin/env python3
"""Distribution-level comparison of the Ground Truth with two to five model populations.

Each `--population LABEL=ROLLOUTS` is one generated population (e.g. Pre-RL and
Post-RL rollouts of the same people).

Instead of per-person match/mismatch, every discrete decision (activity count,
activity types, sequence patterns, dwell times, timing, modes, location scope)
is compared as a population distribution, with Jensen-Shannon divergence (JSD,
base 2, 0 = identical, 1 = disjoint) against Ground Truth.

Modes: with per-trip main modes (ground truth / Student2 `trip_modes`) those are
compared per trip; otherwise the person-level `mode_preference` counts.

Text analysis (Student1 persona/reasoning1, Student2 reasoning2; skipped with
--skip-text):
- cosine similarity between stages (MiniLM sentence embeddings), with a
  shuffled-person baseline, and
- sentiment (3-class RoBERTa), compared with the teacher texts of the SFT
  validation split (the test split has no teacher persona/reasoning).

Within one figure, panels that show the same kind of value (same y label, or all
shares) use one y range, so bar heights compare across panels.

Writes numbered figures and summary.json to --output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tripchain.common import load_config, rows_by_id  # noqa: E402
from tripchain.paths import DATA_ROOT  # noqa: E402
from tripchain.trip_modes import mode_preference_from_trips  # noqa: E402

DAY = 1440
# Populations in drawing order: the Ground Truth, then the --population labels
# (set in main()). Model populations take MODEL_COLORS in the order given.
POPS: tuple[str, ...] = ("Ground Truth",)
MODEL_COLORS = ("#3b7dd8", "#e8833a", "#2f8a55", "#8e5bc0", "#c94f6d")
COLORS = {"Ground Truth": "#4d4d4d", "Teacher": "#4d4d4d"}
MODES = ["walk", "bus", "subway", "car_driver", "car_passenger", "taxi", "bicycle", "other_bus"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_REVISION = "refs/pr/43"  # safetensors conversion of the same weights


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsd(p: dict | np.ndarray, q: dict | np.ndarray, keys=None) -> float:
    if isinstance(p, dict):
        keys = keys or sorted(set(p) | set(q))
        p = np.array([p.get(k, 0) for k in keys], float)
        q = np.array([q.get(k, 0) for k in keys], float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# --------------------------------------------------------------------------- people
def gt_person(row: dict) -> dict:
    acts = row["activities"]
    timeline = [(a["start_time_min"], a["start_time_min"] + a["dwell_time_min"], a["activity_type"]) for a in acts]
    home_dong = acts[0]["location"].get("administrative_dong_code")
    home_gu = acts[0]["location"].get("gu")
    scopes = []
    for a in acts:
        if a["activity_type"] == "home":
            continue
        loc = a["location"]
        if loc.get("sido") != "Seoul":
            scopes.append("outside Seoul")
        elif loc.get("administrative_dong_code") == home_dong:
            scopes.append("same dong as home")
        elif loc.get("gu") == home_gu:
            scopes.append("same gu, other dong")
        else:
            scopes.append("other gu in Seoul")
    if row.get("trip_modes") is not None:
        modes = list(row["trip_modes"])
    else:
        modes = [m["mode"] for m in row.get("mode_preference", []) for _ in range(int(m.get("count", 1)))]
    return {
        "activity_sequence": row["activity_sequence"],
        "dwell": [(a["activity_type"], a["dwell_time_min"]) for a in acts],
        "executed_dwell_total": sum(a["dwell_time_min"] for a in acts),
        "timeline": timeline,
        "travel": [(timeline[i][1], timeline[i + 1][0]) for i in range(len(timeline) - 1)],
        "selected_modes": modes,
        "preferred_modes": modes,
        "top1_mode": (row.get("mode_preference") or [{}])[0].get("mode"),
        "location_scope": scopes,
    }


def model_person(row: dict) -> dict:
    trace = row["decision_trace"]
    s1, s2 = trace["student1"]["output"], trace["student2"]["output"]
    seq = s1["activity_sequence"]
    ldw = s2["location_dwell_by_activity"]
    dwell = [(seq[i] if i < len(seq) else "?", float(item.get("dwell_time_min") or 0)) for i, item in enumerate(ldw)]
    chain = (trace.get("environment") or {}).get("final_trip_chain")
    executed = None
    if chain:
        timeline = [(a["start_min"], a["end_min"], a["activity_type"]) for a in chain["activities"]]
        executed = sum(a["end_min"] - a["start_min"] for a in chain["activities"])
        travel = [(t["departure_min"], t["arrival_min"]) for t in chain.get("trips", [])]
        selected = [t["mode"] for t in chain.get("trips", [])]
    else:  # outside POI coverage / infeasible: no routing, lay dwell times back to back
        timeline, clock = [], 0.0
        for activity, minutes in dwell:
            timeline.append((clock, clock + minutes, activity))
            clock += minutes
        travel, selected = [], []
    home = ldw[0]["location"] if ldw else {}
    scopes = []
    for i, item in enumerate(ldw):
        if i < len(seq) and seq[i] == "home":
            continue
        loc = item.get("location") or {}
        if loc.get("scope") != "seoul":
            scopes.append("outside Seoul")
        elif loc.get("administrative_dong_name") == home.get("administrative_dong_name") and loc.get("gu") == home.get(
            "gu"
        ):
            scopes.append("same dong as home")
        elif loc.get("gu") == home.get("gu"):
            scopes.append("same gu, other dong")
        else:
            scopes.append("other gu in Seoul")
    if s2.get("trip_modes") is not None:
        preferred = list(s2["trip_modes"])
        prefs = mode_preference_from_trips(preferred)
    else:
        prefs = s2.get("mode_preference") or []
        preferred = [m["mode"] for m in prefs for _ in range(int(m.get("count", 1) or 1))]
    persona = s1.get("persona_bundle") or {}
    return {
        "activity_sequence": seq,
        "dwell": dwell,
        "executed_dwell_total": executed,
        "timeline": timeline,
        "travel": travel,
        "selected_modes": selected,
        "preferred_modes": preferred,
        "top1_mode": prefs[0]["mode"] if prefs else None,
        "location_scope": scopes,
        "texts": {
            "persona": " ".join(
                str(persona.get(k, "")) for k in ("professional_persona", "family_persona", "persona")
            ).strip(),
            "reasoning1": str(s1.get("reasoning1") or ""),
            "reasoning2": str(s2.get("reasoning2") or ""),
        },
        "rewards": row["rewards"]["components"],
        "status": row["status"],
    }


def teacher_texts(sft_dir: Path) -> list[dict]:
    """Teacher persona / reasoning texts of the SFT validation split (the Students' training targets)."""
    s1 = {
        r["person_id"]: json.loads(r["messages"][-1]["content"])
        for r in read_jsonl(sft_dir / "student1_residence" / "validation.jsonl")
    }
    s2 = {
        r["person_id"]: json.loads(r["messages"][-1]["content"])
        for r in read_jsonl(sft_dir / "student2_tripmode" / "validation.jsonl")
    }
    out = []
    for pid, a in s1.items():
        if pid not in s2:
            continue
        persona = a.get("persona_bundle") or {}
        out.append(
            {
                "persona": " ".join(
                    str(persona.get(k, "")) for k in ("professional_persona", "family_persona", "persona")
                ).strip(),
                "reasoning1": str(a.get("reasoning1") or ""),
                "reasoning2": str(s2[pid].get("reasoning2") or ""),
            }
        )
    return out


# --------------------------------------------------------------------------- figures
def unify_y(fig):
    """Give the panels of one figure the same y range when they show the same kind of value:
    the same y label, or all of them a share ("share of ..."). Other figures are left alone."""
    axes = [ax for ax in fig.axes if ax.has_data()]
    labels = {ax.get_ylabel() for ax in axes}
    if len(axes) < 2 or not (len(labels) == 1 or all(label.startswith("share of") for label in labels)):
        return
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)


def save(fig, path: Path):
    unify_y(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def occupancy(people: list[dict], categories: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(categories)}
    grid = np.zeros((len(categories), DAY))
    for person in people:
        state = np.full(DAY, index["unscheduled"])
        for start, end in person["travel"]:
            state[int(max(0, start)) : int(min(DAY, end))] = index["travel"]
        for i, (start, end, activity) in enumerate(person["timeline"]):
            if i == len(person["timeline"]) - 1 and activity == "home":
                end = DAY  # the final home stay is open-ended (model day_end_min < 1440)
            # A label outside the KTDB types (a rare invalid Student1 output) counts as "other".
            state[int(max(0, start)) : int(min(DAY, end))] = index.get(
                activity, index.get("other", index["other types"])
            )
        grid[state, np.arange(DAY)] += 1
    return grid / max(1, len(people))


def fig_24h(pops: dict, out: Path, summary: dict):
    counts = Counter(a for p in pops["Ground Truth"] for _, _, a in p["timeline"])
    acts = ["home"] + [a for a, _ in counts.most_common() if a != "home"][:9]
    categories = acts + ["other types", "travel", "unscheduled"]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(acts) + 1)] + ["#222222", "#dddddd"]
    grids = {name: occupancy(people, categories) for name, people in pops.items()}
    # "other types" collects activity types beyond the shown ones; drop it when no population has any
    # (with all KTDB types shown, invalid labels go to "other" instead).
    extra = categories.index("other types")
    if not any(grid[extra].any() for grid in grids.values()):
        categories.pop(extra)
        colors.pop(extra)
        grids = {name: np.delete(grid, extra, axis=0) for name, grid in grids.items()}
    hours = np.arange(DAY) / 60
    fig, axes = plt.subplots(1, len(POPS), figsize=(6 * len(POPS), 5), sharey=True)
    for ax, name in zip(axes, POPS):
        # "unscheduled" (minutes outside every activity and trip) is ~0 in every population
        # (GT 0, models <= 0.1% at any minute), so it is left out of the stack and legend.
        shown = [i for i, c in enumerate(categories) if c != "unscheduled"]
        ax.stackplot(
            hours,
            grids[name][shown],
            labels=[categories[i] for i in shown],
            colors=[colors[i] for i in shown],
            linewidth=0,
        )
        div = np.mean([jsd(grids[name][:, t] + 1e-12, grids["Ground Truth"][:, t] + 1e-12) for t in range(DAY)])
        summary.setdefault("jsd_vs_gt", {}).setdefault("24h_activity_profile", {})[name] = div
        ax.set_title(name if name == "Ground Truth" else f"{name}  (mean JSD vs GT = {div:.3f})")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("hour of day (from day origin)")
        ax.set_ylabel("share of people")
        ax.tick_params(labelleft=True)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle("24-hour activity distribution (who is doing what, minute by minute)")
    save(fig, out / "01_activity_24h.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    home, travel = categories.index("home"), categories.index("travel")
    for name in POPS:
        g = grids[name]
        axes[0].plot(hours, 1 - g[home] - g[categories.index("unscheduled")], color=COLORS[name], label=name, lw=2)
        axes[1].plot(hours, g[travel], color=COLORS[name], label=name, lw=2)
    axes[0].set_title("Out-of-home share (non-home activity or travel)")
    axes[1].set_title("Travelling share (GT: gap between activities; model: routed trips)")
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("hour of day")
        ax.legend()
        ax.set_ylabel("share of people")
    save(fig, out / "02_out_of_home_and_travel.png")

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)
    panels = acts[1:] + (["other types"] if "other types" in categories else [])
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax, activity in zip(axes.flat, panels):
        i = categories.index(activity)
        for name in POPS:
            ax.plot(hours, grids[name][i], color=COLORS[name], label=name, lw=1.8)
        ax.set_title(activity)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        # sharex hides tick labels above the hidden bottom-row slots, so label every panel
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("hour of day (from day origin)")
        ax.set_ylabel("share of people")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Share of people in each non-home activity by hour (same y scale in all panels)")
    save(fig, out / "03_activity_by_hour_per_type.png")


def grouped_bars(ax, dists: dict[str, Counter], keys: list[str], normalize=True, title=""):
    width = 0.8 / len(dists)
    for i, (name, dist) in enumerate(dists.items()):
        total = sum(dist.values()) or 1
        values = [dist.get(k, 0) / total if normalize else dist.get(k, 0) for k in keys]
        ax.bar(np.arange(len(keys)) + (i - (len(dists) - 1) / 2) * width, values, width, label=name, color=COLORS[name])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_title(title)


def with_jsd(name: str, dists: dict, key: str, summary: dict, keys=None) -> str:
    if name == "Ground Truth":
        return name
    value = jsd(dists[name], dists["Ground Truth"], keys)
    summary.setdefault("jsd_vs_gt", {}).setdefault(key, {})[name] = value
    return f"{name} (JSD vs GT {value:.3f})"


def fig_modes(pops: dict, out: Path, summary: dict):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    panels = [
        (
            "selected_modes",
            "Trip mode: GT vs mode of the realized route (one route per trip)",
            "mode_selected_legs",
            "share of trips",
        ),
        ("preferred_modes", "Student2 mode output per trip (trip_modes) vs GT", "mode_policy_output", "share of trips"),
        ("top1_mode", "Top-1 preferred mode per person", "mode_top1", "share of people"),
    ]
    for ax, (field, title, key, ylabel) in zip(axes, panels):
        dists = {}
        for name, people in pops.items():
            values = [p[field] for p in people] if field == "top1_mode" else [m for p in people for m in p[field]]
            dists[name] = Counter(v for v in values if v)
        labelled = {with_jsd(n, dists, key, summary, MODES): d for n, d in dists.items()}
        grouped_bars(ax, dict(zip(POPS, labelled.values())), MODES, title=title)
        ax.legend(list(labelled), fontsize=9)
        ax.set_xlabel("travel mode")
        ax.set_ylabel(ylabel)
    save(fig, out / "04_mode_distribution.png")


def fig_activity(pops: dict, out: Path, summary: dict):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    lengths = {n: Counter(min(len(p["activity_sequence"]), 9) for p in ps) for n, ps in pops.items()}
    keys = list(range(1, 10))  # 1 = home-only day
    labels = [with_jsd(n, lengths, "activity_count", summary, keys) for n in POPS]
    grouped_bars(axes[0], lengths, keys, title="Activities per person (9 = 9+)")
    axes[0].set_xticklabels([str(k) if k < 9 else "9+" for k in keys], rotation=0)
    axes[0].legend(labels)
    axes[0].set_xlabel("activities per person (count, incl. home)")
    axes[0].set_ylabel("share of people")
    types = {n: Counter(a for p in ps for a in p["activity_sequence"] if a != "home") for n, ps in pops.items()}
    keys = [a for a, _ in types["Ground Truth"].most_common(12)]
    labels = [with_jsd(n, types, "non_home_activity_types", summary) for n in POPS]
    grouped_bars(axes[1], types, keys, title="Non-home activity type share (all activities)")
    axes[1].legend(labels)
    axes[1].set_xlabel("non-home activity type")
    axes[1].set_ylabel("share of non-home activities")
    save(fig, out / "05_activity_count_and_types.png")

    patterns = {n: Counter(" > ".join(p["activity_sequence"]) for p in ps) for n, ps in pops.items()}
    top = [s for s, _ in patterns["Ground Truth"].most_common(15)]
    # A last row with everything outside the GT top 15, so each population's bars sum to 1.
    other = "all other sequences"
    fig, ax = plt.subplots(figsize=(12, 7.5))
    height = 0.8 / len(POPS)
    for i, name in enumerate(POPS):
        total = sum(patterns[name].values())
        shares = [patterns[name][s] / total for s in top]
        shares.append(1 - sum(shares))
        ax.barh(
            np.arange(len(top) + 1) + (i - (len(POPS) - 1) / 2) * height,
            shares,
            height,
            color=COLORS[name],
            label=with_jsd(name, patterns, "activity_sequence_patterns", summary),
        )
    ax.axhline(len(top) - 0.5, color="#999999", lw=0.8, ls=":")
    ax.set_yticks(range(len(top) + 1))
    ax.set_yticklabels(top + [other], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of people")
    ax.set_ylabel("daily activity sequence")
    ax.legend()
    ax.set_title("Top-15 Ground Truth daily sequences: frequency in each population")
    save(fig, out / "06_sequence_patterns.png")

    gt_keys = set(patterns["Ground Truth"])
    summary["sequence_pattern_stats"] = {
        n: {
            "unique_patterns": len(patterns[n]),
            "share_of_people_with_pattern_seen_in_gt": sum(c for s, c in patterns[n].items() if s in gt_keys)
            / sum(patterns[n].values()),
        }
        for n in POPS
    }


def hist_bars(ax, values: dict[str, list[float]], edges, labels: dict[str, str]):
    """Grouped bars of the share of values per bin (one bar per population), instead of step lines."""
    n = len(values)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.85 / n
    for i, (name, vals) in enumerate(values.items()):
        hist = np.histogram(np.clip(vals, edges[0], edges[-1] - 1e-6), bins=edges)[0].astype(float)
        ax.bar(
            centers + (i - (n - 1) / 2) * width,
            hist / max(1.0, hist.sum()),
            width * 0.92,
            color=COLORS[name],
            label=labels[name],
        )


def fig_dwell(pops: dict, out: Path, summary: dict):
    counts = Counter(a for p in pops["Ground Truth"] for a, _ in p["dwell"])
    types = [a for a, _ in counts.most_common(8)]
    bins = np.arange(0, 1441, 30)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, activity in zip(axes.flat, types):
        values = {n: [d for p in pops[n] for a, d in p["dwell"] if a == activity] for n in POPS}
        hists = {n: np.histogram(np.clip(values[n], 0, 1439), bins=bins)[0] for n in POPS}
        divs = {n: jsd(hists[n].astype(float), hists["Ground Truth"].astype(float)) for n in POPS[1:]}
        summary.setdefault("jsd_vs_gt", {})[f"dwell_{activity}"] = divs
        labels = {}
        for name in POPS:
            stats = [f"median {np.median(values[name]):.0f}"] if values[name] else []
            if name in divs:
                stats.append(f"JSD {divs[name]:.3f}")
            labels[name] = f"{name} ({', '.join(stats)})" if stats else name
        hist_bars(ax, {n: [v / 60 for v in values[n]] for n in POPS}, np.arange(0, 13.01, 1), labels)
        ax.set_xticks(range(0, 14, 2))
        ax.set_xticklabels([str(h) if h < 12 else "12+" for h in range(0, 14, 2)][:7])
        ax.set_title(activity, fontsize=11)
        ax.set_xlabel("dwell time (hours, 1-hour bins)")
        ax.set_ylabel("share of activities")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Dwell-time distribution per activity type (bars: 1-hour bins, last = 12 h+; JSD vs Ground Truth on 30-min bins)"
    )
    save(fig, out / "07_dwell_by_activity.png")

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    # Executed dwell (final_trip_chain): the environment cuts the open-ended final home stay at the day end,
    # so a planned Student2 total above 1440 shows up here as a shorter final home stay, not an overflow.
    # People without a routed chain (outside POI coverage / infeasible) are left out.
    planned = {n: [sum(d for _, d in p["dwell"]) for p in ps] for n, ps in pops.items()}
    totals = {
        n: [p["executed_dwell_total"] for p in ps if p["executed_dwell_total"] is not None] for n, ps in pops.items()
    }
    labels = {
        name: f"{name}: >24 h {sum(t > 1440 for t in totals[name])} (planned {sum(t > 1440 for t in planned[name])})"
        for name in POPS
    }
    # 1-hour bins (a, b]: a day of exactly 24 h lands in the 23-24 h bar, left of the 24 h line.
    hist_bars(axes[0], {n: [t / 60 - 1e-6 for t in totals[n]] for n in POPS}, np.arange(12, 30.01, 1), labels)
    axes[0].axvline(24, color="red", ls="--", lw=1, label="24 h")
    axes[0].set_title("Executed dwell per person (final trip chain)")
    axes[0].set_xlabel("total executed dwell per person (hours; first bin includes <12 h)")
    axes[0].set_ylabel("share of people")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_xlim(12, 26)
    for ax, (title, pick) in zip(
        axes[1:],
        [
            ("First departure from home", lambda p: p["timeline"][0][1] if len(p["timeline"]) > 1 else None),
            ("Final arrival at home", lambda p: p["timeline"][-1][0] if len(p["timeline"]) > 1 else None),
        ],
    ):
        values = {n: [v / 60 for v in (pick(p) for p in pops[n]) if v is not None] for n in POPS}
        hists = {n: np.histogram(values[n], bins=np.arange(0, 24.5, 0.5))[0] for n in POPS}
        divs = {n: jsd(hists[n].astype(float), hists["Ground Truth"].astype(float)) for n in POPS[1:]}
        summary.setdefault("jsd_vs_gt", {})[title.lower().replace(" ", "_")] = divs
        hist_bars(
            ax, values, np.arange(0, 24.01, 1), {n: f"{n} (JSD vs GT {divs[n]:.3f})" if n in divs else n for n in POPS}
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("hour of day (from day origin, 1-hour bins)")
        ax.set_ylabel("share of people")
        ax.set_xlim(0, 24)
        ax.legend(fontsize=8)
    save(fig, out / "08_dwell_total_and_timing.png")


def fig_location(pops: dict, out: Path, summary: dict):
    keys = ["same dong as home", "same gu, other dong", "other gu in Seoul", "outside Seoul"]
    dists = {n: Counter(s for p in ps for s in p["location_scope"]) for n, ps in pops.items()}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped_bars(ax, dists, keys, title="Student2: where non-home activities are placed relative to home")
    ax.legend([with_jsd(n, dists, "non_home_location_scope", summary, keys) for n in POPS])
    ax.set_ylabel("share of non-home activities")
    ax.set_xlabel("location of non-home activity relative to home")
    save(fig, out / "09_location_scope.png")


def fig_rewards(pops: dict, out: Path):
    mode = (
        "trip_mode" if all("trip_mode" in p["rewards"] for name in POPS[1:] for p in pops[name]) else "mode_preference"
    )
    components = ["activity", "location", "dwell", mode, "feasibility"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    models = POPS[1:]
    width = 0.7 / len(models)
    for i, name in enumerate(models):
        data = [[p["rewards"][c] for p in pops[name]] for c in components]
        parts = ax.violinplot(
            data,
            positions=np.arange(len(components)) + (i - (len(models) - 1) / 2) * width,
            widths=width * 0.95,
            showmeans=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[name])
            body.set_alpha(0.55)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(COLORS[name])
        ax.plot([], [], color=COLORS[name], lw=6, label=name)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components)
    ax.legend()
    ax.set_xlabel("reward component")
    ax.set_ylabel("per-person reward (unitless)")
    ax.set_title("Per-person reward component distributions (lines = means)")
    save(fig, out / "10_reward_components.png")


# --------------------------------------------------------------------------- text
def embed(texts: list[str], device: str, cache: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, cache_dir=cache)
    model = AutoModel.from_pretrained(EMBED_MODEL, cache_dir=cache).to(device).eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), 128):
            batch = tokenizer(
                texts[start : start + 128], padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(vectors)


def sentiment(texts: list[str], device: str, cache: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, cache_dir=cache, revision=SENTIMENT_REVISION)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL, cache_dir=cache, revision=SENTIMENT_REVISION, use_safetensors=True
        )
        .to(device)
        .eval()
    )
    order = [model.config.label2id[k] for k in ("negative", "neutral", "positive")]
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            probs.append(model(**batch).logits.softmax(-1)[:, order].cpu().numpy())
    return np.concatenate(probs)


def fig_text(groups: dict[str, list[dict]], out: Path, summary: dict, device: str, cache: str, seed: int):
    fields = ["persona", "reasoning1", "reasoning2"]
    pairs = [("persona", "reasoning1"), ("reasoning1", "reasoning2"), ("persona", "reasoning2")]
    embeddings, sentiments = {}, {}
    for name, rows in groups.items():
        for field in fields:
            texts = [r[field] or " " for r in rows]
            embeddings[name, field] = embed(texts, device, cache)
            sentiments[name, field] = sentiment(texts, device, cache)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.8), sharey=True)
    bins = np.linspace(0, 1, 51)
    for ax, (a, b) in zip(axes, pairs):
        for name in groups:
            cos = (embeddings[name, a] * embeddings[name, b]).sum(1)
            if name == "Teacher":  # the reference: a filled grey area under the model lines
                ax.hist(
                    cos,
                    bins=bins,
                    histtype="stepfilled",
                    alpha=0.35,
                    density=True,
                    color="#888888",
                    label=f"{name} (mean {cos.mean():.3f})",
                )
            else:
                ax.hist(
                    cos,
                    bins=bins,
                    histtype="step",
                    lw=1.8,
                    density=True,
                    color=COLORS[name],
                    label=f"{name} (mean {cos.mean():.3f})",
                )
            shuffled = (embeddings[name, a] * embeddings[name, b][rng.permutation(len(cos))]).sum(1)
            summary.setdefault("text_cosine", {}).setdefault(f"{a}~{b}", {})[name] = {
                "mean": float(cos.mean()),
                "median": float(np.median(cos)),
                "shuffled_person_mean": float(shuffled.mean()),
            }
            if name == POPS[-1]:
                ax.hist(
                    shuffled,
                    bins=bins,
                    histtype="step",
                    lw=1.0,
                    ls=":",
                    density=True,
                    color="#666666",
                    label=f"different-person pairs (mean {shuffled.mean():.3f})",
                )
        ax.set_title(f"cos({a}, {b})")
        ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=8)
        ax.set_ylabel("density")
        ax.tick_params(labelleft=True)
    fig.suptitle("Cross-stage text consistency: same-person cosine similarity (MiniLM embeddings)")
    save(fig, out / "11_text_cosine.png")

    labels = ["negative", "neutral", "positive"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    names = list(groups)
    xpos, ticks = [], []
    for f_index, field in enumerate(fields):
        for g_index, name in enumerate(names):
            x = f_index * (len(names) + 1) + g_index
            share = np.bincount(sentiments[name, field].argmax(1), minlength=3) / len(sentiments[name, field])
            bottom = 0
            for k, label in enumerate(labels):
                axes[0].bar(
                    x,
                    share[k],
                    bottom=bottom,
                    color=["#d9534f", "#bbbbbb", "#5cb85c"][k],
                    edgecolor=COLORS[name],
                    linewidth=1.5,
                    label=label if x == 0 else None,
                )
                bottom += share[k]
            xpos.append(x)
            ticks.append(f"{field}\n{name}")
            summary.setdefault("text_sentiment", {}).setdefault(field, {})[name] = {
                **{f"share_{lab}": float(share[k]) for k, lab in enumerate(labels)},
                "mean_score_pos_minus_neg": float(
                    (sentiments[name, field][:, 2] - sentiments[name, field][:, 0]).mean()
                ),
            }
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(ticks, fontsize=7, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_title("Sentiment label share")
    axes[0].set_xlabel("text field / population")
    axes[0].set_ylabel("share of texts")
    data, positions, colors = [], [], []
    for f_index, field in enumerate(fields):
        for g_index, name in enumerate(names):
            data.append(sentiments[name, field][:, 2] - sentiments[name, field][:, 0])
            positions.append(f_index * (len(names) + 1) + g_index)
            colors.append(COLORS[name])
    parts = axes[1].violinplot(data, positions=positions, widths=0.85, showmeans=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(colors)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(ticks, fontsize=7, rotation=45, ha="right")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("text field / population")
    axes[1].set_ylabel("P(positive) - P(negative)")
    axes[1].set_title("Sentiment score P(positive) - P(negative)")
    fig.suptitle("Sentiment of persona / reasoning texts (teacher = SFT validation split, different people)")
    save(fig, out / "12_text_sentiment.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="supplies the ground truth of the compared people")
    parser.add_argument(
        "--population",
        action="append",
        required=True,
        metavar="LABEL=ROLLOUTS",
        help="a model population; give two to five",
    )
    parser.add_argument("--output", required=True, help="directory for the figures and summary.json")
    parser.add_argument(
        "--sft-dir",
        default=str(DATA_ROOT / "processed_v2/sft"),
        help="SFT data with the teacher texts (used unless --skip-text)",
    )
    parser.add_argument("--device", default="cuda:0", help="device of the text models (used unless --skip-text)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()
    if not 2 <= len(args.population) <= len(MODEL_COLORS):
        parser.error(f"--population must be given two to {len(MODEL_COLORS)} times")

    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    global POPS
    sources = [item.split("=", 1) for item in args.population]
    sources = [(label, Path(path)) for label, path in sources]
    POPS = ("Ground Truth", *(label for label, _ in sources))
    COLORS.update(zip(POPS[1:], MODEL_COLORS))
    rows = {label: [r for r in read_jsonl(path) if r.get("decision_trace")] for label, path in sources}
    targets = rows_by_id(cfg["paths"]["ground_truth_reference"])
    ids = sorted({str(r["person_id"]) for population in rows.values() for r in population})
    pops = {
        "Ground Truth": [gt_person(targets[pid]) for pid in ids if pid in targets],
        **{label: [model_person(r) for r in population] for label, population in rows.items()},
    }
    summary = {"people": {n: len(p) for n, p in pops.items()}}
    random.seed(args.seed)
    fig_24h(pops, out, summary)
    fig_modes(pops, out, summary)
    fig_activity(pops, out, summary)
    fig_dwell(pops, out, summary)
    fig_location(pops, out, summary)
    fig_rewards(pops, out)
    if not args.skip_text:
        teacher = teacher_texts(ROOT / args.sft_dir)
        summary["people"]["Teacher"] = len(teacher)
        groups = {"Teacher": teacher, **{name: [p["texts"] for p in pops[name]] for name in POPS[1:]}}
        fig_text(groups, out, summary, args.device, cfg["paths"].get("model_cache"), args.seed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"figures + summary.json -> {out}")


if __name__ == "__main__":
    main()
