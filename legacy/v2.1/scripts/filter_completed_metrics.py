#!/usr/bin/env python3
"""Re-aggregate cached per-design metrics over completed v1 designs only."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
from collections import defaultdict


def completed_indices(document: dict) -> set[int]:
    completed_ids = {
        event["design_id"] for event in document["events"]
        if event["kind"] == "completed"
    }
    keep: set[int] = set()
    proposal_index = -1
    for event in document["events"]:
        if event["kind"] == "proposed":
            proposal_index += 1
        elif event["kind"] == "accepted" and event["design_id"] in completed_ids:
            keep.add(proposal_index)
    return keep


def aggregate(slot: dict) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    temperatures: dict[str, list[float]] = defaultdict(list)
    for key, value in slot["raw"].items():
        if key.startswith("__ref__"):
            agent = "ref:" + key[len("__ref__"):].split("#")[0]
        else:
            match = re.match(
                r"transcript_(.+)_[A-Za-z]+_s\d+\.json#\d+$", key)
            if not match:
                raise ValueError(f"unrecognised metric key: {key}")
            agent = match.group(1)
        values[agent].append(float(value))
        temperature = slot.get("T", {}).get(key)
        if temperature is not None:
            temperatures[agent].append(float(temperature))
    return {
        agent: {
            "R_star": sum(items) / len(items),
            "n": len(items),
            "T_median": _median(temperatures[agent]),
        }
        for agent, items in values.items()
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def filter_metrics(source: dict, event_dir: pathlib.Path) -> tuple[dict, dict]:
    corrected = copy.deepcopy(source)
    comparison = []
    total_old = total_new = 0
    index_cache: dict[str, set[int]] = {}
    for task, slot in corrected.items():
        old_agents = source[task].get("agents", aggregate(source[task]))
        kept_raw, kept_t = {}, {}
        for key, value in slot["raw"].items():
            if key.startswith("__ref__"):
                kept_raw[key] = value
                if key in slot.get("T", {}):
                    kept_t[key] = slot["T"][key]
                continue
            match = re.fullmatch(r"(transcript_(.+)_([A-Za-z]+)_s\d+\.json)#(\d+)", key)
            if not match:
                raise ValueError(f"unrecognised metric key: {key}")
            transcript_name, _arm, key_task, index_text = match.groups()
            if key_task != task:
                raise ValueError(f"task mismatch in {key}")
            event_name = transcript_name.replace("transcript_", "events_", 1)
            if event_name not in index_cache:
                event_path = event_dir / event_name
                index_cache[event_name] = completed_indices(
                    json.loads(event_path.read_text()))
            total_old += 1
            if int(index_text) in index_cache[event_name]:
                total_new += 1
                kept_raw[key] = value
                if key in slot.get("T", {}):
                    kept_t[key] = slot["T"][key]
        slot["raw"] = kept_raw
        slot["T"] = kept_t
        slot["design_source"] = "accepted_and_completed_events"
        slot["agents"] = aggregate(slot)
        prior = float(slot["prior"])
        for agent, new in sorted(slot["agents"].items()):
            if agent.startswith("ref:"):
                continue
            old = old_agents[agent]
            comparison.append({
                "task": task,
                "arm": agent,
                "n_old": int(old["n"]),
                "n_completed": int(new["n"]),
                "R_star_old": float(old["R_star"]),
                "R_star_completed": float(new["R_star"]),
                "c_old": 1 - float(old["R_star"]) / prior,
                "c_completed": 1 - float(new["R_star"]) / prior,
            })
    audit = {
        "old_designs": total_old,
        "completed_designs": total_new,
        "excluded_designs": total_old - total_new,
        "comparison": comparison,
    }
    return corrected, audit


def markdown(audit: dict) -> str:
    lines = [
        "# Achievable-regret correction from completed events",
        "",
        f"The old aggregation used {audit['old_designs']} parsed designs. The "
        f"completed-event aggregation uses {audit['completed_designs']} and excludes "
        f"{audit['excluded_designs']} rejected designs.",
        "",
        "| Task | Arm | n old | n completed | R* old | R* completed | c old | c completed | Δc (pp) | η old | η completed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["comparison"]:
        dc = 100 * (row["c_completed"] - row["c_old"])
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['n_old']} | "
            f"{row['n_completed']} | {row['R_star_old']:.6f} | "
            f"{row['R_star_completed']:.6f} | {100*row['c_old']:.1f}% | "
            f"{100*row['c_completed']:.1f}% | {dc:+.1f} | "
            f"{100*row.get('eta_old', float('nan')):.1f}% | "
            f"{100*row.get('eta_completed', float('nan')):.1f}% |")
    return "\n".join(lines) + "\n"


def mean_regrets(directory: pathlib.Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in sorted(directory.glob("episodes_*.json")):
        arm = path.stem.removeprefix("episodes_")
        for episode in json.loads(path.read_text()):
            values[(arm, episode["task"])].append(float(episode["regret"]))
    return {key: sum(items) / len(items) for key, items in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--events", required=True, type=pathlib.Path)
    parser.add_argument("--episodes", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    corrected, audit = filter_metrics(
        json.loads(args.source.read_text()), args.events)
    regrets = mean_regrets(args.episodes)
    for row in audit["comparison"]:
        regret = regrets[(row["arm"], row["task"])]
        row["mean_realized_regret"] = regret
        row["eta_old"] = row["R_star_old"] / regret
        row["eta_completed"] = row["R_star_completed"] / regret
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corrected, indent=2, allow_nan=False))
    audit_path = args.out.with_name(args.out.stem + "_comparison.json")
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False))
    audit_path.with_suffix(".md").write_text(markdown(audit))
    print(json.dumps({key: audit[key] for key in (
        "old_designs", "completed_designs", "excluded_designs")}, indent=2))


if __name__ == "__main__":
    main()
