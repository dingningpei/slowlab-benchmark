#!/usr/bin/env python3
"""Compute design-shape diagnostics from completed execution events."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.eventlog import design_arrays_from_events
from slowlab.llm import _extract_json

TASK_PATTERN = "Sanity|Screen|Optimise|Transfer"


def _add(raw, arm, task_name, arrays):
    task = TASKS[task_name]
    for points, _blocks in arrays:
        unique = np.unique(np.round(points, 12), axis=0)
        rank = np.linalg.matrix_rank(
            np.column_stack([np.ones(len(unique)), unique]))
        raw[(arm, task_name)]["distinct"].append(len(unique))
        raw[(arm, task_name)]["units"].append(len(points))
        raw[(arm, task_name)]["rank_deficient"].append(
            rank < task.d + 1)
        raw[(arm, task_name)]["insufficient_distinct"].append(
            len(unique) < task.d + 1)
        raw[(arm, task_name)]["replication_fraction"].append(
            1 - len(unique) / len(points))
        raw[(arm, task_name)]["width"].append(
            np.ptp(unique, axis=0).tolist())


def _rows(raw):
    rows = []
    for (arm, task_name), values in sorted(raw.items()):
        width = np.asarray(values["width"], float)
        rows.append({
            "arm": arm,
            "task": task_name,
            "n_designs": len(values["distinct"]),
            "median_distinct_treatments": float(np.median(values["distinct"])),
            "rank_deficient_rate": float(np.mean(values["rank_deficient"])),
            "insufficient_distinct_rate": float(
                np.mean(values["insufficient_distinct"])),
            "median_replication_fraction": float(
                np.median(values["replication_fraction"])),
            "mean_normalized_width": width.mean(axis=0).tolist(),
        })
    return rows


def compute(event_dir: pathlib.Path) -> dict:
    raw = defaultdict(lambda: defaultdict(list))
    for path in sorted(event_dir.glob("events_*.json")):
        match = re.fullmatch(
            rf"events_(.+)_({TASK_PATTERN})_s(\d+)\.json", path.name)
        if not match:
            continue
        arm, task_name, seed = match.groups()
        env = SlowLabEnv(TASKS[task_name], seed=int(seed))
        _add(raw, arm, task_name, design_arrays_from_events(path, env))
    return {"design_source": "accepted_and_completed_events", "rows": _rows(raw)}


def legacy_arrays(path: pathlib.Path, env):
    """Reproduce the old transcript parser for comparison only."""
    names = [factor.name for factor in env.task.factors]
    out = []
    for turn in json.loads(path.read_text()):
        try:
            blob = _extract_json(turn["assistant"])
            treatments, allocation = blob["treatments"], blob["allocation"]
        except Exception:
            continue
        points, blocks = [], []
        for treatment_id, units in allocation.items():
            treatment = treatments.get(treatment_id)
            if not treatment:
                continue
            try:
                point = [
                    (float(treatment[name]) - factor.low) /
                    (factor.high - factor.low)
                    for name, factor in zip(names, env.task.factors)
                ]
            except Exception:
                continue
            for unit in units if isinstance(units, list) else [units]:
                match = re.fullmatch(r"c(\d+)l(\d+)", str(unit))
                if match and str(unit) in env.facility._by_id:
                    points.append(point)
                    blocks.append(int(match.group(1)))
        if points:
            out.append((np.clip(np.asarray(points), 0, 1), np.asarray(blocks)))
    return out


def compute_legacy(transcript_dir: pathlib.Path) -> list[dict]:
    """Reproduce v1 geometry solely to quantify the correction."""
    raw = defaultdict(lambda: defaultdict(list))
    for path in sorted(transcript_dir.glob("transcript_*.json")):
        match = re.fullmatch(
            rf"transcript_(.+)_({TASK_PATTERN})_s(\d+)\.json", path.name)
        if not match:
            continue
        arm, task_name, seed = match.groups()
        env = SlowLabEnv(TASKS[task_name], seed=int(seed))
        _add(raw, arm, task_name, legacy_arrays(path, env))
    return _rows(raw)


def markdown(report: dict) -> str:
    lines = [
        "# Completed-event design diagnostics",
        "",
        "| Task | Arm | n | Median distinct | Rank deficient | Too few distinct | Median replication |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['n_designs']} | "
            f"{row['median_distinct_treatments']:.1f} | "
            f"{100*row['rank_deficient_rate']:.1f}% | "
            f"{100*row['insufficient_distinct_rate']:.1f}% | "
            f"{100*row['median_replication_fraction']:.1f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=pathlib.Path)
    parser.add_argument("--transcripts", type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    report = compute(args.events)
    if args.transcripts:
        report["legacy_rows"] = compute_legacy(args.transcripts)
        report["legacy_transcript_design_count"] = sum(
            row["n_designs"] for row in report["legacy_rows"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    args.out.with_suffix(".md").write_text(markdown(report))
    print(sum(row["n_designs"] for row in report["rows"]))


if __name__ == "__main__":
    main()
