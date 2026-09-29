#!/usr/bin/env python3
"""Compare v1 transcript parsing with completed-event design extraction."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.eventlog import completed_design_payloads
from slowlab.llm import _extract_json

TASKS = "Sanity|Screen|Optimise|Transfer"


def legacy_counts(path: pathlib.Path) -> tuple[int, int]:
    """Count designs/units accepted by the old transcript-only parser."""
    designs = units = 0
    for turn in json.loads(path.read_text()):
        try:
            blob = _extract_json(turn["assistant"])
            allocation = blob["allocation"]
            treatments = blob["treatments"]
        except Exception:
            continue
        used = 0
        for treatment_id, unit_ids in allocation.items():
            if treatment_id not in treatments:
                continue
            used += len(unit_ids if isinstance(unit_ids, list) else [unit_ids])
        if used:
            designs += 1
            units += used
    return designs, units


def completed_counts(document: dict) -> tuple[int, int]:
    designs = completed_design_payloads(document)
    return len(designs), sum(
        len(unit_ids)
        for design in designs
        for unit_ids in design["allocation"].values()
    )


def audit(transcripts: pathlib.Path, events: pathlib.Path) -> dict:
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    episodes = 0
    for event_path in sorted(events.glob("events_*.json")):
        match = re.fullmatch(
            rf"events_(.+)_({TASKS})_s(\d+)\.json", event_path.name)
        if not match:
            continue
        arm, task, seed = match.groups()
        transcript_path = transcripts / f"transcript_{arm}_{task}_s{seed}.json"
        if not transcript_path.exists():
            raise FileNotFoundError(transcript_path)
        document = json.loads(event_path.read_text())
        old_designs, old_units = legacy_counts(transcript_path)
        new_designs, new_units = completed_counts(document)
        row = rows[(arm, task)]
        row["episodes"] += 1
        row["legacy_designs"] += old_designs
        row["completed_designs"] += new_designs
        row["excluded_designs"] += old_designs - new_designs
        row["legacy_units"] += old_units
        row["completed_units"] += new_units
        row["rejected_events"] += int(document["counts"].get("rejected", 0))
        episodes += 1

    serial_rows = [
        {"arm": arm, "task": task, **dict(values)}
        for (arm, task), values in sorted(rows.items())
    ]
    totals = {
        key: sum(row.get(key, 0) for row in serial_rows)
        for key in ("episodes", "legacy_designs", "completed_designs",
                    "excluded_designs", "legacy_units", "completed_units",
                    "rejected_events")
    }
    if episodes != totals["episodes"]:
        raise AssertionError("episode aggregation failed")
    if totals["excluded_designs"] != totals["rejected_events"]:
        raise AssertionError(
            "legacy-minus-completed designs does not equal rejected events")
    return {"totals": totals, "rows": serial_rows}


def markdown(report: dict) -> str:
    t = report["totals"]
    lines = [
        "# V1 transcript-only vs completed-event design source",
        "",
        "The old path parsed every syntactically valid design reply. The corrected "
        "path requires a matching `accepted` and `completed` event.",
        "",
        "| Episodes | Old parsed designs | Completed designs | Excluded rejected designs | Old units | Completed units |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {t['episodes']} | {t['legacy_designs']} | {t['completed_designs']} | "
        f"{t['excluded_designs']} | {t['legacy_units']} | {t['completed_units']} |",
        "",
        "| Arm | Task | Episodes | Old | Completed | Excluded |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['arm']} | {row['task']} | {row['episodes']} | "
            f"{row['legacy_designs']} | {row['completed_designs']} | "
            f"{row['excluded_designs']} |")
    lines += [
        "",
        "`excluded_designs == rejected_events` is checked by the script. This "
        "report changes the design sample only; EIG and achievable-regret metrics "
        "must be recomputed from the completed-event designs before paper claims are updated.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", required=True, type=pathlib.Path)
    parser.add_argument("--events", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    report = audit(args.transcripts, args.events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    args.out.with_suffix(".md").write_text(markdown(report))
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
