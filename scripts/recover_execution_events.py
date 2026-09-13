#!/usr/bin/env python3
"""Recover accepted and completed execution events from v1 LLM transcripts.

The command is deterministic and never calls a model API.  Per-episode event
files are derived raw records and remain git-ignored; the compact audit summary
is suitable for review and version control.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import slowlab
from slowlab import TASKS
from slowlab.eventlog import (
    EVENT_SCHEMA_VERSION,
    reconcile_format_failures,
    recover_transcript,
)


def _read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def recover_directory(source: pathlib.Path, output: pathlib.Path) -> dict:
    episode_files = sorted(source.glob("episodes_*.json"))
    if not episode_files:
        raise FileNotFoundError(f"no episodes_*.json files under {source}")

    totals = Counter()
    mismatches = []
    warnings = []
    inferred_parse_failures = 0
    recovered = 0
    expected_episodes = 0

    for episode_file in episode_files:
        tag = episode_file.stem.removeprefix("episodes_")
        episodes = _read_json(episode_file)
        expected_episodes += len(episodes)
        for episode in episodes:
            task_name = str(episode["task"])
            seed = int(episode["seed"])
            transcript_file = source / f"transcript_{tag}_{task_name}_s{seed}.json"
            if not transcript_file.exists():
                mismatches.append({
                    "episode": f"{tag}/{task_name}/s{seed}",
                    "problem": "missing_transcript",
                    "path": str(transcript_file),
                })
                continue

            result = recover_transcript(
                TASKS[task_name], seed, _read_json(transcript_file),
                model=tag, episode_id=f"{tag}/{task_name}/s{seed}")
            inferred_parse_failures += reconcile_format_failures(
                result, int(episode.get("format_failures", 0)))
            recovered += 1
            totals.update(result.counts)
            event_file = output / f"events_{tag}_{task_name}_s{seed}.json"
            _write_json(event_file, result.to_dict())

            checks = {
                "completed": (result.counts.get("completed", 0),
                              int(episode.get("rounds_submitted", 0))),
                "rejected": (result.counts.get("rejected", 0),
                             int(episode.get("infeasible", 0))),
                "parse_failed": (result.counts.get("parse_failed", 0),
                                 int(episode.get("format_failures", 0))),
                "trace": (result.counts.get("completed", 0),
                          len(episode.get("trace", []))),
            }
            bad = {key: {"recovered": got, "episode": want}
                   for key, (got, want) in checks.items() if got != want}
            expected_stop = episode.get("stopped_early_at")
            got_stop = result.counts.get("stopped", 0)
            if (expected_stop is None) != (got_stop == 0):
                bad["stopped"] = {
                    "recovered": got_stop, "episode": expected_stop}
            if result.counts.get("unresolved", 0):
                bad["unresolved"] = result.counts["unresolved"]
            if result.errors:
                warnings.append({
                    "episode": f"{tag}/{task_name}/s{seed}",
                    "messages": result.errors,
                })
            if bad:
                mismatches.append({
                    "episode": f"{tag}/{task_name}/s{seed}", "problems": bad})

    summary = {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "environment_version": slowlab.ENV_VERSION,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "expected_episodes": expected_episodes,
        "recovered_episodes": recovered,
        "event_totals": dict(sorted(totals.items())),
        "inferred_parse_failures": inferred_parse_failures,
        "warning_count": len(warnings),
        "warnings": warnings,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    _write_json(output / "recovery_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=pathlib.Path,
        default=ROOT / "results" / f"llm_env{slowlab.ENV_VERSION}",
        help="directory containing v1 episodes and transcripts")
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=ROOT / "results" / f"execution_events_env{slowlab.ENV_VERSION}")
    args = parser.parse_args()
    summary = recover_directory(args.source, args.out)
    print(json.dumps({key: summary[key] for key in (
        "expected_episodes", "recovered_episodes", "event_totals",
        "mismatch_count")}, indent=2))
    if summary["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
