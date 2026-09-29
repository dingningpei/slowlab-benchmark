#!/usr/bin/env python3
"""Audit Phase 4 completeness, execution outcomes, tool adoption, and API usage."""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np


EXPECTED_ROUNDS = {"T1": 2, "T2": 2, "T3": 3, "T4": 3}


def mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def summarize(rows):
    calls = [call for row in rows for call in row.get("api_calls", [])]
    tool_records = [record for row in rows for record in row.get("tool_use_records", [])]
    design = [record["design_adopted"] for record in tool_records
              if record.get("design_adopted") is not None]
    inference = [record["inference_adopted"] for record in tool_records
                 if record.get("inference_adopted") is not None]
    early = [row for row in rows
             if int(row.get("rounds_submitted") or 0) < EXPECTED_ROUNDS[row["task"]]]
    retry_rows = [row for row in rows
                  if any(int(call.get("attempt", 1)) > 1
                         for call in row.get("api_calls", []))]
    return {
        "episodes": len(rows),
        "mean_final_simple_regret": mean(rows, "regret"),
        "mean_cumulative_regret": mean(rows, "cumulative_regret"),
        "mean_campaign_cash": mean(rows, "cash"),
        "mean_rounds_submitted": mean(rows, "rounds_submitted"),
        "episodes_below_round_cap": len(early),
        "format_failures": int(sum(int(row.get("format_failures") or 0) for row in rows)),
        "infeasible_submissions": int(sum(int(row.get("infeasible") or 0) for row in rows)),
        "design_adoption_rate": float(np.mean(design)) if design else None,
        "inference_adoption_rate": float(np.mean(inference)) if inference else None,
        "successful_api_calls": len(calls),
        "episodes_with_transport_retry": len(retry_rows),
        "max_successful_attempt": max((int(call.get("attempt", 1)) for call in calls),
                                      default=None),
        "reported_cost_usd": sum(float((call.get("usage") or {}).get("cost") or 0)
                                 for call in calls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    identities = defaultdict(list)
    for path in sorted(args.root.rglob("episodes_*.json")):
        relative = path.relative_to(args.root)
        parts = relative.parts
        matrix, variant, condition = parts[0], parts[1], parts[2]
        for row in json.loads(path.read_text()):
            group = f"{matrix}/{variant}/{condition}/{row['task']}/{row['model']}"
            groups[group].append(row)
            identity = (matrix, variant, condition, row["task"], row["model"],
                        int(row["seed"]), int(row["generation_seed"]))
            identities[identity].append(str(path))
    duplicates = {"|".join(map(str, key)): paths for key, paths in identities.items()
                  if len(paths) > 1}
    summaries = {name: summarize(rows) for name, rows in sorted(groups.items())}
    all_rows = [row for rows in groups.values() for row in rows]
    report = {
        "n_episode_rows": len(all_rows),
        "duplicate_episode_identities": duplicates,
        "groups": summaries,
        "overall_api": {
            "reported_cost_usd": sum(value["reported_cost_usd"]
                                     for value in summaries.values()),
            "successful_api_calls": sum(value["successful_api_calls"]
                                        for value in summaries.values()),
            "episodes_with_transport_retry": sum(value["episodes_with_transport_retry"]
                                                 for value in summaries.values()),
            "max_successful_attempt": max(value["max_successful_attempt"] or 0
                                          for value in summaries.values()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
