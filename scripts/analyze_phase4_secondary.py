#!/usr/bin/env python3
"""Run the frozen site-level analysis for a named Phase 4 secondary matrix."""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from analyze_phase4 import bootstrap, load_rows, normal_pvalue, paired_family


ROOT = pathlib.Path(__file__).resolve().parents[1]


def adjust_bh(items):
    """Benjamini-Hochberg adjusted p values, returned in input-name order."""
    ordered = sorted(items, key=lambda item: item[1])
    adjusted = {}
    running = 1.0
    total = len(ordered)
    for rank, (name, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, total * p_value / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def api_audit(rows):
    calls = [call for row in rows for call in row.get("api_calls", [])]
    usage = [call.get("usage") or {} for call in calls]
    return {
        "n_api_calls": len(calls),
        "actual_models": sorted({call.get("actual_model") for call in calls
                                  if call.get("actual_model")}),
        "providers": sorted({call.get("provider") for call in calls
                             if call.get("provider")}),
        "max_attempt": max((int(call.get("attempt", 1)) for call in calls), default=None),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in usage),
        "reported_cost_usd": sum(float(item.get("cost") or 0.0) for item in usage),
    }


def find_matrix(config, name):
    for matrix in config["secondary_matrices"]:
        if matrix["name"] == name:
            return matrix
    raise SystemExit(f"unknown secondary matrix {name!r}")


def paired_condition_contrast(rows, task, model, baseline, treatment,
                              expected_sites, expected_gseeds):
    cell = {}
    duplicates = []
    for row in rows:
        if row.get("task") != task or row.get("model") != model:
            continue
        key = (row["analysis_condition"], int(row["seed"]),
               int(row["generation_seed"]))
        if key in cell:
            duplicates.append(key)
        cell[key] = row
    missing = [(condition, site, gseed)
               for condition in (baseline, treatment)
               for site in expected_sites for gseed in expected_gseeds
               if (condition, site, gseed) not in cell]
    by_site = defaultdict(list)
    for site in expected_sites:
        for gseed in expected_gseeds:
            a = cell.get((treatment, site, gseed))
            b = cell.get((baseline, site, gseed))
            if a is not None and b is not None:
                by_site[site].append(float(a["regret"] - b["regret"]))
    values = np.asarray([np.mean(value) for value in by_site.values()], float)
    return {
        "contrast": {
            "n_sites": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "standardized_effect": float(values.mean() / values.std(ddof=1))
            if len(values) > 1 and values.std(ddof=1) else None,
            "ci95": bootstrap(values) if len(values) else None,
            "p_raw_normal_approx": normal_pvalue(values),
        },
        "missing_cells": missing,
        "duplicate_cells": duplicates,
    }


def analyze_matrix(config, matrix, root):
    model = config["models"][matrix["model"]]["requested_id"]
    sites = list(range(matrix["site_seeds"]["start"],
                       matrix["site_seeds"]["start"] + matrix["site_seeds"]["count"]))
    rows = load_rows(root / matrix["name"])

    if matrix["name"] == "strong_model_replication":
        result = paired_family(rows, matrix["task"], model, matrix["tool_modes"],
                               sites, matrix["generation_seeds"])
        contrasts = result["contrasts"]
    elif matrix["name"] == "task_heterogeneity":
        result = {"contrasts": {}, "missing_cells": [], "duplicate_cells": []}
        for task in matrix["tasks"]:
            task_result = paired_family(rows, task, model, matrix["tool_modes"],
                                        sites, matrix["generation_seeds"])
            for name, value in task_result["contrasts"].items():
                result["contrasts"][f"{task}:{name}"] = value
            result["missing_cells"].extend((task, *item)
                                           for item in task_result["missing_cells"])
            result["duplicate_cells"].extend((task, *item)
                                             for item in task_result["duplicate_cells"])
        contrasts = result["contrasts"]
    elif matrix["name"] == "within_cycle_feedback":
        rows = []
        for condition in matrix["conditions"]:
            condition_rows = load_rows(root / matrix["name"] / "standard" / condition)
            for row in condition_rows:
                row["analysis_condition"] = condition
            rows.extend(condition_rows)
        paired = paired_condition_contrast(
            rows, matrix["task"], model, matrix["conditions"][0],
            matrix["conditions"][1], sites, matrix["generation_seeds"])
        name = f"{matrix['conditions'][1]}_minus_{matrix['conditions'][0]}"
        result = {
            "contrasts": {name: paired.pop("contrast")},
            **paired,
        }
        contrasts = result["contrasts"]
    else:
        raise SystemExit(f"unsupported secondary matrix {matrix['name']!r}")

    adjusted = adjust_bh([(name, value["p_raw_normal_approx"])
                          for name, value in contrasts.items()])
    for name, value in contrasts.items():
        value.pop("p_holm", None)
        value["p_bh"] = adjusted[name]
    return rows, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path,
                        default=ROOT / "configs" / "phase4_preregistered.json")
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    matrix = find_matrix(config, args.matrix)
    rows, result = analyze_matrix(config, matrix, args.root)
    if result["missing_cells"] and not args.allow_incomplete:
        raise SystemExit(
            f"{matrix['name']} incomplete: {len(result['missing_cells'])} missing cells")

    report = {
        "protocol_id": config["protocol_id"],
        "matrix": matrix["name"],
        "analysis_family": "secondary",
        "result": result,
        "n_episode_rows": len(rows),
        "bootstrap_seed": 20260913,
        "bootstrap_resamples": 10_000,
        "api_audit": api_audit(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
