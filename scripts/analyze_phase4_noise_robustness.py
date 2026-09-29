#!/usr/bin/env python3
"""Analyze the frozen Phase 4 adaptive noise-robustness extension."""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from analyze_phase4 import bootstrap, load_rows, normal_pvalue
from analyze_phase4_secondary import adjust_bh

ROOT = pathlib.Path(__file__).resolve().parents[1]


def adjust_holm(items):
    ordered = sorted(items, key=lambda item: item[1])
    adjusted, running = {}, 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[name] = running
    return adjusted


def site_contrasts(rows, multiplier, sites, gseeds, modes):
    cells = {(row["tool_mode"], int(row["seed"]), int(row["generation_seed"])):
             float(row["regret"]) for row in rows
             if float(row.get("noise_multiplier", 1.0)) == float(multiplier)}
    missing = [(mode, site, gseed) for mode in modes for site in sites for gseed in gseeds
               if (mode, site, gseed) not in cells]
    values = {}
    for mode in modes:
        if mode == "bare":
            continue
        by_site = []
        for site in sites:
            diffs = [cells[(mode, site, gseed)] - cells[("bare", site, gseed)]
                     for gseed in gseeds
                     if (mode, site, gseed) in cells and ("bare", site, gseed) in cells]
            if len(diffs) == len(gseeds):
                by_site.append(float(np.mean(diffs)))
        values[mode] = np.asarray(by_site, float)
    return values, missing


def summary(values):
    return {
        "n_sites": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "ci95": bootstrap(values),
        "p_raw_normal_approx": normal_pvalue(values),
    }


def execution_audit(rows, multipliers, modes):
    calls = [call for row in rows for call in row.get("api_calls", [])]
    cells = {}
    for multiplier in multipliers:
        cells[str(multiplier)] = {}
        for mode in modes:
            selected = [row for row in rows
                        if float(row.get("noise_multiplier", 1.0)) == float(multiplier)
                        and row.get("tool_mode") == mode]
            cells[str(multiplier)][mode] = {
                "episodes": len(selected),
                "format_failures": sum(int(row.get("format_failures", 0))
                                       for row in selected),
                "infeasible_submissions": sum(int(row.get("infeasible", 0))
                                              for row in selected),
                "episodes_below_round_cap": sum(
                    int(row.get("rounds_submitted", 0)) < 3 for row in selected),
                "recorded_cost_usd": sum(
                    float((call.get("usage") or {}).get("cost") or 0.0)
                    for row in selected for call in row.get("api_calls", [])),
            }
    return {
        "successful_api_calls": len(calls),
        "actual_models": sorted({call.get("actual_model") for call in calls
                                  if call.get("actual_model")}),
        "providers": sorted({call.get("provider") for call in calls
                             if call.get("provider")}),
        "max_successful_call_attempt": max(
            (int(call.get("attempt", 1)) for call in calls), default=None),
        "recorded_cost_usd": sum(
            float((call.get("usage") or {}).get("cost") or 0.0) for call in calls),
        "note": ("Recorded cost covers successful calls attached to completed episodes. "
                 "The manual scheduler interruption may have left one in-flight call "
                 "outside an episode record."),
        "by_noise_and_mode": cells,
    }


def analyze(config: dict, root: pathlib.Path) -> dict:
    rows = load_rows(root)
    sites = list(range(config["site_seeds"]["start"],
                       config["site_seeds"]["start"] + config["site_seeds"]["count"]))
    gseeds = config["generation_seeds"]
    modes = config["tool_modes"]
    by_multiplier, missing = {}, []
    raw_values = {}
    for multiplier in config["noise_multipliers"]:
        values, absent = site_contrasts(rows, multiplier, sites, gseeds, modes)
        missing.extend((multiplier, *item) for item in absent)
        raw_values[str(multiplier)] = values
        by_multiplier[str(multiplier)] = {
            f"{mode}_minus_bare": summary(value) for mode, value in values.items()}

    primary = by_multiplier["2.0"]
    adjusted = adjust_holm([(name, value["p_raw_normal_approx"])
                            for name, value in primary.items()])
    for name, value in primary.items():
        value["p_holm"] = adjusted[name]

    secondary = {}
    for multiplier in ("0.5", "1.0"):
        for name, value in by_multiplier[multiplier].items():
            secondary[f"noise_{multiplier}:{name}"] = value
    for mode in modes:
        if mode == "bare":
            continue
        interaction = raw_values["2.0"][mode] - raw_values["1.0"][mode]
        secondary[f"noise_2.0_minus_1.0:{mode}_minus_bare"] = summary(interaction)
    bh = adjust_bh([(name, value["p_raw_normal_approx"])
                    for name, value in secondary.items()])
    for name, value in secondary.items():
        value["p_bh"] = bh[name]

    identities = [(float(row.get("noise_multiplier", 1.0)), row.get("tool_mode"),
                   int(row["seed"]), int(row["generation_seed"])) for row in rows]
    return {
        "protocol_id": config["protocol_id"],
        "analysis_family": config["analysis_family"],
        "n_episode_rows": len(rows),
        "expected_episode_rows": (len(config["noise_multipliers"]) * len(modes)
                                  * len(sites) * len(gseeds)),
        "duplicate_identities": len(identities) - len(set(identities)),
        "missing_cells": missing,
        "execution_audit": execution_audit(
            rows, config["noise_multipliers"], modes),
        "primary_high_noise": primary,
        "secondary": secondary,
        "contrasts_by_multiplier": by_multiplier,
        "bootstrap_seed": 20260913,
        "bootstrap_resamples": 10_000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path,
                        default=ROOT / "configs" / "phase4_noise_robustness.json")
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = analyze(config, args.root)
    if report["missing_cells"] and not args.allow_incomplete:
        raise SystemExit(f"noise matrix incomplete: {len(report['missing_cells'])} cells")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
