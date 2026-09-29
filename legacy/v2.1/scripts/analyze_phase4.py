#!/usr/bin/env python3
"""Frozen site-level analysis for complete Phase 4 matrices."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np


def load_rows(folder):
    rows = []
    for path in folder.rglob("episodes_*.json"):
        for row in json.loads(path.read_text()):
            copy = dict(row)
            copy["source"] = str(path)
            rows.append(copy)
    return rows


def normal_pvalue(values):
    values = np.asarray(values, float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return 1.0 if not len(values) or values.mean() == 0 else 0.0
    z = abs(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))
    return float(math.erfc(z / np.sqrt(2)))


def bootstrap(values, seed=20260913, n_boot=10_000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def adjust_holm(items):
    ordered = sorted(items, key=lambda item: item[1])
    adjusted, running = {}, 0.0
    m = len(items)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def paired_family(rows, task, model, modes, expected_sites, expected_gseeds):
    cell = {}
    duplicates = []
    for row in rows:
        if row.get("task") != task or row.get("model") != model:
            continue
        key = (row["tool_mode"], int(row["seed"]), int(row["generation_seed"]))
        if key in cell:
            duplicates.append(key)
        cell[key] = row
    missing = [(mode, site, gseed) for mode in modes for site in expected_sites
               for gseed in expected_gseeds if (mode, site, gseed) not in cell]
    contrasts = {}
    for mode in modes:
        if mode == "bare":
            continue
        by_site = defaultdict(list)
        for site in expected_sites:
            for gseed in expected_gseeds:
                a, b = cell.get((mode, site, gseed)), cell.get(("bare", site, gseed))
                if a is not None and b is not None:
                    by_site[site].append(float(a["regret"] - b["regret"]))
        values = np.asarray([np.mean(v) for v in by_site.values()], float)
        contrasts[f"{mode}_minus_bare"] = {
            "n_sites": len(values), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "standardized_effect": float(values.mean() / values.std(ddof=1))
            if len(values) > 1 and values.std(ddof=1) else None,
            "ci95": bootstrap(values) if len(values) else None,
            "p_raw_normal_approx": normal_pvalue(values),
        }
    holm = adjust_holm([(name, value["p_raw_normal_approx"])
                        for name, value in contrasts.items()])
    for name, value in contrasts.items():
        value["p_holm"] = holm[name]
    return {"contrasts": contrasts, "missing_cells": missing, "duplicate_cells": duplicates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = load_rows(args.root / "primary")
    matrix = config["primary_matrix"]
    sites = range(matrix["site_seeds"]["start"],
                  matrix["site_seeds"]["start"] + matrix["site_seeds"]["count"])
    result = paired_family(rows, matrix["task"],
                           config["models"][matrix["model"]]["requested_id"],
                           matrix["tool_modes"], sites, matrix["generation_seeds"])
    if result["missing_cells"] and not args.allow_incomplete:
        raise SystemExit(f"primary matrix incomplete: {len(result['missing_cells'])} missing cells")
    report = {"protocol_id": config["protocol_id"], "primary": result,
              "n_episode_rows": len(rows), "bootstrap_seed": 20260913,
              "bootstrap_resamples": 10_000}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
