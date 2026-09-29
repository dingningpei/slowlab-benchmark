#!/usr/bin/env python3
"""Analyze the prespecified exploratory Phase 4 prompt-variant matrix."""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from analyze_phase4 import bootstrap, load_rows, normal_pvalue, paired_family
from analyze_phase4_secondary import api_audit


ROOT = pathlib.Path(__file__).resolve().parents[1]


def interaction(rows_by_variant, task, model, sites, gseeds):
    cells = {}
    duplicates = []
    for variant, rows in rows_by_variant.items():
        for row in rows:
            if row.get("task") != task or row.get("model") != model:
                continue
            key = (variant, row["tool_mode"], int(row["seed"]),
                   int(row["generation_seed"]))
            if key in cells:
                duplicates.append(key)
            cells[key] = row
    missing = [(variant, mode, site, gseed)
               for variant in ("standard", "constraint_checklist")
               for mode in ("bare", "inference") for site in sites for gseed in gseeds
               if (variant, mode, site, gseed) not in cells]
    by_site = defaultdict(list)
    for site in sites:
        for gseed in gseeds:
            keys = [(variant, mode, site, gseed)
                    for variant in ("standard", "constraint_checklist")
                    for mode in ("bare", "inference")]
            if all(key in cells for key in keys):
                standard = (cells[("standard", "inference", site, gseed)]["regret"] -
                            cells[("standard", "bare", site, gseed)]["regret"])
                checklist = (cells[("constraint_checklist", "inference", site, gseed)]["regret"] -
                             cells[("constraint_checklist", "bare", site, gseed)]["regret"])
                by_site[site].append(float(checklist - standard))
    values = np.asarray([np.mean(value) for value in by_site.values()], float)
    return {
        "n_sites": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "standardized_effect": float(values.mean() / values.std(ddof=1))
        if len(values) > 1 and values.std(ddof=1) else None,
        "ci95": bootstrap(values) if len(values) else None,
        "p_raw_normal_approx": normal_pvalue(values),
        "missing_cells": missing,
        "duplicate_cells": duplicates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path,
                        default=ROOT / "configs" / "phase4_preregistered.json")
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    matrix = config["exploratory_matrix"]
    model = config["models"][matrix["model"]]["requested_id"]
    sites = list(range(matrix["site_seeds"]["start"],
                       matrix["site_seeds"]["start"] + matrix["site_seeds"]["count"]))
    # run_phase4.py labels this selection "exploratory" while the matrix keeps
    # its semantic name, constraint_prompt_variant, in the frozen config.
    folder = args.root / "exploratory"
    rows_by_variant = {variant: load_rows(folder / variant)
                       for variant in matrix["prompt_variants"]}
    within_variant = {}
    for variant, rows in rows_by_variant.items():
        result = paired_family(rows, matrix["task"], model, matrix["tool_modes"],
                               sites, matrix["generation_seeds"])
        for value in result["contrasts"].values():
            value.pop("p_holm", None)
        within_variant[variant] = result
    interaction_result = interaction(rows_by_variant, matrix["task"], model,
                                     sites, matrix["generation_seeds"])
    missing = sum(len(result["missing_cells"]) for result in within_variant.values())
    missing += len(interaction_result["missing_cells"])
    if missing:
        raise SystemExit(f"exploratory matrix incomplete: {missing} missing references")
    rows = [row for variant_rows in rows_by_variant.values() for row in variant_rows]
    report = {
        "protocol_id": config["protocol_id"],
        "matrix": matrix["name"],
        "analysis_family": "exploratory_not_confirmatory",
        "within_variant": within_variant,
        "checklist_by_inference_interaction": interaction_result,
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
