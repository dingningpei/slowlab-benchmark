#!/usr/bin/env python3
"""Derive the Phase 4 site count from earlier paired confidence intervals."""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", type=pathlib.Path, required=True)
    parser.add_argument("--target-half-width", type=float, default=0.018)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.phase1.read_text())["tool_comparisons"]
    # Phase 1 used n=20 and a two-sided t critical value of 2.093. Recover each
    # paired SD from its published interval rather than accessing exploratory raw
    # transcripts in the confirmatory planner.
    sds = []
    for row in old:
        half = (row["ci95_difference"][1] - row["ci95_difference"][0]) / 2
        sds.append(float(half * np.sqrt(row["n"]) / 2.093))
    quantiles = {str(q): float(np.quantile(sds, q)) for q in [0.5, 0.75, 0.9, 1.0]}
    target_sd = quantiles["0.75"]
    normal_required = int(math.ceil((1.96 * target_sd / args.target_half_width) ** 2))
    # Inflate modestly for finite-site t intervals and unusable episodes. The
    # statistical unit remains the site; two generation seeds estimate model
    # stochasticity but do not double the nominal site count.
    frozen_sites = int(math.ceil(normal_required * 1.10 / 8) * 8)
    report = {
        "source": str(args.phase1), "source_comparisons": len(old),
        "recovered_paired_sd_quantiles": quantiles,
        "planning_sd_quantile": 0.75, "planning_sd": target_sd,
        "target_95ci_half_width": args.target_half_width,
        "normal_approximation_sites": normal_required,
        "inflation": 1.10, "rounded_frozen_primary_sites": frozen_sites,
        "generation_seeds_per_site": 2,
        "note": "Inference is clustered by site; generation seeds estimate within-site model stochasticity.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
