#!/usr/bin/env python3
"""Held-out noise x slot-budget comparison of adaptive and fixed replication."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.strong_baselines import BlockAwareReader, ConstraintAwareBatchBOAgent


def run_one(task, seed, fixed_replicates, reader_config):
    env = SlowLabEnv(task, seed=seed)
    agent = ConstraintAwareBatchBOAgent(
        fixed_replicates=fixed_replicates,
        reader=BlockAwareReader(**reader_config))
    point = agent.run(env)
    result = env.submit_recommendation(point, agent.name)
    return {
        "seed": seed,
        "regret": result.simple_regret,
        "cumulative_regret": result.cumulative_regret,
        "parallel_utilisation": result.parallel_utilisation,
        "replication_history": agent.replication_history,
        "n_rejections": len(result.rejections),
    }


def mean_se(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()),
            "se": float(values.std(ddof=1) / np.sqrt(len(values)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=2200)
    parser.add_argument("--config", default=str(
        ROOT / "configs" / "block_aware_gp_phase3.json"))
    parser.add_argument("--out", default=str(
        ROOT.parent / "slowlab" / "output" / "reviews" /
        "phase3_replication_grid.json"))
    args = parser.parse_args()
    config_blob = json.loads(pathlib.Path(args.config).read_text())
    reader_config = config_blob["task_parameters"]["T3"]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    noise_levels = [0.06, 0.12, 0.24, 0.40]
    budgets = [(2, 8), (2, 12), (3, 12), (5, 12)]
    policies = {"adaptive": None, "fixed_1": 1, "fixed_2": 2, "fixed_4": 4}
    cells = []
    for plant_cv in noise_levels:
        for rounds, width in budgets:
            task = copy.deepcopy(TASKS["T3"])
            task.plant_cv = plant_cv
            task.tau_chamber = plant_cv * (0.10 / 0.12)
            task.tau_loop = plant_cv * (0.05 / 0.12)
            task.tau_batch = plant_cv * (0.05 / 0.12)
            task.n_rounds = rounds
            task.units_per_round = width
            policy_rows = {}
            for policy, fixed in policies.items():
                rows = [run_one(task, seed, fixed, reader_config) for seed in seeds]
                policy_rows[policy] = {
                    "regret": mean_se([row["regret"] for row in rows]),
                    "cumulative_regret": mean_se(
                        [row["cumulative_regret"] for row in rows]),
                    "mean_replicates": float(np.mean([
                        np.mean(row["replication_history"]) for row in rows])),
                    "rejections": int(sum(row["n_rejections"] for row in rows)),
                    "rows": rows,
                }
            best_fixed = min((name for name in policies if name != "adaptive"),
                             key=lambda name: policy_rows[name]["regret"]["mean"])
            cells.append({
                "plant_cv": plant_cv,
                "rounds": rounds,
                "units_per_round": width,
                "total_slots": rounds * width,
                "policies": policy_rows,
                "best_fixed_policy": best_fixed,
                "adaptive_minus_best_fixed": (
                    policy_rows["adaptive"]["regret"]["mean"] -
                    policy_rows[best_fixed]["regret"]["mean"]),
            })
    success = sum(cell["adaptive_minus_best_fixed"] <= 0 for cell in cells)
    report = {
        "protocol_version": "phase3-replication-grid-0.1",
        "task": "T3",
        "heldout_seeds": seeds,
        "reader_config": reader_config,
        "noise_levels": noise_levels,
        "budgets": [{"rounds": r, "units_per_round": w, "total_slots": r*w}
                    for r, w in budgets],
        "cells": cells,
        "adaptive_beats_or_ties_best_fixed_cells": success,
        "n_cells": len(cells),
        "interpretation_gate": (
            "passed" if success >= int(np.ceil(0.75 * len(cells))) else "failed"),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "gate": report["interpretation_gate"],
        "adaptive_beats_or_ties": f"{success}/{len(cells)}",
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
