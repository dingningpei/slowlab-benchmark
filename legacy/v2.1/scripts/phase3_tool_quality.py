#!/usr/bin/env python3
"""Select GP settings on validation sites and evaluate on independent sites."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.agents import GP
from slowlab.registry import make_agent
from slowlab.strong_baselines import BlockAwareReader


def candidates(task, seed=7331, n=5000):
    if task.d == 1:
        return np.linspace(0, 1, 1001)[:, None]
    return np.random.default_rng(seed + task.d).random((n, task.d))


def regret(env, point):
    if env.task.transfer_energy_shock is not None:
        shock = float(env.task.transfer_energy_shock)
        _, best = env.truth.oracle_at(shock)
        return float(best - env.truth.profit_at(point[None, :], shock)[0])
    _, best = env.truth.oracle()
    return float(best - env.truth(point[None, :])[0])


def ordinary(env, points, length_scale, noise):
    X, y = env.as_arrays()
    model = GP(ls=length_scale, noise=noise).fit(X, y)
    return points[int(np.argmax(model.predict(points)[0]))]


def evaluate(histories, configs):
    values = {config["name"]: [] for config in configs}
    for env in histories:
        points = candidates(env.task)
        for config in configs:
            if config["family"] == "ordinary":
                point = ordinary(env, points, config["length_scale"], config["noise"])
            else:
                point = BlockAwareReader(
                    length_scale=config["length_scale"],
                    chamber_multiplier=config["chamber_multiplier"],
                    batch_multiplier=config["batch_multiplier"],
                    uncertainty_multiplier=1.0).recommend(env, candidates=points)
            values[config["name"]].append(regret(env, point))
    return {name: {"n": len(rows), "mean_regret": float(np.mean(rows)),
                   "se_regret": float(np.std(rows, ddof=1) / np.sqrt(len(rows)))
                   if len(rows) > 1 else 0.0}
            for name, rows in values.items()}


def histories(task_name, seeds):
    result = []
    for seed in seeds:
        env = SlowLabEnv(TASKS[task_name], seed=seed)
        make_agent("constraint_aware_batch_bo").run(env)
        result.append(env)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["T1", "T3", "T4"])
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--validation-start", type=int, default=5400)
    parser.add_argument("--test-start", type=int, default=5500)
    parser.add_argument("--out", default=str(
        ROOT.parent / "slowlab" / "output" / "reviews" /
        "phase3_tool_quality.json"))
    args = parser.parse_args()
    ordinary_configs = [
        {"name": f"ordinary_ls{ls}_n{noise}", "family": "ordinary",
         "length_scale": ls, "noise": noise}
        for ls in [0.16, 0.28, 0.45] for noise in [0.12, 0.25, 0.45]]
    block_configs = [
        {"name": f"block_ls{ls}_c{cm}_b{bm}", "family": "block",
         "length_scale": ls, "chamber_multiplier": cm, "batch_multiplier": bm}
        for ls in [0.20, 0.32, 0.45] for cm, bm in [(0.5, 0.5), (1.0, 1.0)]]
    configs = ordinary_configs + block_configs
    report = {"protocol_version": "phase3-tool-quality-0.1", "tasks": {}}
    for task_name in args.tasks:
        validation_seeds = list(range(args.validation_start,
                                      args.validation_start + args.sites))
        test_seeds = list(range(args.test_start, args.test_start + args.sites))
        validation = evaluate(histories(task_name, validation_seeds), configs)
        selected_name = min(validation, key=lambda name: validation[name]["mean_regret"])
        selected_config = next(config for config in configs if config["name"] == selected_name)
        # Evaluate every configuration on test sites for sensitivity, while the
        # reported selected result is locked using validation sites alone.
        test = evaluate(histories(task_name, test_seeds), configs)
        report["tasks"][task_name] = {
            "design_source": "constraint_aware_batch_bo",
            "fixed_actual_histories": True,
            "validation_seeds": validation_seeds, "test_seeds": test_seeds,
            "selected_on_validation": selected_config,
            "selected_validation": validation[selected_name],
            "selected_test": test[selected_name],
            "validation_all": validation, "test_all": test,
        }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({task: {"selected": item["selected_on_validation"],
                             "validation": item["selected_validation"],
                             "test": item["selected_test"]}
                      for task, item in report["tasks"].items()}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
