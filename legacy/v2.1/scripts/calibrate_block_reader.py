#!/usr/bin/env python3
"""Select block-aware GP settings on validation sites and score once on held-out sites."""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.agents import DoEAgent, GP
from slowlab.strong_baselines import BlockAwareReader, completed_history_arrays


def candidates(task, seed=771, n=4000):
    if task.d == 1:
        return np.linspace(0.0, 1.0, 1001)[:, None]
    return np.random.default_rng(seed + task.d).random((n, task.d))


def fixed_histories(task_name, seeds):
    histories = []
    for seed in seeds:
        env = SlowLabEnv(TASKS[task_name], seed=seed)
        DoEAgent().run(env)
        histories.append(env)
    return histories


def reader_metrics(env, reader, points):
    recommendation = reader.recommend(env, candidates=points)
    _, optimum = env.truth.oracle()
    response = env.truth(points)
    scale = max(float(optimum - np.min(response)), 1e-9)
    regret = float(optimum - env.truth(recommendation[None, :])[0])
    probes = env.probe_set()
    truth = env.truth(probes)
    mean, sd = reader.predict_observation(probes)
    lo, hi = mean - 1.959964 * sd, mean + 1.959964 * sd
    return {
        "normalised_regret": regret / scale,
        "regret": regret,
        "predictive_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "coverage": float(np.mean((lo <= truth) & (truth <= hi))),
        "mean_width": float(np.mean(hi - lo)),
        "absolute_standardised_errors": (np.abs(mean - truth) /
                                           np.maximum(sd, 1e-12)).tolist(),
    }


def ordinary_gp_metrics(env, points):
    X, y, *_ = completed_history_arrays(env)
    model = GP(noise=max(0.15, env.task.plant_cv)).fit(X, y)
    mean, _ = model.predict(points)
    recommendation = points[int(np.argmax(mean))]
    _, optimum = env.truth.oracle()
    response = env.truth(points)
    scale = max(float(optimum - np.min(response)), 1e-9)
    regret = float(optimum - env.truth(recommendation[None, :])[0])
    return {"normalised_regret": regret / scale, "regret": regret}


def aggregate(rows):
    keys = [key for key in rows[0]
            if key not in {"task", "seed", "absolute_standardised_errors"}]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate(histories, config):
    rows = []
    for env in histories:
        points = candidates(env.task)
        row = reader_metrics(env, BlockAwareReader(**config), points)
        row.update(task=env.task.name, seed=env.seed)
        rows.append(row)
    return aggregate(rows), rows


def calibrate_uncertainty(histories, mean_config, quantile=0.95):
    _, rows = evaluate(histories, mean_config)
    z = np.concatenate([row["absolute_standardised_errors"] for row in rows])
    return float(max(1.0, np.quantile(z, quantile) / 1.959964))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-seeds", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, default=12)
    parser.add_argument("--tasks", nargs="+", default=["T1", "T3"])
    parser.add_argument("--report", default=str(
        ROOT.parent / "slowlab" / "output" / "reviews" /
        "phase3_block_reader_calibration.json"))
    parser.add_argument("--config-out", default=str(
        ROOT / "configs" / "block_aware_gp_phase3.json"))
    args = parser.parse_args()

    validation_seeds = list(range(200, 200 + args.validation_seeds))
    test_seeds = list(range(1200, 1200 + args.test_seeds))
    mean_grid = [
        {"length_scale": ls, "chamber_multiplier": cm, "batch_multiplier": bm}
        for ls, cm, bm in itertools.product(
            [0.15, 0.28, 0.45], [0.5, 1.0, 2.0], [0.5, 1.0])
    ]
    task_reports, task_parameters = {}, {}
    all_block_rows, all_ordinary_rows = [], []
    for task_name in args.tasks:
        validation = fixed_histories(task_name, validation_seeds)
        grid_rows = []
        for config in mean_grid:
            metrics, _ = evaluate(validation, config)
            grid_rows.append({"config": config, "metrics": metrics})
        chosen = min(grid_rows,
                     key=lambda row: row["metrics"]["normalised_regret"])
        parameters = dict(chosen["config"])
        parameters["uncertainty_multiplier"] = calibrate_uncertainty(
            validation, chosen["config"])
        task_parameters[task_name] = parameters

        heldout = fixed_histories(task_name, test_seeds)
        block_metrics, block_rows = evaluate(heldout, parameters)
        ordinary_rows = []
        for env in heldout:
            row = ordinary_gp_metrics(env, candidates(env.task))
            row.update(task=env.task.name, seed=env.seed)
            ordinary_rows.append(row)
        ordinary_metrics = aggregate(ordinary_rows)
        paired = np.asarray([
            ordinary["normalised_regret"] - block["normalised_regret"]
            for block, ordinary in zip(block_rows, ordinary_rows)], float)
        task_reports[task_name] = {
            "grid": grid_rows,
            "chosen_mean_config": chosen,
            "parameters": parameters,
            "heldout_block_aware": block_metrics,
            "heldout_ordinary_gp": ordinary_metrics,
            "paired_normalised_regret_improvement": {
                "mean": float(paired.mean()),
                "se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
                "fraction_block_aware_better": float(np.mean(paired > 0)),
                "n": int(len(paired)),
            },
        }
        all_block_rows.extend(block_rows)
        all_ordinary_rows.extend(ordinary_rows)

    paired = np.asarray([
        ordinary["normalised_regret"] - block["normalised_regret"]
        for block, ordinary in zip(all_block_rows, all_ordinary_rows)], float)
    paired_se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    calibration_status = (
        "passed_superiority_gate" if paired.mean() > 1.959964 * paired_se
        else "calibrated_but_not_stronger")
    report = {
        "protocol_version": "phase3-block-reader-calibration-0.2",
        "calibration_status": calibration_status,
        "tasks": args.tasks,
        "validation_seeds": validation_seeds,
        "heldout_test_seeds": test_seeds,
        "selection_metric": "per-task mean normalised recommendation regret",
        "uncertainty_target": "validation 95th percentile absolute standardised error",
        "task_reports": task_reports,
        "overall_heldout_block_aware": aggregate(all_block_rows),
        "overall_heldout_ordinary_gp": aggregate(all_ordinary_rows),
        "overall_paired_normalised_regret_improvement": {
            "mean": float(paired.mean()),
            "se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
            "fraction_block_aware_better": float(np.mean(paired > 0)),
            "n": int(len(paired)),
        },
    }
    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    config_path = pathlib.Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "protocol_version": report["protocol_version"],
        "selected_on_tasks": args.tasks,
        "selection_metric": report["selection_metric"],
        "validation_seeds": validation_seeds,
        "heldout_test_seeds": test_seeds,
        "task_parameters": task_parameters,
        "calibration_status": calibration_status,
        "heldout_summary": report["overall_paired_normalised_regret_improvement"],
    }, indent=2) + "\n")
    print(json.dumps({
        "task_parameters": task_parameters,
        "task_heldout": {task: {
            "block": task_reports[task]["heldout_block_aware"],
            "ordinary": task_reports[task]["heldout_ordinary_gp"],
            "paired": task_reports[task]["paired_normalised_regret_improvement"],
        } for task in args.tasks},
        "overall_paired": report["overall_paired_normalised_regret_improvement"],
        "report": str(report_path),
        "config": str(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
