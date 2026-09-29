#!/usr/bin/env python3
"""Held-out scripted counterfactuals for Phase 3 feedback and constraints."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.strong_baselines import (ConstraintAwareBatchBOAgent,
                                      ScheduledObservationBatchBOAgent)


def run_condition(task, seed, condition, agent):
    env = SlowLabEnv(task, seed=seed)
    point = agent.run(env)
    _, optimum = env.truth.oracle()
    regret = float(optimum - env.truth(point[None, :])[0])
    visits = getattr(agent, "visit_records", [])
    detection = next((record["day"] for record in visits
                      if record["separation"] >= 0.10), None)
    return {"condition": condition, "seed": seed, "regret": regret,
            "rounds": task.n_rounds, "slots_per_round": task.units_per_round,
            "total_slots": task.budget_units, "shared_control": task.shared_control,
            "measurement_count": sum(r["n_measurements"] for r in visits),
            "measurement_cost": float(sum(r["cost"] for r in visits)),
            "first_detection_day_within_round": detection,
            "visit_records": visits,
            "final_recommendation": point.tolist()}


def summarize(rows, key="condition"):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: {"n": len(values),
                   "mean_regret": float(np.mean([v["regret"] for v in values])),
                   "se_regret": float(np.std([v["regret"] for v in values], ddof=1)
                                      / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                   "mean_measurements": float(np.mean([v["measurement_count"] for v in values])),
                   "mean_measurement_cost": float(np.mean([v["measurement_cost"] for v in values])),
                   "detection_rate": float(np.mean([
                       v["first_detection_day_within_round"] is not None for v in values]))}
            for name, values in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=5200)
    parser.add_argument("--out", default=str(
        ROOT.parent / "slowlab" / "output" / "reviews" /
        "phase3_feedback_mechanisms.json"))
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    rows = []

    # Fixed 24 slots: more rounds mean more opportunities to use feedback.
    for rounds, batch in [(2, 12), (3, 8), (4, 6), (6, 4)]:
        task = copy.deepcopy(TASKS["T3"])
        task.n_rounds, task.units_per_round = rounds, batch
        for seed in seeds:
            rows.append(run_condition(task, seed, f"feedback_{rounds}x{batch}",
                                      ConstraintAwareBatchBOAgent(fixed_replicates=1)))

    # Same 3x12 budget with/without the physical chamber hierarchy.
    for shared in [True, False]:
        task = copy.deepcopy(TASKS["T3"]); task.shared_control = shared
        for seed in seeds:
            rows.append(run_condition(task, seed, f"shared_control_{shared}",
                                      ConstraintAwareBatchBOAgent(fixed_replicates=1)))

    # Same policy and budget under predeclared plausible relative noise levels.
    for scale in [0.5, 1.0, 1.5]:
        task = copy.deepcopy(TASKS["T3"])
        task.plant_cv *= scale; task.tau_chamber *= scale
        task.tau_loop *= scale; task.tau_batch *= scale
        for seed in seeds:
            rows.append(run_condition(task, seed, f"noise_{scale:.1f}x",
                                      ConstraintAwareBatchBOAgent(fixed_replicates=1)))

    # Equal design policy and resources. Both observation arms use two visits;
    # active timing may bring the second visit forward. Observation cannot change
    # a submitted treatment, so its causal endpoint is interim decision timing.
    observation_rows = []
    for seed in seeds:
        task = copy.deepcopy(TASKS["T3"])
        observation_rows.append(run_condition(
            task, seed, "terminal_only",
            ConstraintAwareBatchBOAgent(fixed_replicates=1)))
        observation_rows.append(run_condition(
            task, seed, "fixed_visits",
            ScheduledObservationBatchBOAgent(active=False, fixed_replicates=1)))
        observation_rows.append(run_condition(
            task, seed, "active_visits",
            ScheduledObservationBatchBOAgent(active=True, fixed_replicates=1)))
    rows.extend(observation_rows)

    # Paired identity check: because later design consumes completed outcomes,
    # passive visits should not alter final recommendations or regret.
    paired = defaultdict(dict)
    for row in observation_rows:
        paired[row["seed"]][row["condition"]] = row
    identity = all(np.allclose(
        paired[s]["terminal_only"]["final_recommendation"],
        paired[s][condition]["final_recommendation"])
        for s in paired for condition in ["fixed_visits", "active_visits"])
    report = {
        "protocol_version": "phase3-feedback-mechanisms-0.1",
        "heldout_seeds": list(seeds), "summary": summarize(rows),
        "observation_final_policy_identity": identity,
        "interpretation": {
            "within_cycle": "Passive inspection changes interim information timing but cannot change the already submitted treatment. Under synchronous rounds and a next-design reader that uses completed outcomes, final policies are expected to match.",
            "early_termination": "Not tested: V2 does not define cleanup time, residual value, replant dynamics, or an early-stop oracle.",
            "asynchrony": "Not tested: the facility currently has one active batch and no staggered-release action space. Smaller synchronous batches are reported only as a feedback-frequency counterfactual."
        },
        "rows": rows,
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": report["summary"],
                      "observation_final_policy_identity": identity,
                      "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
