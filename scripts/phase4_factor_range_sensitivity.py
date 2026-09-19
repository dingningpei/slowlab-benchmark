#!/usr/bin/env python3
"""Phase 4, section 4.4: external-validity sensitivity for the T3 factor ranges.

`slowlab/world.py` records that `density`'s range was once widened from
[2.5, 4.0] to [2.0, 8.0] to keep the optimum off the rail under a now-removed cost
weight, and was narrowed back to [2.2, 4.5] once the true (lambda=1) optimum landed
inside the commercial range with headroom -- i.e. the ranges have moved before, for a
documented and since-corrected reason. This checks whether the Phase 4 qualitative
finding depends on exactly where `day_temp` and `density` (T3's two factors) are
bounded, by widening and narrowing each range around its current midpoint.

Same scripted-agent proxy as `phase4_noise_sensitivity.py` and
`phase4_price_sensitivity.py`:
    bare -> random_spread, design -> split_plot_doe,
    inference -> gp_ucb_profit, both -> constraint_aware_batch_bo

Width multipliers of 0.8x/1.0x/1.2x are applied to each factor's range independently
(one factor perturbed at a time, midpoint held fixed), a plausible "we drew the
commercial range slightly wrong" perturbation rather than the wide swings used for
the confessedly-unsourced noise parameters. `slowlab.world.MANAGEMENT_FACTORS` is a
module-level list read by name on every `Task.factors` access, so bounds are mutated
in place and restored in a `finally` block -- mirroring the save/restore pattern
`World.oracle_at` already uses for `energy_shock`. The oracle cache key includes
`f.low`/`f.high` (`world._oracle_key`), so perturbed bounds cannot silently return a
cached result computed under the unperturbed range.

Also reports the rail-bound rate: the fraction of oracle-optimal points that land
within 2% of either bound, at each width -- the environment's own stated concern
(`economics.py`: "every resource factor acquires an interior optimum... a lambda != 1
is not profit") is exactly that a badly chosen range pins the optimum to a rail and
makes the search problem degenerate.

Uses a fresh, disjoint seed range (9200+).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
from slowlab.world import MANAGEMENT_FACTORS

AGENT_FOR_CONDITION = {
    "bare": "random_spread",
    "design": "split_plot_doe",
    "inference": "gp_ucb_profit",
    "both": "constraint_aware_batch_bo",
}
FACTORS_TO_PERTURB = ("day_temp", "density")
WIDTH_MULTIPLIERS = (0.8, 1.0, 1.2)
SEED_START = 9200
RAIL_TOLERANCE = 0.02  # fraction of [0, 1] normalized range counted as "on the rail"


@contextmanager
def widened(factor_name, width_mult):
    spec = next(f for f in MANAGEMENT_FACTORS if f.name == factor_name)
    old_low, old_high = spec.low, spec.high
    mid = (old_low + old_high) / 2
    half = (old_high - old_low) / 2 * width_mult
    spec.low, spec.high = mid - half, mid + half
    try:
        yield
    finally:
        spec.low, spec.high = old_low, old_high


def run_episode(task, agent_name, seed):
    env = SlowLabEnv(task, seed=seed)
    agent = make_agent(agent_name)
    x = agent.run(env)
    r = env.submit_recommendation(x, agent_name, probes=getattr(agent, "probes", None))
    return float(r.simple_regret)


def optimum_and_rail(task, seed, perturbed_index):
    env = SlowLabEnv(task, seed=seed)
    agent = make_agent("site_oracle")
    x = np.asarray(agent.run(env), float)
    on_rail = bool(x[perturbed_index] < RAIL_TOLERANCE or x[perturbed_index] > 1 - RAIL_TOLERANCE)
    return x.tolist(), on_rail


def sweep_factor(factor_name, seeds, task):
    perturbed_index = task.factor_names.index(factor_name)
    optima, rail_rate, contrasts = {}, {}, {}
    for mult in WIDTH_MULTIPLIERS:
        with widened(factor_name, mult):
            results = [optimum_and_rail(task, seed, perturbed_index) for seed in seeds]
            points, rails = zip(*results)
            optima[str(mult)] = {"mean_point": np.mean(points, axis=0).tolist()}
            rail_rate[str(mult)] = float(np.mean(rails))
            by_cond = defaultdict(list)
            for seed in seeds:
                for condition, agent_name in AGENT_FOR_CONDITION.items():
                    by_cond[condition].append(run_episode(task, agent_name, seed))
        bare = np.asarray(by_cond["bare"], float)
        row = {"mean_regret": {c: float(np.mean(v)) for c, v in by_cond.items()}}
        for condition in ("design", "inference", "both"):
            diff = np.asarray(by_cond[condition], float) - bare
            row[f"{condition}_minus_bare"] = {
                "mean": float(diff.mean()),
                "sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
                "sign": "tool_better" if diff.mean() < 0 else "tool_worse_or_equal",
            }
        contrasts[str(mult)] = row
        print(f"{factor_name} width={mult}: done {len(seeds)} seeds x "
              f"{len(AGENT_FOR_CONDITION)} agents, rail_rate={rail_rate[str(mult)]:.2f}",
              flush=True)
    direction_stable = {
        condition: len({contrasts[str(m)][f"{condition}_minus_bare"]["sign"]
                        for m in WIDTH_MULTIPLIERS}) == 1
        for condition in ("design", "inference", "both")
    }
    return {"width_multipliers": list(WIDTH_MULTIPLIERS), "optimum_location": optima,
            "rail_bound_rate": rail_rate, "contrasts_by_width": contrasts,
            "direction_stable_across_widths": direction_stable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                              "/phase4_factor_range_sensitivity.json"))
    args = parser.parse_args()
    seeds = list(range(SEED_START, SEED_START + args.seeds))
    task = TASKS["Optimise"]

    report = {
        "purpose": "scripted-strategy ranking and task geometry under perturbed factor "
                   "ranges; this tests environment behavior, not robustness of the "
                   "Phase 4 LLM tool-condition effect",
        "interpretation_limit": ("Each label maps to a different scripted algorithm, so "
                                 "these contrasts cannot be interpreted as bare/design/"
                                 "inference/both interventions on one LLM."),
        "task": "Optimise (T3)",
        "n_seeds": len(seeds),
        "seed_range": [SEED_START, SEED_START + args.seeds - 1],
        "agent_for_condition": AGENT_FOR_CONDITION,
        "rail_tolerance": RAIL_TOLERANCE,
        "by_factor": {name: sweep_factor(name, seeds, task) for name in FACTORS_TO_PERTURB},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: {"rail_bound_rate": report["by_factor"][name]["rail_bound_rate"],
                             "direction_stable": report["by_factor"][name]["direction_stable_across_widths"]}
                      for name in FACTORS_TO_PERTURB}, indent=2))


if __name__ == "__main__":
    main()
