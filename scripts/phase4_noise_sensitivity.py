#!/usr/bin/env python3
"""Phase 4, section 4.4: external-validity sensitivity for the noise-layer parameters.

`slowlab/tasks.py` says outright that `plant_cv`, `tau_chamber`, `tau_loop` and
`tau_batch` are "not literature values" -- they were chosen to *match the magnitude*
of published yield CVs, not read off a source. This script asks whether the Phase 4
qualitative finding -- that a more sophisticated (design- or inference-informed)
strategy does not reliably beat a naive one on final simple regret -- survives moving
those four numbers up and down.

This is a *scripted-baseline* proxy for the LLM tool-effect finding, not a replay of
the actual Phase 4 LLM transcripts (those would need `slowlab.replay` reconstruction
from `transcript_*.json` and are out of scope for this pass -- see the report's
"not done" section). It reuses four agents from `slowlab.registry` as stand-ins for
the four Phase 4 tool conditions:

    bare            -> random_spread            (no design or inference help)
    design-tool     -> split_plot_doe            (structured design, naive readout)
    inference-tool  -> gp_ucb_profit              (naive design, GP-informed pick)
    both            -> constraint_aware_batch_bo  (both)

It also reports whether the deterministic optimum moves -- it should not, since these
are noise-layer parameters, not response-mean parameters; a mismatch would mean the
task's difficulty is itself confounded with an unsourced number, which is exactly
what this check exists to catch.

Uses a fresh, disjoint seed range (9000+) never touched by development, calibration,
or the frozen Phase 4 confirmatory protocol (which reserved 6000-6811).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent

AGENT_FOR_CONDITION = {
    "bare": "random_spread",
    "design": "split_plot_doe",
    "inference": "gp_ucb_profit",
    "both": "constraint_aware_batch_bo",
}
MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)
SEED_START = 9000


def perturbed_task(mult: float):
    base = TASKS["Optimise"]
    return replace(base, plant_cv=base.plant_cv * mult, tau_chamber=base.tau_chamber * mult,
                   tau_loop=base.tau_loop * mult, tau_batch=base.tau_batch * mult)


def optimum_location(task, seed):
    """The site's true optimal treatment, read from the privileged oracle agent --
    used only to check whether perturbing the noise layer moves the response mean,
    which it should not."""
    env = SlowLabEnv(task, seed=seed)
    agent = make_agent("site_oracle")
    x = agent.run(env)
    return np.asarray(x, float).tolist()


def run_episode(task, agent_name, seed):
    env = SlowLabEnv(task, seed=seed)
    agent = make_agent(agent_name)
    x = agent.run(env)
    r = env.submit_recommendation(x, agent_name, probes=getattr(agent, "probes", None))
    return float(r.simple_regret)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                              "/phase4_noise_sensitivity.json"))
    args = parser.parse_args()
    seeds = list(range(SEED_START, SEED_START + args.seeds))

    optima = defaultdict(list)
    regret_by_mult_condition = defaultdict(lambda: defaultdict(list))
    for mult in MULTIPLIERS:
        task = perturbed_task(mult)
        for seed in seeds:
            optima[mult].append(optimum_location(task, seed))
            for condition, agent_name in AGENT_FOR_CONDITION.items():
                regret_by_mult_condition[mult][condition].append(
                    run_episode(task, agent_name, seed))
        print(f"mult={mult}: done {len(seeds)} seeds x {len(AGENT_FOR_CONDITION)} agents",
              flush=True)

    # optimum stability: compare each multiplier's mean optimum location to the mult=1.0 baseline
    base_optima = np.asarray(optima[1.0], float)
    optimum_shift = {}
    for mult in MULTIPLIERS:
        arr = np.asarray(optima[mult], float)
        optimum_shift[str(mult)] = {
            "mean_abs_l2_shift_from_mult_1.0": float(np.mean(np.linalg.norm(arr - base_optima, axis=1))),
            "mean_point": arr.mean(0).tolist(),
        }

    contrasts = {}
    for mult in MULTIPLIERS:
        by_cond = regret_by_mult_condition[mult]
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

    # direction stability: does each contrast keep the same sign across all four multipliers?
    direction_stable = {}
    for condition in ("design", "inference", "both"):
        signs = {contrasts[str(m)][f"{condition}_minus_bare"]["sign"] for m in MULTIPLIERS}
        direction_stable[condition] = len(signs) == 1

    report = {
        "purpose": "scripted-baseline proxy for Phase 4 tool-effect robustness under "
                   "perturbed (unsourced) noise-layer parameters; NOT a replay of the "
                   "actual LLM transcripts",
        "task": "Optimise (T3)",
        "base_params": {"plant_cv": TASKS["Optimise"].plant_cv,
                        "tau_chamber": TASKS["Optimise"].tau_chamber,
                        "tau_loop": TASKS["Optimise"].tau_loop,
                        "tau_batch": TASKS["Optimise"].tau_batch},
        "multipliers": list(MULTIPLIERS),
        "n_seeds": len(seeds),
        "seed_range": [SEED_START, SEED_START + args.seeds - 1],
        "agent_for_condition": AGENT_FOR_CONDITION,
        "optimum_location_stability": optimum_shift,
        "contrasts_by_multiplier": contrasts,
        "direction_stable_across_multipliers": direction_stable,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"direction_stable_across_multipliers": direction_stable,
                      "optimum_location_stability": optimum_shift}, indent=2))


if __name__ == "__main__":
    main()
