#!/usr/bin/env python3
"""Phase 4, section 4.4: external-validity sensitivity for the between-site
parameter *distribution* itself -- distinct from `phase4_price_sensitivity.py`,
which moves one drawn value by a multiplier. This asks whether the *spread sites
are drawn from* is wide enough.

`slowlab/economics.py` is explicit that `price_per_kg_fw`'s cited range,
(1.30, 1.62) EUR/kg, is "the weakest-sourced of the six" -- built from two national
annual points (Dutch 1.57, Spanish 1.35) with the range only slightly widened around
them, and the module's own comment says "real within-season variation is far larger,
but we have no series, so we take it narrow." Every other SITE_ECON_SPREAD entry
cites a real Eurostat/DLC cross-country range, so this is the one place the between-
site heterogeneity Phase 4's precision plan relies on could be understated.

This widens price_per_kg_fw's half-width by 1.0x/1.5x/2.0x (holding its center
fixed) via a temporary, restored patch of the module-level `SITE_ECON_SPREAD` dict,
draws fresh sites at each width, and reports:

  1. Between-site SD of the bare agent's regret -- the quantity Phase 4's own
     precision plan (`docs/phase4_analysis_plan.md`) used Phase 1-3 data to
     estimate, to check whether a wider (more honest) spread would have called for
     more than the frozen 56 primary sites.
  2. Whether the design/inference/both-vs-bare direction still holds.

Same four scripted agents as the other Phase 4 sensitivity scripts. Fresh, disjoint
seed range (9300+).
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
from slowlab.economics import SITE_ECON_SPREAD

AGENT_FOR_CONDITION = {
    "bare": "random_spread",
    "design": "split_plot_doe",
    "inference": "gp_ucb_profit",
    "both": "constraint_aware_batch_bo",
}
HALF_WIDTH_MULTIPLIERS = (1.0, 1.5, 2.0)
SEED_START = 9300


@contextmanager
def widened_spread(field, half_width_mult):
    old_lo, old_hi = SITE_ECON_SPREAD[field]
    center = (old_lo + old_hi) / 2
    half = (old_hi - old_lo) / 2 * half_width_mult
    SITE_ECON_SPREAD[field] = (center - half, center + half)
    try:
        yield
    finally:
        SITE_ECON_SPREAD[field] = (old_lo, old_hi)


def run_episode(task, agent_name, seed):
    env = SlowLabEnv(task, seed=seed)
    # SITE_ECON_SPREAD changes the site's sampled economic model, but the oracle
    # disk key contains only seed, factor bounds, cycle length and energy shock.
    # Reusing it across width multipliers can compare a perturbed recommendation
    # with an optimum from a different distribution width.
    env.truth._oracle_cacheable = False
    agent = make_agent(agent_name)
    x = agent.run(env)
    r = env.submit_recommendation(x, agent_name, probes=getattr(agent, "probes", None))
    return float(r.simple_regret)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                              "/phase4_site_distribution_sensitivity.json"))
    args = parser.parse_args()
    seeds = list(range(SEED_START, SEED_START + args.seeds))
    task = TASKS["Optimise"]
    old_lo, old_hi = SITE_ECON_SPREAD["price_per_kg_fw"]

    results = {}
    for mult in HALF_WIDTH_MULTIPLIERS:
        with widened_spread("price_per_kg_fw", mult):
            lo, hi = SITE_ECON_SPREAD["price_per_kg_fw"]
            by_cond = defaultdict(list)
            for seed in seeds:
                for condition, agent_name in AGENT_FOR_CONDITION.items():
                    by_cond[condition].append(run_episode(task, agent_name, seed))
        bare = np.asarray(by_cond["bare"], float)
        row = {
            "price_per_kg_fw_range": [lo, hi],
            "mean_regret": {c: float(np.mean(v)) for c, v in by_cond.items()},
            "bare_between_site_sd": float(bare.std(ddof=1)),
        }
        for condition in ("design", "inference", "both"):
            diff = np.asarray(by_cond[condition], float) - bare
            row[f"{condition}_minus_bare"] = {
                "mean": float(diff.mean()),
                "sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
                "sign": "tool_better" if diff.mean() < 0 else "tool_worse_or_equal",
            }
        results[str(mult)] = row
        print(f"half_width_mult={mult}: range=[{lo:.3f},{hi:.3f}] "
              f"bare_sd={row['bare_between_site_sd']:.4f}", flush=True)

    direction_stable = {
        condition: len({results[str(m)][f"{condition}_minus_bare"]["sign"]
                        for m in HALF_WIDTH_MULTIPLIERS}) == 1
        for condition in ("design", "inference", "both")
    }
    sd_growth = {str(m): results[str(m)]["bare_between_site_sd"] / results["1.0"]["bare_between_site_sd"]
                for m in HALF_WIDTH_MULTIPLIERS}

    report = {
        "purpose": "scripted-strategy check of whether the frozen "
                   "price_per_kg_fw site-distribution width -- explicitly flagged "
                   "in economics.py as the weakest-sourced spread -- understates "
                   "between-site heterogeneity relevant to Phase 4's precision plan; "
                   "not a replay or robustness test of the actual LLM conditions",
        "interpretation_limit": ("Each label maps to a different scripted algorithm; "
                                 "only the bare-strategy SD is used as a rough environment "
                                 "diagnostic, not as a re-estimate of the LLM paired SD."),
        "task": "Optimise (T3)",
        "n_seeds": len(seeds),
        "seed_range": [SEED_START, SEED_START + args.seeds - 1],
        "agent_for_condition": AGENT_FOR_CONDITION,
        "frozen_price_per_kg_fw_range": [old_lo, old_hi],
        "half_width_multipliers": list(HALF_WIDTH_MULTIPLIERS),
        "results_by_half_width": results,
        "bare_between_site_sd_growth_vs_1.0x": sd_growth,
        "direction_stable_across_widths": direction_stable,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"bare_between_site_sd_growth_vs_1.0x": sd_growth,
                      "direction_stable_across_widths": direction_stable}, indent=2))


if __name__ == "__main__":
    main()
