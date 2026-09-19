#!/usr/bin/env python3
"""Phase 4, section 4.4: external-validity sensitivity for price and ambient-light
assumptions.

`slowlab/economics.py` is explicit that its coefficients are "representative values
for European markets and equipment, not measurements from one particular greenhouse"
(produce price cites a real Dutch/Spanish spread; electricity cites an EU average with
unstated national spread; the ambient PAR baseline is a stated approximation of one
climate). This script asks whether the Phase 4 qualitative finding -- a more
sophisticated strategy does not reliably beat a naive one on final simple regret --
survives moving those numbers across their cited or plausible real-world range.

Two independent 1-D sweeps (not a cross product, to keep runtime bounded), each using
the same four scripted-agent stand-ins for the four Phase 4 tool conditions as
`phase4_noise_sensitivity.py`:

    bare -> random_spread, design -> split_plot_doe,
    inference -> gp_ucb_profit, both -> constraint_aware_batch_bo

  price basket (price_per_kg_fw, elec_price, heat_price, co2_price moved together,
  representing a general price-level scenario): 0.85x/1.0x/1.15x, anchored to the
  cited Dutch-vs-Spanish produce-price spread (1.35-1.57 EUR/kg around a 1.50 midpoint,
  ~=0.90x-1.05x) padded outward, since the electricity/heat/CO2 citations do not state
  a comparable cross-country spread.

  ambient light (par_ambient, the natural-baseline PAR the model assumes before any
  supplemental lighting is costed): 0.7x/1.0x/1.3x, a plausible seasonal/latitude swing
  around the stated 18 mol/(m^2.d) baseline -- wider than price because no explicit
  literature range is cited for this one.

This is a scripted-baseline proxy, not a replay of the actual Phase 4 LLM transcripts
(see the same caveat in `phase4_noise_sensitivity.py`). Prices are perturbed by
mutating `env.truth.econ` right after construction and before the agent runs -- the
same mechanism the frozen T4/Transfer task already uses for its energy-price shock
(`Task.transfer_energy_shock` -> `env.truth.econ.energy_shock`), applied here to more
fields and used for a sensitivity sweep rather than a transfer shock.

Uses a fresh, disjoint seed range (9100+), separate from both the frozen Phase 4
confirmatory ranges (6000-6811) and the noise-sensitivity check's range (9000-9019).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

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
PRICE_FIELDS = ("price_per_kg_fw", "elec_price", "heat_price", "co2_price")
PRICE_MULTIPLIERS = (0.85, 1.0, 1.15)
LIGHT_MULTIPLIERS = (0.7, 1.0, 1.3)
SEED_START = 9100


def _perturbed_env(task, seed, econ_overrides):
    env = SlowLabEnv(task, seed=seed)
    for field, mult in econ_overrides.items():
        setattr(env.truth.econ, field, getattr(env.truth.econ, field) * mult)
    # World.oracle() caches its "best" value on disk keyed by
    # (seed, factor bounds, cycle_days, energy_shock) -- NOT by the econ fields this
    # script perturbs (price_per_kg_fw, elec_price, heat_price, co2_price,
    # par_ambient). Left alone, every multiplier after the first would silently read
    # back the first multiplier's oracle value for a given seed, making
    # `best - realized` compare a perturbed realized margin against an unperturbed
    # (or differently-perturbed) reference -- this produced nonsensical results
    # (regret going negative, "optimum" identically frozen across multipliers)
    # before this flag was added. Disabling caching forces oracle() through
    # `_maximise` fresh every call, so it actually sees the mutated econ above.
    env.truth._oracle_cacheable = False
    return env


def run_episode(task, agent_name, seed, econ_overrides):
    env = _perturbed_env(task, seed, econ_overrides)
    agent = make_agent(agent_name)
    x = agent.run(env)
    r = env.submit_recommendation(x, agent_name, probes=getattr(agent, "probes", None))
    return float(r.simple_regret)


def optimum_location(task, seed, econ_overrides):
    env = _perturbed_env(task, seed, econ_overrides)
    agent = make_agent("site_oracle")
    x = agent.run(env)
    return np.asarray(x, float).tolist()


def sweep(name, multipliers, fields, seeds, task):
    optima = {}
    contrasts = {}
    for mult in multipliers:
        overrides = {field: mult for field in fields}
        opt_pts = [optimum_location(task, seed, overrides) for seed in seeds]
        optima[str(mult)] = {"mean_point": np.mean(opt_pts, axis=0).tolist()}
        by_cond = defaultdict(list)
        for seed in seeds:
            for condition, agent_name in AGENT_FOR_CONDITION.items():
                by_cond[condition].append(run_episode(task, agent_name, seed, overrides))
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
        print(f"{name} mult={mult}: done {len(seeds)} seeds x {len(AGENT_FOR_CONDITION)} agents",
              flush=True)
    base = np.asarray(optima[str(1.0 if 1.0 in multipliers else multipliers[len(multipliers)//2])]["mean_point"])
    for mult in multipliers:
        pt = np.asarray(optima[str(mult)]["mean_point"])
        optima[str(mult)]["l2_shift_from_baseline"] = float(np.linalg.norm(pt - base))
    direction_stable = {
        condition: len({contrasts[str(m)][f"{condition}_minus_bare"]["sign"] for m in multipliers}) == 1
        for condition in ("design", "inference", "both")
    }
    return {"multipliers": list(multipliers), "fields_scaled": list(fields),
            "optimum_location": optima, "contrasts_by_multiplier": contrasts,
            "direction_stable_across_multipliers": direction_stable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                              "/phase4_price_sensitivity.json"))
    args = parser.parse_args()
    seeds = list(range(SEED_START, SEED_START + args.seeds))
    task = TASKS["Optimise"]

    report = {
        "purpose": "scripted-strategy ranking under perturbed price and ambient-light "
                   "assumptions; this tests environment behavior, not robustness of the "
                   "Phase 4 LLM tool-condition effect",
        "interpretation_limit": ("Each label maps to a different scripted algorithm, so "
                                 "these contrasts cannot be interpreted as bare/design/"
                                 "inference/both interventions on one LLM."),
        "task": "Optimise (T3)",
        "n_seeds": len(seeds),
        "seed_range": [SEED_START, SEED_START + args.seeds - 1],
        "agent_for_condition": AGENT_FOR_CONDITION,
        "price_basket": sweep("price_basket", PRICE_MULTIPLIERS, PRICE_FIELDS, seeds, task),
        "ambient_light": sweep("ambient_light", LIGHT_MULTIPLIERS, ("par_ambient",), seeds, task),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "price_basket_direction_stable": report["price_basket"]["direction_stable_across_multipliers"],
        "ambient_light_direction_stable": report["ambient_light"]["direction_stable_across_multipliers"],
    }, indent=2))


if __name__ == "__main__":
    main()
