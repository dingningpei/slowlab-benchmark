#!/usr/bin/env python3
"""Calibration suite for the v2 history-conditioned evaluator."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.design import Design
from slowlab.eig import AtomSet
from slowlab.env import SlowLabEnv
from slowlab.history_likelihood import posterior_weights_from_moments, posterior_weights_full_history
from slowlab.history_risk import conditional_design_value_full_history
from slowlab.tasks import TASKS
from slowlab.world import ManagedTomgro


def gaussian_toy(n_sim=200_000, seed=0):
    """Two atoms, two actions, y~N(theta,1): exact Bayes error is analytic."""
    means = np.array([[-1.0], [1.0]])
    cov = np.array([[[1.0]], [[1.0]]])
    grid = np.linspace(-3, 3, 61)
    max_weight_error = 0.0
    for y in grid:
        got = posterior_weights_from_moments([y], means, cov)
        exact_right = 1.0 / (1.0 + math.exp(-2.0 * y))
        max_weight_error = max(max_weight_error, abs(got[1] - exact_right))
    exact_risk = 4.0 * 0.5 * math.erfc(1.0 / math.sqrt(2.0))
    rng = np.random.default_rng(seed)
    theta = rng.choice([-1.0, 1.0], n_sim)
    y = theta + rng.normal(size=n_sim)
    decision = np.where(y >= 0, 1.0, -1.0)
    loss = (decision - theta) ** 2
    mc_risk = float(loss.mean())
    mc_se = float(loss.std(ddof=1) / math.sqrt(n_sim))
    return {
        "max_posterior_weight_error": float(max_weight_error),
        "exact_bayes_risk": exact_risk, "mc_bayes_risk": mc_risk,
        "mc_se": mc_se, "z_error": float((mc_risk - exact_risk) / mc_se),
        "n_sim": int(n_sim), "seed": int(seed),
    }


def sanity_history(seed=7):
    task = copy.deepcopy(TASKS["Sanity"])
    env = SlowLabEnv(task, seed=seed)
    treatments = {"cool": {"day_temp": 0.3}, "warm": {"day_temp": 0.7}}
    allocation = {
        "cool": [u.id for u in env.facility.units if u.chamber == 0][:4],
        "warm": [u.id for u in env.facility.units if u.chamber == 1][:4],
    }
    env.submit_design(Design(treatments, allocation, randomization_seed=19))
    for day in (60, 120):
        env.advance_to(day)
        for modality in ("canopy_lai", "harvested_fresh_mass", "energy_cost_to_date"):
            env.observe(modality=modality)
    env.advance()
    return task, env.history()


def _atoms(task, M):
    worlds, optima = [], []
    for index in range(M):
        world = ManagedTomgro(seed=10_000 + index, factors=task.factors,
                              cycle_days=task.cycle_days)
        optimum, _ = world.oracle()
        worlds.append(world)
        optima.append(optimum)
    return AtomSet(worlds, np.asarray(optima), np.zeros(M, int), 1, 12, task.name)


def _subatoms(atoms, indices):
    indices = np.asarray(indices, int)
    return AtomSet([atoms.worlds[i] for i in indices], atoms.xstar[indices],
                   np.zeros(len(indices), int), 1, atoms.nbins, atoms.task_name)


def _ess(weights):
    return float(1.0 / np.sum(np.asarray(weights) ** 2))


def sanity_calibration(M=16):
    task, history = sanity_history()
    atoms = _atoms(task, M)
    reference, _ = posterior_weights_full_history(
        atoms, history, task, n_mc_plant=512, seed=999, temperature=1.0)
    grid = np.linspace(0, 1, 2001)[:, None]
    response = atoms.mean_response(grid)
    reference_curve = reference @ response
    best = np.array([world.oracle()[1] for world in atoms.worlds])
    regret = best[:, None] - response

    plant_mc = []
    for n_mc in (16, 32, 64, 128):
        for seed in (0, 1, 2):
            weights, _ = posterior_weights_full_history(
                atoms, history, task, n_mc_plant=n_mc, seed=seed)
            plant_mc.append({
                "n_mc_plant": n_mc, "seed": seed,
                "total_variation_to_reference": float(0.5 * np.abs(weights - reference).sum()),
                "posterior_curve_rmse": float(np.sqrt(np.mean((weights @ response - reference_curve) ** 2))),
                "ess": _ess(weights), "max_weight": float(weights.max()),
                "top_atom_matches_reference": bool(np.argmax(weights) == np.argmax(reference)),
            })

    temperature = []
    for value in (1.0, 3.0, 10.0, 30.0, 100.0):
        weights, _ = posterior_weights_full_history(
            atoms, history, task, n_mc_plant=128, seed=0, temperature=value)
        temperature.append({"temperature": value, "ess": _ess(weights),
                            "max_weight": float(weights.max())})

    atom_count = []
    for count in (4, 8, M):
        subset = _subatoms(atoms, np.arange(count))
        weights, _ = posterior_weights_full_history(
            subset, history, task, n_mc_plant=128, seed=0)
        curve = weights @ subset.mean_response(grid)
        atom_count.append({"M": count, "ess": _ess(weights),
                           "curve_rmse_to_Mmax_reference": float(np.sqrt(np.mean((curve - reference_curve) ** 2)))})

    candidate_resolution = []
    for count in (101, 501, 2001):
        idx = np.linspace(0, len(grid) - 1, count).round().astype(int)
        risk = reference @ regret[:, idx]
        candidate_resolution.append({"n_candidates": count,
                                     "bayes_risk": float(risk.min()),
                                     "bayes_action": float(grid[idx[int(np.argmin(risk))], 0])})

    split = []
    for name, indices in (("even", np.arange(0, M, 2)), ("odd", np.arange(1, M, 2))):
        subset = _subatoms(atoms, indices)
        weights, _ = posterior_weights_full_history(
            subset, history, task, n_mc_plant=128, seed=0)
        curve = weights @ subset.mean_response(grid)
        split.append({"split": name, "M": len(indices), "ess": _ess(weights),
                      "curve_rmse_to_Mmax_reference": float(np.sqrt(np.mean((curve - reference_curve) ** 2)))})

    before = history[:1]
    outer_mc = []
    for n_outer in (50, 100, 200):
        value = conditional_design_value_full_history(
            atoms, before, history, task, cand=grid, n_mc_plant=64,
            likelihood_seed=0, n_outer=n_outer, simulation_seed=31)
        outer_mc.append({"n_outer": n_outer, "conditional_value": value.conditional_value,
                         "mc_error": value.mc_error, "failure_flags": list(value.failure_flags)})

    result = {
        "M_reference": M, "n_mc_reference": 512, "likelihood_seed_reference": 999,
        "reference_ess": _ess(reference), "reference_max_weight": float(reference.max()),
        "plant_mc": plant_mc, "temperature": temperature,
        "atom_count": atom_count, "candidate_resolution": candidate_resolution,
        "atom_split": split, "outer_mc": outer_mc,
    }
    result["calibration_status"] = "failed" if (
        result["reference_ess"] < 2.0
        or max(item["curve_rmse_to_Mmax_reference"] for item in split) > 0.005
    ) else "passed"
    result["failure_flags"] = []
    if result["reference_ess"] < 2.0:
        result["failure_flags"].append("posterior_atom_collapse")
    if max(item["curve_rmse_to_Mmax_reference"] for item in split) > 0.005:
        result["failure_flags"].append("atom_split_instability")
    if min(item["bayes_risk"] for item in candidate_resolution) < -1e-9:
        result["failure_flags"].append("negative_regret_from_oracle_tolerance")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--atoms", type=int, default=16)
    args = parser.parse_args()
    result = {"toy": gaussian_toy(), "sanity": sanity_calibration(args.atoms)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
