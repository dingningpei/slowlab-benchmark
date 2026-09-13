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
from slowlab.crop_posterior import (crop_history_moments,
                                    sample_crop_posterior)
from slowlab.economic_posterior import (economic_history_moments,
                                        sample_economic_posterior)
from slowlab.eig import AtomSet
from slowlab.env import SlowLabEnv
from slowlab.factorized_history import factorized_history_posterior
from slowlab.history_likelihood import (atom_history_moments,
                                        posterior_weights_from_moments,
                                        posterior_weights_full_history)
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
    best = np.maximum(np.array([world.oracle()[1] for world in atoms.worlds]),
                      response.max(axis=1))
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

    y_diag, means_diag, cov_diag, records_diag = atom_history_moments(
        atoms, history, task, n_mc_plant=128, seed=0)
    groups = {
        "within_cycle": [i for i, r in enumerate(records_diag) if r.kind == "measured"],
        "terminal_revenue": [i for i, r in enumerate(records_diag)
                             if r.modality == "terminal_revenue_rate"],
        "terminal_costs": [i for i, r in enumerate(records_diag)
                           if r.modality in {"terminal_energy_cost_rate", "terminal_other_cost_rate"}],
        "all_fields": list(range(len(records_diag))),
    }
    field_ablation = {}
    for name, indices in groups.items():
        idx = np.asarray(indices, int)
        items = []
        for value in (1.0, 100.0, 10_000.0, 1_000_000.0, 100_000_000.0):
            weights = posterior_weights_from_moments(
                y_diag[idx], means_diag[:, idx], cov_diag[:, idx[:, None], idx],
                temperature=value)
            items.append({"temperature": value, "ess": _ess(weights),
                          "max_weight": float(weights.max())})
        field_ablation[name] = items

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
        "field_ablation": field_ablation,
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


def economic_smc_calibration(n_particles=128):
    """Check the continuous economic update on independent SMC runs."""
    task, history = sanity_history()
    runs = []
    posterior_means = []
    for seed in (0, 1, 2):
        posterior = sample_economic_posterior(
            history, task, n_particles=n_particles, seed=seed, mcmc_steps=5)
        y, means, covariances, records = economic_history_moments(
            posterior.unit_coordinates, history, task)
        predictive = posterior.weights @ means
        predictive_variance = (posterior.weights @ (
            np.diagonal(covariances, axis1=1, axis2=2) + means ** 2)
            - predictive ** 2)
        standardised_residual = ((y - predictive)
                                 / np.sqrt(np.maximum(predictive_variance, 1e-20)))
        terminal = np.asarray([record.kind == "completed" for record in records])
        posterior_means.append(predictive)
        runs.append({
            "seed": seed, "n_particles": n_particles,
            "n_stages": posterior.n_stages,
            "min_acceptance": float(min(posterior.acceptance_rates)),
            "median_acceptance": float(np.median(posterior.acceptance_rates)),
            "n_unique_particles": int(np.unique(
                np.round(posterior.unit_coordinates, 8), axis=0).shape[0]),
            "max_terminal_absolute_error": float(
                np.max(np.abs(predictive[terminal] - y[terminal]))),
            "max_interim_absolute_error": float(
                np.max(np.abs(predictive[~terminal] - y[~terminal]))),
            "max_absolute_standardised_residual": float(
                np.max(np.abs(standardised_residual))),
        })
    posterior_means = np.asarray(posterior_means)
    between_run_max_range = float(np.max(np.ptp(posterior_means, axis=0)))
    passed = (all(run["n_unique_particles"] >= n_particles // 2
                  and run["max_terminal_absolute_error"] <= 5e-5
                  and run["max_absolute_standardised_residual"] <= 3.0
                  for run in runs)
              and between_run_max_range <= 0.05)
    return {
        "status": "passed" if passed else "failed",
        "runs": runs,
        "between_run_max_predictive_range": between_run_max_range,
        "note": ("This validates only the economic posterior update. Full risk "
                 "calibration still requires its product with crop particles."),
    }


def crop_smc_calibration(n_particles=128):
    """Check continuous crop updating on three independent SMC runs."""
    task, history = sanity_history()
    runs = []
    posterior_means = []
    for seed in (0, 1, 2):
        posterior = sample_crop_posterior(
            history, task, n_particles=n_particles, seed=seed,
            n_mc_plant=16, likelihood_seed=991, mcmc_steps=5)
        y, means, covariances, _ = crop_history_moments(
            posterior.latent_coordinates, history, task,
            n_mc_plant=16, seed=991)
        predictive = posterior.weights @ means
        predictive_variance = (posterior.weights @ (
            np.diagonal(covariances, axis1=1, axis2=2) + means ** 2)
            - predictive ** 2)
        standardised = ((y - predictive)
                        / np.sqrt(np.maximum(predictive_variance, 1e-20)))
        posterior_means.append(predictive)
        runs.append({
            "seed": seed, "n_particles": n_particles,
            "n_stages": posterior.n_stages,
            "final_stage_acceptance": posterior.acceptance_rates[-1],
            "n_unique_particles": int(np.unique(
                np.round(posterior.latent_coordinates, 8), axis=0).shape[0]),
            "max_absolute_standardised_residual": float(
                np.max(np.abs(standardised))),
        })
    posterior_means = np.asarray(posterior_means)
    between_run_max_range = float(np.max(np.ptp(posterior_means, axis=0)))
    passed = (all(run["n_unique_particles"] >= n_particles // 2
                  and run["max_absolute_standardised_residual"] <= 3.0
                  for run in runs)
              and between_run_max_range <= 0.25)
    return {
        "status": "passed" if passed else "failed",
        "runs": runs,
        "between_run_max_predictive_range": between_run_max_range,
        "note": ("This calibrates canopy and harvested-mass updating. Revenue "
                 "coupling is tested in the product posterior separately."),
    }


def factorized_product_calibration():
    """Locate residual collapse after cost evidence is factorised."""
    task, history = sanity_history()
    runs = []
    for seed in (0, 1):
        posterior = factorized_history_posterior(
            history, task, crop_seeds=range(10_000, 10_008),
            n_economic_particles=96, n_economic_product=16,
            economic_seed=seed, likelihood_seed=20 + seed,
            n_mc_plant=16, mcmc_steps=4)
        runs.append({
            "seed": seed,
            "joint_ess": posterior.effective_sample_size,
            "crop_marginal_ess": posterior.crop_effective_sample_size,
            "economic_marginal_ess": posterior.economic_effective_sample_size,
            "max_crop_weight": float(posterior.crop_marginal.max()),
        })
    passed = all(run["crop_marginal_ess"] >= 2.0 for run in runs)
    return {
        "status": "passed" if passed else "failed",
        "runs": runs,
        "failure_flags": ([] if passed else ["crop_particle_collapse"]),
        "note": ("Cost evidence is updated continuously before the product. "
                 "Failure here identifies the remaining discrete crop-prior approximation."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--atoms", type=int, default=16)
    args = parser.parse_args()
    result = {"toy": gaussian_toy(), "sanity": sanity_calibration(args.atoms),
              "economic_smc": economic_smc_calibration(),
              "crop_smc": crop_smc_calibration(),
              "factorized_product": factorized_product_calibration()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
