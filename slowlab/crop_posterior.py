"""Continuous posterior sampling for crop physiology observations."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from statistics import NormalDist
import numpy as np

from .env import SlowLabEnv
from .facility import Facility
from .history_likelihood import visible_likelihood_records
from .world import (CROP_LATENT_DIMENSION, ManagedTomgro,
                    crop_params_from_latent)
from . import tomgro_params as TP


CROP_MODALITIES = {"canopy_lai", "harvested_fresh_mass"}
_STANDARD_NORMAL = NormalDist()


@dataclass
class CropPosterior:
    latent_coordinates: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    n_records: int
    n_stages: int
    acceptance_rates: tuple[float, ...]

    @property
    def effective_sample_size(self):
        return float(1.0 / np.sum(self.weights ** 2))

    @property
    def parameters(self):
        return [crop_params_from_latent(row) for row in self.latent_coordinates]


def _plant_parameter_batch(latents, n_mc_plant, task, rng):
    parameter_sets = [crop_params_from_latent(row).values for row in latents]
    out = {}
    for key in parameter_sets[0]:
        site_values = np.asarray([params[key] for params in parameter_sets], float)
        values = np.repeat(site_values, n_mc_plant)
        if key in TP.VC14_NOMINAL:
            # Use the same standardised plant draws at every proposed site.
            # Otherwise the likelihood depends on a particle's row index and
            # two identical latent coordinates receive different target values.
            # One-dimensional Latin-hypercube normal draws integrate each
            # parameter's marginal far more steadily than an iid sample of the
            # same size. Independent permutations retain cross-parameter
            # variation; the seed controls only those pairings.
            quantiles = (rng.permutation(n_mc_plant) + 0.5) / n_mc_plant
            standard = np.asarray([
                _STANDARD_NORMAL.inv_cdf(float(value)) for value in quantiles])
            deviations = np.tile(task.plant_cv * standard, len(latents))
            values *= np.exp(deviations)
        out[key] = values
    return out


def crop_history_moments(latent_coordinates, events, task, *,
                         n_mc_plant=16, seed=0):
    """Batch crop-only means and covariances with common random numbers."""
    if n_mc_plant < 2:
        raise ValueError("n_mc_plant must be at least 2")
    latents = np.atleast_2d(np.asarray(latent_coordinates, float))
    records = [record for record in visible_likelihood_records(events)
               if record.modality in CROP_MODALITIES]
    y = np.asarray([record.value for record in records], float)
    P, n = len(latents), len(records)
    means = np.zeros((P, n))
    covariances = np.zeros((P, n, n))
    if not n:
        return y, means, covariances, records

    designs = {event.design_id: event.payload for event in events
               if event.kind == "submitted"}
    by_treatment = defaultdict(list)
    for index, record in enumerate(records):
        by_treatment[(record.design_id, record.treatment)].append(index)
    facility = Facility(task.n_chambers, task.loops_per_chamber)
    rng = np.random.default_rng(seed)
    reference = ManagedTomgro(
        seed=0, factors=task.factors, cycle_days=task.cycle_days)

    samples_by_record = {}
    for (design_id, treatment_id), indices in by_treatment.items():
        treatment = designs[design_id]["treatments"][treatment_id]
        point = np.asarray([treatment[factor.name] for factor in task.factors], float)
        X = np.repeat(point.reshape(1, -1), P * n_mc_plant, axis=0)
        plant_params = _plant_parameter_batch(latents, n_mc_plant, task, rng)
        state = reference.start(X, plant_params=plant_params)
        for day in sorted({records[index].day for index in indices}):
            reference.advance_state(state, day)
            for index in [i for i in indices if records[i].day == day]:
                if records[index].modality == "canopy_lai":
                    values = state.LAI
                else:
                    values = reference.econ.marketable_kg(state.WM)
                samples = np.asarray(values, float).reshape(P, n_mc_plant)
                samples_by_record[index] = samples
                means[:, index] = samples.mean(axis=1)

        density = treatment.get("density", 3.0)
        n_plants = facility.plants_per_unit(density)
        for i in indices:
            ai = samples_by_record[i] - means[:, i, None]
            for j in indices:
                if records[i].unit_id != records[j].unit_id:
                    continue
                bj = samples_by_record[j] - means[:, j, None]
                covariances[:, i, j] += np.sum(ai * bj, axis=1) / (
                    (n_mc_plant - 1) * n_plants)

    for i, ri in enumerate(records):
        spec = SlowLabEnv._MEASUREMENT_SPECS[ri.modality]
        for j, rj in enumerate(records):
            if (ri.design_id == rj.design_id and ri.unit_id == rj.unit_id
                    and ri.modality == rj.modality):
                second_moment = (covariances[:, i, j]
                                 + means[:, i] * means[:, j])
                covariances[:, i, j] += spec["bias_sd"] ** 2 * second_moment
        second_moment = covariances[:, i, i] + means[:, i] ** 2
        resolution = 10.0 ** (-int(spec["decimals"]))
        covariances[:, i, i] += (
            spec["record_sd"] ** 2 * second_moment + resolution ** 2 / 12.0)
    covariances += 1e-12 * np.eye(n)[None, :, :]
    return y, means, covariances, records


def _log_likelihood(latents, events, task, n_mc_plant, seed):
    y, means, covariances, records = crop_history_moments(
        latents, events, task, n_mc_plant=n_mc_plant, seed=seed)
    if not len(y):
        return np.zeros(len(np.atleast_2d(latents))), records
    out = np.empty(len(means))
    for index, (mean, covariance) in enumerate(zip(means, covariances)):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("crop-history covariance is not positive definite")
        residual = y - mean
        out[index] = -.5 * (
            residual @ np.linalg.solve(covariance, residual) + logdet)
    return out, records


def _reflect_unit(values):
    values = np.mod(values, 2.0)
    return np.where(values <= 1.0, values, 2.0 - values)


def _systematic_resample(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def sample_crop_posterior(events, task, *, n_particles=128, seed=0,
                          n_mc_plant=16, likelihood_seed=0, ess_fraction=0.7,
                          mcmc_steps=5, max_stages=120):
    """Annealed SMC under the original mixed Gaussian/uniform crop prior."""
    if n_particles < 8:
        raise ValueError("n_particles must be at least 8")
    events = list(events)
    rng = np.random.default_rng(seed)
    particles = np.column_stack([
        rng.normal(size=(n_particles, 4)),
        rng.random((n_particles, CROP_LATENT_DIMENSION - 4)),
    ])
    loglik, records = _log_likelihood(
        particles, events, task, n_mc_plant, likelihood_seed)
    weights = np.full(n_particles, 1.0 / n_particles)
    if not records:
        return CropPosterior(particles, weights, loglik, 0, 0, ())

    beta, step_normal, step_unit = 0.0, 0.18, 0.10
    target_ess = float(ess_fraction) * n_particles
    acceptance, stages = [], 0
    while beta < 1.0 - 1e-12:
        remaining = 1.0 - beta

        def ess_for(delta):
            z = np.log(weights) + delta * loglik
            z -= z.max(); w = np.exp(z); w /= w.sum()
            return 1.0 / np.sum(w * w)

        if ess_for(remaining) >= target_ess:
            delta = remaining
        else:
            lo, hi = 0.0, remaining
            for _ in range(45):
                mid = (lo + hi) / 2.0
                if ess_for(mid) < target_ess:
                    hi = mid
                else:
                    lo = mid
            delta = max(lo, np.finfo(float).eps)
        beta = min(1.0, beta + delta)
        z = np.log(weights) + delta * loglik
        z -= z.max(); weights = np.exp(z); weights /= weights.sum()

        stages += 1
        final_stage = beta >= 1.0 - 1e-12

        indices = _systematic_resample(weights, rng)
        particles, loglik = particles[indices], loglik[indices]
        weights.fill(1.0 / n_particles)
        accepted = attempted = 0
        # The final resampling needs more moves because duplicates count as
        # separate rows but contain no posterior diversity. Continue adapting
        # until narrow beta=1 proposals can separate them.
        stage_steps = int(mcmc_steps) * (10 if final_stage else 2)
        for _ in range(stage_steps):
            proposal = particles.copy()
            # Update one coordinate per particle. Simultaneously moving all ten
            # coordinates made acceptance the product of several narrow
            # directions and left resampled duplicates untouched.
            dimensions = rng.integers(0, CROP_LATENT_DIMENSION, n_particles)
            rows = np.arange(n_particles)
            normal = dimensions < 4
            proposal[rows[normal], dimensions[normal]] += rng.normal(
                0.0, step_normal, int(normal.sum()))
            bounded = ~normal
            proposal[rows[bounded], dimensions[bounded]] = _reflect_unit(
                proposal[rows[bounded], dimensions[bounded]]
                + rng.normal(0.0, step_unit, int(bounded.sum())))
            proposed_ll, _ = _log_likelihood(
                proposal, events, task, n_mc_plant, likelihood_seed)
            log_prior_change = -.5 * (
                np.sum(proposal[:, :4] ** 2, axis=1)
                - np.sum(particles[:, :4] ** 2, axis=1))
            take = np.log(rng.random(n_particles)) < (
                beta * (proposed_ll - loglik) + log_prior_change)
            particles[take], loglik[take] = proposal[take], proposed_ll[take]
            sweep_rate = float(take.mean())
            accepted += int(take.sum()); attempted += n_particles
            # The likelihood becomes much narrower near beta=1. Adapt between
            # sweeps so the final rejuvenation does not spend every move at a
            # step size inherited from a much flatter intermediate target.
            if sweep_rate < 0.15:
                step_normal *= 0.7; step_unit *= 0.7
            elif sweep_rate > 0.4:
                step_normal *= 1.2; step_unit *= 1.2
            step_normal = float(np.clip(step_normal, 1e-4, 0.8))
            step_unit = float(np.clip(step_unit, 1e-5, 0.35))
        rate = accepted / attempted
        acceptance.append(rate)
        if final_stage:
            break
        if stages >= max_stages:
            raise RuntimeError(
                f"crop posterior SMC did not reach beta=1 after {max_stages} stages")
    return CropPosterior(
        particles, weights.copy(), loglik, len(records), stages,
        tuple(float(value) for value in acceptance))
