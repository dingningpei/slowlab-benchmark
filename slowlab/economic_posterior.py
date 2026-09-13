"""Continuous posterior sampling for the economic part of a visible history."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .economics import SITE_ECON_KEYS, site_econ_from_unit
from .env import SlowLabEnv
from .history_likelihood import visible_likelihood_records
from .world import MANAGEMENT_FACTORS


ECONOMIC_MODALITIES = {
    "energy_cost_to_date", "other_cost_to_date",
    "terminal_energy_cost_rate", "terminal_other_cost_rate",
}


@dataclass
class EconomicPosterior:
    unit_coordinates: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    n_records: int
    n_stages: int
    acceptance_rates: tuple[float, ...]

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    @property
    def models(self):
        return [site_econ_from_unit(row) for row in self.unit_coordinates]


def _physical_treatment(treatment, task):
    supplied = {factor.name: factor.denorm(treatment[factor.name])
                for factor in task.factors}
    for factor in MANAGEMENT_FACTORS:
        supplied.setdefault(factor.name, factor.denorm(0.5))
    return supplied


def economic_history_moments(unit_coordinates, events, task):
    """Return cost observations and their moments for continuous econ sites."""
    records = [record for record in visible_likelihood_records(events)
               if record.modality in ECONOMIC_MODALITIES]
    y = np.asarray([record.value for record in records], float)
    U = np.atleast_2d(np.asarray(unit_coordinates, float))
    means = np.zeros((len(U), len(records)))
    covariances = np.zeros((len(U), len(records), len(records)))
    designs = {event.design_id: event.payload for event in events
               if event.kind == "submitted"}
    physical = {}
    for record in records:
        key = record.design_id, record.treatment
        if key not in physical:
            physical[key] = _physical_treatment(
                designs[record.design_id]["treatments"][record.treatment], task)

    for m, row in enumerate(U):
        econ = site_econ_from_unit(row)
        for i, record in enumerate(records):
            x = physical[(record.design_id, record.treatment)]
            if "energy" in record.modality:
                rate = float(econ.energy_cost(x["day_temp"], x["night_temp"], x["par"]))
            else:
                rate = float(econ.other_cost(
                    x["co2"], x["day_temp"], x["night_temp"], x["density"],
                    task.cycle_days))
            means[m, i] = rate * record.day if record.kind == "measured" else rate

        for i, ri in enumerate(records):
            if ri.kind == "completed":
                resolution = 10.0 ** (-SlowLabEnv.TERMINAL_RATE_DECIMALS)
                covariances[m, i, i] += resolution ** 2 / 12.0
                continue
            spec = SlowLabEnv._MEASUREMENT_SPECS[ri.modality]
            for j, rj in enumerate(records):
                if (rj.kind == "measured" and ri.design_id == rj.design_id
                        and ri.unit_id == rj.unit_id and ri.modality == rj.modality):
                    covariances[m, i, j] += (
                        spec["bias_sd"] ** 2 * means[m, i] * means[m, j])
            covariances[m, i, i] += (
                spec["record_sd"] ** 2 * means[m, i] ** 2
                + 10.0 ** (-2 * int(spec["decimals"])) / 12.0)
        covariances[m] += 1e-14 * np.eye(len(records))
    return y, means, covariances, records


def _log_likelihood(unit_coordinates, events, task):
    y, means, covariances, records = economic_history_moments(
        unit_coordinates, events, task)
    if not len(y):
        return np.zeros(len(np.atleast_2d(unit_coordinates))), records
    out = np.empty(len(means))
    for m, (mean, covariance) in enumerate(zip(means, covariances)):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("economic-history covariance is not positive definite")
        residual = y - mean
        out[m] = -.5 * (residual @ np.linalg.solve(covariance, residual) + logdet)
    return out, records


def _reflect_unit(values):
    values = np.mod(values, 2.0)
    return np.where(values <= 1.0, values, 2.0 - values)


def _systematic_resample(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def sample_economic_posterior(events, task, *, n_particles=256, seed=0,
                              ess_fraction=0.7, mcmc_steps=5,
                              max_stages=120):
    """Annealed SMC over the exact six-dimensional economic prior.

    Adaptive temperatures prevent a narrow accounting likelihood from assigning
    all mass to one prior atom. After each resampling step, reflected random-walk
    Metropolis moves restore diversity while preserving the bounded uniform
    prior. Diagnostics are returned so calibration can reject an unresolved run.
    """
    if n_particles < 8:
        raise ValueError("n_particles must be at least 8")
    events = list(events)
    rng = np.random.default_rng(seed)
    dimension = len(SITE_ECON_KEYS) + 1
    particles = rng.random((int(n_particles), dimension))
    loglik, records = _log_likelihood(particles, events, task)
    if not records:
        weights = np.full(n_particles, 1.0 / n_particles)
        return EconomicPosterior(particles, weights, loglik, 0, 0, ())

    weights = np.full(n_particles, 1.0 / n_particles)
    beta, step = 0.0, 0.12
    acceptance, stages = [], 0
    target_ess = float(ess_fraction) * n_particles
    while beta < 1.0 - 1e-12:
        remaining = 1.0 - beta

        def ess_for(delta):
            z = np.log(weights) + delta * loglik
            z -= z.max()
            w = np.exp(z); w /= w.sum()
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

        indices = _systematic_resample(weights, rng)
        particles, loglik = particles[indices], loglik[indices]
        weights.fill(1.0 / n_particles)
        accepted = attempted = 0
        for _ in range(int(mcmc_steps)):
            proposal = _reflect_unit(particles + rng.normal(0.0, step, particles.shape))
            proposed_ll, _ = _log_likelihood(proposal, events, task)
            take = np.log(rng.random(n_particles)) < beta * (proposed_ll - loglik)
            particles[take], loglik[take] = proposal[take], proposed_ll[take]
            accepted += int(take.sum()); attempted += n_particles
        rate = accepted / attempted
        acceptance.append(rate)
        if rate < 0.15:
            step *= 0.55
        elif rate > 0.4:
            step *= 1.35
        step = float(np.clip(step, 1e-5, 0.35))
        stages += 1
        if stages >= max_stages:
            raise RuntimeError(f"economic posterior SMC did not reach beta=1 after {max_stages} stages")

    return EconomicPosterior(
        particles, weights.copy(), loglik, len(records), stages,
        tuple(float(value) for value in acceptance))
