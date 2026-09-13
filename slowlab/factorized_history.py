"""Product posterior for independent crop physiology and site economics."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .economic_posterior import (ECONOMIC_MODALITIES, EconomicPosterior,
                                 sample_economic_posterior)
from .eig import AtomSet
from .history_likelihood import (atom_history_moments,
                                 posterior_weights_from_moments)
from .world import ManagedTomgro


@dataclass
class FactorizedHistoryPosterior:
    atoms: AtomSet
    weights: np.ndarray
    crop_indices: np.ndarray
    economic_indices: np.ndarray
    crop_marginal: np.ndarray
    economic_marginal: np.ndarray
    records: tuple
    economic_posterior: EconomicPosterior

    @property
    def effective_sample_size(self):
        return float(1.0 / np.sum(self.weights ** 2))

    @property
    def crop_effective_sample_size(self):
        return float(1.0 / np.sum(self.crop_marginal ** 2))

    @property
    def economic_effective_sample_size(self):
        return float(1.0 / np.sum(self.economic_marginal ** 2))


def factorized_history_posterior(
        events, task, *, crop_seeds, n_economic_particles=96,
        n_economic_product=32, economic_seed=0, likelihood_seed=0,
        n_mc_plant=64, mcmc_steps=5):
    """Condition a crop-by-economics product approximation on visible history.

    Cost-only fields first update the continuous economic coordinates by SMC.
    The remaining fields then update the Cartesian product. This is an exact
    factorisation of the likelihood because cost fields do not depend on crop
    physiology. Revenue can still couple crop yield and price in the second
    update, while canopy and harvested mass update crop physiology.
    """
    crop_seeds = tuple(int(seed) for seed in crop_seeds)
    if not crop_seeds:
        raise ValueError("crop_seeds must not be empty")
    if not 1 <= n_economic_product <= n_economic_particles:
        raise ValueError("n_economic_product must be between 1 and n_economic_particles")
    events = list(events)
    econ_post = sample_economic_posterior(
        events, task, n_particles=n_economic_particles, seed=economic_seed,
        mcmc_steps=mcmc_steps)

    rng = np.random.default_rng(economic_seed + 8_191)
    chosen = rng.choice(
        n_economic_particles, size=n_economic_product, replace=False,
        p=econ_post.weights)
    econ_models = [econ_post.models[index] for index in chosen]
    econ_prior = econ_post.weights[chosen]
    econ_prior = econ_prior / econ_prior.sum()

    worlds, crop_indices, economic_indices = [], [], []
    for crop_index, crop_seed in enumerate(crop_seeds):
        for economic_index, econ in enumerate(econ_models):
            worlds.append(ManagedTomgro(
                seed=crop_seed, econ=econ, factors=task.factors,
                cycle_days=task.cycle_days))
            crop_indices.append(crop_index)
            economic_indices.append(economic_index)
    crop_indices = np.asarray(crop_indices, int)
    economic_indices = np.asarray(economic_indices, int)
    atoms = AtomSet(
        worlds=worlds, xstar=np.zeros((len(worlds), task.d)),
        cell=np.zeros(len(worlds), int), n_cells=1, nbins=1,
        task_name=task.name)

    y, means, covariances, records = atom_history_moments(
        atoms, events, task, n_mc_plant=n_mc_plant, seed=likelihood_seed)
    retained = np.asarray([
        index for index, record in enumerate(records)
        if record.modality not in ECONOMIC_MODALITIES], int)
    prior = np.asarray([
        econ_prior[economic_index] / len(crop_seeds)
        for economic_index in economic_indices], float)
    if len(retained):
        weights = posterior_weights_from_moments(
            y[retained], means[:, retained],
            covariances[:, retained[:, None], retained], prior=prior)
    else:
        weights = prior / prior.sum()
    crop_marginal = np.bincount(
        crop_indices, weights=weights, minlength=len(crop_seeds))
    economic_marginal = np.bincount(
        economic_indices, weights=weights, minlength=n_economic_product)
    return FactorizedHistoryPosterior(
        atoms=atoms, weights=weights, crop_indices=crop_indices,
        economic_indices=economic_indices, crop_marginal=crop_marginal,
        economic_marginal=economic_marginal, records=tuple(records),
        economic_posterior=econ_post)
