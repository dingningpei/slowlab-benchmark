"""Product posterior for independent crop physiology and site economics."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .crop_posterior import (CROP_MODALITIES, CropPosterior,
                             sample_crop_posterior)
from .economic_posterior import (ECONOMIC_MODALITIES, EconomicPosterior,
                                 sample_economic_posterior)
from .eig import AtomSet
from .history_likelihood import (atom_history_moments,
                                 posterior_weights_from_moments)
from .world import MANAGEMENT_FACTORS, ManagedTomgro


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
    crop_posterior: CropPosterior
    crop_parameters: tuple
    economic_models: tuple

    @property
    def effective_sample_size(self):
        return float(1.0 / np.sum(self.weights ** 2))

    @property
    def crop_effective_sample_size(self):
        return float(1.0 / np.sum(self.crop_marginal ** 2))

    @property
    def economic_effective_sample_size(self):
        return float(1.0 / np.sum(self.economic_marginal ** 2))

    def mean_response(self, points):
        """Posterior-world responses without repeating crop simulation per economy."""
        points = np.atleast_2d(np.asarray(points, float))
        template = ManagedTomgro(
            seed=0, params=self.crop_parameters[0],
            econ=self.economic_models[0], factors=self.atoms.worlds[0].factors,
            cycle_days=self.atoms.worlds[0].cycle_days)
        physical = template._split(points)
        for factor in MANAGEMENT_FACTORS:
            physical.setdefault(factor.name, np.full(len(points), factor.denorm(0.5)))
        days = np.full(len(points), template.cycle_days)

        marketable_rate = []
        for params in self.crop_parameters:
            crop_world = ManagedTomgro(
                seed=0, params=params, econ=self.economic_models[0],
                factors=template.factors, cycle_days=template.cycle_days)
            state = crop_world.start(points)
            crop_world.advance_state(state, int(state.end.max()))
            kg = crop_world.econ.marketable_kg(state.WM_final)
            marketable_rate.append(np.asarray(kg, float) / days)
        marketable_rate = np.asarray(marketable_rate)

        output = np.empty((len(self.crop_parameters),
                           len(self.economic_models), len(points)))
        for index, econ in enumerate(self.economic_models):
            cost = (econ.energy_cost(
                physical["day_temp"], physical["night_temp"], physical["par"])
                + econ.other_cost(
                    physical["co2"], physical["day_temp"],
                    physical["night_temp"], physical["density"], days))
            coefficient = econ.price_per_kg_fw - econ.harvest_labour_per_kg
            output[:, index, :] = marketable_rate * coefficient - cost
        return output.reshape(len(self.atoms.worlds), len(points))


def factorized_history_posterior(
        events, task, *, n_crop_particles=96, n_crop_product=32,
        crop_seed=0, crop_likelihood_seed=0,
        n_economic_particles=96, n_economic_product=32,
        economic_seed=0, likelihood_seed=0, n_mc_plant=64,
        mcmc_steps=5):
    """Condition a crop-by-economics product approximation on visible history.

    Cost-only fields first update the continuous economic coordinates by SMC.
    The remaining fields then update the Cartesian product. This is an exact
    factorisation of the likelihood because cost fields do not depend on crop
    physiology. Revenue can still couple crop yield and price in the second
    update, while canopy and harvested mass update crop physiology.
    """
    if not 1 <= n_crop_product <= n_crop_particles:
        raise ValueError("n_crop_product must be between 1 and n_crop_particles")
    if not 1 <= n_economic_product <= n_economic_particles:
        raise ValueError("n_economic_product must be between 1 and n_economic_particles")
    events = list(events)
    econ_post = sample_economic_posterior(
        events, task, n_particles=n_economic_particles, seed=economic_seed,
        mcmc_steps=mcmc_steps)
    crop_post = sample_crop_posterior(
        events, task, n_particles=n_crop_particles, seed=crop_seed,
        n_mc_plant=n_mc_plant, likelihood_seed=crop_likelihood_seed,
        mcmc_steps=mcmc_steps)

    rng = np.random.default_rng(economic_seed + 8_191)
    chosen_economic = rng.choice(
        n_economic_particles, size=n_economic_product, replace=True,
        p=econ_post.weights)
    chosen_crop = rng.choice(
        n_crop_particles, size=n_crop_product, replace=True,
        p=crop_post.weights)
    econ_models = [econ_post.models[index] for index in chosen_economic]
    crop_params = [crop_post.parameters[index] for index in chosen_crop]

    worlds, crop_indices, economic_indices = [], [], []
    for crop_index, params in enumerate(crop_params):
        for economic_index, econ in enumerate(econ_models):
            worlds.append(ManagedTomgro(
                seed=1_000_000 + crop_index, econ=econ, params=params,
                factors=task.factors,
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
        if (record.modality not in ECONOMIC_MODALITIES
            and record.modality not in CROP_MODALITIES)], int)
    prior = np.full(len(worlds), 1.0 / len(worlds))
    if len(retained):
        weights = posterior_weights_from_moments(
            y[retained], means[:, retained],
            covariances[:, retained[:, None], retained], prior=prior)
    else:
        weights = prior / prior.sum()
    crop_marginal = np.bincount(
        crop_indices, weights=weights, minlength=n_crop_product)
    economic_marginal = np.bincount(
        economic_indices, weights=weights, minlength=n_economic_product)
    return FactorizedHistoryPosterior(
        atoms=atoms, weights=weights, crop_indices=crop_indices,
        economic_indices=economic_indices, crop_marginal=crop_marginal,
        economic_marginal=economic_marginal, records=tuple(records),
        economic_posterior=econ_post, crop_posterior=crop_post,
        crop_parameters=tuple(crop_params), economic_models=tuple(econ_models))
