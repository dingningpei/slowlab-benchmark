import numpy as np

from slowlab.crop_posterior import (crop_history_moments,
                                    sample_crop_posterior)
from slowlab.design import Design
from slowlab.env import SlowLabEnv
from slowlab.tasks import TASKS
from slowlab.world import (CROP_LATENT_DIMENSION, crop_params_from_latent,
                           sample_instance_params)


def test_explicit_crop_latent_map_preserves_seeded_prior():
    seed = 37
    rng = np.random.default_rng(20_000 + seed)
    latent = np.concatenate([rng.normal(size=4), rng.random(6)])
    mapped = crop_params_from_latent(latent)
    sampled = sample_instance_params(seed)
    assert CROP_LATENT_DIMENSION == 10
    assert mapped.values == sampled.values
    assert mapped.intervals == sampled.intervals


def _crop_history():
    task = TASKS["Sanity"]
    env = SlowLabEnv(task, seed=7)
    env.submit_design(Design(
        {"cool": {"day_temp": .3}, "warm": {"day_temp": .7}},
        {"cool": ["c0l0"], "warm": ["c1l0"]}, randomization_seed=3))
    for day in (60, 120):
        env.advance_to(day)
        env.observe(modality="canopy_lai")
        env.observe(modality="harvested_fresh_mass")
    return task, env.history()


def test_batched_crop_moments_have_complete_positive_definite_shape():
    task, history = _crop_history()
    rng = np.random.default_rng(4)
    latents = np.column_stack([rng.normal(size=(3, 4)), rng.random((3, 6))])
    y, means, covariances, records = crop_history_moments(
        latents, history, task, n_mc_plant=4, seed=8)
    assert means.shape == (3, len(y))
    assert covariances.shape == (3, len(y), len(y))
    assert len(records) == len(y) == 8
    assert all(np.linalg.eigvalsh(value).min() > 0 for value in covariances)


def test_crop_smc_preserves_particle_diversity_under_crop_measurements():
    task, history = _crop_history()
    posterior = sample_crop_posterior(
        history, task, n_particles=48, seed=5, n_mc_plant=8,
        likelihood_seed=9, mcmc_steps=5)
    assert posterior.n_records == 8
    assert posterior.n_stages > 1
    assert np.isfinite(posterior.log_likelihood).all()
    assert np.unique(np.round(posterior.latent_coordinates, 7), axis=0).shape[0] > 12
