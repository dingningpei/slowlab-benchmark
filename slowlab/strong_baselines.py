"""Strong Phase-3 baselines built on the public SlowLab history.

Deployable baselines in this module see the same submitted/completed events as an
LLM.  The prior-fixed and site-oracle agents are labelled privileged diagnostics;
they are scale checks and must never appear as deployable competitors.
"""
from __future__ import annotations

import numpy as np

from .agents import (_cands, _preds_empirical, chamber_safe_allocation,
                     respect_hierarchy)
from .design import Design
from .fixed_reference import best_fixed


BASELINE_PROTOCOL_VERSION = "phase3-baseline-0.1"


def completed_history_arrays(env):
    """Return completed public history with block, batch and noise metadata."""
    designs = {event.design_id: event.payload for event in env.history()
               if event.kind == "submitted"}
    density_factor = next(
        (factor for factor in env.task.factors if factor.name == "density"), None)
    X, y, chambers, batches, observation_noise = [], [], [], [], []
    for event in env.history():
        if event.kind != "completed":
            continue
        submitted = designs[event.design_id]
        treatment = submitted["treatments"][event.payload["treatment"]]
        X.append([treatment[factor.name] for factor in env.task.factors])
        y.append(event.payload["value"])
        chambers.append(env.facility.get(event.payload["unit_id"]).chamber)
        batches.append(event.design_id)
        physical_density = (density_factor.denorm(treatment["density"])
                            if density_factor is not None else 3.0)
        n_plants = env.facility.plants_per_unit(physical_density)
        # Relative scales specified by the public task.  Replication reduces the
        # plant component, while loop noise remains unit-specific.
        observation_noise.append(np.sqrt(
            (env.task.plant_cv / np.sqrt(n_plants)) ** 2 + env.task.tau_loop ** 2))
    return (np.asarray(X, float).reshape(-1, env.task.d), np.asarray(y, float),
            np.asarray(chambers, int), np.asarray(batches, int),
            np.asarray(observation_noise, float))


class BlockAwareGP:
    """RBF reader with persistent chamber and per-batch covariance terms.

    Predictions target a new campaign's population response, so cross-covariance
    to training data contains the response kernel but no old chamber or batch
    intercept.  This prevents shared facility noise from being learned as a
    treatment effect.
    """

    def __init__(self, length_scale=0.28, chamber_scale=0.10, batch_scale=0.05):
        self.length_scale = float(length_scale)
        self.chamber_scale = float(chamber_scale)
        self.batch_scale = float(batch_scale)

    def _kernel(self, A, B):
        distance = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-0.5 * distance / self.length_scale ** 2)

    def fit(self, X, y, chambers, batches, observation_noise):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if len(y) == 0:
            raise ValueError("BlockAwareGP needs at least one completed observation")
        self.X = X
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std(ddof=1)) if len(y) > 1 else 1.0
        if self.y_scale < 1e-10:
            self.y_scale = 1.0
        standardized = (y - self.y_mean) / self.y_scale
        chambers, batches = np.asarray(chambers), np.asarray(batches)
        K = self._kernel(X, X)
        K += self.chamber_scale ** 2 * (chambers[:, None] == chambers[None, :])
        K += self.batch_scale ** 2 * (batches[:, None] == batches[None, :])
        K += np.diag(np.asarray(observation_noise, float) ** 2 + 1e-8)
        self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, standardized))
        self.typical_observation_noise = float(np.median(observation_noise))
        return self

    def predict(self, points):
        points = np.atleast_2d(np.asarray(points, float))
        cross = self._kernel(points, self.X)
        mean = (cross @ self.alpha) * self.y_scale + self.y_mean
        solved = np.linalg.solve(self.L, cross.T)
        variance = np.clip(1.0 - (solved ** 2).sum(axis=0), 1e-9, None)
        return mean, variance * self.y_scale ** 2

    def predict_observation(self, points):
        mean, variance = self.predict(points)
        variance += (self.typical_observation_noise * self.y_scale) ** 2
        return mean, np.sqrt(variance)


class BlockAwareReader:
    """Recommendation-only reader for design x reader replay experiments."""

    name = "block_aware_gp"
    privileged = False

    def fit_history(self, env):
        X, y, chambers, batches, noise = completed_history_arrays(env)
        if not len(y):
            self.model = None
            return self
        self.model = BlockAwareGP(
            chamber_scale=env.task.tau_chamber,
            batch_scale=env.task.tau_batch,
        ).fit(X, y, chambers, batches, noise)
        return self

    def recommend(self, env, candidates=None):
        self.fit_history(env)
        if self.model is None:
            return np.full(env.task.d, 0.5)
        if candidates is None:
            candidates = _cands(env.task.d, 8000, np.random.default_rng(7))
        mean, _ = self.model.predict(candidates)
        return np.asarray(candidates[int(np.argmax(mean))], float)


def block_aware_probe_intervals(env):
    """Population-response intervals from the same block-aware final reader."""
    reader = BlockAwareReader().fit_history(env)
    probes = env.probe_set()
    if reader.model is None:
        intervals = _preds_empirical([f"p{i}" for i in range(len(probes))], np.zeros(0))
        return np.asarray([intervals[f"p{i}"] for i in range(len(probes))], float)
    mean, sd = reader.model.predict_observation(probes)
    return np.column_stack([mean - 1.959964 * sd, mean, mean + 1.959964 * sd])


class ConstraintAwareBatchBOAgent:
    """Batch UCB with feasible split-plot allocation and adaptive replication."""

    name = "constraint_aware_batch_bo"
    privileged = False

    def __init__(self, beta=1.8, max_replicates=4):
        self.beta = float(beta)
        self.max_replicates = int(max_replicates)
        self.replication_history = []
        self.reader = BlockAwareReader()

    def replication_count(self, task):
        # Approximate units needed for a two-sided 95% interval narrower than the
        # task's minimum effect.  The public noise regime drives the choice.
        relative_noise = np.sqrt(task.plant_cv ** 2 + task.tau_loop ** 2)
        needed = int(np.ceil((1.96 * relative_noise / max(task.target_mde, 1e-9)) ** 2))
        return int(np.clip(needed, 1, min(self.max_replicates, task.n_chambers)))

    def recommend(self, env):
        return self.reader.recommend(env)

    def run(self, env):
        task = env.task
        rng = np.random.default_rng(env.seed + 3103)
        while env.rounds_left > 0:
            reps = min(self.replication_count(task), task.units_per_round)
            n_points = max(1, task.units_per_round // reps)
            X, y, chambers, batches, noise = completed_history_arrays(env)
            if len(y) < 2:
                points = rng.random((n_points, task.d))
                model = None
            else:
                model = BlockAwareGP(
                    chamber_scale=task.tau_chamber,
                    batch_scale=task.tau_batch,
                ).fit(X, y, chambers, batches, noise)
                candidates = _cands(task.d, 2500, rng)
                mean, variance = model.predict(candidates)
                order = np.argsort(-(mean + self.beta * np.sqrt(variance)))
                chosen = []
                for index in order:
                    if all(np.linalg.norm(candidates[index] - old) > 0.05
                           for old in chosen):
                        chosen.append(candidates[index])
                    if len(chosen) == n_points:
                        break
                points = np.asarray(chosen, float)
            points = respect_hierarchy(points, task, env.facility, rng)
            treatment_ids = [f"t{i}" for i in range(len(points))]
            treatments = {
                tid: {factor.name: float(points[i, j])
                      for j, factor in enumerate(task.factors)}
                for i, tid in enumerate(treatment_ids)
            }
            allocation = chamber_safe_allocation(
                treatment_ids, reps, env.facility, rng, treatments, task,
                avail=env.available_units())
            if model is None:
                predictions = _preds_empirical(treatment_ids, y)
            else:
                mean, sd = model.predict_observation(points)
                predictions = {tid: (float(m - 1.959964 * s), float(m),
                                     float(m + 1.959964 * s))
                               for tid, m, s in zip(treatment_ids, mean, sd)}
            accepted = env.submit_design(Design(
                treatments, allocation, randomization_seed=int(rng.integers(1_000_000)),
                question="constraint-aware block GP-UCB",
                predictions=predictions))
            if not isinstance(accepted, int):
                break
            self.replication_history.append(reps)
            env.advance()
            env.interim_recommendation(self.recommend(env))
        self.probes = block_aware_probe_intervals(env)
        return self.recommend(env)


class PriorOptimalFixedAgent:
    """Prior-optimal constant; privileged scale diagnostic, not deployable."""

    name = "prior_optimal_fixed"
    privileged = True
    privilege_reason = "uses the benchmark site prior"

    def run(self, env):
        point, _ = best_fixed(env.task)
        return point


class SiteOracleAgent:
    """True-site optimum; upper-bound diagnostic, never a deployable agent."""

    name = "site_oracle"
    privileged = True
    privilege_reason = "reads the latent site and forward-model oracle"

    def run(self, env):
        return np.asarray(env.truth.oracle()[0], float)
