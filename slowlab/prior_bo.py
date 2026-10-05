"""Prior-informed local GP/BO baseline (gp-bo-prior-v1): the main non-LLM baseline (decision 2026-10-03).

It starts where a grower would start and searches only nearby:

* The anchor is the frozen fixed reference policy (``configs/fixed_reference_v1.json``): on day 0 one
  compartment plants it, the other day-0 compartment plants a perturbation of it.
* Perturbations come from a two-level orthogonal design (rows of an order-8 Hadamard matrix): every
  field moves by plus or minus the trust-region radius, in the scaled policy box. Row order, column
  choice and column signs are drawn from the method seed.
* The Gaussian process has a frozen kernel (form and hyperparameters, see ``prior_kernel``) and
  noise-to-signal ratio, fitted once on development-site data
  (``scripts/fit_prior_bo_hyperparameters.py``). Within a campaign only a constant mean and a scale are
  estimated (generalised least squares and profile likelihood); neither changes the posterior mean
  ranking or the expected-improvement argmax, so nothing about the response surface is refitted from
  two to six crops.
* Expected improvement is maximised only inside a box trust region centred on the incumbent (the tried
  policy with the highest posterior mean of the scored objective). The radius starts at a value chosen
  on development sites and expands or shrinks by the TuRBO defaults (double after 3 consecutive
  successful batches, halve after ceil(max(4/q, d/q)) failures; side length within [0.5^7, 1.6]).
* The final recommendation is the tried policy with the highest posterior mean of the scored objective;
  untried policies are never recommended.

Scheduling follows the staggered gp-bo-v3: two compartments on day 0, two more after half a crop
cycle (under Full feedback chosen from the public process predictor's predicted final margins, under
Endpoint from further design rows), then a second wave on the day-0 compartments when they finish.
The mean-surface variant (secondary sensitivity analysis) replaces the constant mean by a
development-site mean surface; it is not part of this class.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .bo_agent import BOConfig, GPBOAgent, expected_improvement
from .process_predictor import POLICY_ORDER
from .tools import _matern52, scoring_days, season_features

PRIOR_BO_VERSION = 'gp-bo-prior-v1.1'  # v1.1: inputs in the kernel's field order (decision 2026-10-05)
TURBO = {'side_init_factor': 2.0, 'side_min': 0.5 ** 7, 'side_max': 1.6, 'success_tolerance': 3}


def hadamard8():
    h = np.array([[1]])
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    return h


KERNEL_FORMS = ('product', 'season_interaction')


def prior_kernel(a, b, kernel: dict):
    """Unit-scale prior covariance between rows (six scaled policy fields, two planting-season inputs).

    ``product``: Matern 5/2 over all eight inputs with one length scale each.
    ``season_interaction``: k_p(x, x') * (1 - gamma + gamma * k_s(s, s')) + tau * k_s(s, s'), with Matern
    5/2 k_p over the policy fields and k_s over the season inputs (one shared length scale). The tau term
    is a season effect common to every policy, so it cancels when policies are compared; gamma is the
    share of the policy effect that changes with the planting season.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if kernel['form'] == 'product':
        return _matern52(a, b, np.asarray(kernel['lengthscales'], dtype=float))
    if kernel['form'] != 'season_interaction':
        raise ValueError('unknown kernel form')
    kp = _matern52(a[:, :6], b[:, :6], np.asarray(kernel['policy_lengthscales'], dtype=float))
    ks = _matern52(a[:, 6:], b[:, 6:], np.full(2, float(kernel['season_lengthscale'])))
    gamma, tau = float(kernel['gamma']), float(kernel['tau'])
    return kp * (1.0 - gamma + gamma * ks) + tau * ks


def kernel_variance(kernel: dict) -> float:
    return 1.0 + (float(kernel['tau']) if kernel['form'] == 'season_interaction' else 0.0)


class FrozenGP:
    """GP with a frozen kernel and noise ratio; constant mean and scale estimated from the data."""

    def __init__(self, x, y, kernel, noise_ratio, *, mean=None, scale2=None):
        self.kernel = dict(kernel)
        self.noise_ratio = float(noise_ratio)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.prior_var = kernel_variance(self.kernel)
        k = prior_kernel(self.x, self.x, self.kernel) + (self.noise_ratio + 1e-9) * np.eye(len(self.x))
        self.k_inv = np.linalg.inv(k)
        ones = np.ones(len(self.y))
        self.mean = float(ones @ self.k_inv @ self.y / (ones @ self.k_inv @ ones)) if mean is None else mean
        resid = self.y - self.mean
        if scale2 is None:
            scale2 = float(resid @ self.k_inv @ resid / len(self.y)) if len(self.y) > 1 else 1.0
            scale2 = max(scale2, 1e-6)
        self.scale2 = scale2
        self.alpha = self.k_inv @ resid

    def with_fantasies(self, xs, ys) -> 'FrozenGP':
        """Same mean and scale, conditioned on extra (x, y) pairs (kriging believer)."""
        return FrozenGP(np.vstack([self.x, np.asarray(xs, dtype=float)]),
                        np.concatenate([self.y, np.asarray(ys, dtype=float)]), self.kernel,
                        self.noise_ratio, mean=self.mean, scale2=self.scale2)

    def predict(self, xs):
        ks = prior_kernel(np.asarray(xs, dtype=float), self.x, self.kernel)
        mean = self.mean + ks @ self.alpha
        var = np.maximum(self.prior_var - np.einsum('ij,jk,ik->i', ks, self.k_inv, ks), 1e-12) * self.scale2
        return mean, np.sqrt(var), np.sqrt(var + self.noise_ratio * self.scale2)

    def predict_joint(self, xs):
        xs = np.asarray(xs, dtype=float)
        ks = prior_kernel(xs, self.x, self.kernel)
        cov = (prior_kernel(xs, xs, self.kernel) - ks @ self.k_inv @ ks.T) * self.scale2
        return self.mean + ks @ self.alpha, cov

    def predict_average(self, policies_x, days):
        seasons = [season_features(d) for d in days]
        weights = np.full(len(days), 1.0 / len(days))
        means, sds, per_day = [], [], []
        for x in np.asarray(policies_x, dtype=float):
            mean, cov = self.predict_joint(np.array([list(x) + s for s in seasons]))
            means.append(float(weights @ mean))
            sds.append(math.sqrt(max(float(weights @ cov @ weights), 1e-12)))
            per_day.append(mean.tolist())
        return np.array(means), np.array(sds), per_day


def kernel_from_theta(theta, form):
    if form == 'product':
        return {'form': form, 'lengthscales': np.exp(theta[:8]).tolist()}, math.exp(theta[8])
    return ({'form': form, 'policy_lengthscales': np.exp(theta[:6]).tolist(),
             'season_lengthscale': math.exp(theta[6]), 'gamma': 1.0 / (1.0 + math.exp(-theta[7])),
             'tau': math.exp(theta[8])}, math.exp(theta[9]))


def theta_bounds(form):
    ls = (math.log(0.05), math.log(20.0))
    ratio = (math.log(1e-4), math.log(10.0))
    if form == 'product':
        return [ls] * 8 + [ratio]
    return [ls] * 7 + [(-8.0, 8.0), (math.log(1e-3), math.log(100.0)), ratio]


def profile_nll(theta, groups, form='product'):
    """Negative log likelihood summed over groups, each with its own profiled constant mean and scale."""
    kernel, ratio = kernel_from_theta(theta, form)
    total = 0.0
    for x, y in groups:
        n = len(y)
        k = prior_kernel(x, x, kernel) + (ratio + 1e-9) * np.eye(n)
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return 1e10
        solve = lambda v: np.linalg.solve(chol.T, np.linalg.solve(chol, v))  # noqa: E731
        ones = np.ones(n)
        mean = float(ones @ solve(y) / (ones @ solve(ones)))
        resid = y - mean
        scale2 = max(float(resid @ solve(resid)) / n, 1e-12)
        total += 0.5 * n * math.log(scale2) + float(np.log(np.diag(chol)).sum())
    return total


def fit_shared_hyperparameters(groups, rng, restarts=16, form='product'):
    """Kernel hyperparameters and noise ratio shared by all groups (maximum profile likelihood)."""
    from scipy.optimize import minimize
    bounds = theta_bounds(form)
    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    best = None
    for start in low + rng.random((restarts, len(bounds))) * (high - low):
        fit = minimize(profile_nll, start, args=(groups, form), method='L-BFGS-B', bounds=bounds)
        if best is None or fit.fun < best.fun:
            best = fit
    kernel, ratio = kernel_from_theta(best.x, form)
    at_bound = [i for i, (v, (lo, hi)) in enumerate(zip(best.x, bounds)) if v - lo < 1e-3 or hi - v < 1e-3]
    return {'kernel': kernel, 'noise_ratio': ratio, 'neg_log_likelihood': float(best.fun),
            'parameters_at_bounds': at_bound, 'groups': len(groups),
            'observations': int(sum(len(y) for _, y in groups))}


@dataclass(frozen=True)
class PriorBOConfig:
    anchor: dict
    kernel: dict
    noise_ratio: float
    radius: float = 0.25
    initial_units: int = 2
    stagger_fraction: float = 0.5
    pool_size: int = 256
    xi: float = 0.0
    turbo: dict = field(default_factory=lambda: dict(TURBO))
    # Policy input order of the frozen kernel (its length scales are per position).
    policy_order: tuple = POLICY_ORDER

    def __post_init__(self):
        if set(self.policy_order) != set(self.anchor) or len(self.policy_order) != 6:
            raise ValueError('invalid prior BO configuration')
        if not 0 < self.radius <= 0.8 or self.kernel.get('form') not in KERNEL_FORMS or self.noise_ratio <= 0:
            raise ValueError('invalid prior BO configuration')
        if not 0 < self.stagger_fraction < 1 or self.initial_units < 1 or self.pool_size < 16:
            raise ValueError('invalid prior BO configuration')


def load_prior_config(path, radius: float) -> PriorBOConfig:
    """A prior BO config file (frozen hyperparameters and the anchor reference file) at one radius."""
    import json
    from pathlib import Path
    from .frozen_paths import frozen_path
    record = json.loads(Path(path).read_text())
    if radius not in record['candidate_radii'] and radius != record.get('selected_radius'):
        raise ValueError('radius is not a candidate of this prior BO config')
    root = Path(__file__).resolve().parents[1]
    anchor = json.loads(frozen_path(root, record['anchor']['file']).read_text())['policy']
    hyper = record['hyperparameters']
    if 'policy_order' not in record:
        raise ValueError('this prior BO config does not record its policy input order (v2 or later required)')
    return PriorBOConfig(anchor=anchor, kernel=dict(hyper['kernel']), noise_ratio=hyper['noise_ratio'],
                         radius=radius, turbo=dict(record['turbo']), policy_order=tuple(record['policy_order']))


class PriorLocalBOAgent(GPBOAgent):
    def __init__(self, session, seed: int, config: PriorBOConfig):
        super().__init__(session, seed, BOConfig(schedule='staggered', initial_units=config.initial_units,
                                                 stagger_fraction=config.stagger_fraction))
        self.prior = config
        fields = self.task['policy']['fields']
        if set(config.anchor) != set(fields) or set(config.policy_order) != set(fields):
            raise ValueError('anchor policy fields do not match the task')
        # Inputs follow the kernel's order, not the order of the task mapping (agent sessions sort keys).
        self.fields = list(config.policy_order)
        self.anchor = {k: float(config.anchor[k]) for k in self.fields}
        design = hadamard8()[:, 1:]
        columns = self.rng.permutation(design.shape[1])[:len(self.fields)]
        self._design = design[self.rng.permutation(len(design))][:, columns] * self.rng.choice([-1, 1], len(columns))
        self._design_next = 0
        self.side = 2.0 * config.radius
        self.successes = self.failures = 0
        self.best_observed = None
        self.trust_log = []

    # ── policy space ───────────────────────────────────────────────────────
    def _policy(self, x):
        spec = self.task['policy']['fields']
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        policy = {k: round(spec[k]['min'] + float(u) * (spec[k]['max'] - spec[k]['min']), 2)
                  for k, u in zip(self.fields, x)}
        if policy['night_temperature_c'] > policy['day_temperature_c']:
            policy['night_temperature_c'] = policy['day_temperature_c']
        return policy

    def _tried(self):
        return [r['policy'] for r in self.started]

    def _design_points(self, n):
        centre, radius = self._scale(self.anchor), self.prior.radius
        picks = []
        while len(picks) < n and self._design_next < len(self._design):
            row = self._design[self._design_next]
            self._design_next += 1
            policy = self._policy(centre + radius * row)
            if policy not in self._tried() and policy not in [p for p, _ in picks]:
                picks.append((policy, {'basis': 'anchor_design', 'design_row': self._design_next - 1}))
        return picks

    def _pool(self, centre):
        from scipy.stats import qmc
        half = self.side / 2.0
        low, high = np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)
        sampler = qmc.Sobol(len(self.fields), scramble=True, seed=self.rng)
        points = low + sampler.random_base2(math.ceil(math.log2(self.prior.pool_size))) * (high - low)
        tried = self._tried()
        policies = []
        for x in points:
            policy = self._policy(x)
            if policy not in tried and policy not in policies:
                policies.append(policy)
        return policies, np.array([self._scale(p) for p in policies])

    # ── model ──────────────────────────────────────────────────────────────
    def _model(self, rows, ys):
        if not rows:
            return None
        return FrozenGP(np.array(rows), np.array(ys), self.prior.kernel, self.prior.noise_ratio)

    def _incumbent(self, model, policies):
        """The tried policy with the highest posterior mean of the scored objective."""
        mean, sd, per_day = model.predict_average(np.array([self._scale(p) for p in policies]),
                                                  scoring_days(self.task))
        j = int(np.argmax(mean))
        return policies[j], float(mean[j]), float(sd[j]), per_day[j]

    def _ei_batch(self, model, n, planting_day, centre, pending=()):
        if pending:
            mean, _, _ = model.predict(np.array(pending))
            model = model.with_fantasies(pending, mean)
        policies, xs = self._pool(centre)
        days = scoring_days(self.task)
        chosen = []
        for _ in range(n):
            tried = np.array([self._scale(p) for p in self._tried()] + [self._scale(p) for p, _ in chosen])
            incumbent = float(model.predict_average(tried, days)[0].max())
            mean, sd, _ = model.predict_average(xs, days)
            ei = expected_improvement(mean, sd, incumbent, self.prior.xi)
            j = int(np.argmax(ei))
            chosen.append((policies[j], {'basis': 'expected_improvement', 'ei': float(ei[j]),
                                         'predicted_mean': float(mean[j]), 'predicted_sd': float(sd[j]),
                                         'trust_region_side': self.side}))
            row = np.concatenate([xs[j], season_features(planting_day)])
            believed, _, _ = model.predict(row[None, :])
            model = model.with_fantasies([row], believed)
            policies.pop(j)
            xs = np.delete(xs, j, axis=0)
        return chosen

    def _update_trust_region(self, new_margins):
        """TuRBO rule after a batch of observed (completed) crops."""
        if not new_margins:
            return
        best_new = max(new_margins)
        if self.best_observed is not None:
            if best_new > self.best_observed + 1e-3 * abs(self.best_observed):
                self.successes, self.failures = self.successes + 1, 0
            else:
                self.successes, self.failures = 0, self.failures + 1
            q = max(1, len(new_margins))
            fail_tol = math.ceil(max(4.0 / q, len(self.fields) / q))
            turbo = self.prior.turbo
            if self.successes >= turbo['success_tolerance']:
                self.side, self.successes = min(2.0 * self.side, turbo['side_max']), 0
            elif self.failures >= fail_tol:
                self.side, self.failures = max(self.side / 2.0, turbo['side_min']), 0
        self.best_observed = best_new if self.best_observed is None else max(self.best_observed, best_new)
        self.trust_log.append({'day': self.clock_day, 'side': self.side, 'successes': self.successes,
                               'failures': self.failures})

    def _start_policies(self, units, picks):
        for unit, (policy, basis) in zip(units, picks):
            result = self.session.dispatch({'action': 'start', 'unit': unit, 'policy': policy})
            self.started.append({'unit': unit, 'run_index': result['run_index'], 'policy': policy,
                                 'start_day': self.clock_day})
            self.log.append({'day': self.clock_day, 'action': 'start', 'unit': unit, 'policy': policy, **basis})

    def _collect(self, runs):
        before = len(self.completed)
        for run in runs:
            self._collect_final(run['unit'], run['run_index'])
        self._update_trust_region([m for _, m, _ in self.completed[before:]])

    # ── campaign ───────────────────────────────────────────────────────────
    def run(self) -> dict:
        cal = self.task['calendar']
        units = list(self.task['compartments'])
        full = self.task['feedback']['mode'] == 'full'
        crop, clean, latest, end = cal['crop_days'], cal['cleanup_days'], cal['latest_start_day'], cal['campaign_days']
        first = units[:self.prior.initial_units]
        late = units[len(first):]

        self._start_policies(first, [(dict(self.anchor), {'basis': 'anchor'})] + self._design_points(len(first) - 1))
        if late:
            self._advance(self._align(min(self.prior.stagger_fraction * crop, latest)))
            if full:
                predicted = [(r, *self._predicted_final_margin(r)) for r in self.started]
                self.log.append({'day': self.clock_day, 'action': 'predicted_final_margins',
                                 'values': [round(m, 6) for _, m, _ in predicted],
                                 'typical_error': [round(e, 6) for _, _, e in predicted]})
                model = self._model([self._row(r['policy'], r['start_day']) for r, _, _ in predicted],
                                    [m for _, m, _ in predicted])
                centre, *_ = self._incumbent(model, [r['policy'] for r, _, _ in predicted])
                picks = self._ei_batch(model, len(late), self.clock_day, self._scale(centre))
            else:
                picks = self._design_points(len(late))
            self._start_policies(late, picks)

        wave_two_day = self._align(crop + clean)
        if wave_two_day <= latest:
            self._advance(wave_two_day)
            self._collect([r for r in self.started if r['unit'] in first])
            completed = [p for _, _, p in self.completed]
            if completed:
                model = self._model([x for x, _, _ in self.completed], [m for _, m, _ in self.completed])
                centre, *_ = self._incumbent(model, completed)
                pending = [self._row(r['policy'], r['start_day']) for r in self.started if 'closed' not in r]
                picks = self._ei_batch(model, len(first), self.clock_day, self._scale(centre), pending)
            else:
                picks = self._design_points(len(first))
            self._start_policies(first, picks)

        self._advance(end)
        self._collect([r for r in self.started if 'closed' not in r])
        recommendation, basis = self._recommend()
        result = self.session.dispatch({'action': 'recommend', 'policy': recommendation})
        self.log.append({'day': self.clock_day, 'action': 'recommend', 'policy': recommendation, **basis,
                         'fallback': result['fallback']})
        config = asdict(self.prior)
        return {'version': PRIOR_BO_VERSION, 'config': config, 'recommendation': recommendation,
                'completed_crops': len(self.completed), 'trust_region': self.trust_log, 'log': self.log}

    def _recommend(self):
        if not self.completed:
            return dict(self.anchor), {'basis': 'anchor_no_completed_crops'}
        model = self._model([x for x, _, _ in self.completed], [m for _, m, _ in self.completed])
        tried = []
        for _, _, p in self.completed:
            if p not in tried:
                tried.append(p)
        policy, mean, sd, per_day = self._incumbent(model, tried)
        return dict(policy), {'basis': 'max_posterior_scored_mean_tried', 'predicted_mean': mean,
                              'predicted_sd': sd, 'predicted_by_planting_day': per_day}
