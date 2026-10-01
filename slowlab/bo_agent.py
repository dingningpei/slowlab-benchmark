"""GP/BO baseline agent (gp-bo-v3): a fixed scheduling and acquisition policy.

It acts only through a ``PublicSession`` (the same interface as a language
model) and uses the same Gaussian-process component as the public toolbox.
Its seed is independent of every private site seed.

Schedules (choose on development sites only):

* ``two_waves``: start every compartment at day 0 with a space-filling design;
  when those crops finish and are cleaned, fit a GP to the completed crops and
  start a second wave by batch expected improvement (kriging believer).
  Process information cannot change any decision here, so Full and Endpoint
  runs behave identically.
* ``staggered``: start ``initial_units`` compartments at day 0 and keep the
  rest free until ``stagger_fraction`` of a crop cycle has passed. Then, under
  Full feedback, read each running crop's cumulative channels and canopy proxy,
  predict its final contribution margin with the public process predictor (the
  same component as the predict_crop_outcome tool), fit a GP to those
  predictions and choose the late policies by expected improvement; under
  Endpoint feedback, continue the space-filling design. The second wave is
  chosen as in ``two_waves``.

Completed crops enter the GP with their planting date (position in the year).
Second-wave expected improvement and the final recommendation target the
scoring rule of the task: the mean margin over its planting dates (contract
v5: 1 January and 2 July). The final recommendation maximises that posterior
mean over the candidate pool and every completed policy.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .process_predictor import predict_from_history
from .tools import OutcomeModel, PublicHistory, _predictor, contribution_margin, scoring_days, season_features

BO_VERSION = 'gp-bo-v3'
READINGS = ('cumulative_harvest_fresh_equivalent', 'heating_energy', 'lighting_energy', 'co2_dosed',
            'canopy_lai_proxy')


@dataclass(frozen=True)
class BOConfig:
    schedule: str = 'two_waves'
    initial_units: int = 2
    stagger_fraction: float = 0.5
    pool_size: int = 256
    gp_restarts: int = 8
    xi: float = 0.0

    def __post_init__(self):
        if self.schedule not in ('two_waves', 'staggered'):
            raise ValueError('schedule must be two_waves or staggered')
        if not 0 < self.stagger_fraction < 1 or self.pool_size < 16 or self.initial_units < 1:
            raise ValueError('invalid BO configuration')


def expected_improvement(mean, sd, best, xi):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-12)
    z = (mean - best - xi) / sd
    return (mean - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


class GPBOAgent:
    def __init__(self, session, seed: int, config: BOConfig = BOConfig()):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('seed must be a non-negative integer')
        self.session = session
        self.config = config
        self.task = session.task
        self.fields = list(self.task['policy']['fields'])
        self.rng = np.random.default_rng([seed, 0x60B0])
        self.log: list[dict] = []
        self.completed: list[tuple[np.ndarray, float, dict]] = []
        self.started: list[dict] = []
        self.clock_day = 0.0
        self._pool_policies, self._pool_x = self._make_pool()
        self._next_space_filling = 0

    # ── policy space ───────────────────────────────────────────────────────
    def _scale(self, policy):
        spec = self.task['policy']['fields']
        return np.array([(policy[k] - spec[k]['min']) / (spec[k]['max'] - spec[k]['min']) for k in self.fields])

    def _row(self, policy, planting_day):
        return np.concatenate([self._scale(policy), season_features(planting_day)])

    def _make_pool(self):
        from scipy.stats import qmc
        spec = self.task['policy']['fields']
        sampler = qmc.Sobol(len(self.fields), scramble=True, seed=self.rng)
        policies = []
        for point in sampler.random_base2(math.ceil(math.log2(4 * self.config.pool_size))):
            policy = {k: round(spec[k]['min'] + float(u) * (spec[k]['max'] - spec[k]['min']), 2)
                      for k, u in zip(self.fields, point)}
            if policy['night_temperature_c'] <= policy['day_temperature_c']:
                policies.append(policy)
            if len(policies) == self.config.pool_size:
                break
        return policies, np.array([self._scale(p) for p in policies])

    # ── time ───────────────────────────────────────────────────────────────
    def _align(self, day):
        tick = self.task['calendar']['time_step_seconds']
        return round(day * 86400 / tick) * tick / 86400

    def _advance(self, day):
        day = self._align(day)
        if day > self.clock_day:
            result = self.session.dispatch({'action': 'advance', 'day': day})
            self.clock_day = result['clock'] / 86400

    # ── data ───────────────────────────────────────────────────────────────
    def _collect_final(self, unit, run_index):
        result = self.session.dispatch({'action': 'observe', 'unit': unit, 'run_index': run_index})
        aggregate = result.get('final_aggregate')
        run = next(r for r in self.started if r['unit'] == unit and r['run_index'] == run_index)
        if aggregate and aggregate['reason'] == 'normal_completion':
            margin = contribution_margin(aggregate['accrued'], aggregate['event_cost_eur_m2'], self.task['economics'])
            self.completed.append((self._row(run['policy'], run['start_day']), margin, run['policy']))
            run['margin'] = margin
        run['closed'] = aggregate['reason'] if aggregate else None

    def _predicted_final_margin(self, run):
        """Read every predictor channel now, then predict the crop's final margin (public predictor)."""
        now = self.clock_day
        for channel in READINGS:
            self.session.dispatch({'action': 'observe', 'unit': run['unit'], 'variable': channel,
                                   'start_day': now, 'end_day': now})
        prediction = predict_from_history(self.task, PublicHistory(self.session.transcript), run['unit'],
                                          run['run_index'], _predictor())
        return prediction['predicted_contribution_margin_eur_m2'], prediction['typical_error']['margin_eur_m2']

    def _fit(self, xs, ys):
        if len(xs) < 2:
            return None
        return OutcomeModel(np.array(xs, dtype=float), np.array(ys, dtype=float), self.rng, self.config.gp_restarts)

    def _unused(self):
        tried = [r['policy'] for r in self.started]
        return [i for i, p in enumerate(self._pool_policies) if p not in tried]

    def _space_filling(self, n):
        chosen = []
        for i in self._unused():
            if len(chosen) == n:
                break
            chosen.append((i, {'basis': 'space_filling'}))
        return chosen

    def _batch_ei(self, model, best, n, pending=()):
        """Policy-only model (interim margins of same-season crops)."""
        if model is None:
            return self._space_filling(n)
        available = self._unused()
        chosen = []
        for _ in range(n):
            xs = self._pool_x[available]
            mean, sd, _ = model.predict(xs)
            ei = expected_improvement(mean, sd, best, self.config.xi)
            j = int(np.argmax(ei))
            index = available.pop(j)
            chosen.append((index, {'basis': 'expected_improvement', 'ei': float(ei[j]),
                                   'predicted_mean': float(mean[j]), 'predicted_sd': float(sd[j])}))
            model = model.with_fantasies([self._pool_x[index]], [mean[j]])
            best = max(best, float(mean[j]))
        return chosen

    def _scored_ei(self, model, n, planting_day, pending=()):
        """Batch EI on the scored objective (mean over scoring plantings), kriging believer."""
        if model is None:
            return self._space_filling(n)
        days = scoring_days(self.task)
        if pending:
            mean, _, _ = model.predict(np.array(pending))
            model = model.with_fantasies(pending, mean)
        available = self._unused()
        chosen = []
        for _ in range(n):
            tried = np.array([self._scale(p) for _, _, p in self.completed] +
                             [self._pool_x[i] for i, _ in chosen])
            incumbent = float(model.predict_average(tried, days)[0].max())
            mean, sd, _ = model.predict_average(self._pool_x[available], days)
            ei = expected_improvement(mean, sd, incumbent, self.config.xi)
            j = int(np.argmax(ei))
            index = available.pop(j)
            chosen.append((index, {'basis': 'expected_improvement', 'ei': float(ei[j]),
                                   'predicted_mean': float(mean[j]), 'predicted_sd': float(sd[j]),
                                   'objective': 'scored_mean_over_plantings'}))
            row = np.concatenate([self._pool_x[index], season_features(planting_day)])
            believed, _, _ = model.predict(row[None, :])
            model = model.with_fantasies([row], believed)
        return chosen

    def _start(self, units, picks):
        for unit, (index, basis) in zip(units, picks):
            policy = dict(self._pool_policies[index])
            result = self.session.dispatch({'action': 'start', 'unit': unit, 'policy': policy})
            self.started.append({'unit': unit, 'run_index': result['run_index'], 'policy': policy,
                                 'start_day': self.clock_day})
            self.log.append({'day': self.clock_day, 'action': 'start', 'unit': unit, 'policy': policy, **basis})

    # ── campaign ───────────────────────────────────────────────────────────
    def run(self) -> dict:
        cal = self.task['calendar']
        units = list(self.task['compartments'])
        full = self.task['feedback']['mode'] == 'full'
        crop, clean, latest, end = cal['crop_days'], cal['cleanup_days'], cal['latest_start_day'], cal['campaign_days']
        staggered = self.config.schedule == 'staggered' and self.config.initial_units < len(units)
        first = units[:self.config.initial_units] if staggered else units
        late = units[len(first):]

        self._start(first, self._space_filling(len(first)))
        if late:
            stagger_day = self._align(min(self.config.stagger_fraction * crop, latest))
            self._advance(stagger_day)
            if full:
                predicted = [(self._scale(r['policy']), *self._predicted_final_margin(r)) for r in self.started]
                self.log.append({'day': self.clock_day, 'action': 'predicted_final_margins',
                                 'values': [round(m, 6) for _, m, _ in predicted],
                                 'typical_error': [round(e, 6) for _, _, e in predicted]})
                model = self._fit([x for x, _, _ in predicted], [m for _, m, _ in predicted])
                best = max(m for _, m, _ in predicted)
                picks = self._batch_ei(model, best, len(late))
            else:
                picks = self._space_filling(len(late))
            self._start(late, picks)

        wave_two_day = self._align(crop + clean)
        if wave_two_day <= latest:
            self._advance(wave_two_day)
            for run in [r for r in self.started if r['unit'] in first]:
                self._collect_final(run['unit'], run['run_index'])
            model = self._fit([x for x, _, _ in self.completed], [m for _, m, _ in self.completed])
            pending = [self._row(r['policy'], r['start_day']) for r in self.started if 'closed' not in r]
            self._start(first, self._scored_ei(model, len(first), self.clock_day, pending))

        self._advance(end)
        for run in [r for r in self.started if 'closed' not in r]:
            self._collect_final(run['unit'], run['run_index'])
        recommendation, basis = self._recommend()
        result = self.session.dispatch({'action': 'recommend', 'policy': recommendation})
        self.log.append({'day': self.clock_day, 'action': 'recommend', 'policy': recommendation, **basis,
                         'fallback': result['fallback']})
        return {'version': BO_VERSION, 'config': asdict(self.config), 'recommendation': recommendation,
                'completed_crops': len(self.completed), 'log': self.log}

    def _recommend(self):
        xs = [x for x, _, _ in self.completed]
        ys = [m for _, m, _ in self.completed]
        if not xs:
            return dict(self._pool_policies[0]), {'basis': 'no_completed_crops'}
        candidates = self._pool_policies + [p for _, _, p in self.completed]
        cand_x = np.array([self._scale(p) for p in candidates])
        model = self._fit(xs, ys)
        if model is None:
            best = max(self.completed, key=lambda row: row[1])
            return dict(best[2]), {'basis': 'best_observed', 'observed_margin': best[1]}
        mean, sd, per_day = model.predict_average(cand_x, scoring_days(self.task))
        j = int(np.argmax(mean))
        return dict(candidates[j]), {'basis': 'max_posterior_scored_mean', 'predicted_mean': float(mean[j]),
                                     'predicted_sd': float(sd[j]), 'predicted_by_planting_day': per_day[j]}
