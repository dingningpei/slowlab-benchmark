"""Public numerical analysis tools shared by every agent and the GP/BO baseline.

Every tool is a pure function of the public task view and the public
transcript (what the agent has already received through ``dispatch``). Tools
never query the executor, so they cost no campaign actions and cannot touch
private state. Randomness (GP restarts, space-filling designs) is seeded from
the tool seed and a hash of the tool name and its inputs, so identical public
inputs give byte-identical outputs regardless of call order, and the tool seed
is independent of every private site seed. Numbers are rounded to six
significant digits.

Candidate selection, scheduling and the final recommendation stay with the
agent: there is deliberately no acquisition optimiser here.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from .agent_protocol import canonical

TOOLBOX_VERSION = 'analysis-tools-v1'
LIMITS = {'max_calls': 64, 'max_predict_policies': 32, 'max_candidates': 64,
          'gp_restarts': 8, 'min_completed_runs_for_model': 3, 'max_summary_days': 400}

CATALOG = {
    'runs': {'description': 'Table of every crop you started: compartment, crop number, policy, start day, status, '
                            'and for closed crops with released totals the accrued resources and contribution margin '
                            '(EUR per m2) at the public prices.',
             'args': {}},
    'series_summary': {'description': 'Daily count, mean, minimum and maximum of one channel you have already '
                                      'queried for one compartment.',
                       'args': {'unit': 'integer compartment index', 'variable': 'public channel name',
                                'start_day': 'optional first day', 'end_day': 'optional last day'}},
    'fit_outcome_model': {'description': 'Fit a Gaussian-process model of contribution margin per m2 over one full '
                                         'crop cycle as a function of the policy and the planting date (as a '
                                         'position in the year), using completed crops only. Returns the number of '
                                         'crops used, fitted length scales per input (larger means less sensitive), '
                                         'noise level and leave-one-out error.',
                          'args': {}},
    'predict': {'description': 'Predicted contribution margin per m2 over one crop cycle for up to 32 policies, from '
                               'the model fitted to your completed crops. By default: one prediction per planting '
                               'date of the scoring rule and their mean, which is what your recommendation is scored '
                               'on. With planting_day: that planting date only.',
                'args': {'policies': 'list of policy objects',
                         'planting_day': 'optional calendar day of planting (0-364)'}},
    'space_filling_candidates': {'description': 'Up to 64 feasible policies spread evenly over the allowed ranges, '
                                                'optionally holding some fields fixed.',
                                 'args': {'n': 'number of policies', 'fixed': 'optional object of field: value'}},
}


class ToolError(ValueError):
    """Invalid tool arguments or exhausted tool budget; the agent may correct and retry."""


def _round(obj):
    if isinstance(obj, float):
        return float(f'{obj:.6g}') if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


class PublicHistory:
    """Runs, statuses and measured records reconstructed from a public transcript."""

    def __init__(self, transcript: list):
        self.runs: dict[tuple[int, int], dict] = {}
        self.records: dict[tuple[int, str], dict[float, float]] = {}
        self.last_status: dict[int, dict] = {}
        for entry in transcript:
            if not entry.get('ok'):
                continue
            request, result = entry['request'], entry['result']
            for status in ([result['status']] if isinstance(result.get('status'), dict)
                           else result.get('status') or []):
                self.last_status[int(status['unit'])] = status
            kind = request['action']
            if kind == 'start':
                key = (int(result['unit']), int(result['run_index']))
                self.runs[key] = {'unit': key[0], 'run_index': key[1], 'policy': dict(request['policy']),
                                  'start_day': result['clock'] / 86400, 'status': 'running'}
            elif kind == 'observe':
                aggregate = result.get('final_aggregate')
                if aggregate:
                    key = (int(aggregate['unit']), int(aggregate['run_index']))
                    run = self.runs.setdefault(key, {'unit': key[0], 'run_index': key[1], 'policy': None,
                                                     'start_day': None})
                    run.update(status='closed', reason=aggregate['reason'],
                               closed_day=aggregate['closed_at'] / 86400,
                               accrued=dict(aggregate['accrued']),
                               event_cost_eur_m2=aggregate['event_cost_eur_m2'])
                for record in result.get('records') or []:
                    series = self.records.setdefault((int(record['compartment']), record['variable']), {})
                    series[float(record['measurement_time'])] = float(record['value'])
        for (unit, run_index), run in self.runs.items():
            status = self.last_status.get(unit)
            if run['status'] == 'running' and status and (
                    status['run_index'] > run_index
                    or (status['run_index'] == run_index and status['phase'] != 'active')):
                run['status'] = 'closed_totals_not_yet_observed'


def contribution_margin(accrued: dict, event_cost_eur_m2: float, economics: dict) -> float:
    revenue = accrued['harvest_kg_m2'] * (economics['fruit_price_eur_per_kg_fresh'] - economics['harvest_handling_eur_per_kg'])
    costs = (accrued['heat_kwh_m2'] * economics['delivered_heat_eur_per_kwh']
             + accrued['light_kwh_m2'] * economics['electricity_eur_per_kwh']
             + accrued['co2_kg_m2'] * economics['co2_eur_per_kg']
             + accrued['days'] * economics['background_service_eur_per_m2_day']
             + event_cost_eur_m2)
    return revenue - costs


SEASON_INPUTS = ('planting_season_sin', 'planting_season_cos')


def season_features(day: float) -> list[float]:
    """Planting date as a position in the year, scaled like the policy inputs to [0, 1]."""
    angle = 2.0 * math.pi * (float(day) % 365.0) / 365.0
    return [0.5 * (1.0 + math.sin(angle)), 0.5 * (1.0 + math.cos(angle))]


def scoring_days(task: dict) -> list:
    scoring = task.get('scoring')
    return list(scoring['planting_calendar_days']) if scoring else [0]


def _matern52(a, b, lengthscales):
    diff = (a[:, None, :] - b[None, :, :]) / lengthscales
    r = np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))
    s = math.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


class OutcomeModel:
    """Gaussian process (Matern 5/2, per-field length scales) on scaled policies."""

    def __init__(self, x, y, rng, restarts):
        from scipy.optimize import minimize
        self.x, self.dim = x, x.shape[1]
        self.y_mean, self.y_std = float(y.mean()), float(y.std()) or 1.0
        z = (y - self.y_mean) / self.y_std
        low = np.array([math.log(0.05)] * self.dim + [math.log(1e-2), math.log(1e-6)])
        high = np.array([math.log(20.0)] * self.dim + [math.log(1e2), math.log(1e1)])

        def nll(theta):
            ls, sf2, sn2 = np.exp(theta[:self.dim]), math.exp(theta[-2]), math.exp(theta[-1])
            k = sf2 * _matern52(x, x, ls) + (sn2 + 1e-9) * np.eye(len(x))
            try:
                chol = np.linalg.cholesky(k)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, z))
            return float(0.5 * z @ alpha + np.log(np.diag(chol)).sum())

        best = None
        for start in low + rng.random((restarts, len(low))) * (high - low):
            fit = minimize(nll, start, method='L-BFGS-B', bounds=list(zip(low, high)))
            if best is None or fit.fun < best.fun:
                best = fit
        theta = best.x
        self.lengthscales, self.sf2, self.sn2 = np.exp(theta[:self.dim]), math.exp(theta[-2]), math.exp(theta[-1])
        self._condition(x, z)
        loo = z - self.alpha / np.diag(self.k_inv)
        self.loo_rmse = float(np.sqrt(np.mean((z - loo) ** 2)) * self.y_std)

    def _condition(self, x, z):
        self.x = x
        k = self.sf2 * _matern52(x, x, self.lengthscales) + (self.sn2 + 1e-9) * np.eye(len(x))
        self.k_inv = np.linalg.inv(k)
        self.alpha = self.k_inv @ z
        self.z = z

    def with_fantasies(self, xs, ys) -> 'OutcomeModel':
        """Same hyperparameters, conditioned on extra (x, y) pairs (kriging believer)."""
        other = copy.copy(self)
        zs = (np.asarray(ys, dtype=float) - self.y_mean) / self.y_std
        other._condition(np.vstack([self.x, np.asarray(xs, dtype=float)]), np.concatenate([self.z, zs]))
        return other

    def predict(self, xs):
        ks = self.sf2 * _matern52(xs, self.x, self.lengthscales)
        mean = ks @ self.alpha
        var = np.maximum(self.sf2 - np.einsum('ij,jk,ik->i', ks, self.k_inv, ks), 1e-12)
        return mean * self.y_std + self.y_mean, np.sqrt(var) * self.y_std, np.sqrt(var + self.sn2) * self.y_std

    def predict_joint(self, xs):
        """Posterior mean and latent covariance (original units) at several inputs."""
        ks = self.sf2 * _matern52(xs, self.x, self.lengthscales)
        cov = self.sf2 * _matern52(xs, xs, self.lengthscales) - ks @ self.k_inv @ ks.T
        return (ks @ self.alpha) * self.y_std + self.y_mean, cov * self.y_std ** 2

    def predict_average(self, policies_x, days):
        """Mean over planting days of each policy's margin: mean, sd of that mean, per-day means."""
        seasons = [season_features(d) for d in days]
        means, sds, per_day = [], [], []
        for x in np.asarray(policies_x, dtype=float):
            rows = np.array([list(x) + s for s in seasons])
            mean, cov = self.predict_joint(rows)
            weights = np.full(len(days), 1.0 / len(days))
            means.append(float(weights @ mean))
            sds.append(math.sqrt(max(float(weights @ cov @ weights), 1e-12)))
            per_day.append(mean.tolist())
        return np.array(means), np.array(sds), per_day


class Toolbox:
    def __init__(self, session, tool_seed: int, *, max_calls: int = LIMITS['max_calls']):
        if isinstance(tool_seed, bool) or not isinstance(tool_seed, int) or not 0 <= tool_seed < 2 ** 63:
            raise ValueError('tool seed must be a non-negative integer')
        self._session = session
        self._seed = tool_seed
        self._max_calls = max_calls
        self.calls = 0

    @staticmethod
    def catalog() -> dict:
        return {'version': TOOLBOX_VERSION, 'limits': dict(LIMITS), 'tools': CATALOG}

    # ── plumbing ───────────────────────────────────────────────────────────
    def call(self, name: str, args: dict | None = None) -> dict:
        if name not in CATALOG:
            # Never echo model-supplied text back; see the catalog for valid names.
            raise ToolError('unknown tool; use a name from the tool catalog')
        args = {} if args is None else args
        if not isinstance(args, dict) or not set(args) <= set(CATALOG[name]['args']):
            raise ToolError(f'invalid arguments for {name}')
        if self.calls >= self._max_calls:
            raise ToolError('analysis tool budget exhausted')
        self.calls += 1
        task, transcript = self._session.task, self._session.transcript
        result = getattr(self, '_' + name)(task, PublicHistory(transcript), args)
        return {'tool': name, 'version': TOOLBOX_VERSION, 'result': _round(result)}

    def _rng(self, name, payload):
        key = hashlib.sha256(canonical({'tool': name, 'input': payload}).encode()).digest()
        return np.random.default_rng([self._seed, int.from_bytes(key[:8], 'big')])

    @staticmethod
    def _fields(task):
        return task['policy']['fields']

    def _scale(self, task, policy):
        fields = self._fields(task)
        if not isinstance(policy, dict) or set(policy) != set(fields):
            raise ToolError('each policy must contain exactly the policy fields')
        out = []
        for name, spec in fields.items():
            value = policy[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ToolError(f'policy field {name} must be a finite number')
            out.append((value - spec['min']) / (spec['max'] - spec['min']))
        return out

    def _completed(self, task, history):
        rows = []
        for run in history.runs.values():
            if run.get('reason') == 'normal_completion' and run.get('policy') and run.get('start_day') is not None:
                rows.append((self._scale(task, run['policy']) + season_features(run['start_day']),
                             contribution_margin(run['accrued'], run['event_cost_eur_m2'], task['economics']), run))
        rows.sort(key=lambda row: (row[2]['unit'], row[2]['run_index']))
        return rows

    def _model(self, task, history):
        rows = self._completed(task, history)
        if len(rows) < LIMITS['min_completed_runs_for_model']:
            raise ToolError(f"need at least {LIMITS['min_completed_runs_for_model']} completed crops; have {len(rows)}")
        x = np.array([row[0] for row in rows], dtype=float)
        y = np.array([row[1] for row in rows], dtype=float)
        rng = self._rng('fit_outcome_model', {'x': x.tolist(), 'y': y.tolist()})
        return OutcomeModel(x, y, rng, LIMITS['gp_restarts']), rows

    # ── tools ──────────────────────────────────────────────────────────────
    def _runs(self, task, history, args):
        table = []
        for (unit, run_index) in sorted(history.runs):
            run = history.runs[(unit, run_index)]
            row = {k: run.get(k) for k in ('unit', 'run_index', 'policy', 'start_day', 'status', 'reason', 'closed_day')}
            if run.get('accrued') is not None:
                row['accrued_per_m2'] = run['accrued']
                row['contribution_margin_eur_m2'] = contribution_margin(run['accrued'], run['event_cost_eur_m2'],
                                                                        task['economics'])
            table.append(row)
        return {'runs': table, 'completed_crops': sum(r.get('reason') == 'normal_completion' for r in table)}

    def _series_summary(self, task, history, args):
        unit, variable = args.get('unit'), args.get('variable')
        if type(unit) is not int or variable not in task['observations']['channels']:
            raise ToolError('series_summary needs an integer unit and a public channel name')
        series = history.records.get((unit, variable), {})
        lo, hi = args.get('start_day', -math.inf), args.get('end_day', math.inf)
        days: dict[int, list[float]] = {}
        for t in sorted(series):
            day = int(t // 86400)
            if lo <= day <= hi:
                days.setdefault(day, []).append(series[t])
        if len(days) > LIMITS['max_summary_days']:
            raise ToolError('too many days; narrow the range')
        return {'unit': unit, 'variable': variable, 'unit_of_measure': task['observations']['channels'][variable],
                'days': [{'day': d, 'n': len(v), 'mean': float(np.mean(v)), 'min': min(v), 'max': max(v)}
                         for d, v in sorted(days.items())]}

    def _fit_outcome_model(self, task, history, args):
        model, rows = self._model(task, history)
        return {'completed_crops_used': len(rows),
                'mean_margin_eur_m2': model.y_mean,
                'length_scales_fraction_of_range': dict(zip(list(self._fields(task)) + list(SEASON_INPUTS),
                                                            model.lengthscales.tolist())),
                'noise_sd_eur_m2': math.sqrt(model.sn2) * model.y_std,
                'leave_one_out_rmse_eur_m2': model.loo_rmse}

    def _predict(self, task, history, args):
        policies = args.get('policies')
        if not isinstance(policies, list) or not 0 < len(policies) <= LIMITS['max_predict_policies']:
            raise ToolError(f"predict needs 1 to {LIMITS['max_predict_policies']} policies")
        xs = np.array([self._scale(task, p) for p in policies], dtype=float)
        model, rows = self._model(task, history)
        day = args.get('planting_day')
        if day is not None:
            if isinstance(day, bool) or not isinstance(day, (int, float)) or not 0 <= day < 365:
                raise ToolError('planting_day must be a calendar day from 0 to 364')
            rows_x = np.array([list(x) + season_features(day) for x in xs])
            mean, sd_mean, sd_obs = model.predict(rows_x)
            return {'completed_crops_used': len(rows), 'planting_day': day,
                    'predictions': [{'policy': p, 'margin_mean_eur_m2': float(m), 'sd_of_mean': float(a),
                                     'sd_of_single_crop': float(b)}
                                    for p, m, a, b in zip(policies, mean, sd_mean, sd_obs)]}
        days = scoring_days(task)
        mean, sd, per_day = model.predict_average(xs, days)
        return {'completed_crops_used': len(rows), 'scoring_planting_days': days,
                'predictions': [{'policy': p, 'scored_mean_margin_eur_m2': float(m), 'sd_of_scored_mean': float(s),
                                 'margin_by_planting_day': dict(zip([str(d) for d in days], pd))}
                                for p, m, s, pd in zip(policies, mean, sd, per_day)]}

    def _space_filling_candidates(self, task, history, args):
        from scipy.stats import qmc
        fields = self._fields(task)
        n = args.get('n')
        fixed = args.get('fixed') or {}
        if type(n) is not int or not 0 < n <= LIMITS['max_candidates']:
            raise ToolError(f"n must be an integer from 1 to {LIMITS['max_candidates']}")
        if not isinstance(fixed, dict) or not set(fixed) <= set(fields):
            raise ToolError('fixed must map policy fields to values')
        for name, value in fixed.items():
            spec = fields[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not spec['min'] <= value <= spec['max']:
                raise ToolError(f'fixed value for {name} is outside its range')
        free = [name for name in fields if name not in fixed]
        rng = self._rng('space_filling_candidates', {'n': n, 'fixed': fixed})
        out = []
        if free:
            sampler = qmc.Sobol(len(free), scramble=True, seed=rng)
            m = max(1, math.ceil(math.log2(4 * n)))
            while len(out) < n:
                # Sobol balance needs power-of-two totals: first 2^m points, then doubling.
                batch = sampler.random_base2(m)
                m = int(math.log2(sampler.num_generated))
                for point in batch:
                    policy = dict(fixed)
                    for name, u in zip(free, point):
                        spec = fields[name]
                        policy[name] = round(spec['min'] + float(u) * (spec['max'] - spec['min']), 2)
                    if policy['night_temperature_c'] <= policy['day_temperature_c']:
                        out.append({name: policy[name] for name in fields})
                        if len(out) == n:
                            break
        elif fixed['night_temperature_c'] <= fixed['day_temperature_c']:
            out = [{name: fixed[name] for name in fields}]
        else:
            raise ToolError('fixed policy violates night_temperature_c <= day_temperature_c')
        return {'candidates': out}
