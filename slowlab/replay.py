"""Readers for fixed-history design x reader replay."""
from __future__ import annotations

import json
import numpy as np

from .agents import GP, _cands
from .strong_baselines import BlockAwareReader, completed_history_arrays


class OrdinaryGPReader:
    name = "ordinary_gp"
    privileged = False

    def recommend(self, env, candidates=None):
        X, y = env.as_arrays()
        if not len(y):
            return np.full(env.task.d, 0.5)
        points = candidates if candidates is not None else _cands(
            env.task.d, 8000, np.random.default_rng(7))
        model = GP(noise=max(0.15, env.task.plant_cv)).fit(X, y)
        return np.asarray(points[int(np.argmax(model.predict(points)[0]))], float)


class ComponentGPReader:
    """Reader using the public accounting components, including Transfer shock."""
    name = "component_gp"
    privileged = False

    def recommend(self, env, candidates=None):
        X, fields = env.component_arrays()
        if not len(X):
            return np.full(env.task.d, 0.5)
        points = candidates if candidates is not None else _cands(
            env.task.d, 8000, np.random.default_rng(7))
        noise = max(0.15, env.task.plant_cv)
        revenue = GP(noise=noise).fit(X, fields["rev_rate"])
        energy = GP(noise=0.02).fit(X, fields["energy_cost_rate"])
        other = GP(noise=0.02).fit(X, fields["other_cost_rate"])
        mr = revenue.predict(points)[0]
        me = energy.predict(points)[0]
        mo = other.predict(points)[0]
        shock = 1.0
        if getattr(env.task, "transfer_energy_shock", None) is not None:
            query = env.transfer_query()
            shock = query["energy_shock"] / max(query["old_energy_shock"], 1e-9)
        return np.asarray(points[int(np.argmax(mr - shock * me - mo))], float)


class PrivilegedSiteReader:
    name = "site_oracle"
    privileged = True

    def recommend(self, env, candidates=None):
        del candidates
        return np.asarray(env.truth.oracle()[0], float)


class FixedHistoryLLMReader:
    """Recommendation-only LLM reader; it cannot alter the replayed design."""
    name = "llm_reader"
    privileged = False

    def __init__(self, complete):
        self.complete = complete
        self.records = []

    def recommend(self, env, candidates=None):
        del candidates
        from .llm import SYSTEM_V2, _extract_json, _facility_text, _observations_text, _to_point
        fields = ", ".join(f'"{factor.name}": <number>' for factor in env.task.factors)
        user = "\n\n".join([
            _facility_text(env),
            _observations_text(env),
            "The experimental designs and observations above are fixed. You cannot run, "
            "change, or repeat an experiment. Read this history and return only JSON:",
            "{\n  \"reasoning\": \"brief reason grounded in the observations\",\n"
            f"  \"recommendation\": {{{fields}}}\n}}",
        ])
        reply = self.complete([{"role": "system", "content": SYSTEM_V2},
                               {"role": "user", "content": user}])
        blob = _extract_json(reply)
        self.records.append({"history_events": len(env.history()), "reply": reply,
                             "reasoning": str(blob.get("reasoning", ""))[:800]})
        return _to_point(env, blob["recommendation"])


def calibrated_block_reader(task_name, config):
    parameters = config.get("task_parameters", {}).get(task_name, {})
    return BlockAwareReader(**parameters)
