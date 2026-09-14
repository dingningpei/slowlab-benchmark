"""Baseline registry. Adding an agent needs nothing but an @register."""
from __future__ import annotations
from .agents import RandomAgent, DoEAgent, GPUCBAgent, TransferAgent
from .strong_baselines import (BASELINE_PROTOCOL_VERSION,
                               ConstraintAwareBatchBOAgent,
                               PriorOptimalFixedAgent, SiteOracleAgent)

_REG: dict = {}


def register(name):
    def deco(fn):
        _REG[name] = fn
        return fn
    return deco


register("random_spread")(RandomAgent)
register("classical_doe")(DoEAgent)
register("gp_ucb_rep1")(lambda: GPUCBAgent(reps=1))
register("gp_ucb_rep2")(lambda: GPUCBAgent(reps=2))
register("gp_ucb_profit")(lambda: TransferAgent("profit"))
register("gp_ucb_components")(lambda: TransferAgent("components"))


register("split_plot_doe")(DoEAgent)
register("constraint_aware_batch_bo")(ConstraintAwareBatchBOAgent)
register("component_reconstruction")(lambda: TransferAgent("components"))
register("prior_optimal_fixed")(PriorOptimalFixedAgent)
register("site_oracle")(SiteOracleAgent)


def make_agent(name: str):
    if name not in _REG:
        raise KeyError(f"unknown agent {name!r}; available: {sorted(_REG)}")
    return _REG[name]()


def available() -> list[str]:
    return sorted(_REG)


def describe(name: str) -> dict:
    """Machine-readable permission label for tables and experiment runners."""
    agent = make_agent(name)
    return {
        "name": name,
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "privileged": bool(getattr(agent, "privileged", False)),
        "privilege_reason": getattr(agent, "privilege_reason", None),
    }
