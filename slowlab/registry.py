"""Baseline registry. Adding an agent needs nothing but an @register."""
from __future__ import annotations
from .agents import RandomAgent, DoEAgent, GPUCBAgent, TransferAgent

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


def make_agent(name: str):
    if name not in _REG:
        raise KeyError(f"unknown agent {name!r}; available: {sorted(_REG)}")
    return _REG[name]()


def available() -> list[str]:
    return sorted(_REG)
