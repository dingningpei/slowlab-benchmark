"""The experimental design submitted to the environment, and the rejection semantics."""
from __future__ import annotations
from dataclasses import dataclass, field


class RejectCode:
    OK = "OK"
    SCHEMA = "SCHEMA"
    INFEASIBLE_GRANULARITY = "INFEASIBLE_GRANULARITY"
    UNIT_CONFLICT = "UNIT_CONFLICT"
    UNKNOWN_UNIT = "UNKNOWN_UNIT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    FACTOR_OUT_OF_RANGE = "FACTOR_OUT_OF_RANGE"
    UNIT_RECOVERING = "UNIT_RECOVERING"


@dataclass
class Rejection:
    code: str
    detail: str
    fields: list[str] = field(default_factory=list)

    def __bool__(self):                       # falsy means rejected
        return self.code == RejectCode.OK

    def __repr__(self):
        return f"<{self.code}: {self.detail}>"


OK = Rejection(RejectCode.OK, "")


@dataclass
class Design:
    """The simplified W1 form. interim_rules / analysis_plan / measurement_plan belong to the full one."""
    treatments: dict[str, dict[str, float]]      # tid -> {factor: value}
    allocation: dict[str, list[str]]             # tid -> [unit_id]
    randomization_seed: int | None = None
    question: str = ""
    predictions: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def unit_ids(self) -> list[str]:
        return [u for uids in self.allocation.values() for u in uids]

    def replicates(self) -> dict[str, int]:
        return {t: len(u) for t, u in self.allocation.items()}
