"""SLOWLAB — a benchmark for scientific agents under slow, costly, irreversible experiments.

Ground truth is the reduced state-variable TOMGRO crop model (Jones, Kenig &
Vallejos 1999) plus a greenhouse economic model. It is deliberately not a
synthetic Gaussian-process surface.
"""
from .factors import FactorSpec
from .facility import Facility, Unit
from .design import Design, Rejection, RejectCode
from .validity import score_validity, ValidityReport
from .tomgro import TomgroModel, TomgroParams
from .economics import EconomicModel, TomgroProfitModel, sample_site_econ
from .world import ManagedTomgro, MANAGEMENT_FACTORS, sample_instance_params
from .tasks import TASKS, Task
from .env import SlowLabEnv, EpisodeResult, Observation

# ── Environment version ─────────────────────────────────────
# Frozen 2026-09-10. See ENVIRONMENT_v1.0.md.
#
# What "frozen" means: the environment is a fixed artefact. Known defects are
# registered in sections A-E of ENVIRONMENT_v1.0.md, published rather than all
# fixed -- a benchmark's value is comparability across agents, not correctness
# of the simulator.
#
# The stopping rule: a defect is fixed if it changes a claim the paper makes,
# and recorded if it does not.
#
# Any change to ground truth (numbers in world / economics / tomgro / tasks)
# must bump this version, and invalidates every episode already run.
# tests/test_frozen.py guards this with a set of fingerprints.
ENV_VERSION = "1.0.0"

__version__ = "1.0.0"
__all__ = ["FactorSpec", "Facility", "Unit", "Design", "Rejection", "RejectCode",
           "score_validity", "ValidityReport", "TomgroModel", "TomgroParams",
           "EconomicModel", "TomgroProfitModel", "ManagedTomgro",
           "MANAGEMENT_FACTORS", "sample_instance_params", "TASKS", "Task",
           "SlowLabEnv", "EpisodeResult", "Observation",
           "sample_site_econ", "ENV_VERSION"]
