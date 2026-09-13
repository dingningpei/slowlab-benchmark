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
from .env import SlowLabEnv, EpisodeResult, Measurement, Observation, RecommendationUpdate

# ── Environment version ─────────────────────────────────────
# Version 2 adds persistent within-cycle state and timestamped measurements. See
# ENVIRONMENT_v2.0.md. The terminal forward model remains numerically identical,
# but v1 transcripts do not contain the new decision opportunities.
#
# What "frozen" means: a released version is a fixed artefact. Version-specific
# changes are recorded rather than silently applied to existing results -- a
# benchmark's value depends on comparability across agents.
#
# The stopping rule: a defect is fixed if it changes a claim the paper makes,
# and recorded if it does not.
#
# Any change to ground truth (numbers in world / economics / tomgro / tasks)
# must bump this version, and invalidates every episode already run.
# tests/test_frozen.py guards this with a set of fingerprints.
ENV_VERSION = "2.0.0"

__version__ = "2.0.0"
__all__ = ["FactorSpec", "Facility", "Unit", "Design", "Rejection", "RejectCode",
           "score_validity", "ValidityReport", "TomgroModel", "TomgroParams",
           "EconomicModel", "TomgroProfitModel", "ManagedTomgro",
           "MANAGEMENT_FACTORS", "sample_instance_params", "TASKS", "Task",
           "SlowLabEnv", "EpisodeResult", "Measurement", "Observation",
           "RecommendationUpdate",
           "sample_site_econ", "ENV_VERSION"]
