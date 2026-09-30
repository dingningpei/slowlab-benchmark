"""SlowLab — a benchmark for experimental-design agents when experiments are slow,
noisy, costly and irreversible.

This package is the Version 2.3 campaign environment: an event-driven, multi-compartment
greenhouse in which an agent starts, observes, stops and replants crops over a 365-day
campaign and finally recommends a management policy. Ground truth is the pinned
GreenLight model forced by archived real weather; see RESEARCH_PLAN.md.

The frozen Version 2.1 environment (TOMGRO, synchronous rounds) and its paper live under
``legacy/v2.1`` and are not imported from here.
"""
from .task_contract import validate_policy
from .campaign_executor import CampaignExecutor
from .policy import Policy
from .feedback_view import FeedbackView
from .online_observations import OnlineObservations, PackedOnlineObservations
from .resources import ResourceLedger
from .frozen_paths import frozen_path

__version__ = "2.3.0.dev0"
__all__ = ["validate_policy", "Policy", "CampaignExecutor", "FeedbackView", "OnlineObservations",
           "PackedOnlineObservations", "ResourceLedger", "frozen_path", "__version__"]
