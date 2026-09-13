"""Versioned execution events and deterministic recovery of v1 LLM transcripts.

The v1 transcripts contain every assistant attempt, including rejected drafts.
Scientific analyses must consume only designs that the environment accepted and
completed.  This module replays the original interaction loop and records the
distinction explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import pathlib
import re
from typing import Any, Iterable

import numpy as np

from .design import Design
from .env import Observation, SlowLabEnv
from .llm import _extract_json, _to_design, _to_point


EVENT_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class ExecutionEvent:
    """One ordered fact in an episode's execution history."""

    episode_id: str
    task: str
    model: str
    site_seed: int
    seq: int
    kind: str
    round: int
    timestamp: str | None = None
    status: str | None = None
    reason: str | None = None
    attempt: int | None = None
    design_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryResult:
    schema_version: str
    episode_id: str
    model: str
    task: str
    seed: int
    events: list[ExecutionEvent]
    counts: dict[str, int]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "model": self.model,
            "task": self.task,
            "seed": self.seed,
            "counts": dict(self.counts),
            "errors": list(self.errors),
            "events": [event.to_dict() for event in self.events],
        }


def _design_payload(design: Design) -> dict[str, Any]:
    return {
        "treatments": {
            str(tid): {str(name): float(value) for name, value in values.items()}
            for tid, values in design.treatments.items()
        },
        "allocation": {
            str(tid): [str(unit) for unit in units]
            for tid, units in design.allocation.items()
        },
        "randomization_seed": design.randomization_seed,
        "question": design.question,
    }


def _observation_payload(observation: Observation) -> dict[str, Any]:
    return {
        "design_id": int(observation.design_id),
        "treatment": str(observation.treatment),
        "unit_id": str(observation.unit_id),
        "value": float(observation.value),
        "rev_rate": float(observation.rev_rate),
        "cost_rate": float(observation.cost_rate),
        "energy_cost_rate": float(observation.energy_cost_rate),
        "other_cost_rate": float(observation.other_cost_rate),
    }


def _prompt_kind(user: str) -> str:
    low = user.lower()
    if "energy prices have now changed" in low or "price shock" in low:
        return "transfer"
    if (
        "all rounds are finished" in low
        or ("no further" in low and "experiments are possible" in low)
    ):
        return "final"
    if "ROUNDS REMAINING:" in user:
        return "design"
    return "unknown"


def recover_transcript(
    task,
    seed: int,
    transcript: Iterable[dict[str, str]],
    *,
    max_retries: int = 3,
    model: str = "unknown",
    episode_id: str | None = None,
) -> RecoveryResult:
    """Replay one v1 transcript into explicit execution events.

    The replay invokes the frozen environment for feasibility and observations.
    Invalid replies consume an attempt but no experimental budget.  Three failed
    attempts forfeit exactly one round, matching :class:`LLMAgent`.
    """

    env = SlowLabEnv(task, seed=seed)
    events: list[ExecutionEvent] = []
    errors: list[str] = []
    counts: dict[str, int] = {}
    attempts = 0
    best = np.full(task.d, 0.5)
    stopped = False
    episode_id = episode_id or f"{model}/{task.name}/s{seed}"

    def emit(kind: str, *, attempt=None, design_id=None, payload=None,
             round_index=None) -> None:
        value = payload or {}
        counts[kind] = counts.get(kind, 0) + 1
        events.append(ExecutionEvent(
            episode_id=episode_id, task=str(task.name), model=str(model),
            site_seed=int(seed), seq=len(events), kind=kind,
            round=int(env.round if round_index is None else round_index), attempt=attempt,
            design_id=design_id, status=kind,
            reason=value.get("reason") or value.get("detail"), payload=value))

    def fail_attempt(kind: str, detail: str, raw_reply: str) -> None:
        nonlocal attempts
        attempts += 1
        emit(kind, attempt=attempts,
             payload={"detail": detail, "raw_reply": raw_reply})
        if attempts == max_retries:
            emit("forfeited", attempt=attempts,
                 payload={"reason": "retries_exhausted"})
            env.forfeit_round()
            attempts = 0

    for turn_index, turn in enumerate(transcript):
        user = str(turn.get("user", ""))
        reply = str(turn.get("assistant", ""))
        prompt_kind = _prompt_kind(user)

        if prompt_kind == "unknown":
            errors.append(f"turn {turn_index}: unrecognised prompt")
            emit("unresolved", payload={"turn": turn_index, "raw_reply": reply})
            continue

        if prompt_kind in {"final", "transfer"}:
            try:
                blob = _extract_json(reply)
                point = _to_point(env, blob["recommendation"])
            except Exception as exc:
                emit("parse_failed", payload={
                    "phase": prompt_kind, "detail": str(exc), "raw_reply": reply})
                continue
            emit("recommended", payload={
                "phase": prompt_kind, "point": [float(v) for v in point]})
            if prompt_kind == "final":
                best = point
            continue

        if stopped:
            errors.append(f"turn {turn_index}: design prompt follows stop action")

        attempt = attempts + 1
        try:
            blob = _extract_json(reply)
        except Exception as exc:
            fail_attempt("parse_failed", str(exc), reply)
            continue

        if bool(blob.get("stop", False)):
            emit("stopped", attempt=attempt, payload={"raw": blob})
            if "current_best" in blob:
                try:
                    best = _to_point(env, blob["current_best"])
                    emit("recommended", attempt=attempt, payload={
                        "phase": "stop", "point": [float(v) for v in best]})
                except Exception as exc:
                    emit("parse_failed", attempt=attempt, payload={
                        "phase": "stop_recommendation", "detail": str(exc)})
            stopped = True
            continue

        try:
            design = _to_design(env, blob)
        except Exception as exc:
            fail_attempt("parse_failed", str(exc), reply)
            continue

        emit("proposed", attempt=attempt, payload=_design_payload(design))
        accepted = env.submit_design(design)
        if not isinstance(accepted, int):
            attempts += 1
            emit("rejected", attempt=attempt, payload={
                "code": str(accepted.code), "detail": str(accepted.detail),
                "fields": [str(v) for v in accepted.fields]})
            if attempts == max_retries:
                emit("forfeited", attempt=attempts,
                     payload={"reason": "retries_exhausted"})
                env.forfeit_round()
                attempts = 0
            continue

        emit("accepted", attempt=attempt, design_id=accepted,
             payload=_design_payload(design))
        completed_round = env.round
        observations = env.advance()
        emit("completed", attempt=attempt, design_id=accepted, payload={
            "observations": [_observation_payload(o) for o in observations]},
             round_index=completed_round)
        if "current_best" in blob:
            try:
                best = _to_point(env, blob["current_best"])
            except Exception:
                pass
        emit("recommended", attempt=attempt, design_id=accepted, payload={
            "phase": "interim", "point": [float(v) for v in best]},
             round_index=completed_round)
        attempts = 0

    if attempts:
        errors.append(
            f"transcript ended with {attempts} unresolved failed attempt(s) in round {env.round}")

    return RecoveryResult(
        schema_version=EVENT_SCHEMA_VERSION,
        episode_id=episode_id, model=str(model),
        task=str(task.name), seed=int(seed), events=events,
        counts=counts, errors=errors)


def executed_designs(result: RecoveryResult) -> list[dict[str, Any]]:
    """Return only accepted designs that have a matching completion event."""

    completed = {
        event.design_id for event in result.events
        if event.kind == "completed" and event.design_id is not None
    }
    return [
        event.payload for event in result.events
        if event.kind == "accepted" and event.design_id in completed
    ]


def completed_design_payloads(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract accepted designs whose execution has a completion record."""

    if document.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported event schema {document.get('schema_version')!r}"
        )
    events = document.get("events")
    if not isinstance(events, list):
        raise ValueError("event document has no events list")
    completed = {
        event.get("design_id") for event in events
        if event.get("kind") == "completed" and event.get("design_id") is not None
    }
    accepted = [
        event for event in events
        if event.get("kind") == "accepted" and event.get("design_id") in completed
    ]
    if len(accepted) != len(completed):
        raise ValueError("accepted/completed design ids are not one-to-one")
    return [event["payload"] for event in accepted]


def design_arrays_from_events(path: str | pathlib.Path, env):
    """Load ``(points, chamber_blocks)`` from completed execution events only."""

    document = json.loads(pathlib.Path(path).read_text())
    return design_arrays_from_payloads(completed_design_payloads(document), env)


def design_arrays_from_payloads(designs: Iterable[dict[str, Any]], env):
    """Convert normalized completed-design payloads into scoring arrays."""

    names = [factor.name for factor in env.task.factors]
    out = []
    for design in designs:
        points: list[list[float]] = []
        blocks: list[int] = []
        for treatment_id, units in design["allocation"].items():
            treatment = design["treatments"][treatment_id]
            point = [float(treatment[name]) for name in names]
            for unit in units:
                match = re.fullmatch(r"c(\d+)l(\d+)", str(unit))
                if not match or str(unit) not in env.facility._by_id:
                    raise ValueError(f"unknown unit id in completed event: {unit!r}")
                points.append(point)
                blocks.append(int(match.group(1)))
        if not points:
            raise ValueError("completed design has no allocated units")
        out.append((np.asarray(points, float), np.asarray(blocks, int)))
    return out


def design_arrays_from_environment(env):
    """Use the same completed-design contract for scripted agents.

    A submitted design is included only when the environment has produced at
    least one observation carrying its design id. This excludes pending designs
    and makes reference-policy scoring consistent with recovered LLM events.
    """

    completed = {int(observation.design_id) for observation in env.observations()}
    payloads = [
        _design_payload(design) for design_id, design in enumerate(env._designs)
        if design_id in completed
    ]
    return design_arrays_from_payloads(payloads, env)


def reconcile_format_failures(result: RecoveryResult, expected: int) -> int:
    """Record v1 failures counted in a summary but absent from its transcript.

    Their content and exact position cannot be reconstructed, so the generated
    events carry their provenance and omit a fabricated raw response.
    """

    recovered = result.counts.get("parse_failed", 0)
    missing = int(expected) - recovered
    if missing <= 0:
        return 0
    last_round = result.events[-1].round if result.events else 0
    for _ in range(missing):
        result.events.append(ExecutionEvent(
            episode_id=result.episode_id,
            task=result.task,
            model=result.model,
            site_seed=result.seed,
            seq=len(result.events),
            kind="parse_failed",
            round=last_round,
            status="parse_failed",
            reason="failure counted by v1; assistant response was not persisted",
            payload={
                "inferred": True,
                "source": "episode_summary.format_failures",
                "detail": "failure counted by v1; assistant response was not persisted",
            },
        ))
    result.counts["parse_failed"] = recovered + missing
    result.errors.append(
        f"{missing} format failure(s) exist only in the episode summary"
    )
    return missing
