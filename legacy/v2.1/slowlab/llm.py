"""The LLM agent harness -- turns SlowLab into a turn-based game an LLM can play directly.

Design principles
--------
1. **The model is pluggable.** All that is required is a callable
   `complete(messages) -> str`. Any API (OpenAI, Anthropic, local) connects with
   a thin wrapper, and this file depends on no SDK.
2. **The environment states facts and gives no advice.** The prompt carries the
   facility structure, the hierarchy constraint, the budget and the observations
   so far; there is no hint such as "you should replicate" -- that is precisely
   the capability under test.
3. **Format failures are retryable and free.** A JSON parse failure or a
   mechanically infeasible design feeds the rejection reason back, up to
   `max_retries` times. These do not consume experimental budget (matching
   validate_design's semantics) but are counted and reported: the ability to
   emit format and the ability to design experiments must be read separately.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .design import Design

Completer = Callable[[list], str]


SYSTEM = """You are an agronomist running experiments in a real greenhouse.

You design experiments; you do not control the crop day to day. Each round you
submit a complete design and the growing cycle then runs to completion. You see
nothing until it ends. Cycles are months long and the facility is finite, so a
round you spend badly is a season you cannot get back, and a round you decline to
spend costs nothing.

Answer with a single JSON object and nothing else."""


SYSTEM_V2 = """You are an agronomist running experiments in a real greenhouse.

You submit a fixed design for each crop cycle. While that cycle is running you
may choose integer days to enter the facility and read supported sensors and
ledgers. Final gross margin is unavailable until the cycle ends, and observing
does not change treatments already submitted. The facility remains occupied
throughout the cycle.

Answer with a single JSON object and nothing else."""


def _facility_text(env) -> str:
    t, f = env.task, env.facility
    lines = [
        f"BUDGET: {t.n_rounds} rounds. Each round you may use AT MOST "
        f"{t.units_per_round} units, and a round occupies them for "
        f"{t.duration_days} days.",
        f"FACILITY: {f.n_chambers} chambers x {f.loops_per_chamber} hydroponic "
        f"loops = {len(f.units)} units in total, of which you may use "
        f"{t.units_per_round} per round. Unit ids look like c0l1 "
        f"(chamber 0, loop 1).",
        "",
        "FACTORS (give values in the real units shown; the environment rescales):",
    ]
    for g in t.factors:
        lines.append(f"  - {g.name}: {g.low} to {g.high}, set per {g.control_level.upper()}")
    lines += [
        "",
        "HARD PHYSICAL CONSTRAINT: lamps and heating serve a whole chamber, while",
        "density, pruning and defoliation are set per loop. A CHAMBER-level factor",
        "therefore cannot take two different values inside the same chamber. A design",
        "that violates this is rejected (rejection is free but wastes your turn's",
        "thinking, not your budget).",
        "",
        "OBJECTIVE: net profit per square metre per day. Each unit returns one number.",
    ]
    return "\n".join(lines)


def _observations_text(env) -> str:
    """Serialisation of an observation.

    A past asymmetry in the interface: this used to write out only o.value (total
    gross margin) while Observation has five fields -- value / rev_rate /
    energy_cost_rate / other_cost_rate / cost_rate. Scripted agents read all of
    them and the LLM got two. The consequence was not merely "slightly less
    information": the entire point of the Transfer task is that an agent which
    modelled revenue and cost separately can re-solve after a price change without
    running anything, and the LLM could not see the components at all, so **that
    claim was false for the LLM interface.** Both sides now see the same
    observation object.
    """
    history = env.history()
    completed = [e for e in history if e.kind == "completed"]
    measured = [e for e in history if e.kind == "measured"]
    if not completed and not measured:
        return "VISIBLE OBSERVATIONS SO FAR: none."
    designs = {e.design_id: e.payload for e in history if e.kind == "submitted"}
    sections = []
    if measured:
        rows = ["WITHIN-CYCLE MEASUREMENTS SO FAR:"]
        for event in measured:
            p = event.payload
            rows.append(
                f"  absolute day {event.absolute_day:g}  round {event.round + 1} "
                f"day {event.day}  {p['unit_id']}  {p['modality']} "
                f"= {p['value']!r} {p['unit']}  ({p['method']})")
        sections.append("\n".join(rows))
    if not completed:
        sections.append("TERMINAL OBSERVATIONS SO FAR: none; the running crop has not finished.")
        return "\n\n".join(sections)
    rows = ["OBSERVATIONS SO FAR. All rates are per m2 per day, in EUR.",
            "gross margin = revenue - energy cost - other cost.",
            "  energy cost is electricity + gas; other cost is CO2, transplants, labour.",
            ""]
    for event in completed:
        p = event.payload
        d = designs[event.design_id]
        fv = d["treatments"][p["treatment"]]
        setting = ", ".join(
            f"{g.name}={g.denorm(fv[g.name]):.4g}" for g in env.task.factors)
        parts = (f"  [revenue {p['rev_rate']!r}"
                 f"  energy {p['energy_cost_rate']!r}"
                 f"  other {p['other_cost_rate']!r}]")
        rows.append(f"  round {event.round + 1}  {p['unit_id']}  {setting}"
                    f"  -> margin {p['value']!r}{parts}")
    sections.append("\n".join(rows))
    return "\n\n".join(sections)


FINAL_ASK = """All rounds are finished and every observation is above. No further
experiments are possible.

State your final recommendation to the grower. Reply with JSON and nothing else:

{{
  "reasoning": "<two or three sentences>",
  "recommendation": {{{fields}}}
}}"""


TRANSFER_ASK = """Energy prices have now changed. Electricity and natural gas both
cost {mult:.1f} times what they did while you were running the campaign. Nothing
else moved: the tomato price, transplants, labour and purchased CO2 are unchanged.

No further experiments are possible. Using only the observations above, state the
recommendation you would now make to the grower. Reply with JSON and nothing else:

{{
  "reasoning": "<two or three sentences>",
  "recommendation": {{{fields}}}
}}"""


def _final_prompt(env) -> str:
    names = [g.name for g in env.task.factors]
    return "\n\n".join([
        _facility_text(env),
        _observations_text(env),
        FINAL_ASK.format(fields=", ".join(f'"{n}": <number>' for n in names)),
    ])


def _transfer_prompt(env) -> str:
    """The price-shock question.

    This used not to exist at all. LLMAgent unconditionally did
    self.x_transfer = best, i.e. reused the last recommendation -- every model was
    constructed as a profit-only agent, and the Transfer column measured nothing
    for LLMs. The scripted pair (gp_ucb_profit / gp_ucb_components) *was* asked,
    so the paper's Transfer claim rested entirely on the reference strategies.
    """
    names = [g.name for g in env.task.factors]
    q = env.transfer_query()
    mult = q["energy_shock"] / max(q["old_energy_shock"], 1e-9)
    return "\n\n".join([
        _facility_text(env),
        _observations_text(env),
        TRANSFER_ASK.format(mult=mult,
                            fields=", ".join(f'"{n}": <number>' for n in names)),
    ])


def _schema_text(env) -> str:
    names = [g.name for g in env.task.factors]
    example_units = [u.id for u in env.facility.units[:4]]
    return f"""Reply with JSON of exactly this shape:

{{
  "reasoning": "<two or three sentences on why this design>",
  "treatments": {{
    "A": {{{", ".join(f'"{n}": <number>' for n in names)}}},
    "B": {{{", ".join(f'"{n}": <number>' for n in names)}}}
  }},
  "allocation": {{"A": ["{example_units[0]}", "{example_units[1]}"],
                  "B": ["{example_units[2]}", "{example_units[3]}"]}},
  "randomization_seed": <integer>,
  "stop": false,
  "current_best": {{{", ".join(f'"{n}": <number>' for n in names)}}}
}}

"current_best" is what you would recommend to the grower if the experiment
stopped right now. You may use as many or as few treatments as you like.

You may also stop. Set "stop": true instead of giving treatments and allocation,
and the campaign ends there with your "current_best" as what you are scored on.
The rounds you did not use cost nothing and occupy no units. Stopping is a legitimate
answer whenever a further cycle would not change what you would advise."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in reply")
    depth, end = 0, None
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            end = i + 1
            break
    if end is None:
        raise ValueError("unterminated JSON object")
    return json.loads(text[start:end])


def _to_design(env, blob: dict) -> Design:
    t = env.task
    lo = {g.name: g.low for g in t.factors}
    hi = {g.name: g.high for g in t.factors}
    trt = {}
    for tid, fv in blob["treatments"].items():
        row = {}
        for g in t.factors:
            if g.name not in fv:
                raise ValueError(f"treatment {tid} is missing factor {g.name}")
            v = float(fv[g.name])
            row[g.name] = float(np.clip((v - lo[g.name]) / (hi[g.name] - lo[g.name]), 0, 1))
        trt[tid] = row
    alloc = {k: list(v) for k, v in blob["allocation"].items()}
    seed = blob.get("randomization_seed")
    return Design(treatments=trt, allocation=alloc,
                  randomization_seed=int(seed) if seed is not None else None,
                  question=str(blob.get("reasoning", ""))[:400])


def _to_point(env, fv: dict) -> np.ndarray:
    t = env.task
    return np.clip(np.array(
        [(float(fv[g.name]) - g.low) / (g.high - g.low) for g in t.factors]), 0, 1)


@dataclass
class LLMAgent:
    """Adapt a chat model into a SlowLab agent."""
    complete: Completer
    name: str = "llm"
    max_retries: int = 3
    transcript: list = field(default_factory=list)
    format_failures: int = 0
    infeasible_submissions: int = 0
    tools: bool = False          # backwards-compatible alias for tool_mode="both"
    tool_mode: str | None = None # bare / design / inference / both
    tool_delivery: str = "passive" # passive (legacy aid) / callable
    tool_use_records: list = field(default_factory=list)
    max_observation_times: int = 4
    prompt_variant: str = "standard"

    def effective_tool_mode(self) -> str:
        mode = self.tool_mode or ("both" if self.tools else "bare")
        if mode not in {"bare", "design", "inference", "both"}:
            raise ValueError(f"unknown tool mode {mode!r}")
        return mode

    def _system_prompt(self) -> str:
        if self.prompt_variant == "standard":
            return SYSTEM
        if self.prompt_variant == "constraint_checklist":
            return SYSTEM + "\n\n" + CONSTRAINT_CHECKLIST
        raise ValueError(f"unknown prompt variant {self.prompt_variant!r}")

    def _ask(self, env, feedback: str | None) -> dict:
        pieces = [
            _facility_text(env),
            _observations_text(env),
        ]
        if self.effective_tool_mode() != "bare":
            if self.tool_delivery == "passive":
                pieces.append(tool_text(env, self.effective_tool_mode()))
            elif self.tool_delivery == "callable":
                pieces.append(callable_tool_instructions(self.effective_tool_mode()))
            else:
                raise ValueError(f"unknown tool delivery {self.tool_delivery!r}")
        user = "\n\n".join(pieces + [
            f"ROUNDS REMAINING: {env.rounds_left}",
            f"UNITS YOU MAY DRAW FROM (choose at most {env.task.units_per_round}): "
            + ", ".join(env.available_units()),
            _schema_text(env),
        ] + ([f"YOUR PREVIOUS REPLY WAS REJECTED: {feedback}\nFix it and reply again."]
             if feedback else []))
        msgs = [{"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user}]
        return self._complete_action(env, msgs, phase="design")

    def _available_tools(self, phase: str) -> set[str]:
        mode = self.effective_tool_mode()
        tools = set()
        if phase == "design" and mode in {"design", "both"}:
            tools.add("propose_design")
        if mode in {"inference", "both"}:
            tools.add("analyze_history")
        return tools

    def _complete_action(self, env, msgs: list[dict], *, phase: str) -> dict:
        """Run the model/tool loop. A tool result is returned only after the model
        explicitly emits a call_tool action; availability alone never injects aid."""
        called = set()
        first_user = msgs[-1]["content"]
        turns = []
        for _ in range(3):
            reply = self.complete(msgs)
            turns.append({"assistant": reply})
            blob = _extract_json(reply)
            if blob.get("action") != "call_tool":
                self.transcript.append({"user": first_user, "assistant": reply,
                                        "tool_turns": turns[:-1]})
                return blob
            name = str(blob.get("tool", ""))
            available = self._available_tools(phase)
            if name not in available:
                result = {"error": "tool_unavailable", "available_tools": sorted(available)}
            elif name in called:
                result = {"error": "tool_already_called", "available_tools": sorted(available)}
            else:
                result = callable_tool_result(env, name)
                called.add(name)
                self.tool_use_records.append({
                    "round": int(env.round), "phase": phase, "mode": self.effective_tool_mode(),
                    "tool": name, "delivery": "application_layer_callable",
                    "invoked_by_model": True, "result": result,
                })
            tool_message = "TOOL RESULT:\n" + json.dumps(result, ensure_ascii=False)
            turns[-1]["tool_result"] = result
            msgs = msgs + [{"role": "assistant", "content": reply},
                           {"role": "user", "content": tool_message +
                            "\nNow return either another permitted call_tool action or the requested final JSON."}]
        raise ValueError("tool-call limit exceeded")

    def _record_tool_use(self, env, design, blob) -> None:
        mode = self.effective_tool_mode()
        if mode == "bare":
            return
        if self.tool_delivery == "callable":
            names = [factor.name for factor in env.task.factors]
            proposed = np.asarray([[values[name] for name in names]
                                   for values in design.treatments.values()], float)
            for rec in reversed(self.tool_use_records):
                if rec.get("round") != int(env.round) or rec.get("phase") != "design":
                    continue
                result = rec.get("result", {})
                if rec.get("tool") == "propose_design" and result.get("treatments"):
                    suggested = np.asarray([
                        [(float(v[name]) - env.task.factors[j].low) /
                         (env.task.factors[j].high - env.task.factors[j].low)
                         for j, name in enumerate(names)]
                        for v in result["treatments"].values()], float)
                    distances = np.sqrt(((proposed[:, None] - suggested[None, :]) ** 2).sum(-1))
                    dist = float(0.5 * (distances.min(1).mean() + distances.min(0).mean()))
                    rec["design_distance"] = dist
                    rec["design_adopted"] = dist <= 0.10
                if rec.get("tool") == "analyze_history" and result.get("posterior_best") and "current_best" in blob:
                    current = _to_point(env, blob["current_best"])
                    suggested = _to_point(env, result["posterior_best"])
                    dist = float(np.linalg.norm(current - suggested) / np.sqrt(env.task.d))
                    rec["recommendation_distance"] = dist
                    rec["inference_adopted"] = dist <= 0.05
            return
        artifacts = tool_artifacts(env)
        names = [factor.name for factor in env.task.factors]
        proposed = np.asarray([[values[name] for name in names]
                               for values in design.treatments.values()], float)
        suggested_design = artifacts.get("design") if mode in {"design", "both"} else None
        design_distance = None
        if suggested_design is not None and len(proposed):
            suggested = np.asarray([[values[name] for name in names]
                                    for tid, values in suggested_design.treatments.items()
                                    if tid in suggested_design.allocation], float)
            if len(suggested):
                distances = np.sqrt(((proposed[:, None] - suggested[None, :]) ** 2).sum(-1))
                design_distance = float(0.5 * (distances.min(1).mean() +
                                               distances.min(0).mean()))
        recommendation_distance = None
        if (mode in {"inference", "both"} and
                artifacts.get("inference") is not None and "current_best" in blob):
            try:
                current = _to_point(env, blob["current_best"])
                recommendation_distance = float(np.linalg.norm(
                    current - artifacts["inference"]) / np.sqrt(env.task.d))
            except Exception:
                pass
        self.tool_use_records.append({
            "round": int(env.round), "mode": mode,
            "delivery": "passive_prompt_output", "invoked_by_model": False,
            "design_distance": design_distance,
            "design_adopted": (None if design_distance is None else design_distance <= 0.10),
            "recommendation_distance": recommendation_distance,
            "inference_adopted": (None if recommendation_distance is None
                                  else recommendation_distance <= 0.05),
            "reasoning": str(blob.get("reasoning", ""))[:800],
        })

    def _execute_submitted_round(self, env, best):
        """Terminal-only control arm: reveal nothing until the crop finishes."""
        env.advance()
        env.interim_recommendation(best)
        return best

    def run(self, env):
        best = np.full(env.task.d, 0.5)
        self.stopped_early_at = None
        self.forfeited_rounds = 0
        while env.rounds_left > 0:
            feedback, submitted = None, False
            for _ in range(self.max_retries):
                try:
                    blob = self._ask(env, feedback)
                    # Abstain: submit no design and finish on current_best. Unused
                    # rounds cost nothing and occupy no units. The results show
                    # this can be the right move on a short budget, so it has to be
                    # an action an agent can reach -- otherwise we could observe the
                    # phenomenon but not measure the capability.
                    if bool(blob.get("stop", False)):
                        if "current_best" in blob:
                            try:
                                best = _to_point(env, blob["current_best"])
                            except Exception:
                                pass
                        self.stopped_early_at = env.round
                        self.x_transfer = best
                        return best
                    design = _to_design(env, blob)
                    self._record_tool_use(env, design, blob)
                except Exception as exc:                    # format or parse failure
                    self.format_failures += 1
                    feedback = f"could not parse your reply ({exc})"
                    continue
                rej = env.submit_design(design)
                if isinstance(rej, int):
                    if "current_best" in blob:
                        try:
                            best = _to_point(env, blob["current_best"])
                        except Exception:
                            pass
                    best = self._execute_submitted_round(env, best)
                    submitted = True
                    break
                self.infeasible_submissions += 1
                feedback = f"{rej.code}: {rej.detail}"
            if not submitted:
                # Retries exhausted: void this round but continue the campaign.
                # This used to be a break, which meant one failed submission ended
                # the whole episode -- the wrong severity of penalty.
                self.forfeited_rounds += 1
                env.forfeit_round()

        # The final round's observations are only revealed here. Without asking
        # once more, the last round's data would be paid for and never used.
        for _ in range(self.max_retries):
            try:
                user = _final_prompt(env)
                if self.tool_delivery == "callable" and self.effective_tool_mode() in {"inference", "both"}:
                    user += "\n\n" + callable_tool_instructions("inference")
                msgs = [{"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": user}]
                blob = self._complete_action(env, msgs, phase="final")
                best = _to_point(env, blob["recommendation"])
                env.record_recommendation(best, phase="final")
                break
            except Exception:
                self.format_failures += 1

        # ── The price-shock question ──
        # Asked only on tasks that declare a shock. The default falls back to best
        # (reuse the original recommendation), which is exactly profit-only
        # behaviour; being asked and choosing not to change is a different thing
        # from never being asked.
        self.x_transfer = best
        if getattr(env.task, "transfer_energy_shock", None) is not None:
            for _ in range(self.max_retries):
                try:
                    user = _transfer_prompt(env)
                    if self.tool_delivery == "callable" and self.effective_tool_mode() in {"inference", "both"}:
                        user += "\n\n" + callable_tool_instructions("inference")
                    msgs = [{"role": "system", "content": self._system_prompt()},
                            {"role": "user", "content": user}]
                    blob = self._complete_action(env, msgs, phase="transfer")
                    self.x_transfer = _to_point(env, blob["recommendation"])
                    break
                except Exception:
                    self.format_failures += 1
        return best


class WithinCycleLLMAgent(LLMAgent):
    """Version 2 LLM protocol with agent-chosen within-cycle observation days."""

    def _system_prompt(self) -> str:
        if self.prompt_variant == "standard":
            return SYSTEM_V2
        if self.prompt_variant == "constraint_checklist":
            return SYSTEM_V2 + "\n\n" + CONSTRAINT_CHECKLIST
        raise ValueError(f"unknown prompt variant {self.prompt_variant!r}")

    def _cycle_prompt(self, env, visits_left: int) -> str:
        active = env._active
        modalities = list(env._MEASUREMENT_SPECS)
        modalities.extend(f"setpoint:{f.name}" for f in env.task.factors)
        names = [f.name for f in env.task.factors]
        return "\n\n".join([
            _facility_text(env),
            _observations_text(env),
            f"RUNNING ROUND: {env.round + 1}; current crop day {active.state.day} "
            f"of {env.task.duration_days}. Observation times remaining: {visits_left}.",
            "SUPPORTED MODALITIES: " + ", ".join(modalities),
            f'''Choose the next action. To inspect, choose an integer day strictly after
the current day and before day {env.task.duration_days}. Reading a cached record
again returns the same value. Use null for units to inspect every running unit.

{{
  "action": "observe",
  "day": <integer>,
  "modalities": ["canopy_lai"],
  "units": null,
  "current_best": {{{", ".join(f'"{n}": <number>' for n in names)}}}
}}

To wait for the terminal result instead, reply with action "finish" and
current_best. Finishing does not cancel or modify the treatment.''',
        ])

    def _execute_submitted_round(self, env, best):
        # Starting at day zero fixes the batch and plant draws before the first
        # choice of observation time. Measurement randomness uses a separate
        # deterministic stream and cannot alter this biological path.
        env.advance_to(0)
        visits = 0
        while visits < self.max_observation_times:
            accepted = False
            for _ in range(self.max_retries):
                user = self._cycle_prompt(env, self.max_observation_times - visits)
                msgs = [{"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": user}]
                reply = self.complete(msgs)
                self.transcript.append({"user": user, "assistant": reply})
                try:
                    blob = _extract_json(reply)
                    action = str(blob.get("action", "")).lower()
                    if action == "finish":
                        if "current_best" in blob:
                            best = _to_point(env, blob["current_best"])
                            env.interim_recommendation(best)
                        env.advance()
                        return best
                    if action != "observe":
                        raise ValueError("action must be 'observe' or 'finish'")
                    day = int(blob["day"])
                    if not (env._active.state.day < day < env.task.duration_days):
                        raise ValueError("observation day must be after the current day and before terminal day")
                    modalities = list(blob.get("modalities") or [])
                    if not modalities:
                        raise ValueError("at least one modality is required")
                    units = blob.get("units")
                    if "current_best" in blob:
                        best = _to_point(env, blob["current_best"])
                        env.interim_recommendation(best)
                    env.advance_to(day)
                    for modality in modalities:
                        env.observe(units=units, modality=str(modality))
                    visits += 1
                    accepted = True
                    break
                except Exception:
                    self.format_failures += 1
            if not accepted:
                break
        env.advance()
        return best


# ── A fake model for end-to-end self-test, with no network ──────────
def scripted_completer(env_getter=None):
    """A fake model that only answers in the right format: spread the range evenly,
    one unit per point.

    It exists so the harness can be exercised end to end without an API key, and
    run in CI.
    """
    def complete(messages):
        text = messages[-1]["content"]
        names, lows, highs = [], [], []
        for m in re.finditer(r"- (\w+): ([\d.]+) to ([\d.]+), set per (\w+)", text):
            names.append(m.group(1)); lows.append(float(m.group(2))); highs.append(float(m.group(3)))
        mid_only = {k: lo + 0.5 * (hi - lo) for k, lo, hi in zip(names, lows, highs)}
        if "No further" in text:                      # the final question: a recommendation only
            return json.dumps({"reasoning": "midpoint", "recommendation": mid_only})
        names, lows, highs = [], [], []
        for m in re.finditer(r"- (\w+): ([\d.]+) to ([\d.]+), set per (\w+)", text):
            names.append(m.group(1)); lows.append(float(m.group(2))); highs.append(float(m.group(3)))
        levels = [m.group(4) for m in re.finditer(
            r"- (\w+): ([\d.]+) to ([\d.]+), set per (\w+)", text)]
        units = re.search(r"DRAW FROM \(choose at most \d+\): (.+)", text).group(1).split(", ")
        cap = int(re.search(r"choose at most (\d+)", text).group(1))
        units = units[:cap]
        # Respect the hierarchy: group by chamber, one treatment group per chamber,
        # chamber-level factors constant within a group
        by_ch = {}
        for u in units:
            by_ch.setdefault(u.split("l")[0], []).append(u)
        chs = sorted(by_ch)
        n = max(2, min(4, len(chs)))
        chs = chs[:n]
        trt, alloc = {}, {}
        for i, c in enumerate(chs):
            tid = chr(65 + i)
            frac = i / max(n - 1, 1)
            trt[tid] = {k: lo + frac * (hi - lo) for k, lo, hi in zip(names, lows, highs)}
            alloc[tid] = by_ch[c]
        mid = {k: lo + 0.5 * (hi - lo) for k, lo, hi in zip(names, lows, highs)}
        return json.dumps({"reasoning": "even sweep of the ranges",
                           "treatments": trt, "allocation": alloc,
                           "randomization_seed": 1, "current_best": mid})
    return complete

# ── Recall with no experiment ──────────────────────────────────────────
# This replaces the author's hand-written "literature prior". Those six textbook
# values had no source, and using something recited as the control for recitation
# does not stand up. Here we ask the model itself: with no data at all, what do
# you recommend? The gap between that and the same model's post-campaign score is
# what this model actually gained from experimenting -- one measurement per model,
# not a line we assume on its behalf.

RECALL_ASK = """You will run no experiments at all and see no data.

Recommend the management settings you would give the grower on the strength of what you
already know about this crop. Reply with JSON and nothing else:

{{
  "reasoning": "<two or three sentences>",
  "recommendation": {{{fields}}}
}}"""


# ── The tool ablation ──────────────────────────────────────────────────
# The paper claims the planned / adaptive references mark where correct use of
# standard tools lands. For that claim to have content, the tooled condition has
# to be measured. We do not give the agent call access -- that would introduce a
# whole tool-calling protocol as a confound -- but instead *put the tools' output
# into the prompt*: a ready-made screening design table and a Gaussian-process
# posterior maximum. The agent may use them, adapt them, or ignore them.
#
# BoxingGym ran the corresponding experiment (Box's Apprentice = LLM plus
# statistical modelling) and concluded it "does not reliably improve performance".
# We expected something similar, and "tools were supplied and did not help" is
# itself a reportable result.

TOOLS_HEADER = """You have output from the following standard tools. Use it, adapt it, or ignore it.
{body}
"""

CONSTRAINT_CHECKLIST = """Before submitting a design, explicitly verify internally that
every unit is allocated at most once, the allocation stays within the round cap,
all values are within bounds, and every chamber-level factor is constant inside
each chamber. Return the requested JSON only; do not print the checklist."""


def callable_tool_instructions(mode: str) -> str:
    names = []
    if mode in {"design", "both"}:
        names.append("propose_design")
    if mode in {"inference", "both"}:
        names.append("analyze_history")
    return """OPTIONAL CALLABLE TOOLS
No tool output is shown automatically. You may explicitly call at most once each
by replying with exactly:
{"action":"call_tool","tool":"TOOL_NAME"}
Available tool names: %s.
The tool returns advice only. You remain responsible for the submitted design or
final recommendation. If you do not want a tool, return the requested answer JSON.""" % ", ".join(names)


def _callable_gp_design(env) -> Design:
    """One deterministic, constraint-aware GP-UCB batch proposal without submission."""
    from .agents import GP, _cands, respect_hierarchy, chamber_safe_allocation
    t = env.task
    rng = np.random.default_rng(int(env.seed) * 1009 + int(env.round) * 917 + 23)
    n_pts = t.units_per_round
    X, y = env.as_arrays()
    if len(y) < 2:
        P = rng.random((n_pts, t.d))
    else:
        g = GP(noise=max(0.15, t.plant_cv)).fit(X, y)
        C = _cands(t.d, 2500, rng)
        mu, var = g.predict(C)
        order = np.argsort(-(mu + 1.8 * np.sqrt(var)))
        chosen = []
        for i in order:
            if all(np.linalg.norm(C[i] - p) > 0.05 for p in chosen):
                chosen.append(C[i])
            if len(chosen) == n_pts:
                break
        P = np.asarray(chosen) if chosen else rng.random((n_pts, t.d))
    P = respect_hierarchy(P, t, env.facility, rng)
    tids = [f"gp{i}" for i in range(len(P))]
    treatments = {tid: {f.name: float(P[i][j]) for j, f in enumerate(t.factors)}
                  for i, tid in enumerate(tids)}
    allocation = chamber_safe_allocation(
        tids, 1, env.facility, rng, treatments, t, avail=env.available_units())
    return Design(treatments=treatments, allocation=allocation,
                  randomization_seed=int(rng.integers(1_000_000)),
                  question="callable constraint-aware GP-UCB proposal")


def callable_tool_result(env, name: str) -> dict:
    t = env.task
    if name == "propose_design":
        design = _callable_gp_design(env)
        treatments = {
            tid: {f.name: round(float(f.denorm(values[f.name])), 8) for f in t.factors}
            for tid, values in design.treatments.items()
        }
        return {"tool": name, "method": "constraint-aware GP-UCB batch",
                "treatments": treatments, "allocation": design.allocation,
                "randomization_seed": design.randomization_seed,
                "note": "advice only; no design has been submitted"}
    if name == "analyze_history":
        from .agents import GP, _cands, _gp_posterior_max
        X, y = env.as_arrays()
        if not len(y):
            return {"tool": name, "status": "unavailable_before_observations",
                    "note": "no recommendation has been submitted"}
        x = _gp_posterior_max(env)
        g = GP(noise=max(0.15, t.plant_cv)).fit(X, y)
        mu, var = g.predict(np.asarray([x]))
        point = {f.name: round(float(f.denorm(x[j])), 8) for j, f in enumerate(t.factors)}
        return {"tool": name, "method": "GP posterior-mean reader",
                "posterior_best": point, "predicted_mean": float(mu[0]),
                "posterior_sd": float(np.sqrt(var[0])),
                "n_observations": int(len(y)),
                "note": "advice only; no recommendation has been submitted"}
    raise ValueError(f"unknown callable tool {name!r}")


def tool_artifacts(env) -> dict:
    """Return normalized tool outputs so adoption can be measured geometrically."""
    from .agents import _pb12, respect_hierarchy, chamber_safe_allocation, _gp_posterior_max
    rng = np.random.default_rng(0)
    t, names = env.task, [f.name for f in env.task.factors]
    design = None
    try:
        P = _pb12()[:, :t.d] * 0.25 + 0.5
        P = respect_hierarchy(P, t, env.facility, rng)
        tids = [f"t{i}" for i in range(len(P))]
        treatments = {tid: {name: float(P[i][j]) for j, name in enumerate(names)}
                      for i, tid in enumerate(tids)}
        allocation = chamber_safe_allocation(
            tids, 1, env.facility, rng, treatments, t, avail=env.available_units())
        design = Design(treatments, allocation, randomization_seed=0,
                        question="supplied screening tool")
    except Exception:
        design = None
    X, y = env.as_arrays()
    inference = _gp_posterior_max(env) if len(y) else None
    return {"design": design, "inference": inference}


def design_tool_text(env, artifact=None) -> str:
    artifact = artifact if artifact is not None else tool_artifacts(env)["design"]
    if artifact is None:
        return "TOOL 1 - screening design: unavailable for this facility."
    t, names = env.task, [f.name for f in env.task.factors]
    lines = []
    for tid, units in artifact.allocation.items():
        values = artifact.treatments[tid]
        setting = ", ".join(
            f"{name}={t.factors[j].denorm(values[name]):.1f}"
            for j, name in enumerate(names))
        lines.append(f"  {tid}: {setting}  -> units {list(units)}")
    return ("TOOL 1 - screening design (Plackett-Burman), ready to run:\n" +
            ("\n".join(lines) if lines else "  unavailable"))


def inference_tool_text(env, artifact=None) -> str:
    artifact = artifact if artifact is not None else tool_artifacts(env)["inference"]
    if artifact is None:
        return "TOOL 2 - Gaussian-process posterior maximum: unavailable before observations."
    t = env.task
    point = ", ".join(
        f"{factor.name}={factor.denorm(artifact[j]):.1f}"
        for j, factor in enumerate(t.factors))
    return "TOOL 2 - Gaussian-process posterior maximum from visible observations:\n  " + point


def tool_text(env, mode="both") -> str:
    """Render design-only, inference-only or combined passive tool output."""
    if mode not in {"design", "inference", "both"}:
        raise ValueError(f"tool_text needs design, inference or both; got {mode!r}")
    artifacts = tool_artifacts(env)
    parts = []
    if mode in {"design", "both"}:
        parts.append(design_tool_text(env, artifacts["design"]))
    if mode in {"inference", "both"}:
        parts.append(inference_tool_text(env, artifacts["inference"]))
    return TOOLS_HEADER.format(body="\n\n".join(parts))


def zero_shot_recommendation(env, complete, max_retries: int = 3):
    """Ask the model what it recommends with no experiment at all. Consumes no budget and submits no design."""
    names = [g.name for g in env.task.factors]
    user = "\n\n".join([
        _facility_text(env),
        RECALL_ASK.format(fields=", ".join(f'"{n}": <number>' for n in names)),
    ])
    for _ in range(max_retries):
        try:
            blob = _extract_json(complete([{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": user}]))
            return _to_point(env, blob["recommendation"])
        except Exception:
            continue
    return np.full(env.task.d, 0.5)
