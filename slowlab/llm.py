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
    obs = env.observations()
    if not obs:
        return "OBSERVATIONS SO FAR: none (this is round 1)."
    rows = ["OBSERVATIONS SO FAR. All rates are per m2 per day, in EUR.",
            "gross margin = revenue - energy cost - other cost.",
            "  energy cost is electricity + gas; other cost is CO2, transplants, labour.",
            ""]
    for o in obs:
        d = env._designs[o.design_id]
        fv = d.treatments[o.treatment]
        setting = ", ".join(
            f"{g.name}={g.denorm(fv[g.name]):.4g}" for g in env.task.factors)
        parts = ""
        if o.rev_rate == o.rev_rate:            # not NaN
            parts = (f"  [revenue {o.rev_rate:+.4f}"
                     f"  energy {o.energy_cost_rate:+.4f}"
                     f"  other {o.other_cost_rate:+.4f}]")
        rows.append(f"  round {o.design_id + 1}  {o.unit_id}  {setting}"
                    f"  -> margin {o.value:+.4f}{parts}")
    return "\n".join(rows)


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
    tools: bool = False          # the tool ablation: put standard tools' output into the prompt

    def _ask(self, env, feedback: str | None) -> dict:
        user = "\n\n".join([
            _facility_text(env),
            _observations_text(env),
        ] + ([tool_text(env)] if self.tools else []) + [
            f"ROUNDS REMAINING: {env.rounds_left}",
            f"UNITS YOU MAY DRAW FROM (choose at most {env.task.units_per_round}): "
            + ", ".join(env.available_units()),
            _schema_text(env),
        ] + ([f"YOUR PREVIOUS REPLY WAS REJECTED: {feedback}\nFix it and reply again."]
             if feedback else []))
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        reply = self.complete(msgs)
        self.transcript.append({"user": user, "assistant": reply})
        return _extract_json(reply)

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
                    env.advance()
                    env.interim_recommendation(best)
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
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": _final_prompt(env)}]
                reply = self.complete(msgs)
                self.transcript.append({"user": msgs[-1]["content"],
                                        "assistant": reply})
                blob = _extract_json(reply)
                best = _to_point(env, blob["recommendation"])
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
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": _transfer_prompt(env)}]
                    reply = self.complete(msgs)
                    self.transcript.append({"user": msgs[-1]["content"],
                                            "assistant": reply})
                    blob = _extract_json(reply)
                    self.x_transfer = _to_point(env, blob["recommendation"])
                    break
                except Exception:
                    self.format_failures += 1
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

TOOLS_HEADER = """You also have the output of two standard tools, already run on this
problem. Use them, adapt them, or ignore them.

TOOL 1 - screening design (Plackett-Burman with replicated centre points), a ready-to-run
allocation for this facility:
{design}

TOOL 2 - Gaussian-process fit to every observation so far; its posterior maximum is at:
{gp}
"""


def tool_text(env) -> str:
    """The two tools' output, assembled into a block that can go into the prompt."""
    import numpy as np
    from .agents import _pb12, respect_hierarchy, chamber_safe_allocation, _gp_posterior_max
    rng = np.random.default_rng(0)
    t, names = env.task, [f.name for f in env.task.factors]
    try:
        P = _pb12()[:, :t.d] * 0.25 + 0.5      # +/-1 -> 0.25 / 0.75 within the normalised range
        P = respect_hierarchy(P, t, env.facility, rng)
        tids = [f"t{i}" for i in range(len(P))]
        trt = {tid: {n: float(P[i][j]) for j, n in enumerate(names)}
               for i, tid in enumerate(tids)}
        alloc = chamber_safe_allocation(tids, 1, env.facility, rng, trt, t,
                                        avail=env.available_units())
        lines = []
        for tid, uids in alloc.items():
            vals = ", ".join(f"{n}={t.factors[j].low + trt[tid][n] * (t.factors[j].high - t.factors[j].low):.1f}"
                             for j, n in enumerate(names))
            lines.append(f"  {tid}: {vals}  -> units {list(uids)}")
        design = "\n".join(lines) if lines else "  (facility too small for this design)"
    except Exception as exc:
        design = f"  (unavailable: {exc})"
    X, y = env.as_arrays()
    if len(y):
        x = _gp_posterior_max(env)
        gp = "  " + ", ".join(
            f"{n}={t.factors[j].low + x[j] * (t.factors[j].high - t.factors[j].low):.1f}"
            for j, n in enumerate(names))
    else:
        gp = "  (no observations yet)"
    return TOOLS_HEADER.format(design=design, gp=gp)


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
