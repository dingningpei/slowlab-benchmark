"""Language-model campaign agent (llm-campaign-v1).

The model acts through the same ``PublicSession`` and public ``Toolbox`` as the
BO baseline. Each turn it replies with one JSON object: a campaign action or an
analysis-tool call. Prompts are rebuilt deterministically every turn from
public data only: a system message (public task view, tool catalog, reply
format), the most recent exchanges that fit the prompt budget (older bulky
results compacted), and a final user message with a state digest (day,
remaining budgets, compartment status, crop table) and the latest result.
Identical public histories and identical model replies therefore give
byte-identical prompts.

The harness keeps the campaign settleable: when the call budget runs low it
advances to the final day itself and asks only for the recommendation. Model
formatting errors, invalid actions and tool errors are fed back and counted.
Feedback never repeats model-supplied text (the model's reply is already the
preceding assistant turn), so harness messages contain only harness content;
infrastructure failures, provider failures and firewall blocks propagate.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from .agent_client import InvalidAction
from .agent_protocol import canonical
from .tools import PublicHistory, ToolError, Toolbox, contribution_margin

LLM_AGENT_VERSION = 'llm-campaign-v1'


@dataclass(frozen=True)
class LLMConfig:
    max_llm_calls: int = 200
    prompt_char_budget: int = 80_000
    reserve_tool_calls: int = 2
    max_reason_chars: int = 400
    initial_recommendation_attempts: int = 3
    records_kept_in_history: int = 24


class FormatError(ValueError):
    pass


REPLY_FORMAT = (
    'Reply with exactly one JSON object and nothing else. Either\n'
    '{"type": "campaign", "action": {...}, "reason": "..."} to send one campaign action '
    '(start, observe, advance, stop, recommend; see actions), or\n'
    '{"type": "tool", "name": "...", "args": {...}, "reason": "..."} to run one analysis tool.\n'
    'The reason is optional, at most a few sentences. Analysis tools only analyse data you already '
    'received and do not advance time or spend campaign calls; they have their own call limit.'
)


def system_message(task: dict) -> str:
    return ('You run a greenhouse experimentation campaign as the experimenter. Plan experiments, read '
            'the evidence your feedback condition allows, and recommend one management policy at the '
            'final day.\n\nTASK\n' + canonical(task) + '\n\nANALYSIS TOOLS\n' + canonical(Toolbox.catalog())
            + '\n\nREPLY FORMAT\n' + REPLY_FORMAT)


def parse_reply(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.strip('`')
        stripped = stripped[stripped.find('\n') + 1:] if '\n' in stripped else stripped
    start, end = stripped.find('{'), stripped.rfind('}')
    if start < 0 or end <= start:
        raise FormatError('no JSON object found')
    try:
        reply = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise FormatError(f'invalid JSON: {exc.msg}') from None
    if not isinstance(reply, dict) or reply.get('type') not in ('campaign', 'tool', 'initial_recommendation',
                                                                  'design_complete'):
        raise FormatError('type must be "campaign" or "tool"')
    if reply['type'] == 'campaign' and not isinstance(reply.get('action'), dict):
        raise FormatError('a campaign reply needs an "action" object')
    if reply['type'] == 'tool' and (not isinstance(reply.get('name'), str)
                                    or not isinstance(reply.get('args', {}), dict)):
        raise FormatError('a tool reply needs a "name" string and an "args" object')
    if reply['type'] == 'initial_recommendation' and not isinstance(reply.get('policy'), dict):
        raise FormatError('an initial recommendation needs a "policy" object')
    if 'reason' in reply and not isinstance(reply['reason'], str):
        raise FormatError('reason must be a string')
    return reply


def validate_policy_against_view(task: dict, policy) -> None:
    fields = task['policy']['fields']
    if not isinstance(policy, dict) or set(policy) != set(fields):
        raise ValueError('policy must contain exactly the policy fields')
    for name, spec in fields.items():
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not spec['min'] <= value <= spec['max']:
            raise ValueError(f'policy field {name} outside its range')
    if policy['night_temperature_c'] > policy['day_temperature_c']:
        raise ValueError('night_temperature_c must not exceed day_temperature_c')


def initial_recommendation(task: dict, complete, attempts: int = 3) -> tuple[dict | None, list]:
    """Ask once, from the public task view only, for the policy the model would recommend without
    running any experiment. Returns (policy or None, public log)."""
    system = ('You are a greenhouse grower. Before any experiment, recommend the management policy you '
              'expect to score best under the scoring rule.\n\nTASK\n' + canonical(task) +
              '\n\nReply with exactly one JSON object: {"type": "initial_recommendation", "policy": {...}, '
              '"reason": "..."}')
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': 'Give your initial recommendation now.'}]
    log = []
    for _ in range(attempts):
        text = complete(messages)
        try:
            reply = parse_reply(text)
            if reply['type'] != 'initial_recommendation':
                raise FormatError('expected type "initial_recommendation"')
            validate_policy_against_view(task, reply['policy'])
        except ValueError as exc:
            log.append({'ok': False, 'error': str(exc)})
            messages = messages[:2] + [{'role': 'assistant', 'content': text[:2000]},
                                       {'role': 'user', 'content': f'That reply was not usable: {exc}. '
                                                                   'Reply again with one JSON object.'}]
            continue
        log.append({'ok': True, 'policy': reply['policy']})
        return dict(reply['policy']), log
    return None, log


DAY_ZERO_INSTRUCTION = ('Day-0 design: start crops with start actions and use analysis tools if useful. Reply '
                        '{"type": "design_complete"} when your day-0 design is finished. No other campaign action '
                        'is possible before the design is complete.')


class LLMCampaignAgent:
    """``task`` overrides the session's view (the condition-free view for the shared day-0
    design); ``history``, ``counts`` and ``notice`` continue a conversation after a branch."""

    def __init__(self, session, toolbox: Toolbox, complete, config: LLMConfig = LLMConfig(), *,
                 task: dict | None = None, history: list | None = None, counts: dict | None = None,
                 notice: dict | None = None):
        self.session = session
        self.toolbox = toolbox
        self.complete = complete
        self.config = config
        self.task = task if task is not None else session.task
        self.history: list[dict] = [dict(entry) for entry in history or []]   # {'reply': str, 'feedback': dict}
        self.counts = {'llm_calls': 0, 'format_errors': 0, 'invalid_actions': 0, 'tool_calls': 0,
                       'tool_errors': 0, 'forced_advance': False, 'recommended': False}
        if counts:
            self.counts.update({k: v for k, v in counts.items() if k not in ('forced_advance', 'recommended')})
        self.log: list[dict] = []
        self._harness_note = notice

    # ── public state ───────────────────────────────────────────────────────
    def _latest(self):
        clock, budget = 0.0, None
        for entry in self.session.transcript:
            if entry['ok']:
                clock = entry['result'].get('clock', clock)
                budget = entry['result'].get('budget', budget)
        return clock / 86400, budget

    def _remaining(self):
        _, budget = self._latest()
        b = self.task['budget']
        if budget is None:
            return b['max_tool_calls'], b['max_decision_calls']
        return (budget['tool_calls_max'] - budget['tool_calls_used'],
                budget['decision_calls_max'] - budget['decision_calls_used'])

    def state_digest(self) -> dict:
        day, _ = self._latest()
        tools_left, decisions_left = self._remaining()
        history = PublicHistory(self.session.transcript)
        crops = []
        for (unit, run_index) in sorted(history.runs):
            run = history.runs[(unit, run_index)]
            row = {'unit': unit, 'crop': run_index, 'start_day': run['start_day'], 'status': run['status'],
                   'policy': run['policy']}
            if run.get('accrued') is not None:
                row['reason'] = run['reason']
                row['contribution_margin_eur_m2'] = round(contribution_margin(
                    run['accrued'], run['event_cost_eur_m2'], self.task['economics']), 3)
            crops.append(row)
        return {'day': day, 'days_left': self.task['calendar']['campaign_days'] - day,
                'campaign_calls_left': tools_left, 'decision_calls_left': decisions_left,
                'analysis_tool_calls_left': self.toolbox._max_calls - self.toolbox.calls,
                'model_calls_left': self.config.max_llm_calls - self.counts['llm_calls'],
                'compartments': [history.last_status[u] for u in sorted(history.last_status)],
                'crops': crops}

    # ── prompt ─────────────────────────────────────────────────────────────
    def _compact(self, feedback: dict) -> dict:
        result = feedback.get('result')
        if isinstance(result, dict) and isinstance(result.get('records'), list) \
                and len(result['records']) > self.config.records_kept_in_history:
            records = result['records']
            k = self.config.records_kept_in_history
            step = (len(records) - 1) / (k - 1)
            kept = [records[round(i * step)] for i in range(k)]
            result = {**result, 'records': kept, 'records_note':
                      f'{len(records)} records returned; {k} evenly spaced shown here. Use series_summary.'}
            return {**feedback, 'result': result}
        return feedback

    def messages(self, instruction: str) -> list:
        system = {'role': 'system', 'content': system_message(self.task)}
        final_feedback = self.history[-1]['feedback'] if self.history else {
            'kind': 'start', 'message': 'The campaign is at day 0. No compartment is in use.'}
        if self._harness_note is not None:
            final_feedback = {'harness': self._harness_note, 'previous': final_feedback}
        final = {'role': 'user', 'content': canonical({'state': self.state_digest(), 'latest': final_feedback,
                                                       'instruction': instruction})}
        if len(final['content']) > self.config.prompt_char_budget // 2:
            final['content'] = canonical({'state': self.state_digest(), 'latest': self._compact(final_feedback),
                                          'instruction': instruction})
        budget = self.config.prompt_char_budget - len(system['content']) - len(final['content'])
        window = []
        if self.history:
            last_reply = {'role': 'assistant', 'content': self.history[-1]['reply']}
            budget -= len(last_reply['content'])
            for entry in reversed(self.history[:-1]):
                pair = [{'role': 'assistant', 'content': entry['reply']},
                        {'role': 'user', 'content': canonical(self._compact(entry['feedback']))}]
                size = sum(len(m['content']) for m in pair)
                if size > budget:
                    break
                window = pair + window
                budget -= size
            window.append(last_reply)
        if not window:
            return [system, final]
        opening = {'role': 'user', 'content': 'Begin the campaign.'}
        return [system, opening, *window, final]

    # ── loop ───────────────────────────────────────────────────────────────
    def _ask(self, instruction):
        self.counts['llm_calls'] += 1
        messages = self.messages(instruction)
        self._harness_note = None
        return self.complete(messages)

    def _record(self, reply_text, feedback):
        self.history.append({'reply': reply_text, 'feedback': feedback})

    def _reply_text(self, reply: dict) -> str:
        kept = {k: v for k, v in reply.items() if k in ('type', 'action', 'name', 'args')}
        if isinstance(reply.get('reason'), str):
            kept['reason'] = reply['reason'][:self.config.max_reason_chars]
        return canonical(kept)

    def _endgame(self) -> bool:
        day, _ = self._latest()
        end = self.task['calendar']['campaign_days']
        tools_left, decisions_left = self._remaining()
        low = tools_left <= self.config.reserve_tool_calls or decisions_left <= 1
        if day < end and low and tools_left >= 1:
            result = self.session.dispatch({'action': 'advance', 'day': end})
            self.counts['forced_advance'] = True
            self.log.append({'day': end, 'event': 'forced_advance_to_final_day'})
            self._harness_note = {'kind': 'harness_forced_advance',
                                  'message': 'The call budget is nearly spent, so the harness advanced the '
                                             'campaign to the final day.', 'result': result}
        return self._latest()[0] >= end and (low or self.counts['forced_advance'])

    def run_day_zero_design(self, max_calls: int = 24) -> dict:
        """Shared prefix before branching: only start actions and analysis tools, until the model
        declares the design complete, every compartment is in use, or max_calls is reached."""
        compartments = len(self.task['compartments'])
        started, ended_by = 0, 'max_calls'
        for _ in range(max_calls):
            if started >= compartments:
                ended_by = 'all_compartments_started'
                break
            text = self._ask(DAY_ZERO_INSTRUCTION)
            try:
                reply = parse_reply(text)
                if reply['type'] == 'design_complete':
                    self._record(self._reply_text(reply), {'kind': 'design_complete_acknowledged'})
                    ended_by = 'design_complete'
                    break
                if reply['type'] == 'initial_recommendation' or (
                        reply['type'] == 'campaign' and reply['action'].get('action') != 'start'):
                    raise FormatError('only start actions, analysis tools or design_complete during the day-0 design')
            except FormatError as exc:
                self.counts['format_errors'] += 1
                self._record(text[:2000], {'kind': 'format_error', 'message': str(exc)})
                continue
            reply_text = self._reply_text(reply)
            if reply['type'] == 'tool':
                self.counts['tool_calls'] += 1
                try:
                    out = self.toolbox.call(reply['name'], reply.get('args', {}))
                    self._record(reply_text, {'kind': 'tool_result', 'result': out['result']})
                except ToolError as exc:
                    self.counts['tool_errors'] += 1
                    self._record(reply_text, {'kind': 'tool_error', 'message': str(exc)})
                continue
            try:
                result = self.session.dispatch(reply['action'])
            except InvalidAction as exc:
                self.counts['invalid_actions'] += 1
                self._record(reply_text, {'kind': 'invalid_action', 'message': str(exc)})
                continue
            started += 1
            self._record(reply_text, {'kind': 'campaign_result', 'result': result})
            self.log.append({'day': 0.0, 'action': reply['action'], 'phase': 'shared_day_0'})
        return {'ended_by': ended_by, 'starts': started, 'counts': dict(self.counts),
                'history': [dict(e) for e in self.history], 'log': list(self.log)}

    def run(self) -> dict:
        normal = 'Choose your next step.'
        only_recommend = ('The campaign is at its final day and the call budget is nearly spent. Reply only '
                          'with {"type": "campaign", "action": {"action": "recommend", "policy": {...}}}.')
        while self.counts['llm_calls'] < self.config.max_llm_calls and not self.counts['recommended']:
            restricted = self._endgame()
            text = self._ask(only_recommend if restricted else normal)
            try:
                reply = parse_reply(text)
                if reply['type'] in ('initial_recommendation', 'design_complete'):
                    raise FormatError('type must be "campaign" or "tool"')
                if restricted and not (reply['type'] == 'campaign' and reply['action'].get('action') == 'recommend'):
                    raise FormatError('only a recommend action is possible now')
            except FormatError as exc:
                self.counts['format_errors'] += 1
                self._record(text[:2000], {'kind': 'format_error', 'message': str(exc)})
                continue
            reply_text = self._reply_text(reply)
            if reply['type'] == 'tool':
                self.counts['tool_calls'] += 1
                try:
                    out = self.toolbox.call(reply['name'], reply.get('args', {}))
                    self._record(reply_text, {'kind': 'tool_result', 'result': out['result']})
                except ToolError as exc:
                    self.counts['tool_errors'] += 1
                    self._record(reply_text, {'kind': 'tool_error', 'message': str(exc)})
                continue
            action = reply['action']
            try:
                result = self.session.dispatch(action)
            except InvalidAction as exc:
                self.counts['invalid_actions'] += 1
                self._record(reply_text, {'kind': 'invalid_action', 'message': str(exc)})
                continue
            self._record(reply_text, {'kind': 'campaign_result', 'result': result})
            self.log.append({'day': self._latest()[0], 'action': action})
            if action.get('action') == 'recommend':
                self.counts['recommended'] = True
                self.log[-1]['fallback'] = result.get('fallback')
        if not self.counts['recommended']:
            day, _ = self._latest()
            tools_left, _ = self._remaining()
            if day < self.task['calendar']['campaign_days'] and tools_left >= 1:
                self.session.dispatch({'action': 'advance', 'day': self.task['calendar']['campaign_days']})
                self.counts['forced_advance'] = True
            self.log.append({'event': 'no_recommendation_submitted'})
        return {'version': LLM_AGENT_VERSION, 'config': asdict(self.config), 'counts': dict(self.counts),
                'log': self.log}
