"""Model stand-ins for the full-path test (imported by scripts/run_campaign_job.py via the python: hook)."""
import json

T = 300 / 86400


class Exerciser:
    """Uses every action, every analysis tool it can, notes, and makes one of each recoverable mistake."""

    def __init__(self, task):
        self.task, self.step, self.mistakes = task, 0, ['not json', 'bad_action', 'bad_tool']

    def __call__(self, messages):
        last = messages[-1]['content']
        try:
            final = json.loads(last)
        except json.JSONDecodeError:
            return json.dumps({'type': 'initial_recommendation', 'policy': _policy(self.task, 0.5), 'reason': 'prior'})
        state, instruction = final['state'], final['instruction']
        latest = final.get('latest')
        if isinstance(latest, dict) and isinstance(latest.get('harness'), dict) \
                and latest['harness'].get('kind') == 'feedback_condition_assigned':
            self.step, self.mistakes = 0, ['not json', 'bad_action', 'bad_tool']  # a new branch starts
        if instruction.startswith('Day-0 design'):
            n = len(state['crops'])
            if n < 2:
                return json.dumps({'type': 'campaign', 'action': {'action': 'start', 'unit': n,
                                                                  'policy': _policy(self.task, 0.3 + 0.2 * n)},
                                   'notes': f'started {n + 1}'})
            return json.dumps({'type': 'design_complete'})
        if 'Reply only' in instruction or state['days_left'] <= 1e-9:
            return json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': _policy(self.task, 0.6)}})
        if self.mistakes:
            m = self.mistakes.pop(0)
            if m == 'not json':
                return 'I will think about it.'
            if m == 'bad_action':
                return json.dumps({'type': 'campaign', 'action': {'action': 'start', 'unit': 9, 'policy': _policy(self.task, 0.5)}})
            return json.dumps({'type': 'tool', 'name': 'no_such_tool', 'args': {}})
        self.step += 1
        plan = [
            {'type': 'tool', 'name': 'runs', 'args': {}},
            {'type': 'campaign', 'action': {'action': 'advance', 'day': 4 * T}},
            {'type': 'campaign', 'action': {'action': 'observe', 'unit': 0, 'variable': 'heating_energy'}},
            {'type': 'campaign', 'action': {'action': 'observe', 'unit': 0, 'variable': 'co2_ppm', 'resolution': 'daily'}},
            {'type': 'tool', 'name': 'series_summary', 'args': {'unit': 0, 'variable': 'heating_energy'}},
            {'type': 'campaign', 'action': {'action': 'stop', 'unit': 1}},
            {'type': 'campaign', 'action': {'action': 'advance', 'day': 6 * T}},
            {'type': 'campaign', 'action': {'action': 'start', 'unit': 1, 'policy': _policy(self.task, 0.8)}},
            {'type': 'campaign', 'action': {'action': 'advance', 'day': 13 * T}},
            {'type': 'campaign', 'action': {'action': 'observe', 'unit': 0, 'run_index': 1}},
            {'type': 'campaign', 'action': {'action': 'observe', 'unit': 1, 'run_index': 1}},
            {'type': 'tool', 'name': 'space_filling_candidates', 'args': {'n': 3}},
            {'type': 'campaign', 'action': {'action': 'advance', 'day': 26 * T}},
        ]
        return json.dumps(plan[min(self.step - 1, len(plan) - 1)])


def _policy(task, u):
    out = {}
    for k, s in task['policy']['fields'].items():
        out[k] = round(s['min'] + u * (s['max'] - s['min']), 3)
    out['night_temperature_c'] = min(out['night_temperature_c'], out['day_temperature_c'])
    return out


class Broken:
    """A provider that fails after the initial recommendation (infrastructure failure path)."""

    def __init__(self, task):
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        if self.calls > 1:
            from slowlab.providers import ProviderError
            raise ProviderError('simulated provider outage')
        return 'no json'


class Billed(Exerciser):
    """Exerciser that leaves provider-style call records (tokens only), for the spend-guard test."""

    def __init__(self, task):
        super().__init__(task)
        self.call_records = []

    def __call__(self, messages):
        text = super().__call__(messages)
        chars = sum(len(m['content']) for m in messages)
        self.call_records.append({'usage': {'prompt_tokens': chars // 4, 'completion_tokens': len(text) // 4}})
        return text
