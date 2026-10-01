#!/usr/bin/env python3
"""Calibrate the contract-v7 context budget with two scripted reading habits (toy backend, real calendar).

Both follow the scripted two-wave plan (slowlab/scripted_model.py) and add reads
at checkpoints for every running crop:

* frugal: every 30 days, the current harvest and canopy readings, then the
  predict_crop_outcome tool;
* heavy: every 7 days, the longest raw window (512 records) of five channels.

The budget should leave the frugal habit far from its limit and stop the heavy
habit early. Reports characters used, reads, records and whether the
character budget forced the endgame.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_client import CampaignProcess  # noqa: E402
from slowlab.llm_agent import LLMCampaignAgent  # noqa: E402
from slowlab.outbound_audit import AuditedCompleter  # noqa: E402
from slowlab.prompt_firewall import load_blinding_policy  # noqa: E402
from slowlab.scripted_model import ScriptedModel  # noqa: E402
from slowlab.tools import Toolbox  # noqa: E402

HABITS = {
    'frugal': {'every_days': 30, 'channels': ('cumulative_harvest_fresh_equivalent', 'canopy_lai_proxy'),
               'window_days': 0.0, 'predict': True},
    'heavy': {'every_days': 7, 'channels': ('cumulative_harvest_fresh_equivalent', 'canopy_lai_proxy', 'heating_energy',
                                            'air_temperature_c', 'co2_ppm'), 'window_days': 511 * 300 / 86400,
              'predict': False},
}


class CheckpointReader:
    def __init__(self, task, habit):
        self.base, self.habit, self.queue, self.done_checkpoints = ScriptedModel(task), habit, [], set()
        self.tick = task['calendar']['time_step_seconds'] / 86400

    def __call__(self, messages):
        final = json.loads(messages[-1]['content'])
        state = final['state']
        if self.queue and 'Reply only' not in final['instruction']:
            return json.dumps(self.queue.pop(0))
        reply = json.loads(self.base(messages))
        action = reply.get('action', {})
        if reply.get('type') == 'campaign' and action.get('action') == 'advance':
            day = state['day']
            running = [c for c in state['crops'] if c['status'] == 'running']
            every = self.habit['every_days']
            at_checkpoint = day > 0 and abs(day / every - round(day / every)) < 1e-6
            if running and at_checkpoint and round(day / every) not in self.done_checkpoints:
                self.done_checkpoints.add(round(day / every))
                for c in running:
                    if day - c['start_day'] < 10:
                        continue
                    start = max(c['start_day'], round((day - self.habit['window_days']) / self.tick) * self.tick)
                    for ch in self.habit['channels']:
                        self.queue.append({'type': 'campaign', 'action': {'action': 'observe', 'unit': c['unit'],
                                                                          'variable': ch, 'start_day': start,
                                                                          'end_day': day}})
                    if self.habit['predict'] and c['start_day'] == 0:
                        self.queue.append({'type': 'tool', 'name': 'predict_crop_outcome',
                                           'args': {'unit': c['unit'], 'run_index': c['crop']}})
                if self.queue:
                    return json.dumps(self.queue.pop(0))
            nxt = (int(day // every) + 1) * every
            if running and nxt < action['day'] - 1e-9:
                action['day'] = round(nxt / self.tick) * self.tick
                return json.dumps(reply)
        return json.dumps(reply)


def run(habit_name, contract_path, folder: Path) -> dict:
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    spec = {'backend': 'fake', 'contract': str(contract_path), 'feedback_mode': 'full',
            'fallback_policy': policies['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None,
            'trace': None, 'settlement_out': str(folder / f'{habit_name}_s.json'),
            'failure_out': str(folder / f'{habit_name}_f.json')}
    (folder / f'{habit_name}.json').write_text(json.dumps(spec))
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    with CampaignProcess(folder / f'{habit_name}.json', private_log=folder / f'{habit_name}.log') as c:
        model = CheckpointReader(c.session.task, HABITS[habit_name])
        complete = AuditedCompleter(model, blinding, folder / f'{habit_name}_audit.jsonl')
        summary = LLMCampaignAgent(c.session, Toolbox(c.session, 1), complete).run()
        c.close()
    counts = summary['counts']
    return {'habit': HABITS[habit_name] | {'channels': list(HABITS[habit_name]['channels'])},
            **{k: counts[k] for k in ('llm_calls', 'input_chars', 'output_chars', 'observe_calls', 'records_received',
                                      'tool_calls', 'truncated_results', 'context_exhausted', 'forced_advance',
                                      'recommended')},
            'budget': summary['context_budget'],
            'share_of_budget_used': counts['input_chars'] / summary['context_budget']['max_input_chars_per_campaign'],
            'budget_exhausted_on_day': next((e['from_day'] for e in summary['log'] if e.get('cause') == 'context_budget'), None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budgets', default='2000000', help='comma-separated max_input_chars_per_campaign values')
    args = parser.parse_args()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for budget in (int(b) for b in args.budgets.split(',')):
            contract = json.loads((ROOT / 'configs/task_contract_v7.json').read_text())
            contract['budget']['context']['max_input_chars_per_campaign'] = budget
            path = Path(tmp) / f'contract_{budget}.json'
            path.write_text(json.dumps(contract))
            for name in HABITS:
                folder = Path(tmp) / f'{name}_{budget}'
                folder.mkdir()
                results[f'{name}@{budget}'] = run(name, path, folder)
    out = {'purpose': 'calibration of the contract-v7 context budget with scripted reading habits',
           'backend': 'toy (record formats as in the real executor; no sensor dropouts)', 'results': results}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    for name, r in results.items():
        print(name, {k: r[k] for k in ('llm_calls', 'input_chars', 'observe_calls', 'records_received', 'tool_calls',
                                         'context_exhausted', 'budget_exhausted_on_day', 'share_of_budget_used')})


if __name__ == '__main__':
    main()
