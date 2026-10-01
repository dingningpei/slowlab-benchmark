"""Full and Endpoint branches that share one day-0 start (decision 2026-09-30, option A).

1. The initial recommendation and the day-0 design are made from the
   condition-free public task view, so they cannot depend on the feedback
   condition.
2. The design is executed in the Full process; the Endpoint process then
   replays the same requests, and every response must be byte-identical
   (same response SHA-256), or the run stops as an infrastructure failure.
3. Each branch continues the same conversation, is told its condition by a
   harness notice, and from then on sees the view for that condition. Shared
   model calls and analysis-tool calls count against both branches' budgets.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .agent_client import CampaignProcess, InfrastructureFailure, InvalidAction
from .agent_protocol import public_task_view
from .llm_agent import LLMCampaignAgent, LLMConfig, initial_recommendation
from .tools import Toolbox


class BranchDivergence(InfrastructureFailure):
    """The Endpoint replay of the shared day-0 prefix did not reproduce the Full responses."""


def replay_prefix(session, prefix_transcript: list) -> None:
    for position, entry in enumerate(prefix_transcript):
        try:
            session.dispatch(entry['request'])
        except InvalidAction:
            pass
        replayed = session.transcript[-1]
        if replayed['response_sha256'] != entry['response_sha256']:
            raise BranchDivergence(f'replayed day-0 request {position} gave a different response')


def assignment_notice(task: dict) -> dict:
    return {'kind': 'feedback_condition_assigned', 'mode': task['feedback']['mode'],
            'description': task['feedback']['description'],
            'message': 'Your day-0 design is complete. From now on this feedback condition applies.'}


def run_branched_campaign(contract: dict, base_spec: dict, private_dir: Path, make_completer, *,
                          tool_seed: int, config: LLMConfig = LLMConfig(), default_policy: dict,
                          day_zero_max_calls: int = 24) -> dict:
    """``base_spec`` is a site spec without feedback_mode, fallback_policy or private paths.
    ``make_completer(label)`` returns an audited completer for one phase."""
    private_dir = Path(private_dir)
    shared_view = public_task_view(contract, None)
    initial, initial_log = initial_recommendation(shared_view, make_completer('initial_recommendation'),
                                                  config.initial_recommendation_attempts)
    frozen_default = initial is None
    fallback = initial if initial is not None else default_policy

    def spec_for(mode):
        folder = private_dir / mode
        folder.mkdir(parents=True, exist_ok=True)
        spec = {**copy.deepcopy(base_spec), 'feedback_mode': mode, 'fallback_policy': fallback,
                'trace': str(folder / 'trace.jsonl.gz'), 'settlement_out': str(folder / 'settlement.json'),
                'failure_out': str(folder / 'failure.json')}
        (folder / 'site.json').write_text(json.dumps(spec, indent=2))
        return folder / 'site.json', folder / 'server.log'

    out = {'initial_recommendation': fallback, 'initial_recommendation_log': initial_log,
           'initial_recommendation_is_frozen_default': frozen_default}
    spec, log = spec_for('full')
    with CampaignProcess(spec, private_log=log) as campaign:
        toolbox = Toolbox(campaign.session, tool_seed)
        prefix_agent = LLMCampaignAgent(campaign.session, toolbox, make_completer('shared_day_0'), config,
                                        task=shared_view)
        prefix = prefix_agent.run_day_zero_design(day_zero_max_calls)
        prefix_transcript = campaign.session.transcript
        prefix_tool_calls = toolbox.calls
        view = campaign.session.task
        agent = LLMCampaignAgent(campaign.session, toolbox, make_completer('full'), config, task=view,
                                 history=prefix['history'], counts=prefix['counts'], notice=assignment_notice(view),
                                 notes=prefix.get('notes', ''))
        out['full'] = {'summary': agent.run(), 'transcript': campaign.session.transcript}
        out['full']['server_exit_code'] = campaign.close()
    out['shared_day_0'] = {'ended_by': prefix['ended_by'], 'starts': prefix['starts'], 'counts': prefix['counts'],
                           'transcript': prefix_transcript, 'log': prefix['log']}
    spec, log = spec_for('endpoint')
    with CampaignProcess(spec, private_log=log) as campaign:
        replay_prefix(campaign.session, prefix_transcript)
        toolbox = Toolbox(campaign.session, tool_seed)
        toolbox.calls = prefix_tool_calls
        view = campaign.session.task
        agent = LLMCampaignAgent(campaign.session, toolbox, make_completer('endpoint'), config, task=view,
                                 history=prefix['history'], counts=prefix['counts'], notice=assignment_notice(view),
                                 notes=prefix.get('notes', ''))
        out['endpoint'] = {'summary': agent.run(), 'transcript': campaign.session.transcript}
        out['endpoint']['server_exit_code'] = campaign.close()
    return out
