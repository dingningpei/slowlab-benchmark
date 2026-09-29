#!/usr/bin/env python3
"""Bounded real-GreenLight campaign dispatch integration prefix, not a full episode."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.cabauw_weather import CabauwLc1Weather
from slowlab.campaign_executor import CampaignExecutor
from slowlab.feedback_view import FeedbackView


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    contract = json.loads((ROOT / 'configs/task_contract_v3.json').read_text())
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/weather_gapfilled_plan.json')
    origin = datetime(2016, 12, 31, 23, tzinfo=timezone.utc)
    began = time.monotonic()
    result = {'status': 'running', 'scope': 'four-unit real-GreenLight 2-day-plus-one-tick dynamic campaign prefix; no formal site or LLM'}
    try:
        with gzip.open(args.trace, 'wt', encoding='utf-8') as trace:
            campaign = CampaignExecutor(contract, args.source, weather,
                                        feedback_mode='full', fallback_policy=policies['policy_a'],
                                        origin_utc=origin, trace_sink=trace)
            campaign.dispatch({'action': 'start', 'unit': 0, 'policy': policies['policy_a']})
            campaign.dispatch({'action': 'start', 'unit': 1, 'policy': policies['policy_b']})
            campaign.dispatch({'action': 'advance', 'day': 300 / 86400})
            first = campaign.dispatch({'action': 'observe', 'unit': 0,
                                       'variable': 'air_temperature_c', 'start_day': 0})
            if len(first['records']) != 2 or any(r['available_at'] > campaign.clock for r in first['records']):
                raise AssertionError('first observation time violated')
            campaign.dispatch({'action': 'stop', 'unit': 0})
            campaign.dispatch({'action': 'advance', 'day': 1})
            campaign.dispatch({'action': 'start', 'unit': 2, 'policy': policies['policy_a']})
            campaign.dispatch({'action': 'advance', 'day': 2 + 300 / 86400})
            if campaign._lifecycles['0'].mode != 'empty' or campaign._lifecycles['0'].ready_at > campaign.clock:
                raise AssertionError('stopped crop not ready after two-day cleanup')
            campaign.dispatch({'action': 'start', 'unit': 0, 'policy': policies['policy_b']})
            final = campaign.dispatch({'action': 'observe', 'unit': 0,
                                       'variable': 'air_temperature_c',
                                       'start_day': 2, 'end_day': 2 + 300 / 86400})
            if len(final['records']) != 2 or any(r['available_at'] > campaign.clock for r in final['records']):
                raise AssertionError('final observation time violated')
            if campaign._view().operational_status('0')['run_index'] != 2:
                raise AssertionError('replant did not create second run')
            if campaign._endpoint.final_aggregate('0')['reason'] != 'stop':
                raise AssertionError('early stop aggregate missing')
            if campaign._endpoint.final_aggregate('1') is not None:
                raise AssertionError('running crop final outcome released')
            if any(len(life.engine.model.input_data) != 1 for life in campaign._lifecycles.values()):
                raise AssertionError('future model input rows retained')
            result.update(status='passed_dynamic_campaign_prefix', clock=campaign.clock,
                          starts=campaign._starts, trace_ticks=campaign._trace_ticks,
                          trace_uncompressed_sha256=campaign._trace_sha256.hexdigest(),
                          event_log=campaign.event_log,
                          unit_status={u: campaign._view().operational_status(u) for u in campaign.units},
                          ledgers={u: campaign._ledgers[u].summary() for u in campaign.units},
                          endpoint_final_0=campaign._endpoint.final_aggregate('0'))
        # Independently parse streamed private trace; no in-memory frame accumulation.
        seen = 0
        digest = hashlib.sha256()
        with gzip.open(args.trace, 'rt', encoding='utf-8') as trace:
            for line in trace:
                digest.update(line.encode())
                frame = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                assert frame['tick'] == seen and frame['start'] == seen * 300 and frame['end'] == (seen + 1) * 300
                assert set(frame['units']) == {'0', '1', '2', '3'}
                for unit, entry in frame['units'].items():
                    assert entry['model_input_rows'] == 1
                    assert entry['requested'] == entry['realised']
                    assert all(record['available_at'] <= frame['start']
                               for record in entry['controller_records'].values())
                    assert all(record['available_at'] == frame['end']
                               for record in entry['public_endpoint'].values())
                seen += 1
        assert seen == result['trace_ticks'] == 577
        assert digest.hexdigest() == result['trace_uncompressed_sha256']
        result['trace_audit'] = 'passed_causal_tick_order_and_digest'
    except Exception as exc:
        traceback.print_exc()
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    result['elapsed_seconds'] = time.monotonic() - began
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'clock', 'trace_ticks', 'error', 'elapsed_seconds')}))
    if result['status'] != 'passed_dynamic_campaign_prefix':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
