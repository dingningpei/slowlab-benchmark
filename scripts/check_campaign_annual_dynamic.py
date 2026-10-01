#!/usr/bin/env python3
"""Bounded four-unit GreenLight annual campaign integration; no LLM or benchmark score."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.cabauw_weather import CabauwLc1Weather
from slowlab.greenlight_source import resolve_greenlight_source
from slowlab.campaign_executor import CampaignExecutor
from slowlab.sensor_noise import SensorNoise


def rss_bytes() -> int:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux VmRSS unavailable')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=None,
                        help='GreenLight checkout at the pinned commit; default: the installed [greenlight] extra')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--pilot-days', type=int, default=365)
    parser.add_argument('--max-seconds', type=int, default=10800)
    parser.add_argument('--max-rss-bytes', type=int, default=350_000_000)
    parser.add_argument('--noise-seed', type=int, default=None,
                        help='development sensor-noise seed; omit for deterministic virtual sensors')
    parser.add_argument('--noise-config', type=Path, default=ROOT / 'configs/sensor_noise_v0.json')
    parser.add_argument('--noise-setting', default='main')
    parser.add_argument('--contract', type=Path, default=ROOT / 'configs/task_contract_v6.json')
    args = parser.parse_args()
    if not 1 <= args.pilot_days <= 365 or args.max_seconds <= 0 or args.max_rss_bytes <= 0:
        raise ValueError('invalid pilot or resource bound')
    contract = json.loads(args.contract.read_text())
    pinned_noise = contract['observations'].get('noise')
    if args.noise_seed is not None and isinstance(pinned_noise, dict):
        if hashlib.sha256(args.noise_config.read_bytes()).hexdigest() != pinned_noise['sha256']:
            raise ValueError('noise config differs from the one pinned by the contract')
    args.source = resolve_greenlight_source(args.source, contract)
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    policy_c = dict(policies['policy_b'], day_temperature_c=20, night_temperature_c=16,
                    co2_target_ppm=600, supplemental_light_hours=8)
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/weather_gapfilled_plan.json')
    origin = datetime(2016, 12, 31, 23, tzinfo=timezone.utc)
    noise = (SensorNoise.from_file(args.noise_config, args.noise_seed, args.noise_setting)
             if args.noise_seed is not None else None)
    noise_record = (None if noise is None else
                    {**noise.describe(), 'seed': args.noise_seed,
                     'config': str(args.noise_config.relative_to(ROOT)),
                     'config_sha256': hashlib.sha256(args.noise_config.read_bytes()).hexdigest(),
                     'seed_scope': 'development seed; not a formal site'})
    began = time.monotonic()
    result = {'status': 'running', 'scope': 'four-unit 2017 weather annual event-driven integration pilot; fixed scripted actions; no LLM, no formal site, no real greenhouse validation',
              'pilot_days': args.pilot_days, 'max_seconds': args.max_seconds,
              'max_rss_bytes': args.max_rss_bytes, 'weather_year': 2017,
              'sensor_noise': noise_record,
              'contract_id': contract['contract_id']}
    campaign = None
    try:
        with args.progress.open('w') as progress, gzip.open(args.trace, 'wt', encoding='utf-8') as trace:
            def daily_progress(executor: CampaignExecutor) -> None:
                elapsed = time.monotonic() - began
                rss = rss_bytes()
                if elapsed > args.max_seconds:
                    raise TimeoutError('campaign exceeded wall-time bound')
                if rss > args.max_rss_bytes:
                    raise MemoryError('campaign exceeded RSS bound')
                row = {'day': executor.clock // 86400, 'clock_seconds': executor.clock,
                       'trace_ticks': executor._trace_ticks, 'elapsed_seconds': elapsed,
                       'rss_bytes': rss,
                       'unit_status': {u: executor._view().operational_status(u)
                                       for u in executor.units},
                       'cumulative_per_m2': {u: executor._ledgers[u].summary()['per_m2']
                                             for u in executor.units}}
                progress.write(json.dumps(row, allow_nan=False) + '\n')
                progress.flush()
                trace.flush()

            campaign = CampaignExecutor(contract, args.source, weather,
                                        feedback_mode='full', fallback_policy=policies['policy_a'],
                                        origin_utc=origin, trace_sink=trace,
                                        progress_hook=daily_progress, sensor_noise=noise)
            campaign.dispatch({'action': 'start', 'unit': 0, 'policy': policies['policy_a']})
            campaign.dispatch({'action': 'start', 'unit': 1, 'policy': policies['policy_b']})
            if args.pilot_days >= 14:
                campaign.dispatch({'action': 'advance', 'day': 14})
                observation = campaign.dispatch({'action': 'observe', 'unit': 0,
                                                 'variable': 'air_temperature_c',
                                                 'start_day': 13, 'end_day': 14})
                values = [r['value'] for r in observation['records']]
                if not values or any(not math.isfinite(v) for v in values):
                    raise AssertionError('missing or nonfinite midway observation')
                if any(r['available_at'] > campaign.clock for r in observation['records']):
                    raise AssertionError('future observation leaked')
                mean_air_c = sum(values) / len(values)
                next_policy_name = 'policy_b' if mean_air_c >= 22 else 'policy_c'
                result['midway_decision'] = {'observed_at_day': 14,
                                             'record_count': len(values),
                                             'observed_mean_air_c': mean_air_c,
                                             'predeclared_threshold_c': 22,
                                             'selected_next_policy': next_policy_name}
                campaign.dispatch({'action': 'stop', 'unit': 0})
                campaign.dispatch({'action': 'start', 'unit': 2, 'policy': policies['policy_a']})
            if args.pilot_days >= 16:
                campaign.dispatch({'action': 'advance', 'day': 16})
                campaign.dispatch({'action': 'start', 'unit': 0,
                                   'policy': policies['policy_b'] if next_policy_name == 'policy_b' else policy_c})
            if args.pilot_days >= 30:
                campaign.dispatch({'action': 'advance', 'day': 30})
                campaign.dispatch({'action': 'start', 'unit': 3, 'policy': policies['policy_b']})
            if args.pilot_days >= 60:
                campaign.dispatch({'action': 'advance', 'day': 60})
                campaign.dispatch({'action': 'stop', 'unit': 3})
            campaign.dispatch({'action': 'advance', 'day': args.pilot_days})
            if args.pilot_days == 365:
                campaign.dispatch({'action': 'recommend', 'policy': policies['policy_b']})
                settlement = campaign.settlement()
                if settlement['clock'] != 365 * 86400 or settlement['trace_ticks'] != 365 * 288:
                    raise AssertionError('annual settlement incomplete')
                if settlement['trace_sha256'] != campaign._trace_sha256.hexdigest():
                    raise AssertionError('trace digest mismatch')
                result['settlement'] = settlement
            result.update(status=('passed_annual_dynamic_campaign' if args.pilot_days == 365
                                  else 'passed_dynamic_prefix'),
                          event_log=campaign.event_log, trace_ticks=campaign._trace_ticks,
                          trace_sha256=campaign._trace_sha256.hexdigest(),
                          final_clock=campaign.clock)
    except Exception as exc:
        traceback.print_exc()
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        result['elapsed_seconds'] = time.monotonic() - began
        result['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if campaign is not None:
            result['completed_days'] = campaign.clock / 86400
            result['trace_ticks'] = campaign._trace_ticks
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: result.get(k) for k in
                          ('status', 'completed_days', 'trace_ticks', 'elapsed_seconds', 'error')}),
              flush=True)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
