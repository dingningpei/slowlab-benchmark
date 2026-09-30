"""Private executor server: ``python -m slowlab.executor_server <site-spec.json>``.

Holds every private input of one campaign (site parameters, weather year,
noise seed, physics, private trace) and speaks ``agent_protocol`` on stdin and
stdout. Nothing private is ever written to stdout: agent mistakes return an
``invalid_action`` error with the executor's own message; any failure while
physical time moves, or any unexpected error, returns a generic
``infrastructure_failure``, writes a private failure record and exits with
status 3. The settlement is written to a private file on ``close`` and never
sent back.

Site spec (private JSON; relative paths resolve against the repository root):
``backend`` (greenlight | fake), ``contract``, ``feedback_mode``,
``fallback_policy``, ``origin_utc``, ``weather`` {cache, plan}, ``greenlight_source``,
``sensor_noise`` {config, seed, setting} or null, ``soil_boundary_c``, ``trace``,
``settlement_out``, ``failure_out``.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .agent_protocol import PROTOCOL, canonical, parse, public_task_view
from .site_parameters import site_contract

ROOT = Path(__file__).resolve().parents[1]
AGENT_ERRORS = (ValueError, PermissionError)


def _path(value) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_weather(w: dict, origin: datetime):
    """Private weather reader from a spec: the development archive, or one audited formal year."""
    from .cabauw_weather import CabauwLc1FormalWeather, CabauwLc1Weather
    if w.get('kind') == 'formal':
        if origin.year != int(w['year']) - 1:
            raise ValueError('origin_utc does not start the formal weather year')
        return CabauwLc1FormalWeather(_path(w['cache']), _path(w['audit']), int(w['year']), _path(w.get('dev_cache')))
    return CabauwLc1Weather(_path(w['cache']), _path(w['plan']))


def build_executor(spec: dict):
    from .campaign_executor import CampaignExecutor
    site = spec.get('site') or {}
    contract = site_contract(json.loads(_path(spec['contract']).read_text()), site)
    origin = datetime.fromisoformat(spec['origin_utc'])
    if contract['evaluation'].get('deployment'):
        # Campaign day 0 is 1 January, 00:00 CET (UTC+1) for every site.
        if (origin.utcoffset() is None or origin.utcoffset().total_seconds() != 0
                or (origin.month, origin.day, origin.hour, origin.minute) != (12, 31, 23, 0)):
            raise ValueError('origin_utc must be 31 December 23:00 UTC (1 January 00:00 CET)')
    kwargs = dict(feedback_mode=spec['feedback_mode'], fallback_policy=spec['fallback_policy'],
                  origin_utc=datetime.fromisoformat(spec['origin_utc']),
                  soil_boundary_c=spec.get('soil_boundary_c'),
                  unit_parameters=site.get('unit_parameters'))
    trace_path = _path(spec.get('trace'))
    trace = gzip.open(trace_path, 'wt', encoding='utf-8') if trace_path else None
    kwargs['trace_sink'] = trace
    if spec['backend'] in ('fake', 'fake_failing'):
        from . import fake_backend
        lifecycle = fake_backend.FakeLifecycle if spec['backend'] == 'fake' else fake_backend.FailingLifecycle
        executor = CampaignExecutor(contract, None, None, lifecycle_factory=lifecycle,
                                    sample_endpoint=fake_backend.sample_endpoint, **kwargs)
    elif spec['backend'] == 'greenlight':
        from .greenlight_source import resolve_greenlight_source
        from .sensor_noise import SensorNoise
        source = resolve_greenlight_source(_path(spec.get('greenlight_source')), contract)
        weather = build_weather(spec['weather'], origin)
        noise_spec = spec.get('sensor_noise')
        noise = (SensorNoise.from_file(_path(noise_spec['config']), noise_spec['seed'], noise_spec.get('setting', 'main'))
                 if noise_spec else None)
        executor = CampaignExecutor(contract, source, weather, sensor_noise=noise, **kwargs)
    else:
        raise ValueError('unknown backend')
    return contract, executor, trace


def serve(spec_path: Path, inbound, outbound) -> int:
    spec = json.loads(Path(spec_path).read_text())
    contract, executor, trace = build_executor(spec)
    task = public_task_view(contract, spec['feedback_mode'])
    expected_id = 1

    def reply(message_id, ok, payload):
        body = {'protocol': PROTOCOL, 'id': message_id, 'ok': ok, ('result' if ok else 'error'): payload}
        outbound.write(canonical(body) + '\n')
        outbound.flush()

    def fail(message_id, exc):
        record = {'clock': executor.clock, 'error': type(exc).__name__ + ': ' + str(exc),
                  'executor_failure': executor._failed, 'traceback': traceback.format_exc()}
        failure = _path(spec.get('failure_out'))
        if failure:
            failure.write_text(json.dumps(record, indent=2, default=str) + '\n')
        print(record['traceback'], file=sys.stderr)
        if trace:
            trace.close()
        reply(message_id, False, {'kind': 'infrastructure_failure',
                                  'message': 'infrastructure failure; the campaign is paused'})
        return 3

    for line in inbound:
        try:
            message = parse(line)
            if (message.get('protocol') != PROTOCOL or message.get('id') != expected_id
                    or message.get('op') not in ('hello', 'dispatch', 'close')):
                raise ValueError('malformed request')
        except ValueError as exc:
            reply(None, False, {'kind': 'protocol_error', 'message': str(exc)})
            return 2
        expected_id += 1
        op = message['op']
        if op == 'hello':
            reply(message['id'], True, task)
        elif op == 'dispatch':
            try:
                result = executor.dispatch(message.get('action'))
            except AGENT_ERRORS as exc:
                if executor._failed is not None:
                    return fail(message['id'], exc)
                reply(message['id'], False, {'kind': 'invalid_action', 'message': str(exc)})
            except Exception as exc:  # noqa: BLE001 - any other error is infrastructure
                return fail(message['id'], exc)
            else:
                reply(message['id'], True, result)
        else:
            if trace:
                trace.close()
            at_deadline = executor.clock == executor.deadline_seconds and executor._failed is None
            record = (executor.settlement() if at_deadline else
                      {'status': 'closed_before_deadline', 'clock': executor.clock})
            out = _path(spec.get('settlement_out'))
            if out:
                out.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            reply(message['id'], True, {'closed': True})
            return 0
    return 0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m slowlab.executor_server <site-spec.json>')
    # Keep the protocol on the original stdout; anything else that prints
    # (libraries, compilers) goes to stderr, which the launcher keeps private.
    protocol_out = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    raise SystemExit(serve(Path(sys.argv[1]), sys.stdin, protocol_out))


if __name__ == '__main__':
    main()
