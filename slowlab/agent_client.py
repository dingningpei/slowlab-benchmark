"""Agent-side access to one campaign through a separate executor process.

``CampaignProcess`` (held by the orchestrator) launches
``slowlab.executor_server`` with a private site spec, a scrubbed environment
and stderr sent to a private log. Agents and tools receive only its
``session``: a ``PublicSession`` exposing the public task view, ``dispatch``
and the public transcript. No private object exists in this process: site
parameters, weather, noise seed, physics and the private trace live only in
the server. The orchestrator knows the site-spec path; the session's API never
returns it, although Python introspection could reach the launcher. This is a
process boundary for our own harness code, backed by tests that public output
depends only on public history; it is not an operating-system sandbox.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

from .agent_protocol import PROTOCOL, canonical, digest, parse

ROOT = Path(__file__).resolve().parents[1]
_ENV_KEEP = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR')


class AgentApiError(Exception):
    """Base class for errors returned through the agent API."""


class InvalidAction(AgentApiError):
    """The agent's request was rejected; the campaign continues."""


class InfrastructureFailure(AgentApiError):
    """The campaign is paused; abort and record, never retry into a result."""


class ProtocolError(AgentApiError):
    """The two processes disagree about the protocol; treat as infrastructure."""


class PublicSession:
    """Everything an agent or public tool may touch for one campaign."""
    __slots__ = ('_send', '_task', '_transcript')

    def __init__(self, send, task: dict):
        self._send = send
        self._task = task
        self._transcript = []

    @property
    def task(self) -> dict:
        return copy.deepcopy(self._task)

    @property
    def transcript(self) -> list:
        return copy.deepcopy(self._transcript)

    def dispatch(self, action: dict) -> dict:
        request_line, response_line, response = self._send('dispatch', action=copy.deepcopy(action))
        self._transcript.append({'request': parse(request_line)['action'], 'response_sha256': digest(response_line),
                                 'ok': response['ok'],
                                 ('result' if response['ok'] else 'error'): response.get('result', response.get('error'))})
        if response['ok']:
            return copy.deepcopy(response['result'])
        error = response['error']
        if error['kind'] == 'invalid_action':
            raise InvalidAction(error['message'])
        raise InfrastructureFailure(error['message'])


class CampaignProcess:
    def __init__(self, site_spec: Path, *, private_log: Path, python: str = sys.executable):
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_MAX_THREADS='1')
        self._stderr = open(private_log, 'ab')
        self._process = subprocess.Popen([python, '-B', '-m', 'slowlab.executor_server', str(site_spec)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                         cwd=str(ROOT), env=env, text=True, encoding='utf-8', bufsize=1)
        self._next_id = 1
        self._dead = False
        request, response_line, response = self._exchange('hello')
        if not response['ok']:
            self._terminate()
            raise InfrastructureFailure(response['error']['message'])
        self.session = PublicSession(self._exchange, response['result'])

    def _exchange(self, op: str, **payload):
        if self._dead:
            raise InfrastructureFailure('campaign process is no longer running')
        request = {'protocol': PROTOCOL, 'id': self._next_id, 'op': op, **payload}
        line = canonical(request)
        try:
            self._process.stdin.write(line + '\n')
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            self._terminate()
            raise InfrastructureFailure('campaign process ended unexpectedly') from exc
        if not response_line:
            self._terminate()
            raise InfrastructureFailure('campaign process ended unexpectedly')
        response_line = response_line.rstrip('\n')
        response = parse(response_line)
        if response.get('protocol') != PROTOCOL or response.get('id') != self._next_id:
            self._terminate()
            raise ProtocolError('response does not match request')
        self._next_id += 1
        if not response['ok'] and response['error']['kind'] != 'invalid_action':
            self._terminate()
        return line, response_line, response

    def _terminate(self):
        self._dead = True
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._stderr.close()

    @property
    def returncode(self):
        return self._process.returncode

    def close(self) -> int:
        """Ask the server to write the private settlement and exit."""
        if not self._dead:
            self._exchange('close')
            self._terminate()
        return self._process.returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._dead:
            self._terminate()
        return False
