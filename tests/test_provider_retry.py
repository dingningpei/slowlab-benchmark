import http.client
import io
import json

import pytest

from slowlab import providers


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def reply():
    body = {'model': 'z-ai/glm-5.3-flash', 'provider': 'Z.AI', 'created': 1,
            'choices': [{'message': {'content': '{"ok": true}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 3}}
    return FakeResponse(json.dumps(body).encode())


@pytest.mark.parametrize('error', [http.client.RemoteDisconnected('Remote end closed connection without response'),
                                   ConnectionResetError('reset by peer'), http.client.IncompleteRead(b'')])
def test_a_dropped_connection_is_retried_with_the_same_request(monkeypatch, error):
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(req.data)
        if len(calls) == 1:
            raise error
        return reply()
    monkeypatch.setattr(providers.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(providers.time, 'sleep', lambda s: None)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    complete = providers.openai_compatible('z-ai/glm-5.3-flash', expected_provider='Z.AI')
    assert complete([{'role': 'user', 'content': 'hi'}]) == '{"ok": true}'
    assert len(calls) == 2 and calls[0] == calls[1]
    assert complete.call_records[-1]['attempt'] == 2


def test_persistent_drops_end_as_a_provider_error(monkeypatch):
    def urlopen(req, timeout=None, context=None):
        raise http.client.RemoteDisconnected('gone')
    monkeypatch.setattr(providers.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(providers.time, 'sleep', lambda s: None)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    complete = providers.openai_compatible('z-ai/glm-5.3-flash', expected_provider='Z.AI')
    with pytest.raises(providers.ProviderError):
        complete([{'role': 'user', 'content': 'hi'}])
