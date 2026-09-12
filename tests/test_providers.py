"""Endpoint assembly: *_BASE_URL must work whether it is a host or a full path.

The .env.example once gave a host while the code treated it as a complete
endpoint, so we POSTed to the root and got HTTP 404.
"""
import os
import pytest
from slowlab.providers import openai_compatible


def _url(monkeypatch, model, base=None):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    if base is None:
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base)
    fn = openai_compatible(model)
    return fn.__closure__ and next(
        c.cell_contents for c in fn.__closure__
        if isinstance(c.cell_contents, str) and c.cell_contents.startswith("http"))


@pytest.mark.parametrize("base,want", [
    (None, "https://api.deepseek.com/chat/completions"),
    ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
    ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
    ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
    ("https://api.deepseek.com/chat/completions", "https://api.deepseek.com/chat/completions"),
])
def test_base_url_forms_all_resolve(monkeypatch, base, want):
    assert _url(monkeypatch, "deepseek-chat", base) == want


def test_mandatory_reasoning_endpoint_falls_back_to_a_cap(monkeypatch):
    """Some endpoints make reasoning mandatory (GLM-5.3-flash returns 400
    "Reasoning is mandatory").

    That is not a reason to give up: downgrade to *capping* rather than disabling,
    which still bounds the cost, and do the downgrade once so the rest of the run
    does not hit a 400 on every call.
    """
    import io, json, urllib.error
    import slowlab.providers as P

    calls = []

    class _Resp:
        def __init__(self, d): self._d = json.dumps(d).encode()
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake(req, timeout=None, context=None):
        body = json.loads(req.data.decode())
        calls.append(body)
        if body.get("reasoning", {}).get("enabled") is False:
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":{"message":"Reasoning is mandatory for this '
                           b'endpoint and cannot be disabled."}}'))
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(P.urllib.request, "urlopen", fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    f = P.openai_compatible("z-ai/glm-5.3-flash", reasoning_cap=777)
    assert f([{"role": "user", "content": "hi"}]) == "ok"
    assert calls[0]["reasoning"] == {"enabled": False}
    assert calls[1]["reasoning"] == {"max_tokens": 777}
    f([{"role": "user", "content": "hi"}])
    assert calls[2]["reasoning"] == {"max_tokens": 777}     # no longer retries with it disabled
