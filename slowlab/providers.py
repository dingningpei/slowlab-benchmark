"""Chat-completion providers as ``complete(messages) -> str`` (standard library only).

Ported from the Version 2.1 harness with two changes: failures raise
``ProviderError`` (an infrastructure error the orchestrator records) instead of
exiting the process, and no request ever carries project-identifying headers.
DeepSeek, OpenAI and any OpenAI-compatible endpoint share one path; a model
name containing "/" is routed through OpenRouter.
"""
from __future__ import annotations
import http.client
import json
import socket
import os
import time
import ssl
import urllib.error
import urllib.request
from pathlib import Path


class ProviderError(RuntimeError):
    """A provider call failed after its retries; an infrastructure failure, never a model outcome."""


def _ssl_context() -> ssl.SSLContext:
    """Build a usable TLS context.

    A python.org build of Python on macOS does not use the system certificate
    store, so a plain urlopen raises CERTIFICATE_VERIFY_FAILED. Prefer certifi's
    roots (pip install certifi) and fall back to the system default. Verification
    is never disabled -- an API key travels over this connection.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_CTX = _ssl_context()


def load_dotenv(path: str | Path | None = None) -> None:
    """Load the repository-root .env into the environment. Existing variables are
    not overwritten.

    .env is in .gitignore, so keys are never committed. If the file is absent this
    silently does nothing.
    """
    # SLOWLAB_ENV_FILE points at a private key file outside the repository (run hosts).
    p = Path(path) if path else Path(os.environ.get("SLOWLAB_ENV_FILE") or Path(__file__).resolve().parents[1] / ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

# OpenRouter: one key reaches every vendor, with model names of the form
# "provider/model". Any model name containing "/" goes to OpenRouter and skips
# the prefix table below.
OPENROUTER = ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions")

# For direct vendor access: model-name prefix -> (environment variable, endpoint)
PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions"),
    "gpt":      ("OPENAI_API_KEY",   "https://api.openai.com/v1/chat/completions"),
    "qwen":     ("DASHSCOPE_API_KEY",
                 "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
}


def _guess(model: str):
    if "/" in model:                      # e.g. openai/gpt-5.6-luna -> OpenRouter
        return OPENROUTER
    for prefix, cfg in PROVIDERS.items():
        if model.startswith(prefix):
            return cfg
    raise ProviderError(f"Unrecognised model {model!r}; add its prefix to PROVIDERS in providers.py.")


def openai_compatible(model: str, *, base_url: str | None = None,
                      api_key: str | None = None, temperature: float = 0.7,
                      max_tokens: int = 4096, timeout: float = 90.0,
                      max_attempts: int = 5, reasoning_cap: int = 1024,
                      reasoning_effort: str | None = None,
                      generation_seed: int | None = None,
                      json_mode: bool = False,
                      rpm: float = 0.0,
                      provider_only: list[str] | None = None,
                      expected_provider: str | None = None,
                      expected_actual_model: str | None = None) -> callable:
    """Returns a complete(messages) -> str, retrying network and rate-limit errors with exponential backoff."""
    load_dotenv()                      # allows the key to live in .env
    env_var, default_url = _guess(model)
    url = base_url or os.environ.get(f"{env_var[:-8]}_BASE_URL") or default_url
    # *_BASE_URL may be a full endpoint path or just a host (which is what the
    # vendor docs give). When only a host is supplied, append the OpenAI-compatible
    # path, otherwise we POST to the root and get a 404.
    if not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + ("/chat/completions" if url.rstrip("/").endswith("/v1")
                                 else "/v1/chat/completions"
                                 if "openai" in url or "dashscope" in url
                                 else "/chat/completions")
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ProviderError(
            f"{env_var} not found. Two ways to set it:\n"
            f"  1) export {env_var}=sk-...\n"
            f"  2) cp .env.example .env, then put the key in .env"
            f" (.env is already in .gitignore and will not be committed)")

    # Thinking tokens bill as output and can be 5-20x the visible reply; a long
    # think also eats max_tokens and turns an otherwise fine answer into a format
    # failure. Off by default.
    # Some endpoints make reasoning *mandatory* (GLM-5.3-flash returns 400
    # "Reasoning is mandatory for this endpoint"), so we fall back to capping it
    # rather than disabling it, which still bounds the cost. The downgrade happens
    # once and holds for the rest of the run.
    if "api.deepseek.com" in url:
        # DeepSeek's OpenAI Chat Completions schema uses a top-level
        # reasoning_effort plus a thinking toggle. A Responses/OpenRouter-style
        # `reasoning` object is accepted but ignored, leaving the default high
        # thinking mode enabled and potentially consuming the full output budget.
        mode = ({"thinking": {"type": "enabled"},
                 "reasoning_effort": reasoning_effort}
                if reasoning_effort else
                {"thinking": {"type": "disabled"}})
    else:
        mode = ({"reasoning": {"effort": reasoning_effort}}
                if reasoning_effort else
                {"reasoning": {"enabled": False}} if "openrouter" in url else {})

    def _body(messages):
        d = {"model": model, "messages": messages,
             "temperature": temperature, "max_tokens": max_tokens}
        if generation_seed is not None:
            d["seed"] = int(generation_seed)
        if json_mode:
            d["response_format"] = {"type": "json_object"}
        if "openrouter" in url:
            d["usage"] = {"include": True}   # OpenRouter returns the billed cost per call
        if provider_only:
            if "openrouter" not in url:
                raise ValueError("provider_only is supported only for OpenRouter routes")
            d["provider"] = {"only": list(provider_only), "allow_fallbacks": False}
        d.update(mode)
        return json.dumps(d).encode()

    _last = [0.0]

    def complete(messages):
        nonlocal mode
        if rpm > 0:                       # self-throttle: at most rpm calls per minute
            gap = 60.0 / rpm
            wait = gap - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        req = urllib.request.Request(url, data=_body(messages), headers=hdr)
        last = None
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                    payload = json.loads(r.read())
                choice = payload["choices"][0]
                msg = choice["message"]
                actual_model = payload.get("model")
                actual_provider = payload.get("provider")
                if expected_provider is not None and actual_provider != expected_provider:
                    raise RuntimeError(
                        f"provider mismatch: expected {expected_provider!r}, got {actual_provider!r}")
                if expected_actual_model is not None and actual_model != expected_actual_model:
                    raise RuntimeError(
                        f"model mismatch: expected {expected_actual_model!r}, got {actual_model!r}")
                complete.call_records.append({
                    "requested_model": model,
                    "actual_model": actual_model,
                    "provider": actual_provider,
                    "created": payload.get("created"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": payload.get("usage"),
                    "attempt": attempt + 1,
                    "generation_seed": generation_seed,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                    "reasoning_mode": json.dumps(mode, sort_keys=True),
                    "json_mode": json_mode,
                })
                txt = msg.get("content") or ""
                if not txt.strip():                # some models fill only the reasoning field
                    txt = msg.get("reasoning") or ""
                return txt
            except urllib.error.HTTPError as e:
                last = e
                detail = e.read()[:400].decode(errors="replace")
                if e.code == 400 and "easoning" in detail and mode.get("reasoning", {}).get(
                        "enabled") is False:
                    mode = {"reasoning": {"max_tokens": reasoning_cap}}
                    print(f"  [{model}] endpoint requires reasoning; capping it at {reasoning_cap} tok",
                          flush=True)
                    req = urllib.request.Request(url, data=_body(messages), headers=hdr)
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    wait = 2.0 * (2 ** attempt)
                    # This must be audible. It used to back off silently, so when
                    # 12-way concurrency stretched one call to 15 minutes the
                    # terminal showed only "slow", never the reason.
                    print(f"  [{model}] HTTP {e.code}, retrying in {wait:.0f}s "
                          f"({attempt + 1}/{max_attempts}). If this persists, concurrency is too high.",
                          flush=True)
                    time.sleep(wait)
                    continue
                raise ProviderError(f"HTTP {e.code}: {detail}")
            except (socket.timeout, TimeoutError) as e:
                last = e
                print(f"  [{model}] request timed out after {timeout:.0f}s "
                      f"({attempt + 1}/{max_attempts}). This is what too much concurrency looks like.", flush=True)
                continue
            except urllib.error.URLError as e:
                # A certificate error is permanent; retrying only wastes time. Give an actionable fix.
                if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                    raise ProviderError(
                        "TLS certificate verification failed. This is almost always a "
                        "python.org build of Python on macOS not using the system "
                        "certificate store. Two fixes:\n"
                        "  1) pip install certifi        (this module picks it up automatically)\n"
                        "  2) run the certificate installer shipped with Python, e.g.\n"
                        "     /Applications/Python\\ 3.12/Install\\ Certificates.command\n"
                        "Do not disable certificate verification -- your API key travels over this connection.")
                last = e
                time.sleep(2.0 * (2 ** attempt))
            except TimeoutError as e:
                last = e
                time.sleep(2.0 * (2 ** attempt))
            except (ConnectionError, http.client.HTTPException) as e:
                # The connection dropped after the request was sent (e.g. RemoteDisconnected). urllib
                # raises these unwrapped, so they used to escape the retry loop and end a campaign as an
                # infrastructure failure. Retry the identical request like any other network error
                # (decision 2026-10-05).
                last = e
                print(f"  [{model}] connection dropped ({type(e).__name__}), retrying "
                      f"({attempt + 1}/{max_attempts})", flush=True)
                time.sleep(2.0 * (2 ** attempt))
        raise ProviderError(f"still failing after {max_attempts} attempts: {last}")

    complete.call_records = []
    return complete
