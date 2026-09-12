"""Wrap the common chat APIs into the `complete(messages) -> str` SlowLab needs.

Standard library only (urllib), no SDK to install. DeepSeek, OpenAI and any
OpenAI-compatible endpoint take the same path; only base_url and the model name
differ.

    export DEEPSEEK_API_KEY=sk-...
    python scripts/run_llm.py --model deepseek-chat --seeds 20
"""
from __future__ import annotations
import json
import socket
import os
import time
import ssl
import urllib.error
import urllib.request
from pathlib import Path


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
    p = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
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
    raise SystemExit(
        f"Unrecognised model {model!r}. Add a line to PROVIDERS in "
        f"slowlab/providers.py, or pass your own complete function to LLMAgent.")


def openai_compatible(model: str, *, base_url: str | None = None,
                      api_key: str | None = None, temperature: float = 0.7,
                      max_tokens: int = 4096, timeout: float = 90.0,
                      max_attempts: int = 5, reasoning_cap: int = 1024,
                      rpm: float = 0.0) -> callable:
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
        raise SystemExit(
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
    mode = {"reasoning": {"enabled": False}} if "openrouter" in url else {}

    def _body(messages):
        d = {"model": model, "messages": messages,
             "temperature": temperature, "max_tokens": max_tokens}
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
        if "openrouter" in url:                    # OpenRouter's leaderboard attribution headers
            hdr["HTTP-Referer"] = "https://github.com/slowlab-benchmark"
            hdr["X-Title"] = "SlowLab"
        req = urllib.request.Request(url, data=_body(messages), headers=hdr)
        last = None
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                    payload = json.loads(r.read())
                msg = payload["choices"][0]["message"]
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
                raise SystemExit(f"HTTP {e.code}: {detail}")
            except (socket.timeout, TimeoutError) as e:
                last = e
                print(f"  [{model}] request timed out after {timeout:.0f}s "
                      f"({attempt + 1}/{max_attempts}). This is what too much concurrency looks like.", flush=True)
                continue
            except urllib.error.URLError as e:
                # A certificate error is permanent; retrying only wastes time. Give an actionable fix.
                if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                    raise SystemExit(
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
        raise SystemExit(f"still failing after {max_attempts} attempts: {last}")

    return complete
