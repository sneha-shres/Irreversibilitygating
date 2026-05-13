from __future__ import annotations

import concurrent.futures
import os
from functools import lru_cache


_LAST_CALL_TS = [0.0]
_MIN_INTERVAL_S = float(os.environ.get("IRRGATE_LLM_MIN_INTERVAL", "1.0"))
_CALL_TIMEOUT_S = float(os.environ.get("IRRGATE_LLM_TIMEOUT", "120.0"))


@lru_cache(maxsize=1)
def get_gemini_client():
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set for Gemini calls")
    return genai.Client(vertexai=True, project=project, location=location)


def _throttle() -> None:
    import time as _time
    elapsed = _time.time() - _LAST_CALL_TS[0]
    if elapsed < _MIN_INTERVAL_S:
        _time.sleep(_MIN_INTERVAL_S - elapsed)
    _LAST_CALL_TS[0] = _time.time()


def generate_with_backoff(client, *, model, contents, config, max_retries: int = 6):
    """Call generate_content with throttling + exponential backoff on 429 and network errors.

    Returns the response, or None if every retry was exhausted.
    """
    import sys as _sys
    import time as _time
    from google.genai import errors as _gerrors

    # Network-level errors that are safe to retry (transient disconnects + call timeouts).
    _RETRYABLE_NETWORK = (
        ConnectionError,
        TimeoutError,
        OSError,
        concurrent.futures.TimeoutError,
    )
    try:
        import httpx as _httpx
        _RETRYABLE_NETWORK = _RETRYABLE_NETWORK + (_httpx.RemoteProtocolError, _httpx.ConnectError, _httpx.ReadError)
    except ImportError:
        pass

    delay = 4.0
    for attempt in range(max_retries):
        _throttle()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                _future = _pool.submit(
                    client.models.generate_content,
                    model=model, contents=contents, config=config,
                )
                return _future.result(timeout=_CALL_TIMEOUT_S)
        except _gerrors.ClientError as exc:
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            is_429 = status_code == 429 or "RESOURCE_EXHAUSTED" in str(exc)
            if not is_429:
                raise
            if attempt == max_retries - 1:
                print(f"[gemini] 429 final fallback after {max_retries} attempts", file=_sys.stderr, flush=True)
                return None
            print(f"[gemini] 429 retry {attempt+1}/{max_retries} sleeping {delay:.0f}s",
                  file=_sys.stderr, flush=True)
        except _RETRYABLE_NETWORK as exc:
            if attempt == max_retries - 1:
                print(f"[gemini] network error final fallback after {max_retries} attempts: {exc}",
                      file=_sys.stderr, flush=True)
                return None
            print(f"[gemini] network error retry {attempt+1}/{max_retries} sleeping {delay:.0f}s: {exc}",
                  file=_sys.stderr, flush=True)
        _time.sleep(delay)
        delay = min(delay * 2, 60.0)
    return None
