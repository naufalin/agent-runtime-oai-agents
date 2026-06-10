"""Shared TinyFish client, authentication, and retry logic.

Internal module — not exported as a tool. Used by web_search, web_fetch,
web_agent, and web_browser.
"""

import asyncio

import httpx

_client: httpx.AsyncClient | None = None


class _RetryableError(Exception):
    """Transient failure that should be retried once."""


def _ensure_api_key() -> str:
    """Return the API key from env, raise if missing."""
    import os

    key = os.environ.get("TINYFISH_API_KEY", "")
    if not key:
        raise RuntimeError(
            "TINYFISH_API_KEY is not set. Get a free key at https://agent.tinyfish.ai/api-keys"
        )
    return key


def get_client(timeout: float = 150.0) -> httpx.AsyncClient:
    """Get or create the shared httpx.AsyncClient with API key header.

    Args:
        timeout: Request timeout in seconds. Defaults to 150s (Fetch backend
                 is 110s/URL, CDN ceiling 120s).
    """
    global _client
    if _client is None:
        key = _ensure_api_key()
        _client = httpx.AsyncClient(
            headers={"X-API-Key": key},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
    return _client


async def handle_response(response: httpx.Response) -> dict:
    """Evaluate an httpx response, raise on errors, return JSON on success.

    Raises:
        ValueError: 400 — bad request (caller can fix params).
        RuntimeError: 401 (auth), 402 (billing), exhausted retries.
        _RetryableError: 429 or 503 — caller should retry once.
    """
    status = response.status_code

    if status == 200:
        return response.json()

    if status == 400:
        raise ValueError(f"TinyFish bad request (400): {response.text}")

    if status == 401:
        raise RuntimeError("TinyFish auth failed (401). Check TINYFISH_API_KEY.")

    if status == 402:
        raise RuntimeError("TinyFish insufficient credits (402). Add credits at agent.tinyfish.ai.")

    if status == 429:
        await asyncio.sleep(2.0)
        raise _RetryableError("TinyFish rate limited (429), retrying")

    if status == 503:
        await asyncio.sleep(1.0)
        raise _RetryableError("TinyFish service unavailable (503), retrying")

    response.raise_for_status()
    return {}  # unreachable, keeps type checker happy


async def tinyfish_request(method: str, url: str, **kwargs) -> dict:
    """Make a TinyFish API request with one automatic retry on transient errors.

    Args:
        method: HTTP method ("GET" or "POST").
        url: Full TinyFish endpoint URL.
        **kwargs: Forwarded to httpx client.request() — params, json, timeout, etc.
    """
    client = get_client()
    resp = await client.request(method, url, **kwargs)

    try:
        return await handle_response(resp)
    except _RetryableError:
        resp = await client.request(method, url, **kwargs)
        try:
            return await handle_response(resp)
        except _RetryableError as e:
            raise RuntimeError(str(e)) from e
