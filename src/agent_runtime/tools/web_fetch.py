"""Web fetch tool powered by TinyFish Fetch API (free, no credits).

Endpoint: POST https://api.fetch.tinyfish.ai
Docs: https://docs.tinyfish.ai/fetch-api
Rate limit: 150 URLs/min (free tier). Max 10 URLs per request.
Supports HTML, PDF, JSON endpoints. Rejects binary (images, video).
"""

from agents import function_tool

from agent_runtime.tools._tinyfish_common import tinyfish_request

TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"


async def _web_fetch(urls: list[str], format: str = "markdown") -> str:
    """Fetch and extract clean content from one or more URLs.

    Renders pages in a real browser and returns extracted text — works on
    JavaScript-heavy sites, PDFs, and JSON endpoints.

    Args:
        urls: One or more URLs to fetch. Max 10 per call. Must be http/https.
        format: Output format — "markdown" (default), "html", or "json".
    """
    if not urls:
        return "No URLs provided."

    if len(urls) > 10:
        return f"Too many URLs ({len(urls)}). Maximum is 10 per request."

    data = await tinyfish_request(
        "POST",
        TINYFISH_FETCH_URL,
        json={"urls": urls, "format": format, "ttl": 0},
    )

    results = data.get("results", [])
    errors = data.get("errors", [])

    # Collect per-URL error messages
    error_map: dict[str, str] = {}
    for e in errors:
        url = e.get("url", "unknown")
        code = e.get("error", "unknown_error")
        status = e.get("status")
        msg = f"{code}" + (f" (HTTP {status})" if status else "")
        error_map[url] = msg

    parts: list[str] = []

    for r in results:
        title = r.get("title", "")
        text = r.get("text", "")
        url = r.get("url", "")

        if not text:
            parts.append(f"## {title or url}\n\n[No extractable text content]")
            continue

        header = f"## {title}\n\n" if title else ""
        parts.append(header + text)

    # Report failures
    failed_urls = [u for u in urls if u not in {r.get("url") for r in results}]
    if failed_urls:
        lines = []
        for u in failed_urls:
            reason = error_map.get(u, "no content returned")
            lines.append(f"- {u}: {reason}")
        parts.append("## Failed URLs\n\n" + "\n".join(lines))

    if not parts:
        return "No content could be extracted from the provided URLs."

    return "\n\n---\n\n".join(parts)


web_fetch = function_tool(_web_fetch)
