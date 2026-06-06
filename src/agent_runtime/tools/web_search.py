"""Web search tool powered by TinyFish Search API."""

import os

import httpx
from agents import function_tool

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"X-API-Key": os.environ.get("TINYFISH_API_KEY", "")},
            timeout=15.0,
        )
    return _client


async def _web_search(query: str, location: str = "US", language: str = "en") -> str:
    """Search the web for current information (raw implementation).

    Args:
        query: Search query string.
        location: Country code for geo-targeting (e.g., US, GB, ID).
        language: Language code (e.g., en, id).
    """
    client = _get_client()
    response = await client.get(
        TINYFISH_SEARCH_URL,
        params={"query": query, "location": location, "language": language},
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    if not results:
        return "No results found for your query."

    formatted = []
    for r in results[:5]:
        formatted.append(f"[{r['title']}]({r['url']})\n  {r['snippet']}")

    return "\n\n".join(formatted)


web_search = function_tool(_web_search)
