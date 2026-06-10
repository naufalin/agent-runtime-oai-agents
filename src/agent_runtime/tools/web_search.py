"""Web search tool powered by TinyFish Search API (free, no credits)..

Endpoint: GET https://api.search.tinyfish.ai
Docs: https://docs.tinyfish.ai/search-api
Rate limit: 30 req/min (free tier).
"""

from agents import function_tool

from agent_runtime.tools._tinyfish_common import tinyfish_request

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"


async def _web_search(query: str, location: str = "US", language: str = "en", page: int = 0) -> str:
    """Search the web for current information.

    Args:
        query: Search query string. Supports operators like site:example.com.
        location: Country code for geo-targeting (e.g., US, GB, ID).
        language: Language code for results (e.g., en, id, ja).
        page: Page number for pagination (0-indexed, max 10).
    """
    data = await tinyfish_request(
        "GET",
        TINYFISH_SEARCH_URL,
        params={"query": query, "location": location, "language": language, "page": page},
    )

    results = data.get("results", [])
    if not results:
        return "No results found for your query."

    formatted = []
    for r in results[:10]:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        site = r.get("site_name", "")
        site_tag = f" [{site}]" if site else ""
        formatted.append(f"[{title}]({url}){site_tag}\n  {snippet}")

    return "\n\n".join(formatted)


web_search = function_tool(_web_search)
