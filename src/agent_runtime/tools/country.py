"""Country info tool using REST Countries API (free, no key)."""

import httpx
from agents import function_tool

REST_COUNTRIES_URL = "https://restcountries.com/v3.1/name"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


async def _get_country_info(country_name: str) -> str:
    """Get information about a country — capital, population, region, currency, languages.

    Args:
        country_name: Country name (e.g., "Indonesia", "Japan", "Brazil").
    """
    client = _get_client()
    response = await client.get(f"{REST_COUNTRIES_URL}/{country_name}")
    response.raise_for_status()

    data = response.json()
    if not data:
        return f"Country '{country_name}' not found."

    c = data[0]
    name = c["name"]["common"]
    capital = ", ".join(c.get("capital", ["N/A"]))
    region = c.get("region", "N/A")
    subregion = c.get("subregion", "")
    population = c.get("population", 0)
    flag = c.get("flag", "")

    currencies = ", ".join(
        f"{v['name']} ({v.get('symbol', '')})" for v in c.get("currencies", {}).values()
    )
    languages = ", ".join(c.get("languages", {}).values())

    return (
        f"{flag} {name} ({c['name'].get('official', name)})\n"
        f"  Capital: {capital}\n"
        f"  Region: {region} — {subregion}\n"
        f"  Population: {population:,}\n"
        f"  Currencies: {currencies or 'N/A'}\n"
        f"  Languages: {languages or 'N/A'}"
    )


get_country_info = function_tool(_get_country_info, name_override="get_country_info")
