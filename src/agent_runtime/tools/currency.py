"""Currency conversion tool using Frankfurter API (ECB rates, free, no key)."""

import httpx
from agents import function_tool

FRANKFURTER_URL = "https://api.frankfurter.dev/latest"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


async def _convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using live exchange rates (raw implementation).

    Args:
        amount: Amount to convert.
        from_currency: Source currency code (e.g., "USD", "EUR", "IDR").
        to_currency: Target currency code (e.g., "JPY", "GBP").
    """
    client = _get_client()
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    try:
        response = await client.get(
            FRANKFURTER_URL,
            params={"amount": amount, "from": from_currency, "to": to_currency},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError:
        return f"Error: Invalid currency code(s) — {from_currency} or {to_currency} not supported."

    data = response.json()
    rate = data["rates"][to_currency]

    return f"{amount:,.2f} {from_currency} = {rate:,.2f} {to_currency}\nDate: {data['date']}"


convert_currency = function_tool(_convert_currency)
