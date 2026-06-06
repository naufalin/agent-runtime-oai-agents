"""Tests for currency conversion tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_runtime.tools.currency import _convert_currency as convert_currency


def _make_response(json_data: dict, status_code: int = 200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_convert_currency_success():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response(
        {
            "amount": 100.0,
            "base": "USD",
            "date": "2026-06-06",
            "rates": {"EUR": 92.5},
        }
    )

    with patch("agent_runtime.tools.currency._get_client", return_value=mock_client):
        result = await convert_currency(100, "USD", "EUR")

        assert "USD" in result
        assert "EUR" in result
        assert "92.5" in result


@pytest.mark.asyncio
async def test_convert_currency_api_error():
    resp_for_exc = MagicMock()
    resp_for_exc.status_code = 400

    mock_client = AsyncMock()
    mock_response = _make_response({"error": "bad"}, status_code=400)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad request", request=MagicMock(), response=resp_for_exc
    )
    mock_client.get.return_value = mock_response

    with patch("agent_runtime.tools.currency._get_client", return_value=mock_client):
        result = await convert_currency(100, "USD", "FAKE")

        assert "error" in result.lower() or "invalid" in result.lower()
