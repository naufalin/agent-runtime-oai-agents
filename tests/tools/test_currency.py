"""Tests for currency conversion tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_runtime.tools.currency import _convert_currency as convert_currency


def _make_response(json_data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_convert_currency_success():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response({
        "base": "USD",
        "quote": "EUR",
        "rate": 0.925,
        "date": "2026-06-08",
    })

    with patch("agent_runtime.tools.currency._get_client", return_value=mock_client):
        result = await convert_currency(100, "USD", "EUR")

        assert "USD" in result
        assert "EUR" in result
        assert "92.50" in result  # 100 * 0.925
        assert "0.925" in result  # rate


@pytest.mark.asyncio
async def test_convert_currency_api_error():
    mock_response = _make_response({"message": "Not found"}, status_code=404)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found", request=MagicMock(), response=MagicMock(status_code=404)
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with patch("agent_runtime.tools.currency._get_client", return_value=mock_client):
        result = await convert_currency(100, "USD", "FAKE")

        assert "error" in result.lower() or "invalid" in result.lower()
