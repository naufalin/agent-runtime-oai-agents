"""Tests for country info tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.tools.country import _get_country_info as get_country_info


def _make_response(json_data, status_code: int = 200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_get_country_info_success():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response(
        [
            {
                "name": {"common": "Indonesia", "official": "Republic of Indonesia"},
                "capital": ["Jakarta"],
                "region": "Asia",
                "subregion": "South-Eastern Asia",
                "population": 273523621,
                "currencies": {"IDR": {"name": "Indonesian rupiah", "symbol": "Rp"}},
                "languages": {"ind": "Indonesian"},
                "flag": "🇮🇩",
            }
        ]
    )

    with patch("agent_runtime.tools.country._get_client", return_value=mock_client):
        result = await get_country_info("Indonesia")

        assert "Indonesia" in result
        assert "Jakarta" in result
        assert "273" in result  # population


@pytest.mark.asyncio
async def test_get_country_info_not_found():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response([])

    with patch("agent_runtime.tools.country._get_client", return_value=mock_client):
        result = await get_country_info("FakeCountryXYZ")

        assert "not found" in result.lower()
