"""Tests for weather tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.tools.weather import _get_weather as get_weather


def _make_response(json_data: dict, status_code: int = 200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_get_weather_returns_forecast():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        side_effect=[
            _make_response(
                {
                    "results": [
                        {
                            "latitude": -6.2,
                            "longitude": 106.8,
                            "name": "Jakarta",
                            "country": "Indonesia",
                        }
                    ]
                }
            ),
            _make_response(
                {
                    "current": {
                        "temperature_2m": 30.5,
                        "relative_humidity_2m": 75,
                        "weather_code": 2,
                        "wind_speed_10m": 12.3,
                    },
                    "current_units": {
                        "temperature_2m": "°C",
                        "relative_humidity_2m": "%",
                        "wind_speed_10m": "km/h",
                    },
                }
            ),
        ]
    )

    with patch("agent_runtime.tools.weather._get_client", return_value=mock_client):
        result = await get_weather("Jakarta")

        assert "Jakarta" in result
        assert "30.5" in result


@pytest.mark.asyncio
async def test_get_weather_city_not_found():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response({"results": []})

    with patch("agent_runtime.tools.weather._get_client", return_value=mock_client):
        result = await get_weather("FakeCityXYZ123")

        assert "not found" in result.lower()
