"""Tests for TinyFish web search tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.tools.web_search import _web_search as web_search


def _make_response(json_data: dict, status_code: int = 200):
    """Create a mock httpx response (json/raise_for_status are sync in httpx)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_web_search_returns_results():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response(
        {
            "query": "python tutorial",
            "results": [
                {
                    "position": 1,
                    "title": "Python.org",
                    "snippet": "Official Python docs",
                    "url": "https://python.org",
                },
                {
                    "position": 2,
                    "title": "Learn Python",
                    "snippet": "Free course",
                    "url": "https://learnpython.org",
                },
            ],
            "total_results": 2,
        }
    )

    with patch("agent_runtime.tools.web_search._get_client", return_value=mock_client):
        result = await web_search("python tutorial")

        assert "Python.org" in result
        assert "https://python.org" in result


@pytest.mark.asyncio
async def test_web_search_handles_empty_results():
    mock_client = AsyncMock()
    mock_client.get.return_value = _make_response(
        {
            "query": "nonsense query xyz",
            "results": [],
            "total_results": 0,
        }
    )

    with patch("agent_runtime.tools.web_search._get_client", return_value=mock_client):
        result = await web_search("nonsense query xyz")

        assert "No results found" in result
