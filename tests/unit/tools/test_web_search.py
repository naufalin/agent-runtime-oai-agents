"""Tests for TinyFish web search tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.web_search import _web_search as web_search

_SEARCH_RESPONSE = {
    "query": "python tutorial",
    "results": [
        {
            "position": 1,
            "site_name": "python.org",
            "title": "Python.org",
            "snippet": "Official Python docs",
            "url": "https://python.org",
        },
        {
            "position": 2,
            "site_name": "learnpython.org",
            "title": "Learn Python",
            "snippet": "Free course",
            "url": "https://learnpython.org",
        },
    ],
    "total_results": 2,
    "page": 0,
}


@pytest.mark.asyncio
async def test_web_search_returns_results():
    mock_request = AsyncMock(return_value=_SEARCH_RESPONSE)

    with patch("agent_runtime.tools.web_search.tinyfish_request", mock_request):
        result = await web_search("python tutorial")

        assert "Python.org" in result
        assert "https://python.org" in result
        assert "python.org" in result  # site_name tag
        mock_request.assert_called_once_with(
            "GET",
            "https://api.search.tinyfish.ai",
            params={"query": "python tutorial", "location": "US", "language": "en", "page": 0},
        )


@pytest.mark.asyncio
async def test_web_search_handles_empty_results():
    mock_request = AsyncMock(return_value={"results": [], "total_results": 0})

    with patch("agent_runtime.tools.web_search.tinyfish_request", mock_request):
        result = await web_search("nonsense query xyz")

        assert "No results found" in result


@pytest.mark.asyncio
async def test_web_search_passes_geo_params():
    mock_request = AsyncMock(return_value={"results": [], "total_results": 0})

    with patch("agent_runtime.tools.web_search.tinyfish_request", mock_request):
        await web_search("restaurants", location="ID", language="id", page=2)

        mock_request.assert_called_once_with(
            "GET",
            "https://api.search.tinyfish.ai",
            params={"query": "restaurants", "location": "ID", "language": "id", "page": 2},
        )


@pytest.mark.asyncio
async def test_web_search_missing_api_key():
    mock_request = AsyncMock(side_effect=RuntimeError("TINYFISH_API_KEY is not set."))

    with (
        patch("agent_runtime.tools.web_search.tinyfish_request", mock_request),
        pytest.raises(RuntimeError, match="TINYFISH_API_KEY"),
    ):
        await web_search("test")


@pytest.mark.asyncio
async def test_web_search_rate_limit():
    """Tool should propagate RuntimeError after shared module exhausts retries."""
    mock_request = AsyncMock(side_effect=RuntimeError("TinyFish rate limited (429)"))

    with (
        patch("agent_runtime.tools.web_search.tinyfish_request", mock_request),
        pytest.raises(RuntimeError, match="429"),
    ):
        await web_search("test")


@pytest.mark.asyncio
async def test_web_search_safe_field_access():
    """Results missing optional fields should not crash."""
    sparse_response = {
        "results": [{"url": "https://example.com"}],  # no title, snippet, site_name
        "total_results": 1,
    }
    mock_request = AsyncMock(return_value=sparse_response)

    with patch("agent_runtime.tools.web_search.tinyfish_request", mock_request):
        result = await web_search("test")

        assert "Untitled" in result
        assert "https://example.com" in result
