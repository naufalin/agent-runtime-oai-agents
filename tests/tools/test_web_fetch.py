"""Tests for TinyFish web fetch tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.web_fetch import _web_fetch as web_fetch


def _make_fetch_result(url: str, title: str = "Page", text: str = "Content"):
    return {"url": url, "final_url": url, "title": title, "text": text, "language": "en"}


@pytest.mark.asyncio
async def test_fetch_single_url():
    mock_request = AsyncMock(
        return_value={
            "results": [_make_fetch_result("https://example.com", "Example", "Hello world")],
            "errors": [],
        }
    )

    with patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request):
        result = await web_fetch(["https://example.com"])

        assert "Example" in result
        assert "Hello world" in result
        mock_request.assert_called_once_with(
            "POST",
            "https://api.fetch.tinyfish.ai",
            json={"urls": ["https://example.com"], "format": "markdown", "ttl": 0},
        )


@pytest.mark.asyncio
async def test_fetch_multiple_urls():
    mock_request = AsyncMock(
        return_value={
            "results": [
                _make_fetch_result("https://a.com", "A", "Content A"),
                _make_fetch_result("https://b.com", "B", "Content B"),
                _make_fetch_result("https://c.com", "C", "Content C"),
            ],
            "errors": [],
        }
    )

    with patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request):
        result = await web_fetch(["https://a.com", "https://b.com", "https://c.com"])

        assert "Content A" in result
        assert "Content B" in result
        assert "Content C" in result
        assert "---" in result  # separator between pages


@pytest.mark.asyncio
async def test_fetch_partial_failure():
    mock_request = AsyncMock(
        return_value={
            "results": [_make_fetch_result("https://good.com", "Good", "OK")],
            "errors": [{"url": "https://bad.com", "error": "bot_blocked"}],
        }
    )

    with patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request):
        result = await web_fetch(["https://good.com", "https://bad.com"])

        assert "OK" in result
        assert "https://bad.com" in result
        assert "bot_blocked" in result


@pytest.mark.asyncio
async def test_fetch_all_fail():
    mock_request = AsyncMock(
        return_value={
            "results": [],
            "errors": [
                {"url": "https://a.com", "error": "timeout"},
                {"url": "https://b.com", "error": "page_not_found", "status": 404},
            ],
        }
    )

    with patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request):
        result = await web_fetch(["https://a.com", "https://b.com"])

        assert "Failed URLs" in result
        assert "timeout" in result
        assert "page_not_found" in result


@pytest.mark.asyncio
async def test_fetch_empty_text():
    mock_request = AsyncMock(
        return_value={
            "results": [_make_fetch_result("https://img.com", "Image Page", "")],
            "errors": [],
        }
    )

    with patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request):
        result = await web_fetch(["https://img.com"])

        assert "No extractable text" in result


@pytest.mark.asyncio
async def test_fetch_empty_urls():
    result = await web_fetch([])
    assert "No URLs" in result


@pytest.mark.asyncio
async def test_fetch_too_many_urls():
    urls = [f"https://example{i}.com" for i in range(11)]
    result = await web_fetch(urls)
    assert "Too many URLs" in result


@pytest.mark.asyncio
async def test_fetch_missing_api_key():
    mock_request = AsyncMock(side_effect=RuntimeError("TINYFISH_API_KEY is not set."))

    with (
        patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request),
        pytest.raises(RuntimeError, match="TINYFISH_API_KEY"),
    ):
        await web_fetch(["https://example.com"])


@pytest.mark.asyncio
async def test_fetch_custom_format():
    mock_request = AsyncMock(
        return_value={
            "results": [_make_fetch_result("https://example.com", "Page", "<p>HTML</p>")],
            "errors": [],
        }
    )

    with patch("agent_runtime.tools.web_fetch.tinyfish_request", mock_request):
        result = await web_fetch(["https://example.com"], format="html")

        assert "<p>HTML</p>" in result
        # Verify format was passed through
        call_kwargs = mock_request.call_args
        assert call_kwargs[1]["json"]["format"] == "html"
