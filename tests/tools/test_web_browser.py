"""Tests for TinyFish web browser tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.web_browser import _web_browser as web_browser

_SESSION_RESPONSE = {
    "session_id": "br-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "cdp_url": "wss://example.tinyfish.io/cdp",
    "base_url": "https://example.tinyfish.io",
}


@pytest.mark.asyncio
async def test_browser_session_created():
    mock_request = AsyncMock(return_value=_SESSION_RESPONSE)

    with patch("agent_runtime.tools.web_browser.tinyfish_request", mock_request):
        result = await web_browser("https://www.tinyfish.ai")

        assert "br-a1b2c3d4" in result
        assert "wss://example.tinyfish.io/cdp" in result
        assert "https://example.tinyfish.io" in result
        mock_request.assert_called_once_with(
            "POST",
            "https://api.browser.tinyfish.ai",
            json={"url": "https://www.tinyfish.ai"},
            timeout=60.0,
        )


@pytest.mark.asyncio
async def test_browser_auth_error():
    mock_request = AsyncMock(
        side_effect=RuntimeError("TinyFish auth failed (401). Check TINYFISH_API_KEY.")
    )

    with (
        patch("agent_runtime.tools.web_browser.tinyfish_request", mock_request),
        pytest.raises(RuntimeError, match="401"),
    ):
        await web_browser("https://example.com")


@pytest.mark.asyncio
async def test_browser_insufficient_credits():
    mock_request = AsyncMock(side_effect=RuntimeError("TinyFish insufficient credits (402)"))

    with (
        patch("agent_runtime.tools.web_browser.tinyfish_request", mock_request),
        pytest.raises(RuntimeError, match="402"),
    ):
        await web_browser("https://example.com")


@pytest.mark.asyncio
async def test_browser_no_session_id():
    mock_request = AsyncMock(return_value={"cdp_url": "wss://...", "base_url": "https://..."})

    with patch("agent_runtime.tools.web_browser.tinyfish_request", mock_request):
        result = await web_browser("https://example.com")

        assert "no session id" in result.lower()
