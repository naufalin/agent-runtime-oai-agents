"""Tests for TinyFish web agent tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.web_agent import _web_agent as web_agent


@pytest.mark.asyncio
async def test_agent_success():
    mock_request = AsyncMock(
        return_value={
            "run_id": "run-abc123",
            "status": "COMPLETED",
            "result": {"products": [{"name": "Widget", "price": "$9.99"}]},
            "num_of_steps": 3,
        }
    )

    with patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request):
        result = await web_agent("https://shop.example.com", "Extract product names and prices")

        assert "Widget" in result
        assert "$9.99" in result
        mock_request.assert_called_once_with(
            "POST",
            "https://agent.tinyfish.ai/v1/automation/run",
            json={
                "url": "https://shop.example.com",
                "goal": "Extract product names and prices",
                "browser_profile": "lite",
            },
            timeout=120.0,
        )


@pytest.mark.asyncio
async def test_agent_success_string_result():
    mock_request = AsyncMock(
        return_value={
            "run_id": "run-abc123",
            "status": "COMPLETED",
            "result": "The page title is Example Domain",
        }
    )

    with patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request):
        result = await web_agent("https://example.com", "Get the page title")

        assert "Example Domain" in result


@pytest.mark.asyncio
async def test_agent_failed_status():
    mock_request = AsyncMock(
        return_value={
            "run_id": "run-fail",
            "status": "FAILED",
            "error": {"code": "TASK_FAILED", "message": "Could not find pricing element"},
        }
    )

    with patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request):
        result = await web_agent("https://example.com", "Get pricing")

        assert "failed" in result.lower()
        assert "run-fail" in result
        assert "TASK_FAILED" in result


@pytest.mark.asyncio
async def test_agent_cancelled():
    mock_request = AsyncMock(return_value={"run_id": "run-cancel", "status": "CANCELLED"})

    with patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request):
        result = await web_agent("https://example.com", "Do something")

        assert "cancelled" in result.lower()


@pytest.mark.asyncio
async def test_agent_blocked_detection():
    mock_request = AsyncMock(
        return_value={
            "run_id": "run-blocked",
            "status": "COMPLETED",
            "result": {"error": "captcha detected on page"},
        }
    )

    with patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request):
        result = await web_agent("https://protected-site.com", "Extract data")

        assert "blocked" in result.lower() or "captcha" in result.lower()


@pytest.mark.asyncio
async def test_agent_no_result():
    mock_request = AsyncMock(
        return_value={"run_id": "run-empty", "status": "COMPLETED", "result": None}
    )

    with patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request):
        result = await web_agent("https://example.com", "Do something")

        assert "no result" in result.lower()


@pytest.mark.asyncio
async def test_agent_insufficient_credits():
    mock_request = AsyncMock(side_effect=RuntimeError("TinyFish insufficient credits (402)"))

    with (
        patch("agent_runtime.tools.web_agent.tinyfish_request", mock_request),
        pytest.raises(RuntimeError, match="402"),
    ):
        await web_agent("https://example.com", "Test")
