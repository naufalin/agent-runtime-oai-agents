"""Tests for TinyFish shared client, error handling, and retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.tools._tinyfish_common import (
    _ensure_api_key,
    _RetryableError,
    get_client,
    handle_response,
    tinyfish_request,
)


def _mock_response(status_code: int, json_data: dict | None = None, text: str = ""):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


class TestEnsureApiKey:
    def test_raises_when_missing(self):
        with (
            patch.dict("os.environ", {"TINYFISH_API_KEY": ""}, clear=False),
            pytest.raises(RuntimeError, match="TINYFISH_API_KEY"),
        ):
            _ensure_api_key()

    def test_returns_key_when_set(self):
        with patch.dict("os.environ", {"TINYFISH_API_KEY": "sk-test-123"}, clear=False):
            assert _ensure_api_key() == "sk-test-123"


class TestGetClient:
    def test_raises_when_no_key(self):
        # Reset the global client
        import agent_runtime.tools._tinyfish_common as mod

        mod._client = None
        with (
            patch.dict("os.environ", {"TINYFISH_API_KEY": ""}, clear=False),
            pytest.raises(RuntimeError, match="TINYFISH_API_KEY"),
        ):
            get_client()

    def test_creates_client_with_key(self):
        import agent_runtime.tools._tinyfish_common as mod

        mod._client = None
        with patch.dict("os.environ", {"TINYFISH_API_KEY": "sk-test"}, clear=False):
            client = get_client()
            assert client is not None
            # Clean up
            mod._client = None


class TestHandleResponse:
    @pytest.mark.asyncio
    async def test_success(self):
        resp = _mock_response(200, {"results": [1, 2, 3]})
        result = await handle_response(resp)
        assert result == {"results": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_400_raises_value_error(self):
        resp = _mock_response(400, text="missing query param")
        with pytest.raises(ValueError, match="400"):
            await handle_response(resp)

    @pytest.mark.asyncio
    async def test_401_raises_runtime_error(self):
        resp = _mock_response(401)
        with pytest.raises(RuntimeError, match="401"):
            await handle_response(resp)

    @pytest.mark.asyncio
    async def test_402_raises_runtime_error(self):
        resp = _mock_response(402)
        with pytest.raises(RuntimeError, match="402"):
            await handle_response(resp)

    @pytest.mark.asyncio
    async def test_429_raises_retryable(self):
        resp = _mock_response(429)
        with pytest.raises(_RetryableError, match="429"):
            await handle_response(resp)

    @pytest.mark.asyncio
    async def test_503_raises_retryable(self):
        resp = _mock_response(503)
        with pytest.raises(_RetryableError, match="503"):
            await handle_response(resp)


class TestTinyfishRequest:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        import agent_runtime.tools._tinyfish_common as mod

        mock_client = AsyncMock()
        mock_client.request.return_value = _mock_response(200, {"ok": True})
        mod._client = mock_client

        result = await tinyfish_request("GET", "https://api.test.com")

        assert result == {"ok": True}
        assert mock_client.request.call_count == 1
        mod._client = None

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        import agent_runtime.tools._tinyfish_common as mod

        mock_client = AsyncMock()
        mock_client.request.side_effect = [
            _mock_response(429),
            _mock_response(200, {"results": []}),
        ]
        mod._client = mock_client

        with patch("agent_runtime.tools._tinyfish_common.asyncio.sleep", new_callable=AsyncMock):
            result = await tinyfish_request("GET", "https://api.test.com")

        assert result == {"results": []}
        assert mock_client.request.call_count == 2
        mod._client = None

    @pytest.mark.asyncio
    async def test_retry_on_503(self):
        import agent_runtime.tools._tinyfish_common as mod

        mock_client = AsyncMock()
        mock_client.request.side_effect = [
            _mock_response(503),
            _mock_response(200, {"ok": True}),
        ]
        mod._client = mock_client

        with patch("agent_runtime.tools._tinyfish_common.asyncio.sleep", new_callable=AsyncMock):
            result = await tinyfish_request("POST", "https://api.test.com")

        assert result == {"ok": True}
        assert mock_client.request.call_count == 2
        mod._client = None

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        import agent_runtime.tools._tinyfish_common as mod

        mock_client = AsyncMock()
        mock_client.request.return_value = _mock_response(400, text="bad query")
        mod._client = mock_client

        with pytest.raises(ValueError, match="400"):
            await tinyfish_request("GET", "https://api.test.com")

        assert mock_client.request.call_count == 1
        mod._client = None

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self):
        import agent_runtime.tools._tinyfish_common as mod

        mock_client = AsyncMock()
        mock_client.request.return_value = _mock_response(401)
        mod._client = mock_client

        with pytest.raises(RuntimeError, match="401"):
            await tinyfish_request("GET", "https://api.test.com")

        assert mock_client.request.call_count == 1
        mod._client = None

    @pytest.mark.asyncio
    async def test_exhausted_retry_raises(self):
        """If both attempts return 429, the second should raise RuntimeError."""
        import agent_runtime.tools._tinyfish_common as mod

        mock_client = AsyncMock()
        mock_client.request.return_value = _mock_response(429)
        mod._client = mock_client

        with (
            patch("agent_runtime.tools._tinyfish_common.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError),
        ):
            await tinyfish_request("GET", "https://api.test.com")

        assert mock_client.request.call_count == 2
        mod._client = None
