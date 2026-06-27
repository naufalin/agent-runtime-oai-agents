"""Integration smoke tests — verify the app boots and basic wiring works."""

import pytest
from httpx import ASGITransport, AsyncClient

from agent_runtime.api.app import app

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _set_auth_token(monkeypatch):
    """Configure API auth after the module-level settings object is imported."""
    from agent_runtime.config import Settings

    monkeypatch.setattr(
        "agent_runtime.api.auth.settings",
        Settings(agent_runtime_bearer_token="test-token"),
    )


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_openapi_schema_has_expected_paths():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        paths = set(schema["paths"].keys())

        assert "/" in paths
        assert "/sessions" in paths
        assert "/sessions/{encoded_id}" in paths
        assert "/sessions/{encoded_id}/chat" in paths
        assert "/sessions/{encoded_id}/chat/stream" in paths
        assert "/models" in paths
        assert "/prompts" in paths
        assert "/tools" in paths


@pytest.mark.asyncio
async def test_openapi_schema_info():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get("/openapi.json")
        schema = resp.json()
        assert schema["info"]["title"] == "Agent Runtime"
        assert schema["info"]["version"] == "0.1.0"
