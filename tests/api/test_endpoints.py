"""Tests for API endpoints."""

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from agent_runtime.api.app import app
from agent_runtime.api.deps import get_prompt_repo, get_session_repo
from agent_runtime.db.models import Message, Session, SystemPrompt
from agent_runtime.ids import encode

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture
def mock_session_repo():
    repo = AsyncMock()
    repo.create_session.return_value = Session(id=1, title="Test Session")
    repo.get_session.return_value = Session(id=1, title="Test Session")
    repo.list_sessions.return_value = [Session(id=1, title="Test Session")]
    repo.get_messages.return_value = [
        Message(id=1, session_id=1, role="system", content="You are helpful."),
        Message(id=2, session_id=1, role="user", content="Hello"),
        Message(
            id=3,
            session_id=1,
            role="assistant",
            content="Hi!",
            provider="openrouter",
            model="z-ai/glm-5.2",
            usage_json={"total_tokens": 12, "reasoning_tokens": 4},
            thinking_json={"reasoning": "provider reasoning"},
        ),
    ]
    repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="You are helpful.",
        system_prompt_id=1,
    )
    repo.add_message.return_value = Message(id=4, session_id=1, role="user", content="Hello")
    return repo


@pytest.fixture
def mock_prompt_repo():
    repo = AsyncMock()
    repo.seed_default.return_value = SystemPrompt(id=1, name="default", content="Helpful.")
    repo.list_all.return_value = [SystemPrompt(id=1, name="default", content="Helpful.")]
    repo.get_by_id.return_value = SystemPrompt(id=1, name="default", content="Helpful.")
    repo.get_by_name.return_value = SystemPrompt(id=1, name="default", content="Helpful.")
    repo.create.return_value = SystemPrompt(id=2, name="pirate", content="Arr!")
    return repo


@pytest.fixture(autouse=True)
def _set_auth_token(monkeypatch):
    """Configure API auth for endpoint tests."""
    monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def _reset_overrides():
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


def _setup_deps(mock_session_repo=None, mock_prompt_repo=None):
    if mock_session_repo:
        app.dependency_overrides[get_session_repo] = lambda: mock_session_repo
    if mock_prompt_repo:
        app.dependency_overrides[get_prompt_repo] = lambda: mock_prompt_repo


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_private_endpoint_with_valid_token(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/sessions", headers=AUTH_HEADERS)
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_private_endpoint_without_token_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/sessions")
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_private_endpoint_with_wrong_token_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/sessions", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_private_endpoint_with_malformed_header_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/sessions", headers={"Authorization": "Token test-token"})
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_private_endpoint_with_blank_configured_token_returns_503(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/sessions", headers=AUTH_HEADERS)
        assert resp.status_code == 503


@pytest.mark.asyncio
async def test_docs_require_bearer_token():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        docs_resp = await client.get("/docs")
        redoc_resp = await client.get("/redoc")
        schema_resp = await client.get("/openapi.json")

        assert docs_resp.status_code == 401
        assert redoc_resp.status_code == 401
        assert schema_resp.status_code == 401


@pytest.mark.asyncio
async def test_docs_and_openapi_with_valid_token():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        docs_resp = await client.get("/docs", headers=AUTH_HEADERS)
        redoc_resp = await client.get("/redoc", headers=AUTH_HEADERS)
        schema_resp = await client.get("/openapi.json", headers=AUTH_HEADERS)

        assert docs_resp.status_code == 200
        assert "SwaggerUIBundle" in docs_resp.text
        assert redoc_resp.status_code == 200
        assert "Redoc.init" in redoc_resp.text
        assert schema_resp.status_code == 200
        assert schema_resp.json()["info"]["title"] == "Agent Runtime"


@pytest.mark.asyncio
async def test_list_models_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/models")
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_models_with_valid_token():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get("/models")
        assert resp.status_code == 200
        model_ids = {item["id"] for item in resp.json()["openrouter"]["models"]}
        assert "z-ai/glm-5.2" in model_ids
        assert "deepseek/deepseek-v4-flash" in model_ids


@pytest.mark.asyncio
async def test_create_session(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.post("/sessions", json={"title": "Test"})
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["title"] == "Test Session"  # mock returns this
        mock_session_repo.create_session.assert_called_once_with("Test")


@pytest.mark.asyncio
async def test_list_sessions(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert len(data["sessions"]) >= 1


@pytest.mark.asyncio
async def test_get_session_detail(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get(f"/sessions/{encode(1)}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Test Session"
        assert len(data["messages"]) == 3
        assert data["messages"][2]["provider"] == "openrouter"
        assert data["messages"][2]["model"] == "z-ai/glm-5.2"
        assert data["messages"][2]["usage"]["reasoning_tokens"] == 4
        assert data["messages"][2]["thinking"]["reasoning"] == "provider reasoning"


@pytest.mark.asyncio
async def test_get_session_not_found(mock_session_repo, mock_prompt_repo):
    mock_session_repo.get_session.return_value = None
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get(f"/sessions/{encode(999)}")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_prompts(mock_prompt_repo):
    _setup_deps(mock_prompt_repo=mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.get("/prompts")
        assert resp.status_code == 200
        assert "prompts" in resp.json()


@pytest.mark.asyncio
async def test_create_prompt(mock_prompt_repo):
    mock_prompt_repo.get_by_name.return_value = None
    _setup_deps(mock_prompt_repo=mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.post("/prompts", json={"name": "pirate", "content": "Arr!"})
        assert resp.status_code == 201
        assert resp.json()["name"] == "pirate"


@pytest.mark.asyncio
async def test_create_prompt_duplicate(mock_prompt_repo):
    mock_prompt_repo.get_by_name.return_value = SystemPrompt(id=1, name="default", content="exists")
    _setup_deps(mock_prompt_repo=mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.post("/prompts", json={"name": "default", "content": "dup"})
        assert resp.status_code == 409


@pytest.mark.asyncio
async def test_chat(mock_session_repo, mock_prompt_repo):
    mock_result = AsyncMock()
    mock_result.response = "Hi there!"
    mock_result.session_id = encode(1)
    mock_result.provider = "openrouter"
    mock_result.model = "qwen/qwen3.7-max"
    mock_result.usage = {"total_tokens": 9, "reasoning_tokens": 2}
    mock_result.thinking = {"reasoning": "visible provider reasoning"}

    _setup_deps(mock_session_repo, mock_prompt_repo)
    from unittest.mock import patch

    with patch("agent_runtime.api.routers.sessions.run_agent", return_value=mock_result):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
        ) as client:
            resp = await client.post(
                f"/sessions/{encode(1)}/chat",
                json={
                    "message": "Hello",
                    "provider": "openrouter",
                    "model": "qwen/qwen3.7-max",
                    "reasoning_effort": "high",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["response"] == "Hi there!"
            assert data["provider"] == "openrouter"
            assert data["model"] == "qwen/qwen3.7-max"
            assert data["usage"]["reasoning_tokens"] == 2
            assert data["thinking"]["reasoning"] == "visible provider reasoning"
            assert "messages" in data


@pytest.mark.asyncio
async def test_chat_passes_model_options(mock_session_repo, mock_prompt_repo):
    mock_result = AsyncMock()
    mock_result.response = "Hi there!"
    mock_result.session_id = encode(1)
    mock_result.provider = "openrouter"
    mock_result.model = "deepseek/deepseek-v4-pro"
    mock_result.usage = None
    mock_result.thinking = None

    _setup_deps(mock_session_repo, mock_prompt_repo)
    from unittest.mock import patch

    with patch(
        "agent_runtime.api.routers.sessions.run_agent",
        return_value=mock_result,
    ) as run_mock:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
        ) as client:
            resp = await client.post(
                f"/sessions/{encode(1)}/chat",
                json={
                    "message": "Hello",
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-pro",
                    "reasoning_effort": "xhigh",
                },
            )
            assert resp.status_code == 200
            run_mock.assert_awaited_once_with(
                "Hello",
                session_id=encode(1),
                provider="openrouter",
                model="deepseek/deepseek-v4-pro",
                reasoning_effort="xhigh",
            )


@pytest.mark.asyncio
async def test_chat_session_not_found(mock_session_repo, mock_prompt_repo):
    mock_session_repo.get_session.return_value = None
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.post(f"/sessions/{encode(999)}/chat", json={"message": "Hello"})
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_switch_prompt(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    from unittest.mock import patch

    with patch("agent_runtime.api.routers.prompts.switch_prompt", return_value="pirate"):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
        ) as client:
            resp = await client.post(f"/sessions/{encode(1)}/prompt", json={"name": "pirate"})
            assert resp.status_code == 200
            assert resp.json()["prompt"] == "pirate"


@pytest.mark.asyncio
async def test_chat_stream(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)

    async def fake_streamed(
        message,
        session_id=None,
        provider=None,
        model=None,
        reasoning_effort=None,
    ):
        yield {"type": "text_delta", "delta": "Hello"}
        yield {"type": "text_delta", "delta": " world"}
        yield {
            "type": "done",
            "session_id": session_id or encode(1),
            "provider": provider,
            "model": model,
            "usage": {"total_tokens": 11, "reasoning_tokens": 3},
            "thinking": {"reasoning": "stream reasoning"},
        }

    from unittest.mock import patch

    with patch("agent_runtime.api.routers.sessions.run_agent_streamed", side_effect=fake_streamed):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
        ) as client:
            async with client.stream(
                "POST",
                f"/sessions/{encode(1)}/chat/stream",
                json={
                    "message": "Hello",
                    "provider": "openrouter",
                    "model": "minimax/minimax-m3",
                },
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                lines = []
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        lines.append(json.loads(line[6:]))
                assert len(lines) == 3
                assert lines[0] == {"type": "text_delta", "delta": "Hello"}
                assert lines[1] == {"type": "text_delta", "delta": " world"}
                assert lines[2]["type"] == "done"
                assert lines[2]["provider"] == "openrouter"
                assert lines[2]["model"] == "minimax/minimax-m3"
                assert lines[2]["usage"]["reasoning_tokens"] == 3
                assert lines[2]["thinking"]["reasoning"] == "stream reasoning"


@pytest.mark.asyncio
async def test_chat_stream_session_not_found(mock_session_repo, mock_prompt_repo):
    mock_session_repo.get_session.return_value = None
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        resp = await client.post(
            f"/sessions/{encode(999)}/chat/stream",
            json={"message": "Hello"},
        )
        assert resp.status_code == 404
