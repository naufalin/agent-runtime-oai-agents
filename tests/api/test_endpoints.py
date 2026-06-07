"""Tests for API endpoints."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from agent_runtime.api.app import app
from agent_runtime.api.deps import get_prompt_repo, get_session_repo
from agent_runtime.db.models import Message, Session, SystemPrompt
from agent_runtime.ids import encode


@pytest.fixture
def mock_session_repo():
    repo = AsyncMock()
    repo.create_session.return_value = Session(id=1, title="Test Session")
    repo.get_session.return_value = Session(id=1, title="Test Session")
    repo.list_sessions.return_value = [Session(id=1, title="Test Session")]
    repo.get_messages.return_value = [
        Message(id=1, session_id=1, role="system", content="You are helpful."),
        Message(id=2, session_id=1, role="user", content="Hello"),
        Message(id=3, session_id=1, role="assistant", content="Hi!"),
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
async def test_create_session(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/sessions", json={"title": "Test"})
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["title"] == "Test Session"  # mock returns this
        mock_session_repo.create_session.assert_called_once_with("Test")


@pytest.mark.asyncio
async def test_list_sessions(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert len(data["sessions"]) >= 1


@pytest.mark.asyncio
async def test_get_session_detail(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/sessions/{encode(1)}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Test Session"
        assert len(data["messages"]) == 3


@pytest.mark.asyncio
async def test_get_session_not_found(mock_session_repo, mock_prompt_repo):
    mock_session_repo.get_session.return_value = None
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/sessions/{encode(999)}")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_prompts(mock_prompt_repo):
    _setup_deps(mock_prompt_repo=mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/prompts")
        assert resp.status_code == 200
        assert "prompts" in resp.json()


@pytest.mark.asyncio
async def test_create_prompt(mock_prompt_repo):
    mock_prompt_repo.get_by_name.return_value = None
    _setup_deps(mock_prompt_repo=mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/prompts", json={"name": "pirate", "content": "Arr!"})
        assert resp.status_code == 201
        assert resp.json()["name"] == "pirate"


@pytest.mark.asyncio
async def test_create_prompt_duplicate(mock_prompt_repo):
    mock_prompt_repo.get_by_name.return_value = SystemPrompt(id=1, name="default", content="exists")
    _setup_deps(mock_prompt_repo=mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/prompts", json={"name": "default", "content": "dup"})
        assert resp.status_code == 409


@pytest.mark.asyncio
async def test_chat(mock_session_repo, mock_prompt_repo):
    mock_result = AsyncMock()
    mock_result.response = "Hi there!"
    mock_result.session_id = encode(1)

    _setup_deps(mock_session_repo, mock_prompt_repo)
    from unittest.mock import patch

    with patch("agent_runtime.api.routers.sessions.run_agent", return_value=mock_result):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/sessions/{encode(1)}/chat", json={"message": "Hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["response"] == "Hi there!"
            assert "messages" in data


@pytest.mark.asyncio
async def test_chat_session_not_found(mock_session_repo, mock_prompt_repo):
    mock_session_repo.get_session.return_value = None
    _setup_deps(mock_session_repo, mock_prompt_repo)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(f"/sessions/{encode(999)}/chat", json={"message": "Hello"})
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_switch_prompt(mock_session_repo, mock_prompt_repo):
    _setup_deps(mock_session_repo, mock_prompt_repo)
    from unittest.mock import patch

    with patch("agent_runtime.api.routers.prompts.switch_prompt", return_value="pirate"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/sessions/{encode(1)}/prompt", json={"name": "pirate"})
            assert resp.status_code == 200
            assert resp.json()["prompt"] == "pirate"
