"""Tests for conversation repository."""

from unittest.mock import AsyncMock

import pytest

from agent_runtime.db.conversation_repo import ConversationRepo


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.mark.asyncio
async def test_create_conversation(mock_db):
    mock_db.fetchrow.return_value = {"id": "abc-123", "title": "Test", "created_at": "2026-01-01"}
    repo = ConversationRepo(mock_db)

    result = await repo.create_conversation("abc-123", "Test")

    assert result["id"] == "abc-123"
    mock_db.fetchrow.assert_called_once()


@pytest.mark.asyncio
async def test_add_message(mock_db):
    mock_db.fetchrow.return_value = {
        "id": 1,
        "role": "user",
        "content": "Hello!",
        "created_at": "2026-01-01",
    }
    repo = ConversationRepo(mock_db)

    result = await repo.add_message("abc-123", "user", "Hello!")

    assert result["id"] == 1
    mock_db.fetchrow.assert_called_once()


@pytest.mark.asyncio
async def test_get_messages(mock_db):
    mock_db.fetch.return_value = [
        {"id": 1, "role": "user", "content": "Hello", "created_at": "2026-01-01"},
        {"id": 2, "role": "assistant", "content": "Hi!", "created_at": "2026-01-01"},
    ]
    repo = ConversationRepo(mock_db)

    messages = await repo.get_messages("abc-123")

    assert len(messages) == 2
    mock_db.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_list_conversations(mock_db):
    mock_db.fetch.return_value = [
        {"id": "abc", "title": "First", "updated_at": "2026-01-01"},
        {"id": "def", "title": "Second", "updated_at": "2026-01-02"},
    ]
    repo = ConversationRepo(mock_db)

    convos = await repo.list_conversations(limit=10)

    assert len(convos) == 2


@pytest.mark.asyncio
async def test_get_conversation(mock_db):
    mock_db.fetchrow.return_value = {
        "id": "abc",
        "title": "Test",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
    }
    repo = ConversationRepo(mock_db)

    result = await repo.get_conversation("abc")

    assert result["id"] == "abc"
