"""Tests for conversation repository using real async SQLite."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from agent_runtime.db.connection import Database
from agent_runtime.db.conversation_repo import ConversationRepo
from agent_runtime.db.models import Base


@pytest.fixture
async def db():
    """Create a Database backed by in-memory SQLite for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    database = Database("sqlite+aiosqlite:///:memory:")
    database.engine = engine
    yield database
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_conversation(db):
    repo = ConversationRepo(db)
    result = await repo.create_conversation("abc-123", "Test")

    assert result.id == "abc-123"
    assert result.title == "Test"
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_add_message(db):
    repo = ConversationRepo(db)
    await repo.create_conversation("abc-123", "Test")
    result = await repo.add_message("abc-123", "user", "Hello!")

    assert result.conversation_id == "abc-123"
    assert result.role == "user"
    assert result.content == "Hello!"


@pytest.mark.asyncio
async def test_get_messages(db):
    repo = ConversationRepo(db)
    await repo.create_conversation("abc-123", "Test")
    await repo.add_message("abc-123", "user", "Hello")
    await repo.add_message("abc-123", "assistant", "Hi!")

    messages = await repo.get_messages("abc-123")

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_list_conversations(db):
    repo = ConversationRepo(db)
    await repo.create_conversation("abc", "First")
    await repo.create_conversation("def", "Second")

    convos = await repo.list_conversations(limit=10)

    assert len(convos) == 2


@pytest.mark.asyncio
async def test_get_conversation_found(db):
    repo = ConversationRepo(db)
    await repo.create_conversation("abc", "Test")

    result = await repo.get_conversation("abc")

    assert result is not None
    assert result.id == "abc"
    assert result.title == "Test"


@pytest.mark.asyncio
async def test_get_conversation_not_found(db):
    repo = ConversationRepo(db)

    result = await repo.get_conversation("nonexistent")

    assert result is None
