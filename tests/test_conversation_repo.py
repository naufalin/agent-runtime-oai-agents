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
    result = await repo.create_conversation("Test")

    assert result.id is not None
    assert result.id > 0
    assert result.title == "Test"
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_create_conversation_auto_id(db):
    repo = ConversationRepo(db)
    conv1 = await repo.create_conversation("First")
    conv2 = await repo.create_conversation("Second")

    assert conv2.id > conv1.id


@pytest.mark.asyncio
async def test_update_title(db):
    repo = ConversationRepo(db)
    conv = await repo.create_conversation("Old Title")
    await repo.update_title(conv.id, "New Title")

    updated = await repo.get_conversation(conv.id)
    assert updated is not None
    assert updated.title == "New Title"


@pytest.mark.asyncio
async def test_add_message(db):
    repo = ConversationRepo(db)
    conv = await repo.create_conversation("Test")
    result = await repo.add_message(conv.id, "user", "Hello!")

    assert result.conversation_id == conv.id
    assert result.role == "user"
    assert result.content == "Hello!"


@pytest.mark.asyncio
async def test_get_messages(db):
    repo = ConversationRepo(db)
    conv = await repo.create_conversation("Test")
    await repo.add_message(conv.id, "user", "Hello")
    await repo.add_message(conv.id, "assistant", "Hi!")

    messages = await repo.get_messages(conv.id)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_list_conversations(db):
    repo = ConversationRepo(db)
    await repo.create_conversation("First")
    await repo.create_conversation("Second")

    convos = await repo.list_conversations(limit=10)

    assert len(convos) == 2


@pytest.mark.asyncio
async def test_get_conversation_found(db):
    repo = ConversationRepo(db)
    conv = await repo.create_conversation("Test")

    result = await repo.get_conversation(conv.id)

    assert result is not None
    assert result.id == conv.id
    assert result.title == "Test"


@pytest.mark.asyncio
async def test_get_conversation_not_found(db):
    repo = ConversationRepo(db)

    result = await repo.get_conversation(99999)

    assert result is None
