"""Tests for system prompt repository using real async SQLite."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Base
from agent_runtime.db.prompt_repo import SystemPromptRepo


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    database = Database("sqlite+aiosqlite:///:memory:")
    database.engine = engine
    yield database
    await engine.dispose()


@pytest.mark.asyncio
async def test_seed_default_creates_prompt(db):
    repo = SystemPromptRepo(db)
    prompt = await repo.seed_default()

    assert prompt.name == "default"
    assert "helpful assistant" in prompt.content
    assert prompt.id is not None


@pytest.mark.asyncio
async def test_seed_default_idempotent(db):
    repo = SystemPromptRepo(db)
    p1 = await repo.seed_default()
    p2 = await repo.seed_default()

    assert p1.id == p2.id


@pytest.mark.asyncio
async def test_create_custom_prompt(db):
    repo = SystemPromptRepo(db)
    prompt = await repo.create("pirate", "You are a pirate. Arr!")

    assert prompt.name == "pirate"
    assert prompt.content == "You are a pirate. Arr!"


@pytest.mark.asyncio
async def test_get_by_name(db):
    repo = SystemPromptRepo(db)
    await repo.create("test", "test content")

    result = await repo.get_by_name("test")

    assert result is not None
    assert result.content == "test content"


@pytest.mark.asyncio
async def test_get_by_name_not_found(db):
    repo = SystemPromptRepo(db)

    result = await repo.get_by_name("nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_list_all(db):
    repo = SystemPromptRepo(db)
    await repo.create("alpha", "first")
    await repo.create("beta", "second")

    prompts = await repo.list_all()

    assert len(prompts) == 2
    assert prompts[0].name == "alpha"  # ordered by name


@pytest.mark.asyncio
async def test_update_prompt(db):
    repo = SystemPromptRepo(db)
    prompt = await repo.create("test", "old content")
    await repo.update(prompt.id, "new content")

    updated = await repo.get_by_id(prompt.id)
    assert updated is not None
    assert updated.content == "new content"
