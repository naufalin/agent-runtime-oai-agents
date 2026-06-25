"""Integration test fixtures — in-memory SQLite DB, mock repos for API tests."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Base


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db():
    """Fresh in-memory async SQLite database with all tables created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    database = Database("sqlite+aiosqlite:///:memory:")
    database.engine = engine
    yield database
    await engine.dispose()
