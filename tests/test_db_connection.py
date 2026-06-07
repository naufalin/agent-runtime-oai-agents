"""Tests for database connection (SQLAlchemy async engine)."""

from unittest.mock import patch

import pytest

from agent_runtime.db.connection import Database


def test_database_connect_creates_engine():
    db = Database("postgresql+asyncpg://localhost/test")
    with patch("agent_runtime.db.connection.create_async_engine") as mock_create:
        db.connect()
        mock_create.assert_called_once_with(
            "postgresql+asyncpg://localhost/test", pool_size=5, max_overflow=2
        )


@pytest.mark.asyncio
async def test_database_disconnect():
    db = Database("postgresql+asyncpg://localhost/test")
    db.connect()

    # Replace the real engine with a mock for dispose() tracking
    class FakeEngine:
        disposed = False

        async def dispose(self):
            self.disposed = True

    fake = FakeEngine()
    db.engine = fake
    await db.disconnect()
    assert fake.disposed is True
    assert db.engine is None


@pytest.mark.asyncio
async def test_database_session_raises_if_not_connected():
    db = Database("postgresql+asyncpg://localhost/test")
    with pytest.raises(AssertionError, match="not connected"):
        async with db.session():
            pass
