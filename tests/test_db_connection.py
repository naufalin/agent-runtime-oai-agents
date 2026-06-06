"""Tests for database connection pool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.db.connection import Database


@pytest.mark.asyncio
async def test_database_connect_creates_pool():
    db = Database("postgresql://localhost/test")
    with patch(
        "agent_runtime.db.connection.asyncpg.create_pool", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = AsyncMock()
        await db.connect()
        mock_create.assert_called_once_with("postgresql://localhost/test", min_size=1, max_size=5)


@pytest.mark.asyncio
async def test_database_disconnect():
    db = Database("postgresql://localhost/test")
    mock_pool = AsyncMock()
    db.pool = mock_pool
    await db.disconnect()
    mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_database_fetchrow():
    db = Database("postgresql://localhost/test")
    mock_pool = AsyncMock()
    mock_pool.fetchrow.return_value = {"id": 1, "name": "test"}
    db.pool = mock_pool

    result = await db.fetchrow("SELECT * FROM t WHERE id=$1", 1)

    assert result == {"id": 1, "name": "test"}
    mock_pool.fetchrow.assert_called_once_with("SELECT * FROM t WHERE id=$1", 1)


@pytest.mark.asyncio
async def test_database_fetch():
    db = Database("postgresql://localhost/test")
    mock_pool = AsyncMock()
    mock_pool.fetch.return_value = [{"id": 1}, {"id": 2}]
    db.pool = mock_pool

    result = await db.fetch("SELECT * FROM t")

    assert len(result) == 2
    mock_pool.fetch.assert_called_once_with("SELECT * FROM t")


@pytest.mark.asyncio
async def test_database_execute():
    db = Database("postgresql://localhost/test")
    mock_pool = AsyncMock()
    mock_pool.execute.return_value = "INSERT 0 1"
    db.pool = mock_pool

    result = await db.execute("INSERT INTO t (name) VALUES ($1)", "test")

    assert result == "INSERT 0 1"
    mock_pool.execute.assert_called_once_with("INSERT INTO t (name) VALUES ($1)", "test")
