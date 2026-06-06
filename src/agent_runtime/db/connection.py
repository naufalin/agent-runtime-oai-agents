"""Async database connection pool manager."""

import asyncpg


class Database:
    """Wraps an asyncpg connection pool."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def connect(self, min_size: int = 1, max_size: int = 5) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=min_size, max_size=max_size)

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def fetchrow(self, query: str, *args):
        assert self.pool is not None, "Database not connected — call connect() first"
        return await self.pool.fetchrow(query, *args)

    async def fetch(self, query: str, *args):
        assert self.pool is not None, "Database not connected — call connect() first"
        return await self.pool.fetch(query, *args)

    async def execute(self, query: str, *args):
        assert self.pool is not None, "Database not connected — call connect() first"
        return await self.pool.execute(query, *args)
