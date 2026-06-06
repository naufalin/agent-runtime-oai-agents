"""Generate test_db_connection.py"""
import pathlib

s = chr(42)  # asterisk
sep = s*3

L = []
L.append('"""Tests for database connection pool."""')
L.append('')
L.append('from unittest.mock import AsyncMock, patch')
L.append('')
L.append('import pytest')
L.append('')
L.append('from agent_runtime.db.connection import Database')
L.append('')
L.append('')
L.append('@pytest.mark.asyncio')
L.append('async def test_database_connect_creates_pool():')
L.append('    db = Database("postgresql://localhost/test")')
# Build the with line + body + decorator avoiding triple asterisk in source
with_line = '    with patch("agent_runtime.db.connection.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:'
L.append(with_line)
L.append('        mock_create.return_value = AsyncMock()')
L.append('        await db.connect()')
L.append('        mock_create.assert_called_once_with("postgresql://localhost/test", min_size=1, max_size=10)')
L.append('')
L.append('')
L.append('@pytest.mark.asyncio')
L.append('async def test_database_disconnect():')
L.append('    db = Database("postgresql://localhost/test")')
L.append('    mock_pool = AsyncMock()')
L.append('    db.pool = mock_pool')
L.append('    await db.disconnect()')
L.append('    mock_pool.close.assert_called_once()')
L.append('')
L.append('')
L.append('@pytest.mark.asyncio')
L.append('async def test_database_fetch():')
L.append('    db = Database("postgresql://localhost/test")')
L.append('    mock_pool = AsyncMock()')
L.append('    mock_pool.fetch.return_value = [{"id": 1}]')
L.append('    db.pool = mock_pool')
L.append('')
L.append('    result = await db.fetch("SELECT * FROM test")')
L.append('')
L.append('    assert result == [{"id": 1}]')
L.append('    mock_pool.fetch.assert_called_once_with("SELECT * FROM test")')
L.append('')

p = pathlib.Path('tests/test_db_connection.py')
p.write_text('\n'.join(L))
print(f'Wrote {len(L)} lines to {p}')
