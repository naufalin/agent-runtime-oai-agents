# Agent Runtime Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build an agent runtime powered by the OpenAI Agents SDK with PostgreSQL persistence, TinyFish web search, and several free-to-use API tools.

**Architecture:** A Python package (`agent_runtime`) with async-first design. The OpenAI Agents SDK handles LLM orchestration — we wire up tools (TinyFish search, weather, currency, country info) as `@function_tool` decorated functions. PostgreSQL stores conversation history and agent run metadata via asyncpg. Alembic manages schema migrations.

**Tech Stack:**
- OpenAI Agents SDK (`openai-agents`) — agent runtime, tools, handoffs
- PostgreSQL + asyncpg — async DB access
- Alembic — schema migrations
- httpx — async HTTP client for external APIs
- Pydantic — data models, settings
- Ruff — linting + formatting
- pytest + pytest-asyncio — testing

**External APIs (all free, no paid keys):**

| Tool            | API                        | Auth       | Notes                                      |
|-----------------|----------------------------|------------|---------------------------------------------|
| Web Search      | TinyFish Search API        | API key    | Free tier: 30 req/min, no credits consumed  |
| Weather         | Open-Meteo                 | None       | No API key needed, geocoding + forecast     |
| Currency Rates  | Frankfurter (ECB data)     | None       | No key, EUR-based, covers 30+ currencies    |
| Country Info    | REST Countries             | None       | No key, country metadata (capital, pop, etc) |

---

## Phase 0: Project Setup

### Task 0.1: Add dependencies and tooling config

**Objective:** Install all dependencies and configure ruff + pytest.

**Step 1: Add dependencies**

```bash
cd /Users/wahyu/Dev/personal/agent_runtime_oai_agents
uv add openai-agents asyncpg alembic httpx pydantic pydantic-settings
uv add --dev ruff pytest pytest-asyncio pytest-cov
```

**Step 2: Configure ruff in pyproject.toml**

Add to `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "ASYNC"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 3: Verify tooling works**

```bash
uv run ruff check .
uv run pytest --co  # should collect 0 tests
```

**Step 4: Commit**

```bash
git add -A && git commit -m "chore: add dependencies and configure ruff + pytest"
```

---

### Task 0.2: Create project package structure

**Objective:** Set up the `src/agent_runtime/` package layout.

**Files to create:**

```
src/agent_runtime/__init__.py
src/agent_runtime/config.py
src/agent_runtime/db/__init__.py
src/agent_runtime/tools/__init__.py
src/agent_runtime/agents/__init__.py
tests/__init__.py
tests/conftest.py
```

**Step 1: Create the package directories**

```bash
mkdir -p src/agent_runtime/{db,tools,agents}
mkdir -p tests
```

**Step 2: Create `src/agent_runtime/__init__.py`**

```python
"""Agent runtime powered by OpenAI Agents SDK."""

__version__ = "0.1.0"
```

**Step 3: Create `src/agent_runtime/config.py`**

```python
"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # PostgreSQL
    database_url: str = "postgresql://localhost:5432/agent_runtime"

    # TinyFish
    tinyfish_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

**Step 4: Create `tests/conftest.py`**

```python
"""Shared test fixtures."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

**Step 5: Create empty `__init__.py` files**

Create empty files at:
- `src/agent_runtime/db/__init__.py`
- `src/agent_runtime/tools/__init__.py`
- `src/agent_runtime/agents/__init__.py`
- `tests/__init__.py`

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: create project package structure"
```

---

## Phase 1: Database Layer

### Task 1.1: Alembic init and migration config

**Objective:** Set up Alembic for PostgreSQL migrations.

**Step 1: Initialize Alembic**

```bash
uv run alembic init alembic
```

**Step 2: Edit `alembic/env.py` — import settings and target metadata**

Replace the `run_migrations_online` function's `connectable` section to use our `database_url` from config. Add at the top:

```python
from agent_runtime.config import settings
from agent_runtime.db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)
target_metadata = Base.metadata
```

**Step 3: Add SQLAlchemy to dependencies (for Alembic only)**

```bash
uv add sqlalchemy asyncpg
```

**Step 4: Create `src/agent_runtime/db/models.py`**

```python
"""SQLAlchemy models for conversation persistence."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    title: Mapped[str] = mapped_column(String(255), default="New Conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))  # "user", "assistant", "tool"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
```

**Step 5: Generate initial migration**

```bash
uv run alembic revision --autogenerate -m "create conversations and messages tables"
```

**Step 6: Verify migration file was created**

Check `alembic/versions/` for the new migration file.

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: add Alembic setup and conversation/message models"
```

---

### Task 1.2: Database connection pool

**Objective:** Create an asyncpg connection pool manager.

**Step 1: Write failing test — `tests/test_db_connection.py`**

```python
"""Tests for database connection pool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.db.connection import Database


@pytest.mark.asyncio
async def test_database_connect_creates_pool():
    db = Database("postgresql://localhost/test")
    with patch("agent_runtime.db.connection.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        mock_pool = AsyncMock()
        mock_create.return_value = mock_pool
        await db.connect()
        mock_create.assert_called_once_with("postgresql://localhost/test", min_size=2, max_size=10)
        assert db.pool is mock_pool


@pytest.mark.asyncio
async def test_database_disconnect():
    db = Database("postgresql://localhost/test")
    mock_pool = AsyncMock()
    db.pool = mock_pool
    await db.disconnect()
    mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_database_execute():
    db = Database("postgresql://localhost/test")
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    db.pool = mock_pool

    await db.execute("INSERT INTO test VALUES ($1)", "value")
    mock_conn.execute.assert_called_once_with("INSERT INTO test VALUES ($1)", "value")
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_db_connection.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agent_runtime.db.connection'`

**Step 3: Implement `src/agent_runtime/db/connection.py`**

```python
"""Async PostgreSQL connection pool using asyncpg."""

import asyncpg


class Database:
    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            self.dsn, min_size=self.min_size, max_size=self.max_size
        )

    async def disconnect(self) -> None:
        if self.pool:
            self.pool.close()

    async def execute(self, query: str, *args) -> str:
        assert self.pool is not None, "Database not connected"
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        assert self.pool is not None, "Database not connected"
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        assert self.pool is not None, "Database not connected"
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_db_connection.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add asyncpg database connection pool"
```

---

### Task 1.3: Conversation repository

**Objective:** CRUD operations for conversations and messages.

**Step 1: Write failing tests — `tests/test_conversation_repo.py`**

```python
"""Tests for conversation repository."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.db.conversation_repo import ConversationRepo


@pytest.fixture
def mock_db():
    db = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_create_conversation(mock_db):
    mock_db.fetchrow.return_value = {"id": "abc-123", "title": "Test"}
    repo = ConversationRepo(mock_db)

    result = await repo.create_conversation("abc-123", "Test")

    assert result["id"] == "abc-123"
    mock_db.fetchrow.assert_called_once()


@pytest.mark.asyncio
async def test_add_message(mock_db):
    mock_db.fetchrow.return_value = {"id": 1}
    repo = ConversationRepo(mock_db)

    result = await repo.add_message("abc-123", "user", "Hello!")

    assert result["id"] == 1
    mock_db.fetchrow.assert_called_once()


@pytest.mark.asyncio
async def test_get_messages(mock_db):
    mock_db.fetch.return_value = [
        {"id": 1, "role": "user", "content": "Hello"},
        {"id": 2, "role": "assistant", "content": "Hi!"},
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
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_conversation_repo.py -v
```

Expected: FAIL

**Step 3: Implement `src/agent_runtime/db/conversation_repo.py`**

```python
"""Repository for conversation and message CRUD."""

from agent_runtime.db.connection import Database


class ConversationRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_conversation(self, conversation_id: str, title: str = "New Conversation"):
        return await self.db.fetchrow(
            "INSERT INTO conversations (id, title) VALUES ($1, $2) RETURNING id, title, created_at",
            conversation_id,
            title,
        )

    async def add_message(self, conversation_id: str, role: str, content: str):
        return await self.db.fetchrow(
            "INSERT INTO messages (conversation_id, role, content) "
            "VALUES ($1, $2, $3) RETURNING id, role, content, created_at",
            conversation_id,
            role,
            content,
        )

    async def get_messages(self, conversation_id: str) -> list[dict]:
        rows = await self.db.fetch(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = $1 ORDER BY created_at",
            conversation_id,
        )
        return [dict(row) for row in rows]

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        rows = await self.db.fetch(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT $1",
            limit,
        )
        return [dict(row) for row in rows]

    async def get_conversation(self, conversation_id: str):
        return await self.db.fetchrow(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = $1",
            conversation_id,
        )
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_conversation_repo.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add conversation repository with CRUD operations"
```

---

## Phase 2: Agent Tools

### Task 2.1: TinyFish web search tool

**Objective:** Create an OpenAI Agents SDK `@function_tool` for web search via TinyFish.

**Step 1: Write failing test — `tests/tools/test_web_search.py`**

```python
"""Tests for TinyFish web search tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.web_search import web_search


@pytest.mark.asyncio
async def test_web_search_returns_results():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": "python tutorial",
        "results": [
            {"position": 1, "title": "Python.org", "snippet": "Official Python docs", "url": "https://python.org"},
            {"position": 2, "title": "Learn Python", "snippet": "Free course", "url": "https://learnpython.org"},
        ],
        "total_results": 2,
    }
    mock_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.web_search._client") as mock_client:
        mock_client.get.return_value = mock_response

        result = await web_search("python tutorial")

        assert "Python.org" in result
        assert "https://python.org" in result


@pytest.mark.asyncio
async def test_web_search_handles_empty_results():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": "nonsense query xyz",
        "results": [],
        "total_results": 0,
    }
    mock_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.web_search._client") as mock_client:
        mock_client.get.return_value = mock_response

        result = await web_search("nonsense query xyz")

        assert "No results found" in result
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/tools/test_web_search.py -v
```

Expected: FAIL

**Step 3: Implement `src/agent_runtime/tools/web_search.py`**

```python
"""Web search tool powered by TinyFish Search API."""

import os

import httpx

from agents import function_tool

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"X-API-Key": os.environ.get("TINYFISH_API_KEY", "")},
            timeout=15.0,
        )
    return _client


@function_tool
async def web_search(query: str, location: str = "US", language: str = "en") -> str:
    """Search the web for current information.

    Args:
        query: Search query string.
        location: Country code for geo-targeting (e.g., US, GB, ID).
        language: Language code (e.g., en, id).
    """
    client = _get_client()
    response = await client.get(
        TINYFISH_SEARCH_URL,
        params={"query": query, "location": location, "language": language},
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    if not results:
        return "No results found for your query."

    formatted = []
    for r in results[:5]:
        formatted.append(f"[{r['title']}]({r['url']})\n  {r['snippet']}")

    return "\n\n".join(formatted)
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/tools/test_web_search.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add TinyFish web search tool"
```

---

### Task 2.2: Weather tool (Open-Meteo)

**Objective:** Weather forecast tool using Open-Meteo (no API key needed).

**Step 1: Write failing test — `tests/tools/test_weather.py`**

```python
"""Tests for weather tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.weather import get_weather


@pytest.mark.asyncio
async def test_get_weather_returns_forecast():
    # Mock geocoding response
    geo_response = AsyncMock()
    geo_response.status_code = 200
    geo_response.json.return_value = {
        "results": [{"latitude": -6.2, "longitude": 106.8, "name": "Jakarta"}]
    }
    geo_response.raise_for_status = AsyncMock()

    # Mock weather response
    weather_response = AsyncMock()
    weather_response.status_code = 200
    weather_response.json.return_value = {
        "current": {
            "temperature_2m": 30.5,
            "relative_humidity_2m": 75,
            "weather_code": 2,
            "wind_speed_10m": 12.3,
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h",
        },
    }
    weather_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.weather._client") as mock_client:
        mock_client.get = AsyncMock(side_effect=[geo_response, weather_response])

        result = await get_weather("Jakarta")

        assert "Jakarta" in result
        assert "30.5" in result


@pytest.mark.asyncio
async def test_get_weather_city_not_found():
    geo_response = AsyncMock()
    geo_response.status_code = 200
    geo_response.json.return_value = {"results": []}
    geo_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.weather._client") as mock_client:
        mock_client.get.return_value = geo_response

        result = await get_weather("FakeCityXYZ123")

        assert "not found" in result.lower()
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/tools/test_weather.py -v
```

Expected: FAIL

**Step 3: Implement `src/agent_runtime/tools/weather.py`**

```python
"""Weather forecast tool using Open-Meteo (free, no API key)."""

import httpx

from agents import function_tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


@function_tool
async def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name (e.g., "Jakarta", "Tokyo", "New York").
    """
    client = _get_client()

    # Step 1: Geocode city name → lat/lon
    geo_resp = await client.get(GEOCODING_URL, params={"name": city, "count": 1})
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    results = geo_data.get("results", [])
    if not results:
        return f"City '{city}' not found. Please check the spelling."

    location = results[0]
    lat, lon = location["latitude"], location["longitude"]
    display_name = location.get("name", city)
    country = location.get("country", "")

    # Step 2: Fetch current weather
    weather_resp = await client.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        },
    )
    weather_resp.raise_for_status()
    weather = weather_resp.json()

    current = weather["current"]
    units = weather["current_units"]
    description = WMO_CODES.get(current["weather_code"], "Unknown")

    return (
        f"Weather in {display_name}, {country}:\n"
        f"  Condition: {description}\n"
        f"  Temperature: {current['temperature_2m']}{units['temperature_2m']}\n"
        f"  Humidity: {current['relative_humidity_2m']}{units['relative_humidity_2m']}\n"
        f"  Wind Speed: {current['wind_speed_10m']} {units['wind_speed_10m']}"
    )
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/tools/test_weather.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add weather tool using Open-Meteo"
```

---

### Task 2.3: Currency conversion tool (Frankfurter)

**Objective:** Currency conversion using Frankfurter API (free, no key, ECB rates).

**Step 1: Write failing test — `tests/tools/test_currency.py`**

```python
"""Tests for currency conversion tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.currency import convert_currency


@pytest.mark.asyncio
async def test_convert_currency_success():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "amount": 100.0,
        "base": "USD",
        "date": "2026-06-06",
        "rates": {"EUR": 92.5},
    }
    mock_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.currency._client") as mock_client:
        mock_client.get.return_value = mock_response

        result = await convert_currency(100, "USD", "EUR")

        assert "USD" in result
        assert "EUR" in result
        assert "92.5" in result


@pytest.mark.asyncio
async def test_convert_currency_api_error():
    mock_response = AsyncMock()
    mock_response.status_code = 400
    mock_response.text = "Invalid currency"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad request", request=AsyncMock(), response=mock_response
    )

    with patch("agent_runtime.tools.currency._client") as mock_client:
        mock_client.get.return_value = mock_response

        result = await convert_currency(100, "USD", "FAKE")

        assert "error" in result.lower() or "invalid" in result.lower()
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/tools/test_currency.py -v
```

Expected: FAIL

**Step 3: Implement `src/agent_runtime/tools/currency.py`**

```python
"""Currency conversion tool using Frankfurter API (ECB rates, free, no key)."""

import httpx

from agents import function_tool

FRANKFURTER_URL = "https://api.frankfurter.dev/latest"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


@function_tool
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using live exchange rates.

    Args:
        amount: Amount to convert.
        from_currency: Source currency code (e.g., "USD", "EUR", "IDR").
        to_currency: Target currency code (e.g., "JPY", "GBP").
    """
    client = _get_client()
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    try:
        response = await client.get(
            FRANKFURTER_URL,
            params={"amount": amount, "from": from_currency, "to": to_currency},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError:
        return f"Error: Invalid currency code(s) — {from_currency} or {to_currency} not supported."

    data = response.json()
    rate = data["rates"][to_currency]

    return (
        f"{amount:,.2f} {from_currency} = {rate:,.2f} {to_currency}\n"
        f"Date: {data['date']}"
    )
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/tools/test_currency.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add currency conversion tool using Frankfurter API"
```

---

### Task 2.4: Country info tool (REST Countries)

**Objective:** Country lookup tool using REST Countries API (free, no key).

**Step 1: Write failing test — `tests/tools/test_country.py`**

```python
"""Tests for country info tool."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.tools.country import get_country_info


@pytest.mark.asyncio
async def test_get_country_info_success():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "name": {"common": "Indonesia", "official": "Republic of Indonesia"},
            "capital": ["Jakarta"],
            "region": "Asia",
            "subregion": "South-Eastern Asia",
            "population": 273523621,
            "currencies": {"IDR": {"name": "Indonesian rupiah", "symbol": "Rp"}},
            "languages": {"ind": "Indonesian"},
            "flag": "🇮🇩",
        }
    ]
    mock_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.country._client") as mock_client:
        mock_client.get.return_value = mock_response

        result = await get_country_info("Indonesia")

        assert "Indonesia" in result
        assert "Jakarta" in result
        assert "273" in result  # population


@pytest.mark.asyncio
async def test_get_country_info_not_found():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_response.raise_for_status = AsyncMock()

    with patch("agent_runtime.tools.country._client") as mock_client:
        mock_client.get.return_value = mock_response

        result = await get_country_info("FakeCountryXYZ")

        assert "not found" in result.lower()
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/tools/test_country.py -v
```

Expected: FAIL

**Step 3: Implement `src/agent_runtime/tools/country.py`**

```python
"""Country info tool using REST Countries API (free, no key)."""

import httpx

from agents import function_tool

REST_COUNTRIES_URL = "https://restcountries.com/v3.1/name"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


@function_tool
async def get_country_info(country_name: str) -> str:
    """Get information about a country — capital, population, region, currency, languages.

    Args:
        country_name: Country name (e.g., "Indonesia", "Japan", "Brazil").
    """
    client = _get_client()
    response = await client.get(f"{REST_COUNTRIES_URL}/{country_name}")
    response.raise_for_status()

    data = response.json()
    if not data:
        return f"Country '{country_name}' not found."

    c = data[0]
    name = c["name"]["common"]
    capital = ", ".join(c.get("capital", ["N/A"]))
    region = c.get("region", "N/A")
    subregion = c.get("subregion", "")
    population = c.get("population", 0)
    flag = c.get("flag", "")

    currencies = ", ".join(
        f"{v['name']} ({v.get('symbol', '')})" for v in c.get("currencies", {}).values()
    )
    languages = ", ".join(c.get("languages", {}).values())

    return (
        f"{flag} {name} ({c['name'].get('official', name)})\n"
        f"  Capital: {capital}\n"
        f"  Region: {region} — {subregion}\n"
        f"  Population: {population:,}\n"
        f"  Currencies: {currencies or 'N/A'}\n"
        f"  Languages: {languages or 'N/A'}"
    )
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/tools/test_country.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add country info tool using REST Countries API"
```

---

## Phase 3: Agent Wiring

### Task 3.1: Main agent definition

**Objective:** Create the core agent with all tools registered and a conversational system prompt.

**Step 1: Write failing test — `tests/test_agent.py`**

```python
"""Tests for the main agent definition."""

import pytest

from agent_runtime.agents.runtime import create_agent, run_agent


def test_create_agent_has_all_tools():
    agent = create_agent()
    tool_names = {t.name if hasattr(t, "name") else str(t) for t in agent.tools}
    expected = {"web_search", "get_weather", "convert_currency", "get_country_info"}
    # The tools list may contain tool objects — check length
    assert len(agent.tools) == 4


def test_create_agent_has_name():
    agent = create_agent()
    assert agent.name == "RuntimeAgent"
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: FAIL

**Step 3: Implement `src/agent_runtime/agents/runtime.py`**

```python
"""Main agent definition with all tools."""

from agents import Agent, Runner

from agent_runtime.config import settings
from agent_runtime.tools.web_search import web_search
from agent_runtime.tools.weather import get_weather
from agent_runtime.tools.currency import convert_currency
from agent_runtime.tools.country import get_country_info

SYSTEM_PROMPT = """You are a helpful assistant with access to these tools:
- web_search: Search the web for current information (TinyFish)
- get_weather: Get current weather for any city (Open-Meteo)
- convert_currency: Convert between currencies with live rates (Frankfurter/ECB)
- get_country_info: Look up country details — capital, population, currency, languages (REST Countries)

Use tools when the user's question requires real-time data or lookup.
Be concise and factual. When showing results, format them clearly."""


def create_agent() -> Agent:
    """Create the configured runtime agent."""
    return Agent(
        name="RuntimeAgent",
        instructions=SYSTEM_PROMPT,
        model=settings.openai_model,
        tools=[web_search, get_weather, convert_currency, get_country_info],
    )


async def run_agent(user_message: str, conversation_id: str | None = None) -> str:
    """Run the agent with a user message and return the response."""
    agent = create_agent()
    result = await Runner.run(agent, input=user_message)
    return result.final_output
```

**Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_agent.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: create main agent with all tools registered"
```

---

### Task 3.2: CLI runner

**Objective:** A simple CLI chat loop to interact with the agent.

**Step 1: Implement `main.py`**

```python
"""CLI chat interface for the agent runtime."""

import asyncio
import uuid

from agent_runtime.agents.runtime import run_agent


async def chat_loop() -> None:
    conversation_id = str(uuid.uuid4())
    print(f"Agent Runtime — conversation {conversation_id[:8]}...")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        response = await run_agent(user_input, conversation_id=conversation_id)
        print(f"\nAgent: {response}\n")


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
```

**Step 2: Test manually (optional)**

```bash
# Only if OPENAI_API_KEY is set
uv run python main.py
```

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add CLI chat loop for agent runtime"
```

---

## Phase 4: Polish

### Task 4.1: Ruff check and fix

**Objective:** Ensure all code passes ruff linting.

**Step 1: Run ruff**

```bash
uv run ruff check .
uv run ruff format .
```

**Step 2: Fix any issues**

Address any lint errors reported.

**Step 3: Run full test suite**

```bash
uv run pytest -v --tb=short
```

Expected: All tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "style: ruff lint and format pass"
```

---

### Task 4.2: .env.example and .gitignore

**Objective:** Document required environment variables and protect secrets.

**Step 1: Create `.env.example`**

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgresql://user:pass@localhost:5432/agent_runtime
TINYFISH_API_KEY=your_tinyfish_key_here
```

**Step 2: Ensure `.gitignore` includes `.env`**

Add to `.gitignore` if not present:

```
.env
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
```

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: add .env.example and update .gitignore"
```

---

### Task 4.3: Update README

**Objective:** Write a proper README with setup instructions, architecture overview, and tool descriptions.

**Step 1: Write `README.md`**

```markdown
# Agent Runtime

An agent runtime built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
with PostgreSQL persistence and external API tools.

## Architecture

```
main.py                         CLI chat loop
src/agent_runtime/
  config.py                     Settings (env vars, .env)
  agents/runtime.py             Agent definition + runner
  tools/
    web_search.py               TinyFish Search API
    weather.py                  Open-Meteo (free, no key)
    currency.py                 Frankfurter / ECB rates
    country.py                  REST Countries API
  db/
    connection.py               asyncpg pool
    conversation_repo.py        Conversation & message CRUD
    models.py                   SQLAlchemy models (Alembic)
```

## Setup

```bash
# 1. Install deps
uv sync

# 2. Create .env from example
cp .env.example .env
# Edit .env with your keys

# 3. Create PostgreSQL database
createdb agent_runtime

# 4. Run migrations
uv run alembic upgrade head

# 5. Run the agent
uv run python main.py
```

## Tools

| Tool             | API            | Auth     | What it does                              |
|------------------|----------------|----------|-------------------------------------------|
| `web_search`     | TinyFish       | API key  | Web search with geo-targeting             |
| `get_weather`    | Open-Meteo     | None     | Current weather for any city              |
| `convert_currency` | Frankfurter  | None     | Live ECB exchange rates                   |
| `get_country_info` | REST Countries | None    | Country capital, population, currencies   |

## Development

```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run pytest -v           # Test
uv run pytest --cov        # Coverage
```
```

**Step 2: Commit**

```bash
git add -A && git commit -m "docs: update README with setup and architecture"
```

---

## Final Checklist

- [ ] All tests pass: `uv run pytest -v`
- [ ] No lint errors: `uv run ruff check .`
- [ ] Migrations run cleanly: `uv run alembic upgrade head`
- [ ] CLI works: `uv run python main.py`
- [ ] Each tool responds correctly in conversation
- [ ] `.env` not committed, `.env.example` is present
