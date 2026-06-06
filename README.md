# Agent Runtime

An agent runtime built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
with PostgreSQL persistence and external API tools.

## Architecture

```
main.py                              CLI chat loop
src/agent_runtime/
  config.py                          Settings (env vars, .env)
  agents/runtime.py                  Agent definition + Runner
  tools/
    web_search.py                    TinyFish Search API
    weather.py                       Open-Meteo (free, no key)
    currency.py                      Frankfurter / ECB rates
    country.py                       REST Countries API
  db/
    connection.py                    asyncpg pool
    conversation_repo.py             Conversation & message CRUD
    models.py                        SQLAlchemy models (Alembic)
alembic/                             DB migrations
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

| Tool               | API            | Auth     | What it does                                |
|--------------------|----------------|----------|---------------------------------------------|
| `web_search`       | TinyFish       | API key  | Web search with geo-targeting               |
| `get_weather`      | Open-Meteo     | None     | Current weather for any city                |
| `convert_currency` | Frankfurter    | None     | Live ECB exchange rates                     |
| `get_country_info` | REST Countries | None     | Country capital, population, currencies     |

## Development

```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run pytest -v           # Test
uv run pytest --cov        # Coverage
```

## Environment Variables

| Variable         | Required | Default              | Description                    |
|------------------|----------|----------------------|--------------------------------|
| `OPENAI_API_KEY` | Yes      | —                    | OpenAI API key                 |
| `OPENAI_MODEL`   | No       | `gpt-4o-mini`       | Model for the agent            |
| `DATABASE_URL`   | No       | `postgresql://...`   | PostgreSQL connection string   |
| `TINYFISH_API_KEY`| No      | —                    | TinyFish web search API key    |
