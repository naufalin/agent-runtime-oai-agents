# Agent Runtime

An agent runtime powered by the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
with PostgreSQL persistence, versioned system prompts, and external API tools.

## Quick Start

```bash
# 1. Install
uv sync

# 2. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and DATABASE_URL

# 3. Create database & run migrations
createdb agent_runtime
uv run alembic upgrade head

# 4. Start the server
uv run python main.py
# → http://localhost:8000/docs (Swagger UI)
```

## Usage

### API Server (primary)

```bash
uv run python main.py                              # 0.0.0.0:8000
uv run python main.py --port 3000                   # custom port
uv run python main.py --host 127.0.0.1 --port 3000  # localhost only
uv run python main.py --reload                      # dev mode, auto-reload
uv run python main.py --workers 4                   # multi-worker
uv run python main.py --log-level debug             # verbose
```

### CLI (secondary)

```bash
uv run python main.py cli                    # interactive chat
uv run python main.py cli list               # list sessions
uv run python main.py cli resume <id>        # resume a session
uv run python main.py cli prompts            # list system prompts
uv run python main.py cli create-prompt <name> <content>
```

### Entry points (after `uv sync`)

```bash
agent-runtime                # starts API server
agent-runtime --port 3000    # with args
agent-runtime-cli            # starts CLI
```

## API Endpoints

### Sessions

| Method | Endpoint                   | Description                         |
|--------|----------------------------|-------------------------------------|
| POST   | `/sessions`                | Create a new session                |
| GET    | `/sessions`                | List sessions                       |
| GET    | `/sessions/{id}`           | Get session detail + messages       |
| POST   | `/sessions/{id}/chat`      | Send message, get agent response    |

### Prompts

| Method | Endpoint                   | Description                         |
|--------|----------------------------|-------------------------------------|
| GET    | `/prompts`                 | List all system prompts             |
| POST   | `/prompts`                 | Create a new prompt                 |
| POST   | `/sessions/{id}/prompt`    | Switch prompt mid-session           |

### System

| Method | Endpoint | Description     |
|--------|----------|-----------------|
| GET    | `/`      | Health check    |

## Tools

| Tool               | API            | Auth     | Description                              |
|--------------------|----------------|----------|------------------------------------------|
| `web_search`       | TinyFish       | API key  | Web search with geo-targeting            |
| `get_weather`      | Open-Meteo     | None     | Current weather for any city             |
| `convert_currency` | Frankfurter    | None     | Live ECB exchange rates                  |
| `get_country_info` | REST Countries | None     | Capital, population, currencies, languages |

Tool calls and responses are automatically persisted via `RunHooks`.

## System Prompts

The runtime supports versioned, reusable system prompts with mid-session switching.

```bash
# List prompts
curl http://localhost:8000/prompts

# Create a custom prompt
curl -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{"name": "pirate", "content": "You are a pirate. Always respond with nautical language."}'

# Switch prompt mid-session
curl -X POST http://localhost:8000/sessions/<id>/prompt \
  -H "Content-Type: application/json" \
  -d '{"name": "pirate"}'
```

A `default` prompt is auto-seeded on first run.

## Architecture

```
main.py                                  Entry point (API server or CLI)
src/agent_runtime/
  cli.py                                 CLI interface
  config.py                              Settings (env vars, .env)
  ids.py                                 sqids encode/decode (integer ↔ opaque string)
  api/
    app.py                               FastAPI app + lifespan
    schemas.py                           Pydantic request/response models
    deps.py                              Dependency injection
    routers/
      sessions.py                        Session endpoints
      prompts.py                         Prompt endpoints
  agents/
    runtime.py                           Agent definition, run_agent, switch_prompt
    hooks.py                             PersistenceHooks (tool call tracking)
  db/
    connection.py                        SQLAlchemy async engine
    models.py                            Session, Message, SystemPrompt
    session_repo.py                      Session/message CRUD
    prompt_repo.py                       System prompt CRUD
  tools/
    web_search.py                        TinyFish Search API
    weather.py                           Open-Meteo
    currency.py                          Frankfurter / ECB
    country.py                           REST Countries
alembic/                                 DB migrations
```

## Database

PostgreSQL with 4 tables:

- **sessions** — id (autoincrement), title, timestamps
- **messages** — id, session_id (FK), role, content, tool_name, system_prompt_id (FK), timestamp
- **system_prompts** — id, name (unique), content, timestamp
- **alembic_version** — migration tracking

Session IDs are integers internally, exposed as opaque strings via [sqids](https://sqids.org/) to prevent enumeration.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest -v             # test (49 tests)
uv run pytest --cov          # coverage
```

## Environment Variables

| Variable           | Required | Default                        | Description                  |
|--------------------|----------|--------------------------------|------------------------------|
| `OPENAI_API_KEY`   | Yes      | —                              | OpenAI API key               |
| `OPENAI_MODEL`     | No       | `gpt-5.4-mini`                | Model for the agent          |
| `DATABASE_URL`     | No       | `postgresql://localhost:5432/agent_runtime` | PostgreSQL connection |
| `TINYFISH_API_KEY` | No       | —                              | TinyFish web search API key  |
