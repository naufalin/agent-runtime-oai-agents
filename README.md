# Agent Runtime

An agent runtime powered by the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
with PostgreSQL persistence, versioned system prompts, and external API tools.

## Quick Start

```bash
# 1. Install
uv sync

# 2. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, DATABASE_URL, and AGENT_RUNTIME_BEARER_TOKEN

# 3. Create database & run migrations
createdb agent_runtime
uv run alembic upgrade head

# 4. Start the server
uv run python main.py
# → http://localhost:8000/docs?token=<AGENT_RUNTIME_BEARER_TOKEN> (Swagger UI)
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

All API endpoints except `GET /` require bearer-token authentication. Set
`AGENT_RUNTIME_BEARER_TOKEN` in the server environment, then send the same value
from trusted callers:

```bash
curl http://localhost:8000/sessions \
  -H "Authorization: Bearer $AGENT_RUNTIME_BEARER_TOKEN"
```

If `AGENT_RUNTIME_BEARER_TOKEN` is blank or missing, protected endpoints return
`503` until the service is configured.

Swagger and ReDoc also accept a browser-friendly query token:
`/docs?token=<AGENT_RUNTIME_BEARER_TOKEN>` and
`/redoc?token=<AGENT_RUNTIME_BEARER_TOKEN>`.

### Sessions

| Method | Endpoint                   | Description                         |
|--------|----------------------------|-------------------------------------|
| POST   | `/sessions`                | Create a new session                |
| GET    | `/sessions`                | List sessions                       |
| GET    | `/sessions/{id}`           | Get session detail + messages       |
| POST   | `/sessions/{id}/chat`      | Send message, get agent response    |
| POST   | `/sessions/{id}/chat/stream` | Send message, stream response (SSE) |
| GET    | `/models`                  | List supported model IDs and defaults |

Chat requests accept optional model controls:

```json
{
  "message": "Hello",
  "provider": "openrouter",
  "model": "z-ai/glm-5.2",
  "reasoning_effort": "high"
}
```

If `provider` is omitted and `model` is one of the supported OpenRouter IDs, the
request is routed to OpenRouter for that single run. Responses and session
history include nullable `provider`, `model`, `usage`, and `thinking` metadata.

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

### SSE Streaming

The chat endpoint supports real-time streaming via Server-Sent Events (SSE).

```bash
# Stream a chat response
curl -N -X POST http://localhost:8000/sessions/<id>/chat/stream \
  -H "Authorization: Bearer $AGENT_RUNTIME_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

Each SSE frame is `data: {json}\n\n` with these event types:

| Event Type    | Fields          | Description                          |
|---------------|-----------------|--------------------------------------|
| `text_delta`  | `delta`         | Incremental text chunk from the LLM  |
| `tool_start`  | `tool`          | Tool call begins                     |
| `tool_end`    | `tool`          | Tool call completes                  |
| `done`        | `session_id`, `provider`, `model`, `usage`, `thinking` | Stream finished successfully |
| `error`       | `message`       | Error occurred                       |

Example output:

```
data: {"type": "text_delta", "delta": "Hello"}

data: {"type": "text_delta", "delta": " world"}

data: {"type": "tool_start", "tool": "web_search"}

data: {"type": "tool_end", "tool": "web_search"}

data: {"type": "done", "session_id": "abc123", "provider": "openrouter", "model": "z-ai/glm-5.2", "usage": {"total_tokens": 42, "reasoning_tokens": 8}, "thinking": null}
```

The streaming endpoint accepts the same `ChatRequest` body as the non-streaming `/chat` endpoint. The final assistant message is persisted to the database after streaming completes.

## Tools

### Free (no credits)

| Tool         | API            | Auth    | Description                                          |
|--------------|----------------|---------|------------------------------------------------------|
| `web_search` | TinyFish Search | API key | Web search with geo-targeting and pagination         |
| `web_fetch`  | TinyFish Fetch  | API key | Extract clean content from URLs (batch, PDF support) |
| `get_weather`    | Open-Meteo     | None    | Current weather for any city                         |
| `convert_currency` | Frankfurter    | None    | Live ECB exchange rates                              |
| `get_country_info` | REST Countries | None    | Capital, population, currencies, languages           |

### Paid (uses TinyFish credits — reserved for future use)

| Tool          | API              | Description                                        |
|---------------|------------------|----------------------------------------------------|
| `web_agent`   | TinyFish Agent   | Goal-based browser automation on any website       |
| `web_browser` | TinyFish Browser | Create remote browser session for Playwright/CDP   |

> **Note:** `web_agent` and `web_browser` are reserved for future use. 
> They consume TinyFish credits and are most useful when the runtime
> itself needs to drive browser automation programmatically.

Tool calls and responses are automatically persisted via `RunHooks`.

## System Prompts

The runtime supports versioned, reusable system prompts with mid-session switching.

```bash
# List prompts
curl http://localhost:8000/prompts \
  -H "Authorization: Bearer $AGENT_RUNTIME_BEARER_TOKEN"

# Create a custom prompt
curl -X POST http://localhost:8000/prompts \
  -H "Authorization: Bearer $AGENT_RUNTIME_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "pirate", "content": "You are a pirate. Always respond with nautical language."}'

# Switch prompt mid-session
curl -X POST http://localhost:8000/sessions/<id>/prompt \
  -H "Authorization: Bearer $AGENT_RUNTIME_BEARER_TOKEN" \
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
    runtime.py                           Agent definition, run_agent, run_agent_streamed, switch_prompt
    hooks.py                             PersistenceHooks (tool call tracking)
  db/
    connection.py                        SQLAlchemy async engine
    models.py                            Session, Message, SystemPrompt
    session_repo.py                      Session/message CRUD
    prompt_repo.py                       System prompt CRUD
  tools/
    _tinyfish_common.py                   Shared TinyFish client, auth, retry
    web_search.py                         TinyFish Search API (free)
    web_fetch.py                          TinyFish Fetch API (free, batch)
    web_agent.py                          TinyFish Agent API (credits, future use)
    web_browser.py                        TinyFish Browser API (credits, future use)
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
uv run pytest -v             # test (91 tests)
uv run pytest --cov          # coverage
```

## Environment Variables

| Variable           | Required | Default                        | Description                  |
|--------------------|----------|--------------------------------|------------------------------|
| `OPENAI_API_KEY`   | For OpenAI | —                              | OpenAI API key               |
| `OPENAI_MODEL`     | No       | `gpt-5.4-mini`                | Default OpenAI model          |
| `AGENT_RUNTIME_MODEL_PROVIDER` | No | `openai` | Default provider: `openai` or `openrouter` |
| `OPENROUTER_API_KEY` | For OpenRouter | — | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenRouter OpenAI-compatible base URL |
| `OPENROUTER_MODEL` | No | `z-ai/glm-5.2` | Default OpenRouter model |
| `OPENROUTER_REASONING_EFFORT` | No | — | Optional OpenRouter reasoning effort, e.g. `high` |
| `DATABASE_URL`     | No       | `postgresql://localhost:5432/agent_runtime` | PostgreSQL connection |
| `TINYFISH_API_KEY` | No       | —                              | TinyFish API key (search+fetch free, agent+browser use credits) |
| `AGENT_RUNTIME_BEARER_TOKEN` | Yes for API | — | Shared bearer token for all non-health API routes |
