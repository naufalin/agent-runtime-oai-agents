"""FastAPI application with lifespan management."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from agent_runtime.agents.runtime import get_db  # noqa: E402
from agent_runtime.api.routers import prompts, sessions  # noqa: E402
from agent_runtime.db.prompt_repo import SystemPromptRepo  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect DB, seed default prompt
    db = await get_db()
    prompt_repo = SystemPromptRepo(db)
    await prompt_repo.seed_default()
    yield
    # Shutdown: disconnect DB
    await db.disconnect()


app = FastAPI(
    title="Agent Runtime",
    description="Agent runtime API powered by OpenAI Agents SDK",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(prompts.router)


@app.get("/")
async def health():
    return {"status": "ok", "version": "0.1.0"}
