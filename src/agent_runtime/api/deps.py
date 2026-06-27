"""FastAPI dependency injection for database and repos."""

from agent_runtime.config import Settings, settings
from agent_runtime.db.connection import get_db
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.runtime_model_repo import RuntimeModelRepo
from agent_runtime.db.session_repo import SessionRepo


async def get_settings() -> Settings:
    return settings


async def get_session_repo() -> SessionRepo:
    db = await get_db()
    return SessionRepo(db)


async def get_prompt_repo() -> SystemPromptRepo:
    db = await get_db()
    return SystemPromptRepo(db)


async def get_runtime_model_repo() -> RuntimeModelRepo:
    db = await get_db()
    return RuntimeModelRepo(db)


async def get_agent_factory():
    from agent_runtime.agents.runtime import AgentFactory

    return AgentFactory(default_model=settings.openai_model)
