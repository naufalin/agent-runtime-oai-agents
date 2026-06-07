"""FastAPI dependency injection for database and repos."""

from agent_runtime.agents.runtime import get_db
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.session_repo import SessionRepo


async def get_session_repo() -> SessionRepo:
    db = await get_db()
    return SessionRepo(db)


async def get_prompt_repo() -> SystemPromptRepo:
    db = await get_db()
    return SystemPromptRepo(db)
