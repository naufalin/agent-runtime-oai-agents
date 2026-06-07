"""Main agent definition with all tools and persistence."""

from dataclasses import dataclass

from agents import Agent, Runner

from agent_runtime.agents.hooks import PersistenceHooks, RunContext
from agent_runtime.config import Settings
from agent_runtime.db.connection import Database
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.session_repo import SessionRepo
from agent_runtime.ids import decode, encode
from agent_runtime.tools.country import get_country_info
from agent_runtime.tools.currency import convert_currency
from agent_runtime.tools.weather import get_weather
from agent_runtime.tools.web_search import web_search

_hooks = PersistenceHooks()

_db: Database | None = None


async def get_db() -> Database:
    """Get or create the database engine."""
    global _db
    if _db is None:
        settings = Settings()
        _db = Database(settings.database_url)
        _db.connect()
    return _db


def create_agent(system_prompt: str) -> Agent:
    """Create the configured runtime agent with the given system prompt."""
    settings = Settings()
    return Agent(
        name="RuntimeAgent",
        instructions=system_prompt,
        model=settings.openai_model,
        tools=[web_search, get_weather, convert_currency, get_country_info],
    )


@dataclass
class AgentResponse:
    """Response from run_agent with the session ID for follow-up messages."""

    response: str
    session_id: str  # encoded ID for external use


async def run_agent(user_message: str, session_id: str | None = None) -> AgentResponse:
    """Run the agent with a user message, persist to DB, and return the response.

    Args:
        user_message: The user's message text.
        session_id: Encoded session ID (from ids.encode). None to start new.

    Returns:
        AgentResponse with the agent's reply and the session ID.
    """
    db = await get_db()
    repo = SessionRepo(db)
    prompt_repo = SystemPromptRepo(db)

    # Resolve session: decode existing or create new
    internal_id: int | None = None
    if session_id is not None:
        try:
            internal_id = decode(session_id)
            existing = await repo.get_session(internal_id)
            if not existing:
                internal_id = None
        except ValueError:
            internal_id = None

    if internal_id is None:
        # New session — seed default prompt and insert system message
        default_prompt = await prompt_repo.seed_default()
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        sess = await repo.create_session(title)
        internal_id = sess.id
        await repo.add_message(
            internal_id,
            "system",
            default_prompt.content,
            system_prompt_id=default_prompt.id,
        )

    # Load the active system prompt from session history
    system_msg = await repo.get_latest_system_message(internal_id)
    system_prompt = system_msg.content if system_msg else ""

    # Save user message
    await repo.add_message(internal_id, "user", user_message)

    # Run the agent with the active system prompt and persistence hooks
    agent = create_agent(system_prompt)
    run_context = RunContext(session_id=internal_id, repo=repo)
    result = await Runner.run(
        agent, input=user_message, context=run_context, hooks=_hooks,
    )
    response = result.final_output

    # Save agent response
    await repo.add_message(internal_id, "assistant", response)

    return AgentResponse(response=response, session_id=encode(internal_id))


async def switch_prompt(session_id: str, prompt_name: str) -> str:
    """Switch the system prompt for an existing session.

    Inserts a new system message with the named prompt. Returns the prompt name.
    """
    db = await get_db()
    repo = SessionRepo(db)
    prompt_repo = SystemPromptRepo(db)

    internal_id = decode(session_id)
    sess = await repo.get_session(internal_id)
    if not sess:
        raise ValueError(f"Session not found: {session_id}")

    prompt = await prompt_repo.get_by_name(prompt_name)
    if not prompt:
        raise ValueError(f"Prompt not found: {prompt_name}")

    await repo.add_message(
        internal_id,
        "system",
        prompt.content,
        system_prompt_id=prompt.id,
    )
    return prompt.name
