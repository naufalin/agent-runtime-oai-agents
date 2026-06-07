"""Main agent definition with all tools and persistence."""

from dataclasses import dataclass

from agents import Agent, Runner

from agent_runtime.config import Settings
from agent_runtime.db.connection import Database
from agent_runtime.db.conversation_repo import ConversationRepo
from agent_runtime.ids import decode, encode
from agent_runtime.tools.country import get_country_info
from agent_runtime.tools.currency import convert_currency
from agent_runtime.tools.weather import get_weather
from agent_runtime.tools.web_search import web_search

SYSTEM_PROMPT = """You are a helpful assistant with access to these tools:
- web_search: Search the web for current information (TinyFish)
- get_weather: Get current weather for any city (Open-Meteo)
- convert_currency: Convert between currencies with live rates (Frankfurter/ECB)
- get_country_info: Look up country details — capital, population, currency,
  languages (REST Countries)

Use tools when the user's question requires real-time data or lookup.
Be concise and factual. When showing results, format them clearly."""

_db: Database | None = None


async def get_db() -> Database:
    """Get or create the database engine."""
    global _db
    if _db is None:
        settings = Settings()
        _db = Database(settings.database_url)
        _db.connect()
    return _db


def create_agent() -> Agent:
    """Create the configured runtime agent."""
    settings = Settings()
    return Agent(
        name="RuntimeAgent",
        instructions=SYSTEM_PROMPT,
        model=settings.openai_model,
        tools=[web_search, get_weather, convert_currency, get_country_info],
    )


@dataclass
class AgentResponse:
    """Response from run_agent with the conversation ID for follow-up messages."""

    response: str
    conversation_id: str  # encoded ID for external use


async def run_agent(user_message: str, conversation_id: str | None = None) -> AgentResponse:
    """Run the agent with a user message, persist to DB, and return the response.

    Args:
        user_message: The user's message text.
        conversation_id: Encoded conversation ID (from ids.encode). None to start new.

    Returns:
        The agent's response text.
    """
    db = await get_db()
    repo = ConversationRepo(db)

    # Resolve conversation: decode existing or create new
    internal_id: int | None = None
    if conversation_id is not None:
        try:
            internal_id = decode(conversation_id)
            existing = await repo.get_conversation(internal_id)
            if not existing:
                internal_id = None
        except ValueError:
            internal_id = None

    if internal_id is None:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        conv = await repo.create_conversation(title)
        internal_id = conv.id

    # Save user message
    await repo.add_message(internal_id, "user", user_message)

    # Run the agent
    agent = create_agent()
    result = await Runner.run(agent, input=user_message)
    response = result.final_output

    # Save agent response
    await repo.add_message(internal_id, "assistant", response)

    return AgentResponse(response=response, conversation_id=encode(internal_id))
