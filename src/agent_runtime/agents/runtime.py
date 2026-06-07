"""Main agent definition with all tools and persistence."""

import uuid

from agents import Agent, Runner

from agent_runtime.config import Settings
from agent_runtime.db.connection import Database
from agent_runtime.db.conversation_repo import ConversationRepo
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
    """Get or create the database connection pool."""
    global _db
    if _db is None:
        settings = Settings()
        _db = Database(settings.database_url)
        await _db.connect()
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


async def run_agent(user_message: str, conversation_id: str | None = None) -> str:
    """Run the agent with a user message, persist to DB, and return the response."""
    db = await get_db()
    repo = ConversationRepo(db)

    # Create conversation if new
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    existing = await repo.get_conversation(conversation_id)
    if not existing:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        await repo.create_conversation(conversation_id, title)

    # Save user message
    await repo.add_message(conversation_id, "user", user_message)

    # Run the agent
    agent = create_agent()
    result = await Runner.run(agent, input=user_message)
    response = result.final_output

    # Save agent response
    await repo.add_message(conversation_id, "assistant", response)

    return response
