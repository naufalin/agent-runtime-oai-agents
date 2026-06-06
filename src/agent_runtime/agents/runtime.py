"""Main agent definition with all tools."""

from agents import Agent, Runner

from agent_runtime.config import Settings
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
    """Run the agent with a user message and return the response."""
    agent = create_agent()
    result = await Runner.run(agent, input=user_message)
    return result.final_output
