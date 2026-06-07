"""System prompt repository — CRUD + default seeding."""

from sqlalchemy import select

from agent_runtime.db.connection import Database
from agent_runtime.db.models import SystemPrompt

DEFAULT_PROMPT = """You are a helpful assistant with access to these tools:
- web_search: Search the web for current information (TinyFish)
- get_weather: Get current weather for any city (Open-Meteo)
- convert_currency: Convert between currencies with live rates (Frankfurter/ECB)
- get_country_info: Look up country details — capital, population, currency,
  languages (REST Countries)

Use tools when the user's question requires real-time data or lookup.
Be concise and factual. When showing results, format them clearly."""


class SystemPromptRepo:
    def __init__(self, db: Database):
        self.db = db

    async def get_by_name(self, name: str) -> SystemPrompt | None:
        async with self.db.session() as session:
            result = await session.execute(select(SystemPrompt).where(SystemPrompt.name == name))
            return result.scalar_one_or_none()

    async def get_by_id(self, prompt_id: int) -> SystemPrompt | None:
        async with self.db.session() as session:
            return await session.get(SystemPrompt, prompt_id)

    async def list_all(self) -> list[SystemPrompt]:
        async with self.db.session() as session:
            result = await session.execute(select(SystemPrompt).order_by(SystemPrompt.name))
            return list(result.scalars().all())

    async def create(self, name: str, content: str) -> SystemPrompt:
        async with self.db.session() as session:
            prompt = SystemPrompt(name=name, content=content)
            session.add(prompt)
            await session.flush()
            return prompt

    async def update(self, prompt_id: int, content: str) -> None:
        async with self.db.session() as session:
            prompt = await session.get(SystemPrompt, prompt_id)
            if prompt:
                prompt.content = content

    async def seed_default(self) -> SystemPrompt:
        """Create the default prompt if it doesn't exist. Returns it either way."""
        existing = await self.get_by_name("default")
        if existing:
            return existing
        return await self.create("default", DEFAULT_PROMPT)
