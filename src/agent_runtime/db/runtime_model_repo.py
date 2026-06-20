"""Runtime model registry repository and default seeding."""

from typing import Any

from sqlalchemy import delete, select

from agent_runtime.config import Settings
from agent_runtime.db.connection import Database
from agent_runtime.db.models import RuntimeModel

DEFAULT_OPENROUTER_MODELS: tuple[dict[str, Any], ...] = (
    {
        "provider": "openrouter",
        "model_id": "z-ai/glm-5.2",
        "name": "Z.ai: GLM 5.2",
        "supports_reasoning": True,
        "sort_order": 10,
    },
    {
        "provider": "openrouter",
        "model_id": "qwen/qwen3.7-max",
        "name": "Qwen: Qwen3.7 Max",
        "supports_reasoning": True,
        "sort_order": 20,
    },
    {
        "provider": "openrouter",
        "model_id": "qwen/qwen3.7-plus",
        "name": "Qwen: Qwen3.7 Plus",
        "supports_reasoning": True,
        "sort_order": 30,
    },
    {
        "provider": "openrouter",
        "model_id": "moonshotai/kimi-k2.7-code",
        "name": "MoonshotAI: Kimi K2.7 Code",
        "supports_reasoning": True,
        "sort_order": 40,
    },
    {
        "provider": "openrouter",
        "model_id": "minimax/minimax-m3",
        "name": "MiniMax: MiniMax M3",
        "supports_reasoning": True,
        "sort_order": 50,
    },
    {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v4-pro",
        "name": "DeepSeek: DeepSeek V4 Pro",
        "supports_reasoning": True,
        "sort_order": 60,
    },
    {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v4-flash",
        "name": "DeepSeek: DeepSeek V4 Flash",
        "supports_reasoning": True,
        "sort_order": 70,
    },
)


class RuntimeModelRepo:
    def __init__(self, db: Database):
        self.db = db

    async def list_all(self) -> list[RuntimeModel]:
        async with self.db.session() as session:
            result = await session.execute(
                select(RuntimeModel).order_by(
                    RuntimeModel.provider,
                    RuntimeModel.sort_order,
                    RuntimeModel.name,
                )
            )
            return list(result.scalars().all())

    async def list_enabled(self) -> list[RuntimeModel]:
        async with self.db.session() as session:
            result = await session.execute(
                select(RuntimeModel)
                .where(RuntimeModel.enabled.is_(True))
                .order_by(RuntimeModel.provider, RuntimeModel.sort_order, RuntimeModel.name)
            )
            return list(result.scalars().all())

    async def get_by_id(self, model_row_id: int) -> RuntimeModel | None:
        async with self.db.session() as session:
            return await session.get(RuntimeModel, model_row_id)

    async def get_by_provider_model(
        self,
        provider: str,
        model_id: str,
        *,
        enabled_only: bool = False,
    ) -> RuntimeModel | None:
        async with self.db.session() as session:
            statement = select(RuntimeModel).where(
                RuntimeModel.provider == provider,
                RuntimeModel.model_id == model_id,
            )
            if enabled_only:
                statement = statement.where(RuntimeModel.enabled.is_(True))
            result = await session.execute(statement)
            return result.scalar_one_or_none()

    async def create(
        self,
        *,
        provider: str,
        model_id: str,
        name: str,
        enabled: bool = True,
        supports_reasoning: bool = False,
        sort_order: int = 0,
        config_json: dict[str, Any] | None = None,
    ) -> RuntimeModel:
        async with self.db.session() as session:
            model = RuntimeModel(
                provider=provider,
                model_id=model_id,
                name=name,
                enabled=enabled,
                supports_reasoning=supports_reasoning,
                sort_order=sort_order,
                config_json=config_json,
            )
            session.add(model)
            await session.flush()
            return model

    async def update(
        self,
        model_row_id: int,
        *,
        name: str | None = None,
        enabled: bool | None = None,
        supports_reasoning: bool | None = None,
        sort_order: int | None = None,
        config_json: dict[str, Any] | None = None,
        replace_config: bool = False,
    ) -> RuntimeModel | None:
        async with self.db.session() as session:
            model = await session.get(RuntimeModel, model_row_id)
            if model is None:
                return None
            if name is not None:
                model.name = name
            if enabled is not None:
                model.enabled = enabled
            if supports_reasoning is not None:
                model.supports_reasoning = supports_reasoning
            if sort_order is not None:
                model.sort_order = sort_order
            if replace_config:
                model.config_json = config_json
            await session.flush()
            return model

    async def delete(self, model_row_id: int) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                delete(RuntimeModel).where(RuntimeModel.id == model_row_id)
            )
            return result.rowcount > 0

    async def seed_defaults(self) -> None:
        settings = Settings()
        defaults = (
            {
                "provider": "openai",
                "model_id": settings.openai_model,
                "name": settings.openai_model,
                "supports_reasoning": False,
                "sort_order": 0,
            },
            *DEFAULT_OPENROUTER_MODELS,
        )

        for entry in defaults:
            existing = await self.get_by_provider_model(entry["provider"], entry["model_id"])
            if existing:
                continue
            await self.create(**entry)
