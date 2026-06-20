"""Tests for the runtime model registry repository."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Base
from agent_runtime.db.runtime_model_repo import RuntimeModelRepo


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    database = Database("sqlite+aiosqlite:///:memory:")
    database.engine = engine
    yield database
    await engine.dispose()


@pytest.mark.asyncio
async def test_seed_defaults_is_idempotent(db):
    repo = RuntimeModelRepo(db)
    await repo.seed_defaults()
    await repo.seed_defaults()

    rows = await repo.list_all()
    model_ids = [row.model_id for row in rows if row.provider == "openrouter"]

    assert len(model_ids) == len(set(model_ids))
    assert "z-ai/glm-5.2" in model_ids
    assert "deepseek/deepseek-v4-flash" in model_ids


@pytest.mark.asyncio
async def test_create_update_delete_runtime_model(db):
    repo = RuntimeModelRepo(db)

    created = await repo.create(
        provider="openrouter",
        model_id="vendor/custom",
        name="Vendor Custom",
        enabled=True,
        supports_reasoning=False,
        sort_order=99,
        config_json={"tier": "test"},
    )

    fetched = await repo.get_by_provider_model("openrouter", "vendor/custom")
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.config_json == {"tier": "test"}

    updated = await repo.update(
        created.id,
        name="Vendor Custom Updated",
        enabled=False,
        supports_reasoning=True,
        sort_order=100,
        config_json={"tier": "prod"},
        replace_config=True,
    )
    assert updated is not None
    assert updated.name == "Vendor Custom Updated"
    assert updated.enabled is False
    assert updated.supports_reasoning is True
    assert updated.sort_order == 100
    assert updated.config_json == {"tier": "prod"}

    assert await repo.get_by_provider_model(
        "openrouter", "vendor/custom", enabled_only=True
    ) is None
    assert await repo.delete(created.id) is True
    assert await repo.get_by_provider_model("openrouter", "vendor/custom") is None


@pytest.mark.asyncio
async def test_list_enabled_filters_and_orders(db):
    repo = RuntimeModelRepo(db)
    await repo.create(provider="openrouter", model_id="vendor/b", name="B", sort_order=20)
    await repo.create(provider="openrouter", model_id="vendor/a", name="A", sort_order=10)
    disabled = await repo.create(
        provider="openrouter",
        model_id="vendor/disabled",
        name="Disabled",
        enabled=False,
        sort_order=0,
    )

    rows = await repo.list_enabled()

    assert disabled.id not in {row.id for row in rows}
    assert [row.model_id for row in rows] == ["vendor/a", "vendor/b"]
