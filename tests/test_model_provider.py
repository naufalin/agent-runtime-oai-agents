"""Tests for model provider resolution and metadata extraction."""

import pytest
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Content
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from sqlalchemy.ext.asyncio import create_async_engine

from agent_runtime.agents.model_provider import (
    OpenRouterChatCompletionsModel,
    _copy_reasoning_details_to_delta_reasoning,
    extract_thinking,
    resolve_runtime_model,
    serialize_usage,
    supported_models_payload,
)
from agent_runtime.db.connection import Database
from agent_runtime.db.models import Base
from agent_runtime.db.runtime_model_repo import RuntimeModelRepo


@pytest.fixture
async def model_repo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    database = Database("sqlite+aiosqlite:///:memory:")
    database.engine = engine
    repo = RuntimeModelRepo(database)
    await repo.seed_defaults()
    yield repo
    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_openrouter_model_per_request(monkeypatch, model_repo):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    glm = await resolve_runtime_model(
        model_repo=model_repo,
        provider="openrouter",
        model="z-ai/glm-5.2",
    )
    qwen = await resolve_runtime_model(
        model_repo=model_repo,
        provider="openrouter",
        model="qwen/qwen3.7-max",
    )

    assert glm.provider == "openrouter"
    assert glm.model_name == "z-ai/glm-5.2"
    assert qwen.model_name == "qwen/qwen3.7-max"
    assert isinstance(glm.model, OpenRouterChatCompletionsModel)
    assert isinstance(qwen.model, OpenRouterChatCompletionsModel)
    assert glm.model is not qwen.model


@pytest.mark.asyncio
async def test_resolve_openrouter_model_from_model_id(monkeypatch, model_repo):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("AGENT_RUNTIME_MODEL_PROVIDER", "openai")

    resolved = await resolve_runtime_model(
        model_repo=model_repo,
        model="deepseek/deepseek-v4-flash",
    )

    assert resolved.provider == "openrouter"
    assert resolved.model_name == "deepseek/deepseek-v4-flash"


@pytest.mark.asyncio
async def test_resolve_openai_model_applies_reasoning_effort(model_repo):
    resolved = await resolve_runtime_model(
        model_repo=model_repo,
        provider="openai",
        reasoning_effort="high",
    )

    assert resolved.provider == "openai"
    assert resolved.model_settings.reasoning is not None
    assert resolved.model_settings.reasoning.effort == "high"
    assert resolved.model_settings.reasoning.summary == "auto"


@pytest.mark.asyncio
async def test_resolve_openrouter_rejects_unsupported_model(monkeypatch, model_repo):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    try:
        await resolve_runtime_model(
            model_repo=model_repo,
            provider="openrouter",
            model="not/a-model",
        )
    except ValueError as exc:
        assert "Unsupported or disabled openrouter model" in str(exc)
    else:
        raise AssertionError("unsupported OpenRouter model should fail")


@pytest.mark.asyncio
async def test_resolve_openrouter_rejects_disabled_model(monkeypatch, model_repo):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    row = await model_repo.get_by_provider_model("openrouter", "z-ai/glm-5.2")
    assert row is not None
    await model_repo.update(row.id, enabled=False)

    with pytest.raises(ValueError, match="Unsupported or disabled openrouter model"):
        await resolve_runtime_model(
            model_repo=model_repo,
            provider="openrouter",
            model="z-ai/glm-5.2",
        )


def test_serialize_usage_includes_reasoning_tokens():
    request_usage = Usage(
        requests=1,
        input_tokens=10,
        output_tokens=7,
        total_tokens=17,
        input_tokens_details=InputTokensDetails(cached_tokens=3),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=4),
    )
    usage = Usage()
    usage.add(request_usage)

    result = serialize_usage(usage)

    assert result == {
        "requests": 1,
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
        "cached_tokens": 3,
        "reasoning_tokens": 4,
        "request_usage_entries": [
            {
                "input_tokens": 10,
                "output_tokens": 7,
                "total_tokens": 17,
                "cached_tokens": 3,
                "reasoning_tokens": 4,
            }
        ],
    }


def test_extract_thinking_from_reasoning_item():
    reasoning_item = ResponseReasoningItem(
        id="rs_123",
        type="reasoning",
        summary=[],
        content=[Content(text="visible provider reasoning", type="reasoning_text")],
        reasoning_details=[{"type": "reasoning.text", "text": "detail"}],
    )
    response = ModelResponse(output=[reasoning_item], usage=Usage(), response_id=None)

    thinking = extract_thinking([response])

    assert thinking is not None
    assert thinking["reasoning"] == "visible provider reasoning"
    assert thinking["reasoning_details"] == [[{"type": "reasoning.text", "text": "detail"}]]


def test_reasoning_details_stream_chunks_are_mirrored_to_reasoning_delta():
    chunk = ChatCompletionChunk(
        id="chunk-1",
        choices=[
            Choice(
                delta=ChoiceDelta(reasoning_details=[{"text": "streamed reasoning"}]),
                index=0,
                finish_reason=None,
            )
        ],
        created=1,
        model="z-ai/glm-5.2",
        object="chat.completion.chunk",
    )

    _copy_reasoning_details_to_delta_reasoning(chunk)

    assert chunk.choices[0].delta.reasoning == "streamed reasoning"


@pytest.mark.asyncio
async def test_supported_models_payload_includes_required_openrouter_models(model_repo):
    models = {
        item["model_id"]
        for item in (await supported_models_payload(model_repo))["openrouter"]["models"]
    }

    assert {
        "z-ai/glm-5.2",
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-plus",
        "moonshotai/kimi-k2.7-code",
        "minimax/minimax-m3",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
    }.issubset(models)
