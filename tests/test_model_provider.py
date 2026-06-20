"""Tests for model provider resolution and metadata extraction."""

from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Content
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agent_runtime.agents.model_provider import (
    OpenRouterChatCompletionsModel,
    _copy_reasoning_details_to_delta_reasoning,
    extract_thinking,
    resolve_runtime_model,
    serialize_usage,
    supported_models_payload,
)


def test_resolve_openrouter_model_per_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    glm = resolve_runtime_model(provider="openrouter", model="z-ai/glm-5.2")
    qwen = resolve_runtime_model(provider="openrouter", model="qwen/qwen3.7-max")

    assert glm.provider == "openrouter"
    assert glm.model_name == "z-ai/glm-5.2"
    assert qwen.model_name == "qwen/qwen3.7-max"
    assert isinstance(glm.model, OpenRouterChatCompletionsModel)
    assert isinstance(qwen.model, OpenRouterChatCompletionsModel)
    assert glm.model is not qwen.model


def test_resolve_openrouter_model_from_model_id(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("AGENT_RUNTIME_MODEL_PROVIDER", "openai")

    resolved = resolve_runtime_model(model="deepseek/deepseek-v4-flash")

    assert resolved.provider == "openrouter"
    assert resolved.model_name == "deepseek/deepseek-v4-flash"


def test_resolve_openrouter_rejects_unsupported_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    try:
        resolve_runtime_model(provider="openrouter", model="not/a-model")
    except ValueError as exc:
        assert "Unsupported OpenRouter model" in str(exc)
    else:
        raise AssertionError("unsupported OpenRouter model should fail")


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


def test_supported_models_payload_includes_required_openrouter_models():
    models = {item["id"] for item in supported_models_payload()["openrouter"]["models"]}

    assert {
        "z-ai/glm-5.2",
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-plus",
        "moonshotai/kimi-k2.7-code",
        "minimax/minimax-m3",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
    }.issubset(models)
