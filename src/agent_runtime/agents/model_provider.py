"""Model provider resolution and response metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

from agents import AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel
from agents.items import ModelResponse
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import ModelTracing
from agents.usage import Usage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Content, Summary

from agent_runtime.config import Settings

SUPPORTED_OPENROUTER_MODELS: dict[str, str] = {
    "z-ai/glm-5.2": "Z.ai: GLM 5.2",
    "qwen/qwen3.7-max": "Qwen: Qwen3.7 Max",
    "qwen/qwen3.7-plus": "Qwen: Qwen3.7 Plus",
    "moonshotai/kimi-k2.7-code": "MoonshotAI: Kimi K2.7 Code",
    "minimax/minimax-m3": "MiniMax: MiniMax M3",
    "deepseek/deepseek-v4-pro": "DeepSeek: DeepSeek V4 Pro",
    "deepseek/deepseek-v4-flash": "DeepSeek: DeepSeek V4 Flash",
}

ProviderName = Literal["openai", "openrouter"]


@dataclass(frozen=True)
class RuntimeModelConfig:
    provider: ProviderName
    model_name: str
    model: str | OpenAIChatCompletionsModel
    model_settings: ModelSettings
    tracing_disabled: bool = False


class _ReasoningDetailsStream:
    """Adapt OpenRouter reasoning_details chunks to SDK reasoning deltas."""

    def __init__(self, stream: Any):
        self._stream = stream

    async def __aiter__(self):
        async for chunk in self._stream:
            _copy_reasoning_details_to_delta_reasoning(chunk)
            yield chunk


class OpenRouterChatCompletionsModel(OpenAIChatCompletionsModel):
    """OpenRouter Chat Completions model with provider thinking preservation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_chat_completion: ChatCompletion | None = None

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any | None,
        handoffs: list[Any],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> ModelResponse:
        model_response = await super().get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

        response = self._last_chat_completion
        reasoning_item = _reasoning_item_from_chat_completion(response, self.model)
        if reasoning_item is not None:
            model_response.output.insert(0, reasoning_item)
        return model_response

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        response = await super()._fetch_response(*args, **kwargs)
        if kwargs.get("stream"):
            response_head, stream = response
            return response_head, _ReasoningDetailsStream(stream)
        self._last_chat_completion = cast(ChatCompletion, response)
        return response


def resolve_runtime_model(
    *,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> RuntimeModelConfig:
    """Resolve provider/model for a single agent run."""
    settings = Settings()
    resolved_provider = _resolve_provider(settings, provider, model)
    resolved_model = _resolve_model(settings, resolved_provider, model)

    if resolved_provider == "openai":
        return RuntimeModelConfig(
            provider="openai",
            model_name=resolved_model,
            model=resolved_model,
            model_settings=ModelSettings(),
        )

    if not settings.openrouter_api_key.strip():
        raise ValueError("OPENROUTER_API_KEY is required when using the openrouter provider")

    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    model_settings = _openrouter_model_settings(settings, reasoning_effort)
    return RuntimeModelConfig(
        provider="openrouter",
        model_name=resolved_model,
        model=OpenRouterChatCompletionsModel(
            model=resolved_model,
            openai_client=client,
        ),
        model_settings=model_settings,
        tracing_disabled=True,
    )


def supported_models_payload() -> dict[str, Any]:
    """Return the local supported model registry for API callers."""
    settings = Settings()
    return {
        "default_provider": settings.agent_runtime_model_provider,
        "openai": {
            "default_model": settings.openai_model,
            "models": [settings.openai_model],
        },
        "openrouter": {
            "default_model": settings.openrouter_model,
            "models": [
                {"id": model_id, "name": name}
                for model_id, name in SUPPORTED_OPENROUTER_MODELS.items()
            ],
        },
    }


def serialize_usage(usage: Usage | None) -> dict[str, Any] | None:
    """Serialize SDK usage into the API's stable token summary."""
    if usage is None:
        return None

    cached_tokens = _detail_value(usage.input_tokens_details, "cached_tokens")
    reasoning_tokens = _detail_value(usage.output_tokens_details, "reasoning_tokens")
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "request_usage_entries": [
            {
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "total_tokens": entry.total_tokens,
                "cached_tokens": _detail_value(entry.input_tokens_details, "cached_tokens"),
                "reasoning_tokens": _detail_value(
                    entry.output_tokens_details, "reasoning_tokens"
                ),
            }
            for entry in usage.request_usage_entries
        ],
    }


def extract_thinking(raw_responses: Iterable[ModelResponse]) -> dict[str, Any] | None:
    """Extract provider-returned thinking content from SDK model responses."""
    reasoning_texts: list[str] = []
    summaries: list[str] = []
    reasoning_details: list[Any] = []
    encrypted_content: list[str] = []

    for response in raw_responses:
        for item in response.output:
            if getattr(item, "type", None) != "reasoning":
                continue

            for content in getattr(item, "content", None) or []:
                text = getattr(content, "text", None)
                if text:
                    reasoning_texts.append(text)

            for summary in getattr(item, "summary", None) or []:
                text = getattr(summary, "text", None)
                if text:
                    summaries.append(text)

            details = getattr(item, "reasoning_details", None)
            if details:
                reasoning_details.append(_jsonable(details))

            encrypted = getattr(item, "encrypted_content", None)
            if encrypted:
                encrypted_content.append(encrypted)

    if not any((reasoning_texts, summaries, reasoning_details, encrypted_content)):
        return None

    return {
        "reasoning": "\n".join(reasoning_texts) if reasoning_texts else None,
        "summary": "\n".join(summaries) if summaries else None,
        "reasoning_details": reasoning_details or None,
        "encrypted_content": encrypted_content or None,
    }


def _resolve_provider(settings: Settings, provider: str | None, model: str | None) -> ProviderName:
    if provider:
        normalized = provider.lower()
    elif model in SUPPORTED_OPENROUTER_MODELS:
        normalized = "openrouter"
    else:
        normalized = settings.agent_runtime_model_provider.lower()

    if normalized not in ("openai", "openrouter"):
        raise ValueError(f"Unsupported model provider: {provider or normalized}")
    return cast(ProviderName, normalized)


def _resolve_model(settings: Settings, provider: ProviderName, model: str | None) -> str:
    resolved = model or (
        settings.openrouter_model if provider == "openrouter" else settings.openai_model
    )
    if provider == "openrouter" and resolved not in SUPPORTED_OPENROUTER_MODELS:
        supported = ", ".join(SUPPORTED_OPENROUTER_MODELS)
        raise ValueError(f"Unsupported OpenRouter model: {resolved}. Supported models: {supported}")
    return resolved


def _openrouter_model_settings(
    settings: Settings,
    reasoning_effort: str | None,
) -> ModelSettings:
    effort = (reasoning_effort or settings.openrouter_reasoning_effort).strip()
    reasoning: dict[str, Any] = {"exclude": False}
    if effort:
        reasoning["effort"] = effort
    return ModelSettings(
        include_usage=True,
        extra_body={"reasoning": reasoning},
    )


def _detail_value(details: Any, field: str) -> int:
    value = getattr(details, field, None)
    return value if isinstance(value, int) else 0


def _reasoning_item_from_chat_completion(
    response: ChatCompletion | None,
    model: str,
) -> ResponseReasoningItem | None:
    if response is None or not response.choices:
        return None

    message = response.choices[0].message
    thinking = _thinking_from_message(message)
    if thinking is None:
        return None

    text = thinking.get("reasoning") or _reasoning_details_to_text(
        thinking.get("reasoning_details")
    )
    reasoning_kwargs: dict[str, Any] = {
        "id": FAKE_RESPONSES_ID,
        "summary": [],
        "type": "reasoning",
        "provider_data": {"model": model},
    }
    if text:
        reasoning_kwargs["content"] = [Content(text=text, type="reasoning_text")]
    if thinking.get("reasoning_details"):
        reasoning_kwargs["reasoning_details"] = thinking["reasoning_details"]
    if thinking.get("encrypted_content"):
        reasoning_kwargs["encrypted_content"] = thinking["encrypted_content"]
    if thinking.get("summary"):
        reasoning_kwargs["summary"] = [Summary(text=thinking["summary"], type="summary_text")]
    return ResponseReasoningItem(**reasoning_kwargs)


def _thinking_from_message(message: Any) -> dict[str, Any] | None:
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    reasoning_details = getattr(message, "reasoning_details", None)
    thinking_blocks = getattr(message, "thinking_blocks", None)

    thinking_texts: list[str] = []
    encrypted_content: list[str] = []
    if thinking_blocks:
        for block in thinking_blocks:
            if isinstance(block, dict):
                if block.get("thinking"):
                    thinking_texts.append(block["thinking"])
                if block.get("signature"):
                    encrypted_content.append(block["signature"])

    if not any((reasoning, reasoning_details, thinking_texts, encrypted_content)):
        return None

    return {
        "reasoning": reasoning or ("\n".join(thinking_texts) if thinking_texts else None),
        "reasoning_details": _jsonable(reasoning_details) if reasoning_details else None,
        "encrypted_content": "\n".join(encrypted_content) if encrypted_content else None,
    }


def _copy_reasoning_details_to_delta_reasoning(chunk: ChatCompletionChunk) -> None:
    for choice in chunk.choices or []:
        delta = choice.delta
        if delta is None or getattr(delta, "reasoning", None):
            continue

        details = getattr(delta, "reasoning_details", None)
        text = _reasoning_details_to_text(details)
        if text:
            delta.reasoning = text


def _reasoning_details_to_text(details: Any) -> str | None:
    if not details:
        return None

    texts: list[str] = []
    detail_items = details if isinstance(details, list) else [details]
    for detail in detail_items:
        if hasattr(detail, "model_dump"):
            detail = detail.model_dump()
        if not isinstance(detail, dict):
            continue
        for key in ("text", "thinking", "content"):
            value = detail.get(key)
            if isinstance(value, str) and value:
                texts.append(value)
    return "\n".join(texts) if texts else None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
