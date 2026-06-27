"""Unit tests for Langfuse tracing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import agent_runtime.tracing as tracing


class FakeObservation:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def __enter__(self) -> FakeObservation:
        return self

    def __exit__(self, *args: object) -> None:
        self.ended = True

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []
        self.current_observations: list[FakeObservation] = []
        self.flushed = False

    def create_trace_id(self) -> str:
        return "0" * 32

    def start_as_current_observation(self, **kwargs: Any) -> FakeObservation:
        observation = FakeObservation(**kwargs)
        self.current_observations.append(observation)
        return observation

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        observation = FakeObservation(**kwargs)
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        self.flushed = True


@pytest.fixture(autouse=True)
def reset_langfuse_state():
    original_initialized = tracing._langfuse_initialized
    original_available = tracing._langfuse_available
    original_client = tracing._langfuse_client
    tracing._langfuse_initialized = True
    tracing._langfuse_available = True
    tracing._langfuse_client = FakeLangfuseClient()
    yield
    tracing._langfuse_initialized = original_initialized
    tracing._langfuse_available = original_available
    tracing._langfuse_client = original_client


@pytest.mark.asyncio
async def test_langfuse_trace_sets_root_and_propagated_attributes(monkeypatch):
    propagated: list[dict[str, Any]] = []

    @contextmanager
    def fake_propagate_attributes(**kwargs: Any):
        propagated.append(kwargs)
        yield

    import langfuse

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate_attributes)

    async with tracing.langfuse_trace(
        name="agent-run",
        input_data="hello",
        session_id="75",
        user_id="user-1",
        metadata={"provider": "openrouter", "model": "minimax/minimax-m3"},
        tags=["agent-runtime"],
    ) as span:
        span.update(output="world")

    client = tracing._langfuse_client
    root = client.current_observations[0]
    assert root.kwargs["name"] == "agent-run"
    assert root.kwargs["as_type"] == "agent"
    assert root.kwargs["input"] == "hello"
    assert root.kwargs["metadata"]["session_id"] == "75"
    assert root.kwargs["metadata"]["provider"] == "openrouter"
    assert root.updates == [{"output": "world"}]
    assert root.ended is True
    assert client.flushed is True
    assert propagated == [
        {
            "trace_name": "agent-run",
            "metadata": {"provider": "openrouter", "model": "minimax/minimax-m3"},
            "session_id": "75",
            "user_id": "user-1",
            "tags": ["agent-runtime"],
        }
    ]


def test_log_generation_creates_generation_with_usage_details():
    tracing.log_generation(
        name="agent-llm-call",
        model="minimax/minimax-m3",
        input_data=[{"role": "user", "content": "hi"}],
        output="hello",
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cached_tokens": 3,
            "reasoning_tokens": 2,
        },
        metadata={"provider": "openrouter"},
        model_parameters={"temperature": 0.2},
    )

    observation = tracing._langfuse_client.observations[0]
    assert observation.kwargs["name"] == "agent-llm-call"
    assert observation.kwargs["as_type"] == "generation"
    assert observation.kwargs["model"] == "minimax/minimax-m3"
    assert observation.kwargs["usage_details"] == {
        "input": 10,
        "output": 5,
        "total": 15,
        "input_cached": 3,
        "output_reasoning": 2,
    }
    assert observation.kwargs["model_parameters"] == {"temperature": 0.2}
    assert observation.ended is True


def test_tool_observation_updates_and_ends_once():
    tool = tracing.start_tool_observation(
        "web_search",
        input_data={"query": "world cup"},
        metadata={"call_id": None},
    )

    tracing.finish_tool_observation(
        tool,
        output={"results": []},
        metadata={"output_preview": "{'results': []}"},
    )

    assert tool.kwargs["name"] == "web_search"
    assert tool.kwargs["as_type"] == "tool"
    assert tool.kwargs["input"] == {"query": "world cup"}
    assert tool.updates == [
        {
            "output": {"results": []},
            "metadata": {"output_preview": "{'results': []}"},
        }
    ]
    assert tool.ended is True


def test_mark_observation_error_updates_level_and_metadata():
    observation = FakeObservation()

    tracing.mark_observation_error(observation, RuntimeError("stream broke"))

    assert observation.updates == [
        {
            "level": "ERROR",
            "metadata": {
                "error": "stream broke",
                "status": "error",
            },
        }
    ]
