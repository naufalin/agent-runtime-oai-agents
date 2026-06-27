"""Tests for the main agent definition."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from agents.usage import Usage

from agent_runtime.agents.runtime import (
    AgentFactory,
    AgentResponse,
    create_agent,
    run_agent,
    run_agent_streamed,
)
from agent_runtime.db.models import Message, RuntimeModel, Session, SystemPrompt
from agent_runtime.ids import encode


class FakeTraceSpan:
    def __init__(self) -> None:
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def test_create_agent_has_all_tools():
    agent = create_agent("test prompt")
    assert len(agent.tools) == 6


def test_create_agent_has_name():
    agent = create_agent("test prompt")
    assert agent.name == "MainAgent"


def test_create_agent_uses_prompt():
    agent = create_agent("You are a pirate.")
    assert agent.instructions == "You are a pirate."


@pytest.mark.asyncio
async def test_run_agent_inserts_system_message_for_new_session(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = None
    mock_repo.create_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.add_message.return_value = Message(id=2, role="user", content="Hi")

    mock_prompt_repo.seed_default.return_value = SystemPrompt(
        id=1,
        name="default",
        content="Default prompt",
    )
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    mock_result = AsyncMock()
    mock_result.final_output = "Hello!"
    mock_result.context_wrapper.usage = Usage()
    mock_result.raw_responses = []

    with patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result) as mock_run:
        result = await run_agent(
            "Hi there",
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
            model_repo=mock_model_repo,
            agent_factory=AgentFactory(),
        )

        assert isinstance(result, AgentResponse)
        assert result.response == "Hello!"
        mock_prompt_repo.seed_default.assert_called_once()
        mock_repo.create_session.assert_called_once()
        assert mock_repo.add_message.call_count == 3
        final_message_kwargs = mock_repo.add_message.call_args.kwargs
        assert final_message_kwargs["provider"] == "openai"
        assert final_message_kwargs["model"] == "gpt-5.4-mini"
        assert final_message_kwargs["usage_json"]["total_tokens"] == 0
        # Verify hooks and context were passed
        call_kwargs = mock_run.call_args
        assert "hooks" in call_kwargs.kwargs
        assert "context" in call_kwargs.kwargs
        assert call_kwargs.kwargs["run_config"].tracing_disabled is False


@pytest.mark.asyncio
async def test_run_agent_uses_latest_system_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=5,
        session_id=1,
        role="system",
        content="You are a pirate.",
    )
    mock_repo.add_message.return_value = Message(id=6, role="user", content="Ahoy")

    mock_result = AsyncMock()
    mock_result.final_output = "Arr!"
    mock_result.context_wrapper.usage = Usage()
    mock_result.raw_responses = []
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    with patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result):
        result = await run_agent(
            "Ahoy",
            session_id=encode(1),
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
            model_repo=mock_model_repo,
            agent_factory=AgentFactory(),
        )

        assert result.response == "Arr!"
        mock_prompt_repo.seed_default.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_validates_model_before_persisting_user_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openrouter",
        model_id="deepseek/deepseek-v4-flash",
        name="DeepSeek: DeepSeek V4 Flash",
        enabled=True,
    )

    with (
        patch("agent_runtime.agents.runtime.Runner.run") as mock_run,
        pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"),
    ):
        await run_agent(
            "Hello",
            session_id=encode(1),
            provider="openrouter",
            model="deepseek/deepseek-v4-flash",
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
            model_repo=mock_model_repo,
            agent_factory=AgentFactory(),
        )

    mock_repo.add_message.assert_not_called()
    mock_repo.create_session.assert_not_called()
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_streamed_validates_model_before_persisting_user_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openrouter",
        model_id="deepseek/deepseek-v4-flash",
        name="DeepSeek: DeepSeek V4 Flash",
        enabled=True,
    )

    with (
        patch("agent_runtime.agents.runtime.Runner.run_streamed") as mock_run_streamed,
        pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"),
    ):
        _ = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                provider="openrouter",
                model="deepseek/deepseek-v4-flash",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    mock_repo.add_message.assert_not_called()
    mock_repo.create_session.assert_not_called()
    mock_run_streamed.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_streamed_separates_thinking_deltas_from_text(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    AsyncMock()
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hello"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(usage=Usage())
        raw_responses = []

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=SimpleNamespace(
                    type="response.reasoning_text.delta",
                    delta="thinking first",
                )
            )
            yield RawResponsesStreamEvent(
                data=SimpleNamespace(
                    type="response.output_text.delta",
                    delta="Visible answer",
                )
            )
            yield RawResponsesStreamEvent(
                data=SimpleNamespace(
                    type="response.reasoning_summary_text.delta",
                    delta="summary bit",
                )
            )

    with patch(
        "agent_runtime.agents.runtime.Runner.run_streamed",
        return_value=FakeStreamedResult(),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    assert events[:3] == [
        {"type": "thinking_delta", "delta": "thinking first", "kind": "reasoning"},
        {"type": "text_delta", "delta": "Visible answer"},
        {"type": "thinking_delta", "delta": "summary bit", "kind": "summary"},
    ]
    assert events[3]["type"] == "done"

    final_message_args = mock_repo.add_message.call_args.args
    assert final_message_args[2] == "Visible answer"
    assert final_message_args[2] != "thinking firstVisible answersummary bit"


@pytest.mark.asyncio
async def test_run_agent_streamed_does_not_emit_mismatched_final_suffix(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hello"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(usage=Usage())
        raw_responses = []

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=SimpleNamespace(
                    type="response.output_text.delta",
                    delta="Visible streamed text.",
                )
            )
            yield RunItemStreamEvent(
                name="message_output_created",
                item=SimpleNamespace(),
            )

    with (
        patch(
            "agent_runtime.agents.runtime.Runner.run_streamed",
            return_value=FakeStreamedResult(),
        ),
        patch(
            "agents.items.ItemHelpers.text_message_output",
            return_value="Corrected final text.",
        ),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    assert [event for event in events if event["type"] == "text_delta"] == [
        {"type": "text_delta", "delta": "Visible streamed text."}
    ]
    final_message_args = mock_repo.add_message.call_args.args
    assert final_message_args[2] == "Corrected final text."


@pytest.mark.asyncio
async def test_run_agent_streamed_includes_tool_output_preview(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hello"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(usage=Usage())
        raw_responses = []

        async def stream_events(self):
            yield RunItemStreamEvent(
                name="tool_called",
                item=SimpleNamespace(
                    tool_name="web_search",
                    call_id="call-1",
                    raw_item=SimpleNamespace(arguments='{"query": "gold price"}'),
                ),
            )
            yield RunItemStreamEvent(
                name="tool_output",
                item=SimpleNamespace(
                    tool_name="web_search",
                    call_id="call-1",
                    output={"results": [{"title": "Gold price"}]},
                    raw_item=None,
                ),
            )

    with patch(
        "agent_runtime.agents.runtime.Runner.run_streamed",
        return_value=FakeStreamedResult(),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    tool_end = next(event for event in events if event["type"] == "tool_end")
    assert tool_end["output_preview"] == "{'results': [{'title': 'Gold price'}]}"


@pytest.mark.asyncio
async def test_run_agent_streamed_includes_visualization_html(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Make a chart"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    viz_html = "<!DOCTYPE html><html><body><div id='chart'></div></body></html>"

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(usage=Usage())
        raw_responses = []

        async def stream_events(self):
            yield RunItemStreamEvent(
                name="tool_called",
                item=SimpleNamespace(
                    tool_name="generate_visualization",
                    call_id="viz-1",
                    raw_item=SimpleNamespace(arguments='{"chart_type": "bar"}'),
                ),
            )
            yield RunItemStreamEvent(
                name="tool_output",
                item=SimpleNamespace(
                    tool_name="generate_visualization",
                    call_id="viz-1",
                    output=f'{{"html": "{viz_html}", "title": "Chart"}}',
                    raw_item=None,
                ),
            )

    with patch(
        "agent_runtime.agents.runtime.Runner.run_streamed",
        return_value=FakeStreamedResult(),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Make a chart",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    tool_end = next(event for event in events if event["type"] == "tool_end")
    assert tool_end["tool"] == "generate_visualization"
    assert tool_end["viz_html"] == viz_html


@pytest.mark.asyncio
async def test_run_agent_streamed_infers_generic_visualization_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Make a chart"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    viz_html = "<!DOCTYPE html><html><body><div id='chart'></div></body></html>"
    args = {
        "chart_type": "bar",
        "data": '{"series": [{"data": [1], "type": "bar"}]}',
        "title": "Chart",
    }

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(usage=Usage())
        raw_responses = []

        async def stream_events(self):
            yield RunItemStreamEvent(
                name="tool_called",
                item=SimpleNamespace(
                    tool_name=None,
                    call_id="viz-1",
                    raw_item=SimpleNamespace(arguments=json.dumps(args)),
                ),
            )
            yield RunItemStreamEvent(
                name="tool_output",
                item=SimpleNamespace(
                    tool_name=None,
                    call_id="viz-1",
                    output=json.dumps({"html": viz_html, "title": "Chart"}),
                    raw_item=None,
                ),
            )

    with patch(
        "agent_runtime.agents.runtime.Runner.run_streamed",
        return_value=FakeStreamedResult(),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Make a chart",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    assert events[0]["type"] == "tool_start"
    assert events[0]["tool"] == "generate_visualization"
    tool_end = next(event for event in events if event["type"] == "tool_end")
    assert tool_end["tool"] == "generate_visualization"
    assert tool_end["viz_html"] == viz_html
    assert mock_repo.add_message.call_args_list[-1].kwargs["tool_name"] == (
        "generate_visualization"
    )


@pytest.mark.asyncio
async def test_run_agent_streamed_infers_generic_web_search_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Search"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(usage=Usage())
        raw_responses = []

        async def stream_events(self):
            yield RunItemStreamEvent(
                name="tool_called",
                item=SimpleNamespace(
                    tool_name=None,
                    call_id="search-1",
                    raw_item=SimpleNamespace(
                        arguments=json.dumps({"query": "DKI Jakarta population", "location": "ID"})
                    ),
                ),
            )
            yield RunItemStreamEvent(
                name="tool_output",
                item=SimpleNamespace(
                    tool_name=None,
                    call_id="search-1",
                    output="[DKI Jakarta](https://example.com)\n  population data",
                    raw_item=None,
                ),
            )

    with patch(
        "agent_runtime.agents.runtime.Runner.run_streamed",
        return_value=FakeStreamedResult(),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Search",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    assert events[0]["type"] == "tool_start"
    assert events[0]["tool"] == "web_search"
    tool_end = next(event for event in events if event["type"] == "tool_end")
    assert tool_end["tool"] == "web_search"
    assert mock_repo.add_message.call_args_list[-1].kwargs["tool_name"] == "web_search"


@pytest.mark.asyncio
async def test_run_agent_logs_generation_and_root_output(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()
    span = FakeTraceSpan()

    @asynccontextmanager
    async def fake_langfuse_trace(*args, **kwargs):
        yield span

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hello"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )
    mock_result = AsyncMock()
    mock_result.final_output = "Hello back"
    mock_result.context_wrapper.usage = Usage(input_tokens=3, output_tokens=2, total_tokens=5)
    mock_result.raw_responses = []

    with (
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
        patch("agent_runtime.agents.runtime.langfuse_trace", fake_langfuse_trace),
        patch("agent_runtime.agents.runtime.log_generation") as log_generation,
    ):
        result = await run_agent(
            "Hello",
            session_id=encode(1),
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
            model_repo=mock_model_repo,
            agent_factory=AgentFactory(),
        )

    assert result.response == "Hello back"
    assert span.updates == [{"output": "Hello back"}]
    log_generation.assert_called_once()
    assert log_generation.call_args.kwargs["output"] == "Hello back"
    assert log_generation.call_args.kwargs["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_run_agent_streamed_traces_generation_tool_and_root(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()
    span = FakeTraceSpan()
    generation = object()
    tool_observation = object()

    @asynccontextmanager
    async def fake_langfuse_trace(*args, **kwargs):
        yield span

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hello"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FakeStreamedResult:
        context_wrapper = SimpleNamespace(
            usage=Usage(input_tokens=4, output_tokens=6, total_tokens=10)
        )
        raw_responses = []

        async def stream_events(self):
            yield RunItemStreamEvent(
                name="tool_called",
                item=SimpleNamespace(
                    tool_name="web_search",
                    call_id="call-1",
                    raw_item=SimpleNamespace(arguments='{"query": "world cup"}'),
                ),
            )
            yield RunItemStreamEvent(
                name="tool_output",
                item=SimpleNamespace(
                    tool_name="web_search",
                    call_id="call-1",
                    output={"results": [{"title": "World Cup"}]},
                    raw_item=None,
                ),
            )
            yield RawResponsesStreamEvent(
                data=SimpleNamespace(type="response.output_text.delta", delta="Final answer")
            )

    with (
        patch(
            "agent_runtime.agents.runtime.Runner.run_streamed",
            return_value=FakeStreamedResult(),
        ),
        patch("agent_runtime.agents.runtime.langfuse_trace", fake_langfuse_trace),
        patch(
            "agent_runtime.agents.runtime.start_generation",
            return_value=generation,
        ) as start_gen,
        patch("agent_runtime.agents.runtime.finish_generation") as finish_gen,
        patch(
            "agent_runtime.agents.runtime.start_tool_observation",
            return_value=tool_observation,
        ) as start_tool,
        patch("agent_runtime.agents.runtime.finish_tool_observation") as finish_tool,
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    assert [event["type"] for event in events] == ["tool_start", "tool_end", "text_delta", "done"]
    assert events[1]["output_preview"] == "{'results': [{'title': 'World Cup'}]}"
    start_gen.assert_called_once()
    assert start_gen.call_args.kwargs["input_data"] == [{"role": "user", "content": "Hello"}]
    start_tool.assert_called_once()
    assert start_tool.call_args.kwargs["input_data"] == {"query": "world cup"}
    finish_tool.assert_called_once()
    assert finish_tool.call_args.kwargs["output"] == {"results": [{"title": "World Cup"}]}
    finish_gen.assert_called_once()
    assert finish_gen.call_args.args[0] is generation
    assert finish_gen.call_args.kwargs["output"] == "Final answer"
    assert finish_gen.call_args.kwargs["usage"]["total_tokens"] == 10
    assert span.updates == [{"output": "Final answer"}]


@pytest.mark.asyncio
async def test_run_agent_streamed_traces_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()
    span = FakeTraceSpan()
    generation = object()
    tool_observation = object()

    @asynccontextmanager
    async def fake_langfuse_trace(*args, **kwargs):
        yield span

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default prompt",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hello"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FailingAfterToolStartResult:
        async def stream_events(self):
            yield RunItemStreamEvent(
                name="tool_called",
                item=SimpleNamespace(
                    tool_name="web_search",
                    call_id="call-1",
                    raw_item=SimpleNamespace(arguments='{"query": "world cup"}'),
                ),
            )
            raise RuntimeError("stream broke")

    with (
        patch(
            "agent_runtime.agents.runtime.Runner.run_streamed",
            return_value=FailingAfterToolStartResult(),
        ),
        patch("agent_runtime.agents.runtime.langfuse_trace", fake_langfuse_trace),
        patch("agent_runtime.agents.runtime.start_generation", return_value=generation),
        patch(
            "agent_runtime.agents.runtime.start_tool_observation",
            return_value=tool_observation,
        ),
        patch("agent_runtime.agents.runtime.mark_observation_error") as mark_error,
        patch("agent_runtime.agents.runtime.finish_generation") as finish_gen,
        patch("agent_runtime.agents.runtime.finish_tool_observation") as finish_tool,
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    assert events[-1]["type"] == "error"
    assert "stream broke" in events[-1]["message"]
    marked_observations = [call.args[0] for call in mark_error.call_args_list]
    assert generation in marked_observations
    assert tool_observation in marked_observations
    assert span in marked_observations
    finish_tool.assert_called_once_with(tool_observation)
    finish_gen.assert_called_once_with(generation)


@pytest.mark.asyncio
async def test_run_agent_existing_session_with_history(monkeypatch):
    """Existing session loads prior messages into agent context."""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Existing")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="You are helpful.",
    )
    mock_repo.get_messages.return_value = [
        Message(id=1, session_id=1, role="system", content="You are helpful."),
        Message(id=2, session_id=1, role="user", content="First message"),
        Message(id=3, session_id=1, role="assistant", content="First reply"),
    ]
    mock_repo.add_message.return_value = Message(id=4, role="user", content="Second")

    mock_result = AsyncMock()
    mock_result.final_output = "Second reply"
    mock_result.context_wrapper.usage = Usage()
    mock_result.raw_responses = []

    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    with patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result):
        result = await run_agent(
            "Second",
            session_id=encode(1),
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
            model_repo=mock_model_repo,
            agent_factory=AgentFactory(),
        )

    assert result.response == "Second reply"
    mock_prompt_repo.seed_default.assert_not_called()
    mock_repo.create_session.assert_not_called()
    mock_repo.get_messages.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_run_agent_runner_error_propagates(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default",
    )
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    with (
        patch("agent_runtime.agents.runtime.Runner.run", side_effect=RuntimeError("API down")),
        pytest.raises(RuntimeError, match="API down"),
    ):
        await run_agent(
            "Hello",
            session_id=encode(1),
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
            model_repo=mock_model_repo,
            agent_factory=AgentFactory(),
        )


@pytest.mark.asyncio
async def test_run_agent_streamed_runner_error_yields_error_event(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()
    mock_model_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        session_id=1,
        role="system",
        content="Default",
    )
    mock_repo.get_messages.return_value = [
        Message(id=2, session_id=1, role="user", content="Hi"),
    ]
    mock_model_repo.get_by_provider_model.return_value = RuntimeModel(
        id=1,
        provider="openai",
        model_id="gpt-5.4-mini",
        name="gpt-5.4-mini",
        enabled=True,
    )

    class FailingStreamedResult:
        async def stream_events(self):
            raise RuntimeError("stream broke")
            yield  # make it an async generator

    with patch(
        "agent_runtime.agents.runtime.Runner.run_streamed",
        return_value=FailingStreamedResult(),
    ):
        events = [
            e
            async for e in run_agent_streamed(
                "Hi",
                session_id=encode(1),
                session_repo=mock_repo,
                prompt_repo=mock_prompt_repo,
                model_repo=mock_model_repo,
                agent_factory=AgentFactory(),
            )
        ]

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "stream broke" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_switch_prompt_success(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_prompt_repo.get_by_name.return_value = SystemPrompt(
        id=2,
        name="pirate",
        content="You are a pirate.",
    )
    mock_repo.add_message.return_value = Message(id=10, role="system", content="You are a pirate.")

    from agent_runtime.agents.runtime import switch_prompt

    result = await switch_prompt(
        encode(1),
        "pirate",
        session_repo=mock_repo,
        prompt_repo=mock_prompt_repo,
    )

    assert result == "pirate"
    mock_repo.add_message.assert_called_once_with(
        1,
        "system",
        "You are a pirate.",
        system_prompt_id=2,
    )


@pytest.mark.asyncio
async def test_switch_prompt_not_found_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()

    mock_repo.get_session.return_value = Session(id=1, title="Test")
    mock_prompt_repo.get_by_name.return_value = None

    from agent_runtime.agents.runtime import switch_prompt

    with pytest.raises(ValueError, match="Prompt not found"):
        await switch_prompt(
            encode(1),
            "nonexistent",
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
        )


@pytest.mark.asyncio
async def test_switch_prompt_session_not_found_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()

    mock_repo.get_session.return_value = None

    from agent_runtime.agents.runtime import switch_prompt

    with pytest.raises(ValueError, match="Session not found"):
        await switch_prompt(
            encode(999),
            "pirate",
            session_repo=mock_repo,
            prompt_repo=mock_prompt_repo,
        )
