"""Tests for the main agent definition."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from agents.stream_events import RawResponsesStreamEvent
from agents.usage import Usage

from agent_runtime.agents.runtime import AgentResponse, create_agent, run_agent, run_agent_streamed
from agent_runtime.db.models import Message, RuntimeModel, Session, SystemPrompt
from agent_runtime.ids import encode


def test_create_agent_has_all_tools():
    agent = create_agent("test prompt")
    assert len(agent.tools) == 5


def test_create_agent_has_name():
    agent = create_agent("test prompt")
    assert agent.name == "MainAgent"


def test_create_agent_uses_prompt():
    agent = create_agent("You are a pirate.")
    assert agent.instructions == "You are a pirate."


@pytest.mark.asyncio
async def test_run_agent_inserts_system_message_for_new_session(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_db = AsyncMock()
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

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.RuntimeModelRepo", return_value=mock_model_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result) as mock_run,
    ):
        result = await run_agent("Hi there")

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
    mock_db = AsyncMock()
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

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.RuntimeModelRepo", return_value=mock_model_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
    ):
        from agent_runtime.ids import encode

        result = await run_agent("Ahoy", session_id=encode(1))

        assert result.response == "Arr!"
        mock_prompt_repo.seed_default.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_validates_model_before_persisting_user_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    mock_db = AsyncMock()
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
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.RuntimeModelRepo", return_value=mock_model_repo),
        patch("agent_runtime.agents.runtime.Runner.run") as mock_run,
        pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"),
    ):
        await run_agent(
            "Hello",
            session_id=encode(1),
            provider="openrouter",
            model="deepseek/deepseek-v4-flash",
        )

    mock_repo.add_message.assert_not_called()
    mock_repo.create_session.assert_not_called()
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_streamed_validates_model_before_persisting_user_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    mock_db = AsyncMock()
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
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.RuntimeModelRepo", return_value=mock_model_repo),
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
            )
        ]

    mock_repo.add_message.assert_not_called()
    mock_repo.create_session.assert_not_called()
    mock_run_streamed.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_streamed_separates_thinking_deltas_from_text(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    mock_db = AsyncMock()
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

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.RuntimeModelRepo", return_value=mock_model_repo),
        patch(
            "agent_runtime.agents.runtime.Runner.run_streamed",
            return_value=FakeStreamedResult(),
        ),
    ):
        events = [
            event
            async for event in run_agent_streamed(
                "Hello",
                session_id=encode(1),
                provider="openai",
                model="gpt-5.4-mini",
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
