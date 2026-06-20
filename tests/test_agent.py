"""Tests for the main agent definition."""

from unittest.mock import AsyncMock, patch

import pytest
from agents.usage import Usage

from agent_runtime.agents.runtime import AgentResponse, create_agent, run_agent
from agent_runtime.db.models import Message, Session, SystemPrompt


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

    mock_result = AsyncMock()
    mock_result.final_output = "Hello!"
    mock_result.context_wrapper.usage = Usage()
    mock_result.raw_responses = []

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
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

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.SessionRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
    ):
        from agent_runtime.ids import encode

        result = await run_agent("Ahoy", session_id=encode(1))

        assert result.response == "Arr!"
        mock_prompt_repo.seed_default.assert_not_called()
