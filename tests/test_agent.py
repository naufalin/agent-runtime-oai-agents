"""Tests for the main agent definition."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.agents.runtime import create_agent, run_agent


def test_create_agent_has_all_tools():
    agent = create_agent()
    assert len(agent.tools) == 4


def test_create_agent_has_name():
    agent = create_agent()
    assert agent.name == "RuntimeAgent"


def test_create_agent_has_instructions():
    agent = create_agent()
    instructions = agent.instructions
    assert callable(instructions) or isinstance(instructions, str)


@pytest.mark.asyncio
async def test_run_agent_persists_messages():
    mock_db = AsyncMock()
    mock_repo = AsyncMock()
    mock_repo.get_conversation.return_value = {"id": "test-123", "title": "Test"}
    mock_repo.add_message.return_value = {"id": 1}

    mock_result = AsyncMock()
    mock_result.final_output = "Hello!"

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.ConversationRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
    ):
        result = await run_agent("Hi there", conversation_id="test-123")

        assert result == "Hello!"
        # User message saved
        assert mock_repo.add_message.call_count == 2
        mock_repo.add_message.assert_any_call("test-123", "user", "Hi there")
        mock_repo.add_message.assert_any_call("test-123", "assistant", "Hello!")


@pytest.mark.asyncio
async def test_run_agent_creates_conversation_if_new():
    mock_db = AsyncMock()
    mock_repo = AsyncMock()
    mock_repo.get_conversation.return_value = None  # new conversation
    mock_repo.create_conversation.return_value = {"id": "new-123"}
    mock_repo.add_message.return_value = {"id": 1}

    mock_result = AsyncMock()
    mock_result.final_output = "Response"

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.ConversationRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
    ):
        result = await run_agent("First message", conversation_id="new-123")

        mock_repo.create_conversation.assert_called_once()
        assert result == "Response"
