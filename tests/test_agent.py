"""Tests for the main agent definition."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_runtime.agents.runtime import AgentResponse, create_agent, run_agent
from agent_runtime.db.models import Conversation, Message, SystemPrompt


def test_create_agent_has_all_tools():
    agent = create_agent("test prompt")
    assert len(agent.tools) == 4


def test_create_agent_has_name():
    agent = create_agent("test prompt")
    assert agent.name == "RuntimeAgent"


def test_create_agent_uses_prompt():
    agent = create_agent("You are a pirate.")
    instructions = agent.instructions
    assert instructions == "You are a pirate."


@pytest.mark.asyncio
async def test_run_agent_inserts_system_message_for_new_conversation():
    mock_db = AsyncMock()
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()

    # New conversation flow
    mock_repo.get_conversation.return_value = None
    mock_repo.create_conversation.return_value = Conversation(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=1,
        conversation_id=1,
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

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.ConversationRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
    ):
        result = await run_agent("Hi there")

        assert isinstance(result, AgentResponse)
        assert result.response == "Hello!"
        # Should have seeded default prompt
        mock_prompt_repo.seed_default.assert_called_once()
        # Should have created conversation
        mock_repo.create_conversation.assert_called_once()
        # 3 messages: system + user + assistant
        assert mock_repo.add_message.call_count == 3


@pytest.mark.asyncio
async def test_run_agent_uses_latest_system_prompt():
    mock_db = AsyncMock()
    mock_repo = AsyncMock()
    mock_prompt_repo = AsyncMock()

    mock_repo.get_conversation.return_value = Conversation(id=1, title="Test")
    mock_repo.get_latest_system_message.return_value = Message(
        id=5,
        conversation_id=1,
        role="system",
        content="You are a pirate.",
    )
    mock_repo.add_message.return_value = Message(id=6, role="user", content="Ahoy")

    mock_result = AsyncMock()
    mock_result.final_output = "Arr!"

    with (
        patch("agent_runtime.agents.runtime.get_db", return_value=mock_db),
        patch("agent_runtime.agents.runtime.ConversationRepo", return_value=mock_repo),
        patch("agent_runtime.agents.runtime.SystemPromptRepo", return_value=mock_prompt_repo),
        patch("agent_runtime.agents.runtime.Runner.run", return_value=mock_result),
    ):
        from agent_runtime.ids import encode

        result = await run_agent("Ahoy", conversation_id=encode(1))

        assert result.response == "Arr!"
        # Should NOT have seeded — resuming existing conversation
        mock_prompt_repo.seed_default.assert_not_called()
