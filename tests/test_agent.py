"""Tests for the main agent definition."""

from agent_runtime.agents.runtime import create_agent


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
