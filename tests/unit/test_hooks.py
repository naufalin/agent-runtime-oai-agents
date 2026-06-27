"""Unit tests for PersistenceHooks (tool call persistence)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_runtime.agents.hooks import PersistenceHooks, RunContext


@pytest.fixture
def hooks():
    return PersistenceHooks()


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.add_message.return_value = SimpleNamespace(id=1)
    return repo


@pytest.fixture
def run_context(mock_repo):
    return RunContext(session_id=42, repo=mock_repo)


@pytest.fixture
def context_wrapper(run_context):
    wrapper = SimpleNamespace(context=run_context)
    return wrapper


@pytest.fixture
def agent():
    return SimpleNamespace(name="TestAgent")


@pytest.fixture
def tool():
    return SimpleNamespace(name="web_search")


@pytest.mark.asyncio
async def test_on_tool_start_saves_message(hooks, context_wrapper, agent, tool, mock_repo):
    await hooks.on_tool_start(context_wrapper, agent, tool)

    mock_repo.add_message.assert_awaited_once_with(
        42,
        role="tool",
        content="[calling web_search...]",
        tool_name="web_search",
    )


@pytest.mark.asyncio
async def test_on_tool_end_saves_short_result(hooks, context_wrapper, agent, tool, mock_repo):
    await hooks.on_tool_end(context_wrapper, agent, tool, "short result")

    mock_repo.add_message.assert_awaited_once_with(
        42,
        role="tool",
        content="short result",
        tool_name="web_search",
    )


@pytest.mark.asyncio
async def test_on_tool_end_truncates_long_result(hooks, context_wrapper, agent, tool, mock_repo):
    long_result = "x" * 3000
    await hooks.on_tool_end(context_wrapper, agent, tool, long_result)

    call_args = mock_repo.add_message.call_args
    content = call_args.kwargs.get("content") or call_args.args[2]
    assert len(content) <= 2050  # 2000 + "... [truncated]"
    assert content.endswith("... [truncated]")


@pytest.mark.asyncio
async def test_on_tool_end_exactly_2000_not_truncated(
    hooks, context_wrapper, agent, tool, mock_repo
):
    exact_result = "y" * 2000
    await hooks.on_tool_end(context_wrapper, agent, tool, exact_result)

    call_args = mock_repo.add_message.call_args
    content = call_args.kwargs.get("content") or call_args.args[2]
    assert content == exact_result
    assert "truncated" not in content


@pytest.mark.asyncio
async def test_on_tool_start_uses_correct_session_id(hooks, agent, tool, mock_repo):
    ctx = RunContext(session_id=999, repo=mock_repo)
    wrapper = SimpleNamespace(context=ctx)

    await hooks.on_tool_start(wrapper, agent, tool)

    call_args = mock_repo.add_message.call_args
    session_id = call_args.args[0] if call_args.args else call_args.kwargs.get("session_id")
    assert session_id == 999


@pytest.mark.asyncio
async def test_on_tool_end_converts_non_string_result(
    hooks, context_wrapper, agent, tool, mock_repo
):
    dict_result = {"key": "value", "count": 42}
    await hooks.on_tool_end(context_wrapper, agent, tool, dict_result)

    call_args = mock_repo.add_message.call_args
    content = call_args.kwargs.get("content") or call_args.args[2]
    assert "key" in content
    assert "value" in content
