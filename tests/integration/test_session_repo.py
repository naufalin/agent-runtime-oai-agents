"""Tests for session repository using real async SQLite."""

import pytest

from agent_runtime.db.session_repo import SessionRepo


@pytest.mark.asyncio
async def test_create_session(db):
    repo = SessionRepo(db)
    result = await repo.create_session("Test")
    assert result.id is not None
    assert result.id > 0
    assert result.title == "Test"


@pytest.mark.asyncio
async def test_create_session_auto_id(db):
    repo = SessionRepo(db)
    s1 = await repo.create_session("First")
    s2 = await repo.create_session("Second")
    assert s2.id > s1.id


@pytest.mark.asyncio
async def test_update_title(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Old Title")
    await repo.update_title(sess.id, "New Title")
    updated = await repo.get_session(sess.id)
    assert updated is not None
    assert updated.title == "New Title"


@pytest.mark.asyncio
async def test_add_message(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    result = await repo.add_message(sess.id, "user", "Hello!")
    assert result.session_id == sess.id
    assert result.role == "user"
    assert result.content == "Hello!"


@pytest.mark.asyncio
async def test_add_assistant_message_with_model_metadata(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    usage = {"total_tokens": 12, "reasoning_tokens": 4}
    thinking = {"reasoning": "provider returned reasoning"}

    result = await repo.add_message(
        sess.id,
        "assistant",
        "Hello!",
        provider="openrouter",
        model="z-ai/glm-5.2",
        usage_json=usage,
        thinking_json=thinking,
    )

    assert result.provider == "openrouter"
    assert result.model == "z-ai/glm-5.2"
    assert result.usage_json == usage
    assert result.thinking_json == thinking


@pytest.mark.asyncio
async def test_get_messages(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    await repo.add_message(sess.id, "user", "Hello")
    await repo.add_message(sess.id, "assistant", "Hi!")
    messages = await repo.get_messages(sess.id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_list_sessions(db):
    repo = SessionRepo(db)
    await repo.create_session("First")
    await repo.create_session("Second")
    sessions = await repo.list_sessions(limit=10)
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_get_session_found(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    result = await repo.get_session(sess.id)
    assert result is not None
    assert result.id == sess.id


@pytest.mark.asyncio
async def test_get_session_not_found(db):
    repo = SessionRepo(db)
    result = await repo.get_session(99999)
    assert result is None


@pytest.mark.asyncio
async def test_add_message_with_system_prompt_id(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    result = await repo.add_message(sess.id, "system", "You are helpful.", system_prompt_id=42)
    assert result.role == "system"
    assert result.system_prompt_id == 42


@pytest.mark.asyncio
async def test_get_latest_system_message(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    await repo.add_message(sess.id, "system", "First prompt", system_prompt_id=1)
    await repo.add_message(sess.id, "user", "Hello")
    await repo.add_message(sess.id, "assistant", "Hi!")
    await repo.add_message(sess.id, "system", "Second prompt", system_prompt_id=2)
    latest = await repo.get_latest_system_message(sess.id)
    assert latest is not None
    assert latest.content == "Second prompt"
    assert latest.system_prompt_id == 2


@pytest.mark.asyncio
async def test_get_latest_system_message_none_if_no_system(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    await repo.add_message(sess.id, "user", "Hello")
    result = await repo.get_latest_system_message(sess.id)
    assert result is None


@pytest.mark.asyncio
async def test_add_tool_message(db):
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    result = await repo.add_message(
        sess.id,
        "tool",
        "search results here",
        tool_name="web_search",
        tool_call_id="call_abc123",
        tool_input={"query": "weather Jakarta"},
        tool_output={"results": [{"title": "Weather", "url": "https://example.com"}]},
        output_preview='{"results": [{"title": "Weather"',
    )
    assert result.role == "tool"
    assert result.tool_name == "web_search"
    assert result.tool_call_id == "call_abc123"
    assert result.tool_input == {"query": "weather Jakarta"}
    assert result.tool_output == {"results": [{"title": "Weather", "url": "https://example.com"}]}
    assert result.output_preview == '{"results": [{"title": "Weather"'


@pytest.mark.asyncio
async def test_add_tool_message_minimal(db):
    """Tool message with only tool_name — no call_id, input, or output."""
    repo = SessionRepo(db)
    sess = await repo.create_session("Test")
    result = await repo.add_message(
        sess.id,
        "tool",
        "raw output text",
        tool_name="weather",
    )
    assert result.role == "tool"
    assert result.tool_name == "weather"
    assert result.tool_call_id is None
    assert result.tool_input is None
    assert result.tool_output is None
    assert result.output_preview is None
